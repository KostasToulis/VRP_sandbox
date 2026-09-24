r"""
Relocate (shift / or-opt-1) neighbourhood operator for the CVRP.

A relocate move lifts a single customer out of its current position and
reinserts it somewhere else - either into a different route or at a different
position of the same route:

    before:   ... a - u - b ...        ... i - j ...
    after:    ... a - b ...            ... i - u - j ...

The move cost is the removal saving plus the insertion detour:

    delta = [ d(a,b) - d(a,u) - d(u,b) ]  +  [ d(i,u) + d(u,j) - d(i,j) ]
             \_______ <= 0 _______/          \_______ >= 0 _______/

`find_best_move` scans the whole neighbourhood and returns the single cheapest
feasible move, or None when the neighbourhood is empty.

Dependencies: standard library only.

NOTE: this file lives under 'metaheuristics/local search/operators/'. The space
in 'local search' means the directory is not a valid Python package name, so
this module cannot be reached with a plain `import`. Load it with importlib
(see the module docstring of a driver) or rename the folder to 'local_search'.
"""

from __future__ import annotations

# Running this file directly puts the operators folder on sys.path rather than
# the repository root, which would make the `model` import below fail. Add the
# root up front. This is a no-op when a driver has already imported the module.
if __name__ == "__main__":
    import os as _os
    import sys as _sys

    _sys.path.insert(
        0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", ".."))
    )

import logging
from collections import Counter
from dataclasses import dataclass
from typing import AbstractSet, Dict, List, Mapping, Optional, Tuple

from model import VRPModel, Route, Solution, Node

logger = logging.getLogger(__name__)

DistanceMatrix = Dict[int, Dict[int, float]]
Arc = Tuple[int, int]


def arc(a: int, b: int) -> Arc:
    """
    Canonical key for the undirected arc between two nodes.

    Distances in this project are symmetric and 2-opt reverses whole segments,
    so (a,b) and (b,a) are the same physical link and must share one tabu entry.
    """
    return (a, b) if a <= b else (b, a)


@dataclass(frozen=True)
class RelocateMove:
    """
    A single relocate move.

    Attributes:
        from_route:  index into solution.routes of the route losing the customer.
        from_pos:    index of the customer inside that route's sequence.
        to_route:    index into solution.routes of the route gaining the customer.
        to_pos:      insertion index within the TARGET route's sequence *after*
                     the customer has been removed. For an intra-route move this
                     means to_pos is an index into the shortened sequence, which
                     keeps `apply` unambiguous.
        customer_id: id of the relocated customer (for logging/tabu lists).
        delta:       change in total cost; negative means an improvement.
        removed_arcs: arcs this move destroys. A tabu driver forbids restoring
                     these for the length of the tenure.
        added_arcs:  arcs this move creates. The move is tabu when any of them
                     is currently forbidden.

    Arcs that the move both destroys and recreates are netted out of both
    tuples, since such a link is left untouched and must not become tabu.
    """

    from_route: int
    from_pos: int
    to_route: int
    to_pos: int
    customer_id: int
    delta: float
    removed_arcs: Tuple[Arc, ...] = ()
    added_arcs: Tuple[Arc, ...] = ()

    def apply(
        self,
        solution: Solution,
        model: VRPModel,
        dist: Optional[DistanceMatrix] = None,
    ) -> Solution:
        """
        Return a NEW Solution with this move applied; `solution` is left untouched.

        Routes emptied by the move are dropped and the survivors are renumbered
        from 1, matching how solutions are built elsewhere in the project.
        """
        if dist is None:
            dist = build_distance_matrix(model)

        sequences = [list(r.sequence_of_nodes) for r in solution.routes]
        customer = sequences[self.from_route].pop(self.from_pos)
        sequences[self.to_route].insert(self.to_pos, customer)

        return _rebuild(sequences, model, dist)


def _net_arcs(removed, added):
    """
    Reduce a move's raw arc lists to the traversals it actually changes.

    Bookkeeping is done on TRAVERSAL COUNTS, not mere presence, because a route
    serving a single customer drives the depot-customer link twice: in
    depot -> u -> depot the arc (depot, u) is used twice over. Inserting a
    customer ahead of u therefore consumes one of those two traversals while
    leaving the link in place, and a tabu driver wants that change recorded -
    it is exactly what stops the search undoing the insertion next iteration.

    Self-loops are dropped (predecessor and successor are both the depot when a
    route holds one customer), and a link destroyed and immediately recreated
    cancels out, since an untouched traversal must not become tabu nor block
    the move that leaves it alone.
    """
    removed_count = Counter(a for a in removed if a[0] != a[1])
    added_count = Counter(a for a in added if a[0] != a[1])
    return (
        tuple(sorted((removed_count - added_count).elements())),
        tuple(sorted((added_count - removed_count).elements())),
    )


def _promise_blocks(promises, added, new_cost):
    """
    Promise rule: an arc carries the cost of the solution it was deleted from.

    Re-creating it is only worth doing if the search can do better than it was
    doing when it let the arc go, so the move is blocked unless the solution it
    produces is strictly cheaper than that recorded label. An arc absent from
    the mapping has an implicit promise of infinity and is therefore free.
    """
    for a in added:
        limit = promises.get(a)
        if limit is not None and new_cost >= limit:
            return True
    return False


def build_distance_matrix(model: VRPModel) -> DistanceMatrix:
    """
    Pre-compute all pairwise distances.

    VRPModel.get_distance recomputes a square root and a rounding on every call,
    and a neighbourhood scan touches it O(n^2) times, so a driver running many
    iterations should build this once and pass it to every call.
    """
    return {
        a.id: {b.id: model.get_distance(a.id, b.id) for b in model.nodes}
        for a in model.nodes
    }


def find_best_move(
    solution: Solution,
    model: VRPModel,
    *,
    only_improving: bool = False,
    dist: Optional[DistanceMatrix] = None,
    tabu_arcs: Optional[AbstractSet[Arc]] = None,
    aspiration_delta: Optional[float] = None,
    arc_promises: Optional[Mapping[Arc, float]] = None,
) -> Optional[RelocateMove]:
    """
    Find the cheapest feasible relocate move in `solution`.

    Args:
        solution:       the incumbent solution to search around.
        model:          the VRPModel, needed for distances, capacity and the depot.
        only_improving: if True, return None unless the best move strictly
                        reduces cost (delta < 0). If False (the default) the
                        globally cheapest move is returned even when it worsens
                        the solution, which is what tabu search and other
                        non-monotone drivers need.
        dist:           optional pre-computed distance matrix.
        tabu_arcs:      arcs that must not be recreated. Any move that would add
                        one of them is skipped, so the returned move is the best
                        NON-TABU move rather than the best move overall. Pass a
                        set (or dict keys) for O(1) membership tests.
        aspiration_delta: standard aspiration-by-objective escape hatch. A tabu
                        move is admitted anyway when its delta is strictly below
                        this value; a driver sets it to
                        `best_cost - current_cost` so a move is allowed when it
                        would beat the best solution seen so far. None disables
                        aspiration, making the tabu ban absolute.
        arc_promises:   per-arc alternative to the flat `tabu_arcs` ban, used by
                        the promises search. Maps an arc to the cost of the
                        solution it was deleted from; a move that re-creates
                        that arc is skipped unless the solution it produces
                        costs strictly less than the recorded label. An arc that
                        is absent has an implicit promise of infinity and is
                        free. The aspiration criterion is built into this rule,
                        so `aspiration_delta` does not apply to it; a driver
                        uses either this scheme or `tabu_arcs`, not both.

    Returns:
        The best admissible RelocateMove, or None when no feasible non-tabu move
        exists.

    Ties are broken by the move's index tuple, so the result is deterministic.
    """
    if dist is None:
        dist = build_distance_matrix(model)

    depot_id = model.depot_id
    capacity = model.capacity
    routes = solution.routes
    sequences = [r.sequence_of_nodes for r in routes]
    # Recompute loads rather than trusting Route.load, which may be stale.
    loads = [sum(n.demand for n in seq) for seq in sequences]

    # Recomputed rather than trusting Solution.cost, which may be stale.
    base_cost = (
        sum(_route_cost(r.sequence_of_nodes, model, dist) for r in solution.routes)
        if arc_promises
        else 0.0
    )
    best: Optional[RelocateMove] = None

    for ri, seq in enumerate(sequences):
        for p, customer in enumerate(seq):
            prev_id = depot_id if p == 0 else seq[p - 1].id
            next_id = depot_id if p == len(seq) - 1 else seq[p + 1].id

            # Saving from closing the gap left behind (<= 0 under triangle inequality).
            removal_gain = (
                dist[prev_id][next_id] - dist[prev_id][customer.id] - dist[customer.id][next_id]
            )

            # The sequence the customer is inserted into, as seen after removal.
            reduced = seq[:p] + seq[p + 1:]

            for rj, target in enumerate(sequences):
                if rj == ri:
                    candidate_seq = reduced
                else:
                    if loads[rj] + customer.demand > capacity:
                        continue
                    candidate_seq = target

                for q in range(len(candidate_seq) + 1):
                    # Putting the customer straight back where it came from is a no-op.
                    if rj == ri and q == p:
                        continue

                    left_id = depot_id if q == 0 else candidate_seq[q - 1].id
                    right_id = (
                        depot_id if q == len(candidate_seq) else candidate_seq[q].id
                    )
                    insertion_cost = (
                        dist[left_id][customer.id]
                        + dist[customer.id][right_id]
                        - dist[left_id][right_id]
                    )

                    delta = removal_gain + insertion_cost

                    # Only a candidate that would become the new best is worth
                    # the arc bookkeeping, so do the cheap comparison first.
                    if best is not None and not (
                        delta < best.delta
                        or (
                            delta == best.delta
                            and (ri, p, rj, q)
                            < (best.from_route, best.from_pos, best.to_route, best.to_pos)
                        )
                    ):
                        continue

                    removed_net, added_net = _net_arcs(
                        (
                            arc(prev_id, customer.id),
                            arc(customer.id, next_id),
                            arc(left_id, right_id),
                        ),
                        (
                            arc(prev_id, next_id),
                            arc(left_id, customer.id),
                            arc(customer.id, right_id),
                        ),
                    )

                    if tabu_arcs and any(a in tabu_arcs for a in added_net):
                        # Blocked, unless it is good enough to aspire past the ban.
                        if aspiration_delta is None or delta >= aspiration_delta:
                            continue

                    if arc_promises and _promise_blocks(arc_promises, added_net, base_cost + delta):
                        # Re-creating a promised arc without beating its label.
                        continue

                    best = RelocateMove(
                        from_route=ri,
                        from_pos=p,
                        to_route=rj,
                        to_pos=q,
                        customer_id=customer.id,
                        delta=delta,
                        removed_arcs=removed_net,
                        added_arcs=added_net,
                    )

    if best is None:
        logger.debug("Relocate: neighbourhood is empty.")
        return None

    if only_improving and best.delta >= 0:
        logger.debug("Relocate: no improving move (best delta %.2f).", best.delta)
        return None

    logger.debug(
        "Relocate: customer %d from route %d to route %d, delta %.2f.",
        best.customer_id, best.from_route, best.to_route, best.delta,
    )
    return best


def _route_cost(seq: List[Node], model: VRPModel, dist: DistanceMatrix) -> float:
    """Closed-tour cost depot -> seq -> depot."""
    if not seq:
        return 0.0
    depot_id = model.depot_id
    cost = dist[depot_id][seq[0].id]
    for a, b in zip(seq, seq[1:]):
        cost += dist[a.id][b.id]
    return float(cost + dist[seq[-1].id][depot_id])


def _rebuild(
    sequences: List[List[Node]], model: VRPModel, dist: DistanceMatrix
) -> Solution:
    """Turn customer sequences into a Solution, dropping any emptied route."""
    routes: List[Route] = []
    for seq in sequences:
        if not seq:
            continue
        routes.append(
            Route(
                id=len(routes) + 1,
                load=sum(n.demand for n in seq),
                cost=_route_cost(seq, model, dist),
                sequence_of_nodes=list(seq),
            )
        )
    return Solution(cost=sum(r.cost for r in routes), routes=routes)


if __name__ == "__main__":
    import setup
    from heuristics.minimum_insertion import solve as construct

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", ".."))
    _model = setup.read_vrp_file(_os.path.join(_root, "instances", "A-n32-k5.vrp"))
    _solution = construct(_model)
    _move = find_best_move(_solution, _model)

    print(f"start cost : {_solution.cost:.0f} ({len(_solution.routes)} routes)")
    print(f"best move  : {_move}")
    if _move is not None:
        _improved = _move.apply(_solution, _model)
        print(f"after move : {_improved.cost:.0f} ({len(_improved.routes)} routes)")
