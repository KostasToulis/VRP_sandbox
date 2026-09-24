"""
Swap (exchange) neighbourhood operator for the CVRP.

A swap move exchanges two customers u and v, each keeping the other's position.
The routes involved may be the same or different:

    before:   ... a - u - b ...        ... i - v - j ...
    after:    ... a - v - b ...        ... i - u - j ...

For the general (non-adjacent) case the two endpoint deltas are independent:

    delta = [ d(a,v) + d(v,b) - d(a,u) - d(u,b) ]
          + [ d(i,u) + d(u,j) - d(i,v) - d(v,j) ]

When u and v are adjacent inside the same route the shared edge d(u,v) appears
on both sides and cancels (distances here are symmetric), so that case is
handled separately - applying the general formula to it would double-count.

Unlike relocate, a swap never changes how many customers a route serves, so an
intra-route swap is always capacity-feasible and no route can be emptied.

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
class SwapMove:
    """
    A single swap move.

    Attributes:
        first_route/first_pos:   location of customer u.
        second_route/second_pos: location of customer v.
        first_id/second_id:      the two customer ids (for logging/tabu lists).
        delta:                   change in total cost; negative is an improvement.
        removed_arcs:            arcs this move destroys. A tabu driver forbids
                                 restoring these for the length of the tenure.
        added_arcs:              arcs this move creates. The move is tabu when
                                 any of them is currently forbidden.

    Arcs that the move both destroys and recreates are netted out of both
    tuples. This matters for the adjacent case, where the link between the two
    swapped customers survives the exchange untouched.
    """

    first_route: int
    first_pos: int
    second_route: int
    second_pos: int
    first_id: int
    second_id: int
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

        Route ids are renumbered from 1, matching how solutions are built
        elsewhere in the project. A swap cannot empty a route.
        """
        if dist is None:
            dist = build_distance_matrix(model)

        sequences = [list(r.sequence_of_nodes) for r in solution.routes]
        first = sequences[self.first_route][self.first_pos]
        second = sequences[self.second_route][self.second_pos]
        sequences[self.first_route][self.first_pos] = second
        sequences[self.second_route][self.second_pos] = first

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
) -> Optional[SwapMove]:
    """
    Find the cheapest feasible swap move in `solution`.

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
        The best admissible SwapMove, or None when no feasible non-tabu move
        exists.

    Each unordered pair of customers is evaluated once, and ties are broken by
    the move's index tuple, so the result is deterministic.
    """
    if dist is None:
        dist = build_distance_matrix(model)

    depot_id = model.depot_id
    capacity = model.capacity
    sequences = [r.sequence_of_nodes for r in solution.routes]
    # Recompute loads rather than trusting Route.load, which may be stale.
    loads = [sum(n.demand for n in seq) for seq in sequences]

    # Flattened customer positions; taking a < b covers every unordered pair once.
    positions: List[Tuple[int, int]] = [
        (ri, p) for ri, seq in enumerate(sequences) for p in range(len(seq))
    ]

    # Recomputed rather than trusting Solution.cost, which may be stale.
    base_cost = (
        sum(_route_cost(r.sequence_of_nodes, model, dist) for r in solution.routes)
        if arc_promises
        else 0.0
    )
    best: Optional[SwapMove] = None

    for a in range(len(positions)):
        ri, p = positions[a]
        seq_i = sequences[ri]
        u = seq_i[p]
        prev_u = depot_id if p == 0 else seq_i[p - 1].id
        next_u = depot_id if p == len(seq_i) - 1 else seq_i[p + 1].id

        for b in range(a + 1, len(positions)):
            rj, q = positions[b]
            seq_j = sequences[rj]
            v = seq_j[q]

            if rj != ri:
                # Both routes must still fit after trading the two demands.
                if loads[ri] - u.demand + v.demand > capacity:
                    continue
                if loads[rj] - v.demand + u.demand > capacity:
                    continue

            prev_v = depot_id if q == 0 else seq_j[q - 1].id
            next_v = depot_id if q == len(seq_j) - 1 else seq_j[q + 1].id

            adjacent = ri == rj and q == p + 1

            if adjacent:
                # Adjacent in the same route: u and v trade places across the
                # shared edge, which is unchanged and cancels out.
                delta = (
                    dist[prev_u][v.id]
                    + dist[u.id][next_v]
                    - dist[prev_u][u.id]
                    - dist[v.id][next_v]
                )
            else:
                delta = (
                    dist[prev_u][v.id]
                    + dist[v.id][next_u]
                    - dist[prev_u][u.id]
                    - dist[u.id][next_u]
                ) + (
                    dist[prev_v][u.id]
                    + dist[u.id][next_v]
                    - dist[prev_v][v.id]
                    - dist[v.id][next_v]
                )

            # Only a candidate that would become the new best is worth the arc
            # bookkeeping, so do the cheap comparison first.
            if best is not None and not (
                delta < best.delta
                or (
                    delta == best.delta
                    and (ri, p, rj, q)
                    < (best.first_route, best.first_pos, best.second_route, best.second_pos)
                )
            ):
                continue

            if adjacent:
                # The u-v link survives the exchange, merely reversed, so it is
                # listed on both sides and nets out. Deriving it from the general
                # lists below would instead collapse to self-loops and wrongly
                # report the link as destroyed.
                raw_removed = (arc(prev_u, u.id), arc(u.id, v.id), arc(v.id, next_v))
                raw_added = (arc(prev_u, v.id), arc(v.id, u.id), arc(u.id, next_v))
            else:
                raw_removed = (
                    arc(prev_u, u.id),
                    arc(u.id, next_u),
                    arc(prev_v, v.id),
                    arc(v.id, next_v),
                )
                raw_added = (
                    arc(prev_u, v.id),
                    arc(v.id, next_u),
                    arc(prev_v, u.id),
                    arc(u.id, next_v),
                )

            removed_net, added_net = _net_arcs(raw_removed, raw_added)

            if tabu_arcs and any(a in tabu_arcs for a in added_net):
                # Blocked, unless it is good enough to aspire past the ban.
                if aspiration_delta is None or delta >= aspiration_delta:
                    continue

            if arc_promises and _promise_blocks(arc_promises, added_net, base_cost + delta):
                # Re-creating a promised arc without beating its label.
                continue

            best = SwapMove(
                first_route=ri,
                first_pos=p,
                second_route=rj,
                second_pos=q,
                first_id=u.id,
                second_id=v.id,
                delta=delta,
                removed_arcs=removed_net,
                added_arcs=added_net,
            )

    if best is None:
        logger.debug("Swap: neighbourhood is empty.")
        return None

    if only_improving and best.delta >= 0:
        logger.debug("Swap: no improving move (best delta %.2f).", best.delta)
        return None

    logger.debug(
        "Swap: customer %d (route %d) with customer %d (route %d), delta %.2f.",
        best.first_id, best.first_route, best.second_id, best.second_route, best.delta,
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
