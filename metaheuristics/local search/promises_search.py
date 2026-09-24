"""
Promises search for the CVRP over the relocate / swap / 2-opt neighbourhoods.

A variant of tabu search in which the tabu list stores, alongside each deleted
arc, the COST OF THE SOLUTION IT WAS DELETED FROM. That recorded value is the
arc's promise:

    p(i,j) = cost of the solution that contained arc (i,j) when it was removed

A move that re-introduces arc (i,j) is forbidden unless the solution it
produces costs strictly less than p(i,j). The reasoning is that the search
already walked away from a solution of that cost while holding this arc, so
putting the arc back is only worth doing if it can now do better than it was
doing then.

Every arc starts with an implicit promise of infinity - absent from the mapping
means free - and every `p_iter` iterations the whole mapping is re-initialised
to infinity, wiping the memory. `p_iter` therefore plays the role the tenure
plays in classical tabu search.

How this differs from tabu_search.py:

  * Classical tabu bans an arc OUTRIGHT for a fixed tenure, with a separate,
    global aspiration criterion (beat the best solution ever found) bolted on
    to let exceptional moves through.
  * The promise rule has no separate aspiration criterion because the rule IS
    one: the threshold is per-arc, fixed at the moment of deletion, and a move
    is admitted precisely when it beats that arc's own label.
  * Tabu bans expire one arc at a time as each tenure runs out; promises are
    wiped wholesale every p_iter iterations, which gives the search a sawtooth
    of tightening constraint followed by a clean slate.

Everything else matches tabu_search.py: steepest descent over all three
neighbourhoods each iteration, stop after `max_non_improving` consecutive
iterations without improving the best-known solution, return that best
solution. The method is fully deterministic and needs no seed.

Dependencies: standard library only.

NOTE: this file lives under 'metaheuristics/local search/'. The space in
'local search' means the directory is not a valid Python package name, so this
module cannot be reached with a plain `import`; solver.py loads it with
importlib, and so must any other caller. Renaming the folder to 'local_search'
would remove the need.
"""

from __future__ import annotations

# Running this file directly puts its own folder on sys.path rather than the
# repository root, which would make the `model` import below fail. Add the root
# up front. This is a no-op when a driver has already imported the module.
if __name__ == "__main__":
    import os as _os
    import sys as _sys

    _sys.path.insert(
        0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
    )

import importlib.util
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

from model import VRPModel, Solution

logger = logging.getLogger(__name__)


# ===========================================================================
# Parameters
# ===========================================================================

#: Stop once this many consecutive iterations fail to improve the best-known
#: solution. The main effort dial: each iteration costs a full scan of all
#: three neighbourhoods.
MAX_NON_IMPROVING_ITERATIONS = 1000

#: Re-initialise every promise to infinity every this many iterations, wiping
#: the memory. Plays the role the tenure plays in classical tabu search: small
#: values keep the search barely constrained, large values let promises pile up
#: (roughly 3 arcs are labelled per iteration) until the wipe frees them all.
P_ITER = 100

#: Optional wall-clock cap in seconds, checked at the end of each iteration.
#: None disables it and lets MAX_NON_IMPROVING_ITERATIONS decide alone.
TIME_LIMIT_SECONDS: Optional[float] = None

# ===========================================================================


Arc = Tuple[int, int]

_OPERATOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "operators")

#: (attribute name, file name) for each neighbourhood. The scan order also
#: breaks ties between operators that offer equally good moves.
_OPERATOR_FILES: Sequence[Tuple[str, str]] = (
    ("relocate", "relocate.py"),
    ("swap", "swap.py"),
    ("two_opt", "2-opt.py"),
)


def load_operators() -> List[Tuple[str, object]]:
    """
    Load the three operator modules by path.

    'local search' contains a space and '2-opt.py' starts with a digit and
    contains a hyphen, so none of these are importable as module names. They
    have to be loaded from their file paths instead.
    """
    loaded: List[Tuple[str, object]] = []
    for name, filename in _OPERATOR_FILES:
        path = os.path.join(_OPERATOR_DIR, filename)
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load operator module from {path}")
        module = importlib.util.module_from_spec(spec)
        # Registering before exec_module is required: @dataclass looks the
        # defining class's module up in sys.modules and fails without it.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded.append((name, module))
    return loaded


class PromisesSearch:
    """
    Steepest-descent search under the promise policy.

    The search keeps two solutions: `current`, which moves every iteration and
    is allowed to get worse, and `best`, the cheapest solution ever visited,
    which is what solve() returns.
    """

    def __init__(
        self,
        model: VRPModel,
        initial_solution: Optional[Solution] = None,
        max_non_improving: int = MAX_NON_IMPROVING_ITERATIONS,
        p_iter: int = P_ITER,
        time_limit: Optional[float] = TIME_LIMIT_SECONDS,
    ):
        if max_non_improving < 1:
            raise ValueError("max_non_improving must be at least 1.")
        if p_iter < 1:
            raise ValueError("p_iter must be at least 1.")
        if time_limit is not None and time_limit <= 0:
            raise ValueError("time_limit must be positive when given.")

        self.model = model
        self.initial_solution = initial_solution
        self.max_non_improving = max_non_improving
        self.p_iter = p_iter
        self.time_limit = time_limit

        self.operators = load_operators()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def solve(self) -> Solution:
        """Run the search and return the best solution found."""
        dist = self.operators[0][1].build_distance_matrix(self.model)

        current = self.initial_solution or self._construct()
        best = current

        # arc -> cost of the solution it was deleted from. An arc missing from
        # this mapping carries an implicit promise of infinity, i.e. is free.
        promises: Dict[Arc, float] = {}
        iteration = 0
        non_improving = 0
        resets = 0
        started = time.perf_counter()

        logger.info(
            "Promises search on %s: start cost %.2f, p_iter %d, stop after %d "
            "non-improving iterations.",
            self.model.name, current.cost, self.p_iter, self.max_non_improving,
        )

        while non_improving < self.max_non_improving:
            iteration += 1

            # p(i,j) <- infinity for every arc.
            if iteration % self.p_iter == 0:
                logger.debug(
                    "Iteration %d: re-initialising %d promises to infinity.",
                    iteration, len(promises),
                )
                promises.clear()
                resets += 1

            move, source = self._best_move(current, dist, promises)

            if move is None:
                # Every candidate in every neighbourhood re-creates a promised
                # arc without beating its label.
                logger.info(
                    "Iteration %d: no admissible move remains; stopping early.", iteration
                )
                break

            # The label is the cost of the solution the arc is being taken out
            # of, so it is read BEFORE the move is applied. A later deletion of
            # the same arc overwrites the label with the more recent cost.
            label = current.cost
            for deleted in move.removed_arcs:
                promises[deleted] = label

            current = move.apply(current, self.model, dist)

            if current.cost < best.cost:
                best = current
                non_improving = 0
                logger.info(
                    "Iteration %d: new best %.2f (%s, delta %.2f).",
                    iteration, best.cost, source, move.delta,
                )
            else:
                non_improving += 1

            if self.time_limit is not None and time.perf_counter() - started >= self.time_limit:
                logger.info(
                    "Iteration %d: time limit of %.1fs reached; stopping.",
                    iteration, self.time_limit,
                )
                break

        logger.info(
            "Finished after %d iterations (%d promise resets): best cost %.2f "
            "over %d routes (%.1fs).",
            iteration, resets, best.cost, len(best.routes),
            time.perf_counter() - started,
        )

        self._validate(best)
        return best

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _best_move(self, solution, dist, promises):
        """
        Steepest descent across all three neighbourhoods.

        Returns the single cheapest admissible move and the name of the
        neighbourhood it came from. Ties go to the earlier operator in
        _OPERATOR_FILES, which keeps the search deterministic.
        """
        best_move = None
        best_source = ""

        for name, module in self.operators:
            candidate = module.find_best_move(
                solution,
                self.model,
                dist=dist,
                arc_promises=promises,
            )
            if candidate is None:
                continue
            if best_move is None or candidate.delta < best_move.delta:
                best_move, best_source = candidate, name

        return best_move, best_source

    def _construct(self) -> Solution:
        """Build a starting solution when the caller did not supply one."""
        from heuristics.minimum_insertion import solve as minimum_insertion

        return minimum_insertion(self.model, strategy="parallel")

    def _validate(self, solution: Solution) -> None:
        """Assert the returned solution is feasible before handing it back."""
        depot_id = self.model.depot_id
        expected = {n.id for n in self.model.nodes if n.id != depot_id}
        served: List[int] = []

        for route in solution.routes:
            ids = [n.id for n in route.sequence_of_nodes]
            assert ids, f"Route {route.id} is empty."
            assert depot_id not in ids, f"Route {route.id} contains the depot in its sequence."
            assert len(ids) == len(set(ids)), f"Route {route.id} revisits a customer."
            load = sum(n.demand for n in route.sequence_of_nodes)
            assert load <= self.model.capacity, (
                f"Route {route.id} exceeds capacity: {load} > {self.model.capacity}."
            )
            assert route.load == load, (
                f"Route {route.id} load does not match the demands it serves."
            )
            served.extend(ids)

        assert len(served) == len(set(served)), "A customer is served by more than one route."
        assert set(served) == expected, (
            f"Customer coverage mismatch: missing {sorted(expected - set(served))}, "
            f"unexpected {sorted(set(served) - expected)}."
        )


def solve(instance: VRPModel, **kwargs) -> Solution:
    """
    Solve the given CVRP instance using the Promises search.

    Args:
        instance: A fully constructed VRPModel from setup.py.
        **kwargs: initial_solution, max_non_improving, p_iter, time_limit.

    Returns:
        A Solution object with routes and total cost.
    """
    return PromisesSearch(instance, **kwargs).solve()


if __name__ == "__main__":
    import setup

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
    _model = setup.read_vrp_file(_os.path.join(_root, "instances", "A-n32-k5.vrp"))

    _best = solve(_model, max_non_improving=200)

    print(f"\nTotal Cost: {_best.cost:.2f}")
    print(f"Number of Routes: {len(_best.routes)}")
    for _r in _best.routes:
        _seq = " -> ".join(str(_n.id) for _n in _r.sequence_of_nodes)
        print(f"  Route {_r.id} (Load: {_r.load}): {_model.depot_id} -> {_seq} -> {_model.depot_id}")
