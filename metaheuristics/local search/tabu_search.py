"""
Tabu search for the CVRP over the relocate / swap / 2-opt neighbourhoods.

Tabu search (Glover, 1986) escapes local optima by always moving to the best
solution in the neighbourhood, even when that move makes things worse, and
forbidding recently undone changes so the search cannot immediately walk back.

This driver is a steepest-descent tabu search:

  * every iteration scans ALL THREE neighbourhoods in full - relocate, swap and
    2-opt - and applies the single cheapest admissible move among them;
  * the tabu attribute is the ARC. Applying a move puts the arcs it destroys on
    the tabu list for `tabu_tenure` iterations, and the operators refuse to
    evaluate any candidate that would recreate one of them;
  * aspiration by objective lets a tabu move through anyway when it would beat
    the best solution found so far, which is the standard guard against the
    tabu list hiding a genuinely good move;
  * the search stops after `max_non_improving` consecutive iterations that fail
    to improve the best-known solution, and returns that best solution.

The method is fully deterministic - steepest descent with deterministic
tie-breaking - so it needs no seed parameter to be reproducible.

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
#: solution. This is the main effort dial: higher means a longer, more thorough
#: search. Each iteration costs a full scan of all three neighbourhoods.
MAX_NON_IMPROVING_ITERATIONS = 1000

#: How many iterations an arc stays forbidden after a move destroys it. Higher
#: means a more diversified search that is harder to cycle, but also a more
#: constrained one: with ~3 arcs banned per iteration, a tenure of T keeps
#: roughly 3T arcs off the table at once, which on a small instance can be a
#: large fraction of all arcs that exist.
TABU_TENURE = 100

#: Allow a tabu move when it would produce a better solution than any seen so
#: far (aspiration by objective). Turning this off makes the tabu ban absolute.
USE_ASPIRATION = True

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


class TabuSearch:
    """
    Steepest-descent tabu search over relocate, swap and 2-opt.

    The search keeps two solutions: `current`, which moves every iteration and
    is allowed to get worse, and `best`, the cheapest solution ever visited,
    which is what solve() returns.
    """

    def __init__(
        self,
        model: VRPModel,
        initial_solution: Optional[Solution] = None,
        max_non_improving: int = MAX_NON_IMPROVING_ITERATIONS,
        tabu_tenure: int = TABU_TENURE,
        use_aspiration: bool = USE_ASPIRATION,
        time_limit: Optional[float] = TIME_LIMIT_SECONDS,
    ):
        if max_non_improving < 1:
            raise ValueError("max_non_improving must be at least 1.")
        if tabu_tenure < 0:
            raise ValueError("tabu_tenure must be non-negative.")
        if time_limit is not None and time_limit <= 0:
            raise ValueError("time_limit must be positive when given.")

        self.model = model
        self.initial_solution = initial_solution
        self.max_non_improving = max_non_improving
        self.tabu_tenure = tabu_tenure
        self.use_aspiration = use_aspiration
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

        # arc -> the iteration at which the ban lapses
        tabu: Dict[Arc, int] = {}
        iteration = 0
        non_improving = 0
        started = time.perf_counter()

        logger.info(
            "Tabu search on %s: start cost %.2f, tenure %d, stop after %d "
            "non-improving iterations, aspiration %s.",
            self.model.name, current.cost, self.tabu_tenure,
            self.max_non_improving, "on" if self.use_aspiration else "off",
        )

        while non_improving < self.max_non_improving:
            iteration += 1

            # Retire lapsed bans so the set handed to the operators stays small.
            if tabu:
                for expired in [a for a, ends in tabu.items() if ends <= iteration]:
                    del tabu[expired]

            # A tabu move is admissible when it would beat the incumbent best,
            # i.e. current.cost + delta < best.cost.
            aspiration = (best.cost - current.cost) if self.use_aspiration else None

            move, source = self._best_move(current, dist, tabu.keys(), aspiration)

            if move is None:
                # Every neighbour of every neighbourhood is currently banned.
                logger.info(
                    "Iteration %d: no admissible move remains; stopping early.", iteration
                )
                break

            current = move.apply(current, self.model, dist)

            # Forbid restoring what this move just destroyed.
            for destroyed in move.removed_arcs:
                tabu[destroyed] = iteration + self.tabu_tenure

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
            "Finished after %d iterations: best cost %.2f over %d routes (%.1fs).",
            iteration, best.cost, len(best.routes), time.perf_counter() - started,
        )

        self._validate(best)
        return best

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _best_move(self, solution, dist, tabu_arcs, aspiration):
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
                tabu_arcs=tabu_arcs,
                aspiration_delta=aspiration,
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
    Solve the given CVRP instance using Tabu Search.

    Args:
        instance: A fully constructed VRPModel from setup.py.
        **kwargs: initial_solution, max_non_improving, tabu_tenure,
                  use_aspiration, time_limit.

    Returns:
        A Solution object with routes and total cost.
    """
    return TabuSearch(instance, **kwargs).solve()


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
