import os
import setup
from typing import Dict, List, Tuple
from metaheuristics.genetic_algorithm import GeneticAlgorithm
from exact.branch_and_cut import BranchAndCut
from exact.column_generation import ColumnGeneration
from metaheuristics.random_key_optimizer import RandomKeyOptimizer
from heuristics.minimum_insertion import MinimumInsertion


def _load_local_search(module_name: str):
    """
    Load a module from metaheuristics/local search/ by path.

    'local search' contains a space, so it is not a valid package name and its
    modules cannot be imported normally. Loading is deferred to call time to
    keep these off the import path of every other methodology.
    """
    import importlib.util
    import os
    import sys

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "metaheuristics", "local search", f"{module_name}.py",
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    # Registering before exec_module is required: @dataclass looks the defining
    # class's module up in sys.modules and fails without it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_solution(model, solution) -> Tuple[bool, List[str]]:
    """
    Validate a CVRP solution against core feasibility constraints:
    1) Capacity constraints per route.
    2) Each vehicle route departs from and returns to the central depot (implicit representation).
    3) No subtours/revisits inside routes.
    4) Every customer is served exactly once.
    """
    errors: List[str] = []
    depot_id = model.depot_id
    customer_ids = {n.id for n in model.nodes if n.id != depot_id}
    served_count: Dict[int, int] = {cid: 0 for cid in customer_ids}

    for route in solution.routes:
        seq = route.sequence_of_nodes

        if not seq:
            errors.append(f"Route {route.id} is empty.")
            continue

        route_ids = [n.id for n in seq]

        # Depot should never appear in an explicit route sequence.
        if depot_id in route_ids:
            errors.append(f"Route {route.id} contains depot node {depot_id} in sequence.")

        # Subtour/revisit guard: repeated customer in the same route implies a cycle/revisit.
        if len(route_ids) != len(set(route_ids)):
            errors.append(f"Route {route.id} contains repeated customers (subtour/revisit detected).")

        # Capacity check based on actual node demands.
        actual_load = sum(node.demand for node in seq)
        if actual_load > model.capacity:
            errors.append(
                f"Route {route.id} exceeds capacity: load={actual_load}, capacity={model.capacity}."
            )
        if route.load != actual_load:
            errors.append(
                f"Route {route.id} load mismatch: stored={route.load}, recomputed={actual_load}."
            )

        # Depot departure/return are implicit, so verify both endpoint depot links are valid.
        first_id = seq[0].id
        last_id = seq[-1].id
        try:
            _ = model.get_distance(depot_id, first_id)
            _ = model.get_distance(last_id, depot_id)
        except KeyError:
            errors.append(
                f"Route {route.id} cannot connect to depot {depot_id} from endpoints ({first_id}, {last_id})."
            )

        for nid in route_ids:
            if nid not in customer_ids:
                errors.append(f"Route {route.id} visits invalid customer id {nid}.")
                continue
            served_count[nid] += 1

    missing = [cid for cid, cnt in served_count.items() if cnt == 0]
    duplicates = [cid for cid, cnt in served_count.items() if cnt > 1]

    if missing:
        errors.append(f"Unserved customers: {sorted(missing)}")
    if duplicates:
        errors.append(f"Customers served more than once: {sorted(duplicates)}")

    return len(errors) == 0, errors

_METHOD_ALIASES = {
    "branch_and_cut": "branch_and_cut",
    "column_generation": "column_generation",
    "genetic_algorithm": "genetic_algorithm",
    "brkga": "random_key_optimizer",
    "random_key_optimizer": "random_key_optimizer",
    "minimum_insertion": "minimum_insertion",
    "min_insertion": "minimum_insertion",
    "cheapest_insertion": "minimum_insertion",
    "tabu_search": "tabu_search",
    "tabu": "tabu_search",
    "promises_search": "promises_search",
    "promises": "promises_search",
}

_MENU = {
    "1": "branch_and_cut",
    "2": "column_generation",
    "3": "genetic_algorithm",
    "4": "random_key_optimizer",
    "5": "minimum_insertion",
    "6": "tabu_search",
    "7": "promises_search",
}


def solve(instance, method: str, **kwargs):
    """
    Solve a CVRP instance with the named methodology.

    Args:
        instance: A VRPModel produced by setup.read_vrp_file().
        method:   One of 'branch_and_cut', 'column_generation',
                  'genetic_algorithm', 'random_key_optimizer' (or 'brkga'),
                  'minimum_insertion' (or 'min_insertion'/'cheapest_insertion'),
                  'tabu_search' (or 'tabu'),
                  'promises_search' (or 'promises').
        **kwargs: Methodology-specific hyperparameters forwarded to the solver.

    Returns:
        A Solution object with routes and total cost.
    """
    method_key = _METHOD_ALIASES.get(method.lower())
    if method_key is None:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Valid options: {sorted(_METHOD_ALIASES)}"
        )

    if method_key == "branch_and_cut":
        solver = BranchAndCut(instance, **kwargs)
    elif method_key == "column_generation":
        solver = ColumnGeneration(instance, **kwargs)
    elif method_key == "genetic_algorithm":
        solver = GeneticAlgorithm(instance, **kwargs)
    elif method_key == "minimum_insertion":
        solver = MinimumInsertion(instance, **kwargs)
    elif method_key == "tabu_search":
        solver = _load_local_search("tabu_search").TabuSearch(instance, **kwargs)
    elif method_key == "promises_search":
        solver = _load_local_search("promises_search").PromisesSearch(instance, **kwargs)
    else:  # random_key_optimizer
        solver = RandomKeyOptimizer(instance, **kwargs)

    return solver.solve()


def run(choice: str, model) -> None:
    """Interactive menu dispatcher used by main.py."""
    method_key = _MENU.get(choice)
    if choice == "8":
        print("Exiting program.")
        return
    if method_key is None:
        print("Invalid option. Program terminated.")
        return

    label = {
        "branch_and_cut": "Branch and Cut",
        "column_generation": "Column Generation",
        "genetic_algorithm": "Genetic Algorithm",
        "random_key_optimizer": "Random Key Optimizer (BRKGA)",
        "minimum_insertion": "Minimum Insertion",
        "tabu_search": "Tabu Search",
        "promises_search": "Promises Search",
    }[method_key]
    print(f"Solving with {label}...")

    defaults = {
        "branch_and_cut": {"time_limit": 300},
        "column_generation": {},
        "genetic_algorithm": {"population_size": 100, "generations": 200},
        "random_key_optimizer": {"population_size": 100, "generations": 300},
        "minimum_insertion": {"strategy": "sequential", "seed_rule": "farthest"},
        # Empty: tabu_search.py hosts its own parameter defaults at the top.
        "tabu_search": {},
        # Empty: promises_search.py hosts its own parameter defaults at the top.
        "promises_search": {},
    }
    solution = solve(model, method=method_key, **defaults[method_key])

    is_valid, validation_errors = validate_solution(model, solution)
    print(f"\nSolution Feasibility: {'VALID' if is_valid else 'INVALID'}")
    if not is_valid:
        for err in validation_errors:
            print(f"  - {err}")

    print("\nBest Solution Found:")
    print(f"Total Cost: {solution.cost:.2f}")
    print(f"Number of Routes: {len(solution.routes)}")
    for r in solution.routes:
        route_str = " -> ".join([str(n.id) for n in r.sequence_of_nodes])
        print(f"  Route {r.id} (Load: {r.load}): {model.depot.id} -> {route_str} -> {model.depot.id}")

