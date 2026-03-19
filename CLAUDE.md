# CVRP Sandbox — Claude Code Instructions

## Project Overview

This repository is a sandbox for implementing and benchmarking multiple solution methodologies in python for the **Capacitated Vehicle Routing Problem (CVRP)**. The goal is to systematically explore and compare approaches ranging from exact methods to heuristics, metaheuristics, and machine learning, using a shared set of standard benchmark instances.

---

## Repository Structure

```
vrp-sandbox/
├── CLAUDE.md                  # This file
├── instances/                 # Standard CVRP benchmark instances (e.g., Augerat, Christofides)
├── model.py                   # Problem domain classes (Node, Route, VRPModel, Solution, etc.)
├── setup.py                   # Instance parser that creates a vrp model from an instance      
├── solver.py                  # Solver containing the different solution methodology invocations
├── main.py                    # Entry point — selects and dispatches a methodology to solve a given 
|
├── exact/                     # Exact methods (guaranteed optimal solutions)
│   └── ...
├── heuristics/                # Construction and improvement heuristics
│   └── ...
├── metaheuristics/            # Population-based and trajectory-based metaheuristics
│   └── ...
├── ml/                        # Machine learning and learning-based approaches
│   └── ...
│
└── docs/                      # Methodology documentation (auto-generated .md files)
    └── ...
```

---

## Core Modules

### `model.py`
Defines the domain model for the CVRP. All methodology implementations must use these classes — do not redefine problem structure in solver files.

Key classes to define/maintain here:
- `Node` — a customer or depot with coordinates and demand
- `VRPInstance` — holds all nodes, the depot, vehicle capacity, and distance matrix
- `Route` — an ordered list of nodes served by one vehicle
- `Solution` — a collection of routes with a total cost

### `setup.py`
Parses a benchmark instance file (e.g., `.vrp` in TSPLIB format) and returns a fully constructed `VRPInstance`. All methodologies consume instances produced by this module.

### `solver.py`
Acts as the unified entry point. Accepts an instance and a methodology name, imports the appropriate module, runs the solver, and returns a `Solution`. Keep this file clean — it should orchestrate, not implement.

---

## Implementing a New Methodology

When asked to implement a new solution methodology, follow this protocol precisely.

### 1. Determine the category

Place the implementation in the correct subfolder:

| Category | Folder | Examples |
|---|---|---|
| Exact | `exact/` | Branch-and-Bound, Branch-and-Cut, Column Generation |
| Heuristic | `heuristics/` | Nearest Neighbor, Clarke-Wright Savings, 2-opt |
| Metaheuristic | `metaheuristics/` | Simulated Annealing, Tabu Search, Genetic Algorithm, ACO |
| Machine Learning | `ml/` | Pointer Networks, Attention Models, Graph Neural Nets |

### 2. Create the implementation file

Name the file descriptively in `snake_case` (e.g., `simulated_annealing.py`, `clarke_wright.py`, `pointer_network.py`).

Each implementation file must expose a single public function with this signature:

```python
def solve(instance: CVRPInstance, **kwargs) -> Solution:
    """
    Solve the given CVRP instance using <Methodology Name>.

    Args:
        instance: A fully constructed CVRPInstance from setup.py.
        **kwargs: Methodology-specific hyperparameters.

    Returns:
        A Solution object with routes and total cost.
    """
```

### 3. Register in `solver.py`

Add the new methodology to the dispatcher in `solver.py` so it can be called by name.

### 4. Create the documentation file

For every new methodology, create a corresponding `.md` file in the `docs/` folder (e.g., `docs/simulated_annealing.md`).

The documentation file must cover:
- **Conceptual overview** — what the methodology is, its origins, and its main ideas
- **Algorithm outline** — pseudocode or step-by-step description
- **Key parameters** — what they control and typical ranges
- **Strengths and limitations** — when to use it and what to watch out for
- **Code snippet examples** — short, self-contained examples illustrating the core logic (not the full implementation)
- **References** — seminal papers or textbook chapters

---

## Running a Methodology

```python
from setup import load_instance
from solver import solve

instance = load_instance("instances/A-n32-k5.vrp")
solution = solve(instance, method="simulated_annealing", max_iter=5000, T_init=100.0)

print(f"Total cost: {solution.total_cost:.2f}")
for i, route in enumerate(solution.routes):
    print(f"  Route {i+1}: {[n.id for n in route.nodes]}")
```

---

## Conventions and Standards

- **Language**: Python 3.10+. Use type hints throughout.
- **Dependencies**: Prefer the standard library and `numpy`. For ML approaches, `torch` is permitted. For exact methods, `pulp` or `ortools` are acceptable. Declare any non-standard dependency clearly at the top of the file.
- **No global state**: Solvers must be stateless and re-entrant.
- **Reproducibility**: Any stochastic method must accept a `seed` parameter.
- **Solution validity**: Every solver must return a feasible solution (capacity constraints respected, all customers visited exactly once). Include a validation step or assertion before returning.
- **Logging**: Use Python's `logging` module, not `print`, for progress output.

---

## Benchmarking

After implementing a methodology, test it against at least the following instances from the `instances/` folder and report the results in the methodology's `.md` doc:

- One small instance (~20–35 customers)
- One medium instance (~50–75 customers)
- One large instance (~100+ customers)

Report: total cost, runtime, and gap to best-known solution (BKS) where available.