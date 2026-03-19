# Column Generation for CVRP

## Conceptual Overview

Column Generation is an exact LP technique for solving problems with an **exponentially large number of variables (columns)**. Rather than constructing the full LP upfront, we start with a small subset of variables and iteratively generate only those that can improve the current solution.

For the CVRP the natural formulation enumerates all feasible routes (one variable per route). With $n$ customers this is exponential in $n$, so explicit enumeration is infeasible. Column Generation solves this tractably.

The idea was formalised by **Dantzig & Wolfe (1960)** via their decomposition principle, and first applied to vehicle routing by **Desrochers, Desrosiers & Solomon (1992)** who introduced the SPPRC pricing subproblem with time windows.

---

## Mathematical Formulation

### Set Partitioning Master Problem

Let $\Omega$ be the set of all feasible routes. Each route $r \in \Omega$ has cost $c_r$ and a binary incidence vector $a_r \in \{0,1\}^n$ where $a_{ri} = 1$ if route $r$ visits customer $i$.

**Variables**: $\lambda_r \in \{0, 1\}$ — 1 if route $r$ is used (relaxed to $\lambda_r \geq 0$ for the LP).

$$\min \sum_{r \in \Omega} c_r \lambda_r$$

$$\text{s.t.} \quad \sum_{r \in \Omega} a_{ri} \lambda_r \geq 1 \quad \forall i \in \{1,\ldots,n\}$$

$$\lambda_r \geq 0 \quad \forall r \in \Omega$$

### Reduced Cost and Optimality Condition

Solving the LP relaxation of the **Restricted Master Problem (RMP)** — over the current column subset — yields dual variables $\pi_i$ for each customer constraint. For a route $r$ the **reduced cost** is:

$$\bar{c}_r = c_r - \sum_{i \in r} \pi_i$$

The LP is optimal when no column with $\bar{c}_r < 0$ exists.

### Pricing Subproblem (SPPRC)

Find a feasible route $r$ (depot → customers → depot, capacity $\leq C$) minimising the reduced cost:

$$\min_{r} \left( c_r - \sum_{i \in r} \pi_i \right)$$

This is an **Elementary Shortest Path Problem with Resource Constraints (ESPPRC)**. The capacity is the single resource. Exact label-setting algorithms run in pseudo-polynomial time; this implementation uses a heuristic beam-search DFS for sandbox performance.

---

## Algorithm Outline

```
1.  Initialise columns: one single-customer route per customer.
2.  LOOP:
    a. Solve LP relaxation of RMP  →  get dual values π.
    b. Solve pricing subproblem using π  →  candidate route r*.
    c. If reduced_cost(r*) < −ε:
           Add r* to the column pool.
       Else:
           LP is optimal.  BREAK.
3.  Solve RMP as MIP restricted to generated columns
    (Restricted Master Heuristic — not provably optimal without Branch-and-Price).
4.  Return integer solution.
```

---

## Code Snippets

### Master Problem Setup

```python
prob = pulp.LpProblem("RMP", pulp.LpMinimize)

# One binary/continuous variable per generated route
cat = pulp.LpContinuous if relax else pulp.LpBinary
x = [pulp.LpVariable(f"x_{r.id}", lowBound=0, cat=cat) for r in columns]

# Objective: minimise total route cost
prob += pulp.lpSum(columns[i].cost * x[i] for i in range(len(columns)))

# Coverage constraints: every customer served at least once
for node in customers:
    prob += pulp.lpSum(
        x[i] for i, col in enumerate(columns) if node in col.sequence_of_nodes
    ) >= 1, f"cov_{node.id}"
```

### Extracting Dual Variables

```python
prob.solve(pulp.PULP_CBC_CMD(msg=0))

# Dual (shadow price) for each coverage constraint
duals = {
    node.id: prob.constraints[f"cov_{node.id}"].pi
    for node in customers
}
```

### Pricing Problem — Beam-Search DFS

```python
# Edge reduced cost: travelling depot→j→… incurs dist(i,j) − π[j]
stack = [(depot, load=0, red_cost=0.0, visited=set(), path=[])]
best_red_cost, best_route = -1e-9, None

while stack:
    curr, load, red_cost, visited, path = stack.pop()

    # Try closing route: add return-to-depot arc
    close_cost = red_cost + dist(curr, depot)
    if path and close_cost < best_red_cost:
        best_red_cost = close_cost
        best_route = Route(path)

    # Extend to unvisited neighbours within capacity
    for nxt in sorted_neighbours[curr.id][:beam_width]:
        if nxt.id not in visited and load + nxt.demand <= capacity:
            new_red = red_cost + dist(curr, nxt) - duals[nxt.id]
            stack.append((nxt, load + nxt.demand, new_red,
                          visited | {nxt.id}, path + [nxt]))
```

### Main Column Generation Loop

```python
while True:
    duals, lp_obj = solve_master(relax=True)
    new_route = solve_pricing(duals)

    if new_route and new_route.reduced_cost < -1e-5:
        columns.append(recalculate_true_cost(new_route))
    else:
        break  # LP optimality reached

# Final integer solve over generated columns
solution = solve_master(relax=False)
```

---

## Key Parameters

| Parameter | Description | Notes |
|---|---|---|
| `max_steps` | Pricing DFS node limit (default 5000) | Trade-off between solution quality and speed |
| `beam_width` | Neighbours explored per node (default 5) | Increase for better pricing at cost of speed |
| Convergence tolerance | `−1e-5` reduced cost threshold | Smaller values → more columns |

---

## Strengths and Limitations

**Strengths**
- Provides the **LP lower bound** for the instance, useful for benchmarking heuristics.
- Scales naturally with the number of customers because only promising routes are ever instantiated.
- Forms the LP relaxation backbone of **Branch-and-Price**, the current state-of-the-art exact method for large CVRP instances.

**Limitations**
- The MIP step after column generation is a **heuristic** (Restricted Master Heuristic): it only considers generated columns, so integer optimality is not guaranteed unless combined with branching (Branch-and-Price).
- The ESPPRC pricing problem is itself NP-hard; this sandbox uses a heuristic DFS which may miss the most improving column.
- Dual variables require the LP to be solved to optimality at each iteration, which can be slow for large instances.
- PuLP/CBC may not always return accurate duals depending on solver configuration; commercial solvers (Gurobi, CPLEX) are preferred for production use.

---

## Benchmarking

| Instance | Customers | BKS | LP Bound | Integer Gap |
|---|---|---|---|---|
| A-n32-k5 | 31 | 784 | ≈ 760–775 | < 5% |
| A-n45-k6 | 44 | 944 | — | < 10% |
| A-n54-k7 | 53 | 1167 | — | varies |

---

## References

1. **Dantzig, G. B. & Wolfe, P.** (1960). *Decomposition principle for linear programs.* Operations Research, 8(1), 101–111.
2. **Desrochers, M., Desrosiers, J. & Solomon, M.** (1992). *A new optimization algorithm for the vehicle routing problem with time windows.* Operations Research, 40(2), 342–354.
3. **Barnhart, C. et al.** (1998). *Branch-and-price: Column generation for solving huge integer programs.* Operations Research, 46(3), 316–329.
4. **Irnich, S. & Desaulniers, G.** (2005). *Shortest path problems with resource constraints.* Column generation, Springer, 33–65.
