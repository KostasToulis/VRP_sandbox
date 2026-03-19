# Branch-and-Cut for CVRP

## Conceptual Overview

Branch-and-Cut is an **exact method** for solving combinatorial optimisation problems, guaranteeing a provably optimal solution. It is the algorithmic backbone of state-of-the-art MILP solvers such as CPLEX, Gurobi, and CBC.

For the CVRP it combines two classical ideas:

| Technique | Role |
|---|---|
| **Branch-and-Bound** | Systematically partitions the search space into sub-problems and prunes sub-trees whose lower bound exceeds the incumbent. |
| **Cutting Planes** | Tightens the LP relaxation at each node by adding valid inequalities (cuts) that remove fractional solutions without excluding any integer feasible point. |

The method was first applied rigorously to vehicle routing by **Padberg & Rinaldi (1991)** for the TSP and later extended to the CVRP by Fisher (1981) and Augerat et al. (1998).

---

## Mathematical Formulation

We use the **2-index vehicle flow formulation** over a complete directed graph $G = (V, A)$ where $V = \{0, 1, \ldots, n\}$, node $0$ is the depot, and nodes $1 \ldots n$ are customers.

**Decision Variables**

$$x_{ij} \in \{0, 1\}, \quad (i, j) \in A, \; i \neq j$$

$x_{ij} = 1$ if a vehicle travels directly from node $i$ to node $j$.

**Objective**

$$\min \sum_{(i,j) \in A} c_{ij} \, x_{ij}$$

**Degree Constraints** — each customer visited exactly once:

$$\sum_{j \neq i} x_{ij} = 1 \quad \forall i \in \{1, \ldots, n\}$$
$$\sum_{j \neq i} x_{ji} = 1 \quad \forall i \in \{1, \ldots, n\}$$

**Depot Flow Balance** — vehicles that leave must return:

$$\sum_{j=1}^{n} x_{0j} = \sum_{j=1}^{n} x_{j0}$$

---

## Algorithm Outline

```
1.  Build ILP with degree & depot-flow constraints only.
2.  LOOP:
    a. Solve current LP/ILP relaxation.
    b. Extract edges where x[i,j] ≈ 1.
    c. Identify connected components (cycles) in the solution graph.
    d. For each cycle S not containing the depot:
           Add Subtour Elimination Cut (SEC).
    e. For each route (cycle through depot) violating capacity:
           Add Generalized Subtour Elimination Constraint (GSEC).
    f. If no cuts were added → solution is feasible and optimal. STOP.
3.  Return the optimal solution.
```

---

## Cut Types

### Subtour Elimination Constraint (SEC)
If a cycle $S \subseteq \{1,\ldots,n\}$ does not pass through the depot, it is infeasible. The cut forces at least one edge to leave $S$:

$$\sum_{i \in S} \sum_{j \in S,\, j \neq i} x_{ij} \leq |S| - 1$$

### Generalized Subtour Elimination Constraint (GSEC / Capacity Cut)
For a customer set $S$ with total demand $D(S)$, the minimum number of vehicles required is $k(S) = \lceil D(S) / C \rceil$. The cut forces the routes over $S$ to be decomposed into at least $k(S)$ sub-routes:

$$\sum_{i \in S} \sum_{j \in S,\, j \neq i} x_{ij} \leq |S| - k(S)$$

---

## Code Snippets

### Building Decision Variables and Objective

```python
# Separate variable creation from objective construction
x = {}
for i in node_ids:
    for j in node_ids:
        if i != j:
            x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)

# Add objective as a single lpSum — do NOT use += inside a loop,
# as each += would REPLACE the objective in PuLP.
prob += pulp.lpSum(
    model.get_distance(i, j) * x[i, j]
    for i in node_ids for j in node_ids if i != j
)
```

### Iterative Cut-and-Solve Loop

```python
while True:
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))

    edges = [(i, j) for i in node_ids for j in node_ids
             if i != j and pulp.value(x[i, j]) > 0.9]

    cycles = find_cycles(edges)          # follow next-node pointers
    cuts_added = 0

    for cycle in cycles:
        if depot_id not in cycle:
            # SEC: cut disconnected subtour
            prob += pulp.lpSum(x[i, j] for i in cycle
                               for j in cycle if i != j) <= len(cycle) - 1
            cuts_added += 1

    if cuts_added == 0:
        routes, capacity_ok = extract_routes(edges, depot_id)
        if capacity_ok:
            break  # optimal & feasible
        for route_nodes in routes:
            load = sum(demand[n] for n in route_nodes)
            if load > capacity:
                k = math.ceil(load / capacity)
                prob += pulp.lpSum(x[i, j] for i in route_nodes
                                   for j in route_nodes if i != j) \
                       <= len(route_nodes) - k
                cuts_added += 1
        if cuts_added == 0:
            break
```

### Cycle Detection via Next-Node Pointers

```python
def find_cycles(edges):
    next_node = {u: v for u, v in edges}   # unique successor (degree = 1)
    processed = set()
    cycles = []

    for start in next_node:
        if start in processed:
            continue
        cycle, curr = [], start
        while curr not in processed:
            processed.add(curr)
            cycle.append(curr)
            curr = next_node.get(curr)
            if curr is None:
                break
        cycles.append(cycle)

    return cycles
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `time_limit` | 300 s | Maximum solver wall-clock time. Increase for harder instances. |

---

## Strengths and Limitations

**Strengths**
- Provides a **provably optimal** solution.
- Well-understood theoretically; decades of research on valid inequalities.
- Leverages mature LP/ILP solvers (CBC, CPLEX, Gurobi).

**Limitations**
- **Exponential worst-case complexity** — impractical for instances with > ~50–100 customers without strong cuts and preprocessing.
- Quality heavily depends on the strength of the cuts added.
- The pricing of cuts (separation problem) can itself be NP-hard in general.
- This implementation is a simplified sandbox version; production-grade B&C requires branch-and-bound trees, stronger cuts (comb inequalities, etc.), and LP re-solving at each tree node.

---

## Benchmarking

Test against the following Augerat instances and compare to Best-Known Solutions (BKS):

| Instance | Customers | BKS | Expected Gap |
|---|---|---|---|
| A-n32-k5 | 31 | 784 | < 5% |
| A-n45-k6 | 44 | 944 | < 10% |
| A-n54-k7 | 53 | 1167 | > 10% (time-limited) |

---

## References

1. **Padberg, M. & Rinaldi, G.** (1991). *A branch-and-cut algorithm for the resolution of large-scale symmetric traveling salesman problems.* SIAM Review, 33(1), 60–100.
2. **Fisher, M. L.** (1981). *The Lagrangian relaxation method for solving integer programming problems.* Management Science, 27(1), 1–18.
3. **Augerat, P. et al.** (1998). *Computational results with a branch-and-cut code for the capacitated vehicle routing problem.* IMAG Research Report.
4. **Laporte, G.** (1992). *The vehicle routing problem: An overview of exact and approximate algorithms.* European Journal of Operational Research, 59(3), 345–358.
