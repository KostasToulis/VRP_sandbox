# Branch-and-Cut for CVRP

## Conceptual Overview

Branch-and-Cut is an **exact method** for solving combinatorial optimisation problems, guaranteeing a provably optimal solution. It is the algorithmic backbone of state-of-the-art MILP solvers such as CPLEX, Gurobi, and CBC.

For the CVRP it combines two classical ideas:

| Technique | Role |
| - | - |
| Branch-and-Bound | Systematically partitions the search space into sub-problems and prunes sub-trees whose lower bound exceeds the incumbent. |
| Cutting Planes | Tightens the LP relaxation at each node by adding valid inequalities (cuts) that remove fractional solutions without excluding any integer feasible point. |

The method was first applied rigorously to vehicle routing by **Padberg & Rinaldi (1991)** for the TSP and later extended to the CVRP by Fisher (1981) and Augerat et al. (1998).

---

## Mathematical Formulation

We use the **2-index vehicle flow formulation** over a complete directed graph $G = (V, A)$ where $V = \{0, 1, \ldots, n\}$, node $0$ is the depot, and nodes $1 \ldots n$ are customers.

### Decision Variables

$$x_{ij} \in \{0, 1\}, \quad (i, j) \in A, \; i \neq j$$

$x_{ij} = 1$ if a vehicle travels directly from node $i$ to node $j$.

### Objective

$$\min \sum_{(i,j) \in A} c_{ij} \, x_{ij}$$

**Degree Constraints** — each customer visited exactly once:

$$\sum_{j \neq i} x_{ij} = 1 \quad \forall i \in \{1, \ldots, n\}$$
$$\sum_{j \neq i} x_{ji} = 1 \quad \forall i \in \{1, \ldots, n\}$$

**Depot Flow Balance** — vehicles that leave must return:

$$\sum_{j=1}^{n} x_{0j} = \sum_{j=1}^{n} x_{j0}$$

---

## Algorithm Outline

```text
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

## Subtour Elimination: MTZ Constraints

This implementation uses **Miller–Tucker–Zemlin (MTZ)** constraints to eliminate subtours and enforce capacity in a single, static pass — no iterative cut-adding loop is required.

### Load Variables

For each customer $i \in \{1,\ldots,n\}$ introduce a continuous variable:

$$u_i \in [q_i,\; Q]$$

$u_i$ represents the **cumulative vehicle load after serving node $i$**. The lower bound $q_i$ ensures at least node $i$'s own demand has been loaded; the upper bound $Q$ is the capacity constraint.

### MTZ Constraint

For every ordered pair of customers $(i, j)$, $i \neq j$:

$$u_j - u_i \geq q_j - Q(1 - x_{ij})$$

**When $x_{ij} = 1$** (vehicle travels $i \to j$):

$$u_j \geq u_i + q_j$$

The load strictly increases along each route arc, making a closed customer-only cycle algebraically impossible (any cycle would require $u$ to both increase around the loop and return to its starting value — a contradiction).

**When $x_{ij} = 0$** (edge not used):

$$u_j - u_i \geq q_j - Q$$

Since $u_j \geq q_j$ and $u_i \leq Q$, this is always satisfied. The constraint is inactive.

> **Note on the depot**: Edges from the depot ($i = 0$) reduce to $u_j \geq q_j - Q(1-x_{0j})$, which is always dominated by $u_j \geq q_j$. These constraints are omitted.

---

## Code Snippets

### Building All Variables and Constraints

```python
prob = pulp.LpProblem("CVRP_Branch_and_Cut", pulp.LpMinimize)

# Routing variables
x = {
    (i, j): pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
    for i in node_ids for j in node_ids if i != j
}

# Objective — single lpSum (using += inside a loop replaces the objective in PuLP)
prob += pulp.lpSum(
    model.get_distance(i, j) * x[i, j]
    for i in node_ids for j in node_ids if i != j
)

# Degree constraints
for i in customer_ids:
    prob += pulp.lpSum(x[i, j] for j in node_ids if i != j) == 1
    prob += pulp.lpSum(x[j, i] for j in node_ids if i != j) == 1

# Depot flow balance
prob += (pulp.lpSum(x[depot_id, j] for j in customer_ids)
         == pulp.lpSum(x[j, depot_id] for j in customer_ids))
```

### MTZ Load Variables and Constraints

```python
demand = {n.id: n.demand for n in model.nodes}

# u[i] ∈ [q_i, Q] — cumulative load after serving customer i
u = {
    i: pulp.LpVariable(f"u_{i}", lowBound=demand[i], upBound=capacity)
    for i in customer_ids
}

# MTZ constraints: enforce load monotonicity and eliminate subtours
for i in customer_ids:
    for j in customer_ids:
        if i != j:
            prob += u[j] - u[i] >= demand[j] - capacity * (1 - x[i, j])
```

### Single Solve (No Loop Required)

```python
# One solve is sufficient — all constraints are static
status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))

if status == pulp.LpStatusOptimal:
    edges = [(i, j) for i in node_ids for j in node_ids
             if i != j and pulp.value(x[i, j]) > 0.9]
    routes = extract_routes(edges, depot_id)
    return build_solution(routes)
```

### Route Extraction via Next-Node Pointers

```python
def extract_routes(edges, depot_id):
    next_node = {u: v for u, v in edges}
    routes = []
    for start in (v for u, v in edges if u == depot_id):
        route, curr = [], start
        while curr != depot_id:
            route.append(curr)
            curr = next_node[curr]
        routes.append(route)
    return routes
```

---

## Key Parameters

| Parameter | Default | Description |
| - | - | - |
| `time_limit` | 300 s | Maximum solver wall-clock time. Increase for harder instances. |

---

## Strengths and Limitations

### Strengths

- Provides a **provably optimal** solution.
- Well-understood theoretically; decades of research on valid inequalities.
- Leverages mature LP/ILP solvers (CBC, CPLEX, Gurobi).
- MTZ constraints are simple to implement and eliminate the need for an iterative cut-adding loop.

### Limitations

- **Exponential worst-case complexity** — impractical for instances with > ~50–100 customers without strong cuts and preprocessing.
- MTZ gives a **weaker LP relaxation** than SEC/GSEC cutting planes; the solver may need more branching to close the gap.
- This implementation is a simplified sandbox version; production-grade B&C requires branch-and-bound trees, stronger cuts (comb inequalities, etc.), and LP re-solving at each tree node.

---

## Benchmarking

Test against the following Augerat instances and compare to Best-Known Solutions (BKS):

| Instance  | Customers | BKS  | Expected Gap          |
| --------- | --------- | ---- | --------------------- |
| A-n32-k5  | 31        | 784  | < 5%                  |
| A-n45-k6  | 44        | 944  | < 10%                 |
| A-n54-k7  | 53        | 1167 | > 10% (time-limited)  |

---

## References

1. **Padberg, M. & Rinaldi, G.** (1991). *A branch-and-cut algorithm for the resolution of large-scale symmetric traveling salesman problems.* SIAM Review, 33(1), 60–100.
2. **Miller, C. E., Tucker, A. W. & Zemlin, R. A.** (1960). *Integer programming formulation of traveling salesman problems.* Journal of the ACM, 7(4), 326–329.
3. **Fisher, M. L.** (1981). *The Lagrangian relaxation method for solving integer programming problems.* Management Science, 27(1), 1–18.
4. **Augerat, P. et al.** (1998). *Computational results with a branch-and-cut code for the capacitated vehicle routing problem.* IMAG Research Report.
5. **Laporte, G.** (1992). *The vehicle routing problem: An overview of exact and approximate algorithms.* European Journal of Operational Research, 59(3), 345–358.
