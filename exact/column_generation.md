# Column Generation for CVRP

## Overview
Column Generation is an exact method (or heuristic framework) used for solving large-scale linear programming problems with many variables (columns). In the context of VRP, it is used to solve the **Set Partitioning** formulation of the problem.

Instead of enumerating all exponentially many possible routes, we start with a small set of routes and iteratively generate new "promising" routes (columns) that can improve the current solution.

## Components

### 1. Master Problem (Restricted Master Problem - RMP)
The Master Problem is a Set Partitioning (or Set Covering) problem that selects the best combination of known routes to satisfy customer demand at minimum cost.
- **Variables**: $\lambda_r \ge 0$ (selection of route $r$).
- **Constraints**: Every customer must be visited at least once.
- **Objective**: Minimize total route cost.

Solving the linear relaxation of RMP provides **Dual Variables** ($\pi_i$) for each customer. These duals represent the "price" or "saving" of visiting a customer.

### 2. Pricing Problem (Subproblem)
The Pricing Problem tries to find a new route with negative **Reduced Cost**.
$$\bar{c}_r = c_r - \sum_{i \in r} \pi_i < 0$$
Where $c_r$ is the actual travel cost of the route and $\pi_i$ are the dual values from the Master Problem.

This is a **Shortest Path Problem with Resource Constraints (SPPRC)** (specifically Capacity). We search for a path starting and ending at the depot such that its "reduced cost" is negative.

### 3. Algorithm Loop
1. **Initialize**: Start with simple dummy routes (e.g., one vehicle per customer).
2. **Loop**:
   - Solve RMP (LP Relaxation) to get Duals.
   - Solve Pricing Problem using Duals to find a new route.
   - If a route with negative reduced cost is found, add it to RMP.
   - If not, the LP solution is optimal. Stop.
3. **Integer Solution**: After convergence, solve the RMP as an Integer Problem (MIP) using the generated columns to get a feasible routing plan.

## Code Example

```python
# Master Problem loop
while True:
    # 1. Solve LP
    prob.solve()
    duals = {node: constraint.pi for node in customers}
    
    # 2. Solve Pricing
    # Edge cost = distance(i,j) - dual[j]
    new_route = solve_pricing(duals)
    
    # 3. Check Reduced Cost
    if new_route.reduced_cost < 0:
        add_column(new_route)
    else:
        break # Optimality reached

# 4. Final Integer Solve
prob_mip.solve()
```
