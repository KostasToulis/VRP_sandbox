# Branch-and-Cut for CVRP

## Overview
Branch-and-Cut is an exact method for solving the Vehicle Routing Problem (VRP). It combines **Branch-and-Bound** with **Cutting Planes**. 

In this implementation, we use an iterative approach using an Integer Linear Programming (ILP) formulation. We start with a relaxed set of constraints (assignments and flow conservation) and iteratively add "lazy constraints" or cuts when we detect violations in the solution found.

## Formulation
We use a 2-index vehicle flow formulation.
- **Variables**: $x_{ij} \in \{0, 1\}$ is 1 if a vehicle travels from customer $i$ to $j$.
- **Objective**: Minimize total distance $\sum c_{ij} x_{ij}$.

### Core Constraints
1. **Degree constraints**: Every customer must have exactly one incoming and one outgoing edge.
   $$ \sum_{j} x_{ij} = 1 \quad \forall i \in Customers $$
   $$ \sum_{j} x_{ji} = 1 \quad \forall i \in Customers $$

2. **Flow at Depot**: The number of vehicles leaving the depot must equal the number entering.
   $$ \sum_{j} x_{0j} = \sum_{j} x_{j0} $$

### Subtour Elimination & Capacity Cuts
Initially, the problem is solved allowing disconnected cycles (subtours). We then check the solution:

1. **Subtour Elimination**: If we find a cycle $S$ that does not include the depot, it is invalid. We add a cut to break it:
   $$ \sum_{i,j \in S} x_{ij} \le |S| - 1 $$
   This forces at least one edge to leave the set $S$.

2. **Capacity Cuts**: If we find a route (connected to depot) whose total demand exceeds capacity $C$, we add a Generalized Subtour Elimination Constraint (GSEC). For a set of customers $S$ with total demand $D(S)$, the minimum number of vehicles needed is $k(S) = \lceil D(S)/C \rceil$.
   $$ \sum_{i,j \in S} x_{ij} \le |S| - k(S) $$
   This forces the route to be split into at least $k(S)$ pieces (or handled by that many vehicles).

## Code Example

```python
# Iterative loop
while True:
    prob.solve()
    
    # Extract edges from solution
    edges = [(i,j) for i in nodes for j in nodes if value(x[i,j]) > 0.9]
    subtours = find_subtours(edges)
    
    cuts_added = 0
    for S in subtours:
        if depot not in S:
             # Add Subtour Elimination Cut
             prob += lpSum(x[i,j] for i in S for j in S) <= len(S) - 1
             cuts_added += 1
        else:
             # Check Capacity
             load = sum(demand[i] for i in S)
             if load > capacity:
                 k = ceil(load / capacity)
                 prob += lpSum(x[i,j] for i in S for j in S) <= len(S) - k
                 cuts_added += 1
    
    if cuts_added == 0:
        break # Optimal solution found
```
