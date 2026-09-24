# Minimum Insertion Heuristic for CVRP

## Conceptual Overview

The **minimum insertion** heuristic (also called *cheapest insertion*) is one of the classical **route-construction** heuristics for vehicle routing. It descends from the cheapest-insertion rule for the TSP (**Rosenkrantz, Stearns & Lewis, 1977**) and was adapted to the capacitated VRP most influentially by **Mole & Jameson (1976)** and later generalised into the time-window insertion framework of **Solomon (1987)**.

The idea is deliberately simple. Rather than appending customers to the end of a tour — the way nearest-neighbour does — the heuristic considers inserting each unrouted customer **between every pair of consecutive nodes already on a route**, and commits the insertion that increases the route length by the least. Because a customer can be spliced into the middle of an existing path, the resulting routes are far less prone to the long "return leg" that plagues purely sequential construction.

Minimum insertion is a **greedy, deterministic, single-pass** method. It produces no optimality guarantee, but it is fast ($O(n^3)$ worst case), always feasible by construction, and yields a well-structured starting solution for local search or metaheuristics.

---

## The Insertion Criterion

For an unrouted customer $u$, and a consecutive pair $(i, j)$ on a partial route, the **insertion cost** is the detour that $u$ causes:

$$\delta(i, u, j) = d(i, u) + d(u, j) - \mu \cdot d(i, j)$$

With $\mu = 1$ this is the pure detour — the extra distance travelled relative to going straight from $i$ to $j$. By the triangle inequality $\delta \geq 0$ always holds in a Euclidean instance.

The parameter $\mu$ (Mole & Jameson's *route shape* parameter) rewards or penalises the removed edge:

- $\mu = 1$ — the classical minimum-insertion criterion.
- $\mu > 1$ — extra credit for breaking a long edge; routes become more circular/compact.
- $\mu < 1$ — long edges are cheaper to keep; routes become more radial and elongated.

The endpoints of a route are handled by treating the depot as both the predecessor of the first customer and the successor of the last, so insertions at the head and tail of a route compete on the same footing as interior ones.

**Feasibility** is enforced at evaluation time: a candidate $u$ is only considered for a route whose residual capacity $C - \text{load}$ is at least $q_u$. Every insertion that is ever committed is therefore capacity-feasible, and no repair step is needed.

---

## Algorithm Outline

Two construction strategies are implemented; they differ only in how many routes compete for each insertion.

### Sequential (one route at a time)

```
unrouted <- all customers
while unrouted is not empty:
    seed <- select_seed(unrouted)             # opens a new route
    route <- [seed];  load <- demand(seed)
    loop:
        (delta*, u*, p*) <- min over u in unrouted with demand(u) <= C - load,
                                     over positions p in route,
                                     of delta(prev(p), u, next(p))
        if no feasible (u, p) exists:  break   # route is closed
        insert u* into route at position p*
        load <- load + demand(u*)
        remove u* from unrouted
    emit route
```

### Parallel (all routes compete)

```
K <- ceil(total_demand / C)                   # bin-packing lower bound on vehicles
seeds <- K mutually dispersed customers
routes <- one single-customer route per seed

while unrouted is not empty:
    (delta*, u*, p*, r*) <- global minimum over ALL open routes r
    if no feasible insertion exists anywhere:
        open a new route seeded with select_seed(unrouted)   # overflow
        continue
    insert u* into route r* at position p*
```

The sequential variant fills each vehicle greedily before moving on, which tends to use **fewer vehicles**. The parallel variant lets a customer choose its best route globally, which usually gives **lower total distance** — the results below bear this out on three of the four benchmark instances.

### Seeding

The first customer of a route anchors its direction, so the seed rule matters:

| Rule | Behaviour |
|---|---|
| `farthest` (default) | The unrouted customer furthest from the depot. Remote customers are the most expensive to serve, so committing them first avoids stranding them in an otherwise-full route. |
| `max_demand` | The unrouted customer with the largest demand. Packs the awkward, bulky customers early. |
| `random` | Uniform random choice from a seeded local RNG — useful for multi-start / GRASP-style restarts. |

For the parallel strategy the first seed follows the rule above and the remaining $K-1$ seeds are chosen by **max-min dispersion**: each new seed is the customer whose distance to its *closest* already-chosen seed is largest. This spreads the initial routes across the map instead of clustering them in one corner.

---

## Complexity

Each insertion scans every unrouted customer against every position of every open route. With $n$ customers, one insertion costs $O(n \cdot L)$ for a route of length $L$ (sequential) or $O(n \cdot n)$ across all routes (parallel), and $n$ insertions are performed — giving $O(n^3)$ in the worst case.

The distance matrix is pre-computed once in the constructor (`self._dist`), since `VRPModel.get_distance` recomputes a square root and a rounding on every call and the criterion is evaluated millions of times on large instances. In practice $n = 200$ solves in under 0.2 s.

---

## Key Parameters

| Parameter | Default | Effect | Typical range |
|---|---|---|---|
| `strategy` | `"sequential"` | `"sequential"` fills routes one at a time; `"parallel"` grows all routes at once | — |
| `seed_rule` | `"farthest"` | How the customer opening a new route is chosen | `farthest`, `max_demand`, `random` |
| `mu` | `1.0` | Route-shape parameter in $\delta = d(i,u) + d(u,j) - \mu \, d(i,j)$ | 0.5 – 2.0 |
| `num_routes` | `None` | Parallel only: number of routes seeded up front. Defaults to $\lceil \sum q_i / C \rceil$ | $\geq \lceil \sum q_i / C \rceil$ |
| `seed` | `None` | RNG seed; only affects `seed_rule="random"` | any int |

`num_routes` below the bin-packing bound is not an error — the construction simply opens overflow routes as needed, so the final vehicle count can exceed the requested one.

---

## Strengths and Limitations

**Strengths**

- **Fast and deterministic.** Sub-second on every instance in `instances/`, with no parameters that need tuning to get a usable answer.
- **Always feasible.** Capacity is checked before an insertion is ever committed, so no repair or penalty machinery is required.
- **Good route topology.** Because customers are spliced into the interior of routes, the output contains far fewer crossing edges than nearest-neighbour or a greedy giant-tour split, which makes it a much better warm start for 2-opt / Or-opt.
- **Anytime, incremental.** The partial solution is feasible at every step, so the construction can be interrupted and completed by another method.

**Limitations**

- **Greedy myopia.** Each choice is locally cheapest and never reconsidered. Customers left until last are inserted into whatever capacity remains, often at high cost — visible as the small, expensive "leftover" route the sequential variant tends to produce (e.g. the single-customer Route 5 on A-n32-k5).
- **No optimality guarantee.** Expect roughly 20–45 % above BKS on Augerat-class instances. This is normal for a pure construction heuristic; Clarke-Wright savings typically lands closer.
- **Cubic scaling.** Fine to $n \approx 1000$; beyond that the full candidate scan should be restricted to a neighbour list.
- **Needs a companion improvement step.** The output is a *starting* solution. Pairing it with 2-opt/Or-opt, or using it to seed a metaheuristic, recovers most of the gap.

---

## Code Snippet Examples

The criterion itself is the whole method — evaluating one candidate against one position:

```python
prev_id = depot_id if position == 0 else seq[position - 1].id
next_id = depot_id if position == len(seq) else seq[position].id
delta = dist[prev_id][node.id] + dist[node.id][next_id] - mu * dist[prev_id][next_id]
```

Scanning for the cheapest feasible insertion into a single route, with the capacity filter applied before any distance work:

```python
def _best_insertion(self, seq, load, candidates):
    residual = self.model.capacity - load
    best = None

    for node in candidates:
        if node.demand > residual:          # capacity filter => always feasible
            continue
        for position in range(len(seq) + 1):
            delta = ...                      # criterion above
            if best is None or delta < best[0]:
                best = (delta, node, position)

    return best                              # None => nothing fits, close the route
```

Growing one route until it is full (the sequential strategy):

```python
seq, load = [seed_node], seed_node.demand
while unrouted:
    best = self._best_insertion(seq, load, unrouted)
    if best is None:
        break
    _, node, position = best
    seq.insert(position, node)
    load += node.demand
    unrouted.remove(node)
```

Max-min dispersion for the parallel seeds:

```python
pick = max(remaining, key=lambda n: min(self._dist[s.id][n.id] for s in seeds))
```

---

## Usage

```python
from setup import read_vrp_file
from solver import solve

model = read_vrp_file("instances/A-n32-k5.vrp")

# Default: sequential construction, farthest-from-depot seeding
solution = solve(model, method="minimum_insertion")

# Parallel construction, compact routes
solution = solve(model, method="minimum_insertion", strategy="parallel", mu=1.5)

print(f"Total cost: {solution.cost:.2f}")
```

Registered aliases in `solver.py`: `minimum_insertion`, `min_insertion`, `cheapest_insertion`. It is also menu option **5** in `main.py`.

---

## Benchmarking

Measured on this repository's instances, best-known solutions taken from `solutions/*.sol`. Defaults `seed_rule="farthest"`, `mu=1.0`; runtimes are the mean of five runs on Python 3.14, single core.

| Instance | Customers | BKS | Strategy | Cost | Routes | Gap | Runtime |
|---|---|---|---|---|---|---|---|
| A-n32-k5 | 31 | 784 | sequential | 1072 | 5 | +36.7 % | 0.001 s |
| A-n32-k5 | 31 | 784 | parallel | **969** | 5 | **+23.6 %** | 0.001 s |
| A-n54-k7 | 53 | 1167 | sequential | **1536** | 7 | **+31.6 %** | 0.001 s |
| A-n54-k7 | 53 | 1167 | parallel | 1561 | 8 | +33.8 % | 0.005 s |
| P-n101-k4 | 100 | 681 | sequential | 904 | 4 | +32.7 % | 0.010 s |
| P-n101-k4 | 100 | 681 | parallel | **734** | 4 | **+7.8 %** | 0.027 s |
| M-n200-k17 | 199 | 1275 | sequential | 1841 | 16 | +44.4 % | 0.023 s |
| M-n200-k17 | 199 | 1275 | parallel | **1724** | 17 | **+35.2 %** | 0.168 s |

All eight solutions pass `solver.validate_solution`.

**Observations**

- **Parallel wins on three of four instances**, by 3–25 percentage points, at roughly 2–7× the runtime. It is the better default when runtime is not the binding constraint.
- **Sequential uses fewer vehicles** (16 vs 17 on M-n200-k17, 7 vs 8 on A-n54-k7) because it packs each route to capacity before opening the next. If vehicle count is the objective, sequential is preferable.
- **P-n101-k4 is the standout** at +7.8 % — the P-series has few, large-capacity vehicles, which gives the insertion criterion many positions to choose from and plays directly to the method's strength. Conversely, instances with many tightly-loaded vehicles leave little slack and the greedy choices bind early.
- The absolute runtimes confirm the practical cost is negligible: the expensive part of any pipeline built on this will be the improvement step, not the construction.

---

## References

1. **Rosenkrantz, D. J., Stearns, R. E. & Lewis, P. M.** (1977). *An analysis of several heuristics for the traveling salesman problem.* SIAM Journal on Computing, 6(3), 563–581.
2. **Mole, R. H. & Jameson, S. R.** (1976). *A sequential route-building algorithm employing a generalised savings criterion.* Operational Research Quarterly, 27(2), 503–511.
3. **Solomon, M. M.** (1987). *Algorithms for the vehicle routing and scheduling problems with time window constraints.* Operations Research, 35(2), 254–265.
4. **Christofides, N., Mingozzi, A. & Toth, P.** (1979). *The vehicle routing problem.* In Christofides et al. (eds.), *Combinatorial Optimization*, Wiley, 315–338.
5. **Laporte, G. & Semet, F.** (2002). *Classical heuristics for the capacitated VRP.* In Toth & Vigo (eds.), *The Vehicle Routing Problem*, SIAM Monographs on Discrete Mathematics and Applications, 109–128.
6. **Toth, P. & Vigo, D.** (2014). *Vehicle Routing: Problems, Methods, and Applications* (2nd ed.), SIAM. Chapter 4.
