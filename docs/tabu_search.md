# Tabu Search for CVRP

## Conceptual Overview

Tabu search is a **trajectory-based metaheuristic** introduced by **Fred Glover (1986)** and given its standard form in Glover & Laguna (1997). It takes plain local search and fixes its fatal flaw: a descent method stops the moment no neighbour improves on the current solution, leaving it stuck in the first local optimum it stumbles into.

Tabu search escapes by **always moving to the best solution in the neighbourhood, even when that move makes things worse**. On its own that would immediately cycle — the best move out of a local optimum is uphill, and the best move from there is straight back down again. So the method keeps a short-term memory, the **tabu list**, recording attributes of recent moves and forbidding their reversal for a number of iterations called the **tenure**. The search is pushed to explore elsewhere rather than oscillate.

For the CVRP the classical reference is **Taillard (1993)**, whose tabu search held many best-known solutions for years, and **Gendreau, Hertz & Laporte's TABUROUTE (1994)**.

---

## The Three Neighbourhoods

Each iteration scans **all three** operators in `operators/` in full and applies the single cheapest admissible move among them — steepest descent across the union of the neighbourhoods:

| Operator | Move | Size | Changes route membership? |
|---|---|---|---|
| `relocate.py` | Move one customer to any position in any route | $O(n^2)$ | Yes — can empty a route |
| `swap.py` | Exchange two customers between any positions | $O(n^2)$ | Yes |
| `2-opt.py` | Reverse a segment inside one route | $O(n^2)$ | No |

Relocate and swap do the inter-route work — moving load between vehicles — while 2-opt cleans up the ordering inside a route. The combination matters: 2-opt alone can never fix a bad assignment of customers to vehicles, and relocate/swap alone leave crossing edges behind.

---

## The Tabu Attribute: Arcs

The tabu attribute here is the **arc** (the link between two consecutive nodes), which is the standard choice for routing problems.

When a move is applied, every arc it **destroys** is banned for `tabu_tenure` iterations. A candidate move is tabu when it would **create** any currently banned arc. This is exactly what stops the search undoing what it just did: if relocating customer 14 destroys the links `8-14` and `11-14`, those links cannot be rebuilt for the next 100 iterations, so 14 cannot simply slide back.

The ban is enforced **inside the operators**, not by filtering afterwards. `find_best_move` takes a `tabu_arcs` set and never evaluates a candidate that would recreate one, so what comes back is the best *admissible* move rather than the best move overall.

### Arcs are undirected

Distances in this project are symmetric, and 2-opt reverses whole segments, so `(a,b)` and `(b,a)` are the same physical link and share a single tabu entry:

```python
def arc(a: int, b: int) -> Arc:
    return (a, b) if a <= b else (b, a)
```

This is also why a 2-opt move only ever changes **two** arcs: every link interior to the reversed segment is traversed in the opposite direction but is the same undirected arc.

### Bookkeeping is on traversal counts, not presence

One subtlety is easy to get wrong. A route serving a single customer, `depot -> u -> depot`, drives the link `(depot, u)` **twice**. Inserting a customer ahead of `u` consumes one of those two traversals while leaving the link in place.

Accounting on mere presence would miss that change entirely and let the search immediately undo the insertion. The operators therefore net arcs as **multisets**:

```python
removed_count = Counter(a for a in removed if a[0] != a[1])
added_count = Counter(a for a in added if a[0] != a[1])
return (
    tuple(sorted((removed_count - added_count).elements())),
    tuple(sorted((added_count - removed_count).elements())),
)
```

The same routine drops degenerate self-loops (predecessor and successor are both the depot when a route holds one customer) and cancels any link destroyed and immediately recreated — an untouched traversal must not become tabu, nor block the move that leaves it alone. That cancellation is what makes the **adjacent swap** correct: when `u` and `v` are neighbours, the `u-v` link survives the exchange merely reversed.

### Aspiration

A pure ban can hide a genuinely excellent move, because the tabu attribute is an *attribute*, not a whole solution — forbidding an arc forbids many solutions, most of which were never visited. The standard remedy is **aspiration by objective**: admit a tabu move anyway when it would produce a solution better than any seen so far.

```python
aspiration = best.cost - current.cost     # admit if current.cost + delta < best.cost
```

Setting `use_aspiration=False` makes the ban absolute. See the benchmarks below — on these instances that is often the *better* setting, which is not the textbook expectation.

---

## Algorithm Outline

```
current <- initial solution (minimum insertion, parallel)
best    <- current
tabu    <- {}                                  # arc -> iteration the ban lapses

while non_improving < max_non_improving:
    iteration += 1
    retire arcs whose ban has lapsed

    aspiration <- best.cost - current.cost     # or None if disabled
    move <- cheapest admissible move over relocate ∪ swap ∪ 2-opt
    if move is None:                           # everything banned
        stop

    current <- move.apply(current)
    for a in move.removed_arcs:
        tabu[a] <- iteration + tabu_tenure

    if current.cost < best.cost:
        best <- current
        non_improving <- 0
    else:
        non_improving += 1

return best
```

Two solutions are tracked throughout: `current`, which moves every iteration and is allowed to get worse, and `best`, the cheapest ever visited, which is what `solve()` returns.

The method is **fully deterministic** — steepest descent with deterministic tie-breaking, ties going to the earlier operator in scan order — so it takes no `seed` parameter.

---

## Key Parameters

Hosted at the top of `tabu_search.py`:

| Parameter | Default | Effect |
|---|---|---|
| `MAX_NON_IMPROVING_ITERATIONS` | 1000 | Stop after this many consecutive iterations that fail to improve the best solution. The main effort dial. |
| `TABU_TENURE` | 100 | How many iterations an arc stays banned. |
| `USE_ASPIRATION` | True | Admit a tabu move that would beat the best solution so far. |
| `TIME_LIMIT_SECONDS` | None | Optional wall-clock cap, checked at the end of each iteration. |

All four are overridable per call: `solve(model, method="tabu_search", tabu_tenure=20)`.

**On tenure.** Roughly 3 arcs are banned per iteration, so a tenure of $T$ keeps about $3T$ arcs off the table at once. An instance with $n$ customers has $\binom{n+1}{2}$ possible arcs — 496 for A-n32-k5 — so the default tenure of 100 bans a large fraction of a small instance's entire arc set. The right tenure scales with instance size, and the sweep below shows exactly that.

---

## Strengths and Limitations

**Strengths**

- **Large, immediate gains over construction.** Cuts the minimum-insertion gap from 24–35 % down to 3–8 % in seconds.
- **Deterministic and reproducible.** No seed, no variance between runs — the same input always gives the same answer.
- **Feasible throughout.** Every operator checks capacity before admitting a move, so every intermediate solution is valid and the search can be stopped at any point.
- **Cheap memory.** The tabu list holds ~3×tenure arcs, independent of instance size.

**Limitations**

- **Parameter sensitivity is real and non-monotone.** On A-n32-k5 the tenure sweep runs 868 → 850 → **796** → 834 → 834. There is no safe "larger is better" direction, and the best setting depends on instance size.
- **Steepest descent is expensive.** Every iteration rescans all three neighbourhoods in full, $O(n^2)$ each. A candidate-list or granular neighbourhood (Toth & Vigo, 2003) restricted to near neighbours would cut this sharply.
- **Short-term memory only.** No diversification or intensification: no frequency-based penalties, no restarts from elite solutions, no path relinking. These are what separate a basic tabu search from Taillard's.
- **Fixed tenure.** Reactive tabu search (Battiti & Tecchiolli, 1994) adapts the tenure when cycling is detected, which would remove most of the tuning burden documented here.
- **Can stall.** If every move in every neighbourhood is banned the search stops early rather than aging the list out; with the default tenure on a small instance this is a live possibility.

---

## Usage

```python
from setup import read_vrp_file
from solver import solve

model = read_vrp_file("instances/A-n32-k5.vrp")

# Module defaults: 1000 non-improving iterations, tenure 100, aspiration on
solution = solve(model, method="tabu_search")

# Tuned for a small instance
solution = solve(model, method="tabu", tabu_tenure=20, use_aspiration=False)

# Warm-start from a solution you already have
solution = solve(model, method="tabu_search", initial_solution=existing)
```

Registered aliases in `solver.py`: `tabu_search`, `tabu`. It is menu option **6** in `main.py`. When no `initial_solution` is given it builds one with the parallel minimum-insertion heuristic.

> **Import note.** `metaheuristics/local search/` contains a space, so it is not a valid Python package name and `tabu_search.py` cannot be imported normally. `solver.py` loads it by path with `importlib`, deferred to call time. Any other caller must do the same — see `_load_tabu_search()` in `solver.py`. Renaming the folder to `local_search` would remove the need.

---

## Benchmarking

All runs start from the parallel minimum-insertion solution and are deterministic, so each figure is a single run. Python 3.14, single core.

### Module defaults (`max_non_improving=1000`, `tabu_tenure=100`, aspiration on)

| Instance | Customers | Start | Tabu | Routes | BKS | Gap | Runtime |
|---|---|---|---|---|---|---|---|
| A-n32-k5 | 31 | 969 | 850 | 5 | 784 | +8.4 % | 2.5 s |
| A-n54-k7 | 53 | 1561 | 1250 | 8 | 1167 | +7.1 % | 2.6 s |
| P-n101-k4 | 100 | 734 | 701 | 4 | 681 | +2.9 % | 5.2 s |
| M-n200-k17 | 199 | 1724 | 1368 | 17 | 1275 | +7.3 % | 12.4 s |

All eight solutions (four here, four tuned below) pass `solver.validate_solution`.

### Tenure × aspiration sweep

Gap to BKS; best per instance in bold.

| Tenure | A-n32-k5 on | A-n32-k5 off | A-n54-k7 on | A-n54-k7 off | P-n101-k4 on | P-n101-k4 off |
|---|---|---|---|---|---|---|
| 5 | 10.7 % | 10.7 % | 7.1 % | 11.6 % | 2.8 % | 2.8 % |
| 10 | 8.4 % | 10.7 % | 7.1 % | 3.7 % | 2.6 % | 2.6 % |
| 20 | 8.4 % | **1.5 %** | 3.2 % | **2.1 %** | **1.8 %** | **1.8 %** |
| 50 | 8.4 % | 6.4 % | 2.6 % | 4.0 % | 2.9 % | 3.2 % |
| 100 | 8.4 % | 6.4 % | 7.1 % | 11.2 % | 2.9 % | 3.2 % |

On M-n200-k17 the ordering inverts: tenure 20 gives 1430 (+12.2 %), tenure 50 gives 1379 (+8.2 %), and the default 100 gives **1368 (+7.3 %)**.

**Observations**

- **Tenure 20 is the sweet spot on instances up to ~100 customers**, and the default of 100 is too long for them — it bans so much of a small instance's arc set that the search is starved of moves. On the 199-customer instance the relationship reverses and a long tenure wins, which is consistent with tenure needing to scale against the number of arcs.
- **Turning aspiration off often helps**, most sharply on A-n32-k5 (1.5 % vs 8.4 %). This runs against the textbook advice. The likely reason is that on small instances aspiration readmits moves leading straight back toward the region the tabu list was trying to push the search out of, undercutting diversification — the very thing a short-ish tenure is there to provide.
- **The response surface is bumpy, not monotone.** 868 → 850 → 796 → 834 → 834 across the tenure sweep. Anyone tuning this should sweep rather than hill-climb, and the sensitivity is a strong argument for a reactive tenure.
- **Best results found**: A-n32-k5 **796 (+1.5 %)**, A-n54-k7 **1192 (+2.1 %)**, P-n101-k4 **693 (+1.8 %)**, M-n200-k17 **1368 (+7.3 %)** — all within seconds, and a large improvement on the 23.6–35.2 % gaps of the construction heuristic alone.

---

## References

1. **Glover, F.** (1986). *Future paths for integer programming and links to artificial intelligence.* Computers & Operations Research, 13(5), 533–549.
2. **Glover, F. & Laguna, M.** (1997). *Tabu Search.* Kluwer Academic Publishers.
3. **Taillard, É.** (1993). *Parallel iterative search methods for vehicle routing problems.* Networks, 23(8), 661–673.
4. **Gendreau, M., Hertz, A. & Laporte, G.** (1994). *A tabu search heuristic for the vehicle routing problem.* Management Science, 40(10), 1276–1290.
5. **Battiti, R. & Tecchiolli, G.** (1994). *The reactive tabu search.* ORSA Journal on Computing, 6(2), 126–140.
6. **Toth, P. & Vigo, D.** (2003). *The granular tabu search and its application to the vehicle routing problem.* INFORMS Journal on Computing, 15(4), 333–346.
7. **Cordeau, J.-F. & Laporte, G.** (2005). *Tabu search heuristics for the vehicle routing problem.* In Rego & Alidaee (eds.), *Metaheuristic Optimization via Memory and Evolution*, Springer, 145–163.
