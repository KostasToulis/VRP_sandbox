# Promises Search for CVRP

## Conceptual Overview

Promises search is a tabu search whose tabu list carries **more than the forbidden attribute** — it carries a number. When a move deletes an arc $(i,j)$, the list records the arc together with the **cost of the solution it was deleted from**:

$$p(i,j) \;=\; \text{cost of the solution that contained } (i,j) \text{ at the moment it was removed}$$

A move that re-introduces $(i,j)$ is forbidden **unless the solution it produces costs strictly less than $p(i,j)$**.

The reasoning behind the name: when the search abandons an arc, it does so from a solution of some known quality. Bringing the arc back is a *promise* to do better than that. If the move cannot keep the promise, it is refused. If it can, the arc was never really the problem and the move goes through.

Every arc begins with an implicit promise of $\infty$ — absent from the mapping means free — and every `p_iter` iterations the entire mapping is re-initialised to $\infty$, wiping the memory. `p_iter` therefore plays the role that tenure plays in classical tabu search.

---

## Why the Aspiration Criterion Is Built In

Classical tabu search (see `docs/tabu_search.md`) bans an arc **outright** for a fixed tenure. That ban is blunt: the tabu attribute is an *attribute*, not a solution, so forbidding one arc forbids an enormous set of solutions, almost none of which were ever visited. To stop the list hiding genuinely good moves, a separate **aspiration criterion** is bolted on — usually "admit a tabu move if it beats the best solution ever found".

The promise rule needs no such bolt-on, because **the rule is itself an aspiration criterion**:

| | Classical tabu | Promises |
|---|---|---|
| Stored per arc | nothing (membership only) | cost at deletion, $p(i,j)$ |
| Ban | absolute for `tenure` iterations | conditional on cost |
| Aspiration | separate, global (beat global best) | intrinsic, per-arc (beat this arc's label) |
| Threshold | one number for the whole search, moving as `best` improves | one number per arc, fixed when the arc was dropped |
| Memory released | one arc at a time as each tenure expires | all arcs at once every `p_iter` iterations |

The per-arc threshold is the substantive difference. A global aspiration asks "is this the best solution ever?", which is a very high bar that almost nothing clears, so in practice the global ban dominates. The promise asks "is this better than where we were when we dropped *this particular* arc?", which is a bar calibrated to each arc's own history — much easier to clear for arcs abandoned early from poor solutions, and much harder for arcs abandoned recently from good ones. The result is a ban that self-tightens as the search improves.

---

## Algorithm Outline

```
current <- initial solution (minimum insertion, parallel)
best    <- current
p       <- {}                                  # arc -> label; absent means infinity

while non_improving < max_non_improving:
    iteration += 1

    if iteration mod p_iter == 0:
        p <- {}                                # p(i,j) <- infinity for every arc

    # cheapest move over relocate ∪ swap ∪ 2-opt whose every created arc (i,j)
    # satisfies  current.cost + delta  <  p(i,j)
    move <- best admissible move
    if move is None:
        stop

    label <- current.cost                      # read BEFORE applying the move
    for a in move.removed_arcs:
        p[a] <- label

    current <- move.apply(current)

    if current.cost < best.cost:
        best <- current
        non_improving <- 0
    else:
        non_improving += 1

return best
```

Two points of order matter:

- **The label is read before the move is applied.** It is the cost of the solution the arc was taken *out of*, not the cost of the solution that results.
- **A later deletion overwrites an earlier label.** If an arc is dropped at cost 1000, restored, then dropped again at cost 900, its promise is 900 — the most recent abandonment is the one the search must beat.

Admission is enforced **inside the operators** via the `arc_promises` parameter, so a blocked candidate is never evaluated and what comes back is the best *admissible* move, not a filtered afterthought:

```python
def _promise_blocks(promises, added, new_cost):
    for a in added:
        limit = promises.get(a)
        if limit is not None and new_cost >= limit:
            return True
    return False
```

The comparison is strict (`new_cost >= limit` blocks), so merely matching the recorded cost is not enough to keep the promise.

---

## Key Parameters

Hosted at the top of `promises_search.py`:

| Parameter | Default | Effect |
|---|---|---|
| `MAX_NON_IMPROVING_ITERATIONS` | 1000 | Stop after this many consecutive iterations that fail to improve the best solution. The main effort dial. |
| `P_ITER` | 100 | Re-initialise every promise to $\infty$ every this many iterations. |
| `TIME_LIMIT_SECONDS` | None | Optional wall-clock cap, checked at the end of each iteration. |

All three are overridable per call: `solve(model, method="promises", p_iter=50)`.

Roughly 3 arcs are labelled per iteration, so just before a wipe about $3 \times$ `p_iter` arcs carry promises. `p_iter = 1` wipes the memory every iteration and degenerates to unconstrained steepest descent, which stalls in the first local optimum (cost 870 on A-n32-k5, versus 784 at the default).

---

## Strengths and Limitations

**Strengths**

- **Consistently better than the classical tabu ban** on every instance tested, sometimes by a wide margin — it reaches the known optimum on A-n32-k5 where tabu stalls 8.4 % above it.
- **Self-calibrating restriction.** The bar rises automatically as the search finds better solutions, with no tuning of an aspiration rule.
- **Far less parameter-sensitive than tenure.** The `p_iter` response is close to monotone and flat above ~50, where tabu's tenure response is bumpy and non-monotone (868 → 850 → 796 → 834 → 834).
- **No separate aspiration parameter** to set or get wrong.
- **Deterministic and reproducible.** No seed, no run-to-run variance.

**Limitations**

- **Memory grows between wipes.** Nothing expires individually; the mapping only ever grows until `p_iter` clears it. Fine in practice (~3·`p_iter` entries) but it is a sawtooth, not a steady state.
- **Weaker diversification than a hard ban.** Because the rule admits any improving-enough move, the search is less forcefully pushed out of a basin. On the largest instance the margin over tabu narrows to 0.7 points, suggesting the advantage shrinks as the space grows.
- **Steepest descent is expensive.** Every iteration rescans all three neighbourhoods in full, $O(n^2)$ each. A granular or candidate-list neighbourhood would cut this sharply.
- **Short-term memory only.** No frequency-based diversification, no restarts from elite solutions, no path relinking.
- **Can stall.** If every candidate in every neighbourhood re-creates a promised arc without beating its label, the search stops early rather than forcing a wipe. Reaching the next scheduled reset would be a reasonable alternative.

---

## Usage

```python
from setup import read_vrp_file
from solver import solve

model = read_vrp_file("instances/A-n32-k5.vrp")

# Module defaults: 1000 non-improving iterations, promises wiped every 100
solution = solve(model, method="promises_search")

# Longer memory between wipes
solution = solve(model, method="promises", p_iter=200)

# Warm-start from a solution you already have
solution = solve(model, method="promises", initial_solution=existing)
```

Registered aliases in `solver.py`: `promises_search`, `promises`. It is menu option **7** in `main.py`. When no `initial_solution` is given it builds one with the parallel minimum-insertion heuristic.

> **Import note.** `metaheuristics/local search/` contains a space, so it is not a valid Python package name and `promises_search.py` cannot be imported normally. `solver.py` loads it by path with `importlib` via `_load_local_search()`, deferred to call time. Any other caller must do the same. Renaming the folder to `local_search` would remove the need.

---

## Benchmarking

All runs start from the parallel minimum-insertion solution and are deterministic, so each figure is a single run. Python 3.14, single core. Tabu figures are at its own default tenure of 100, for a like-for-like comparison of the two memory policies at the same setting.

### Promises vs. classical tabu (`max_non_improving=1000`, `p_iter` = tenure = 100)

| Instance | Customers | Start | **Promises** | Gap | Time | Tabu | Gap | Time | BKS |
|---|---|---|---|---|---|---|---|---|---|
| A-n32-k5 | 31 | 969 | **784** | **0.0 %** | 0.6 s | 850 | +8.4 % | 2.4 s | 784 |
| A-n54-k7 | 53 | 1561 | **1199** | **+2.7 %** | 2.2 s | 1250 | +7.1 % | 2.4 s | 1167 |
| P-n101-k4 | 100 | 734 | **690** | **+1.3 %** | 6.0 s | 701 | +2.9 % | 4.8 s | 681 |
| M-n200-k17 | 199 | 1724 | **1359** | **+6.6 %** | 19.9 s | 1368 | +7.3 % | 11.7 s | 1275 |

All solutions pass `solver.validate_solution`.

### `p_iter` sweep (gap to BKS)

| Instance | p=5 | p=10 | p=20 | p=50 | p=100 | p=200 |
|---|---|---|---|---|---|---|
| A-n32-k5 | 11.0 % | 11.0 % | 11.0 % | **0.0 %** | **0.0 %** | **0.0 %** |
| A-n54-k7 | 7.1 % | 7.1 % | 3.1 % | **2.6 %** | 2.7 % | 2.7 % |
| P-n101-k4 | 2.8 % | 2.6 % | 2.6 % | 1.2 % | 1.3 % | **0.0 %** |

**Observations**

- **The promise policy beats the flat ban on all four instances**, and reaches the **known optimum on A-n32-k5 (784)** where tabu at the same setting stalls at 850. On P-n101-k4 with `p_iter=200` it also reaches the optimum (681).
- **Why it wins:** the flat ban refuses a move on membership alone, so at tenure 100 a large fraction of a small instance's arc set is simply unavailable and the search is starved. The promise refuses only moves that fail to improve on a recorded cost, so improving moves are never blocked no matter how many arcs carry labels. It gets the diversification benefit of a long memory without the starvation.
- **`p_iter` is far better behaved than tenure.** The sweep is close to monotone and flat above 50, so the default of 100 is a safe choice on every instance tested — unlike tabu, where the best tenure was 20 on small instances and 100 on the large one, with a bumpy response in between.
- **Short memory is actively bad here.** At `p_iter` ≤ 20 the wipes come so often that almost nothing is ever constrained, and A-n32-k5 sits at 11.0 %. This is the opposite of the tabu tenure finding, and follows directly from the rule being conditional rather than absolute.
- **Cost:** on the small instances promises is also *faster* (0.6 s vs 2.4 s), because it finds improvements sooner and the non-improving counter trips later. On the 199-customer instance it is slower (19.9 s vs 11.7 s) for a 0.7-point gain, as more admissible moves means more iterations before stalling.

---

## References

1. **Glover, F.** (1986). *Future paths for integer programming and links to artificial intelligence.* Computers & Operations Research, 13(5), 533–549.
2. **Glover, F. & Laguna, M.** (1997). *Tabu Search.* Kluwer Academic Publishers. — Chapter 4 covers aspiration criteria and attribute-based memory in depth.
3. **Hertz, A. & de Werra, D.** (1991). *The tabu search metaheuristic: how we used it.* Annals of Mathematics and Artificial Intelligence, 1, 111–121.
4. **Taillard, É.** (1993). *Parallel iterative search methods for vehicle routing problems.* Networks, 23(8), 661–673.
5. **Gendreau, M., Hertz, A. & Laporte, G.** (1994). *A tabu search heuristic for the vehicle routing problem.* Management Science, 40(10), 1276–1290.
6. **Battiti, R. & Tecchiolli, G.** (1994). *The reactive tabu search.* ORSA Journal on Computing, 6(2), 126–140.
7. **Cordeau, J.-F. & Laporte, G.** (2005). *Tabu search heuristics for the vehicle routing problem.* In Rego & Alidaee (eds.), *Metaheuristic Optimization via Memory and Evolution*, Springer, 145–163.
