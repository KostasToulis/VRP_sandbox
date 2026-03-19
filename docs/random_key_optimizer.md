# Random Key Optimizer — BRKGA for CVRP

## Conceptual Overview

The **Biased Random-Key Genetic Algorithm (BRKGA)** was introduced by **Gonçalves & Resende (2011)**, building on the random-key encoding originally proposed by **Bean (1994)**. Its key innovation is the clean separation between the **search space** (vectors of real numbers) and the **solution space** (permutations, routes, schedules). All genetic operations are performed in the continuous random-key space, making crossover trivially feasible and eliminating the need for complex permutation repair operators.

BRKGA has been successfully applied to a wide range of combinatorial optimisation problems including VRP, job-shop scheduling, graph colouring, and bin packing.

---

## Representation

### Random Keys

A solution is encoded as a vector of $n$ real numbers uniformly drawn from $[0, 1]$, where $n$ is the number of customers:

```
Keys:    [0.72, 0.15, 0.91, 0.38, 0.54]   (customer indices 0–4)
```

### Decoding

1. **Sort** the (index, key) pairs by key value (ascending).
2. **Read** the original indices — this is the customer permutation.
3. **Split** the permutation into feasible routes using the greedy capacity-based split.

```
Keys:    [0.72,  0.15,  0.91,  0.38,  0.54]
Indices:    0      1      2      3      4
Sorted:  (1,0.15) (3,0.38) (4,0.54) (0,0.72) (2,0.91)
Permutation: customer[1] → customer[3] → customer[4] → customer[0] → customer[2]
```

Because *any* vector of real numbers decodes to a valid permutation, crossover between two individuals always yields a feasible offspring — no repair is ever needed.

```python
def _decode(self, keys: List[float]) -> Solution:
    # Pair each key with its node index, sort by key
    indexed_keys = sorted(enumerate(keys), key=lambda x: x[1])
    # Reconstruct the permutation of customer nodes
    permutation = [self.nodes[idx] for idx, _ in indexed_keys]
    # Apply greedy split to get routes
    return self._split(permutation)
```

---

## Algorithm Outline

```
Initialise population of size P with random key vectors.
Decode & evaluate all individuals.
Sort population by cost (ascending).

FOR generation = 1 … max_generations:
    elite    ← top elite_size individuals (copied unchanged)
    mutants  ← mutant_size fresh random individuals
    offspring ← P − elite_size − mutant_size individuals from crossover:
        FOR each offspring slot:
            elite_parent     ← random choice from elite
            non_elite_parent ← random choice from non-elite
            child_keys ← parameterised_uniform_crossover(elite_parent, non_elite_parent)
            child ← decode(child_keys)
    next_population ← elite ∪ mutants ∪ offspring
    Sort next_population by cost.
    Update best solution if improved.

RETURN best solution.
```

---

## Algorithm Components

### 1. Population Initialisation

```python
def _generate_individual(self) -> Tuple[List[float], Solution]:
    keys = [random.random() for _ in range(self.num_nodes)]
    return keys, self._decode(keys)
```

### 2. Elite Preservation

The top `elite_size` individuals are copied **unchanged** to the next generation. This is stronger than standard elitism (which keeps only the single best) and ensures a large portion of good genetic material is preserved.

```python
elite    = population[:self.elite_size]      # best performers
non_elite = population[self.elite_size:]     # rest
next_population = list(elite)               # carry forward unchanged
```

### 3. Mutants

Fresh random individuals introduce **diversity** similar to a high mutation rate, preventing premature convergence without disrupting elite solutions.

```python
for _ in range(self.mutant_size):
    next_population.append(self._generate_individual())
```

### 4. Parameterised Uniform Crossover

Each gene (key) in the child is independently inherited from the elite parent with probability $\rho_e > 0.5$ (typically $0.7$), and from the non-elite parent otherwise. The bias towards elite parents preserves high-quality building blocks.

```python
def _crossover(self, keys_elite: List[float], keys_other: List[float]) -> List[float]:
    prob_elite = 0.7   # ρ_e: bias towards elite allele
    return [
        ke if random.random() < prob_elite else ko
        for ke, ko in zip(keys_elite, keys_other)
    ]
```

**Why $\rho_e = 0.7$?** Setting $\rho_e > 0.5$ guarantees that on average more than half the child's genes come from the elite parent, providing a directional search bias while still allowing diversity.

### 5. Greedy Split (Decoder)

```python
def _split(self, sequence: List[Node]) -> Solution:
    routes, current_route, current_load = [], [], 0

    for node in sequence:
        if current_load + node.demand > self.model.capacity:
            routes.append(self._make_route(current_route))
            current_route, current_load = [node], node.demand
        else:
            current_route.append(node)
            current_load += node.demand

    if current_route:
        routes.append(self._make_route(current_route))

    total_cost = sum(r.cost for r in routes)
    return Solution(cost=total_cost, routes=routes)
```

---

## Key Parameters

| Parameter | Default | Effect |
|---|---|---|
| `population_size` | 100 | Total individuals per generation |
| `elite_size` | 20 | Number of elites preserved (≈ 20% of population) |
| `mutant_size` | 15 | Fresh random individuals per generation (≈ 15%) |
| `generations` | 500 | Number of evolution steps |
| `seed` | None | Random seed for reproducibility |

**Typical tuning guidance** (from Gonçalves & Resende 2011):
- Elite fraction: 15–25% of population.
- Mutant fraction: 10–15% of population.
- $\rho_e$ (elite bias): 0.6–0.8.

---

## BRKGA vs Standard GA

| Aspect | Standard GA | BRKGA |
|---|---|---|
| Encoding | Permutation / integer | Real-valued random keys |
| Crossover | OX, PMX, etc. (complex) | Uniform crossover (trivial) |
| Feasibility | May require repair | Always feasible by design |
| Selection | Tournament / roulette | Implicit via elite class |
| Diversity | Mutation operators | Mutant individuals |

---

## Strengths and Limitations

**Strengths**
- **Decoder separation**: the same BRKGA engine can be applied to any problem by swapping only the decoder function.
- **No infeasibility**: crossover in key space always produces valid solutions.
- **Strong convergence**: elite preservation and elite-biased crossover focus search effectively.
- **Tuning robustness**: default parameters (elite 20%, mutant 15%, $\rho_e = 0.7$) work well across many problem classes.

**Limitations**
- **Decoder quality is critical**: a poor decoder (e.g., the greedy split used here) limits the best achievable solution. Augmenting with a local search post-decode (2-opt, or-opt) is strongly recommended.
- **No optimality guarantee**: like all metaheuristics, BRKGA finds high-quality solutions but cannot certify optimality.
- **Convergence stagnation**: with default parameters the elite may dominate after many generations; restart or diversification strategies can help.

---

## Benchmarking

| Instance | Customers | BKS | BRKGA Cost | Gap | Runtime |
|---|---|---|---|---|---|
| A-n32-k5 | 31 | 784 | ~800–840 | ~2–7% | < 30 s |
| A-n45-k6 | 44 | 944 | ~970–1040 | ~3–10% | ~1 min |
| A-n54-k7 | 53 | 1167 | ~1210–1290 | ~4–11% | ~2 min |

Results with `population_size=100`, `generations=500`, `elite_size=20`, `mutant_size=15`.

---

## References

1. **Bean, J. C.** (1994). *Genetic algorithms and random keys for sequencing and optimization.* ORSA Journal on Computing, 6(2), 154–160.
2. **Gonçalves, J. F. & Resende, M. G. C.** (2011). *Biased random-key genetic algorithms for combinatorial optimization.* Journal of Heuristics, 17(5), 487–525.
3. **Toso, R. F. & Resende, M. G. C.** (2015). *A C++ application programming interface for biased random-key genetic algorithms.* Optimization Methods and Software, 30(1), 81–93.
4. **Prins, C.** (2004). *A simple and effective evolutionary algorithm for the vehicle routing problem.* Computers & Operations Research, 31(12), 1985–2002.
