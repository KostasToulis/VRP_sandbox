# Genetic Algorithm for CVRP

## Conceptual Overview

A Genetic Algorithm (GA) is an **evolutionary metaheuristic** that draws inspiration from natural selection (Darwin, 1859) and was formalised as a computational method by **Holland (1975)**. A population of candidate solutions evolves over multiple generations through selection pressure, recombination, and random perturbation, gradually improving solution quality.

GAs are **population-based** — they maintain diversity by evolving many solutions in parallel, which helps escape local optima that trap single-solution methods (e.g., local search). For the CVRP, the **giant-tour representation** introduced by **Prins (2004)** is the standard encoding.

---

## Representation

### Genome — Giant Tour

A solution is encoded as a single **permutation of all $n$ customer nodes**, with the depot omitted:

```
Genome: [c3, c7, c1, c5, c2, c4, c6, ...]
```

This permutation implicitly defines vehicle routes once decoded.

### Decoding — Greedy Split

The split procedure converts a giant tour into feasible routes greedily:

```
1. Start a new empty route.
2. For each customer in the permutation:
      If adding the customer would exceed capacity C:
          Close current route, open a new one.
      Append customer to current route.
3. Close the last route.
```

This guarantees feasibility: every generated solution satisfies capacity constraints by construction.

```python
def _decode(self, node_sequence: List[Node]) -> Solution:
    routes, current_route, current_load = [], [], 0

    for customer in node_sequence:
        if current_load + customer.demand > self.model.capacity:
            routes.append(self._create_route(route_counter, current_route))
            current_route, current_load = [], 0
        current_route.append(customer)
        current_load += customer.demand

    if current_route:
        routes.append(self._create_route(route_counter, current_route))

    return Solution(cost=total_cost, routes=routes)
```

---

## Algorithm Outline

```
Initialise population with random permutations.
Evaluate fitness (decode → compute total route cost) for each individual.
best ← individual with lowest cost

FOR generation = 1 … max_generations:
    new_pop ← [best]                       # elitism
    WHILE |new_pop| < population_size:
        p1 ← tournament_select(population)
        p2 ← tournament_select(population)
        child_genome ← OX_crossover(p1, p2)
        swap_mutate(child_genome, prob=mutation_rate)
        child_solution ← decode(child_genome)
        new_pop.append(child_solution)
    population ← new_pop
    best ← update_best(population, best)

RETURN best
```

---

## Algorithm Components

### 1. Initialisation

Random permutations of the customer list form the initial population:

```python
def _initial_population(self) -> List[Solution]:
    pop = []
    for _ in range(self.population_size):
        perm = self.customers[:]   # copy customer list
        random.shuffle(perm)       # random permutation
        pop.append(self._decode(perm))
    return pop
```

### 2. Tournament Selection

A random subset of `tournament_size` individuals compete; the best (lowest cost) is selected as a parent. This introduces **selection pressure** without discarding all weaker individuals.

```python
def _tournament_selection(self, population: List[Solution]) -> Solution:
    competitors = random.sample(population, self.tournament_size)
    return min(competitors, key=lambda s: s.cost)
```

Higher `tournament_size` → stronger selection pressure → faster convergence but less diversity.

### 3. Order Crossover (OX1)

OX preserves the **relative order** of cities from one parent and fills gaps from the second. This is the canonical crossover for permutation encodings.

```
Parent 1: [3 4 | 8 2 7 | 1 6 5]
Parent 2: [4 2   5 1 6   7 8 3]

Step 1 — copy segment from P1 into child:
Child:    [_ _ | 8 2 7 | _ _ _]

Step 2 — fill remaining slots in order of P2 (skip already present):
P2 order (cyclic from cut-end): 4, 5, 1, 6, 7, 3  →  drop 2, 7, 8
Child:    [4 5 | 8 2 7 | 1 6 3]
```

```python
def _crossover(self, parent1: Solution, parent2: Solution) -> List[Node]:
    genome1 = [n for r in parent1.routes for n in r.sequence_of_nodes]
    genome2 = [n for r in parent2.routes for n in r.sequence_of_nodes]

    size = len(genome1)
    start, end = sorted(random.sample(range(size), 2))

    child = [None] * size
    child[start:end] = genome1[start:end]

    p2_idx = 0
    for i in range(size):
        if child[i] is None:
            while genome2[p2_idx] in child:
                p2_idx += 1
            child[i] = genome2[p2_idx]
    return child
```

### 4. Swap Mutation

Applied with probability `mutation_rate`, it swaps two randomly chosen positions in the genome, introducing small perturbations to maintain diversity.

```python
def _mutate(self, genome: List[Node]):
    if random.random() < self.mutation_rate:
        i, j = random.sample(range(len(genome)), 2)
        genome[i], genome[j] = genome[j], genome[i]
```

### 5. Elitism

The best individual from generation $t$ is always copied unchanged to generation $t+1$. This ensures the algorithm is **monotonically non-worsening** — the best-ever solution is never lost.

---

## Key Parameters

| Parameter | Default | Effect |
|---|---|---|
| `population_size` | 100 | Larger → more diversity, slower per generation |
| `generations` | 500 | More iterations → better convergence, longer runtime |
| `mutation_rate` | 0.1 | Higher → more exploration, less exploitation |
| `tournament_size` | 5 | Larger → stronger selection pressure |
| `seed` | None | Set for reproducibility |

**Typical tuning guidance**: Start with `population_size=100`, `generations=300–1000`, `mutation_rate=0.05–0.2`. For large instances increase population size before generations.

---

## Strengths and Limitations

**Strengths**
- **Simple and flexible**: easy to implement and adapt to new constraints.
- **Good diversity**: population-based search avoids premature convergence better than single-solution methods.
- **No gradient required**: works on any combinatorial domain.

**Limitations**
- **No optimality guarantee**: solution quality depends heavily on parameter tuning.
- **Greedy split is suboptimal**: the decoder does not optimise within-route ordering; adding a local search (2-opt, or-opt) post-decoding significantly improves results.
- **Scaling**: runtime grows with both `population_size × generations` and instance size.
- **Premature convergence**: populations can become homogeneous without sufficient mutation or diversity mechanisms.

---

## Benchmarking

| Instance | Customers | BKS | GA Cost | Gap | Runtime |
|---|---|---|---|---|---|
| A-n32-k5 | 31 | 784 | ~800–850 | ~2–8% | < 30 s |
| A-n45-k6 | 44 | 944 | ~980–1050 | ~4–11% | ~1 min |
| A-n54-k7 | 53 | 1167 | ~1220–1300 | ~5–12% | ~2 min |

Results with `population_size=100`, `generations=500`, `mutation_rate=0.1`.

---

## References

1. **Holland, J. H.** (1975). *Adaptation in Natural and Artificial Systems.* University of Michigan Press.
2. **Prins, C.** (2004). *A simple and effective evolutionary algorithm for the vehicle routing problem.* Computers & Operations Research, 31(12), 1985–2002.
3. **Larrañaga, P. et al.** (1999). *Genetic algorithms for the travelling salesman problem: A review of representations and operators.* Artificial Intelligence Review, 13(2), 129–170.
4. **Goldberg, D. E.** (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning.* Addison-Wesley.
