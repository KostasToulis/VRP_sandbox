# Random Key Optimizer (BRKGA) for CVRP

## Overview
The Random Key Optimizer is an implementation of a **Biased Random-Key Genetic Algorithm (BRKGA)**. This metaheuristic is powerful because it separates the search space (random keys) from the solution space (permutations/routes). All genetic operators (crossover) are performed on the random keys, ensuring feasibility is always maintained during the decoding process.

## Representation
**Random Keys**: A solution is represented by a vector of $N$ real numbers (keys) in the interval $[0, 1]$, where $N$ is the number of customer nodes.

**Decoding**:
1.  **Sorting**: The keys are sorted in ascending order.
2.  **Permutation**: The original indices of the sorted keys form a permutation of the customers.
3.  **Split**: The permutation is greedily split into feasible vehicle routes based on capacity (similar to the standard Genetic Algorithm decoding).

Example:
- Keys: `[0.2, 0.9, 0.4]` (for customers A, B, C)
- Sorted: `0.2 (A)`, `0.4 (C)`, `0.9 (B)`
- Permutation: `A -> C -> B`
- Routes (Cap=10, Demands=4): `[A, C] (Load 8)`, `[B] (Load 4)`

## Algorithm Components

### 1. Evolution Strategy (BRKGA)
Instead of standard selection and mutation, BRKGA divides the population into:
- **Elite**: The top % of best solutions (e.g., 20%). These are copied to the next generation unchanged.
- **Mutants**: Totally new random individuals (e.g., 15%). These introduce diversity similar to high-rate mutation.
- **Crossover Offspring**: The rest of the population is filled by mating Elite parents with Non-Elite parents.

### 2. Parameterized Uniform Crossover
When mating an Elite parent with a Non-Elite parent, the child inherits the "key" from the Elite parent with a high probability (e.g., $\rho_e = 0.7$). This preserves good gene blocks while allowing mixing.

```python
def _crossover(self, keys_elite, keys_other):
    prob_elite = 0.7
    child_keys = []
    for ke, ko in zip(keys_elite, keys_other):
        # Biased coin toss favoring elite gene
        val = ke if random.random() < prob_elite else ko
        child_keys.append(val)
    return child_keys
```

### 3. Benefits
- **Feasibility**: Any vector of random numbers corresponds to a valid permutation, so crossover never produces invalid chromosomes.
- **Simplicity**: No need for complex permutation crossovers (like OX, PMX) or specific mutation operators.
- **Convergence**: Elitism combined with the bias towards elite genes drives convergence, while mutants prevent stagnation.

## Parameters
- **`population_size`**: 100
- **`elite_size`**: 20 (Top 20% survive)
- **`mutant_size`**: 15 (15% refreshed)
- **`generations`**: 500
