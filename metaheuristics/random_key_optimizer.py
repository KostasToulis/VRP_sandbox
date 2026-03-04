import random
from typing import List, Tuple
from model import VRPModel, Solution, Route, Node

class RandomKeyOptimizer:
    def __init__(self, model: VRPModel, population_size: int = 100, generations: int = 500, elite_size: int = 20, mutant_size: int = 15):
        self.model = model
        self.population_size = population_size
        self.generations = generations
        # Parameters for Biased Random-Key Genetic Algorithm (BRKGA) style
        self.elite_size = elite_size
        self.mutant_size = mutant_size
        self.nodes = [n for n in model.nodes if n.id != model.depot_id]
        self.num_nodes = len(self.nodes)

    def solve(self) -> Solution:
        # A solution is represented by a vector of random keys (floats in [0, 1])
        # Length = number of customers.
        
        # 1. Initialize Population
        # Each individual is: (keys: List[float], solution: Solution)
        population = [self._generate_individual() for _ in range(self.population_size)]
        
        # Evaluate initial population
        population.sort(key=lambda x: x[1].cost)
        best_solution = population[0][1]

        for gen in range(self.generations):
            # 2. Classify Population
            elite = population[:self.elite_size]
            non_elite = population[self.elite_size:]
            
            next_population = []
            
            # 3. Elitism: Cope elite individuals directly
            next_population.extend(elite)
            
            # 4. Mutants: Introduce fresh random individuals
            for _ in range(self.mutant_size):
                next_population.append(self._generate_individual())
                
            # 5. Crossover
            # Combine Elite parent with Non-Elite parent to fill the rest
            slots_remaining = self.population_size - len(next_population)
            for _ in range(slots_remaining):
                parent_elite = random.choice(elite)
                parent_non_elite = random.choice(non_elite) # or random from whole pop? BRKGA usually uses non-elite.
                
                child_keys = self._crossover(parent_elite[0], parent_non_elite[0])
                child_sol = self._decode(child_keys)
                next_population.append((child_keys, child_sol))
            
            population = next_population
            population.sort(key=lambda x: x[1].cost)
            
            if population[0][1].cost < best_solution.cost:
                best_solution = population[0][1]
                # print(f"Generation {gen}: New Best Cost {best_solution.cost:.2f}")

        return best_solution

    def _generate_individual(self) -> Tuple[List[float], Solution]:
        keys = [random.random() for _ in range(self.num_nodes)]
        sol = self._decode(keys)
        return (keys, sol)

    def _decode(self, keys: List[float]) -> Solution:
        # Random Keys to Permutation:
        # Pair (index, key) and sort by key. The indices form the permutation.
        indexed_keys = []
        for i, key in enumerate(keys):
            indexed_keys.append((i, key))
        
        # Sort by key value (ascending)
        indexed_keys.sort(key=lambda x: x[1])
        
        # Extract permutation of nodes
        # nodes[0] corresponds to index 0, etc.
        permutation = [self.nodes[idx] for idx, _ in indexed_keys]
        
        # Standard Split algorithm to evaluate
        return self._split(permutation)

    def _split(self, sequence: List[Node]) -> Solution:
        routes = []
        current_route_nodes = []
        current_load = 0
        total_cost = 0.0
        route_id = 1
        
        for node in sequence:
            if current_load + node.demand > self.model.capacity:
                # Close route
                r_cost, r_load = self._calculate_route_metrics(current_route_nodes)
                routes.append(Route(id=route_id, load=r_load, cost=r_cost, sequence_of_nodes=current_route_nodes))
                total_cost += r_cost
                route_id += 1
                
                current_route_nodes = [node]
                current_load = node.demand
            else:
                current_route_nodes.append(node)
                current_load += node.demand
                
        if current_route_nodes:
            r_cost, r_load = self._calculate_route_metrics(current_route_nodes)
            routes.append(Route(id=route_id, load=r_load, cost=r_cost, sequence_of_nodes=current_route_nodes))
            total_cost += r_cost
            
        return Solution(cost=total_cost, routes=routes)

    def _calculate_route_metrics(self, nodes: List[Node]) -> Tuple[float, int]:
        cost = 0.0
        load = sum(n.demand for n in nodes)
        
        if nodes:
            depot = self.model.depot
            cost += self.model.get_distance(depot.id, nodes[0].id)
            for i in range(len(nodes) - 1):
                cost += self.model.get_distance(nodes[i].id, nodes[i+1].id)
            cost += self.model.get_distance(nodes[-1].id, depot.id)
            
        return cost, load

    def _crossover(self, keys_elite: List[float], keys_other: List[float]) -> List[float]:
        # Parameterized Uniform Crossover
        # Probability for elite allele usually > 0.5 (e.g. 0.7)
        prob_elite = 0.7
        child_keys = []
        for ke, ko in zip(keys_elite, keys_other):
            if random.random() < prob_elite:
                child_keys.append(ke)
            else:
                child_keys.append(ko)
        return child_keys
