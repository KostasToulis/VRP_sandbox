import random
from typing import List, Tuple
from model import VRPModel, Route, Solution, Node

class GeneticAlgorithm:
    def __init__(self, model: VRPModel, population_size: int = 100, generations: int = 500, mutation_rate: float = 0.1, tournament_size: int = 5):
        self.model = model
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.customers = [n for n in model.nodes if n.id != model.depot_id]

    def solve(self) -> Solution:
        population = self._initial_population()
        best_solution = min(population, key=lambda s: s.cost)

        for _ in range(self.generations):
            new_population = []
            
            # Elitism: keep best
            new_population.append(best_solution)

            while len(new_population) < self.population_size:
                p1 = self._tournament_selection(population)
                p2 = self._tournament_selection(population)
                
                child_genome = self._crossover(p1, p2)
                self._mutate(child_genome)
                
                child_solution = self._decode(child_genome)
                new_population.append(child_solution)
            
            population = new_population
            current_best = min(population, key=lambda s: s.cost)
            if current_best.cost < best_solution.cost:
                best_solution = current_best

        return best_solution

    def _initial_population(self) -> List[Solution]:
        pop = []
        for _ in range(self.population_size):
            perm = self.customers[:]
            random.shuffle(perm)
            pop.append(self._decode(perm))
        return pop

    def _decode(self, node_sequence: List[Node]) -> Solution:
        routes = []
        current_route_nodes = []
        current_load = 0
        total_cost = 0
        route_counter = 1

        for customer in node_sequence:
            if current_load + customer.demand > self.model.capacity:
                # Close current route
                routes.append(self._create_route(route_counter, current_route_nodes))
                total_cost += routes[-1].cost
                route_counter += 1
                current_route_nodes = []
                current_load = 0
            
            current_route_nodes.append(customer)
            current_load += customer.demand

        if current_route_nodes:
            routes.append(self._create_route(route_counter, current_route_nodes))
            total_cost += routes[-1].cost

        return Solution(cost=total_cost, routes=routes)

    def _create_route(self, rid: int, nodes: List[Node]) -> Route:
        # Calculate cost: Depot -> Node1 -> ... -> NodeN -> Depot
        cost = 0.0
        load = sum(n.demand for n in nodes)
        
        # From depot to first
        if nodes:
            cost += self.model.get_distance(self.model.depot_id, nodes[0].id)
            for i in range(len(nodes) - 1):
                cost += self.model.get_distance(nodes[i].id, nodes[i+1].id)
            # From last to depot
            cost += self.model.get_distance(nodes[-1].id, self.model.depot_id)
            
        return Route(id=rid, load=load, cost=cost, sequence_of_nodes=nodes)

    def _tournament_selection(self, population: List[Solution]) -> Solution:
        competitors = random.sample(population, self.tournament_size)
        return min(competitors, key=lambda s: s.cost)

    def _crossover(self, parent1: Solution, parent2: Solution) -> List[Node]:
        # Extract genomes (flat list of customers excluding depot)
        # Note: Solution stores routes. We need to reconstruct the permutation.
        genome1 = [n for r in parent1.routes for n in r.sequence_of_nodes]
        genome2 = [n for r in parent2.routes for n in r.sequence_of_nodes]
        
        # Order Crossover (OX)
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

    def _mutate(self, genome: List[Node]):
        if random.random() < self.mutation_rate:
            # Swap two random nodes
            i, j = random.sample(range(len(genome)), 2)
            genome[i], genome[j] = genome[j], genome[i]
