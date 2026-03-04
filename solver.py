import os
import setup
from metaheuristics.genetic_algorithm import GeneticAlgorithm
from exact.branch_and_cut import BranchAndCut
from exact.column_generation import ColumnGeneration
from metaheuristics.random_key_optimizer import RandomKeyOptimizer

def run():
    instance_path = os.path.join("instances", "A-n32-k5.vrp")
    
    # Ensure folder structure existence for the user if they don't have it locally yet
    # (In a real scenario, we assume the file exists as per instructions)
    if not os.path.exists(instance_path):
        print(f"Instance file not found at {instance_path}")
        return

    print(f"Loading instance: {instance_path}")
    model = setup.read_vrp_file(instance_path)
    
    # METHODOLOGY SELECTION
    # Uncomment the methodology you want to run
    
    # 1. Genetic Algorithm
    # print("Solving with Genetic Algorithm...")
    # ga = GeneticAlgorithm(model, population_size=100, generations=200)
    # solution = ga.solve()
    
    # 2. Branch and Cut
    # print("Solving with Branch and Cut...")
    # bc = BranchAndCut(model, time_limit=300) 
    # solution = bc.solve()

    # 3. Column Generation
    # print("Solving with Column Generation...")
    # cg = ColumnGeneration(model)
    # solution = cg.solve()

    # 4. Random Key Optimizer (BRKGA)
    print("Solving with Random Key Optimizer (BRKGA)...")
    rk_opt = RandomKeyOptimizer(model, population_size=100, generations=300)
    solution = rk_opt.solve()
    
    print("\nBest Solution Found:")
    print(f"Total Cost: {solution.cost:.2f}")
    print(f"Number of Routes: {len(solution.routes)}")
    for r in solution.routes:
        route_str = " -> ".join([str(n.id) for n in r.sequence_of_nodes])
        print(f"  Route {r.id} (Load: {r.load}): {model.depot.id} -> {route_str} -> {model.depot.id}")

if __name__ == "__main__":
    run()
