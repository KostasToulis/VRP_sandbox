import os
import setup
from solver import run

if __name__ == "__main__":
    instance_path = os.path.join("instances", "P-n16-k8.vrp")
    
    # Ensure folder structure existence for the user if they don't have it locally yet
    # (In a real scenario, we assume the file exists as per instructions)
    if not os.path.exists(instance_path):
        print(f"Instance file not found at {instance_path}")
        

    print(f"Loading instance: {instance_path}")
    model = setup.read_vrp_file(instance_path)

    print("\nSelect the methodology to solve the VRP:")
    print("1. Branch and Cut")
    print("2. Column Generation")
    print("3. Genetic Algorithm")
    print("4. Random Key Optimizer (BRKGA)")
    print("5. Exit")

    try:
        choice = input("Enter your choice (1-5): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nProgram terminated.")
        raise SystemExit(0)

    run(choice, model)