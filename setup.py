import os
from typing import List, Tuple, Dict
from model import Node, VRPModel

def read_vrp_file(filepath: str) -> VRPModel:
    """
    Reads a .vrp file and returns a VRPModel instance.
    Assumes EUC_2D edge weight type and CVRP format.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    name = ""
    capacity = 0
    dimension = 0
    nodes: List[Node] = []
    depot_id = 1
    
    section = None
    node_coords: Dict[int, Tuple[float, float]] = {}
    demands: Dict[int, int] = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Parse Header Information
        if line.startswith("NAME"):
            name = line.split(":")[-1].strip()
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(":")[-1].strip())
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[-1].strip())
        
        # Identify Sections
        elif line.startswith("NODE_COORD_SECTION"):
            section = "NODE_COORD"
        elif line.startswith("DEMAND_SECTION"):
            section = "DEMAND"
        elif line.startswith("DEPOT_SECTION"):
            section = "DEPOT"
        elif line.startswith("EOF"):
            break
        
        # Parse Section Data
        else:
            parts = line.split()
            if section == "NODE_COORD":
                # Format: Id X Y
                if len(parts) >= 3:
                    nid = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    node_coords[nid] = (x, y)
            elif section == "DEMAND":
                # Format: Id Demand
                if len(parts) >= 2:
                    nid = int(parts[0])
                    dem = int(parts[1])
                    demands[nid] = dem
            elif section == "DEPOT":
                # Format: Id (ends with -1)
                val = int(parts[0])
                if val != -1:
                    depot_id = val

    # Construct Node objects
    # Note: Assumes that node IDs in COORDINATES and DEMANDS match
    for nid, (x, y) in node_coords.items():
        demand = demands.get(nid, 0)
        nodes.append(Node(nid, x, y, demand))

    # Sort nodes by ID just to be safe/consistent
    nodes.sort(key=lambda n: n.id)

    return VRPModel(name, capacity, nodes, depot_id)

if __name__ == "__main__":
    # Example usage for testing
    import sys
    if len(sys.argv) > 1:
        fp = sys.argv[1]
        try:
            model = read_vrp_file(fp)
            print(f"Successfully loaded: {model}")
            print(f"Nodes: {len(model.nodes)}")
            print(f"Depot: {model.depot}")
        except Exception as e:
            print(f"Error loading file: {e}")
    else:
        print("Usage: python setup.py <path_to_vrp_file>")
