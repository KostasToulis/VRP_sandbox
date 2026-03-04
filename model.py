from dataclasses import dataclass
from typing import List, Dict
import math

@dataclass
class Node:
    id: int
    x: float
    y: float
    demand: int

@dataclass
class Route:
    id: int
    load: int
    cost: float
    sequence_of_nodes: List[Node]

@dataclass
class Solution:
    cost: float
    routes: List[Route]

class VRPModel:
    def __init__(self, name: str, capacity: int, nodes: List[Node], depot_id: int):
        self.name = name
        self.capacity = capacity
        self.nodes = nodes
        self.depot_id = depot_id
        self.dimension = len(nodes)
        
        # Helper to get node by ID for quick access
        self.node_map: Dict[int, Node] = {node.id: node for node in nodes}

    @property
    def depot(self) -> Node:
        return self.node_map[self.depot_id]
        
    def get_distance(self, node1_id: int, node2_id: int) -> float:
        """Calculates Euclidean distance (EUC_2D) between two nodes."""
        n1 = self.node_map[node1_id]
        n2 = self.node_map[node2_id]
        return math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)

    def __repr__(self):
        return f"VRPModel(name={self.name}, dimension={self.dimension}, capacity={self.capacity}, depot={self.depot_id})"
