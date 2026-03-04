import pulp
from typing import List, Dict, Tuple
from model import VRPModel, Solution, Route, Node
import math

class ColumnGeneration:
    def __init__(self, model: VRPModel):
        self.model = model
        self.nodes = [n for n in model.nodes if n.id != model.depot_id]
        self.depot = model.depot
        
        # Generated columns (routes)
        self.columns: List[Route] = []

    def solve(self) -> Solution:
        # 1. Initialization: Create dummy routes (one per customer) to ensure feasibility
        self._initialize_columns()
        
        iteration = 0
        while True:
            iteration += 1
            # 2. Solve Master Problem (LP Relaxation)
            duals, lp_obj = self._solve_master_problem(relax=True)
            
            # 3. Solve Pricing Problem
            # Find a route with negative reduced cost
            new_route = self._solve_pricing_problem(duals)
            
            if new_route and new_route.cost < -1e-5: # Significant negative reduced cost
                # Add real cost route to columns (re-evaluate cost without duals)
                real_cost, load = self._calculate_route_metrics(new_route.sequence_of_nodes)
                final_route = Route(id=len(self.columns)+1, load=load, cost=real_cost, sequence_of_nodes=new_route.sequence_of_nodes)
                
                self.columns.append(final_route)
                # print(f"Iteration {iteration}: Added col with red. cost {new_route.cost:.2f}")
            else:
                # No more columns with negative reduced cost -> LP Optimal found
                break
        
        # 4. Final Step: Solve Master Problem as Integer (MIP) restricted to generated columns
        # This is a heuristic (Restricted Master Heuristic), as we don't branch-and-price.
        solution = self._solve_master_problem(relax=False)
        return solution

    def _initialize_columns(self):
        """Create single-customer routes: Depot -> Customer -> Depot"""
        for node in self.nodes:
            seq = [node]
            cost, load = self._calculate_route_metrics(seq)
            r = Route(id=len(self.columns)+1, load=load, cost=cost, sequence_of_nodes=seq)
            self.columns.append(r)

    def _calculate_route_metrics(self, nodes: List[Node]) -> Tuple[float, int]:
        cost = 0.0
        load = sum(n.demand for n in nodes)
        
        if nodes:
            cost += self.model.get_distance(self.depot.id, nodes[0].id)
            for i in range(len(nodes) - 1):
                cost += self.model.get_distance(nodes[i].id, nodes[i+1].id)
            cost += self.model.get_distance(nodes[-1].id, self.depot.id)
            
        return cost, load

    def _solve_master_problem(self, relax: bool = True):
        """
        Solves the Set Partitioning Problem.
        If relax=True, returns (duals, obj_value).
        If relax=False, returns Solution object.
        """
        prob = pulp.LpProblem("MasterProblem", pulp.LpMinimize)
        
        # Variables: x_r for each route r
        # If relaxed, 0 <= x_r. If integer, x_r in {0, 1}
        cat = pulp.LpContinuous if relax else pulp.LpBinary
        x = [pulp.LpVariable(f"x_{r.id}", lowBound=0, cat=cat) for r in self.columns]
        
        # Objective: minimize sum(c_r * x_r)
        prob += pulp.lpSum(self.columns[i].cost * x[i] for i in range(len(self.columns)))
        
        # Constraints: Each customer covered exactly once (or >= 1 for feasibility easier)
        # sum(a_ir * x_r) >= 1  forall i
        constraints = {}
        for node in self.nodes:
            # Finding which columns visit this node
            # Ideally cache this mapping for performance
            prob += pulp.lpSum(x[i] for i, col in enumerate(self.columns) if node in col.sequence_of_nodes) >= 1, f"cov_{node.id}"
            if relax:
                constraints[node.id] = prob.constraints[f"cov_{node.id}"]

        # Solve
        # Suppress output
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if relax:
            # Extract duals
            # For >= constraints, duals are non-negative in minimization? 
            # Pulp duals convention: usually positive for >= constraints in minimization.
            duals = {nid: constraints[nid].pi for nid in constraints}
            return duals, pulp.value(prob.objective)
        else:
            # Reconstruct Solution
            selected_routes = []
            for i, var in enumerate(x):
                if pulp.value(var) and pulp.value(var) > 0.5:
                    selected_routes.append(self.columns[i])
            
            return Solution(cost=pulp.value(prob.objective), routes=selected_routes)

    def _solve_pricing_problem(self, duals: Dict[int, float]) -> Route:
        """
        Solves the ESPPRC (Elementary Shortest Path with Resource Constraints)
        using a Label Setting / DFS approach.
        Reduce cost of edge (i,j): d_ij - dual_j/2 - dual_i/2 ? 
        Standard: reduced_cost_route = cost_route - sum(dual_i).
        Edge cost scheme: moving to node j incurs cost = dist(i,j) - dual[j].
        """
        
        # Simple depth-limited DFS or heuristics for sandbox performance
        # Exact Label Setting is PSPACE/NP-hard. We use a heuristic search here for demonstration.
        
        best_route = None
        best_red_cost = -1e-9
        
        # Stack: (current_node, current_load, current_red_cost, visited_ids_set, path_nodes)
        # Note: visited is a set for O(1) lookup
        stack = [(self.depot, 0, 0.0, set(), [])]
        
        # Limit exploration for sandbox responsiveness
        max_steps = 5000 
        steps = 0
        
        # Sort neighbors by proximity to guide search (heuristic)
        neighbors = {
            n.id: sorted(self.nodes, key=lambda x: self.model.get_distance(n.id, x.id)) 
            for n in self.model.nodes
        }
        
        while stack:
            steps += 1
            # Heuristic pruning
            if steps > max_steps:
               break

            curr, load, red_cost, visited, path = stack.pop()
            
            # Try closing route to depot
            # Edge cost to depot: dist(curr, depot) - 0
            dist_to_depot = self.model.get_distance(curr.id, self.depot.id)
            total_red_cost = red_cost + dist_to_depot
            
            if len(path) > 0 and total_red_cost < best_red_cost:
                best_red_cost = total_red_cost
                best_route = Route(id=-1, load=load, cost=total_red_cost, sequence_of_nodes=list(path))
            
            # Extend to neighbors
            # For simple heuristic: pick top 5 closest unvisited neighbors
            candidates = [n for n in neighbors[curr.id] if n.id not in visited]
            
            for next_node in candidates[:5]: # Beam width 5
                if load + next_node.demand <= self.model.capacity:
                    # Cost to move to next_node: dist(curr, next) - dual[next]
                    d = self.model.get_distance(curr.id, next_node.id)
                    dual_val = duals.get(next_node.id, 0.0)
                    new_red_cost = red_cost + d - dual_val
                    
                    # Pruning: if current reduced cost is already very high, maybe skip?
                    # But reduced cost can decrease (due to positive duals).
                    # Simple bound check if we had minimum possible dist and max duals implies no hope? (Skip for simplicity)
                    
                    new_visited = visited.copy()
                    new_visited.add(next_node.id)
                    new_path = path + [next_node]
                    
                    stack.append((next_node, load + next_node.demand, new_red_cost, new_visited, new_path))
                    
        return best_route
