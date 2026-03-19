import pulp
from typing import List, Tuple
from model import VRPModel, Solution, Route

class BranchAndCut:
    def __init__(self, model: VRPModel, time_limit: int = 600):
        self.model = model
        self.time_limit = time_limit

    def solve(self) -> Solution:
        node_ids = [n.id for n in self.model.nodes]
        customer_ids = [n.id for n in self.model.nodes if n.id != self.model.depot_id]
        depot_id = self.model.depot_id
        
        # Create problem
        prob = pulp.LpProblem("CVRP_Branch_and_Cut", pulp.LpMinimize)

        # Variables: x[i][j] = 1 if edge (i, j) is used
        # We use a directed graph formulation
        x = {}
        for i in node_ids:
            for j in node_ids:
                if i != j:
                    x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)

        # Objective: minimise total travel distance.
        # Must be set as a single lpSum — using `prob += term` inside a loop
        # would replace (not accumulate) the objective on every iteration in PuLP.
        prob += pulp.lpSum(
            self.model.get_distance(i, j) * x[i, j]
            for i in node_ids for j in node_ids if i != j
        )

        # Constraints
        # 1. Degree constraints: Each customer has exactly one outgoing and one incoming edge
        for i in customer_ids:
            prob += pulp.lpSum(x[i, j] for j in node_ids if i != j) == 1
            prob += pulp.lpSum(x[j, i] for j in node_ids if i != j) == 1

        # 2. Depot flow balance: Outgoing = Incoming = K (number of vehicles)
        # Note: In pure VRP, K might be fixed or variable. Here we let it be free but bounded by N
        # Generally, sum(x[depot, j]) = sum(x[j, depot])
        prob += pulp.lpSum(x[depot_id, j] for j in customer_ids) == pulp.lpSum(x[j, depot_id] for j in customer_ids)
        
        # We solve iteratively, adding subtour elimination constraints
        while True:
            # Solve with current constraints
            status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=self.time_limit))
            
            if status != pulp.LpStatusOptimal:
                print("Optimal solution not found or time limit reached.")
                break

            # Reconstruct routes from edges
            edges = []
            for i in node_ids:
                for j in node_ids:
                    if i != j and pulp.value(x[i, j]) > 0.9:
                        edges.append((i, j))

            subtours = self._find_subtours(edges)

            # Check for generic subtours (not connected to depot) or capacity violations
            cuts_added = 0

            for subtour in subtours:
                is_depot_in_subtour = depot_id in subtour

                # Case 1: Subtour does not contain depot -> Valid SEC required
                if not is_depot_in_subtour:
                    # Cut: sum(x_ij for i,j in S) <= |S| - 1
                    prob += pulp.lpSum(x[i, j] for i in subtour for j in subtour if i != j) <= len(subtour) - 1
                    cuts_added += 1
                
                # Case 2: Subtour contains depot (looks like a valid route) but exceeds capacity
                elif is_depot_in_subtour:
                    # This "subtour" is actually a set of routes connected to depot.
                    # We need to check individual connected components attached to depot? 
                    # Actually, _find_subtours returns connected components. 
                    # If depot is in component, it might be multiple routes merged or just one valid route.
                    # Simple capacity cut: Required vehicles k >= ceil(Demand / Capacity)
                    # r(S) >= Demand / Capacity
                    # Cut: sum(x_ij for i in S, j not in S) >= 2 * k(S) (Standard Generalized Subtour Elimination)
                    # For directed: sum(x_ij for i in S, j not in S) >= k(S)
                    
                    # However, strictly for the CVRP formulation loops:
                    # If we have a single component with depot, we must check if it's decomposable or if it violates capacity simply as a set.
                    # A better way in simple B&C is to traverse the edges starting from depot to find individual routes.
                    pass

            # If we found disjoint subtours not containing depot, we handled them above.
            # Now we must specifically check Capacity constraints for cycles connected to depot.
            # Let's clean up routes parsing.
            routes_data, is_valid_capacity = self._extract_routes_and_check_capacity(edges, depot_id)
            
            if cuts_added == 0 and is_valid_capacity:
                # OPTIMAL FOUND
                return self._construct_solution(routes_data)
            
            if cuts_added == 0 and not is_valid_capacity:
                # Add capacity cuts for violating routes
                for route_nodes in routes_data:
                    load = sum(self.model.node_map[nid].demand for nid in route_nodes)
                    if load > self.model.capacity:
                        # Add Capacity Cut for this set S
                        # Cut: sum(x_ij for i,j in S) <= |S| - r(S)
                        # Minimal number of vehicles needed
                        import math
                        min_vehicles = math.ceil(load / self.model.capacity)
                        prob += pulp.lpSum(x[i, j] for i in route_nodes for j in route_nodes if i != j) <= len(route_nodes) - min_vehicles
                        cuts_added += 1
            
            if cuts_added == 0:
                break

        return Solution(cost=float('inf'), routes=[])

    def _find_subtours(self, edges: List[Tuple[int, int]]) -> List[List[int]]:
        """Detect cycles by following unique next-node pointers.

        In a feasible integer solution each node has out-degree exactly 1,
        so edges form a set of disjoint directed cycles.  Following successor
        pointers from each unprocessed start node recovers those cycles without
        any ambiguity.
        """
        next_node = {u: v for u, v in edges}
        processed: set = set()
        cycles: List[List[int]] = []

        for start_node in next_node:
            if start_node in processed:
                continue

            cycle: List[int] = []
            curr = start_node
            while curr not in processed:
                processed.add(curr)
                cycle.append(curr)
                curr = next_node.get(curr)
                if curr is None:
                    break  # open path — shouldn't occur when degree constraints hold

            cycles.append(cycle)

        return cycles

    def _extract_routes_and_check_capacity(self, edges: List[Tuple[int, int]], depot_id: int):
        next_node = {u: v for u, v in edges}
        routes = []
        is_valid = True
        
        # Start from depot. Depot degree constraint is sum out >= 1? 
        # Actually in formulation above: sum out = sum in.
        # So there might be multiple routes starting from depot.
        
        # We need to find all edges starting from depot
        starts = [v for u, v in edges if u == depot_id]
        
        visited_nodes = set()
        
        for start in starts:
            route = []
            curr = start
            while curr != depot_id:
                route.append(curr)
                visited_nodes.add(curr)
                curr = next_node.get(curr)
                if curr is None: break 
            routes.append(route)

        # There might be cycles NOT connected to depot, which _find_subtours handles.
        # But here we focus on routes connected to depot for capacity check.
        
        for r in routes:
            load = sum(self.model.node_map[n_id].demand for n_id in r)
            if load > self.model.capacity:
                is_valid = False
        
        return routes, is_valid

    def _construct_solution(self, routes_data: List[List[int]]) -> Solution:
        routes_objs = []
        total_cost = 0.0
        for idx, r_nodes_ids in enumerate(routes_data):
            nodes = [self.model.node_map[nid] for nid in r_nodes_ids]
            
            # Recalculate cost
            cost = 0.0
            load = sum(n.demand for n in nodes)
            if nodes:
                cost += self.model.get_distance(self.model.depot_id, nodes[0].id)
                for i in range(len(nodes) - 1):
                    cost += self.model.get_distance(nodes[i].id, nodes[i+1].id)
                cost += self.model.get_distance(nodes[-1].id, self.model.depot_id)
            
            total_cost += cost
            routes_objs.append(Route(id=idx+1, load=load, cost=cost, sequence_of_nodes=nodes))
            
        return Solution(cost=total_cost, routes=routes_objs)