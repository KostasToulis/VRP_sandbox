import math
import pulp
from typing import Dict, List, Set, Tuple
from model import VRPModel, Solution, Route


class BranchAndCut:
    def __init__(self, model: VRPModel, time_limit: int = 600):
        self.model = model
        self.time_limit = time_limit

    def solve(self) -> Solution:
        node_ids     = [n.id for n in self.model.nodes]
        customer_ids = [n.id for n in self.model.nodes if n.id != self.model.depot_id]
        depot_id     = self.model.depot_id
        capacity     = self.model.capacity
        demand: Dict[int, int] = {n.id: n.demand for n in self.model.nodes}

        prob = pulp.LpProblem("CVRP_Branch_and_Cut", pulp.LpMinimize)

        # ------------------------------------------------------------------
        # Routing variables: x[i,j] = 1 iff vehicle travels i → j
        # ------------------------------------------------------------------
        x: Dict[Tuple[int, int], pulp.LpVariable] = {
            (i, j): pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
            for i in node_ids for j in node_ids if i != j
        }

        # Objective — single lpSum (assigning inside a loop would replace
        # rather than accumulate the PuLP objective on every iteration).
        prob += pulp.lpSum(
            self.model.get_distance(i, j) * x[i, j]
            for i in node_ids for j in node_ids if i != j
        )

        # ------------------------------------------------------------------
        # Degree constraints: each customer has exactly one in- and out-edge
        # ------------------------------------------------------------------
        for i in customer_ids:
            prob += pulp.lpSum(x[i, j] for j in node_ids if i != j) == 1
            prob += pulp.lpSum(x[j, i] for j in node_ids if i != j) == 1

        # Depot flow balance: vehicles that leave must return
        prob += (
            pulp.lpSum(x[depot_id, j] for j in customer_ids)
            == pulp.lpSum(x[j, depot_id] for j in customer_ids)
        )

        # ------------------------------------------------------------------
        # Minimum vehicles lower bound
        #
        # At least ceil(total_demand / Q) routes must depart the depot.
        # This tightens the LP relaxation at the root node and helps CBC
        # avoid exploring configurations with too few vehicles.
        # ------------------------------------------------------------------
        total_demand = sum(demand[i] for i in customer_ids)
        min_vehicles = math.ceil(total_demand / capacity)
        prob += pulp.lpSum(x[depot_id, j] for j in customer_ids) >= min_vehicles

        # ------------------------------------------------------------------
        # MTZ load variables: u[i] ∈ [q_i, Q]  for each customer i
        #
        # u[i] is the cumulative vehicle load *after* serving node i.
        # Lower bound q_i: the vehicle must have at least loaded i's demand.
        # Upper bound Q: enforces the capacity constraint implicitly.
        # ------------------------------------------------------------------
        u: Dict[int, pulp.LpVariable] = {
            i: pulp.LpVariable(f"u_{i}", lowBound=demand[i], upBound=capacity)
            for i in customer_ids
        }

        # ------------------------------------------------------------------
        # MTZ subtour-elimination constraints
        #
        # For every pair of customer nodes i ≠ j:
        #   u[j] - u[i] >= q[j] - Q * (1 - x[i,j])
        #
        # x[i,j]=1  →  u[j] >= u[i] + q[j]   (load strictly increases)
        # x[i,j]=0  →  u[j] - u[i] >= q[j]-Q  (always satisfied, inactive)
        #
        # Monotone load accumulation makes any closed customer-only cycle
        # algebraically impossible, so no iterative cut-adding loop is needed.
        # Edges from the depot are omitted: those reduce to u[j] >= q[j],
        # already enforced by the variable lower bound.
        # ------------------------------------------------------------------
        for i in customer_ids:
            for j in customer_ids:
                if i != j:
                    prob += u[j] - u[i] >= demand[j] - capacity * (1 - x[i, j])

        # ------------------------------------------------------------------
        # Warm start: greedy nearest-neighbour heuristic
        #
        # Providing CBC with a feasible incumbent before the ILP solve lets it
        # prune every branch whose bound already exceeds the greedy cost.
        # For small instances this can reduce the B&B tree by orders of magnitude.
        # ------------------------------------------------------------------
        self._warm_start(x, customer_ids, depot_id, demand, capacity)

        # ------------------------------------------------------------------
        # Solve — a single call is sufficient because all constraints are static
        # ------------------------------------------------------------------
        status = prob.solve(
            pulp.PULP_CBC_CMD(msg=0, timeLimit=self.time_limit, warmStart=True)
        )

        if status != pulp.LpStatusOptimal:
            print("Optimal solution not found or time limit reached.")
            return Solution(cost=float('inf'), routes=[])

        edges = [
            (i, j)
            for i in node_ids for j in node_ids
            if i != j and pulp.value(x[i, j]) > 0.9
        ]
        return self._construct_solution(self._extract_routes(edges, depot_id))

    # ------------------------------------------------------------------
    # Warm start: nearest-neighbour construction heuristic
    # ------------------------------------------------------------------
    def _warm_start(
        self,
        x: Dict[Tuple[int, int], pulp.LpVariable],
        customer_ids: List[int],
        depot_id: int,
        demand: Dict[int, int],
        capacity: int,
    ) -> None:
        """Build a greedy feasible solution and inject it as an ILP incumbent.

        Nearest-neighbour: from the current node always move to the closest
        unvisited customer that still fits in the current vehicle.  When no
        customer fits, return to the depot and open a new route.
        """
        unvisited: Set[int] = set(customer_ids)
        active_edges: Set[Tuple[int, int]] = set()

        while unvisited:
            curr, load = depot_id, 0

            while True:
                best, best_dist = None, float('inf')
                for nxt in unvisited:
                    if load + demand[nxt] <= capacity:
                        d = self.model.get_distance(curr, nxt)
                        if d < best_dist:
                            best, best_dist = nxt, d

                if best is None:
                    break

                active_edges.add((curr, best))
                load += demand[best]
                unvisited.discard(best)
                curr = best

            active_edges.add((curr, depot_id))

        for (i, j), var in x.items():
            var.setInitialValue(1 if (i, j) in active_edges else 0)

    # ------------------------------------------------------------------
    # Helper: extract ordered customer sequences for each route
    # ------------------------------------------------------------------
    def _extract_routes(
        self, edges: List[Tuple[int, int]], depot_id: int
    ) -> List[List[int]]:
        next_node = {u: v for u, v in edges}
        routes: List[List[int]] = []

        for start in (v for u, v in edges if u == depot_id):
            route: List[int] = []
            curr = start
            while curr != depot_id:
                route.append(curr)
                curr = next_node.get(curr)
                if curr is None:
                    break
            routes.append(route)

        return routes

    # ------------------------------------------------------------------
    # Helper: build Solution object from raw node-id sequences
    # ------------------------------------------------------------------
    def _construct_solution(self, routes_data: List[List[int]]) -> Solution:
        route_objs: List[Route] = []
        total_cost = 0.0

        for idx, node_ids in enumerate(routes_data):
            nodes = [self.model.node_map[nid] for nid in node_ids]
            load  = sum(n.demand for n in nodes)
            cost  = 0.0

            if nodes:
                cost += self.model.get_distance(self.model.depot_id, nodes[0].id)
                for k in range(len(nodes) - 1):
                    cost += self.model.get_distance(nodes[k].id, nodes[k + 1].id)
                cost += self.model.get_distance(nodes[-1].id, self.model.depot_id)

            total_cost += cost
            route_objs.append(Route(id=idx + 1, load=load, cost=cost, sequence_of_nodes=nodes))

        return Solution(cost=total_cost, routes=route_objs)
