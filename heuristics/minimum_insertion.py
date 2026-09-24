"""
Minimum (cheapest) insertion construction heuristic for the CVRP.

Dependencies: standard library only.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Dict, List, Optional, Tuple

from model import VRPModel, Route, Solution, Node

logger = logging.getLogger(__name__)

_STRATEGIES = ("sequential", "parallel")
_SEED_RULES = ("farthest", "max_demand", "random")

# (delta_cost, customer, position_in_route, route_index)
_Candidate = Tuple[float, Node, int, int]


class MinimumInsertion:
    """
    Construct a feasible CVRP solution with the minimum insertion criterion.

    At every step the heuristic evaluates the insertion of each unrouted
    customer u between every pair of consecutive nodes (i, j) of a partial
    route and charges the detour it causes:

        delta(i, u, j) = d(i, u) + d(u, j) - mu * d(i, j)

    The cheapest feasible (customer, route, position) triple is committed, and
    the process repeats until every customer is routed.

    Two construction strategies are supported:

    * 'sequential' - routes are filled one at a time. A route is seeded, grown
      until no unrouted customer fits within capacity, closed, and the next
      route is opened.
    * 'parallel'   - a pool of routes is seeded up front (by default
      ceil(total_demand / capacity) of them, i.e. the bin-packing lower bound)
      and every insertion competes across all open routes. Extra routes are
      opened on demand if a customer no longer fits anywhere.
    """

    def __init__(
        self,
        model: VRPModel,
        strategy: str = "sequential",
        seed_rule: str = "farthest",
        mu: float = 1.0,
        num_routes: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}. Valid options: {list(_STRATEGIES)}")
        if seed_rule not in _SEED_RULES:
            raise ValueError(f"Unknown seed_rule {seed_rule!r}. Valid options: {list(_SEED_RULES)}")
        if num_routes is not None and num_routes < 1:
            raise ValueError("num_routes must be a positive integer.")

        self.model = model
        self.strategy = strategy
        self.seed_rule = seed_rule
        self.mu = mu
        self.num_routes = num_routes
        # Local RNG keeps the solver re-entrant: no global random state is touched.
        self.rng = random.Random(seed)

        self.customers: List[Node] = [n for n in model.nodes if n.id != model.depot_id]

        oversized = [n.id for n in self.customers if n.demand > model.capacity]
        if oversized:
            raise ValueError(
                f"Instance is infeasible: customers {oversized} have demand above "
                f"vehicle capacity {model.capacity}."
            )

        # Cache the distance matrix once; the criterion is evaluated O(n^3) times.
        self._dist: Dict[int, Dict[int, float]] = {
            a.id: {b.id: model.get_distance(a.id, b.id) for b in model.nodes}
            for a in model.nodes
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def solve(self) -> Solution:
        """Build a feasible solution and return it."""
        logger.info(
            "Minimum insertion on %s: %d customers, capacity %d, strategy=%s, seed_rule=%s, mu=%.2f",
            self.model.name, len(self.customers), self.model.capacity,
            self.strategy, self.seed_rule, self.mu,
        )

        unrouted = list(self.customers)
        if not unrouted:
            return Solution(cost=0.0, routes=[])

        if self.strategy == "sequential":
            sequences = self._build_sequential(unrouted)
        else:
            sequences = self._build_parallel(unrouted)

        routes = [self._make_route(rid, seq) for rid, seq in enumerate(sequences, start=1)]
        solution = Solution(cost=sum(r.cost for r in routes), routes=routes)

        self._validate(solution)
        logger.info("Constructed %d routes, total cost %.2f", len(routes), solution.cost)
        return solution

    # ------------------------------------------------------------------ #
    # Construction strategies
    # ------------------------------------------------------------------ #

    def _build_sequential(self, unrouted: List[Node]) -> List[List[Node]]:
        """Fill one route at a time until it admits no further customer."""
        sequences: List[List[Node]] = []

        while unrouted:
            seed_node = self._pick_seed(unrouted)
            unrouted.remove(seed_node)
            seq = [seed_node]
            load = seed_node.demand

            while unrouted:
                best = self._best_insertion(seq, load, unrouted)
                if best is None:
                    break
                _, node, position = best
                seq.insert(position, node)
                load += node.demand
                unrouted.remove(node)

            logger.debug(
                "Route %d closed with %d customers (load %d).", len(sequences) + 1, len(seq), load
            )
            sequences.append(seq)

        return sequences

    def _build_parallel(self, unrouted: List[Node]) -> List[List[Node]]:
        """Grow every route simultaneously, always committing the global minimum."""
        total_demand = sum(n.demand for n in unrouted)
        target = self.num_routes or math.ceil(total_demand / self.model.capacity)
        target = max(1, min(target, len(unrouted)))

        sequences: List[List[Node]] = []
        loads: List[int] = []
        for seed_node in self._pick_parallel_seeds(unrouted, target):
            unrouted.remove(seed_node)
            sequences.append([seed_node])
            loads.append(seed_node.demand)

        while unrouted:
            best: Optional[_Candidate] = None
            for route_index, seq in enumerate(sequences):
                local = self._best_insertion(seq, loads[route_index], unrouted)
                if local is None:
                    continue
                delta, node, position = local
                if best is None or delta < best[0] or (
                    delta == best[0]
                    and (node.id, position, route_index) < (best[1].id, best[2], best[3])
                ):
                    best = (delta, node, position, route_index)

            if best is None:
                # Every open route is too full for every remaining customer.
                seed_node = self._pick_seed(unrouted)
                unrouted.remove(seed_node)
                sequences.append([seed_node])
                loads.append(seed_node.demand)
                logger.debug(
                    "Opened overflow route %d seeded with customer %d.", len(sequences), seed_node.id
                )
                continue

            _, node, position, route_index = best
            sequences[route_index].insert(position, node)
            loads[route_index] += node.demand
            unrouted.remove(node)

        return sequences

    # ------------------------------------------------------------------ #
    # Minimum insertion criterion
    # ------------------------------------------------------------------ #

    def _best_insertion(
        self, seq: List[Node], load: int, candidates: List[Node]
    ) -> Optional[Tuple[float, Node, int]]:
        """
        Cheapest feasible insertion of any candidate into `seq`.

        Returns (delta_cost, customer, position) or None if nothing fits. Ties
        are broken by lowest customer id, then lowest position, so the
        construction is deterministic for a given input ordering.
        """
        dist = self._dist
        depot_id = self.model.depot_id
        residual = self.model.capacity - load
        mu = self.mu

        best: Optional[Tuple[float, Node, int]] = None

        for node in candidates:
            if node.demand > residual:
                continue
            node_dist = dist[node.id]

            for position in range(len(seq) + 1):
                prev_id = depot_id if position == 0 else seq[position - 1].id
                next_id = depot_id if position == len(seq) else seq[position].id
                delta = dist[prev_id][node.id] + node_dist[next_id] - mu * dist[prev_id][next_id]

                if best is None or delta < best[0] or (
                    delta == best[0] and (node.id, position) < (best[1].id, best[2])
                ):
                    best = (delta, node, position)

        return best

    # ------------------------------------------------------------------ #
    # Seeding
    # ------------------------------------------------------------------ #

    def _pick_seed(self, candidates: List[Node]) -> Node:
        """Choose the customer that opens a new route."""
        if self.seed_rule == "random":
            return self.rng.choice(candidates)
        if self.seed_rule == "max_demand":
            return max(candidates, key=lambda n: (n.demand, -n.id))
        # 'farthest': the customer most expensive to reach pins the route's direction.
        depot_id = self.model.depot_id
        return max(candidates, key=lambda n: (self._dist[depot_id][n.id], -n.id))

    def _pick_parallel_seeds(self, candidates: List[Node], count: int) -> List[Node]:
        """
        Pick `count` mutually dispersed seeds so the parallel routes start out
        pointing at different parts of the map.
        """
        seeds = [self._pick_seed(candidates)]
        remaining = [n for n in candidates if n is not seeds[0]]

        while len(seeds) < count and remaining:
            if self.seed_rule == "random":
                pick = self.rng.choice(remaining)
            else:
                # Max-min dispersion: farthest from the closest seed already chosen.
                pick = max(
                    remaining,
                    key=lambda n: (min(self._dist[s.id][n.id] for s in seeds), -n.id),
                )
            seeds.append(pick)
            remaining.remove(pick)

        return seeds

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _make_route(self, rid: int, seq: List[Node]) -> Route:
        return Route(
            id=rid,
            load=sum(n.demand for n in seq),
            cost=self._route_cost(seq),
            sequence_of_nodes=list(seq),
        )

    def _route_cost(self, seq: List[Node]) -> float:
        """Closed-tour cost depot -> seq -> depot."""
        if not seq:
            return 0.0
        dist = self._dist
        depot_id = self.model.depot_id
        cost = dist[depot_id][seq[0].id]
        for a, b in zip(seq, seq[1:]):
            cost += dist[a.id][b.id]
        cost += dist[seq[-1].id][depot_id]
        return float(cost)

    def _validate(self, solution: Solution) -> None:
        """Assert the constructed solution is feasible before handing it back."""
        depot_id = self.model.depot_id
        expected = {n.id for n in self.customers}
        served: List[int] = []

        for route in solution.routes:
            ids = [n.id for n in route.sequence_of_nodes]
            assert ids, f"Route {route.id} is empty."
            assert depot_id not in ids, f"Route {route.id} contains the depot in its sequence."
            assert len(ids) == len(set(ids)), f"Route {route.id} revisits a customer."
            assert route.load <= self.model.capacity, (
                f"Route {route.id} exceeds capacity: {route.load} > {self.model.capacity}."
            )
            assert route.load == sum(n.demand for n in route.sequence_of_nodes), (
                f"Route {route.id} load does not match the demands it serves."
            )
            served.extend(ids)

        assert len(served) == len(set(served)), "A customer is served by more than one route."
        assert set(served) == expected, (
            f"Customer coverage mismatch: missing {sorted(expected - set(served))}, "
            f"unexpected {sorted(set(served) - expected)}."
        )


def solve(instance: VRPModel, **kwargs) -> Solution:
    """
    Solve the given CVRP instance using the Minimum Insertion heuristic.

    Args:
        instance: A fully constructed VRPModel from setup.py.
        **kwargs: strategy ('sequential' | 'parallel'), seed_rule
                  ('farthest' | 'max_demand' | 'random'), mu, num_routes, seed.

    Returns:
        A Solution object with routes and total cost.
    """
    return MinimumInsertion(instance, **kwargs).solve()
