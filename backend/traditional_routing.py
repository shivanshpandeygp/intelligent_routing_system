"""
Traditional Routing Algorithms: Dijkstra and Bellman-Ford
"""
import numpy as np
import networkx as nx
from typing import List, Tuple, Optional
import time
import logging

logger = logging.getLogger(__name__)


class TraditionalRouting:
    """Implementation of Dijkstra and Bellman-Ford routing algorithms"""

    def __init__(self, network_graph):
        """
        Initialize traditional routing

        Args:
            network_graph: NetworkGraph instance
        """
        self.network = network_graph

    def dijkstra(self, source: int, destination: int) -> Tuple[List[int], float, float]:
        """
        Dijkstra's shortest path algorithm

        Returns:
            path: List of nodes in the path
            cost: Total path cost
            time_taken: Algorithm execution time
        """
        start_time = time.time()

        # Initialize distances and previous nodes
        distances = {node: float('inf') for node in self.network.graph.nodes()}
        distances[source] = 0
        previous = {node: None for node in self.network.graph.nodes()}
        unvisited = set(self.network.graph.nodes())

        while unvisited:
            # Find node with minimum distance
            current = min(unvisited, key=lambda node: distances[node])

            if distances[current] == float('inf'):
                break

            if current == destination:
                break

            unvisited.remove(current)

            # Update distances to neighbors
            for neighbor in self.network.get_neighbors(current):
                if neighbor in unvisited:
                    edge_weight = self.network.get_edge_weight(current, neighbor)
                    new_distance = distances[current] + edge_weight

                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current

        # Reconstruct path
        path = []
        current = destination

        if previous[current] is not None or current == source:
            while current is not None:
                path.insert(0, current)
                current = previous[current]

        time_taken = time.time() - start_time
        cost = distances[destination] if distances[destination] != float('inf') else -1

        logger.info(f"Dijkstra: {source} -> {destination}, "
                    f"cost={cost:.2f}, time={time_taken:.4f}s")

        return path, cost, time_taken

    def bellman_ford(self, source: int, destination: int) -> Tuple[List[int], float, float]:
        """
        Bellman-Ford shortest path algorithm (handles negative weights)

        Returns:
            path: List of nodes in the path
            cost: Total path cost
            time_taken: Algorithm execution time
        """
        start_time = time.time()

        # Initialize distances and previous nodes
        distances = {node: float('inf') for node in self.network.graph.nodes()}
        distances[source] = 0
        previous = {node: None for node in self.network.graph.nodes()}

        # Relax edges repeatedly
        for _ in range(self.network.num_nodes - 1):
            for u in self.network.graph.nodes():
                for v in self.network.get_neighbors(u):
                    edge_weight = self.network.get_edge_weight(u, v)

                    if distances[u] + edge_weight < distances[v]:
                        distances[v] = distances[u] + edge_weight
                        previous[v] = u

        # Check for negative cycles
        for u in self.network.graph.nodes():
            for v in self.network.get_neighbors(u):
                edge_weight = self.network.get_edge_weight(u, v)
                if distances[u] + edge_weight < distances[v]:
                    logger.warning("Negative cycle detected!")
                    return [], -1, time.time() - start_time

        # Reconstruct path
        path = []
        current = destination

        if previous[current] is not None or current == source:
            while current is not None:
                path.insert(0, current)
                current = previous[current]

        time_taken = time.time() - start_time
        cost = distances[destination] if distances[destination] != float('inf') else -1

        logger.info(f"Bellman-Ford: {source} -> {destination}, "
                    f"cost={cost:.2f}, time={time_taken:.4f}s")

        return path, cost, time_taken

    def get_all_paths(self, source: int, destination: int,
                      max_paths: int = 5) -> List[Tuple[List[int], float]]:
        """Get multiple paths between source and destination"""
        try:
            paths = []
            for path in nx.all_simple_paths(self.network.graph.to_undirected(),
                                            source, destination):
                cost = sum(self.network.get_edge_weight(path[i], path[i + 1])
                           for i in range(len(path) - 1))
                paths.append((path, cost))

                if len(paths) >= max_paths:
                    break

            # Sort by cost
            paths.sort(key=lambda x: x[1])
            return paths
        except nx.NetworkXNoPath:
            return []