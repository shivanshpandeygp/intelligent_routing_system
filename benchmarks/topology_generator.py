"""
Network Topology Generator for Benchmarking
"""
import networkx as nx
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class TopologyGenerator:
    """Generate various network topologies for benchmarking"""

    @staticmethod
    def generate_mesh(size: int) -> Dict:
        """Generate mesh topology"""
        grid_size = int(np.sqrt(size))
        actual_size = grid_size * grid_size

        graph = nx.DiGraph()
        graph.add_nodes_from(range(actual_size))

        edges = []
        for i in range(grid_size):
            for j in range(grid_size):
                node = i * grid_size + j

                # Right neighbor
                if j < grid_size - 1:
                    right = node + 1
                    weight = np.random.randint(1, 20)
                    edges.append((node, right, weight))
                    edges.append((right, node, weight))

                # Bottom neighbor
                if i < grid_size - 1:
                    bottom = node + grid_size
                    weight = np.random.randint(1, 20)
                    edges.append((node, bottom, weight))
                    edges.append((bottom, node, weight))

        for u, v, w in edges:
            graph.add_edge(u, v, weight=w)

        return {
            'graph': graph,
            'type': 'mesh',
            'size': actual_size,
            'edges': len(edges)
        }

    @staticmethod
    def generate_ring(size: int) -> Dict:
        """Generate ring topology"""
        graph = nx.DiGraph()
        graph.add_nodes_from(range(size))

        edges = []
        for i in range(size):
            next_node = (i + 1) % size
            weight = np.random.randint(1, 20)
            edges.append((i, next_node, weight))
            edges.append((next_node, i, weight))

        for u, v, w in edges:
            graph.add_edge(u, v, weight=w)

        return {
            'graph': graph,
            'type': 'ring',
            'size': size,
            'edges': len(edges)
        }

    @staticmethod
    def generate_tree(size: int) -> Dict:
        """Generate binary tree topology"""
        graph = nx.DiGraph()
        graph.add_nodes_from(range(size))

        edges = []
        for i in range(size):
            left = 2 * i + 1
            right = 2 * i + 2

            if left < size:
                weight = np.random.randint(1, 20)
                edges.append((i, left, weight))
                edges.append((left, i, weight))

            if right < size:
                weight = np.random.randint(1, 20)
                edges.append((i, right, weight))
                edges.append((right, i, weight))

        for u, v, w in edges:
            graph.add_edge(u, v, weight=w)

        return {
            'graph': graph,
            'type': 'tree',
            'size': size,
            'edges': len(edges)
        }

    @staticmethod
    def generate_random(size: int, density: float = 0.3) -> Dict:
        """Generate random sparse graph"""
        graph = nx.DiGraph()
        graph.add_nodes_from(range(size))

        # Spanning tree for connectivity
        spanning_tree = nx.random_tree(size)

        edges = []
        for u, v in spanning_tree.edges():
            weight = np.random.randint(1, 20)
            edges.append((u, v, weight))
            edges.append((v, u, weight))

        # Add random edges
        max_edges = size * (size - 1)
        target_edges = int(max_edges * density)
        current_edges = len(edges)

        while current_edges < target_edges:
            u = np.random.randint(0, size)
            v = np.random.randint(0, size)

            if u != v and not graph.has_edge(u, v):
                weight = np.random.randint(1, 20)
                edges.append((u, v, weight))
                current_edges += 1

        for u, v, w in edges:
            graph.add_edge(u, v, weight=w)

        return {
            'graph': graph,
            'type': 'random',
            'size': size,
            'edges': len(edges)
        }

    @staticmethod
    def generate_all_topologies(sizes: List[int] = [10, 20, 30]) -> Dict[str, List[Dict]]:
        """Generate all topology types for multiple sizes"""
        topologies = {
            'mesh': [],
            'ring': [],
            'tree': [],
            'random': []
        }

        for size in sizes:
            topologies['mesh'].append(TopologyGenerator.generate_mesh(size))
            topologies['ring'].append(TopologyGenerator.generate_ring(size))
            topologies['tree'].append(TopologyGenerator.generate_tree(size))
            topologies['random'].append(TopologyGenerator.generate_random(size))

        logger.info(f"Generated topologies for sizes: {sizes}")
        return topologies
