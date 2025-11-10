"""
Unit tests for traditional routing algorithms
"""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.graph_manager import NetworkGraph
from backend.traditional_routing import TraditionalRouting


class TestTraditionalRouting(unittest.TestCase):
    """Test traditional routing algorithms"""

    def setUp(self):
        """Set up test network"""
        self.network = NetworkGraph(num_nodes=10, topology='random', sparsity=0.3)
        self.routing = TraditionalRouting(self.network)

    def test_dijkstra_finds_path(self):
        """Test that Dijkstra finds a path"""
        path, cost, time_taken = self.routing.dijkstra(0, 5)

        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 5)
        self.assertGreaterEqual(cost, 0)

    def test_bellman_ford_finds_path(self):
        """Test that Bellman-Ford finds a path"""
        path, cost, time_taken = self.routing.bellman_ford(0, 5)

        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 5)

    def test_dijkstra_same_source_dest(self):
        """Test same source and destination"""
        path, cost, time_taken = self.routing.dijkstra(3, 3)

        self.assertEqual(len(path), 1)
        self.assertEqual(path[0], 3)
        self.assertEqual(cost, 0)

    def test_paths_are_valid(self):
        """Test that paths are valid (consecutive nodes are connected)"""
        path, cost, time_taken = self.routing.dijkstra(0, 8)

        for i in range(len(path) - 1):
            neighbors = list(self.network.graph.neighbors(path[i]))
            self.assertIn(path[i + 1], neighbors,
                          f"Node {path[i + 1]} not neighbor of {path[i]}")


if __name__ == '__main__':
    unittest.main()
