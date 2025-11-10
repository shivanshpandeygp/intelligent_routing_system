"""
Unit tests for network graph manager
"""
import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.graph_manager import NetworkGraph


class TestNetworkGraph(unittest.TestCase):
    """Test network graph operations"""

    def test_mesh_creation(self):
        """Test mesh topology creation"""
        network = NetworkGraph(num_nodes=16, topology='mesh')

        self.assertEqual(network.num_nodes, 16)
        self.assertEqual(network.topology, 'mesh')
        self.assertGreater(network.graph.number_of_edges(), 0)

    def test_ring_creation(self):
        """Test ring topology creation"""
        network = NetworkGraph(num_nodes=10, topology='ring')

        self.assertEqual(network.num_nodes, 10)
        self.assertEqual(network.topology, 'ring')

    def test_tree_creation(self):
        """Test tree topology creation"""
        network = NetworkGraph(num_nodes=15, topology='tree')

        self.assertEqual(network.num_nodes, 15)
        self.assertEqual(network.topology, 'tree')

    def test_random_creation(self):
        """Test random topology creation"""
        network = NetworkGraph(num_nodes=10, topology='random', sparsity=0.3)

        self.assertEqual(network.num_nodes, 10)
        self.assertTrue(network.is_connected())

    def test_get_neighbors(self):
        """Test neighbor retrieval"""
        network = NetworkGraph(num_nodes=10, topology='ring')
        neighbors = network.get_neighbors(0)

        self.assertIsInstance(neighbors, list)
        self.assertGreater(len(neighbors), 0)

    def test_link_failure(self):
        """Test link failure simulation"""
        network = NetworkGraph(num_nodes=10, topology='mesh')
        initial_active = sum(1 for metrics in network.edge_metrics.values()
                             if metrics['active'])

        failed = network.simulate_link_failure(failure_prob=0.3)

        current_active = sum(1 for metrics in network.edge_metrics.values()
                             if metrics['active'])

        self.assertLess(current_active, initial_active)

    def test_adjacency_matrix(self):
        """Test adjacency matrix generation"""
        network = NetworkGraph(num_nodes=5, topology='ring')
        adj_matrix = network.get_adjacency_matrix()

        self.assertEqual(adj_matrix.shape, (5, 5))
        self.assertTrue(np.all(np.diag(adj_matrix) == 0))


if __name__ == '__main__':
    unittest.main()
