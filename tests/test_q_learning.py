"""
Unit tests for Q-Learning routing
"""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.graph_manager import NetworkGraph
from backend.q_learning_routing import QLearningRouting
from backend.config import Config


class TestQLearningRouting(unittest.TestCase):
    """Test Q-Learning routing"""

    def setUp(self):
        """Set up test environment"""
        self.network = NetworkGraph(num_nodes=10, topology='mesh')
        config = Config()
        config.QL_EPISODES = 100  # Reduced for testing
        self.ql = QLearningRouting(self.network, config)

    def test_initialization(self):
        """Test Q-table initialization"""
        self.assertIsInstance(self.ql.q_table, dict)
        self.assertGreater(len(self.ql.q_table), 0)

    def test_training(self):
        """Test training process"""
        self.ql.train(0, 8, num_episodes=50)

        # Check training history
        self.assertEqual(len(self.ql.training_history['episodes']), 50)
        self.assertEqual(len(self.ql.training_history['rewards']), 50)

    def test_find_path_after_training(self):
        """Test path finding after training"""
        self.ql.train(0, 8, num_episodes=100)
        path, cost, time_taken = self.ql.find_path(0, 8)

        if len(path) > 0:
            self.assertEqual(path[0], 0)
            self.assertEqual(path[-1], 8)
            self.assertGreater(cost, 0)

    def test_q_table_updates(self):
        """Test that Q-table is being updated"""
        initial_q = dict(self.ql.q_table)
        self.ql.train(0, 5, num_episodes=50)

        # Check that some Q-values changed
        changed = False
        for state in initial_q:
            if state in self.ql.q_table:
                for action in initial_q[state]:
                    if action in self.ql.q_table[state]:
                        if initial_q[state][action] != self.ql.q_table[state][action]:
                            changed = True
                            break

        self.assertTrue(changed, "Q-table did not update during training")


if __name__ == '__main__':
    unittest.main()
