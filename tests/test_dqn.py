"""
Unit tests for DQN routing
"""
import unittest
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.graph_manager import NetworkGraph
from backend.dqn_routing import DQNRouting, DQNetwork
from backend.config import Config


class TestDQNRouting(unittest.TestCase):
    """Test DQN routing"""

    def setUp(self):
        """Set up test environment"""
        self.network = NetworkGraph(num_nodes=10, topology='ring')
        config = Config()
        config.DQN_EPISODES = 50  # Reduced for testing
        self.dqn = DQNRouting(self.network, config)

    def test_network_initialization(self):
        """Test DQN network initialization"""
        self.assertIsInstance(self.dqn.policy_net, DQNetwork)
        self.assertIsInstance(self.dqn.target_net, DQNetwork)

    def test_state_encoding(self):
        """Test state encoding"""
        state = self.dqn._encode_state(0, 5, set([0]), 0)

        self.assertIsInstance(state, torch.Tensor)
        self.assertEqual(state.shape[1], self.dqn.state_size)

    def test_training(self):
        """Test training process"""
        self.dqn.train(0, 7, num_episodes=30)

        # Check training history
        self.assertGreater(len(self.dqn.training_history['episodes']), 0)
        self.assertGreater(len(self.dqn.training_history['rewards']), 0)

    def test_memory_storage(self):
        """Test experience replay memory"""
        initial_size = len(self.dqn.memory)
        self.dqn.train(0, 5, num_episodes=10)

        self.assertGreater(len(self.dqn.memory), initial_size)

    def test_find_path(self):
        """Test path finding"""
        self.dqn.train(0, 8, num_episodes=50)
        path, cost, time_taken = self.dqn.find_path(0, 8)

        if len(path) > 0:
            self.assertEqual(path[0], 0)
            self.assertGreater(cost, 0)


if __name__ == '__main__':
    unittest.main()
