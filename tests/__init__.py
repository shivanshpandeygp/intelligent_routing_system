"""
Test Suite for Intelligent Routing System

This module contains unit tests for all components including
graph management, traditional routing, and RL-based routing.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

__all__ = [
    'test_traditional',
    'test_q_learning',
    'test_dqn',
    'test_graph_manager',
]

__version__ = "1.0.0"

def run_all_tests():
    """Run all test suites"""
    loader = unittest.TestLoader()
    suite = loader.discover(str(Path(__file__).parent), pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)

# Test configuration
TEST_CONFIG = {
    'small_network_size': 10,
    'medium_network_size': 20,
    'large_network_size': 50,
    'training_episodes': 100,  # Reduced for testing
    'timeout': 30,  # seconds
}
