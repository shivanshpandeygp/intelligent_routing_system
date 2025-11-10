"""
Backend Module - Core Routing Algorithms and Network Management

This module provides the core functionality for network graph management,
routing algorithms (traditional and RL-based), reward design, and
performance metrics calculation.
"""

from .graph_manager import NetworkGraph
from .traditional_routing import TraditionalRouting
from .q_learning_routing import QLearningRouting
from .dqn_routing import DQNRouting, DQNetwork
from .reward_designer import RewardDesigner
from .performance_metrics import PerformanceMetrics
from .config import Config

__all__ = [
    # Graph Management
    'NetworkGraph',

    # Traditional Algorithms
    'TraditionalRouting',

    # RL Algorithms
    'QLearningRouting',
    'DQNRouting',
    'DQNetwork',

    # Utilities
    'RewardDesigner',
    'PerformanceMetrics',
    'Config',
]

# Version info
__version__ = "1.0.0"

# Module-level configuration
import logging

logger = logging.getLogger(__name__)
logger.info("Backend module initialized")
