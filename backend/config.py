"""
Configuration settings for Intelligent Routing System
"""
import torch


class Config:
    """Central configuration for the routing system"""

    # Network settings
    DEFAULT_NETWORK_SIZE = 10
    MIN_EDGE_WEIGHT = 1
    MAX_EDGE_WEIGHT = 100
    SPARSITY_FACTOR = 0.3  # 30% connectivity for sparse graph

    # Q-Learning parameters
    QL_LEARNING_RATE = 0.1
    QL_DISCOUNT_FACTOR = 0.95
    QL_EPSILON = 0.1  # Exploration rate
    QL_EPSILON_DECAY = 0.995
    QL_MIN_EPSILON = 0.01
    QL_EPISODES = 1000

    # Deep Q-Learning parameters
    DQN_LEARNING_RATE = 0.001
    DQN_DISCOUNT_FACTOR = 0.99
    DQN_EPSILON_START = 1.0
    DQN_EPSILON_END = 0.01
    DQN_EPSILON_DECAY = 0.995
    DQN_BATCH_SIZE = 64
    DQN_MEMORY_SIZE = 10000
    DQN_TARGET_UPDATE = 10
    DQN_EPISODES = 500
    DQN_HIDDEN_SIZE = 128

    # Reward design
    REWARD_DESTINATION = 100.0
    REWARD_HOP_PENALTY = -1.0
    REWARD_CONGESTION_PENALTY = -5.0
    REWARD_ENERGY_PENALTY = -2.0
    REWARD_LOOP_PENALTY = -50.0
    REWARD_INVALID_ACTION = -100.0

    # Dynamic topology
    LINK_FAILURE_PROBABILITY = 0.1
    LATENCY_INCREASE_FACTOR = 1.5

    # Performance metrics
    METRICS = ['path_cost', 'latency', 'energy_efficiency',
               'packet_delivery_rate', 'convergence_time']

    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Visualization
    ANIMATION_SPEED = 0.5  # seconds per hop
    NODE_SIZE = 1000
    EDGE_WIDTH = 2.0

    # Export settings
    RESULTS_DIR = "results"
    MODELS_DIR = "models"

    @staticmethod
    def get_network_topologies():
        """Return available network topology types"""
        return ['mesh', 'ring', 'tree', 'random', 'custom']
