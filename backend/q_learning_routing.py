"""
Q-Learning Based Routing Algorithm with Transfer Learning
"""
import numpy as np
import logging
from typing import List, Tuple, Dict
import time
import pickle
from pathlib import Path
import hashlib
from backend.reward_designer import RewardDesigner
from backend.config import Config

logger = logging.getLogger(__name__)

class QLearningRouting:
    """Q-Learning based adaptive routing with transfer learning"""

    def __init__(self, network_graph, config: Config = None, load_pretrained: bool = True):
        """
        Initialize Q-Learning routing

        Args:
            network_graph: NetworkGraph instance
            config: Configuration object
            load_pretrained: Whether to load pre-trained global model
        """
        self.network = network_graph
        self.config = config or Config()
        self.reward_designer = RewardDesigner(network_graph, config)

        # Q-table: Q[state][action] = value
        self.q_table = {}

        # Paths for models
        self.pretrained_path = Path("models/pretrained/q_learning_global.pkl")
        self.pretrained_path.parent.mkdir(parents=True, exist_ok=True)

        # Network-specific model path
        self.network_id = self._generate_network_id()
        self.network_specific_path = Path(f"models/network_specific/network_{self.network_id}_ql.pkl")
        self.network_specific_path.parent.mkdir(parents=True, exist_ok=True)

        # CRITICAL: Initialize to 0, will be loaded from model if exists
        self.total_training_episodes = 0  # ← Start at 0 for this instance

        # Load models (priority: network-specific > global > fresh)
        if self._load_network_specific_model():
            logger.info("✓ Loaded network-specific Q-Learning model")
        elif load_pretrained and self.pretrained_path.exists():
            self._load_pretrained_model()
        else:
            self._initialize_q_table()

        # Training metrics
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'path_lengths': [],
            'convergence': []
        }

        self.epsilon = self.config.QL_EPSILON

    def _generate_network_id(self) -> str:
        """Generate unique ID for network based on topology and size"""
        network_str = f"{self.network.topology}_{self.network.num_nodes}_{self.network.graph.number_of_edges()}"
        return hashlib.md5(network_str.encode()).hexdigest()[:8]

    def _initialize_q_table(self):
        """Initialize Q-table with zeros"""
        for node in self.network.graph.nodes():
            self.q_table[node] = {}
            for neighbor in self.network.graph.neighbors(node):
                self.q_table[node][neighbor] = 0.0
        logger.info("Q-table initialized from scratch")

    def _load_pretrained_model(self):
        """Load pre-trained global Q-Learning model"""
        try:
            with open(self.pretrained_path, 'rb') as f:
                pretrained_data = pickle.load(f)

            # Transfer Q-values from pre-trained model
            pretrained_q_table = pretrained_data.get('q_table', {})
            # DON'T load total_training_episodes here - keep it at 0 for this instance

            # Initialize current network's Q-table
            self._initialize_q_table()

            # Transfer knowledge: copy Q-values for matching state-action pairs
            transferred = 0
            for state in self.q_table:
                if state in pretrained_q_table:
                    for action in self.q_table[state]:
                        if action in pretrained_q_table[state]:
                            self.q_table[state][action] = pretrained_q_table[state][action]
                            transferred += 1

            logger.info(f"✓ Loaded pre-trained Q-Learning model")
            logger.info(f"✓ Transferred {transferred} Q-values from global model")
            logger.info(f"✓ Global model has {pretrained_data.get('total_episodes', 0):,} cumulative episodes")
            logger.info(f"✓ This instance starts fresh at 0 episodes")

        except Exception as e:
            logger.warning(f"Could not load pre-trained model: {e}. Starting fresh.")
            self._initialize_q_table()

    def _load_network_specific_model(self) -> bool:
        """Load network-specific Q-Learning model if exists"""
        if not self.network_specific_path.exists():
            return False

        try:
            with open(self.network_specific_path, 'rb') as f:
                network_data = pickle.load(f)

            self.q_table = network_data.get('q_table', {})
            self.total_training_episodes = network_data.get('total_episodes', 0)
            self.training_history = network_data.get('training_history', self.training_history)

            logger.info(f"✓ Loaded network-specific model: {self.network_specific_path.name}")
            logger.info(f"✓ Episodes trained: {self.total_training_episodes}")

            return True

        except Exception as e:
            logger.warning(f"Could not load network-specific model: {e}")
            return False

    def save_network_specific_model(self):
        """Save network-specific Q-Learning model"""
        try:
            network_data = {
                'q_table': self.q_table,
                'total_episodes': self.total_training_episodes,
                'training_history': self.training_history,
                'network_info': {
                    'topology': self.network.topology,
                    'num_nodes': self.network.num_nodes,
                    'num_edges': self.network.graph.number_of_edges(),
                    'network_id': self.network_id
                },
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            with open(self.network_specific_path, 'wb') as f:
                pickle.dump(network_data, f)

            logger.info(f"✓ Saved network-specific model: {self.network_specific_path.name}")

        except Exception as e:
            logger.error(f"Error saving network-specific model: {e}")

    def save_pretrained_model(self):
        """Save/update global pre-trained model with proper cumulative episode tracking"""
        try:
            # Load existing pre-trained model if it exists
            global_q_table = {}
            global_total_episodes = 0
            version = 1

            if self.pretrained_path.exists():
                with open(self.pretrained_path, 'rb') as f:
                    pretrained_data = pickle.load(f)
                global_q_table = pretrained_data.get('q_table', {})
                global_total_episodes = pretrained_data.get('total_episodes', 0)
                version = pretrained_data.get('version', 0) + 1

            # Merge current Q-table with global Q-table (keep higher Q-values)
            for state in self.q_table:
                if state not in global_q_table:
                    global_q_table[state] = {}

                for action, q_value in self.q_table[state].items():
                    if action not in global_q_table[state]:
                        global_q_table[state][action] = q_value
                    else:
                        global_q_table[state][action] = max(
                            global_q_table[state][action],
                            q_value
                        )

            # CRITICAL FIX: Add current instance total to global total
            # This ensures cumulative tracking across all sessions

            if hasattr(self, '_episodes_before_training'):
                new_episodes = self.total_training_episodes - self._episodes_before_training
            else:
                # Fallback: use total if no tracking
                new_episodes = self.total_training_episodes

            new_global_total = global_total_episodes + new_episodes

            # Save updated global model
            pretrained_data = {
                'q_table': global_q_table,
                'total_episodes': new_global_total,  # ← Cumulative sum
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
                'version': version
            }

            with open(self.pretrained_path, 'wb') as f:
                pickle.dump(pretrained_data, f)

            logger.info(f"✓ Updated global pre-trained Q-Learning model")
            logger.info(f"✓ Model version: {version}")
            logger.info(f"✓ Previous total: {global_total_episodes:,} episodes")
            logger.info(f"✓ Current session: {new_episodes:,} episodes")
            logger.info(f"✓ New cumulative total: {new_global_total:,} episodes")

            # Update instance variable to reflect new total
            #self.total_training_episodes = new_global_total

        except Exception as e:
            logger.error(f"Error saving pre-trained model: {e}")

    def choose_action(self, state: int, destination: int,
                     training: bool = True) -> int:
        """Choose next node using epsilon-greedy policy"""
        neighbors = self.network.get_neighbors(state)

        if not neighbors:
            return state

        if training and np.random.random() < self.epsilon:
            return np.random.choice(neighbors)
        else:
            if state in self.q_table:
                q_values = {neighbor: self.q_table[state].get(neighbor, 0.0)
                           for neighbor in neighbors}
                return max(q_values, key=q_values.get)
            else:
                return np.random.choice(neighbors)

    def train(self, source: int, destination: int, num_episodes: int = None,
              save_specific: bool = True):
        """
        Train Q-Learning agent

        Args:
            source: Source node
            destination: Destination node
            num_episodes: Number of training episodes
            save_specific: Whether to save network-specific model
        """
        num_episodes = num_episodes or self.config.QL_EPISODES

        # CRITICAL FIX: Track episodes BEFORE training starts
        self._episodes_before_training = self.total_training_episodes

        logger.info(f"Training Q-Learning: {source} -> {destination}, {num_episodes} episodes")
        logger.info(f"Current cumulative episodes: {self.total_training_episodes}")  # ← ADD THIS

        self.reward_designer.precompute_distances(destination)

        for episode in range(num_episodes):
            state = source
            visited = set()
            total_reward = 0
            path_length = 0
            max_steps = self.network.num_nodes * 2

            while state != destination and path_length < max_steps:
                visited.add(state)
                action = self.choose_action(state, destination, training=True)
                reward = self.reward_designer.calculate_reward(
                    state, action, destination, visited, path_length
                )
                next_state = action

                old_q = self.q_table[state].get(action, 0.0)

                if next_state in self.q_table and next_state != destination:
                    next_neighbors = self.network.get_neighbors(next_state)
                    if next_neighbors:
                        max_next_q = max(self.q_table[next_state].get(n, 0.0)
                                         for n in next_neighbors)
                    else:
                        max_next_q = 0.0
                else:
                    max_next_q = 0.0

                new_q = old_q + self.config.QL_LEARNING_RATE * (
                        reward + self.config.QL_DISCOUNT_FACTOR * max_next_q - old_q
                )

                if state not in self.q_table:
                    self.q_table[state] = {}
                self.q_table[state][action] = new_q

                total_reward += reward
                state = next_state
                path_length += 1

            self.epsilon = max(self.config.QL_MIN_EPSILON,
                               self.epsilon * self.config.QL_EPSILON_DECAY)

            self.training_history['episodes'].append(episode)
            self.training_history['rewards'].append(total_reward)
            self.training_history['path_lengths'].append(path_length)

            if episode % 100 == 0:
                avg_reward = np.mean(self.training_history['rewards'][-100:])
                logger.info(f"Episode {episode}/{num_episodes}, "
                            f"Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        # CRITICAL: Accumulate episodes BEFORE saving
        self.total_training_episodes += num_episodes  # ← This should already be here

        logger.info(f"✅ Training completed. New cumulative total: {self.total_training_episodes}")  # ← ADD THIS

        # Save both models
        if save_specific:
            self.save_network_specific_model()
        self.save_pretrained_model()

        logger.info("✅ Q-Learning training completed and models saved!")
        logger.info(f"✓ Network-specific model saved: {save_specific}")

    def find_path(self, source: int, destination: int) -> Tuple[List[int], float, float]:
        """Find path using trained Q-table"""
        start_time = time.time()

        path = [source]
        state = source
        visited = set([source])
        cost = 0.0
        max_steps = self.network.num_nodes * 2

        while state != destination and len(path) < max_steps:
            action = self.choose_action(state, destination, training=False)

            if action in visited:
                logger.warning(f"Loop detected at node {action}")
                break

            path.append(action)
            visited.add(action)
            cost += self.network.get_edge_weight(state, action)
            state = action

        time_taken = time.time() - start_time

        if state != destination:
            logger.warning(f"Failed to reach destination. Path: {path}")
            return [], -1, time_taken

        logger.info(f"Q-Learning path found: {path}, cost={cost:.2f}")
        return path, cost, time_taken

    def get_q_table_stats(self) -> Dict:
        """Get statistics about Q-table"""
        total_entries = sum(len(actions) for actions in self.q_table.values())
        avg_q_value = np.mean([q for actions in self.q_table.values()
                              for q in actions.values()])

        return {
            'total_states': len(self.q_table),
            'total_state_action_pairs': total_entries,
            'average_q_value': avg_q_value,
            'epsilon': self.epsilon,
            'total_training_episodes': self.total_training_episodes,
            'network_id': self.network_id,
            'network_specific_saved': self.network_specific_path.exists(),
            'using_pretrained': self.pretrained_path.exists()
        }
