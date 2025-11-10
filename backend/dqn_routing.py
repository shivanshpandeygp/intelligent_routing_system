"""
Deep Q-Network (DQN) Based Routing Algorithm
STABLE VERSION - Fixes exploding Q-values and degrading performance

Critical Stability Improvements:
1. ✅ Q-value clamping to prevent explosion
2. ✅ Reward normalization
3. ✅ Slower target network updates
4. ✅ Lower learning rate
5. ✅ Huber loss with proper delta
6. ✅ Better exploration schedule
7. ✅ Proper initialization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import logging
from typing import List, Tuple, Dict, Set
import time
import hashlib
from pathlib import Path
from backend.reward_designer import RewardDesigner
from backend.config import Config

logger = logging.getLogger(__name__)

Experience = namedtuple('Experience',
                       ['state', 'action', 'reward', 'next_state', 'done', 'valid_actions'])


class DQNetwork(nn.Module):
    """Deep Q-Network with Xavier initialization for stability"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)

        # Xavier initialization for better convergence
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)  # No activation on output


class ReplayMemory:
    """Experience replay memory"""

    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, experience: Experience):
        self.memory.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[i] for i in indices]

    def __len__(self):
        return len(self.memory)

    def is_ready(self, min_size: int) -> bool:
        return len(self.memory) >= min_size


class DQNRouting:
    """Deep Q-Network based routing with stability improvements"""

    def __init__(self, network_graph, config: Config = None, load_pretrained: bool = True):
        self.network = network_graph
        self.config = config or Config()
        self.reward_designer = RewardDesigner(network_graph, self.config)
        self.device = self.config.DEVICE

        # State and action space
        self.state_size = self.network.num_nodes * 3 + 1
        self.action_size = self.network.num_nodes

        # CRITICAL: Q-value bounds to prevent explosion
        self.q_value_min = -100.0
        self.q_value_max = 100.0

        # Initialize networks
        self.policy_net = DQNetwork(
            self.state_size,
            self.action_size,
            self.config.DQN_HIDDEN_SIZE
        ).to(self.device)

        self.target_net = DQNetwork(
            self.state_size,
            self.action_size,
            self.config.DQN_HIDDEN_SIZE
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # CRITICAL: Lower learning rate for stability
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=0.00001,  # Much lower!
            weight_decay=1e-5
        )

        # Model paths
        self.pretrained_path = Path("models/pretrained/dqn_global.pt")
        self.pretrained_path.parent.mkdir(parents=True, exist_ok=True)

        self.network_id = self._generate_network_id()
        self.network_specific_path = Path(f"models/network_specific/network_{self.network_id}_dqn.pt")
        self.network_specific_path.parent.mkdir(parents=True, exist_ok=True)

        # Training counters
        self.total_training_episodes = 0
        self.steps_done = 0

        # Initialize training history BEFORE loading
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'path_lengths': [],
            'success_rate': []
        }

        # Load models
        if self._load_network_specific_model():
            logger.info("✓ Loaded network-specific DQN model")
        elif load_pretrained and self.pretrained_path.exists():
            self._load_pretrained_model()
        else:
            logger.info("✓ Initialized fresh DQN model")

        # Huber loss with smaller delta for stability
        self.criterion = nn.SmoothL1Loss(beta=1.0)

        # Experience replay
        self.memory = ReplayMemory(self.config.DQN_MEMORY_SIZE)

        # Epsilon
        self.epsilon = self.config.DQN_EPSILON_START

        logger.info(f"✓ DQN initialized on {self.device}")
        logger.info(f"✓ State size: {self.state_size}, Action size: {self.action_size}")
        logger.info(f"✓ Hidden layer size: {self.config.DQN_HIDDEN_SIZE}")
        logger.info(f"✓ Learning rate: 0.00001 (stabilized)")

    def _generate_network_id(self) -> str:
        network_str = f"{self.network.topology}_{self.network.num_nodes}_{self.network.graph.number_of_edges()}"
        return hashlib.md5(network_str.encode()).hexdigest()[:8]

    def _load_pretrained_model(self):
        try:
            checkpoint = torch.load(self.pretrained_path, map_location=self.device)

            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])

            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            global_total = checkpoint.get('total_episodes', 0)
            logger.info(f"✓ Loaded pre-trained DQN model")
            logger.info(f"✓ Global model: {global_total:,} cumulative episodes")

        except Exception as e:
            logger.warning(f"Could not load pre-trained model: {e}. Starting fresh.")
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _load_network_specific_model(self) -> bool:
        if not self.network_specific_path.exists():
            return False

        try:
            checkpoint = torch.load(self.network_specific_path, map_location=self.device)

            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.total_training_episodes = checkpoint.get('total_episodes', 0)
            self.training_history = checkpoint.get('training_history', self.training_history)

            logger.info(f"✓ Loaded network-specific DQN: {self.network_specific_path.name}")
            logger.info(f"✓ Episodes trained: {self.total_training_episodes:,}")

            return True

        except Exception as e:
            logger.warning(f"Could not load network-specific model: {e}")
            return False

    def save_network_specific_model(self):
        try:
            checkpoint = {
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
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

            torch.save(checkpoint, self.network_specific_path)
            logger.info(f"✓ Saved network-specific DQN model")

        except Exception as e:
            logger.error(f"Error saving network-specific model: {e}")

    def save_pretrained_model(self):
        try:
            version = 1
            global_total_episodes = 0

            if self.pretrained_path.exists():
                checkpoint = torch.load(self.pretrained_path, map_location=self.device)
                version = checkpoint.get('version', 0) + 1
                global_total_episodes = checkpoint.get('total_episodes', 0)

            if hasattr(self, '_episodes_before_training'):
                new_episodes = self.total_training_episodes - self._episodes_before_training
            else:
                new_episodes = self.total_training_episodes

            new_global_total = global_total_episodes + new_episodes

            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'total_episodes': new_global_total,
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
                'version': version
            }, self.pretrained_path)

            logger.info(f"✓ Updated global model (v{version})")
            logger.info(f"✓ Previous global total: {global_total_episodes:,} episodes")
            logger.info(f"✓ Episodes added this session: {new_episodes:,}")
            logger.info(f"✓ New cumulative total: {new_global_total:,} episodes")

            #logger.info(f"✓ Updated global model (v{version}): {new_global_total:,} episodes")

        except Exception as e:
            logger.error(f"Error saving pre-trained model: {e}")

    def _encode_state(self, current_node: int, destination: int,
                     visited: Set[int], path_length: int) -> torch.Tensor:
        state = np.zeros(self.state_size)

        if current_node < self.network.num_nodes:
            state[current_node] = 1.0

        if destination < self.network.num_nodes:
            state[self.network.num_nodes + destination] = 1.0

        for node in visited:
            if node < self.network.num_nodes:
                state[self.network.num_nodes * 2 + node] = 1.0

        state[-1] = min(path_length / self.network.num_nodes, 1.0)

        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def _get_valid_actions(self, current_node: int) -> List[int]:
        neighbors = self.network.get_neighbors(current_node)
        return list(neighbors) if neighbors else []

    def choose_action(self, state_tensor: torch.Tensor, current_node: int,
                     training: bool = True) -> int:
        valid_actions = self._get_valid_actions(current_node)

        if not valid_actions:
            return current_node

        if training and np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]

            # CRITICAL: Clamp Q-values to prevent explosion
            q_values = np.clip(q_values, self.q_value_min, self.q_value_max)

            valid_q_values = {action: q_values[action] for action in valid_actions}
            return max(valid_q_values.items(), key=lambda x: x[1])[0]

    def train(self, source: int, destination: int, num_episodes: int = None,
             save_specific: bool = True):
        num_episodes = num_episodes or self.config.DQN_EPISODES

        # CRITICAL FIX: Track episodes BEFORE training starts
        self._episodes_before_training = self.total_training_episodes

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting STABLE DQN Training: {source} → {destination}")
        logger.info(f"Episodes: {num_episodes:,}")
        logger.info(f"Current cumulative: {self.total_training_episodes:,}")
        logger.info(f"{'='*60}\n")

        self.reward_designer.precompute_distances(destination)

        warmup_episodes = min(100, num_episodes // 10)
        logger.info(f"Warm-up phase: {warmup_episodes} episodes")

        success_count = 0

        for episode in range(num_episodes):
            state_node = source
            visited = set([source])
            path_length = 0
            total_reward = 0.0
            episode_loss = 0.0
            loss_count = 0
            max_steps = self.network.num_nodes * 3

            state = self._encode_state(state_node, destination, visited, path_length)

            while state_node != destination and path_length < max_steps:
                valid_actions = self._get_valid_actions(state_node)

                if not valid_actions:
                    break

                action = self.choose_action(state, state_node, training=True)

                reward = self.reward_designer.calculate_reward(
                    state_node, action, destination, visited, path_length
                )

                # CRITICAL: Normalize reward
                reward = np.clip(reward, -10.0, 100.0) / 10.0  # Scale to [-1, 10]

                next_state_node = action
                visited.add(next_state_node)
                path_length += 1
                done = (next_state_node == destination)

                next_state = self._encode_state(
                    next_state_node, destination, visited, path_length
                )

                next_valid_actions = self._get_valid_actions(next_state_node)

                self.memory.push(Experience(
                    state, action, reward, next_state, done, next_valid_actions
                ))

                total_reward += reward
                state = next_state
                state_node = next_state_node

                if episode >= warmup_episodes and self.memory.is_ready(self.config.DQN_BATCH_SIZE):
                    loss = self._train_step()
                    if not np.isnan(loss) and not np.isinf(loss):
                        episode_loss += loss
                        loss_count += 1
                    self.steps_done += 1

            if state_node == destination:
                success_count += 1

            # CRITICAL: Slower target network update (every 50 episodes instead of 10)
            if episode % 50 == 0 and episode > 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                logger.info(f"✓ Target network updated at episode {episode}")

            # Slower epsilon decay
            self.epsilon = max(
                self.config.DQN_EPSILON_END,
                self.epsilon * 0.999  # Slower decay
            )

            self.training_history['episodes'].append(episode)
            self.training_history['rewards'].append(total_reward)
            avg_loss = episode_loss / max(loss_count, 1)
            self.training_history['losses'].append(avg_loss)
            self.training_history['path_lengths'].append(path_length)

            if episode > 0 and episode % 50 == 0:
                recent_success_rate = success_count / 50
                self.training_history['success_rate'].append(recent_success_rate)
                success_count = 0

            if episode % 50 == 0 or episode == num_episodes - 1:
                recent_rewards = [r for r in self.training_history['rewards'][-50:] if r is not None]
                recent_losses = [l for l in self.training_history['losses'][-50:] if l > 0]
                recent_paths = [p for p in self.training_history['path_lengths'][-50:] if p > 0]

                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                avg_loss = np.mean(recent_losses) if recent_losses else 0
                avg_path = np.mean(recent_paths) if recent_paths else 0
                recent_sr = self.training_history['success_rate'][-1] if self.training_history['success_rate'] else 0.0

                logger.info(
                    f"Episode {episode:4d}/{num_episodes} | "
                    f"Reward: {avg_reward:7.2f} | "
                    f"Loss: {avg_loss:8.4f} | "
                    f"Path: {avg_path:4.1f} | "
                    f"Success: {recent_sr:5.1%} | "
                    f"ε: {self.epsilon:.3f}"
                )

        self.total_training_episodes += num_episodes

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Training completed!")
        logger.info(f"{'='*60}\n")

        if save_specific:
            self.save_network_specific_model()
        self.save_pretrained_model()

    def _train_step(self) -> float:
        """Training step with Q-value clamping"""
        batch = self.memory.sample(self.config.DQN_BATCH_SIZE)

        states = torch.cat([exp.state for exp in batch])
        actions = torch.LongTensor([exp.action for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
        next_states = torch.cat([exp.next_state for exp in batch])
        dones = torch.FloatTensor([exp.done for exp in batch]).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # CRITICAL: Clamp current Q-values
        current_q_values = torch.clamp(current_q_values, self.q_value_min, self.q_value_max)

        with torch.no_grad():
            next_q_values = torch.zeros(self.config.DQN_BATCH_SIZE, device=self.device)

            for i, exp in enumerate(batch):
                if exp.done:
                    next_q_values[i] = 0.0
                else:
                    valid_next_actions = exp.valid_actions

                    if valid_next_actions:
                        policy_q = self.policy_net(exp.next_state).squeeze()
                        target_q = self.target_net(exp.next_state).squeeze()

                        # CRITICAL: Clamp before selecting max
                        policy_q = torch.clamp(policy_q, self.q_value_min, self.q_value_max)
                        target_q = torch.clamp(target_q, self.q_value_min, self.q_value_max)

                        valid_mask = torch.full((self.action_size,), float('-inf'), device=self.device)
                        valid_mask[valid_next_actions] = policy_q[valid_next_actions]

                        best_action = torch.argmax(valid_mask).item()
                        next_q_values[i] = float(target_q[best_action].item())
                    else:
                        next_q_values[i] = 0.0

            # CRITICAL: Clamp target Q-values
            target_q_values = rewards + (1 - dones) * 0.95 * next_q_values  # Lower gamma
            target_q_values = torch.clamp(target_q_values, self.q_value_min, self.q_value_max)

        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        # CRITICAL: Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)

        self.optimizer.step()

        return loss.item()

    def find_path(self, source: int, destination: int) -> Tuple[List[int], float, float]:
        start_time = time.time()

        self.policy_net.eval()

        path = [source]
        state_node = source
        visited = set([source])
        cost = 0.0
        path_length = 0
        max_steps = self.network.num_nodes * 2  # Shorter max steps

        with torch.no_grad():
            while state_node != destination and path_length < max_steps:
                state = self._encode_state(state_node, destination, visited, path_length)

                valid_actions = self._get_valid_actions(state_node)
                unvisited_actions = [a for a in valid_actions if a not in visited]

                if not unvisited_actions:
                    break  # Prevent loops

                q_values = self.policy_net(state).cpu().numpy()[0]
                q_values = np.clip(q_values, self.q_value_min, self.q_value_max)

                unvisited_q = {a: q_values[a] for a in unvisited_actions}
                action = max(unvisited_q.items(), key=lambda x: x[1])[0]

                path.append(action)
                visited.add(action)

                edge_weight = self.network.get_edge_weight(state_node, action)
                if edge_weight > 0:
                    cost += edge_weight
                else:
                    break

                state_node = action
                path_length += 1

        time_taken = time.time() - start_time

        if state_node != destination:
            logger.warning(f"DQN failed: {source} → {destination}")
            return [], -1.0, time_taken

        logger.info(f"✓ DQN path: {path}, cost={cost:.2f}, time={time_taken:.4f}s")
        return path, cost, time_taken
