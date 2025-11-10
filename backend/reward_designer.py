"""
Reward Function Designer for Reinforcement Learning
"""
import numpy as np
import networkx as nx  # ← MISSING IMPORT - ADD THIS LINE
from typing import Dict
import logging
from backend.config import Config

logger = logging.getLogger(__name__)

class RewardDesigner:
    """Designs rewards for RL-based routing"""

    def __init__(self, network_graph, config: Config = None):
        """
        Initialize reward designer

        Args:
            network_graph: NetworkGraph instance
            config: Configuration object
        """
        self.network = network_graph
        self.config = config or Config()

    def calculate_reward(self, current_node: int, next_node: int,
                        destination: int, visited_nodes: set,
                        path_length: int) -> float:
        """
        Calculate reward for taking action (moving to next_node)

        Reward components:
        1. Destination reached: +100
        2. Hop penalty: -1 per hop
        3. Congestion penalty: -5 if edge is congested
        4. Energy penalty: -2 * energy_cost
        5. Loop penalty: -50 if revisiting node
        6. Invalid action: -100 if edge doesn't exist

        Args:
            current_node: Current node
            next_node: Next node to visit
            destination: Destination node
            visited_nodes: Set of already visited nodes
            path_length: Current path length

        Returns:
            reward: Calculated reward value
        """
        reward = 0.0

        # Check if next_node is a valid neighbor
        if next_node not in self.network.get_neighbors(current_node):
            return self.config.REWARD_INVALID_ACTION

        # Destination reached
        if next_node == destination:
            reward += self.config.REWARD_DESTINATION
            logger.debug(f"Destination reached! Reward: {reward}")
            return reward

        # Hop penalty (encourages shorter paths)
        reward += self.config.REWARD_HOP_PENALTY

        # Loop detection penalty
        if next_node in visited_nodes:
            reward += self.config.REWARD_LOOP_PENALTY
            logger.debug(f"Loop detected at node {next_node}")

        # Congestion penalty
        edge_metrics = self.network.edge_metrics.get((current_node, next_node), {})
        if edge_metrics.get('congestion', 0) > 0.5:
            reward += self.config.REWARD_CONGESTION_PENALTY
            logger.debug(f"Congestion on edge ({current_node}, {next_node})")

        # Energy penalty (based on edge weight)
        energy_cost = edge_metrics.get('energy', 0)
        reward += self.config.REWARD_ENERGY_PENALTY * (energy_cost / 10.0)

        # Bonus for moving closer to destination (heuristic)
        # This helps guide exploration towards the destination
        if hasattr(self, '_distance_to_dest'):
            current_dist = self._distance_to_dest.get(current_node, float('inf'))
            next_dist = self._distance_to_dest.get(next_node, float('inf'))

            if next_dist < current_dist:
                reward += 2.0  # Bonus for getting closer

        return reward

    def precompute_distances(self, destination: int):
        """Precompute distances from all nodes to destination (for heuristic)"""
        self._distance_to_dest = {}

        for node in self.network.graph.nodes():
            try:
                # Use Dijkstra to get distance
                path_length = nx.dijkstra_path_length(
                    self.network.graph.to_undirected(),
                    node,
                    destination
                )
                self._distance_to_dest[node] = path_length
            except nx.NetworkXNoPath:
                self._distance_to_dest[node] = float('inf')
            except Exception as e:
                logger.warning(f"Error computing distance from {node} to {destination}: {e}")
                self._distance_to_dest[node] = float('inf')

    def get_reward_summary(self) -> Dict[str, float]:
        """Get summary of reward parameters"""
        return {
            'destination_reward': self.config.REWARD_DESTINATION,
            'hop_penalty': self.config.REWARD_HOP_PENALTY,
            'congestion_penalty': self.config.REWARD_CONGESTION_PENALTY,
            'energy_penalty': self.config.REWARD_ENERGY_PENALTY,
            'loop_penalty': self.config.REWARD_LOOP_PENALTY,
            'invalid_action_penalty': self.config.REWARD_INVALID_ACTION
        }
