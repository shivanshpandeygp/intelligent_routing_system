"""
Network Graph Manager - Handles graph creation and dynamic topology changes
"""
import networkx as nx
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkGraph:
    """Manages network topology as weighted directed graph"""

    def __init__(self, num_nodes: int = 10, topology: str = 'random',
                 sparsity: float = 0.3):
        """
        Initialize network graph

        Args:
            num_nodes: Number of nodes in the network
            topology: Type of topology (mesh, ring, tree, random)
            sparsity: Connection density (0-1)
        """
        self.num_nodes = num_nodes
        self.topology = topology
        self.sparsity = sparsity
        self.graph = nx.DiGraph()
        self.node_positions = {}
        self.edge_metrics = {}  # Store latency, bandwidth, energy

        self._create_topology()
        self._initialize_edge_metrics()

        logger.info(f"Created {topology} topology with {num_nodes} nodes, "
                    f"{self.graph.number_of_edges()} edges")

    def _create_topology(self):
        """Create network topology based on specified type"""
        nodes = list(range(self.num_nodes))
        self.graph.add_nodes_from(nodes)

        if self.topology == 'mesh':
            self._create_mesh()
        elif self.topology == 'ring':
            self._create_ring()
        elif self.topology == 'tree':
            self._create_tree()
        elif self.topology == 'random':
            self._create_random()
        else:
            raise ValueError(f"Unknown topology: {self.topology}")

    def _create_mesh(self):
        """Create mesh topology (grid-like structure)"""
        grid_size = int(np.ceil(np.sqrt(self.num_nodes)))  # Use ceiling to fit all nodes

        logger.info(f"Creating mesh: {self.num_nodes} nodes in {grid_size}x{grid_size} grid")

        # Create grid connections
        for node in range(self.num_nodes):
            i = node // grid_size
            j = node % grid_size

            # Connect to right neighbor
            if j < grid_size - 1:
                right = node + 1
                if right < self.num_nodes:
                    weight = np.random.randint(1, 20)
                    self.graph.add_edge(node, right, weight=weight)
                    self.graph.add_edge(right, node, weight=weight)

            # Connect to bottom neighbor
            bottom = node + grid_size
            if bottom < self.num_nodes:
                weight = np.random.randint(1, 20)
                self.graph.add_edge(node, bottom, weight=weight)
                self.graph.add_edge(bottom, node, weight=weight)

        # Store grid positions for ALL nodes (CRITICAL FIX)
        for node in range(self.num_nodes):
            i = node // grid_size
            j = node % grid_size
            self.node_positions[node] = (j, grid_size - i - 1)

        logger.info(f"Mesh created with {len(self.node_positions)} node positions")

    def _create_ring(self):
        """Create ring topology"""
        for i in range(self.num_nodes):
            next_node = (i + 1) % self.num_nodes
            weight = np.random.randint(1, 20)
            self.graph.add_edge(i, next_node, weight=weight)
            self.graph.add_edge(next_node, i, weight=weight)

        # Circular layout positions
        angles = np.linspace(0, 2 * np.pi, self.num_nodes, endpoint=False)
        for i, angle in enumerate(angles):
            self.node_positions[i] = (np.cos(angle), np.sin(angle))

    def _create_tree(self):
        """Create tree topology (binary tree)"""
        for i in range(self.num_nodes):
            left_child = 2 * i + 1
            right_child = 2 * i + 2

            if left_child < self.num_nodes:
                weight = np.random.randint(1, 20)
                self.graph.add_edge(i, left_child, weight=weight)
                self.graph.add_edge(left_child, i, weight=weight)

            if right_child < self.num_nodes:
                weight = np.random.randint(1, 20)
                self.graph.add_edge(i, right_child, weight=weight)
                self.graph.add_edge(right_child, i, weight=weight)

        # Hierarchical positions
        self.node_positions = nx.spring_layout(self.graph.to_undirected())

    def _create_random(self):
        """Create random sparse connected graph"""
        # Ensure connectivity by creating spanning tree first
        spanning_tree = nx.random_tree(self.num_nodes, seed=42)

        for u, v in spanning_tree.edges():
            weight = np.random.randint(1, 20)
            self.graph.add_edge(u, v, weight=weight)
            self.graph.add_edge(v, u, weight=weight)

        # Add random edges based on sparsity
        max_edges = self.num_nodes * (self.num_nodes - 1)
        target_edges = int(max_edges * self.sparsity)
        current_edges = self.graph.number_of_edges()

        while current_edges < target_edges:
            u = np.random.randint(0, self.num_nodes)
            v = np.random.randint(0, self.num_nodes)

            if u != v and not self.graph.has_edge(u, v):
                weight = np.random.randint(1, 20)
                self.graph.add_edge(u, v, weight=weight)
                current_edges += 1

        self.node_positions = nx.spring_layout(self.graph.to_undirected())

    @classmethod
    def create_custom(cls, num_nodes: int, edges: list):
        """
        Create custom network from user-defined edges

        Args:
            num_nodes: Number of nodes in the network
            edges: List of tuples (source, destination, weight)

        Returns:
            NetworkGraph instance
        """
        import networkx as nx
        import numpy as np

        # Create instance
        instance = cls.__new__(cls)
        instance.num_nodes = num_nodes
        instance.topology = 'custom'
        instance.sparsity = len(edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        instance.graph = nx.DiGraph()
        instance.node_positions = {}
        instance.edge_metrics = {}

        # Add nodes
        instance.graph.add_nodes_from(range(num_nodes))

        # Add edges
        for source, dest, weight in edges:
            instance.graph.add_edge(source, dest, weight=weight)

        # Initialize edge metrics
        for source, dest, data in instance.graph.edges(data=True):
            weight = data['weight']
            instance.edge_metrics[(source, dest)] = {
                'latency': weight,
                'bandwidth': np.random.randint(10, 100),
                'energy': weight * 0.1,
                'congestion': 0.0,
                'active': True
            }

        # Generate layout positions
        instance.node_positions = nx.spring_layout(instance.graph.to_undirected())

        logger.info(f"Created custom topology with {num_nodes} nodes, {len(edges)} edges")

        return instance

    def _initialize_edge_metrics(self):
        """Initialize edge metrics (latency, bandwidth, energy)"""
        for u, v, data in self.graph.edges(data=True):
            weight = data['weight']
            self.edge_metrics[(u, v)] = {
                'latency': weight,
                'bandwidth': np.random.randint(10, 100),  # Mbps
                'energy': weight * 0.1,  # Proportional to distance
                'congestion': 0.0,  # Initially no congestion
                'active': True  # Link is active
            }

    def simulate_link_failure(self, failure_prob: float = 0.1):
        """Simulate random link failures"""
        failed_links = []
        for u, v in list(self.edge_metrics.keys()):
            if np.random.random() < failure_prob:
                self.edge_metrics[(u, v)]['active'] = False
                failed_links.append((u, v))
                logger.info(f"Link failure: {u} -> {v}")
        return failed_links

    def simulate_congestion(self, edge: Tuple[int, int], increase: float = 1.5):
        """Simulate congestion on specific edge"""
        if edge in self.edge_metrics:
            self.edge_metrics[edge]['congestion'] += 1.0
            self.edge_metrics[edge]['latency'] *= increase
            logger.info(f"Congestion on edge {edge}: "
                        f"latency={self.edge_metrics[edge]['latency']:.2f}")

    def restore_link(self, edge: Tuple[int, int]):
        """Restore a failed link"""
        if edge in self.edge_metrics:
            self.edge_metrics[edge]['active'] = True
            logger.info(f"Link restored: {edge}")

    def get_neighbors(self, node: int) -> List[int]:
        """Get active neighbors of a node"""
        neighbors = []
        for neighbor in self.graph.neighbors(node):
            if self.edge_metrics.get((node, neighbor), {}).get('active', False):
                neighbors.append(neighbor)
        return neighbors

    def get_edge_weight(self, u: int, v: int) -> float:
        """Get current weight of edge considering congestion"""
        if (u, v) not in self.edge_metrics:
            return float('inf')

        metrics = self.edge_metrics[(u, v)]
        if not metrics['active']:
            return float('inf')

        return metrics['latency']

    def save_graph(self, filepath: str):
        """Save graph to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'positions': self.node_positions,
                'metrics': self.edge_metrics,
                'topology': self.topology
            }, f)
        logger.info(f"Graph saved to {filepath}")

    @classmethod
    def load_graph(cls, filepath: str):
        """Load graph from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        instance = cls.__new__(cls)
        instance.graph = data['graph']
        instance.node_positions = data['positions']
        instance.edge_metrics = data['metrics']
        instance.topology = data['topology']
        instance.num_nodes = instance.graph.number_of_nodes()

        logger.info(f"Graph loaded from {filepath}")
        return instance

    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix representation"""
        adj_matrix = np.full((self.num_nodes, self.num_nodes), np.inf)

        for u in range(self.num_nodes):
            adj_matrix[u][u] = 0
            for v in self.get_neighbors(u):
                adj_matrix[u][v] = self.get_edge_weight(u, v)

        return adj_matrix

    def is_connected(self) -> bool:
        """Check if graph is connected (considering active edges only)"""
        # Create subgraph with only active edges
        active_graph = nx.DiGraph()
        active_graph.add_nodes_from(self.graph.nodes())

        for u, v in self.graph.edges():
            if self.edge_metrics.get((u, v), {}).get('active', False):
                active_graph.add_edge(u, v)

        return nx.is_weakly_connected(active_graph)
