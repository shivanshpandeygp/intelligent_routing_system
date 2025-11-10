"""
Network Visualization and Animation Module
"""
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class NetworkVisualizer:
    """Visualize network topology and routing paths"""

    def __init__(self, network_graph):
        """
        Initialize visualizer

        Args:
            network_graph: NetworkGraph instance
        """
        self.network = network_graph
        self.colors = {
            'node': '#1f77b4',
            'edge': '#999999',
            'path_edge': '#ff7f0e',
            'source': '#2ca02c',
            'destination': '#d62728',
            'visited': '#9467bd'
        }

    def plot_network(self, title: str = "Network Topology") -> go.Figure:
        """
        Plot complete network topology with edge weights

        Args:
            title: Plot title

        Returns:
            fig: Plotly figure
        """
        # Get positions
        pos = self.network.node_positions

        # Safety check for missing positions
        missing_nodes = [n for n in self.network.graph.nodes() if n not in pos]
        if missing_nodes:
            logger.warning(f"Missing positions for nodes: {missing_nodes}")
            temp_pos = nx.spring_layout(self.network.graph)
            for node in missing_nodes:
                pos[node] = temp_pos[node]

        # Create edge traces with midpoint markers for displaying weights
        edge_x = []
        edge_y = []
        edge_mid_x = []
        edge_mid_y = []
        edge_texts = []

        for u, v, data in self.network.graph.edges(data=True):
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]

                # Add edge line
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

                # Calculate midpoint for label
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                edge_mid_x.append(mid_x)
                edge_mid_y.append(mid_y)

                weight = data.get('weight', 0)
                edge_texts.append(f"Edge {u}→{v}<br>Weight: {weight:.2f}")

        # Edge lines (without hover)
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=2, color=self.colors['edge']),
            hoverinfo='skip',
            mode='lines',
            name='Edges',
            showlegend=True
        )

        # Edge labels (invisible markers at midpoints with hover text)
        edge_label_trace = go.Scatter(
            x=edge_mid_x,
            y=edge_mid_y,
            mode='markers',
            hoverinfo='text',
            hovertext=edge_texts,
            marker=dict(
                size=10,
                color='rgba(255,255,255,0)',  # Invisible
                line=dict(width=0)
            ),
            name='Edge Weights',
            showlegend=False
        )

        # Edge weight text annotations (visible on graph)
        edge_annotations = []
        for i, (mid_x, mid_y) in enumerate(zip(edge_mid_x, edge_mid_y)):
            # Extract weight from hover text
            weight_str = edge_texts[i].split("Weight: ")[1].split("<br>")[0]
            edge_annotations.append(
                dict(
                    x=mid_x,
                    y=mid_y,
                    text=weight_str,
                    showarrow=False,
                    font=dict(size=9, color='#dc143c', family='Arial'),  # Crimson red
                    bgcolor='rgba(255,255,255,0.7)',
                    borderpad=1
                )
            )

        # Create node trace with BLACK text for visibility
        node_x = []
        node_y = []
        node_text = []
        node_hover = []

        for node in self.network.graph.nodes():
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(str(node))

                degree = self.network.graph.out_degree(node)
                node_hover.append(f"Node {node}<br>Degree: {degree}")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_hover,
            marker=dict(
                size=25,
                color=self.colors['node'],
                line=dict(width=2, color='white')
            ),
            text=node_text,
            textposition="middle center",  # Center the text inside node
            textfont=dict(
                size=12,
                color='white',  # White text inside blue nodes
                family='Arial Black'
            ),
            name='Nodes',
            showlegend=True
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, edge_label_trace, node_trace])

        # Add edge weight annotations
        fig.update_layout(
            annotations=edge_annotations,
            title=title,
            titlefont_size=20,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=600
        )

        return fig

    def plot_path(self, path: List[int], title: str = "Routing Path") -> go.Figure:
        """
        Plot network with highlighted path and edge weights

        Args:
            path: List of nodes in path
            title: Plot title

        Returns:
            fig: Plotly figure
        """
        pos = self.network.node_positions

        # Safety check
        missing_nodes = [n for n in self.network.graph.nodes() if n not in pos]
        if missing_nodes:
            temp_pos = nx.spring_layout(self.network.graph)
            for node in missing_nodes:
                pos[node] = temp_pos[node]

        # Regular edges
        regular_edge_x = []
        regular_edge_y = []

        for u, v in self.network.graph.edges():
            if not any((path[i] == u and path[i+1] == v) for i in range(len(path)-1)):
                if u in pos and v in pos:
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    regular_edge_x.extend([x0, x1, None])
                    regular_edge_y.extend([y0, y1, None])

        regular_edges = go.Scatter(
            x=regular_edge_x,
            y=regular_edge_y,
            line=dict(width=1, color=self.colors['edge']),
            mode='lines',
            name='Network Edges',
            hoverinfo='skip',
            showlegend=True
        )

        # Path edges with annotations
        path_edge_x = []
        path_edge_y = []
        path_annotations = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                path_edge_x.extend([x0, x1, None])
                path_edge_y.extend([y0, y1, None])

                # Midpoint annotation
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                weight = self.network.get_edge_weight(u, v)

                path_annotations.append(
                    dict(
                        x=mid_x,
                        y=mid_y,
                        text=f"{weight:.1f}",
                        showarrow=False,
                        font=dict(size=10, color='white', family='Arial Black'),
                        bgcolor='rgba(255,127,14,0.8)',
                        borderpad=2
                    )
                )

        path_edges = go.Scatter(
            x=path_edge_x,
            y=path_edge_y,
            line=dict(width=4, color=self.colors['path_edge']),
            mode='lines',
            name='Route Path',
            hoverinfo='skip',
            showlegend=True
        )

        # Nodes
        regular_node_x = []
        regular_node_y = []
        regular_node_text = []

        path_node_x = []
        path_node_y = []
        path_node_text = []

        for node in self.network.graph.nodes():
            if node in pos:
                x, y = pos[node]
                if node in path and node != path[0] and node != path[-1]:
                    path_node_x.append(x)
                    path_node_y.append(y)
                    path_node_text.append(str(node))
                elif node not in path:
                    regular_node_x.append(x)
                    regular_node_y.append(y)
                    regular_node_text.append(str(node))

        regular_nodes = go.Scatter(
            x=regular_node_x,
            y=regular_node_y,
            mode='markers+text',
            marker=dict(size=20, color=self.colors['node'], line=dict(width=2, color='white')),
            text=regular_node_text,
            textposition="middle center",
            textfont=dict(size=10, color='white', family='Arial Black'),
            name='Nodes',
            hoverinfo='text',
            hovertext=[f"Node {t}" for t in regular_node_text]
        )

        path_nodes = go.Scatter(
            x=path_node_x,
            y=path_node_y,
            mode='markers+text',
            marker=dict(size=25, color=self.colors['visited'], line=dict(width=2, color='white')),
            text=path_node_text,
            textposition="middle center",
            textfont=dict(size=11, color='white', family='Arial Black'),
            name='Path Nodes',
            hoverinfo='text',
            hovertext=[f"Node {t} (in path)" for t in path_node_text]
        )

        # Source and destination
        source_node = go.Scatter(
            x=[pos[path[0]][0]],
            y=[pos[path[0]][1]],
            mode='markers+text',
            marker=dict(size=30, color=self.colors['source'], symbol='star',
                       line=dict(width=2, color='white')),
            text=[str(path[0])],
            textposition="middle center",
            textfont=dict(size=13, color='white', family='Arial Black'),
            name='Source',
            hoverinfo='text',
            hovertext=f"Source: Node {path[0]}"
        )

        dest_node = go.Scatter(
            x=[pos[path[-1]][0]],
            y=[pos[path[-1]][1]],
            mode='markers+text',
            marker=dict(size=30, color=self.colors['destination'], symbol='star',
                       line=dict(width=2, color='white')),
            text=[str(path[-1])],
            textposition="middle center",
            textfont=dict(size=13, color='white', family='Arial Black'),
            name='Destination',
            hoverinfo='text',
            hovertext=f"Destination: Node {path[-1]}"
        )

        # Create figure
        fig = go.Figure(data=[regular_edges, path_edges, regular_nodes,
                            path_nodes, source_node, dest_node])

        fig.update_layout(
            annotations=path_annotations,
            title=f"{title}<br>Path: {' → '.join(map(str, path))}",
            titlefont_size=18,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=600
        )

        return fig

    def create_animation(self, path: List[int], title: str = "Packet Routing Animation"):
        """
        Create animated visualization of packet routing

        Args:
            path: List of nodes in path
            title: Animation title

        Returns:
            fig: Animated plotly figure
        """
        pos = self.network.node_positions

        # Create frames for animation
        frames = []

        for step in range(len(path)):
            current_path = path[:step+1]

            # Network edges
            edge_x, edge_y = [], []
            for u, v in self.network.graph.edges():
                if u in pos and v in pos:
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

            # Path edges so far
            path_x, path_y = [], []
            for i in range(len(current_path) - 1):
                u, v = current_path[i], current_path[i+1]
                if u in pos and v in pos:
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    path_x.extend([x0, x1, None])
                    path_y.extend([y0, y1, None])

            # Nodes
            node_x = [pos[n][0] for n in self.network.graph.nodes() if n in pos]
            node_y = [pos[n][1] for n in self.network.graph.nodes() if n in pos]
            node_text = [str(n) for n in self.network.graph.nodes() if n in pos]

            # Packet position
            packet_x = pos[current_path[-1]][0]
            packet_y = pos[current_path[-1]][1]

            frame = go.Frame(
                data=[
                    go.Scatter(x=edge_x, y=edge_y, mode='lines',
                             line=dict(width=1, color='#ccc'),
                             hoverinfo='skip'),
                    go.Scatter(x=path_x, y=path_y, mode='lines',
                             line=dict(width=3, color='orange'),
                             hoverinfo='skip'),
                    go.Scatter(x=node_x, y=node_y, mode='markers+text',
                             marker=dict(size=20, color='lightblue'),
                             text=node_text,
                             textposition='middle center',
                             textfont=dict(size=10, color='#333'),
                             hoverinfo='text'),
                    go.Scatter(x=[packet_x], y=[packet_y], mode='markers',
                             marker=dict(size=25, color='red', symbol='circle'),
                             name='Packet', hoverinfo='text',
                             hovertext=f"Packet at Node {current_path[-1]}")
                ],
                name=str(step)
            )
            frames.append(frame)

        # Initial figure
        fig = go.Figure(data=frames[0].data, frames=frames)

        fig.update_layout(
            title=title,
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': 'Play', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 500, 'redraw': True},
                                    'fromcurrent': True}]},
                    {'label': 'Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                      'mode': 'immediate'}]}
                ]
            }],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            plot_bgcolor='white'
        )

        return fig
