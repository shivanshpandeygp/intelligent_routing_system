"""
Performance Comparison Dashboard
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class PerformanceDashboard:
    """Interactive performance comparison dashboard"""

    def __init__(self, comparison: Dict, network_graph):
        """
        Initialize dashboard

        Args:
            comparison: Comparison dictionary from PerformanceMetrics
            network_graph: NetworkGraph instance
        """
        self.comparison = comparison
        self.network = network_graph

    def render(self):
        """Render complete dashboard"""

        st.markdown("### 📊 Algorithm Comparison Overview")

        # Metrics summary cards
        self._render_metrics_cards()

        st.markdown("---")

        # Comparison charts
        col1, col2 = st.columns(2)

        with col1:
            self._render_cost_comparison()
            self._render_latency_comparison()

        with col2:
            self._render_time_comparison()
            self._render_energy_comparison()

        st.markdown("---")

        # Detailed comparison table
        self._render_comparison_table()

        st.markdown("---")

        # Rankings
        self._render_rankings()

        st.markdown("---")

        # Radar chart
        self._render_radar_chart()

    def _render_metrics_cards(self):
        """Render metric summary cards with proper failure indicators"""

        algorithms = self.comparison['algorithms']
        cols = st.columns(len(algorithms))

        for idx, algo_name in enumerate(algorithms):
            metrics = self.comparison['metrics'][algo_name]

            with cols[idx]:
                st.markdown(f"#### {algo_name}")

                # Color coding based on success
                if metrics['success']:
                    st.success("✅ Route Found")
                    st.metric("Path Cost", f"{metrics['path_cost']:.2f}")
                    st.metric("Latency", f"{metrics['latency']:.2f} ms")
                    st.metric("Hops", metrics['hop_count'])
                else:
                    st.error("❌ Failed")
                    st.metric("Path Cost", "N/A")
                    st.metric("Latency", "N/A")
                    st.metric("Hops", "0")
                    st.warning("Algorithm could not find a valid path")

    def _render_cost_comparison(self):
        """Render path cost comparison chart"""

        algorithms = self.comparison['algorithms']
        costs = [self.comparison['metrics'][algo]['path_cost']
                for algo in algorithms]

        # Filter out failed attempts
        valid_algos = [algo for algo, cost in zip(algorithms, costs) if cost >= 0]
        valid_costs = [cost for cost in costs if cost >= 0]

        if not valid_costs:
            st.warning("No valid paths to compare")
            return

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        fig = go.Figure(data=[
            go.Bar(
                x=valid_algos,
                y=valid_costs,
                marker_color=colors[:len(valid_algos)],
                text=[f"{cost:.2f}" for cost in valid_costs],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="Path Cost Comparison",
            xaxis_title="Algorithm",
            yaxis_title="Total Path Cost",
            height=350,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_time_comparison(self):
        """Render computation time comparison"""

        algorithms = self.comparison['algorithms']
        times = [self.comparison['metrics'][algo]['computation_time'] * 1000  # Convert to ms
                for algo in algorithms]

        valid_algos = [algo for algo, t in zip(algorithms, times) if t >= 0]
        valid_times = [t for t in times if t >= 0]

        if not valid_times:
            st.warning("No valid times to compare")
            return

        colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        fig = go.Figure(data=[
            go.Bar(
                x=valid_algos,
                y=valid_times,
                marker_color=colors[:len(valid_algos)],
                text=[f"{t:.2f} ms" for t in valid_times],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="Computation Time Comparison",
            xaxis_title="Algorithm",
            yaxis_title="Time (milliseconds)",
            height=350,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_latency_comparison(self):
        """Render latency comparison"""

        algorithms = self.comparison['algorithms']
        latencies = [self.comparison['metrics'][algo]['latency']
                    for algo in algorithms]

        valid_algos = [algo for algo, lat in zip(algorithms, latencies) if lat >= 0]
        valid_latencies = [lat for lat in latencies if lat >= 0]

        if not valid_latencies:
            return

        fig = go.Figure(data=[
            go.Bar(
                x=valid_algos,
                y=valid_latencies,
                marker_color='lightcoral',
                text=[f"{lat:.2f}" for lat in valid_latencies],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="End-to-End Latency",
            xaxis_title="Algorithm",
            yaxis_title="Latency (ms)",
            height=350,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_energy_comparison(self):
        """Render energy efficiency comparison"""

        algorithms = self.comparison['algorithms']
        energies = [self.comparison['metrics'][algo]['energy']
                   for algo in algorithms]

        valid_algos = [algo for algo, e in zip(algorithms, energies) if e >= 0]
        valid_energies = [e for e in energies if e >= 0]

        if not valid_energies:
            return

        fig = go.Figure(data=[
            go.Bar(
                x=valid_algos,
                y=valid_energies,
                marker_color='lightgreen',
                text=[f"{e:.2f}" for e in valid_energies],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="Energy Consumption",
            xaxis_title="Algorithm",
            yaxis_title="Energy Units",
            height=350,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_comparison_table(self):
        """Render detailed comparison table with proper failure handling"""

        st.markdown("### 📋 Detailed Metrics Table")

        # Prepare table data
        table_data = []
        for algo in self.comparison['algorithms']:
            metrics = self.comparison['metrics'][algo]

            # Check if algorithm succeeded
            success = metrics['success']

            table_data.append({
                'Algorithm': algo,
                'Path Cost': f"{metrics['path_cost']:.2f}" if success else "Failed",
                'Computation Time (ms)': f"{metrics['computation_time'] * 1000:.4f}",
                'Latency (ms)': f"{metrics['latency']:.2f}" if success else "N/A",
                'Energy': f"{metrics['energy']:.2f}" if success else "N/A",
                'Hop Count': metrics['hop_count'] if success else "N/A",
                'Success': '✅' if success else '❌'
            })

        import pandas as pd
        df = pd.DataFrame(table_data)

        # Style the dataframe with PROPER failure handling
        def highlight_best(s):
            """Highlight best values - EXCLUDING FAILED ALGORITHMS"""
            # Skip non-comparable columns
            if s.name in ['Algorithm', 'Success']:
                return [''] * len(s)

            # Get the algorithms that succeeded
            success_mask = df['Success'] == '✅'

            # If no successful algorithms, no highlighting
            if not success_mask.any():
                return [''] * len(s)

            # Handle Hop Count (integer column)
            if s.name == 'Hop Count':
                try:
                    # Get only successful values
                    valid_values = []
                    valid_indices = []

                    for idx, (val, success) in enumerate(zip(s, df['Success'])):
                        if success == '✅' and val != "N/A":
                            try:
                                valid_values.append(int(val))
                                valid_indices.append(idx)
                            except:
                                pass

                    if not valid_values:
                        return [''] * len(s)

                    min_val = min(valid_values)

                    styles = []
                    for idx, val in enumerate(s):
                        if idx in valid_indices and int(val) == min_val:
                            styles.append('background-color: lightgreen')
                        else:
                            styles.append('')

                    return styles
                except:
                    return [''] * len(s)

            # Handle other numeric columns
            try:
                # Convert to numeric, filtering out failed algorithms
                numeric_values = []
                numeric_indices = []

                for idx, (val, success) in enumerate(zip(s, df['Success'])):
                    if success == '✅' and val not in ["N/A", "Failed"]:
                        try:
                            clean_val = float(str(val).replace(',', ''))
                            # Skip negative values (failure indicators)
                            if clean_val >= 0:
                                numeric_values.append(clean_val)
                                numeric_indices.append(idx)
                        except:
                            pass

                # If no valid values, no highlighting
                if not numeric_values:
                    return [''] * len(s)

                # Find minimum value (best)
                min_val = min(numeric_values)

                # Apply highlighting only to successful algorithms with best value
                styles = []
                for idx, val in enumerate(s):
                    if idx in numeric_indices:
                        try:
                            clean_val = float(str(val).replace(',', ''))
                            if abs(clean_val - min_val) < 0.01:  # Epsilon comparison
                                styles.append('background-color: lightgreen')
                            else:
                                styles.append('')
                        except:
                            styles.append('')
                    else:
                        styles.append('')

                return styles

            except Exception as e:
                logger.warning(f"Styling error for column {s.name}: {e}")
                return [''] * len(s)

        try:
            styled_df = df.style.apply(highlight_best, axis=0)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        except Exception as e:
            logger.warning(f"Styling failed: {e}. Showing unstyled table.")
            st.dataframe(df, use_container_width=True, hide_index=True)

    def _render_rankings(self):
        """Render algorithm rankings - EXCLUDING FAILED ALGORITHMS"""

        st.markdown("### 🏆 Algorithm Rankings")

        rankings = self.comparison.get('rankings', {})

        if not rankings:
            st.info("No rankings available")
            return

        # Filter out failed algorithms from rankings
        successful_algos = [algo for algo in self.comparison['algorithms']
                            if self.comparison['metrics'][algo]['success']]

        if not successful_algos:
            st.warning("⚠️ No algorithms succeeded in finding a path!")
            return

        # Display only successful algorithms in rankings
        num_cols = min(3, len(rankings))
        cols = st.columns(num_cols)

        for idx, (metric, ranks) in enumerate(rankings.items()):
            col_idx = idx % num_cols
            with cols[col_idx]:
                st.markdown(f"**{metric.replace('_', ' ').title()}**")

                # Filter ranks to include only successful algorithms
                filtered_ranks = {algo: rank for algo, rank in ranks.items()
                                  if algo in successful_algos}

                if not filtered_ranks:
                    st.info("No successful algorithms for this metric")
                    continue

                sorted_ranks = sorted(filtered_ranks.items(), key=lambda x: x[1])

                for rank, (algo, value) in enumerate(sorted_ranks, 1):
                    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "🏅"
                    st.write(f"{medal} {rank}. {algo}")

    def _render_radar_chart(self):
        """Render radar chart for multi-dimensional comparison"""

        st.markdown("### 🎯 Multi-Dimensional Comparison")

        algorithms = self.comparison['algorithms']

        # Normalize metrics (0-100 scale, higher is better)
        metrics_list = ['path_cost', 'computation_time', 'latency', 'energy', 'hop_count']

        fig = go.Figure()

        for algo in algorithms:
            algo_metrics = self.comparison['metrics'][algo]

            # Get values and normalize (invert so higher is better)
            values = []
            for metric in metrics_list:
                val = algo_metrics[metric]
                if val < 0:
                    val = 0
                values.append(val)

            # Normalize to 0-100 (inverted so lower original values = higher score)
            max_vals = [max([self.comparison['metrics'][a][m]
                           for a in algorithms]) for m in metrics_list]

            normalized = []
            for val, max_val in zip(values, max_vals):
                if max_val > 0:
                    normalized.append(100 - (val / max_val * 100))
                else:
                    normalized.append(100)

            # Close the polygon
            normalized.append(normalized[0])

            fig.add_trace(go.Scatterpolar(
                r=normalized,
                theta=metrics_list + [metrics_list[0]],
                fill='toself',
                name=algo
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Performance Radar (Higher = Better)",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 **Note**: Metrics are normalized and inverted. Higher values indicate better performance.")
