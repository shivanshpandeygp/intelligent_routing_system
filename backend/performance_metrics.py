"""
Performance Metrics Calculator for Routing Algorithms
"""
import numpy as np
from typing import Dict, List, Tuple
import time
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Calculate and compare routing algorithm performance"""

    def __init__(self, network_graph):
        """
        Initialize performance metrics calculator

        Args:
            network_graph: NetworkGraph instance
        """
        self.network = network_graph

    def calculate_path_cost(self, path: List[int]) -> float:
        """
        Calculate total path cost

        Args:
            path: List of nodes in path

        Returns:
            cost: Total path cost
        """
        if len(path) < 2:
            return 0.0

        cost = 0.0
        for i in range(len(path) - 1):
            cost += self.network.get_edge_weight(path[i], path[i + 1])

        return cost

    def calculate_latency(self, path: List[int]) -> float:
        """
        Calculate end-to-end latency

        Args:
            path: List of nodes in path

        Returns:
            latency: Total latency in ms
        """
        if len(path) < 2:
            return 0.0

        latency = 0.0
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            if edge in self.network.edge_metrics:
                latency += self.network.edge_metrics[edge]['latency']

        return latency

    def calculate_energy_efficiency(self, path: List[int]) -> float:
        """
        Calculate energy efficiency (lower is better)

        Args:
            path: List of nodes in path

        Returns:
            energy: Total energy consumption
        """
        if len(path) < 2:
            return 0.0

        energy = 0.0
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            if edge in self.network.edge_metrics:
                energy += self.network.edge_metrics[edge]['energy']

        return energy

    def calculate_hop_count(self, path: List[int]) -> int:
        """
        Calculate number of hops

        Args:
            path: List of nodes in path

        Returns:
            hops: Number of hops
        """
        return max(0, len(path) - 1)

    def calculate_packet_delivery_rate(self, paths: List[List[int]],
                                       total_attempts: int) -> float:
        """
        Calculate packet delivery rate

        Args:
            paths: List of successful paths
            total_attempts: Total routing attempts

        Returns:
            pdr: Packet delivery rate (0-1)
        """
        successful = sum(1 for path in paths if len(path) > 1)
        return successful / total_attempts if total_attempts > 0 else 0.0

    def calculate_path_optimality(self, path: List[int],
                                  optimal_cost: float) -> float:
        """
        Calculate how close path is to optimal (percentage)

        Args:
            path: Computed path
            optimal_cost: Cost of optimal path

        Returns:
            optimality: Percentage of optimal (100 = perfect)
        """
        if optimal_cost == 0:
            return 100.0

        path_cost = self.calculate_path_cost(path)
        if path_cost < 0:
            return 0.0

        return (optimal_cost / path_cost) * 100.0 if path_cost > 0 else 0.0

    def compare_algorithms(self, results: Dict) -> Dict:
        """
        Compare multiple routing algorithms with proper failure handling

        Args:
            results: Dictionary with algorithm results

        Returns:
            comparison: Comprehensive comparison dictionary
        """
        comparison = {
            'algorithms': list(results.keys()),
            'metrics': {},
            'rankings': {}
        }

        # Calculate metrics for each algorithm
        for algo_name, algo_result in results.items():
            path = algo_result['path']
            cost = algo_result['cost']
            time_taken = algo_result['time']

            # Check if algorithm succeeded
            success = len(path) > 0 and cost >= 0

            if success:
                # Calculate full metrics for successful algorithms
                latency = sum(
                    self.network.edge_metrics.get((path[i], path[i + 1]), {}).get('latency', 0)
                    for i in range(len(path) - 1)
                )

                energy = sum(
                    self.network.edge_metrics.get((path[i], path[i + 1]), {}).get('energy', 0)
                    for i in range(len(path) - 1)
                )

                hop_count = len(path) - 1
            else:
                # Failed algorithm - use sentinel values
                latency = -1
                energy = -1
                hop_count = 0

            comparison['metrics'][algo_name] = {
                'path_cost': cost,
                'computation_time': time_taken,
                'latency': latency,
                'energy': energy,
                'hop_count': hop_count,
                'success': success,
                'path_length': len(path)
            }

        # Calculate rankings ONLY for successful algorithms
        successful_algos = [algo for algo, metrics in comparison['metrics'].items()
                            if metrics['success']]

        if successful_algos:
            metrics_to_rank = ['path_cost', 'computation_time', 'latency', 'energy', 'hop_count']

            for metric in metrics_to_rank:
                # Get values only from successful algorithms
                values = {algo: comparison['metrics'][algo][metric]
                          for algo in successful_algos
                          if comparison['metrics'][algo][metric] >= 0}  # Exclude negative values

                if values:
                    # Sort and assign ranks (1 = best)
                    sorted_algos = sorted(values.items(), key=lambda x: x[1])
                    comparison['rankings'][metric] = {algo: rank
                                                      for rank, (algo, _) in enumerate(sorted_algos, 1)}
                else:
                    comparison['rankings'][metric] = {}
        else:
            comparison['rankings'] = {}

        return comparison

    def _calculate_rankings(self, metrics: Dict) -> Dict:
        """Calculate rankings for each metric"""
        rankings = {}

        # Metrics where lower is better
        lower_better = ['path_cost', 'computation_time', 'latency',
                        'energy', 'hop_count']

        for metric in ['path_cost', 'computation_time', 'latency',
                       'energy', 'hop_count']:
            values = {algo: m[metric] for algo, m in metrics.items()
                      if m[metric] >= 0}

            if values:
                sorted_algos = sorted(values.items(), key=lambda x: x[1])
                rankings[metric] = {algo: rank + 1
                                    for rank, (algo, _) in enumerate(sorted_algos)}

        return rankings

    def generate_performance_report(self, comparison: Dict) -> str:
        """
        Generate human-readable performance report

        Args:
            comparison: Comparison dictionary from compare_algorithms

        Returns:
            report: Formatted report string
        """
        report = "=" * 70 + "\n"
        report += "ROUTING ALGORITHM PERFORMANCE REPORT\n"
        report += "=" * 70 + "\n\n"

        for algo_name in comparison['algorithms']:
            metrics = comparison['metrics'][algo_name]
            report += f"Algorithm: {algo_name}\n"
            report += "-" * 70 + "\n"
            report += f"  Path Cost:          {metrics['path_cost']:.2f}\n"
            report += f"  Computation Time:   {metrics['computation_time']:.4f} seconds\n"
            report += f"  Latency:            {metrics['latency']:.2f} ms\n"
            report += f"  Energy:             {metrics['energy']:.2f} units\n"
            report += f"  Hop Count:          {metrics['hop_count']}\n"
            report += f"  Success:            {'✓' if metrics['success'] else '✗'}\n"
            report += "\n"

        report += "=" * 70 + "\n"
        report += "RANKINGS (1 = Best)\n"
        report += "=" * 70 + "\n"

        for metric, ranks in comparison['rankings'].items():
            report += f"\n{metric.replace('_', ' ').title()}:\n"
            for algo, rank in sorted(ranks.items(), key=lambda x: x[1]):
                report += f"  {rank}. {algo}\n"

        return report

    def export_to_csv(self, comparison: Dict, filepath: str):
        """Export comparison to CSV"""
        import csv

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = ['Algorithm', 'Path Cost', 'Computation Time (s)',
                      'Latency (ms)', 'Energy', 'Hop Count', 'Success']
            writer.writerow(header)

            # Data
            for algo_name in comparison['algorithms']:
                metrics = comparison['metrics'][algo_name]
                row = [
                    algo_name,
                    f"{metrics['path_cost']:.2f}",
                    f"{metrics['computation_time']:.4f}",
                    f"{metrics['latency']:.2f}",
                    f"{metrics['energy']:.2f}",
                    metrics['hop_count'],
                    'Yes' if metrics['success'] else 'No'
                ]
                writer.writerow(row)

        logger.info(f"Metrics exported to {filepath}")

    def export_to_pdf(self, comparison: Dict, filepath: str):
        """Export comparison to PDF"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.units import inch

            doc = SimpleDocTemplate(filepath, pagesize=letter)
            elements = []
            styles = getSampleStyleSheet()

            # Title
            title = Paragraph("Routing Algorithm Performance Report",
                              styles['Title'])
            elements.append(title)
            elements.append(Spacer(1, 0.3 * inch))

            # Create table data
            table_data = [['Algorithm', 'Path Cost', 'Time (s)', 'Latency',
                           'Energy', 'Hops', 'Success']]

            for algo_name in comparison['algorithms']:
                metrics = comparison['metrics'][algo_name]
                table_data.append([
                    algo_name,
                    f"{metrics['path_cost']:.2f}",
                    f"{metrics['computation_time']:.4f}",
                    f"{metrics['latency']:.2f}",
                    f"{metrics['energy']:.2f}",
                    str(metrics['hop_count']),
                    '✓' if metrics['success'] else '✗'
                ])

            # Create table
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            elements.append(table)
            doc.build(elements)

            logger.info(f"PDF report exported to {filepath}")
        except ImportError:
            logger.warning("reportlab not installed. PDF export skipped.")
            logger.info("Install with: pip install reportlab")
