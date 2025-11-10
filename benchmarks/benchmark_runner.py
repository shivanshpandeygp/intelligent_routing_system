"""
Automated Benchmarking System
"""
import time
import numpy as np
from typing import Dict, List
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run comprehensive benchmarks across topologies"""

    def __init__(self, output_dir: str = "results/benchmarks"):
        """
        Initialize benchmark runner

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def run_benchmark(self, network_graph, algorithms: Dict,
                      source: int, destination: int) -> Dict:
        """
        Run benchmark for a single topology

        Args:
            network_graph: NetworkGraph instance
            algorithms: Dictionary of algorithm instances
            source: Source node
            destination: Destination node

        Returns:
            results: Benchmark results
        """
        results = {
            'topology': network_graph.topology,
            'nodes': network_graph.num_nodes,
            'edges': network_graph.graph.number_of_edges(),
            'source': source,
            'destination': destination,
            'algorithms': {}
        }

        for algo_name, algo_instance in algorithms.items():
            try:
                logger.info(f"Benchmarking {algo_name}...")

                start_time = time.time()
                path, cost, algo_time = algo_instance.find_path(source, destination)
                total_time = time.time() - start_time

                results['algorithms'][algo_name] = {
                    'path': path,
                    'cost': cost,
                    'time': algo_time,
                    'total_time': total_time,
                    'success': len(path) > 1,
                    'hop_count': len(path) - 1 if len(path) > 1 else 0
                }

                logger.info(f"{algo_name}: cost={cost:.2f}, time={algo_time:.4f}s")

            except Exception as e:
                logger.error(f"Error benchmarking {algo_name}: {e}")
                results['algorithms'][algo_name] = {
                    'error': str(e),
                    'success': False
                }

        self.results.append(results)
        return results

    def run_multiple_pairs(self, network_graph, algorithms: Dict,
                           num_pairs: int = 10) -> List[Dict]:
        """Run benchmarks for multiple source-destination pairs"""

        pair_results = []
        nodes = list(network_graph.graph.nodes())

        for i in range(num_pairs):
            source = np.random.choice(nodes)
            destination = np.random.choice([n for n in nodes if n != source])

            logger.info(f"Pair {i + 1}/{num_pairs}: {source} -> {destination}")

            result = self.run_benchmark(network_graph, algorithms, source, destination)
            pair_results.append(result)

        return pair_results

    def generate_summary_report(self) -> pd.DataFrame:
        """Generate summary report from all results"""

        summary_data = []

        for result in self.results:
            for algo_name, algo_result in result['algorithms'].items():
                if algo_result.get('success', False):
                    summary_data.append({
                        'Topology': result['topology'],
                        'Nodes': result['nodes'],
                        'Edges': result['edges'],
                        'Algorithm': algo_name,
                        'Path Cost': algo_result['cost'],
                        'Computation Time (ms)': algo_result['time'] * 1000,
                        'Hop Count': algo_result['hop_count'],
                        'Success Rate': 1.0
                    })

        df = pd.DataFrame(summary_data)

        # Save to CSV
        output_file = self.output_dir / "benchmark_summary.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Summary saved to {output_file}")

        return df

    def generate_comparison_report(self) -> Dict:
        """Generate algorithm comparison statistics"""

        comparison = {
            'by_algorithm': {},
            'by_topology': {}
        }

        # Group by algorithm
        for result in self.results:
            for algo_name, algo_result in result['algorithms'].items():
                if algo_name not in comparison['by_algorithm']:
                    comparison['by_algorithm'][algo_name] = {
                        'costs': [],
                        'times': [],
                        'successes': []
                    }

                if algo_result.get('success', False):
                    comparison['by_algorithm'][algo_name]['costs'].append(algo_result['cost'])
                    comparison['by_algorithm'][algo_name]['times'].append(algo_result['time'])
                    comparison['by_algorithm'][algo_name]['successes'].append(1)
                else:
                    comparison['by_algorithm'][algo_name]['successes'].append(0)

        # Calculate statistics
        for algo_name, data in comparison['by_algorithm'].items():
            comparison['by_algorithm'][algo_name]['stats'] = {
                'avg_cost': np.mean(data['costs']) if data['costs'] else 0,
                'std_cost': np.std(data['costs']) if data['costs'] else 0,
                'avg_time': np.mean(data['times']) if data['times'] else 0,
                'success_rate': np.mean(data['successes'])
            }

        return comparison

    def save_results(self, filename: str = "benchmark_results.pkl"):
        """Save all results to file"""
        import pickle

        output_file = self.output_dir / filename
        with open(output_file, 'wb') as f:
            pickle.dump(self.results, f)

        logger.info(f"Results saved to {output_file}")
