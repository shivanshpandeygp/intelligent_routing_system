"""
Benchmarking Module - Performance Evaluation Tools

This module provides tools for generating network topologies and
running comprehensive benchmarks across multiple algorithms and
network configurations.
"""

from .topology_generator import TopologyGenerator
from .benchmark_runner import BenchmarkRunner

__all__ = [
    'TopologyGenerator',
    'BenchmarkRunner',
]

__version__ = "1.0.0"

# Benchmark configuration
BENCHMARK_CONFIG = {
    'default_topologies': ['mesh', 'ring', 'tree', 'random'],
    'default_sizes': [10, 20, 30, 50],
    'num_trials': 10,
    'output_dir': 'results/benchmarks',
}

import logging
logger = logging.getLogger(__name__)
logger.info("Benchmarks module initialized")
