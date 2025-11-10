"""
Frontend Module - Streamlit UI Components

This module provides the Streamlit-based user interface including
the main application, network visualizer, and performance dashboard.
"""

from .visualizer import NetworkVisualizer
from .dashboard import PerformanceDashboard

__all__ = [
    'NetworkVisualizer',
    'PerformanceDashboard',
]

__version__ = "1.0.0"

# UI Configuration
UI_CONFIG = {
    'theme': 'light',
    'layout': 'wide',
    'sidebar_state': 'expanded'
}
