"""Graph generation module for DCSBM graphs with block jumper rules."""

from src.graph.degree_correction import sample_theta
from src.graph.types import GraphData

__all__ = [
    "GraphData",
    "sample_theta",
]
