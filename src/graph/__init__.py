"""Graph generation module for DCSBM graphs with block jumper rules."""

from src.graph.cache import (
    generate_or_load_graph,
    graph_cache_key,
    load_graph,
    save_graph,
)
from src.graph.dcsbm import (
    GraphGenerationError,
    build_probability_matrix,
    generate_dcsbm_graph,
    validate_graph,
)
from src.graph.degree_correction import sample_theta
from src.graph.jumpers import (
    R_SCALES,
    JumperInfo,
    compute_r_values,
    designate_jumpers,
)
from src.graph.types import GraphData
from src.graph.validation import (
    check_non_trivial,
    reachable_blocks_at_distance,
    verify_all_jumpers,
)

__all__ = [
    "GraphData",
    "GraphGenerationError",
    "JumperInfo",
    "R_SCALES",
    "build_probability_matrix",
    "check_non_trivial",
    "compute_r_values",
    "designate_jumpers",
    "generate_dcsbm_graph",
    "generate_or_load_graph",
    "graph_cache_key",
    "load_graph",
    "reachable_blocks_at_distance",
    "sample_theta",
    "save_graph",
    "validate_graph",
    "verify_all_jumpers",
]
