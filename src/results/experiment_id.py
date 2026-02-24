"""Experiment ID generation with scannable parameter slug format."""

from datetime import datetime, timezone

from src.config.experiment import ExperimentConfig


def generate_experiment_id(config: ExperimentConfig) -> str:
    """Generate a scannable experiment ID from config parameters.

    Format: n{n}_w{w}_r{r}_d{d_model}_L{n_layers}_s{seed}_{YYYYMMDD}_{HHMMSS}
    Example: n500_w64_r57_d128_L4_s42_20260224_143012

    The slug encodes key parameters so experiment directories are identifiable
    at a glance in file listings without opening result.json.
    """
    ts = datetime.now(timezone.utc)
    return (
        f"n{config.graph.n}"
        f"_w{config.training.w}"
        f"_r{config.training.r}"
        f"_d{config.model.d_model}"
        f"_L{config.model.n_layers}"
        f"_s{config.seed}"
        f"_{ts.strftime('%Y%m%d_%H%M%S')}"
    )
