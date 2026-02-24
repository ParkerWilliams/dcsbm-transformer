#!/usr/bin/env python3
"""Entry point for running DCSBM transformer experiments.

Usage:
    python run_experiment.py --config config.json
    python run_experiment.py --config config.json --dry-run
"""

import argparse
import sys
from pathlib import Path

from src.config import config_from_json, config_hash, full_config_hash
from src.results import generate_experiment_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a DCSBM transformer experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config JSON file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load and print config without running the experiment",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    json_str = config_path.read_text()
    config = config_from_json(json_str)

    # Print config summary
    experiment_id = generate_experiment_id(config)
    print(f"Experiment ID: {experiment_id}")
    print(f"Config hash:   {full_config_hash(config)}")
    print(f"Graph hash:    {config_hash(config.graph)}")
    print()
    print(f"Graph:    n={config.graph.n}, K={config.graph.K}, "
          f"p_in={config.graph.p_in}, p_out={config.graph.p_out}")
    print(f"Model:    d_model={config.model.d_model}, "
          f"n_layers={config.model.n_layers}, n_heads={config.model.n_heads}")
    print(f"Training: w={config.training.w}, walk_length={config.training.walk_length}, "
          f"corpus_size={config.training.corpus_size}, r={config.training.r}")
    print(f"Seed:     {config.seed}")

    if args.dry_run:
        print("\n[dry-run] Config loaded successfully. Exiting.")
        return

    # TODO: Phase 2+ — add graph generation, training, and evaluation pipeline
    print("\n[stub] Experiment execution not yet implemented (Phase 2+).")


if __name__ == "__main__":
    main()
