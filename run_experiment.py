#!/usr/bin/env python3
"""Entry point for running DCSBM transformer experiments.

Chains all pipeline stages into a single executable command:
graph generation -> walk generation -> training -> evaluation ->
analysis -> visualization -> reporting.

Usage:
    python run_experiment.py --config config.json
    python run_experiment.py --config config.json --dry-run
    python run_experiment.py --config config.json --verbose
"""

import argparse
import json
import logging
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import torch

from src.config import config_from_json, config_hash, full_config_hash
from src.results import generate_experiment_id

log = logging.getLogger(__name__)


@contextmanager
def stage_timer(name: str) -> Generator[None, None, None]:
    """Context manager that prints stage banners with elapsed time."""
    print(f"\n=== {name} ===")
    log.info("Starting: %s", name)
    t0 = time.monotonic()
    yield
    elapsed = time.monotonic() - t0
    print(f"... done in {elapsed:.1f}s")
    log.info("Completed: %s in %.1fs", name, elapsed)


def run_pipeline(config_path: Path, results_dir: str = "results") -> Path:
    """Execute the full experiment pipeline.

    Args:
        config_path: Path to experiment config JSON file.
        results_dir: Base directory for results output.

    Returns:
        Path to the output directory.
    """
    # Lazy imports to keep --dry-run fast
    from src.analysis.auroc_horizon import run_auroc_analysis
    from src.analysis.statistical_controls import apply_statistical_controls
    from src.evaluation import fused_evaluate, save_evaluation_results
    from src.evaluation.split import assign_split
    from src.graph import generate_or_load_graph
    from src.model import create_model
    from src.reporting import generate_single_report
    from src.reproducibility import get_git_hash, set_seed
    from src.training import run_training_pipeline
    from src.visualization.render import render_all
    from src.walk import generate_or_load_walks

    pipeline_start = time.monotonic()

    # Load config
    json_str = config_path.read_text()
    config = config_from_json(json_str)

    # Note: experiment_id includes a timestamp, so we generate it once here
    # and pass the resulting output_dir to all stages. The training pipeline
    # generates its own experiment_id (different timestamp), so we use its
    # result_path to find the actual output directory.
    log.info("Config loaded from %s", config_path)
    log.info("Seed: %d", config.seed)
    log.info("Git hash: %s", get_git_hash())

    # ── Stage 1: Seed ──────────────────────────────────────────────
    with stage_timer("Reproducibility Seeding"):
        set_seed(config.seed)
        log.info("Seed set: %d", config.seed)

    # ── Device selection ───────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Stage 2: Graph Generation ──────────────────────────────────
    with stage_timer("Graph Generation"):
        graph_data, jumpers = generate_or_load_graph(config)
        log.info(
            "Graph: n=%d, K=%d, edges=%d, jumpers=%d",
            graph_data.n, graph_data.K,
            graph_data.adjacency.nnz, len(jumpers),
        )

    # ── Stage 3: Walk Generation ───────────────────────────────────
    with stage_timer("Walk Generation"):
        train_result, eval_result = generate_or_load_walks(
            graph_data, jumpers, config,
        )
        train_walks = train_result.walks
        eval_walks = eval_result.walks
        log.info(
            "Walks: %d train, %d eval (length %d)",
            train_walks.shape[0], eval_walks.shape[0],
            train_walks.shape[1],
        )

    # ── Stage 4: Model Creation ────────────────────────────────────
    with stage_timer("Model Creation"):
        model = create_model(config)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        log.info(
            "Model: d_model=%d, n_layers=%d, n_heads=%d, params=%d",
            config.model.d_model, config.model.n_layers,
            config.model.n_heads, n_params,
        )

    # ── Stage 5: Training ──────────────────────────────────────────
    with stage_timer("Training"):
        training_result = run_training_pipeline(
            model=model,
            train_walks=train_walks,
            eval_walks=eval_walks,
            graph_data=graph_data,
            jumpers=jumpers,
            config=config,
            device=device,
            results_dir=results_dir,
        )
        if training_result.gate_passed:
            log.info(
                "Training gate PASSED at epoch %d",
                training_result.final_epoch,
            )
        else:
            log.warning(
                "Training gate NOT PASSED after %d epochs: %s",
                training_result.final_epoch,
                training_result.failure_reason,
            )

    # Determine the output directory from the training pipeline's result path.
    # run_training_pipeline calls generate_experiment_id (with its own timestamp),
    # so the actual output dir is the parent of result.json it wrote.
    if training_result.result_path:
        output_dir = Path(training_result.result_path).parent
    else:
        # Fallback: generate a new experiment_id
        experiment_id = generate_experiment_id(config)
        output_dir = Path(results_dir) / experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Output directory: %s", output_dir)

    # ── Copy config to output dir ──────────────────────────────────
    shutil.copy2(str(config_path), str(output_dir / "config.json"))
    log.info("Config copied to %s", output_dir / "config.json")

    # ── Stage 6: Evaluation ────────────────────────────────────────
    with stage_timer("Evaluation"):
        model.eval()
        eval_result = fused_evaluate(
            model=model,
            eval_walks=eval_walks,
            graph_data=graph_data,
            jumpers=jumpers,
            config=config,
            device=device,
        )
        n_violations = int((eval_result.failure_index >= 0).sum())
        log.info(
            "Evaluation: %d sequences, %d violations",
            eval_result.sequence_lengths.shape[0], n_violations,
        )

    # ── Stage 6b: Split Assignment ─────────────────────────────────
    with stage_timer("Split Assignment"):
        split_labels = assign_split(eval_result.failure_index)
        log.info("Split: assigned exploratory/confirmatory labels")

    # ── Stage 6c: Save Evaluation Results (NPZ) ───────────────────
    with stage_timer("Save Evaluation Results"):
        eval_summary = save_evaluation_results(
            eval_result,
            output_dir=str(output_dir),
            split_labels=split_labels,
        )
        log.info("Evaluation NPZ written to %s", output_dir)

    # ── Stage 7: AUROC Analysis ────────────────────────────────────
    with stage_timer("AUROC Analysis"):
        # Build metric keys from SVD metrics
        metric_keys = [
            k for k in eval_result.svd_metrics.keys()
            if "." in k and k.split(".")[0] in ("qkt", "avwo", "wvwo")
        ]
        jumper_map = {j.vertex_id: j for j in jumpers}

        # Build eval_result_data dict
        eval_result_data: dict[str, Any] = {
            "generated": eval_result.generated,
            "rule_outcome": eval_result.rule_outcome,
            "failure_index": eval_result.failure_index,
            "sequence_lengths": eval_result.sequence_lengths,
        }
        eval_result_data.update(eval_result.svd_metrics)

        predictive_horizon = run_auroc_analysis(
            eval_result_data=eval_result_data,
            jumper_map=jumper_map,
            metric_keys=metric_keys,
        )
        log.info("Predictive horizon analysis complete")

    # ── Stage 7b: Statistical Controls ─────────────────────────────
    with stage_timer("Statistical Controls"):
        statistical_controls = apply_statistical_controls(
            auroc_results=predictive_horizon,
            eval_data=eval_result_data,
            jumper_map=jumper_map,
        )
        log.info("Statistical controls complete")

    # ── Stage 8: Update result.json ────────────────────────────────
    with stage_timer("Update Result JSON"):
        result_json_path = output_dir / "result.json"
        with open(result_json_path) as f:
            result_data = json.load(f)

        # Add analysis blocks
        result_data["metrics"]["predictive_horizon"] = predictive_horizon
        result_data["metrics"]["statistical_controls"] = statistical_controls

        # Merge evaluation summary scalars
        if "scalars" in eval_summary:
            for k, v in eval_summary["scalars"].items():
                result_data["metrics"]["scalars"][k] = v

        # Ensure metadata has seed and git hash
        result_data["metadata"]["seed"] = config.seed
        result_data["metadata"]["git_hash"] = get_git_hash()

        with open(result_json_path, "w") as f:
            json.dump(result_data, f, indent=2)
        log.info("result.json updated with analysis blocks")

    # ── Stage 9: Visualization ─────────────────────────────────────
    with stage_timer("Visualization"):
        figures = render_all(str(output_dir))
        log.info("Generated %d figure files", len(figures))

    # ── Stage 10: Reporting ────────────────────────────────────────
    with stage_timer("Reporting"):
        report_path = generate_single_report(str(output_dir))
        log.info("Report written to %s", report_path)

    # ── Final Summary ──────────────────────────────────────────────
    total_elapsed = time.monotonic() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {total_elapsed:.1f}s")
    print(f"  Experiment: {output_dir.name}")
    print(f"  Output:     {output_dir}")
    print(f"  Result:     {output_dir / 'result.json'}")
    print(f"  Figures:    {len(figures)} files")
    print(f"  Report:     {report_path}")
    print(f"  Gate:       {'PASSED' if training_result.gate_passed else 'NOT PASSED'}")
    print(f"{'=' * 60}")

    return output_dir


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
        help="Show pipeline plan without running the experiment",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

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
        print(f"\nPipeline plan for experiment {experiment_id}:")
        print(f"  1. Set seed: {config.seed}")
        print(f"  2. Graph generation: n={config.graph.n}, K={config.graph.K}, "
              f"p_in={config.graph.p_in}, p_out={config.graph.p_out}")
        print(f"  3. Walk generation: corpus_size={config.training.corpus_size}, "
              f"walk_length={config.training.walk_length}")
        print(f"  4. Model creation: d_model={config.model.d_model}, "
              f"n_layers={config.model.n_layers}, n_heads={config.model.n_heads}")
        print(f"  5. Training: batch_size={config.training.batch_size}, "
              f"max_steps={config.training.max_steps}")
        print(f"  6. Evaluation: fused SVD + behavioral")
        print(f"  7. Analysis: AUROC horizon + statistical controls")
        print(f"  8. Visualization: all plots to figures/")
        print(f"  9. Reporting: HTML report")
        print(f"\nOutput: results/{experiment_id}/")
        print(f"  - result.json")
        print(f"  - token_metrics.npz")
        print(f"  - spectrum_trajectories.npz")
        print(f"  - figures/ (PNG + SVG)")
        print(f"  - report.html")
        print(f"  - config.json (copy)")
        print(f"\n[dry-run] Config loaded successfully. Exiting.")
        return

    # Run full pipeline
    try:
        run_pipeline(config_path)
    except Exception:
        log.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
