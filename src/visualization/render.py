"""Orchestrator: render all figures for a single experiment.

Reads result.json + token_metrics.npz, calls all plot functions,
saves to results/{experiment_id}/figures/ as PNG + SVG.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.visualization.style import apply_style, save_figure

log = logging.getLogger(__name__)


def load_result_data(result_dir: str | Path) -> dict[str, Any]:
    """Load result.json and token_metrics.npz from an experiment directory.

    Args:
        result_dir: Path to results/{experiment_id}/ directory.

    Returns:
        Dict with keys:
        - result: The parsed result.json dict
        - metric_arrays: Dict of arrays from token_metrics.npz
        - curves: Training curves from result.metrics.curves (or empty dict)
    """
    result_dir = Path(result_dir)

    # Load result.json
    result_path = result_dir / "result.json"
    with open(result_path) as f:
        result = json.load(f)

    # Load token_metrics.npz
    npz_path = result_dir / "token_metrics.npz"
    metric_arrays: dict[str, Any] = {}
    if npz_path.exists():
        npz = np.load(str(npz_path), allow_pickle=False)
        metric_arrays = dict(npz)

    # Extract curves
    curves = result.get("metrics", {}).get("curves", {})

    return {
        "result": result,
        "metric_arrays": metric_arrays,
        "curves": curves,
    }


def render_all(result_dir: str | Path) -> list[Path]:
    """Generate all figures for a single experiment.

    Loads data, applies style, generates each plot type, saves
    as PNG + SVG to {result_dir}/figures/.

    Each plot type is wrapped in try/except to ensure one failure
    doesn't block the others.

    Args:
        result_dir: Path to results/{experiment_id}/ directory.

    Returns:
        List of paths to generated figure files.
    """
    apply_style()

    result_dir = Path(result_dir)
    data = load_result_data(result_dir)
    figures_dir = result_dir / "figures"
    generated_files: list[Path] = []

    result = data["result"]
    metric_arrays = data["metric_arrays"]
    curves = data["curves"]

    # ── PLOT-02: Training convergence curves ──────────────────────────
    if curves and ("train_loss" in curves or "edge_compliance" in curves):
        try:
            from src.visualization.training import plot_training_curves

            fig = plot_training_curves(curves)
            paths = save_figure(fig, figures_dir, "training_curves")
            generated_files.extend(paths)
            log.info("Generated: training_curves")
        except Exception as e:
            log.warning("Failed to generate training_curves: %s", e)

    # ── PLOT-04: Confusion matrix ─────────────────────────────────────
    if "edge_valid" in metric_arrays and "rule_outcome" in metric_arrays:
        try:
            from src.visualization.confusion import plot_confusion_matrix

            fig = plot_confusion_matrix(
                metric_arrays["edge_valid"],
                metric_arrays["rule_outcome"],
            )
            paths = save_figure(fig, figures_dir, "confusion_matrix")
            generated_files.extend(paths)
            log.info("Generated: confusion_matrix")
        except Exception as e:
            log.warning("Failed to generate confusion_matrix: %s", e)

    # ── PLOT-01: Event-aligned SVD metric plots ───────────────────────
    failure_index = metric_arrays.get("failure_index")
    if failure_index is not None and np.any(failure_index >= 0):
        try:
            from src.analysis.event_extraction import AnalysisEvent
            from src.evaluation.behavioral import RuleOutcome

            # Find SVD metric keys in NPZ
            svd_metric_keys = [
                k for k in metric_arrays.keys()
                if "." in k and k.split(".")[0] in ("qkt", "avwo", "wvwo")
                and k not in ("edge_valid", "rule_outcome", "failure_index",
                              "sequence_lengths", "generated")
            ]

            # Build simple events from failure_index for alignment
            generated = metric_arrays.get("generated")
            rule_outcome = metric_arrays.get("rule_outcome")

            if generated is not None and rule_outcome is not None:
                from src.visualization.event_aligned import plot_event_aligned

                # Build events from failure_index (simplified: one event per violated sequence)
                events = []
                for seq_idx in range(len(failure_index)):
                    fi = int(failure_index[seq_idx])
                    if fi >= 0:
                        events.append(
                            AnalysisEvent(
                                walk_idx=seq_idx,
                                encounter_step=max(0, fi - 5),
                                resolution_step=fi,
                                r_value=5,
                                outcome=RuleOutcome.VIOLATED,
                                is_first_violation=True,
                            )
                        )
                    else:
                        # Control events at similar positions
                        pos = min(50, metric_arrays[svd_metric_keys[0]].shape[1] - 1) if svd_metric_keys else 50
                        events.append(
                            AnalysisEvent(
                                walk_idx=seq_idx,
                                encounter_step=max(0, pos - 5),
                                resolution_step=pos,
                                r_value=5,
                                outcome=RuleOutcome.FOLLOWED,
                                is_first_violation=False,
                            )
                        )

                # Plot primary metrics
                primary_prefixes = ("qkt.layer_0.stable_rank", "qkt.layer_0.spectral_entropy",
                                    "qkt.layer_0.grassmannian_distance", "avwo.layer_0.stable_rank",
                                    "avwo.layer_0.grassmannian_distance")
                for metric_key in svd_metric_keys:
                    # Limit to primary-like metrics for manageable output
                    if not any(metric_key.startswith(p) or metric_key == p for p in primary_prefixes):
                        if len(svd_metric_keys) > 10:
                            continue

                    try:
                        fig = plot_event_aligned(
                            metric_arrays[metric_key],
                            events,
                            window=10,
                            metric_name=metric_key,
                        )
                        safe_name = metric_key.replace(".", "_")
                        paths = save_figure(fig, figures_dir, f"event_aligned_{safe_name}")
                        generated_files.extend(paths)
                        log.info("Generated: event_aligned_%s", safe_name)
                    except Exception as e:
                        log.warning("Failed to generate event_aligned_%s: %s", metric_key, e)

        except Exception as e:
            log.warning("Failed to generate event-aligned plots: %s", e)

    # ── PLOT-05: Distribution plots ───────────────────────────────────
    if failure_index is not None:
        try:
            from src.visualization.distributions import plot_pre_post_distributions

            svd_metric_keys = [
                k for k in metric_arrays.keys()
                if "." in k and k.split(".")[0] in ("qkt", "avwo", "wvwo")
                and k not in ("edge_valid", "rule_outcome", "failure_index",
                              "sequence_lengths", "generated")
            ]

            # Plot distributions for a few key metrics
            primary_prefixes = ("qkt.layer_0.stable_rank", "avwo.layer_0.stable_rank")
            for metric_key in svd_metric_keys:
                if not any(metric_key.startswith(p) or metric_key == p for p in primary_prefixes):
                    if len(svd_metric_keys) > 10:
                        continue

                try:
                    fig = plot_pre_post_distributions(
                        metric_arrays[metric_key],
                        failure_index,
                        window=5,
                        metric_name=metric_key,
                    )
                    safe_name = metric_key.replace(".", "_")
                    paths = save_figure(fig, figures_dir, f"distribution_{safe_name}")
                    generated_files.extend(paths)
                    log.info("Generated: distribution_%s", safe_name)
                except Exception as e:
                    log.warning("Failed to generate distribution_%s: %s", metric_key, e)

        except Exception as e:
            log.warning("Failed to generate distribution plots: %s", e)

    # ── PLOT-03: AUROC curves ─────────────────────────────────────────
    pred_horizon = result.get("metrics", {}).get("predictive_horizon", {})
    if pred_horizon:
        try:
            from src.visualization.auroc import plot_auroc_curves

            for r_val_str, r_data in pred_horizon.get("by_r_value", {}).items():
                by_metric = r_data.get("by_metric", {})
                if not by_metric:
                    continue

                r_value = int(r_val_str)
                fig = plot_auroc_curves(by_metric, r_value=r_value)
                paths = save_figure(fig, figures_dir, f"auroc_r{r_value}")
                generated_files.extend(paths)
                log.info("Generated: auroc_r%d", r_value)

        except Exception as e:
            log.warning("Failed to generate AUROC curves: %s", e)

    # ── PLOT-06: Heatmap (skip for single experiment) ─────────────────
    # Heatmap requires multiple (r, w) configs. For single experiment,
    # log a message and skip. Use render_horizon_heatmap() for sweep data.
    config = result.get("config", {})
    training_cfg = config.get("training", {})
    r_val = training_cfg.get("r")
    w_val = training_cfg.get("w")
    if r_val is not None and w_val is not None:
        log.info(
            "Heatmap skipped for single experiment (r=%d, w=%d). "
            "Use render_horizon_heatmap() with sweep directory for multi-config heatmap.",
            r_val, w_val,
        )

    log.info("Generated %d figures to %s", len(generated_files), figures_dir)
    return generated_files
