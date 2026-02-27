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

    # ── NULL MODEL: Null overlay on event-aligned plots ───────────────
    null_model = result.get("metrics", {}).get("null_model")
    null_npz_path = result_dir / "null_token_metrics.npz"
    null_metric_arrays: dict[str, np.ndarray] = {}
    if null_npz_path.exists():
        null_npz = np.load(str(null_npz_path), allow_pickle=False)
        null_metric_arrays = dict(null_npz)

    if null_metric_arrays and failure_index is not None and np.any(failure_index >= 0):
        try:
            from src.visualization.null_overlay import (
                compute_null_distribution_stats,
                plot_event_aligned_with_null,
            )

            # Rebuild events (reuse logic from event-aligned section above)
            generated = metric_arrays.get("generated")
            rule_outcome = metric_arrays.get("rule_outcome")

            if generated is not None and rule_outcome is not None:
                from src.analysis.event_extraction import AnalysisEvent
                from src.evaluation.behavioral import RuleOutcome

                null_events = []
                event_positions = []
                for seq_idx in range(len(failure_index)):
                    fi = int(failure_index[seq_idx])
                    if fi >= 0:
                        null_events.append(
                            AnalysisEvent(
                                walk_idx=seq_idx,
                                encounter_step=max(0, fi - 5),
                                resolution_step=fi,
                                r_value=5,
                                outcome=RuleOutcome.VIOLATED,
                                is_first_violation=True,
                            )
                        )
                        event_positions.append(fi)
                    else:
                        svd_keys = [
                            k for k in metric_arrays.keys()
                            if "." in k and k.split(".")[0] in ("qkt", "avwo", "wvwo")
                        ]
                        pos = min(50, metric_arrays[svd_keys[0]].shape[1] - 1) if svd_keys else 50
                        null_events.append(
                            AnalysisEvent(
                                walk_idx=seq_idx,
                                encounter_step=max(0, pos - 5),
                                resolution_step=pos,
                                r_value=5,
                                outcome=RuleOutcome.FOLLOWED,
                                is_first_violation=False,
                            )
                        )

                # Generate null overlay for primary metrics
                primary_prefixes = (
                    "qkt.layer_0.stable_rank",
                    "qkt.layer_0.grassmannian_distance",
                    "avwo.layer_0.grassmannian_distance",
                )
                for metric_key in null_metric_arrays.keys():
                    if not any(metric_key.startswith(p) or metric_key == p for p in primary_prefixes):
                        continue
                    if metric_key not in metric_arrays:
                        continue

                    try:
                        null_stats = compute_null_distribution_stats(
                            null_metric_arrays[metric_key],
                            event_positions,
                            window=10,
                        )
                        fig = plot_event_aligned_with_null(
                            metric_arrays[metric_key],
                            null_events,
                            null_stats,
                            window=10,
                            metric_name=metric_key,
                        )
                        safe_name = metric_key.replace(".", "_")
                        paths = save_figure(fig, figures_dir, f"null_overlay_{safe_name}")
                        generated_files.extend(paths)
                        log.info("Generated: null_overlay_%s", safe_name)
                    except Exception as e:
                        log.warning("Failed to generate null_overlay_%s: %s", metric_key, e)

        except Exception as e:
            log.warning("Failed to generate null overlay plots: %s", e)

    # ── NULL MODEL: MP histogram ──────────────────────────────────────
    if null_model and "marchenko_pastur" in null_model:
        try:
            from src.visualization.mp_histogram import plot_mp_histogram

            mp_data = null_model["marchenko_pastur"]
            gamma_val = mp_data.get("gamma", 0.5)

            for anchor_name, anchor_data in mp_data.get("anchor_positions", {}).items():
                # Check if we have raw SV data in null NPZ for this anchor
                sv_key = f"mp_svs_{anchor_name}"
                if sv_key in null_metric_arrays:
                    sv_data = null_metric_arrays[sv_key]
                else:
                    # Fall back to using null metric arrays at a representative position
                    continue

                try:
                    mp_result_for_plot = {
                        "sigma2": mp_data.get("sigma2", 1.0),
                        "lambda_minus": mp_data.get("lambda_minus"),
                        "lambda_plus": mp_data.get("lambda_plus"),
                        "ks_statistic": anchor_data.get("ks_statistic", 0.0),
                        "ks_p_value": anchor_data.get("ks_p_value", 0.0),
                    }
                    fig = plot_mp_histogram(
                        sv_data, gamma_val, mp_result_for_plot,
                        position_label=anchor_name,
                    )
                    paths = save_figure(fig, figures_dir, f"mp_histogram_{anchor_name}")
                    generated_files.extend(paths)
                    log.info("Generated: mp_histogram_%s", anchor_name)
                except Exception as e:
                    log.warning("Failed to generate mp_histogram_%s: %s", anchor_name, e)

        except Exception as e:
            log.warning("Failed to generate MP histogram: %s", e)

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

    # ── PRCL-01/03: PR Curves ──────────────────────────────────────────
    pr_curves_data = result.get("metrics", {}).get("pr_curves", {})
    if pr_curves_data:
        try:
            from src.visualization.pr_curves import plot_pr_curves

            for r_val_str, r_data in pr_curves_data.get("by_r_value", {}).items():
                by_metric = r_data.get("by_metric", {})
                if not by_metric:
                    continue
                r_value = int(r_val_str)
                fig = plot_pr_curves(by_metric, r_value=r_value)
                paths = save_figure(fig, figures_dir, f"pr_curve_r{r_value}")
                generated_files.extend(paths)
                log.info("Generated: pr_curve_r%d", r_value)

        except Exception as e:
            log.warning("Failed to generate PR curves: %s", e)

    # ── PRCL-02/03: Calibration diagnostics ────────────────────────────
    calibration_data = result.get("metrics", {}).get("calibration", {})
    if calibration_data:
        try:
            from src.visualization.calibration import plot_reliability_diagram

            for r_val_str, r_data in calibration_data.get("by_r_value", {}).items():
                by_metric = r_data.get("by_metric", {})
                if not by_metric:
                    continue
                r_value = int(r_val_str)

                for metric_key, m_data in by_metric.items():
                    ece_list = m_data.get("ece_by_lookback", [])
                    lookback_data = [{"ece": ece_val} for ece_val in ece_list]

                    try:
                        fig = plot_reliability_diagram(
                            metric_name=metric_key,
                            lookback_data=lookback_data,
                            r_value=r_value,
                        )
                        safe_name = metric_key.replace(".", "_")
                        paths = save_figure(
                            fig, figures_dir, f"calibration_{safe_name}_r{r_value}"
                        )
                        generated_files.extend(paths)
                        log.info("Generated: calibration_%s_r%d", safe_name, r_value)
                    except Exception as e:
                        log.warning(
                            "Failed to generate calibration_%s_r%d: %s",
                            safe_name, r_value, e,
                        )

        except Exception as e:
            log.warning("Failed to generate calibration plots: %s", e)

    # ── OVHD-01/02/03: SVD Benchmark ─────────────────────────────────
    svd_benchmark = result.get("metrics", {}).get("svd_benchmark", {})
    if svd_benchmark and svd_benchmark.get("by_target"):
        try:
            from src.visualization.svd_benchmark import (
                plot_svd_accuracy_tradeoff,
                plot_svd_benchmark_bars,
            )

            fig = plot_svd_benchmark_bars(svd_benchmark)
            paths = save_figure(fig, figures_dir, "svd_benchmark_bars")
            generated_files.extend(paths)
            log.info("Generated: svd_benchmark_bars")

            fig = plot_svd_accuracy_tradeoff(svd_benchmark)
            paths = save_figure(fig, figures_dir, "svd_benchmark_tradeoff")
            generated_files.extend(paths)
            log.info("Generated: svd_benchmark_tradeoff")

        except Exception as e:
            log.warning("Failed to generate SVD benchmark plots: %s", e)

    # ── SFTX-02/03: Perturbation Bound ──────────────────────────────
    perturbation_bound = result.get("metrics", {}).get("perturbation_bound", {})
    if perturbation_bound and perturbation_bound.get("by_magnitude"):
        try:
            from src.visualization.perturbation_bound import (
                plot_bound_by_magnitude,
                plot_bound_tightness,
            )

            fig = plot_bound_tightness(perturbation_bound)
            paths = save_figure(fig, figures_dir, "perturbation_bound_tightness")
            generated_files.extend(paths)
            log.info("Generated: perturbation_bound_tightness")

            fig = plot_bound_by_magnitude(perturbation_bound)
            paths = save_figure(fig, figures_dir, "perturbation_bound_detail")
            generated_files.extend(paths)
            log.info("Generated: perturbation_bound_detail")

        except Exception as e:
            log.warning("Failed to generate perturbation bound plots: %s", e)

    # ── SPEC-03: Spectrum Analysis ──────────────────────────────────
    spectrum_analysis = result.get("metrics", {}).get("spectrum_analysis", {})
    spectrum_npz_path = result_dir / "spectrum_trajectories.npz"
    if spectrum_analysis or spectrum_npz_path.exists():
        try:
            from src.visualization.spectrum import (
                plot_spectrum_auroc,
                plot_spectrum_trajectory_sample,
            )

            # AUROC plots per r-value
            by_r = spectrum_analysis.get("by_r_value", {})
            for r_val_str, r_data in by_r.items():
                by_metric = r_data.get("by_metric", {})
                if by_metric:
                    r_value = int(r_val_str)
                    fig = plot_spectrum_auroc(by_metric, r_value=r_value)
                    paths = save_figure(fig, figures_dir, f"spectrum_auroc_r{r_value}")
                    generated_files.extend(paths)
                    log.info("Generated: spectrum_auroc_r%d", r_value)

            # Sample trajectory plot (first sequence, first layer)
            if spectrum_npz_path.exists():
                spectrum_data = np.load(str(spectrum_npz_path), allow_pickle=False)
                for key in spectrum_data.files:
                    if key.endswith(".spectrum"):
                        spectra = spectrum_data[key]
                        fig = plot_spectrum_trajectory_sample(spectra, sequence_idx=0)
                        layer_tag = key.replace(".spectrum", "").replace(".", "_")
                        paths = save_figure(fig, figures_dir, f"spectrum_traj_{layer_tag}")
                        generated_files.extend(paths)
                        log.info("Generated: spectrum_traj_%s", layer_tag)
                        break  # Only plot first layer for brevity

        except Exception as e:
            log.warning("Failed to generate spectrum analysis plots: %s", e)

    # ── COMP-01/02: Compliance Curve ─────────────────────────────────
    compliance_curve = result.get("metrics", {}).get("compliance_curve", {})
    if compliance_curve:
        try:
            from src.visualization.compliance import plot_compliance_curve

            curve_data = compliance_curve.get("curve", {})
            if curve_data and curve_data.get("r_over_w_values"):
                fig = plot_compliance_curve(curve_data)
                paths = save_figure(fig, figures_dir, "compliance_curve")
                generated_files.extend(paths)
                log.info("Generated: compliance_curve")

        except Exception as e:
            log.warning("Failed to generate compliance curve: %s", e)

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
