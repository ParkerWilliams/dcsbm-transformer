"""Multi-experiment comparison HTML report generator.

Produces a self-contained HTML report comparing multiple experiments
with scalar metrics sparklines, curve overlay grids, config diff
highlighting, auto-generated verdict, and per-experiment reproduction blocks.
"""

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.reporting.embed import embed_figure
from src.reporting.reproduction import build_reproduction_block
from src.visualization.render import load_result_data
from src.visualization.style import PALETTE

log = logging.getLogger(__name__)

# Template directory relative to this file
_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _flatten_config(config: dict, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten nested config dict with dot-separated keys.

    Args:
        config: Possibly nested dict.
        prefix: Key prefix for recursion (internal use).

    Returns:
        Flat dict with dot-separated keys, e.g.
        ``{"model.d_model": 128, "graph.n": 100}``.
    """
    flat: dict[str, Any] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_config(value, full_key))
        else:
            flat[full_key] = value
    return flat


def compute_config_diff(configs: list[dict]) -> list[dict]:
    """Compare flattened configs across experiments.

    Args:
        configs: List of raw config dicts (one per experiment).

    Returns:
        Sorted list of dicts with keys ``param``, ``values`` (list, one
        per experiment), and ``differs`` (bool, True when any value
        differs from the first).
    """
    flat_configs = [_flatten_config(c) for c in configs]

    # Collect all parameter names across all configs
    all_keys: set[str] = set()
    for fc in flat_configs:
        all_keys.update(fc.keys())

    rows: list[dict] = []
    for param in sorted(all_keys):
        values = [fc.get(param, "N/A") for fc in flat_configs]
        # Check if any value differs from the first
        differs = len(set(str(v) for v in values)) > 1
        rows.append({"param": param, "values": values, "differs": differs})

    return rows


def generate_sparkline(
    values: list[float],
    width: int = 120,
    height: int = 30,
) -> str:
    """Create a tiny horizontal bar chart as a base64 data URI.

    Uses the project's colorblind-safe palette for bar colours.

    Args:
        values: One value per experiment.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        ``data:image/png;base64,...`` URI string.
    """
    dpi = 96
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(values))]
    ax.barh(range(len(values)), values, color=bar_colors, height=0.7)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(left=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def compute_verdict(experiment_data: list[dict]) -> str:
    """Determine which experiment outperforms across scalar metrics.

    Compares ``edge_compliance``, ``rule_compliance``, and
    ``predictive_horizon``-related AUROC across experiments and returns
    a plain-language summary sentence.

    Args:
        experiment_data: List of dicts, each containing at least a
            ``scalars`` key mapping metric names to numeric values.

    Returns:
        Human-readable verdict string.
    """
    if len(experiment_data) < 2:
        return "Single experiment -- no comparison available."

    # Gather all scalar metric names present in any experiment
    all_metrics: set[str] = set()
    for ed in experiment_data:
        scalars = ed.get("scalars", {})
        all_metrics.update(scalars.keys())

    if not all_metrics:
        return "No scalar metrics available for comparison."

    # Metrics where higher is better
    higher_better = {"edge_compliance", "rule_compliance"}

    # Count wins per experiment index
    wins: dict[int, int] = {i: 0 for i in range(len(experiment_data))}
    ties = 0
    compared = 0

    for metric in sorted(all_metrics):
        values: list[tuple[int, float]] = []
        for i, ed in enumerate(experiment_data):
            v = ed.get("scalars", {}).get(metric)
            if v is not None and isinstance(v, (int, float)):
                values.append((i, float(v)))

        if len(values) < 2:
            continue

        compared += 1

        # Determine if higher or lower is better
        if metric in higher_better:
            best_val = max(v for _, v in values)
        elif "loss" in metric.lower() or "error" in metric.lower():
            best_val = min(v for _, v in values)
        else:
            # Default: higher is better
            best_val = max(v for _, v in values)

        winners = [i for i, v in values if v == best_val]
        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            ties += 1

    if compared == 0:
        return "No common metrics to compare."

    # Build verdict
    # Experiment labels: A, B, C, ...
    labels = [chr(ord("A") + i) for i in range(len(experiment_data))]

    best_idx = max(wins, key=lambda k: wins[k])
    best_wins = wins[best_idx]

    if best_wins == 0:
        return f"All {compared} metrics tied across experiments."

    # Check if there is a clear winner
    runner_up_wins = sorted(wins.values(), reverse=True)
    if len(runner_up_wins) > 1 and runner_up_wins[0] == runner_up_wins[1]:
        tied_leaders = [
            labels[i] for i, w in wins.items() if w == best_wins
        ]
        return (
            f"Experiments {', '.join(tied_leaders)} are tied, each winning "
            f"{best_wins}/{compared} metrics."
        )

    verdict = (
        f"Experiment {labels[best_idx]} outperforms on "
        f"{best_wins}/{compared} metrics"
    )
    if ties > 0:
        verdict += f" ({ties} tied)"
    verdict += "."
    return verdict


def _build_curve_overlay_grid(
    all_figure_paths: list[dict[str, list[Path]]],
    labels: list[str],
) -> list[dict[str, str]]:
    """Build composite subplot grids for same-named figures across experiments.

    For each figure type found across experiments, creates a single
    composite figure with N side-by-side subplots (one per experiment)
    sharing the same y-axis limits.

    Args:
        all_figure_paths: List (per experiment) of dicts mapping figure
            stem names to lists of matching PNG paths.
        labels: Experiment labels (one per experiment).

    Returns:
        List of dicts with ``title`` and ``data_uri`` keys.
    """
    # Collect all unique figure names
    all_names: set[str] = set()
    for fp_dict in all_figure_paths:
        all_names.update(fp_dict.keys())

    composites: list[dict[str, str]] = []

    for name in sorted(all_names):
        # Collect paths for this figure across experiments
        paths: list[Path | None] = []
        for fp_dict in all_figure_paths:
            p_list = fp_dict.get(name, [])
            paths.append(p_list[0] if p_list else None)

        # Only include if at least two experiments have this figure
        available = [(i, p) for i, p in enumerate(paths) if p is not None]
        if len(available) < 2:
            continue

        n_plots = len(available)
        fig, axes = plt.subplots(
            1, n_plots,
            figsize=(4 * n_plots, 3.5),
            squeeze=False,
        )

        for ax_idx, (exp_idx, img_path) in enumerate(available):
            ax = axes[0, ax_idx]
            try:
                img_data = plt.imread(str(img_path))
                ax.imshow(img_data)
            except Exception:
                ax.text(
                    0.5, 0.5, "Error loading image",
                    ha="center", va="center", transform=ax.transAxes,
                )
            ax.set_title(labels[exp_idx], fontsize=10)
            ax.axis("off")

        fig.suptitle(
            name.replace("_", " ").title(),
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
        composites.append({
            "title": name.replace("_", " ").title(),
            "data_uri": f"data:image/png;base64,{encoded}",
        })

    return composites


def generate_comparison_report(
    result_dirs: list[str | Path],
    output_path: str | Path | None = None,
) -> Path:
    """Generate a multi-experiment comparison HTML report.

    Loads data for each experiment, builds scalar metrics with sparklines,
    config diff with highlighting, curve overlay grids, auto-generated
    verdict, and per-experiment reproduction blocks.

    Args:
        result_dirs: List of paths to results/{experiment_id}/ directories.
        output_path: Where to write the HTML. Defaults to
            ``{first_result_dir.parent}/comparison_report.html``.

    Returns:
        Path to the generated HTML report file.
    """
    result_dirs = [Path(d) for d in result_dirs]

    # Load data for each experiment
    experiments: list[dict[str, Any]] = []
    for rd in result_dirs:
        data = load_result_data(rd)
        experiments.append(data)

    # Build experiment labels (A, B, C, ...)
    labels = [chr(ord("A") + i) for i in range(len(experiments))]
    experiment_ids = [
        exp["result"].get("experiment_id", f"Experiment {labels[i]}")
        for i, exp in enumerate(experiments)
    ]

    # ── Scalar Metrics Comparison ────────────────────────────────────
    # Gather all scalar metric names
    all_metric_names: set[str] = set()
    for exp in experiments:
        scalars = exp["result"].get("metrics", {}).get("scalars", {})
        all_metric_names.update(scalars.keys())

    metric_rows: list[dict[str, Any]] = []
    for metric_name in sorted(all_metric_names):
        values: list[Any] = []
        float_values: list[float] = []
        for exp in experiments:
            v = exp["result"].get("metrics", {}).get("scalars", {}).get(metric_name)
            values.append(v)
            if v is not None and isinstance(v, (int, float)):
                float_values.append(float(v))

        sparkline = ""
        if len(float_values) >= 2:
            sparkline = generate_sparkline(float_values)

        metric_rows.append({
            "name": metric_name,
            "experiment_values": values,
            "sparkline": sparkline,
        })

    # ── Config Diff ──────────────────────────────────────────────────
    configs = [exp["result"].get("config", {}) for exp in experiments]
    config_diff = compute_config_diff(configs)

    # ── Verdict ──────────────────────────────────────────────────────
    experiment_data_for_verdict = [
        {"scalars": exp["result"].get("metrics", {}).get("scalars", {})}
        for exp in experiments
    ]
    verdict = compute_verdict(experiment_data_for_verdict)

    # ── Curve Overlay Grid ───────────────────────────────────────────
    all_figure_paths: list[dict[str, list[Path]]] = []
    for rd in result_dirs:
        figures_dir = rd / "figures"
        fig_dict: dict[str, list[Path]] = {}
        if figures_dir.exists():
            for png_file in sorted(figures_dir.glob("*.png")):
                fig_dict.setdefault(png_file.stem, []).append(png_file)
        all_figure_paths.append(fig_dict)

    curve_overlays = _build_curve_overlay_grid(all_figure_paths, labels)

    # ── Reproduction Blocks ──────────────────────────────────────────
    reproductions: list[dict[str, Any]] = []
    for i, exp in enumerate(experiments):
        repro = build_reproduction_block(exp["result"])
        repro["label"] = labels[i]
        repro["experiment_id"] = experiment_ids[i]
        reproductions.append(repro)

    # ── Render Template ──────────────────────────────────────────────
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("comparison_report.html")

    html = template.render(
        experiment_ids=experiment_ids,
        labels=labels,
        metric_rows=metric_rows,
        config_diff=config_diff,
        verdict=verdict,
        curve_overlays=curve_overlays,
        reproductions=reproductions,
    )

    # ── Write Output ─────────────────────────────────────────────────
    if output_path is None:
        output_path = result_dirs[0].parent / "comparison_report.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    log.info("Comparison report written to %s", output_path)
    return output_path
