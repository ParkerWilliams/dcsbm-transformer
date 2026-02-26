"""Single-experiment HTML report generator.

Produces a self-contained HTML file with base64-embedded figures,
structured configuration tables, scalar metrics, statistical test
results, and a copy-pasteable reproduction block.
"""

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.reporting.embed import embed_figure
from src.reporting.reproduction import build_reproduction_block
from src.visualization.render import load_result_data

log = logging.getLogger(__name__)

# Template directory relative to this file
_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _build_config_tables(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Categorize config parameters into structured table groups.

    Args:
        config: Raw config dict from result.json.

    Returns:
        Dict mapping category names to lists of {name, value} dicts.
    """
    model = config.get("model", {})
    training = config.get("training", {})
    graph = config.get("graph", {})

    tables: dict[str, list[dict[str, Any]]] = {}

    # Model parameters
    model_params = [
        {"name": "d_model", "value": model.get("d_model", "N/A")},
        {"name": "n_layers", "value": model.get("n_layers", "N/A")},
        {"name": "n_heads", "value": model.get("n_heads", "N/A")},
        {"name": "vocab_size", "value": model.get("vocab_size", "N/A")},
    ]
    tables["Model"] = model_params

    # Training parameters
    seed = config.get("seed")
    if seed is None:
        seed = training.get("seed", "N/A")
    training_params = [
        {"name": "learning_rate", "value": training.get("learning_rate", "N/A")},
        {"name": "num_epochs", "value": training.get("num_epochs", training.get("max_steps", "N/A"))},
        {"name": "batch_size", "value": training.get("batch_size", "N/A")},
        {"name": "w (context window)", "value": training.get("w", "N/A")},
        {"name": "seed", "value": seed},
    ]
    tables["Training"] = training_params

    # Data parameters
    data_params = [
        {"name": "n (vertices)", "value": graph.get("n", "N/A")},
        {"name": "K (blocks)", "value": graph.get("K", "N/A")},
        {"name": "p_in", "value": graph.get("p_in", "N/A")},
        {"name": "p_out", "value": graph.get("p_out", "N/A")},
        {"name": "walk_length", "value": training.get("walk_length", "N/A")},
        {"name": "corpus_size", "value": training.get("corpus_size", "N/A")},
    ]
    tables["Data"] = data_params

    return tables


def _collect_figures(figures_dir: Path) -> dict[str, Any]:
    """Collect and embed all PNG figures from a figures directory.

    Categorizes figures by name pattern into specific template slots
    (training_curves, confusion_matrix) and generic lists (auroc,
    event_aligned, distribution).

    Args:
        figures_dir: Path to {result_dir}/figures/ directory.

    Returns:
        Dict with keys for each figure slot in the template.
    """
    result: dict[str, Any] = {
        "training_curves_figure": None,
        "confusion_matrix_figure": None,
        "auroc_figures": [],
        "event_aligned_figures": [],
        "distribution_figures": [],
    }

    if not figures_dir.exists():
        return result

    for png_file in sorted(figures_dir.glob("*.png")):
        data_uri = embed_figure(png_file)
        if not data_uri:
            continue

        name = png_file.stem
        # Derive a readable title from filename
        title = name.replace("_", " ").replace(".", " ").title()

        if name == "training_curves":
            result["training_curves_figure"] = data_uri
        elif name == "confusion_matrix":
            result["confusion_matrix_figure"] = data_uri
        elif name.startswith("auroc"):
            result["auroc_figures"].append({"title": title, "data_uri": data_uri})
        elif name.startswith("event_aligned"):
            result["event_aligned_figures"].append({"title": title, "data_uri": data_uri})
        elif name.startswith("distribution"):
            result["distribution_figures"].append({"title": title, "data_uri": data_uri})
        else:
            # Unknown figure type -- add to event_aligned as general
            result["event_aligned_figures"].append({"title": title, "data_uri": data_uri})

    return result


def _extract_statistical_tests(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract statistical test results from metrics for display.

    Args:
        metrics: The metrics dict from result.json.

    Returns:
        List of dicts with keys: metric, auroc, p_value, ci_lower, ci_upper, effect_size.
    """
    tests: list[dict[str, Any]] = []

    # Check statistical_controls for per-metric results
    stat_controls = metrics.get("statistical_controls", {})
    headline = stat_controls.get("headline_comparison", {})
    primary = headline.get("primary_metrics", {})

    for metric_name, metric_data in primary.items():
        tests.append({
            "metric": metric_name,
            "auroc": metric_data.get("auroc", "N/A"),
            "p_value": metric_data.get("p_value_corrected", metric_data.get("p_value", "N/A")),
            "ci_lower": metric_data.get("ci_lower", "N/A"),
            "ci_upper": metric_data.get("ci_upper", "N/A"),
            "effect_size": metric_data.get("cohens_d", metric_data.get("effect_size", "N/A")),
        })

    return tests


def _extract_predictive_horizon(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract predictive horizon summary from metrics.

    Args:
        metrics: The metrics dict from result.json.

    Returns:
        List of dicts with keys: r_value, best_metric, best_auroc, significant.
    """
    rows: list[dict[str, Any]] = []
    pred = metrics.get("predictive_horizon", {})
    by_r = pred.get("by_r_value", {})

    for r_str, r_data in sorted(by_r.items(), key=lambda x: int(x[0])):
        by_metric = r_data.get("by_metric", {})
        if not by_metric:
            continue

        # Find best metric for this r-value
        best_name = ""
        best_auroc = 0.0
        for m_name, m_data in by_metric.items():
            auroc_vals = m_data.get("auroc_by_lookback", [])
            if auroc_vals:
                max_auroc = max(auroc_vals)
                if max_auroc > best_auroc:
                    best_auroc = max_auroc
                    best_name = m_name

        rows.append({
            "r_value": int(r_str),
            "best_metric": best_name,
            "best_auroc": best_auroc,
            "significant": "Yes" if best_auroc >= 0.75 else "No",
        })

    return rows


def generate_single_report(
    result_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Generate a single-experiment HTML report.

    Loads experiment data, embeds figures as base64, builds structured
    configuration tables, and renders a self-contained HTML report
    using the Jinja2 template.

    Args:
        result_dir: Path to results/{experiment_id}/ directory.
        output_path: Where to write the HTML. Defaults to {result_dir}/report.html.

    Returns:
        Path to the generated HTML report file.
    """
    result_dir = Path(result_dir)

    # Load experiment data
    data = load_result_data(result_dir)
    result = data["result"]
    config = result.get("config", {})
    metrics = result.get("metrics", {})

    # Build template context
    config_tables = _build_config_tables(config)
    scalar_metrics = metrics.get("scalars", {})

    # Collect and embed figures
    figures_dir = result_dir / "figures"
    figures = _collect_figures(figures_dir)

    # Extract statistical tests and predictive horizon
    statistical_tests = _extract_statistical_tests(metrics)
    predictive_horizon = _extract_predictive_horizon(metrics)

    # Build reproduction block
    reproduction = build_reproduction_block(result)

    # Set up Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("single_report.html")

    # Render
    html = template.render(
        experiment_id=result.get("experiment_id", "Unknown"),
        timestamp=result.get("timestamp", ""),
        description=result.get("description", ""),
        config_tables=config_tables,
        scalar_metrics=scalar_metrics,
        training_curves_figure=figures["training_curves_figure"],
        confusion_matrix_figure=figures["confusion_matrix_figure"],
        auroc_figures=figures["auroc_figures"],
        event_aligned_figures=figures["event_aligned_figures"],
        distribution_figures=figures["distribution_figures"],
        statistical_tests=statistical_tests,
        predictive_horizon=predictive_horizon,
        sequence_analysis=None,
        reproduction=reproduction,
    )

    # Write output
    if output_path is None:
        output_path = result_dir / "report.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    log.info("Report written to %s", output_path)
    return output_path
