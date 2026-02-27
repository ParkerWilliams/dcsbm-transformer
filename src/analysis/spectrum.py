"""Full spectrum trajectory analysis with Frenet-Serret curvature and torsion.

Computes discrete curvature and torsion on spectral trajectories (singular value
vectors over time) with Savitzky-Golay smoothing for noise suppression. Handles
ordering crossings, zero-velocity degeneracies, and boundary effects.

Phase 15: Advanced Analysis (SPEC-01, SPEC-02, SPEC-03).
"""

import logging
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from src.analysis.auroc_horizon import (
    auroc_from_groups,
    compute_auroc_curve,
)
from src.analysis.event_extraction import (
    AnalysisEvent,
    extract_events,
    filter_contaminated_events,
    stratify_by_r,
)
from src.evaluation.behavioral import RuleOutcome

log = logging.getLogger(__name__)

# Default parameters
DEFAULT_TOP_K = 8
DEFAULT_SAVGOL_WINDOW = 7
DEFAULT_SAVGOL_POLYORDER = 3
VELOCITY_EPS = 1e-10  # Threshold for zero-velocity guard


def smooth_spectrum(
    spectra: np.ndarray,
    window_length: int = DEFAULT_SAVGOL_WINDOW,
    polyorder: int = DEFAULT_SAVGOL_POLYORDER,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to spectral trajectory.

    Args:
        spectra: Singular value trajectory, shape [n_steps, k].
        window_length: Smoothing window length (must be odd, >= polyorder+2).
        polyorder: Polynomial order for Savitzky-Golay filter.

    Returns:
        Smoothed array of same shape as input.
    """
    n_steps = spectra.shape[0]
    if n_steps < window_length:
        return spectra.copy()
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(spectra, window_length, polyorder, axis=0)


def detect_ordering_crossings(spectra: np.ndarray) -> np.ndarray:
    """Detect steps where adjacent singular values swap ordering.

    Singular values from SVD are returned in descending order. An ordering
    crossing at step t means sigma_i(t) < sigma_{i+1}(t) for some i, indicating
    a sorting artifact rather than a physical signal.

    Args:
        spectra: Singular value trajectory, shape [n_steps, k].

    Returns:
        Boolean mask of shape [n_steps], True where crossing detected.
        Marks t-1, t, t+1 as invalid (derivative stencil contamination).
    """
    n_steps, k = spectra.shape
    crossing_mask = np.zeros(n_steps, dtype=bool)

    if k < 2:
        return crossing_mask

    # Check each pair of adjacent singular values
    for i in range(k - 1):
        # A crossing occurs when sigma_i < sigma_{i+1} (should be descending)
        crossed = spectra[:, i] < spectra[:, i + 1]
        for t in range(n_steps):
            if crossed[t]:
                # Mark t and neighbors
                for dt in [-1, 0, 1]:
                    idx = t + dt
                    if 0 <= idx < n_steps:
                        crossing_mask[idx] = True

    return crossing_mask


def spectral_curvature(
    spectra: np.ndarray,
    window_length: int = DEFAULT_SAVGOL_WINDOW,
    polyorder: int = DEFAULT_SAVGOL_POLYORDER,
) -> np.ndarray:
    """Compute discrete Frenet-Serret curvature of spectral trajectory.

    Curvature measures how sharply the spectral curve bends. High curvature
    indicates a rapid change in the direction of spectral evolution, which
    may precede behavioral transitions.

    Uses the formula: kappa = ||a_perp|| / ||v||^2
    where a_perp = a - (a.v_hat)*v_hat is acceleration perpendicular to velocity.

    Args:
        spectra: Singular value trajectory, shape [n_steps, k].
        window_length: Savitzky-Golay window length.
        polyorder: Savitzky-Golay polynomial order.

    Returns:
        Curvature array of shape [n_steps], with NaN at boundaries,
        zero-velocity steps, and ordering crossings.
    """
    n_steps, k = spectra.shape
    curvature = np.full(n_steps, np.nan)

    if n_steps < 3:
        return curvature

    # Detect ordering crossings on original (unsmoothed) spectra
    crossing_mask = detect_ordering_crossings(spectra)

    # Smooth before differentiation
    smoothed = smooth_spectrum(spectra, window_length, polyorder)

    # First derivative (velocity): v[t] = smoothed[t+1] - smoothed[t]
    v = np.diff(smoothed, axis=0)  # [n_steps-1, k]

    # Second derivative (acceleration): a[t] = v[t+1] - v[t]
    a = np.diff(v, axis=0)  # [n_steps-2, k]

    # Compute curvature for valid range
    for t in range(len(a)):
        v_t = v[t]  # [k]
        a_t = a[t]  # [k]

        speed = np.linalg.norm(v_t)
        if speed < VELOCITY_EPS:
            continue  # curvature stays NaN

        v_hat = v_t / speed
        # Perpendicular acceleration: a - (a.v_hat)*v_hat
        a_parallel = np.dot(a_t, v_hat) * v_hat
        a_perp = a_t - a_parallel
        a_perp_norm = np.linalg.norm(a_perp)

        # Map t in derivative space to t+1 in original space
        # (diff shifts index: v[t] corresponds to interval [t, t+1],
        #  a[t] corresponds to step t+1 in original spectra)
        orig_idx = t + 1
        curvature[orig_idx] = a_perp_norm / (speed**2)

    # Apply ordering crossing mask
    curvature[crossing_mask] = np.nan

    return curvature


def spectral_torsion(
    spectra: np.ndarray,
    window_length: int = DEFAULT_SAVGOL_WINDOW,
    polyorder: int = DEFAULT_SAVGOL_POLYORDER,
) -> np.ndarray:
    """Compute discrete torsion of spectral trajectory.

    Torsion measures how much the curve twists out of its osculating plane.
    For a curve in R^k, torsion captures out-of-plane acceleration changes.

    Uses projection-based formula for k-dimensional curves:
    tau = ||j_perp|| / (||v|| * ||a_perp||)
    where j_perp is the jerk component orthogonal to both velocity and
    normal acceleration directions.

    Args:
        spectra: Singular value trajectory, shape [n_steps, k].
        window_length: Savitzky-Golay window length.
        polyorder: Savitzky-Golay polynomial order.

    Returns:
        Torsion array of shape [n_steps], with NaN at boundaries,
        degenerate steps, and ordering crossings.
    """
    n_steps, k = spectra.shape
    torsion = np.full(n_steps, np.nan)

    if n_steps < 4:
        return torsion

    # Detect ordering crossings on original spectra
    crossing_mask = detect_ordering_crossings(spectra)

    # Smooth before differentiation
    smoothed = smooth_spectrum(spectra, window_length, polyorder)

    # Derivatives
    v = np.diff(smoothed, axis=0)   # [n_steps-1, k] velocity
    a = np.diff(v, axis=0)          # [n_steps-2, k] acceleration
    j = np.diff(a, axis=0)          # [n_steps-3, k] jerk

    for t in range(len(j)):
        v_t = v[t]
        a_t = a[t]
        j_t = j[t]

        speed = np.linalg.norm(v_t)
        if speed < VELOCITY_EPS:
            continue

        v_hat = v_t / speed

        # Normal acceleration (perpendicular to velocity)
        a_parallel = np.dot(a_t, v_hat) * v_hat
        a_perp = a_t - a_parallel
        a_perp_norm = np.linalg.norm(a_perp)

        if a_perp_norm < VELOCITY_EPS:
            continue  # straight line, torsion undefined

        n_hat = a_perp / a_perp_norm

        # Project jerk out of velocity and normal directions
        j_v = np.dot(j_t, v_hat) * v_hat
        j_n = np.dot(j_t, n_hat) * n_hat
        j_perp = j_t - j_v - j_n
        j_perp_norm = np.linalg.norm(j_perp)

        # Map to original index (jerk at t corresponds to step t+2)
        orig_idx = t + 2
        torsion[orig_idx] = j_perp_norm / (speed * a_perp_norm)

    # Apply ordering crossing mask
    torsion[crossing_mask] = np.nan

    return torsion


def compute_spectrum_analysis(
    spectra: np.ndarray,
    window_length: int = DEFAULT_SAVGOL_WINDOW,
    polyorder: int = DEFAULT_SAVGOL_POLYORDER,
) -> dict[str, np.ndarray]:
    """Compute full spectrum analysis for a single sequence.

    Args:
        spectra: Singular value trajectory, shape [n_steps, k].
        window_length: Savitzky-Golay window length.
        polyorder: Savitzky-Golay polynomial order.

    Returns:
        Dict with keys:
            curvature: [n_steps] curvature values
            torsion: [n_steps] torsion values
            crossing_mask: [n_steps] boolean mask of ordering crossings
            smoothed: [n_steps, k] smoothed spectra
    """
    return {
        "curvature": spectral_curvature(spectra, window_length, polyorder),
        "torsion": spectral_torsion(spectra, window_length, polyorder),
        "crossing_mask": detect_ordering_crossings(spectra),
        "smoothed": smooth_spectrum(spectra, window_length, polyorder),
    }


def compute_spectrum_analysis_batch(
    spectra_batch: np.ndarray,
    window_length: int = DEFAULT_SAVGOL_WINDOW,
    polyorder: int = DEFAULT_SAVGOL_POLYORDER,
) -> dict[str, np.ndarray]:
    """Compute spectrum analysis for a batch of sequences.

    Args:
        spectra_batch: Singular value trajectories, shape [n_sequences, n_steps, k].
        window_length: Savitzky-Golay window length.
        polyorder: Savitzky-Golay polynomial order.

    Returns:
        Dict with keys:
            curvature: [n_sequences, n_steps]
            torsion: [n_sequences, n_steps]
            crossing_mask: [n_sequences, n_steps]
    """
    n_sequences, n_steps, k = spectra_batch.shape
    curvature = np.full((n_sequences, n_steps), np.nan)
    torsion_arr = np.full((n_sequences, n_steps), np.nan)
    crossing_mask = np.zeros((n_sequences, n_steps), dtype=bool)

    for i in range(n_sequences):
        result = compute_spectrum_analysis(
            spectra_batch[i], window_length, polyorder
        )
        curvature[i] = result["curvature"]
        torsion_arr[i] = result["torsion"]
        crossing_mask[i] = result["crossing_mask"]

    return {
        "curvature": curvature,
        "torsion": torsion_arr,
        "crossing_mask": crossing_mask,
    }


def run_spectrum_auroc_analysis(
    spectrum_path: str | Path,
    eval_result_data: dict,
    jumper_map: dict,
    layers: list[int] | None = None,
    min_events_per_class: int = 5,
    window_length: int = DEFAULT_SAVGOL_WINDOW,
    polyorder: int = DEFAULT_SAVGOL_POLYORDER,
) -> dict:
    """Run AUROC analysis on curvature/torsion as secondary predictive metrics.

    Loads spectrum trajectories, computes curvature/torsion, then feeds them
    into the existing AUROC pipeline alongside primary metrics.

    Args:
        spectrum_path: Path to spectrum_trajectories.npz.
        eval_result_data: Dict with generated, rule_outcome, failure_index arrays.
        jumper_map: Mapping from vertex_id to JumperInfo.
        layers: Layer indices to analyze (default: all available).
        min_events_per_class: Minimum events per class for AUROC.
        window_length: Savitzky-Golay window length.
        polyorder: Savitzky-Golay polynomial order.

    Returns:
        Nested dict with structure:
            status: "exploratory"
            config: analysis parameters
            by_r_value: {r: {n_violations, n_controls, by_metric: {...}}}
    """
    spectrum_path = Path(spectrum_path)
    if not spectrum_path.exists():
        log.warning("Spectrum trajectories not found: %s", spectrum_path)
        return {"status": "exploratory", "error": "spectrum file not found"}

    spectrum_data = np.load(str(spectrum_path))

    # Discover available layers
    available_keys = [k for k in spectrum_data.files if k.endswith(".spectrum")]
    if not available_keys:
        log.warning("No spectrum keys found in %s", spectrum_path)
        return {"status": "exploratory", "error": "no spectrum keys"}

    if layers is None:
        layers = sorted(set(
            int(k.split(".layer_")[1].split(".")[0])
            for k in available_keys
        ))

    # Extract events
    generated = eval_result_data["generated"]
    rule_outcome = eval_result_data["rule_outcome"]
    failure_index = eval_result_data["failure_index"]

    events = extract_events(generated, rule_outcome, failure_index, jumper_map)
    clean_events, _filter_stats = filter_contaminated_events(events)
    by_r = stratify_by_r(clean_events)

    # Compute curvature/torsion for each layer
    metric_arrays: dict[str, np.ndarray] = {}

    for layer_idx in layers:
        spec_key = f"qkt.layer_{layer_idx}.spectrum"
        if spec_key not in spectrum_data.files:
            continue

        spectra = spectrum_data[spec_key].astype(np.float32)  # [n_seqs, n_steps, k]
        batch_result = compute_spectrum_analysis_batch(
            spectra, window_length, polyorder
        )

        # Curvature and torsion arrays: [n_seqs, n_steps]
        # Trim to match SVD metric array shape [n_seqs, n_steps-1]
        n_steps_metric = spectra.shape[1]
        curv = batch_result["curvature"][:, :n_steps_metric]
        tors = batch_result["torsion"][:, :n_steps_metric]

        metric_arrays[f"qkt.layer_{layer_idx}.curvature"] = curv
        metric_arrays[f"qkt.layer_{layer_idx}.torsion"] = tors

    # Run AUROC analysis per r-value
    result: dict = {
        "status": "exploratory",
        "config": {
            "layers": layers,
            "smoothing": {"window": window_length, "polyorder": polyorder},
            "min_events_per_class": min_events_per_class,
        },
        "by_r_value": {},
    }

    for r_val, events_for_r in by_r.items():
        violations = [e for e in events_for_r if e.outcome == RuleOutcome.VIOLATED]
        controls = [e for e in events_for_r if e.outcome == RuleOutcome.FOLLOWED]

        if len(violations) < min_events_per_class or len(controls) < min_events_per_class:
            continue

        r_result: dict = {
            "n_violations": len(violations),
            "n_controls": len(controls),
            "by_metric": {},
        }

        for metric_key, metric_arr in metric_arrays.items():
            auroc_curve = compute_auroc_curve(
                violations, controls, metric_arr, r_val, min_per_class=2
            )
            # Find peak AUROC
            valid_aurocs = auroc_curve[~np.isnan(auroc_curve)]
            if len(valid_aurocs) > 0:
                peak_auroc = float(np.nanmax(auroc_curve))
                peak_lookback = int(np.nanargmax(auroc_curve)) + 1  # 1-indexed
            else:
                peak_auroc = float("nan")
                peak_lookback = -1

            r_result["by_metric"][metric_key] = {
                "auroc_by_lookback": [
                    float(v) if np.isfinite(v) else None
                    for v in auroc_curve
                ],
                "peak_auroc": peak_auroc,
                "peak_lookback": peak_lookback,
            }

        result["by_r_value"][str(r_val)] = r_result

    return result
