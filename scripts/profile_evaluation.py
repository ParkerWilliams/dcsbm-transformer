#!/usr/bin/env python3
"""Profile the evaluation pipeline to diagnose performance bottlenecks.

Measures and reports on every stage of the post-training evaluation pipeline
to diagnose long (17+ hour) evaluation runs. Creates a structured performance
report with per-stage timings, GPU utilization, memory usage, and specific
optimization recommendations.

Usage:
    python scripts/profile_evaluation.py --config config.json
    python scripts/profile_evaluation.py --config config.json --n-sequences 10 --skip-analysis
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def sync_cuda(device):
    """Synchronize CUDA if on GPU, no-op otherwise."""
    try:
        import torch
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass


def fmt_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f}us"
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 3600:
        return f"{seconds:.2f}s"
    hours = seconds / 3600
    return f"{hours:.2f}h"


def fmt_bytes(n_bytes: int) -> str:
    """Format bytes into human-readable string."""
    if n_bytes < 1024:
        return f"{n_bytes}B"
    if n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.1f}KB"
    if n_bytes < 1024 ** 3:
        return f"{n_bytes / (1024 ** 2):.1f}MB"
    return f"{n_bytes / (1024 ** 3):.2f}GB"


class Timer:
    """Simple context-manager timer using monotonic clock."""

    def __init__(self, device=None):
        self.device = device
        self.elapsed = 0.0

    def __enter__(self):
        sync_cuda(self.device)
        self._start = time.monotonic()
        return self

    def __exit__(self, *args):
        sync_cuda(self.device)
        self.elapsed = time.monotonic() - self._start


# ---------------------------------------------------------------------------
# Section 1: Setup
# ---------------------------------------------------------------------------

def setup_fixtures(config_path: str, n_sequences: int, seed: int = 42):
    """Set up graph, walks, and model fixtures for profiling.

    Returns:
        Tuple of (config, graph_data, jumpers, eval_walks, model, device)
    """
    import numpy as np
    import torch

    from src.config import config_from_json
    from src.graph import generate_or_load_graph
    from src.model import create_model
    from src.reproducibility import set_seed
    from src.walk import generate_or_load_walks

    print("Setting up profiling fixtures...")

    # Load config
    json_str = Path(config_path).read_text()
    config = config_from_json(json_str)

    # Set seed for reproducibility
    set_seed(seed)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate graph
    print(f"  Generating graph (n={config.graph.n}, K={config.graph.K})...")
    graph_data, jumpers = generate_or_load_graph(config)

    # Generate walks
    print(f"  Generating walks...")
    _train_result, eval_result = generate_or_load_walks(graph_data, jumpers, config)
    eval_walks = eval_result.walks

    # Subsample to n_sequences
    if eval_walks.shape[0] > n_sequences:
        rng = np.random.default_rng(seed)
        indices = rng.choice(eval_walks.shape[0], n_sequences, replace=False)
        eval_walks = eval_walks[indices]
    print(f"  Eval walks: {eval_walks.shape[0]} sequences, length {eval_walks.shape[1]}")

    # Create model (randomly initialized)
    print(f"  Creating model (d_model={config.model.d_model}, "
          f"n_layers={config.model.n_layers}, n_heads={config.model.n_heads})...")
    model = create_model(config)
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters on {device}")

    return config, graph_data, jumpers, eval_walks, model, device


# ---------------------------------------------------------------------------
# Section 2: Profile fused_evaluate
# ---------------------------------------------------------------------------

def profile_fused_evaluate(config, graph_data, jumpers, eval_walks, model, device):
    """Profile the fused_evaluate function and its sub-components.

    Returns:
        Dict with timing results.
    """
    import numpy as np
    import torch

    from src.config.experiment import ExperimentConfig
    from src.evaluation.behavioral import classify_steps
    from src.evaluation.pipeline import fused_evaluate
    from src.evaluation.svd_metrics import compute_all_metrics, guard_matrix_for_svd
    from src.graph.types import GraphData
    from src.model.types import ExtractionMode

    results = {}
    w = config.training.w
    n_sequences = eval_walks.shape[0]
    batch_size = 32
    max_steps = 4 * w

    print(f"\n--- Profiling fused_evaluate ---")
    print(f"  Sequences: {n_sequences}, batch_size: {batch_size}, max_steps: {max_steps}")

    # 2a. Full fused_evaluate call
    print("  Running full fused_evaluate...")
    with Timer(device) as t_full:
        try:
            eval_result = fused_evaluate(
                model=model,
                eval_walks=eval_walks,
                graph_data=graph_data,
                jumpers=jumpers,
                config=config,
                device=device,
                batch_size=batch_size,
            )
        except Exception as e:
            print(f"  WARNING: fused_evaluate failed: {e}")
            eval_result = None
    results["fused_evaluate_total"] = t_full.elapsed
    print(f"  fused_evaluate total: {fmt_time(t_full.elapsed)}")

    # 2b. Model forward pass micro-benchmark
    print("  Micro-benchmarking model forward pass...")
    n_forward_iters = 10
    forward_times = []
    try:
        # Create a dummy batch
        dummy_input = torch.randint(0, config.graph.n, (batch_size, w), device=device)
        with torch.no_grad():
            for i in range(n_forward_iters):
                with Timer(device) as t_fwd:
                    output = model(dummy_input, mode=ExtractionMode.SVD_TARGETS)
                forward_times.append(t_fwd.elapsed)
        fwd_mean = np.mean(forward_times)
        fwd_std = np.std(forward_times)
        results["forward_pass_mean"] = fwd_mean
        results["forward_pass_std"] = fwd_std
        print(f"  Forward pass (SVD_TARGETS): {fwd_mean*1000:.2f} +/- {fwd_std*1000:.2f} ms")
    except Exception as e:
        print(f"  WARNING: Forward pass benchmark failed: {e}")
        results["forward_pass_mean"] = float("nan")
        results["forward_pass_std"] = float("nan")

    # 2c. SVD computation micro-benchmark
    print("  Micro-benchmarking SVD computation...")
    n_svd_iters = 100
    svd_gpu_times = []
    svd_cpu_times = []
    try:
        # Create a representative QK^T matrix
        qkt_matrix = torch.randn(batch_size, w, w, device=device)
        qkt_clean, _ = guard_matrix_for_svd(qkt_matrix)

        # GPU SVD timing
        for i in range(n_svd_iters):
            with Timer(device) as t_svd:
                U, S, Vh = torch.linalg.svd(qkt_clean, full_matrices=False)
            svd_gpu_times.append(t_svd.elapsed)

        svd_gpu_mean = np.mean(svd_gpu_times)
        svd_gpu_std = np.std(svd_gpu_times)
        results["svd_gpu_mean"] = svd_gpu_mean
        results["svd_gpu_std"] = svd_gpu_std
        print(f"  SVD ({device}): {svd_gpu_mean*1000:.3f} +/- {svd_gpu_std*1000:.3f} ms "
              f"(batch={batch_size}, size={w}x{w})")

        # CPU SVD timing for comparison
        qkt_cpu = qkt_clean.cpu()
        for i in range(min(n_svd_iters, 20)):
            with Timer(None) as t_svd_cpu:
                U_cpu, S_cpu, Vh_cpu = torch.linalg.svd(qkt_cpu, full_matrices=False)
            svd_cpu_times.append(t_svd_cpu.elapsed)

        svd_cpu_mean = np.mean(svd_cpu_times)
        svd_cpu_std = np.std(svd_cpu_times)
        results["svd_cpu_mean"] = svd_cpu_mean
        results["svd_cpu_std"] = svd_cpu_std
        print(f"  SVD (cpu): {svd_cpu_mean*1000:.3f} +/- {svd_cpu_std*1000:.3f} ms "
              f"(batch={batch_size}, size={w}x{w})")

        svd_device = str(qkt_clean.device)
        results["svd_device"] = svd_device
    except Exception as e:
        print(f"  WARNING: SVD benchmark failed: {e}")
        results["svd_gpu_mean"] = float("nan")
        results["svd_cpu_mean"] = float("nan")

    # 2d. WvWo pre-computation timing
    print("  Timing WvWo pre-computation...")
    try:
        with Timer(device) as t_wvwo:
            wvwo = model.get_wvwo()  # [n_layers, n_heads, d_model, d_model]
            # SVD on each layer/head
            n_layers = config.model.n_layers
            n_heads = config.model.n_heads
            for layer_idx in range(n_layers):
                for head_idx in range(n_heads):
                    wvwo_m = wvwo[layer_idx, head_idx]
                    wvwo_clean, _ = guard_matrix_for_svd(wvwo_m)
                    U, S, Vh = torch.linalg.svd(wvwo_clean, full_matrices=False)
                    _ = compute_all_metrics(S, U=U, Vh=Vh)
        results["wvwo_precompute"] = t_wvwo.elapsed
        print(f"  WvWo pre-compute ({n_layers} layers x {n_heads} heads): {fmt_time(t_wvwo.elapsed)}")
    except Exception as e:
        print(f"  WARNING: WvWo benchmark failed: {e}")
        results["wvwo_precompute"] = float("nan")

    # 2e. Behavioral classification timing
    print("  Timing behavioral classification...")
    try:
        # Create some generated sequences for classify_steps
        dummy_generated = torch.randint(0, config.graph.n, (batch_size, max_steps), device=device)
        jumper_map = {j.vertex_id: j for j in jumpers}
        with Timer(device) as t_classify:
            edge_valid, rule_outcome, failure_index = classify_steps(
                dummy_generated, graph_data, jumper_map
            )
        results["classify_steps"] = t_classify.elapsed
        print(f"  classify_steps ({batch_size} seq, {max_steps} steps): {fmt_time(t_classify.elapsed)}")
    except Exception as e:
        print(f"  WARNING: classify_steps benchmark failed: {e}")
        results["classify_steps"] = float("nan")

    # 2f. Per-element metric storage loop vs vectorized benchmark
    print("  Benchmarking metric storage: per-element vs vectorized...")
    try:
        n_metric_keys = 10  # Representative number of metric keys per step
        fake_arrays = {f"metric_{i}": np.full((n_sequences, max_steps), np.nan, dtype=np.float32)
                       for i in range(n_metric_keys)}
        fake_vals = np.random.randn(batch_size).astype(np.float32)
        test_step = w  # representative step index

        # Per-element loop (the current approach in pipeline.py)
        n_storage_iters = 100
        loop_times = []
        for _ in range(n_storage_iters):
            t0 = time.monotonic()
            for key in fake_arrays:
                for b_idx in range(batch_size):
                    fake_arrays[key][b_idx, test_step] = fake_vals[b_idx]
            loop_times.append(time.monotonic() - t0)

        loop_mean = np.mean(loop_times)
        results["storage_loop_mean"] = loop_mean

        # Vectorized slice assignment
        vec_times = []
        for _ in range(n_storage_iters):
            t0 = time.monotonic()
            for key in fake_arrays:
                fake_arrays[key][0:batch_size, test_step] = fake_vals
            vec_times.append(time.monotonic() - t0)

        vec_mean = np.mean(vec_times)
        results["storage_vec_mean"] = vec_mean

        speedup = loop_mean / vec_mean if vec_mean > 0 else float("inf")
        results["storage_speedup"] = speedup
        print(f"  Per-element loop: {loop_mean*1e6:.1f}us/iter, "
              f"Vectorized: {vec_mean*1e6:.1f}us/iter, "
              f"Speedup: {speedup:.1f}x")
    except Exception as e:
        print(f"  WARNING: Storage benchmark failed: {e}")
        results["storage_loop_mean"] = float("nan")
        results["storage_vec_mean"] = float("nan")

    return results, eval_result


# ---------------------------------------------------------------------------
# Section 3: Profile AUROC analysis
# ---------------------------------------------------------------------------

def profile_auroc_analysis(eval_result, jumpers, config, n_shuffle=100):
    """Profile AUROC analysis with reduced shuffle count.

    Returns:
        Dict with timing results.
    """
    import numpy as np

    from src.analysis.auroc_horizon import run_auroc_analysis, run_shuffle_control
    from src.analysis.event_extraction import extract_events, filter_contaminated_events, stratify_by_r
    from src.evaluation.behavioral import RuleOutcome
    from src.graph.jumpers import JumperInfo

    results = {}
    print(f"\n--- Profiling AUROC Analysis (n_shuffle={n_shuffle}) ---")

    if eval_result is None:
        print("  SKIPPED: fused_evaluate did not produce results")
        return results

    try:
        # Build eval_result_data dict
        eval_result_data = {
            "generated": eval_result.generated,
            "rule_outcome": eval_result.rule_outcome,
            "failure_index": eval_result.failure_index,
            "sequence_lengths": eval_result.sequence_lengths,
        }
        eval_result_data.update(eval_result.svd_metrics)

        # Determine metric keys
        metric_keys = [
            k for k in eval_result.svd_metrics.keys()
            if "." in k and k.split(".")[0] in ("qkt", "avwo", "wvwo")
        ]
        jumper_map = {j.vertex_id: j for j in jumpers}

        print(f"  Metric keys: {len(metric_keys)}")

        # Full AUROC analysis
        with Timer() as t_auroc:
            auroc_results = run_auroc_analysis(
                eval_result_data=eval_result_data,
                jumper_map=jumper_map,
                metric_keys=metric_keys,
                n_shuffle=n_shuffle,
            )
        results["auroc_total"] = t_auroc.elapsed
        results["auroc_n_shuffle"] = n_shuffle
        print(f"  run_auroc_analysis ({n_shuffle} shuffles): {fmt_time(t_auroc.elapsed)}")

        # Extrapolate to 10,000 shuffles
        scale_factor = 10_000 / n_shuffle
        est_full = t_auroc.elapsed * scale_factor
        results["auroc_est_full"] = est_full
        print(f"  Estimated at 10,000 shuffles: {fmt_time(est_full)}")

        # Profile single shuffle control call
        all_events = extract_events(
            eval_result.generated, eval_result.rule_outcome,
            eval_result.failure_index, jumper_map
        )
        filtered, _ = filter_contaminated_events(all_events)
        by_r = stratify_by_r(filtered)

        if by_r:
            r_val = list(by_r.keys())[0]
            r_events = by_r[r_val]
            violations = [e for e in r_events if e.outcome == RuleOutcome.VIOLATED]
            controls = [e for e in r_events if e.outcome == RuleOutcome.FOLLOWED]

            if len(violations) >= 2 and len(controls) >= 2 and metric_keys:
                first_metric_key = metric_keys[0]
                metric_arr = eval_result.svd_metrics.get(first_metric_key)
                if metric_arr is not None:
                    with Timer() as t_single_shuffle:
                        run_shuffle_control(
                            violations, controls, metric_arr, r_val,
                            n_permutations=100,
                        )
                    results["single_shuffle_control"] = t_single_shuffle.elapsed
                    print(f"  Single shuffle_control (100 perms, r={r_val}): "
                          f"{fmt_time(t_single_shuffle.elapsed)}")

    except Exception as e:
        print(f"  WARNING: AUROC profiling failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Section 4: Profile statistical controls
# ---------------------------------------------------------------------------

def profile_statistical_controls(eval_result, jumpers, n_bootstrap=100):
    """Profile statistical controls with reduced bootstrap count.

    Returns:
        Dict with timing results.
    """
    from src.analysis.statistical_controls import apply_statistical_controls

    results = {}
    print(f"\n--- Profiling Statistical Controls (n_bootstrap={n_bootstrap}) ---")

    if eval_result is None:
        print("  SKIPPED: fused_evaluate did not produce results")
        return results

    try:
        # Build eval_data dict
        eval_data = {
            "generated": eval_result.generated,
            "rule_outcome": eval_result.rule_outcome,
            "failure_index": eval_result.failure_index,
            "sequence_lengths": eval_result.sequence_lengths,
        }
        eval_data.update(eval_result.svd_metrics)

        jumper_map = {j.vertex_id: j for j in jumpers}

        with Timer() as t_stats:
            apply_statistical_controls(
                auroc_results=None,
                eval_data=eval_data,
                jumper_map=jumper_map,
                n_bootstrap=n_bootstrap,
            )
        results["stats_total"] = t_stats.elapsed
        results["stats_n_bootstrap"] = n_bootstrap
        print(f"  apply_statistical_controls ({n_bootstrap} bootstrap): "
              f"{fmt_time(t_stats.elapsed)}")

        # Extrapolate to 10,000 bootstrap
        scale_factor = 10_000 / n_bootstrap
        est_full = t_stats.elapsed * scale_factor
        results["stats_est_full"] = est_full
        print(f"  Estimated at 10,000 bootstrap: {fmt_time(est_full)}")

    except Exception as e:
        print(f"  WARNING: Statistical controls profiling failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Section 5: Hardware utilization checks
# ---------------------------------------------------------------------------

def check_hardware(model, device, config):
    """Check and report hardware utilization.

    Returns:
        Dict with hardware info.
    """
    import torch

    from src.model.types import ExtractionMode

    results = {}
    print(f"\n--- Hardware Utilization ---")

    # Basic device info
    results["device"] = str(device)
    results["cpu_cores"] = os.cpu_count()
    print(f"  Device: {device}")
    print(f"  CPU cores: {os.cpu_count()}")

    results["cuda_available"] = torch.cuda.is_available()

    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(device)
            results["gpu_name"] = props.name
            results["gpu_memory_total"] = props.total_mem
            results["gpu_memory_allocated"] = torch.cuda.memory_allocated(device)
            results["gpu_memory_reserved"] = torch.cuda.memory_reserved(device)

            print(f"  GPU: {props.name}")
            print(f"  GPU Memory Total: {fmt_bytes(props.total_mem)}")
            print(f"  GPU Memory Allocated: {fmt_bytes(torch.cuda.memory_allocated(device))}")
            print(f"  GPU Memory Reserved: {fmt_bytes(torch.cuda.memory_reserved(device))}")

            # Peak memory during forward pass
            torch.cuda.reset_peak_memory_stats(device)
            w = config.training.w
            dummy = torch.randint(0, config.graph.n, (32, w), device=device)
            with torch.no_grad():
                _ = model(dummy, mode=ExtractionMode.SVD_TARGETS)
            peak_mem = torch.cuda.max_memory_allocated(device)
            results["peak_memory_forward"] = peak_mem
            print(f"  Peak GPU Memory (forward pass): {fmt_bytes(peak_mem)}")

            # GPU utilization
            try:
                util = torch.cuda.utilization(device)
                results["gpu_utilization"] = util
                print(f"  GPU Utilization: {util}%")
            except Exception:
                print("  GPU Utilization: unavailable (try nvidia-smi)")
                results["gpu_utilization"] = None

        except Exception as e:
            print(f"  WARNING: GPU info collection failed: {e}")
    else:
        print("  No GPU available -- all computation on CPU")
        print("  Note: SVD and forward passes will be significantly slower on CPU")

    # Check for CPU tensors that should be on GPU
    if torch.cuda.is_available():
        cpu_params = []
        for name, param in model.named_parameters():
            if param.device.type == "cpu":
                cpu_params.append(name)
        if cpu_params:
            print(f"  WARNING: {len(cpu_params)} parameters on CPU (should be on GPU):")
            for p in cpu_params[:5]:
                print(f"    - {p}")
            results["cpu_params_count"] = len(cpu_params)
        else:
            print("  All model parameters on GPU")
            results["cpu_params_count"] = 0

    return results


# ---------------------------------------------------------------------------
# Section 6: Optimization opportunity analysis
# ---------------------------------------------------------------------------

def analyze_optimizations(eval_timings, auroc_timings, stats_timings, hw_info, config):
    """Analyze timing results and flag optimization opportunities.

    Returns:
        List of optimization recommendations.
    """
    import numpy as np

    recommendations = []
    w = config.training.w
    n_layers = config.model.n_layers
    n_heads = config.model.n_heads
    default_n_sequences = 10_000  # Full-scale default
    batch_size = 32
    n_batches = (default_n_sequences + batch_size - 1) // batch_size
    max_steps = 4 * w
    svd_steps = max_steps - w  # Steps where SVD is computed (step >= w)

    # Estimate total full-scale time components
    fwd_per_batch = eval_timings.get("forward_pass_mean", 0)
    svd_per_call = eval_timings.get("svd_gpu_mean", eval_timings.get("svd_cpu_mean", 0))

    # Number of SVD calls per step per batch: 2 targets (qkt, avwo) x n_layers x n_heads
    # Each target does a full SVD + compute_all_metrics
    svd_calls_per_step = 2 * n_layers * n_heads
    total_svd_calls = svd_calls_per_step * svd_steps * n_batches

    fwd_total_est = fwd_per_batch * max_steps * n_batches
    svd_total_est = svd_per_call * total_svd_calls

    # Storage loop cost (per-element Python loops)
    storage_loop_cost = eval_timings.get("storage_loop_mean", 0)
    storage_vec_cost = eval_timings.get("storage_vec_mean", 0)
    storage_speedup = eval_timings.get("storage_speedup", 1.0)

    # Number of metric keys: ~9 metrics x 3 targets x n_layers x n_heads
    n_metric_keys_approx = 9 * 3 * n_layers * n_heads
    # Each storage iteration loops over batch_size elements per key per step
    storage_total_est = storage_loop_cost * svd_steps * n_batches

    classify_per_batch = eval_timings.get("classify_steps", 0)
    classify_total_est = classify_per_batch * n_batches

    # Total estimated fused_evaluate time at full scale
    total_est = fwd_total_est + svd_total_est + storage_total_est + classify_total_est

    # AUROC extrapolation
    auroc_est = auroc_timings.get("auroc_est_full", 0)
    stats_est = stats_timings.get("stats_est_full", 0)

    grand_total_est = total_est + auroc_est + stats_est

    # --- Build recommendations ---

    # 1. Per-element Python loops
    if not np.isnan(storage_loop_cost) and not np.isnan(storage_vec_cost) and storage_speedup > 1.5:
        pct_of_total = (storage_total_est / grand_total_est * 100) if grand_total_est > 0 else 0
        recommendations.append({
            "title": "Vectorize per-element metric storage loops",
            "description": (
                "The `for b_idx in range(B_actual)` loops in pipeline.py (~lines 349-362, "
                "419-430, 457-470) copy SVD metrics one element at a time. Replace with "
                "vectorized slice: svd_metric_arrays[key][batch_start:batch_end, step] = vals_np"
            ),
            "est_pct_of_total": round(pct_of_total, 1),
            "est_speedup": f"{storage_speedup:.0f}x on storage, saves ~{fmt_time(storage_total_est * (1 - 1/storage_speedup))} at full scale",
        })

    # 2. Sequential SVD calls
    svd_pct = (svd_total_est / grand_total_est * 100) if grand_total_est > 0 else 0
    if svd_per_call > 0:
        recommendations.append({
            "title": "Batch or parallelize SVD decompositions",
            "description": (
                f"{svd_calls_per_step} SVD calls per step "
                f"({2} targets x {n_layers} layers x {n_heads} heads). "
                "Consider batching QK^T matrices across layers into a single batched SVD call, "
                "or using torch.linalg.svd on a stacked tensor."
            ),
            "est_pct_of_total": round(svd_pct, 1),
            "est_speedup": "2-3x potential with batched SVD (reduces kernel launch overhead)",
        })

    # 3. CPU-GPU transfers
    recommendations.append({
        "title": "Defer .cpu().numpy() transfers to end of batch",
        "description": (
            "Multiple .cpu().numpy() calls inside the per-step inner loop force "
            "GPU synchronization. Accumulate results in GPU tensors per batch, "
            "then transfer to CPU once at the end of each batch."
        ),
        "est_pct_of_total": "5-15% (synchronization overhead)",
        "est_speedup": "1.2-1.5x on fused_evaluate (reduces sync stalls)",
    })

    # 4. Shuffle control scale
    if auroc_est > 0:
        auroc_pct = (auroc_est / grand_total_est * 100) if grand_total_est > 0 else 0
        recommendations.append({
            "title": "Reduce or parallelize shuffle controls",
            "description": (
                f"10,000 permutations x N metrics x N r-values. "
                f"Estimated {fmt_time(auroc_est)} at full scale ({auroc_pct:.0f}% of total). "
                "Consider: (1) reducing to 1,000 permutations with calibrated p-value correction, "
                "(2) parallelizing with multiprocessing, or (3) vectorizing the permutation loop."
            ),
            "est_pct_of_total": round(auroc_pct, 1),
            "est_speedup": "10x with 1000 perms, 4-8x with multiprocessing",
        })

    # 5. classify_steps pure Python
    if classify_per_batch > 0:
        classify_pct = (classify_total_est / grand_total_est * 100) if grand_total_est > 0 else 0
        recommendations.append({
            "title": "Vectorize classify_steps with numpy",
            "description": (
                "classify_steps uses nested Python loops (for b in range(B), for t in range(L-1)). "
                "The CSR adjacency lookup and jumper tracking can be vectorized with scipy.sparse "
                "indexing and numpy operations."
            ),
            "est_pct_of_total": round(classify_pct, 1),
            "est_speedup": "5-20x with numpy vectorization",
        })

    return recommendations, {
        "fwd_total_est": fwd_total_est,
        "svd_total_est": svd_total_est,
        "storage_total_est": storage_total_est,
        "classify_total_est": classify_total_est,
        "auroc_est": auroc_est,
        "stats_est": stats_est,
        "grand_total_est": grand_total_est,
    }


# ---------------------------------------------------------------------------
# Section 7: Report generation
# ---------------------------------------------------------------------------

def print_report(eval_timings, auroc_timings, stats_timings, hw_info, recommendations,
                 full_scale_estimates, config, n_sequences, skip_analysis):
    """Print the structured performance report."""
    import numpy as np

    w = config.training.w
    n_layers = config.model.n_layers
    n_heads = config.model.n_heads

    print("\n" + "=" * 70)
    print("=== EVALUATION PIPELINE PERFORMANCE PROFILE ===")
    print("=" * 70)

    # --- Hardware ---
    print("\n--- Hardware ---")
    print(f"Device: {hw_info.get('device', 'unknown')}", end="")
    if hw_info.get('gpu_name'):
        print(f" ({hw_info['gpu_name']})")
    else:
        print()
    if hw_info.get('gpu_memory_total'):
        print(f"GPU Memory: {fmt_bytes(hw_info['gpu_memory_total'])} total, "
              f"{fmt_bytes(hw_info.get('gpu_memory_allocated', 0))} allocated, "
              f"{fmt_bytes(hw_info.get('gpu_memory_reserved', 0))} reserved")
    if hw_info.get('peak_memory_forward'):
        print(f"Peak GPU Memory (forward): {fmt_bytes(hw_info['peak_memory_forward'])}")
    if hw_info.get('gpu_utilization') is not None:
        print(f"GPU Utilization: {hw_info['gpu_utilization']}%")
    print(f"CPU Cores: {hw_info.get('cpu_cores', 'unknown')}")

    # --- Stage Timings ---
    print(f"\n--- Stage Timings (measured at reduced scale: {n_sequences} sequences) ---")
    print(f"{'Stage':<42} {'Time':>10}  {'Notes'}")
    print("-" * 70)

    fused_total = eval_timings.get("fused_evaluate_total", float("nan"))
    print(f"{'fused_evaluate':<42} {fmt_time(fused_total):>10}")

    fwd_mean = eval_timings.get("forward_pass_mean", float("nan"))
    fwd_std = eval_timings.get("forward_pass_std", float("nan"))
    if not np.isnan(fwd_mean):
        print(f"{'  model forward (per batch, SVD mode)':<42} {fmt_time(fwd_mean):>10}  "
              f"+/- {fmt_time(fwd_std)}")

    svd_mean = eval_timings.get("svd_gpu_mean", eval_timings.get("svd_cpu_mean", float("nan")))
    if not np.isnan(svd_mean):
        svd_device = eval_timings.get("svd_device", "?")
        print(f"{'  SVD per call (batch, WxW)':<42} {fmt_time(svd_mean):>10}  "
              f"on {svd_device}")

    svd_cpu = eval_timings.get("svd_cpu_mean", float("nan"))
    if not np.isnan(svd_cpu) and not np.isnan(svd_mean) and abs(svd_cpu - svd_mean) > 1e-6:
        print(f"{'  SVD per call (CPU comparison)':<42} {fmt_time(svd_cpu):>10}")

    wvwo = eval_timings.get("wvwo_precompute", float("nan"))
    if not np.isnan(wvwo):
        print(f"{'  WvWo pre-compute':<42} {fmt_time(wvwo):>10}  "
              f"{n_layers}L x {n_heads}H")

    classify = eval_timings.get("classify_steps", float("nan"))
    if not np.isnan(classify):
        print(f"{'  behavioral classify (per batch)':<42} {fmt_time(classify):>10}")

    loop_mean = eval_timings.get("storage_loop_mean", float("nan"))
    vec_mean = eval_timings.get("storage_vec_mean", float("nan"))
    if not np.isnan(loop_mean):
        speedup = eval_timings.get("storage_speedup", 1.0)
        print(f"{'  metric storage (per-element loop)':<42} {fmt_time(loop_mean):>10}  "
              f"vs {fmt_time(vec_mean)} vectorized ({speedup:.0f}x)")

    if not skip_analysis:
        auroc_total = auroc_timings.get("auroc_total", float("nan"))
        if not np.isnan(auroc_total):
            n_shuf = auroc_timings.get("auroc_n_shuffle", "?")
            print(f"{'AUROC analysis':<42} {fmt_time(auroc_total):>10}  "
                  f"({n_shuf} shuffles)")

        stats_total = stats_timings.get("stats_total", float("nan"))
        if not np.isnan(stats_total):
            n_bs = stats_timings.get("stats_n_bootstrap", "?")
            print(f"{'Statistical controls':<42} {fmt_time(stats_total):>10}  "
                  f"({n_bs} bootstrap)")

    # --- Estimated Full-Scale ---
    print(f"\n--- Estimated Full-Scale Timings (10,000 sequences, 10,000 shuffles/bootstrap) ---")
    print(f"{'Stage':<42} {'Est. Time':>12}  {'% of Total':>10}")
    print("-" * 70)

    est = full_scale_estimates
    grand = est.get("grand_total_est", 1)

    def pct(val):
        return f"{val / grand * 100:.1f}%" if grand > 0 and not np.isnan(val) else "?"

    fwd_est = est.get("fwd_total_est", float("nan"))
    svd_est = est.get("svd_total_est", float("nan"))
    storage_est = est.get("storage_total_est", float("nan"))
    classify_est = est.get("classify_total_est", float("nan"))

    fused_est_total = fwd_est + svd_est + storage_est + classify_est
    print(f"{'fused_evaluate':<42} {fmt_time(fused_est_total):>12}  {pct(fused_est_total):>10}")
    print(f"{'  model forward passes':<42} {fmt_time(fwd_est):>12}  {pct(fwd_est):>10}")
    print(f"{'  SVD computations':<42} {fmt_time(svd_est):>12}  {pct(svd_est):>10}")
    print(f"{'  metric storage loops':<42} {fmt_time(storage_est):>12}  {pct(storage_est):>10}")
    print(f"{'  behavioral classification':<42} {fmt_time(classify_est):>12}  {pct(classify_est):>10}")

    if not skip_analysis:
        auroc_est = est.get("auroc_est", float("nan"))
        stats_est = est.get("stats_est", float("nan"))
        if not np.isnan(auroc_est):
            print(f"{'AUROC analysis (10k shuffles)':<42} {fmt_time(auroc_est):>12}  {pct(auroc_est):>10}")
        if not np.isnan(stats_est):
            print(f"{'Statistical controls (10k bootstrap)':<42} {fmt_time(stats_est):>12}  {pct(stats_est):>10}")

    print("-" * 70)
    print(f"{'TOTAL ESTIMATED':<42} {fmt_time(grand):>12}")

    # --- GPU Utilization Summary ---
    print(f"\n--- GPU Utilization ---")
    svd_device = eval_timings.get("svd_device", "unknown")
    print(f"SVD running on: {svd_device}")
    if hw_info.get("peak_memory_forward"):
        print(f"Peak GPU memory during forward: {fmt_bytes(hw_info['peak_memory_forward'])}")
    if hw_info.get("cpu_params_count", 0) > 0:
        print(f"WARNING: {hw_info['cpu_params_count']} model parameters on CPU!")
    elif hw_info.get("cuda_available", False):
        print("All model parameters on GPU")
    else:
        print("No GPU available -- all operations on CPU")

    # --- Top Bottlenecks ---
    print(f"\n--- Top Bottlenecks ---")
    for i, rec in enumerate(recommendations, 1):
        pct_str = rec.get("est_pct_of_total", "?")
        if isinstance(pct_str, (int, float)):
            pct_str = f"{pct_str:.0f}%"
        print(f"{i}. {rec['title']}")
        print(f"   Est. {pct_str} of total, potential speedup: {rec.get('est_speedup', '?')}")

    # --- Recommendations ---
    print(f"\n--- Recommendations ---")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']}")
        print(f"   {rec['description']}")
        print()

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile the evaluation pipeline to diagnose performance bottlenecks."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to experiment config JSON file (default: config.json)",
    )
    parser.add_argument(
        "--n-sequences",
        type=int,
        default=50,
        help="Number of evaluation sequences for profiling (default: 50)",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip AUROC and statistical controls profiling (just profile fused_evaluate)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"Evaluation Pipeline Profiler")
    print(f"Config: {args.config}")
    print(f"Sequences: {args.n_sequences}")
    print(f"Skip analysis: {args.skip_analysis}")
    print(f"Seed: {args.seed}")
    print()

    overall_start = time.monotonic()

    # Section 1: Setup
    try:
        config, graph_data, jumpers, eval_walks, model, device = setup_fixtures(
            config_path=args.config,
            n_sequences=args.n_sequences,
            seed=args.seed,
        )
    except Exception as e:
        print(f"\nFATAL: Setup failed: {e}")
        print("Ensure all dependencies are installed and config.json is valid.")
        sys.exit(1)

    # Section 5: Hardware checks (run early to get device info)
    try:
        hw_info = check_hardware(model, device, config)
    except Exception as e:
        print(f"WARNING: Hardware check failed: {e}")
        hw_info = {"device": str(device), "cpu_cores": os.cpu_count()}

    # Section 2: Profile fused_evaluate
    try:
        eval_timings, eval_result = profile_fused_evaluate(
            config, graph_data, jumpers, eval_walks, model, device
        )
    except Exception as e:
        print(f"WARNING: fused_evaluate profiling failed: {e}")
        eval_timings = {}
        eval_result = None

    # Sections 3 & 4: AUROC and Statistical Controls
    auroc_timings = {}
    stats_timings = {}
    if not args.skip_analysis:
        try:
            auroc_timings = profile_auroc_analysis(
                eval_result, jumpers, config, n_shuffle=100
            )
        except Exception as e:
            print(f"WARNING: AUROC profiling failed: {e}")

        try:
            stats_timings = profile_statistical_controls(
                eval_result, jumpers, n_bootstrap=100
            )
        except Exception as e:
            print(f"WARNING: Statistical controls profiling failed: {e}")

    # Section 6: Optimization analysis
    try:
        recommendations, full_scale_estimates = analyze_optimizations(
            eval_timings, auroc_timings, stats_timings, hw_info, config
        )
    except Exception as e:
        print(f"WARNING: Optimization analysis failed: {e}")
        recommendations = []
        full_scale_estimates = {}

    # Section 7: Print report
    print_report(
        eval_timings, auroc_timings, stats_timings, hw_info,
        recommendations, full_scale_estimates,
        config, args.n_sequences, args.skip_analysis,
    )

    overall_elapsed = time.monotonic() - overall_start
    print(f"\nProfiling completed in {fmt_time(overall_elapsed)}")


if __name__ == "__main__":
    main()
