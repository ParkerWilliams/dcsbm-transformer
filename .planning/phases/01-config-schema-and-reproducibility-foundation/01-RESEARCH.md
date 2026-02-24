# Phase 1: Config, Schema, and Reproducibility Foundation - Research

**Researched:** 2026-02-24
**Domain:** Python dataclasses, JSON serialization, deterministic hashing, seed management, git integration
**Confidence:** HIGH

## Summary

This phase builds the foundational infrastructure that every subsequent phase depends on: a frozen, hashable experiment configuration system, a validated result.json schema, deterministic seed management across all RNG sources, and automatic git hash capture. The technical landscape is well-established and uses only Python standard library features plus PyTorch/NumPy seed APIs. There are no bleeding-edge dependencies or risky unknowns.

The core pattern is: nested frozen dataclasses with `@dataclass(frozen=True, slots=True)` for immutability and memory efficiency, JSON round-trip via `dataclasses.asdict()` + a custom `from_dict()` reconstruction function (or `dacite` for convenience), deterministic content hashing via `json.dumps(sort_keys=True)` piped through `hashlib.sha256`, and a centralized `set_seed(master_seed)` function that seeds `random`, `numpy`, `torch`, and configures CUDA determinism.

**Primary recommendation:** Use stdlib `dataclasses` with `frozen=True, slots=True` for config objects, `dacite.from_dict()` for deserialization, `hashlib.sha256` over sorted-JSON for config hashing, and a single `set_seed()` function that configures all RNG sources plus CUDA determinism flags.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Nested dataclasses: GraphConfig, ModelConfig, TrainingConfig, SweepConfig (and a top-level ExperimentConfig composing them)
- Typed, validated, IDE-friendly with frozen=True for immutability
- Strict validation with cross-parameter constraints (walk_length >= 2*w, corpus_size >= 100*n, n_heads == 1)
- Config hashing: all params except seed -- graph cache shared across seeds, model/training cache includes seed
- Serialization: JSON format (matches result.json, easy to diff)
- Hybrid JSON + npz: result.json holds summary metrics, scalars, curves, statistical tests, and sequence metadata; token_metrics.npz holds full per-step SVD metric arrays referenced by sequence_id
- Python validation function checks required fields, types, and array length consistency before writing -- not a formal JSON Schema file
- All evaluation walks containing block jumper events get full sequence data in result.json (not sampled)
- Experiment ID format: key params in slug -- e.g., n500_w64_r32_d128_L4_s42_{timestamp} -- scannable at a glance in file listings
- Anchor Config Defaults (LOCKED): n=500, w=64, t=200000, d_model=128, n_layers=4, n_heads=1, r=0.9w=57, walk_length=4w=256, 3 random seeds per config
- Makefile as primary entry point: `make run`, `make sweep`, `make test`, `make pdf`
- Python script (`run_experiment.py`) with argparse for custom invocations
- All Python commands run in venv

### Claude's Discretion
- Exact dataclass field names and type annotations
- Validation error message formatting
- pyproject.toml structure and dependency pinning
- Makefile target organization beyond the required ones
- Internal helper functions for serialization/deserialization

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MGMT-01 | System defines experiment configuration as a frozen, serializable, hashable dataclass with all governing parameters | Frozen dataclass pattern with `slots=True`, JSON serialization via `asdict()`, deterministic hashing via `hashlib.sha256` over sorted JSON. Nested composition of GraphConfig, ModelConfig, TrainingConfig, SweepConfig into ExperimentConfig. |
| MGMT-05 | System writes result.json per configuration conforming to the project schema (schema_version, experiment_id, timestamp, description, tags, config, metrics, sequences, metadata) | Python validation function (not jsonschema), RESULTS_SCHEMA.md defines the contract. Hybrid JSON + npz storage for large arrays. |
| TRNG-02 | System controls all random seeds (torch, numpy, python random, CUDA deterministic) from a single master seed | `set_seed()` function seeding `random.seed()`, `np.random.seed()`, `torch.manual_seed()`, `torch.cuda.manual_seed_all()`, plus `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. |
| TRNG-07 | System tracks git code hash (short SHA) and stores it with results for reproducibility | `subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])` with error handling, stored in result.json metadata block as `code_hash`. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `dataclasses` | stdlib (3.12) | Frozen, typed config objects | Built-in, zero dependencies, full IDE support, `frozen=True` + `slots=True` for immutability and performance |
| `json` | stdlib | Serialization/deserialization | Deterministic with `sort_keys=True`, human-readable diffs, matches result.json format |
| `hashlib` | stdlib | Deterministic config hashing | SHA-256 is stable across Python versions and sessions (unlike `hash()`) |
| `subprocess` | stdlib | Git hash capture | `git rev-parse --short HEAD` for code provenance |
| `dacite` | 1.8+ | Dict-to-dataclass deserialization | Type-safe nested dataclass reconstruction from JSON dicts, handles Optional/Union/List fields |
| `torch` | 2.x | Seed management (GPU) | `manual_seed()`, `cuda.manual_seed_all()`, deterministic algorithm flags |
| `numpy` | 2.x | Seed management + npz storage | `random.seed()` for RNG, `savez_compressed()` for large array storage |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pathlib` | stdlib | Path manipulation | All file path operations for results directories |
| `datetime` | stdlib | ISO 8601 timestamps | `datetime.now(timezone.utc).isoformat()` for result.json |
| `os` | stdlib | Environment variables | Setting `CUBLAS_WORKSPACE_CONFIG` for CUDA determinism |
| `copy` | stdlib | Deep copy for config mutation | `dataclasses.replace()` for creating config variants |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `dacite` | Hand-rolled `from_dict()` | Dacite handles edge cases (Optional, Union, nested lists) automatically; hand-rolled is zero-dependency but more code and more bugs |
| `dacite` | `pydantic` | Pydantic is heavier, adds runtime validation overhead, and this project uses frozen dataclasses (not Pydantic models) per user decision |
| `json` + `hashlib` | `hash()` built-in | Python's `hash()` is randomized per session (PYTHONHASHSEED), not stable across runs -- completely unsuitable for caching/reproducibility |
| `json.dumps(sort_keys=True)` | `pickle` | Pickle is not binary-stable across Python versions; JSON is human-readable and diffable |

**Installation:**
```bash
pip install dacite torch numpy
```

## Architecture Patterns

### Recommended Project Structure
```
src/
    config/
        __init__.py
        experiment.py      # ExperimentConfig, GraphConfig, ModelConfig, TrainingConfig, SweepConfig
        defaults.py        # ANCHOR_CONFIG with locked defaults
        hashing.py         # config_hash() using JSON + SHA-256
        serialization.py   # to_json(), from_json() round-trip functions
    results/
        __init__.py
        schema.py          # validate_result(), write_result(), ResultWriter
        experiment_id.py   # generate_experiment_id() with param slug + timestamp
    reproducibility/
        __init__.py
        seed.py            # set_seed(), get_rng_state(), verify_seed_determinism()
        git_hash.py        # get_git_hash() with error handling
```

### Pattern 1: Nested Frozen Dataclass Composition
**What:** Top-level ExperimentConfig composed of sub-configs, all frozen and slotted
**When to use:** Always -- this is the locked decision for config structure
**Example:**
```python
# Source: Python stdlib dataclasses + CONTEXT.md decisions
from dataclasses import dataclass, field, asdict
from typing import Optional
import math

@dataclass(frozen=True, slots=True)
class GraphConfig:
    n: int = 500                    # number of vertices
    K: int = 4                      # number of blocks
    p_in: float = 0.25              # in-group edge probability
    p_out: float = 0.03             # out-group edge probability
    n_jumpers_per_block: int = 2    # block jumper vertices per block

@dataclass(frozen=True, slots=True)
class ModelConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 1    # locked: always 1

@dataclass(frozen=True, slots=True)
class TrainingConfig:
    w: int = 64                     # context window
    walk_length: int = 256          # 4 * w
    corpus_size: int = 200_000      # t
    r: int = 57                     # jump length (0.9 * w rounded)
    learning_rate: float = 3e-4
    batch_size: int = 64
    max_steps: int = 50_000

@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    description: str = ""
    tags: tuple[str, ...] = ()      # tuple, not list, for hashability

    def __post_init__(self):
        # Cross-parameter validation (use object.__setattr__ for frozen)
        if self.training.walk_length < 2 * self.training.w:
            raise ValueError(
                f"walk_length ({self.training.walk_length}) must be >= 2 * w ({2 * self.training.w})"
            )
        if self.training.corpus_size < 100 * self.graph.n:
            raise ValueError(
                f"corpus_size ({self.training.corpus_size}) must be >= 100 * n ({100 * self.graph.n})"
            )
        if self.model.n_heads != 1:
            raise ValueError("n_heads must be exactly 1 (single-head constraint)")
```

### Pattern 2: Deterministic Config Hashing (Excluding Seed)
**What:** SHA-256 hash of sorted JSON serialization, with field exclusion for cache sharing
**When to use:** Config identity for caching (graph cache excludes seed, full config includes seed)
**Example:**
```python
# Source: hashlib docs + death.andgravity.com/stable-hashing
import json
import hashlib
from dataclasses import asdict

def config_hash(config, exclude_fields=None):
    """Deterministic SHA-256 hash of config, optionally excluding fields."""
    d = asdict(config)
    if exclude_fields:
        for field_path in exclude_fields:
            # Support dotted paths like "seed" or nested removal
            _remove_nested(d, field_path)
    serialized = json.dumps(
        d,
        sort_keys=True,
        ensure_ascii=True,
        separators=(',', ':'),   # compact, no whitespace variation
        indent=None,
    )
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]

def graph_config_hash(config):
    """Hash for graph caching -- excludes seed and training/model params."""
    return config_hash(config.graph)

def full_config_hash(config):
    """Hash for full experiment identity -- includes everything."""
    return config_hash(config)
```

### Pattern 3: JSON Round-Trip with dacite
**What:** Serialize via `asdict()` + `json.dumps()`, deserialize via `json.loads()` + `dacite.from_dict()`
**When to use:** Config persistence, result.json config block, config loading from files
**Example:**
```python
# Source: dacite GitHub + dataclasses stdlib
import json
from dataclasses import asdict
from dacite import from_dict, Config as DaciteConfig

def config_to_json(config: ExperimentConfig) -> str:
    """Serialize config to JSON string."""
    return json.dumps(asdict(config), indent=2, sort_keys=True)

def config_from_json(json_str: str) -> ExperimentConfig:
    """Deserialize JSON string to ExperimentConfig."""
    data = json.loads(json_str)
    return from_dict(
        data_class=ExperimentConfig,
        data=data,
        config=DaciteConfig(
            cast=[tuple],  # JSON arrays -> tuples for tags field
            check_types=True,
        ),
    )

# Round-trip test:
# original = ExperimentConfig(...)
# json_str = config_to_json(original)
# restored = config_from_json(json_str)
# assert config_hash(original) == config_hash(restored)
```

### Pattern 4: Centralized Seed Management
**What:** Single function that seeds all RNG sources and configures CUDA determinism
**When to use:** At experiment start, before any random operation
**Example:**
```python
# Source: PyTorch reproducibility docs
import os
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Must be called before any random operations (model init, data loading, etc.).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CUDA determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def verify_seed_determinism(seed: int) -> bool:
    """Verify that setting the seed produces identical sequences."""
    set_seed(seed)
    r1 = [random.random() for _ in range(10)]
    n1 = np.random.rand(10).tolist()
    t1 = torch.rand(10).tolist()

    set_seed(seed)
    r2 = [random.random() for _ in range(10)]
    n2 = np.random.rand(10).tolist()
    t2 = torch.rand(10).tolist()

    return r1 == r2 and n1 == n2 and t1 == t2
```

### Pattern 5: Result Schema Validation and Writing
**What:** Python validation function (not jsonschema) that checks structure, types, and array consistency
**When to use:** Before every result.json write and after every load
**Example:**
```python
# Source: RESULTS_SCHEMA.md + CONTEXT.md decisions
import json
import os
from datetime import datetime, timezone

REQUIRED_TOP_FIELDS = {
    "schema_version", "experiment_id", "timestamp",
    "description", "tags", "config", "metrics"
}

def validate_result(result: dict) -> list[str]:
    """Validate result dict against schema. Returns list of errors (empty = valid)."""
    errors = []
    missing = REQUIRED_TOP_FIELDS - set(result.keys())
    if missing:
        errors.append(f"Missing required top-level fields: {missing}")
    if "metrics" in result and "scalars" not in result["metrics"]:
        errors.append("metrics.scalars is required")
    for seq in result.get("sequences", []):
        tokens = seq.get("tokens", [])
        for array_key in ["token_logprobs", "token_entropy"]:
            arr = seq.get(array_key, [])
            if arr and len(arr) != len(tokens):
                errors.append(
                    f"{array_key} length ({len(arr)}) != tokens length ({len(tokens)}) "
                    f"in {seq.get('sequence_id', 'unknown')}"
                )
    return errors
```

### Anti-Patterns to Avoid
- **Using Python's built-in `hash()` for config identity:** Randomized per session via PYTHONHASHSEED. Will produce different hashes on every run. Use `hashlib.sha256` over sorted JSON instead.
- **Mutable fields in frozen dataclasses:** Lists and dicts inside frozen dataclasses can still be mutated. Use `tuple` for sequence fields and ensure all nested objects are also frozen.
- **Seeding only torch and forgetting numpy/random:** NumPy and Python's random module have separate RNG states. Data loading code often uses these silently. Seed all three.
- **Using `json.dumps()` without `sort_keys=True` for hashing:** Dict ordering in JSON output depends on insertion order. Without sorted keys, identical configs can hash differently.
- **Storing config hash as Python `int` from `hash()`:** Not portable. Store as hex string from hashlib.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dict-to-nested-dataclass deserialization | Custom recursive `from_dict()` | `dacite.from_dict()` | Handles Optional, Union, nested lists, type casting; edge cases are numerous and subtle |
| JSON Schema validation | `jsonschema` library + `.json` schema file | Simple Python validation function | User decision: keep it simple and maintainable, avoid formal schema dependency |
| Experiment ID generation | UUID or random string | Param-slug + timestamp format | User decision: `n500_w64_r32_d128_L4_s42_{timestamp}` is scannable in `ls` output |
| CUDA determinism configuration | Manual per-module settings | Single `set_seed()` function | Easy to miss one source of non-determinism; centralize the checklist |

**Key insight:** The config system is pure Python infrastructure with no exotic dependencies. The complexity lies in discipline (always hash the same way, always seed everything, always validate before write) rather than in technology choices.

## Common Pitfalls

### Pitfall 1: Python hash() Is Session-Random
**What goes wrong:** Using `hash(config)` to generate cache keys or experiment IDs produces different values on each Python process start.
**Why it happens:** Python randomizes hash seeds for security (PYTHONHASHSEED) since Python 3.3.
**How to avoid:** Use `hashlib.sha256` over `json.dumps(asdict(config), sort_keys=True)`. This is deterministic across sessions, machines, and Python versions.
**Warning signs:** Cache misses on every run; identical configs getting different IDs.

### Pitfall 2: Frozen Dataclass with Mutable Nested Fields
**What goes wrong:** A frozen dataclass containing a `list` field can still have that list mutated in place, breaking immutability guarantees.
**Why it happens:** `frozen=True` only prevents reassignment of top-level attributes. Nested mutable containers remain mutable.
**How to avoid:** Use `tuple` instead of `list` for sequence fields. In `__post_init__`, convert any lists to tuples via `object.__setattr__(self, 'tags', tuple(self.tags))`. All sub-configs must also be `frozen=True`.
**Warning signs:** Config objects that were supposed to be identical now have different hashes; unexpected state changes.

### Pitfall 3: Forgetting CUBLAS_WORKSPACE_CONFIG
**What goes wrong:** Even with `torch.use_deterministic_algorithms(True)`, certain CUDA operations (cuBLAS matrix multiplications) remain non-deterministic.
**Why it happens:** cuBLAS uses non-deterministic stream scheduling by default for performance. The environment variable forces a deterministic workspace allocation.
**How to avoid:** Set `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` in the `set_seed()` function, before any CUDA operations.
**Warning signs:** Slightly different training loss curves across identical-seed runs on GPU.

### Pitfall 4: JSON Float Precision in Round-Trip
**What goes wrong:** Floating-point values like `0.9 * 64 = 57.6` may serialize with different precision, causing hash mismatches on round-trip.
**Why it happens:** Python's `json.dumps` uses `repr()` for floats which can produce platform-dependent digit counts in edge cases.
**How to avoid:** Pre-compute derived values as integers where possible (r=57, not r=0.9*w). For float fields that must remain, verify round-trip hash equality in tests.
**Warning signs:** Config hash mismatch after JSON round-trip despite identical logical values.

### Pitfall 5: Git Hash in Dirty Working Tree
**What goes wrong:** `git rev-parse --short HEAD` returns the committed hash, not reflecting uncommitted changes. Two runs with different code but same commit produce the same code_hash.
**Why it happens:** `rev-parse HEAD` only reflects committed state.
**How to avoid:** Append a `-dirty` suffix when `git diff --quiet` fails. Store as `"a3f9c1d-dirty"` to flag unreproducible state.
**Warning signs:** Results claiming same code_hash but behaving differently.

### Pitfall 6: dacite Silently Dropping Extra Keys
**What goes wrong:** Loading a config JSON from a newer schema version silently ignores unknown fields, masking version incompatibilities.
**Why it happens:** `dacite` default behavior is non-strict (ignores extra keys).
**How to avoid:** Use `dacite.Config(strict=True)` to reject unexpected keys. This catches schema drift early.
**Warning signs:** Configs loading successfully but missing expected parameters.

## Code Examples

Verified patterns from official sources:

### Experiment ID Generation
```python
# Source: CONTEXT.md decision -- scannable slug format
from datetime import datetime, timezone

def generate_experiment_id(config: ExperimentConfig) -> str:
    """Generate scannable experiment ID from config params.

    Format: n{n}_w{w}_r{r}_d{d_model}_L{n_layers}_s{seed}_{YYYYMMDD}_{HHMMSS}
    Example: n500_w64_r57_d128_L4_s42_20260224_143012
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
```

### Git Hash Capture with Dirty Detection
```python
# Source: git docs + RESULTS_SCHEMA.md
import subprocess

def get_git_hash() -> str:
    """Get short git SHA with dirty flag for reproducibility tracking."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

    try:
        subprocess.check_output(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        sha += "-dirty"

    return sha
```

### Result Writer with Validation
```python
# Source: RESULTS_SCHEMA.md + CONTEXT.md decisions
import json
import os
import numpy as np
from pathlib import Path

def write_result(
    config: ExperimentConfig,
    metrics: dict,
    sequences: list | None = None,
    metadata: dict | None = None,
    token_metrics: dict | None = None,  # {sequence_id: {metric_name: np.ndarray}}
    results_dir: str = "results",
) -> str:
    """Write result.json and optional token_metrics.npz.

    Returns experiment_id.
    """
    experiment_id = generate_experiment_id(config)
    out_dir = Path(results_dir) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": config.description,
        "tags": list(config.tags),
        "config": asdict(config),
        "metrics": metrics,
        "sequences": sequences or [],
        "metadata": {
            "code_hash": get_git_hash(),
            "config_hash": full_config_hash(config),
            "graph_config_hash": graph_config_hash(config),
            **(metadata or {}),
        },
    }

    # Validate before writing
    errors = validate_result(result)
    if errors:
        raise ValueError(f"Result validation failed:\n" + "\n".join(errors))

    # Write JSON
    result_path = out_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # Write npz for large token-level arrays
    if token_metrics:
        npz_path = out_dir / "token_metrics.npz"
        flat = {}
        for seq_id, metrics_dict in token_metrics.items():
            for metric_name, arr in metrics_dict.items():
                flat[f"{seq_id}/{metric_name}"] = arr
        np.savez_compressed(str(npz_path), **flat)

    return experiment_id
```

### DataLoader Seed Worker (for future phases)
```python
# Source: PyTorch reproducibility docs
# https://docs.pytorch.org/docs/stable/notes/randomness.html
import torch
import numpy
import random

def seed_worker(worker_id):
    """Worker init function for reproducible DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

# Usage:
# g = torch.Generator()
# g.manual_seed(config.seed)
# DataLoader(..., worker_init_fn=seed_worker, generator=g)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `@dataclass(frozen=True)` alone | `@dataclass(frozen=True, slots=True)` | Python 3.10 | 10-20% memory savings, faster attribute access |
| `np.random.seed()` global seeding | `np.random.default_rng(seed)` Generator API | NumPy 1.17 (2019) | Per-generator state avoids global mutation; but `np.random.seed()` still needed for libraries using legacy API |
| `torch.backends.cudnn.deterministic = True` only | `torch.use_deterministic_algorithms(True)` + CUBLAS env var | PyTorch 1.8+ | Covers more operations beyond cuDNN (scatter, index_put, etc.) |
| `hash()` for object identity | `hashlib` over canonical serialization | Always true | `hash()` was never suitable for cross-session identity |

**Deprecated/outdated:**
- `torch.backends.cudnn.deterministic = True` alone: Still useful but insufficient. Must combine with `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG` for full determinism.
- `np.random.seed()` as sole NumPy seeding: Works but the newer Generator API (`np.random.default_rng()`) is preferred for new code. However, many libraries still use the legacy global state, so setting both is prudent.

## Open Questions

1. **Exact DCSBM default parameters (K, p_in, p_out)**
   - What we know: The anchor config locks n=500, w=64, t=200k, d_model=128, n_layers=4, n_heads=1. The scaffold mentions sweep ranges for K (4,8,16), p_in (0.15,0.25,0.40), p_out (0.01,0.03,0.07).
   - What's unclear: Which specific values are the "anchor defaults" for K, p_in, p_out. The CONTEXT.md says "per spec defaults" but does not specify them.
   - Recommendation: Use K=4, p_in=0.25, p_out=0.03 as anchor defaults (middle of the sweep range). Confirm with user if needed, but these are reasonable defaults that ensure non-trivial block structure.

2. **SweepConfig scope in Phase 1**
   - What we know: CONTEXT.md lists SweepConfig as one of the nested dataclasses. Phase 1 requirements do not include MGMT-02 (parameter sweep definition).
   - What's unclear: Whether SweepConfig should be a stub or fully defined in Phase 1.
   - Recommendation: Define SweepConfig as a frozen dataclass with fields for sweep ranges (as tuples of allowed values) but defer the sweep execution logic to Phase 10. This lets Phase 1 validate that sweep configs serialize correctly without implementing the sweep runner.

3. **n_jumpers_per_block default**
   - What we know: Sweep range is 1, 2, 5 per block.
   - What's unclear: Anchor default.
   - Recommendation: Use 2 as the anchor default (moderate density, not trivial, not overwhelming).

## Sources

### Primary (HIGH confidence)
- [Python dataclasses documentation](https://docs.python.org/3/library/dataclasses.html) - frozen, slots, __post_init__, asdict
- [Python hashlib documentation](https://docs.python.org/3/library/hashlib.html) - SHA-256 deterministic hashing
- [PyTorch Reproducibility docs](https://docs.pytorch.org/docs/stable/notes/randomness.html) - seed management, CUDA determinism, DataLoader workers
- [NumPy savez documentation](https://numpy.org/doc/stable/reference/generated/numpy.savez.html) - npz compressed array storage
- [dacite GitHub](https://github.com/konradhalas/dacite) - from_dict API, Config options, nested dataclass support

### Secondary (MEDIUM confidence)
- [Deterministic hashing of Python data objects](https://death.andgravity.com/stable-hashing) - JSON + hashlib pattern, field exclusion, stability guarantees
- [PyTorch CUDA Environment Variables](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html) - CUBLAS_WORKSPACE_CONFIG documentation

### Tertiary (LOW confidence)
- None -- all findings verified with primary or secondary sources.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all stdlib plus well-established libraries (dacite, torch, numpy)
- Architecture: HIGH - patterns are straightforward frozen dataclass composition with no novel design decisions
- Pitfalls: HIGH - well-documented issues (hash randomization, CUDA determinism, mutable nested fields) with clear solutions

**Research date:** 2026-02-24
**Valid until:** 2026-06-24 (stable domain, no fast-moving dependencies)
