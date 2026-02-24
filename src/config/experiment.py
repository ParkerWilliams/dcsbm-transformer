"""Experiment configuration dataclasses — all frozen and slotted for immutability."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class GraphConfig:
    """DCSBM graph generation parameters."""

    n: int = 500  # number of vertices
    K: int = 4  # number of blocks
    p_in: float = 0.25  # in-group edge probability
    p_out: float = 0.03  # out-group edge probability
    n_jumpers_per_block: int = 2  # block jumper vertices per block


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """NanoGPT-scale transformer parameters."""

    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 1  # locked: always 1 (single-head constraint)
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training loop parameters."""

    w: int = 64  # context window
    walk_length: int = 256  # 4 * w
    corpus_size: int = 200_000  # t — number of training walks
    r: int = 57  # jump length (0.9 * w rounded)
    learning_rate: float = 3e-4
    batch_size: int = 64
    max_steps: int = 50_000
    eval_interval: int = 1000
    checkpoint_interval: int = 5000


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """Parameter sweep ranges. Structure defined here; execution logic in Phase 10."""

    n_values: tuple[int, ...] = (500,)
    w_values: tuple[int, ...] = (64,)
    r_scale_values: tuple[float, ...] = (0.9,)
    d_model_values: tuple[int, ...] = (128,)
    n_layers_values: tuple[int, ...] = (4,)
    K_values: tuple[int, ...] = (4,)
    p_in_values: tuple[float, ...] = (0.25,)
    p_out_values: tuple[float, ...] = (0.03,)
    seeds: tuple[int, ...] = (42, 123, 7)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Top-level experiment configuration composing all sub-configs.

    All fields are frozen and typed. Cross-parameter validation runs
    in __post_init__ to reject invalid configurations early.
    """

    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sweep: SweepConfig | None = None
    seed: int = 42
    description: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Cross-parameter validation (uses object.__setattr__ since frozen)."""
        if self.training.walk_length < 2 * self.training.w:
            raise ValueError(
                f"walk_length ({self.training.walk_length}) must be "
                f">= 2 * w ({2 * self.training.w})"
            )
        if self.training.corpus_size < 100 * self.graph.n:
            raise ValueError(
                f"corpus_size ({self.training.corpus_size}) must be "
                f">= 100 * n ({100 * self.graph.n})"
            )
        if self.model.n_heads != 1:
            raise ValueError("n_heads must be exactly 1 (single-head constraint)")
        if self.training.r > self.training.walk_length:
            raise ValueError(
                f"r ({self.training.r}) must be "
                f"<= walk_length ({self.training.walk_length})"
            )
