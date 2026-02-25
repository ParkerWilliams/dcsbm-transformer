"""TDD tests for greedy generation and compliance evaluation (Phase 05-02).

Uses small fixtures with known graph structure for deterministic compliance testing.
"""

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from src.config.experiment import ExperimentConfig, GraphConfig, ModelConfig, TrainingConfig
from src.graph.jumpers import JumperInfo
from src.graph.types import GraphData
from src.model.transformer import TransformerLM
from src.training.evaluate import (
    ComplianceResult,
    evaluate_compliance,
    greedy_generate,
)


@pytest.fixture
def small_config():
    return ExperimentConfig(
        graph=GraphConfig(n=20, K=4, p_in=0.5, p_out=0.1, n_jumpers_per_block=1),
        model=ModelConfig(d_model=32, n_layers=2, n_heads=1, dropout=0.0),
        training=TrainingConfig(
            w=16, walk_length=32, corpus_size=2000, r=10,
            learning_rate=3e-4, batch_size=8, max_steps=1000,
            eval_interval=100, checkpoint_interval=500,
        ),
        seed=42,
    )


@pytest.fixture
def small_model(small_config):
    return TransformerLM(
        vocab_size=small_config.graph.n,
        d_model=small_config.model.d_model,
        n_layers=small_config.model.n_layers,
        max_seq_len=small_config.training.w,
        dropout=0.0,
    )


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def small_graph():
    """Create a small 20-vertex graph with known edges for compliance testing."""
    n = 20
    K = 4
    block_size = n // K  # 5

    # Create block assignments: [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3]
    block_assignments = np.repeat(np.arange(K), block_size).astype(np.int32)

    # Create a complete graph (all edges valid) for simplicity in some tests
    rows, cols = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                rows.append(i)
                cols.append(j)
    data = np.ones(len(rows), dtype=np.float64)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))

    return GraphData(
        adjacency=adjacency,
        block_assignments=block_assignments,
        theta=np.ones(n),
        n=n,
        K=K,
        block_size=block_size,
        generation_seed=42,
        attempt=0,
    )


@pytest.fixture
def small_graph_sparse():
    """Create a sparse 20-vertex graph where only some edges exist."""
    n = 20
    K = 4
    block_size = n // K

    block_assignments = np.repeat(np.arange(K), block_size).astype(np.int32)

    # Only create edges within blocks (no inter-block edges)
    rows, cols = [], []
    for b in range(K):
        start = b * block_size
        end = (b + 1) * block_size
        for i in range(start, end):
            for j in range(start, end):
                if i != j:
                    rows.append(i)
                    cols.append(j)
    data = np.ones(len(rows), dtype=np.float64)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))

    return GraphData(
        adjacency=adjacency,
        block_assignments=block_assignments,
        theta=np.ones(n),
        n=n,
        K=K,
        block_size=block_size,
        generation_seed=42,
        attempt=0,
    )


@pytest.fixture
def jumpers():
    """Block jumpers: vertex 0 (block 0) targets block 2, vertex 5 (block 1) targets block 3."""
    return [
        JumperInfo(vertex_id=0, block=0, target_block=2, r=10),
        JumperInfo(vertex_id=5, block=1, target_block=3, r=10),
    ]


class TestGreedyGenerate:
    def test_greedy_generate_shape(self, small_model, device):
        """greedy_generate returns tensor of shape [B, length]."""
        start_tokens = torch.tensor([[5], [10]], dtype=torch.long)
        result = greedy_generate(
            small_model, start_tokens, length=20,
            max_seq_len=small_model.max_seq_len, device=device,
        )
        assert result.shape == (2, 20)
        assert result.dtype == torch.long

    def test_greedy_generate_argmax(self, small_model, device):
        """Generated tokens match argmax of model logits at each step."""
        torch.manual_seed(42)
        start_tokens = torch.tensor([[3]], dtype=torch.long)
        result = greedy_generate(
            small_model, start_tokens, length=10,
            max_seq_len=small_model.max_seq_len, device=device,
        )

        # Verify by manual forward pass
        small_model.eval()
        generated = start_tokens.clone()
        with torch.no_grad():
            for i in range(9):
                from src.model.types import ExtractionMode
                output = small_model(generated[:, -small_model.max_seq_len:], mode=ExtractionMode.NONE)
                next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

        assert torch.equal(result, generated)


class TestEvaluateCompliance:
    def test_evaluate_compliance_perfect_edges(
        self, small_model, small_config, small_graph, jumpers, device
    ):
        """With complete graph, edge compliance is 1.0."""
        eval_walks = np.random.default_rng(42).integers(
            0, 20, size=(100, 32), dtype=np.int32
        )
        result = evaluate_compliance(
            small_model, eval_walks, small_graph, jumpers, small_config, device,
            n_sequences=10,
        )
        assert isinstance(result, ComplianceResult)
        # Complete graph means all edges are valid
        assert result.edge_compliance == 1.0

    def test_evaluate_compliance_invalid_edges(
        self, small_model, small_config, small_graph_sparse, jumpers, device
    ):
        """With sparse graph, edge compliance < 1.0 if model generates cross-block edges."""
        eval_walks = np.random.default_rng(42).integers(
            0, 20, size=(100, 32), dtype=np.int32
        )
        result = evaluate_compliance(
            small_model, eval_walks, small_graph_sparse, jumpers, small_config, device,
            n_sequences=10,
        )
        assert isinstance(result, ComplianceResult)
        # Sparse graph (only intra-block edges) -- untrained model will generate
        # many invalid cross-block edges
        assert result.edge_compliance < 1.0
        assert result.n_edge_checks > 0

    def test_evaluate_compliance_no_jumpers(
        self, small_model, small_config, small_graph, device
    ):
        """When no jumpers, rule compliance is 1.0 (vacuous truth)."""
        eval_walks = np.random.default_rng(42).integers(
            0, 20, size=(100, 32), dtype=np.int32
        )
        result = evaluate_compliance(
            small_model, eval_walks, small_graph, [], small_config, device,
            n_sequences=10,
        )
        assert result.rule_compliance == 1.0
        assert result.n_rule_checks == 0

    def test_compliance_result_dataclass(
        self, small_model, small_config, small_graph, jumpers, device
    ):
        """ComplianceResult has expected fields."""
        eval_walks = np.random.default_rng(42).integers(
            0, 20, size=(100, 32), dtype=np.int32
        )
        result = evaluate_compliance(
            small_model, eval_walks, small_graph, jumpers, small_config, device,
            n_sequences=5,
        )
        assert hasattr(result, "edge_compliance")
        assert hasattr(result, "rule_compliance")
        assert hasattr(result, "n_sequences")
        assert hasattr(result, "n_edge_checks")
        assert hasattr(result, "n_rule_checks")
        assert result.n_sequences == 5

    def test_evaluate_compliance_rule_check(
        self, small_model, small_config, small_graph, jumpers, device
    ):
        """Rule compliance checking works with known jumper info."""
        eval_walks = np.random.default_rng(42).integers(
            0, 20, size=(100, 32), dtype=np.int32
        )
        result = evaluate_compliance(
            small_model, eval_walks, small_graph, jumpers, small_config, device,
            n_sequences=20,
        )
        # With an untrained model and jumpers, we should have some rule checks
        # The compliance value itself depends on model output, but checks happen
        assert result.n_sequences == 20
        assert isinstance(result.rule_compliance, float)
        assert 0.0 <= result.rule_compliance <= 1.0
        assert 0.0 <= result.edge_compliance <= 1.0
