"""TDD tests for the full training pipeline (Phase 05-02).

Tests the orchestration: training loop + evaluation + checkpoint + gate + result writing.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from src.config.experiment import ExperimentConfig, GraphConfig, ModelConfig, TrainingConfig
from src.graph.jumpers import JumperInfo
from src.graph.types import GraphData
from src.model.transformer import TransformerLM
from src.training.evaluate import ComplianceResult
from src.training.pipeline import TrainingPipelineResult, run_training_pipeline


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
def complete_graph():
    n = 20
    K = 4
    block_size = n // K
    block_assignments = np.repeat(np.arange(K), block_size).astype(np.int32)

    # Include self-loops so all possible token transitions are valid edges
    rows, cols = [], []
    for i in range(n):
        for j in range(n):
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows), dtype=np.float64)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))

    return GraphData(
        adjacency=adjacency,
        block_assignments=block_assignments,
        theta=np.ones(n),
        n=n, K=K, block_size=block_size,
        generation_seed=42, attempt=0,
    )


@pytest.fixture
def jumpers():
    return [
        JumperInfo(vertex_id=0, source_block=0, target_block=2, r=10),
        JumperInfo(vertex_id=5, source_block=1, target_block=3, r=10),
    ]


@pytest.fixture
def train_walks():
    rng = np.random.default_rng(42)
    return rng.integers(0, 20, size=(100, 32), dtype=np.int32)


@pytest.fixture
def eval_walks():
    rng = np.random.default_rng(123)
    return rng.integers(0, 20, size=(50, 32), dtype=np.int32)


class TestPipelineResult:
    def test_pipeline_returns_result(
        self, small_model, small_config, train_walks, eval_walks,
        complete_graph, jumpers, device, tmp_path
    ):
        """run_training_pipeline returns TrainingPipelineResult."""
        result = run_training_pipeline(
            model=small_model,
            train_walks=train_walks,
            eval_walks=eval_walks,
            graph_data=complete_graph,
            jumpers=jumpers,
            config=small_config,
            device=device,
            results_dir=str(tmp_path / "results"),
            max_epochs=2,
        )
        assert isinstance(result, TrainingPipelineResult)
        assert hasattr(result, "gate_passed")
        assert hasattr(result, "final_epoch")
        assert hasattr(result, "curves")
        assert hasattr(result, "result_path")

    def test_pipeline_logs_curves(
        self, small_model, small_config, train_walks, eval_walks,
        complete_graph, jumpers, device, tmp_path
    ):
        """Result contains curves with train_loss, edge_compliance, rule_compliance."""
        result = run_training_pipeline(
            model=small_model,
            train_walks=train_walks,
            eval_walks=eval_walks,
            graph_data=complete_graph,
            jumpers=jumpers,
            config=small_config,
            device=device,
            results_dir=str(tmp_path / "results"),
            max_epochs=2,
        )
        assert "train_loss" in result.curves
        assert "edge_compliance" in result.curves
        assert "rule_compliance" in result.curves
        # train_loss should have one entry per step
        assert len(result.curves["train_loss"]) > 0
        # compliance should have one entry per epoch
        assert len(result.curves["edge_compliance"]) == result.final_epoch
        assert len(result.curves["rule_compliance"]) == result.final_epoch

    def test_pipeline_stops_on_gate_pass(
        self, small_model, small_config, train_walks, eval_walks,
        complete_graph, jumpers, device, tmp_path
    ):
        """With rigged compliance always passing, pipeline stops early."""
        always_pass = ComplianceResult(
            edge_compliance=0.99, rule_compliance=0.95,
            n_sequences=10, n_edge_checks=100, n_rule_checks=50,
        )
        with patch(
            "src.training.pipeline.evaluate_compliance",
            return_value=always_pass,
        ):
            result = run_training_pipeline(
                model=small_model,
                train_walks=train_walks,
                eval_walks=eval_walks,
                graph_data=complete_graph,
                jumpers=jumpers,
                config=small_config,
                device=device,
                results_dir=str(tmp_path / "results"),
                max_epochs=50,
            )
        assert result.gate_passed is True
        assert result.final_epoch == 1  # Should stop after first epoch

    def test_pipeline_writes_failure_metadata(
        self, small_model, small_config, train_walks, eval_walks,
        complete_graph, jumpers, device, tmp_path
    ):
        """With rigged compliance never passing, writes failure metadata."""
        always_fail = ComplianceResult(
            edge_compliance=0.5, rule_compliance=0.3,
            n_sequences=10, n_edge_checks=100, n_rule_checks=50,
        )
        with patch(
            "src.training.pipeline.evaluate_compliance",
            return_value=always_fail,
        ):
            result = run_training_pipeline(
                model=small_model,
                train_walks=train_walks,
                eval_walks=eval_walks,
                graph_data=complete_graph,
                jumpers=jumpers,
                config=small_config,
                device=device,
                results_dir=str(tmp_path / "results"),
                max_epochs=3,
            )
        assert result.gate_passed is False
        assert result.final_epoch == 3
        assert result.failure_reason is not None
        assert "not passed" in result.failure_reason.lower() or "fail" in result.failure_reason.lower()
