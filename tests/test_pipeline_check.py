"""Tests for run_pipeline_check: it produces finite, in-range, reproducible metrics."""

import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from cogfm.binding.train import run_pipeline_check

ZUCO_ROOT = Path("data/zuco/processed")


def _cfg(data: dict | None = None):
    return OmegaConf.create(
        {
            "seed": 0,
            "data": data or {"name": "dummy", "n_samples": 16, "n_fixations": 6},
            "encoder": {"name": "scanpath", "embed_dim": 16},
            "connector": {"name": "mlp", "hidden_dim": 32, "dropout": 0.0},
            "anchor": {"name": "standin", "dim": 16, "vocab_size": 500},
            "loss": {"name": "infonce", "temperature": 0.07},
            "optimizer": {"lr": 0.001, "weight_decay": 0.0},
            "training": {"max_steps": 3, "batch_size": 8},
        }
    )


def test_returns_finite_in_range_metrics():
    m = run_pipeline_check(_cfg())
    assert math.isfinite(m["final_loss"])
    assert 0.0 <= m["retrieval@1"] <= 1.0
    assert 0.0 <= m["retrieval@5"] <= 1.0
    assert m["steps"] == 3


def test_is_reproducible():
    a = run_pipeline_check(_cfg())
    b = run_pipeline_check(_cfg())
    assert a["final_loss"] == b["final_loss"]
    assert a["retrieval@1"] == b["retrieval@1"]


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="unknown dataset"):
        run_pipeline_check(_cfg({"name": "nope"}))


@pytest.mark.skipif(
    not (ZUCO_ROOT / "scanpaths.npz").is_file(),
    reason="ZuCo has not been extracted; run scripts/extract_zuco.py",
)
def test_runs_on_zuco_fold():
    """The same loop on real data, over one subject- and item-disjoint fold."""
    cfg = _cfg(
        {
            "name": "zuco",
            "root": str(ZUCO_ROOT),
            "task": None,
            "n_folds": 4,
            "fold": 0,
            "split_seed": 0,
        }
    )
    m = run_pipeline_check(cfg)
    assert math.isfinite(m["final_loss"])
    assert 0.0 <= m["retrieval@1"] <= 1.0
