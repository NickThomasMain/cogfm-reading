"""Binding training loop and an end-to-end pipeline check.

``run_pipeline_check`` builds every component from the config, runs a few
training steps on the synthetic data, and reports the final loss plus
retrieval@k. Its job is to prove the pipeline flows end-to-end and produces a
reproducible number, not to reach any particular quality.
"""

from __future__ import annotations

import logging
from itertools import cycle

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

# importing these packages registers the built-in components
import cogfm.anchor  # noqa: F401
import cogfm.connectors  # noqa: F401
import cogfm.encoders  # noqa: F401
import cogfm.losses  # noqa: F401
from cogfm.binding.model import BindingModel
from cogfm.data.batching import batch_reading_samples
from cogfm.data.dummy import DummyReadingDataset
from cogfm.eval.retrieval import retrieval_at_k
from cogfm.registry import ANCHORS, CONNECTORS, ENCODERS, LOSSES
from cogfm.seed import set_seed

log = logging.getLogger(__name__)


def _build_model(cfg: DictConfig) -> BindingModel:
    encoder = ENCODERS.build(cfg.encoder.name, embed_dim=cfg.encoder.embed_dim)
    connector_params = {k: v for k, v in cfg.connector.items() if k != "name"}
    connector = CONNECTORS.build(
        cfg.connector.name,
        in_dim=cfg.encoder.embed_dim,
        out_dim=cfg.anchor.dim,
        **connector_params,
    )
    anchor = ANCHORS.build(cfg.anchor.name, dim=cfg.anchor.dim, vocab_size=cfg.anchor.vocab_size)
    return BindingModel(encoder, connector, anchor)


def _evaluate(model: BindingModel, loss_fn, dataset, batch_size: int) -> dict:
    model.eval()
    n = min(batch_size, len(dataset))
    batch = batch_reading_samples([dataset[i] for i in range(n)])
    with torch.no_grad():
        modality = model.encode_modality(batch["scanpath"])
        text = model.encode_text(batch["text"])
        _, logits = loss_fn(modality, text)
    return {
        "retrieval@1": retrieval_at_k(logits, k=1),
        "retrieval@5": retrieval_at_k(logits, k=5),
    }


def run_pipeline_check(cfg: DictConfig) -> dict:
    set_seed(cfg.seed)

    model = _build_model(cfg)
    loss_fn = LOSSES.build(cfg.loss.name, temperature=cfg.loss.temperature)

    # frozen backbones: only the connector is trained
    model.encoder.requires_grad_(False)
    model.anchor.requires_grad_(False)
    optimizer = torch.optim.Adam(
        model.connector.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    dataset = DummyReadingDataset(
        cfg.data.n_samples, seed=cfg.seed, n_fixations=cfg.data.n_fixations
    )
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=batch_reading_samples,
        generator=generator,
    )

    model.train()
    batches = cycle(loader)
    final_loss = float("nan")
    for step in range(1, cfg.training.max_steps + 1):
        batch = next(batches)
        modality = model.encode_modality(batch["scanpath"])
        text = model.encode_text(batch["text"])
        loss, _ = loss_fn(modality, text)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        log.info("step %d/%d  loss=%.4f", step, cfg.training.max_steps, final_loss)

    metrics = _evaluate(model, loss_fn, dataset, cfg.training.batch_size)
    metrics["final_loss"] = final_loss
    metrics["steps"] = cfg.training.max_steps
    log.info(
        "pipeline check done: loss=%.4f  R@1=%.3f  R@5=%.3f",
        metrics["final_loss"],
        metrics["retrieval@1"],
        metrics["retrieval@5"],
    )
    return metrics
