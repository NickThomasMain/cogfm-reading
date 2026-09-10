"""Binding training loop and an end-to-end pipeline check.

``run_pipeline_check`` builds every component from the config, trains the
connector for a few steps and reports the final loss plus retrieval@k. Its job
is to prove that the pipeline flows end to end and produces a reproducible
number, not to reach any particular quality.

The reported retrieval is measured inside a single batch, so its chance level
is one over the batch size and it says nothing about binding quality. A real
measurement needs a fixed, length-matched candidate pool and a permutation
null, neither of which lives here.
"""

from __future__ import annotations

import logging
from itertools import cycle

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset

# importing these packages registers the built-in components
import cogfm.anchor  # noqa: F401
import cogfm.connectors  # noqa: F401
import cogfm.encoders  # noqa: F401
import cogfm.losses  # noqa: F401
from cogfm.binding.model import BindingModel
from cogfm.data.adapters.zuco_et import ZuCoETDataset
from cogfm.data.batching import batch_reading_samples
from cogfm.data.dummy import DummyReadingDataset
from cogfm.data.splits import make_folds, untested
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
    anchor_params = {k: v for k, v in cfg.anchor.items() if k not in ("name", "dim")}
    anchor = ANCHORS.build(cfg.anchor.name, dim=cfg.anchor.dim, **anchor_params)
    return BindingModel(encoder, connector, anchor)


def _build_datasets(cfg: DictConfig) -> tuple[Dataset, Dataset]:
    """Return the training and evaluation datasets named by the config.

    Synthetic data has no split and serves both roles. ZuCo is divided into
    subject- and item-disjoint folds; the configured fold becomes the test
    split and the rest of the grid becomes training. Trials that overlap the
    test split on exactly one axis belong to neither and are left out.
    """
    name = cfg.data.name
    if name == "dummy":
        dataset = DummyReadingDataset(
            cfg.data.n_samples, seed=cfg.seed, n_fixations=cfg.data.n_fixations
        )
        log.info("dummy data: %d samples, %d fixations each", len(dataset), cfg.data.n_fixations)
        return dataset, dataset

    if name == "zuco_et":
        dataset = ZuCoETDataset(root=cfg.data.root, task=cfg.data.task)
        folds = make_folds(
            dataset.subject_ids,
            dataset.sentence_ids,
            dataset.sentences,
            n_folds=cfg.data.n_folds,
            seed=cfg.data.split_seed,
        )
        if not 0 <= cfg.data.fold < len(folds):
            raise ValueError(f"fold {cfg.data.fold} outside 0..{len(folds) - 1}")
        fold = folds[cfg.data.fold]
        dropped = len(dataset) - len(fold.train) - len(fold.test)
        log.info(
            "zuco_et fold %d of %d: %d train, %d test, %d dropped (%d trials, %d sentences total)",
            fold.index,
            len(folds),
            len(fold.train),
            len(fold.test),
            dropped,
            len(dataset),
            len(dataset.sentences),
        )
        log.info("test subjects: %s", ", ".join(fold.test_subjects))
        never = untested(folds, dataset.sentence_ids)
        if len(never):
            log.info("sentences no fold tests: %d", len(never))
        return Subset(dataset, fold.train.tolist()), Subset(dataset, fold.test.tolist())

    raise ValueError(f"unknown dataset '{name}'; expected 'dummy' or 'zuco_et'")


def _evaluate(model: BindingModel, loss_fn, dataset: Dataset, batch_size: int) -> dict:
    """Retrieval inside one batch drawn from the evaluation split."""
    model.eval()
    n = min(batch_size, len(dataset))
    batch = batch_reading_samples([dataset[i] for i in range(n)])
    with torch.no_grad():
        modality = model.encode_modality(batch["scanpath"], batch["mask"])
        text = model.encode_text(batch["text"])
        _, logits = loss_fn(modality, text)
    return {
        "retrieval@1": retrieval_at_k(logits, k=1),
        "retrieval@5": retrieval_at_k(logits, k=5),
        "chance@1": 1.0 / n,
        "eval_batch": n,
    }


def train_connector(
    model: BindingModel,
    loss_fn,
    optimizer,
    train_set: Dataset,
    max_steps: int,
    batch_size: int,
    seed: int = 0,
    log_every: int = 0,
) -> tuple[float, float]:
    """Train the connector for a fixed number of steps and report the loss ends.

    Encoder and anchor stay frozen, so gradients reach the connector alone. The
    loader cycles, which means a step count above one epoch simply revisits the
    data.

    Args:
        model: the two towers, already built.
        loss_fn: returns (loss, logits) for a modality and a text batch.
        optimizer: updates whatever parameters it was given.
        train_set: the training split of one fold.
        max_steps: optimisation steps to run.
        batch_size: samples per step; the number of negatives a contrastive
            loss sees is one less than this, so small batches weaken it.
        seed: controls shuffling.
        log_every: log the loss every so many steps; silent when zero.

    Returns:
        The loss after the first step and after the last.
    """
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=batch_reading_samples,
        generator=generator,
        drop_last=len(train_set) > batch_size,
    )

    model.train()
    batches = cycle(loader)
    first_loss = final_loss = float("nan")
    for step in range(1, max_steps + 1):
        batch = next(batches)
        modality = model.encode_modality(batch["scanpath"], batch["mask"])
        text = model.encode_text(batch["text"])
        loss, _ = loss_fn(modality, text)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        if step == 1:
            first_loss = final_loss
        if log_every and step % log_every == 0:
            log.info("step %d/%d  loss=%.4f", step, max_steps, final_loss)
    return first_loss, final_loss


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

    train_set, eval_set = _build_datasets(cfg)
    first_loss, final_loss = train_connector(
        model,
        loss_fn,
        optimizer,
        train_set,
        max_steps=cfg.training.max_steps,
        batch_size=cfg.training.batch_size,
        seed=cfg.seed,
    )

    metrics = _evaluate(model, loss_fn, eval_set, cfg.training.batch_size)
    metrics["first_loss"] = first_loss
    metrics["final_loss"] = final_loss
    metrics["steps"] = cfg.training.max_steps
    log.info(
        "pipeline check done: loss %.4f -> %.4f  R@1=%.3f (chance %.3f)  R@5=%.3f",
        metrics["first_loss"],
        metrics["final_loss"],
        metrics["retrieval@1"],
        metrics["chance@1"],
        metrics["retrieval@5"],
    )
    return metrics
