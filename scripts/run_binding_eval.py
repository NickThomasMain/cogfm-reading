"""Measure binding on ZuCo across every fold and every condition.

Produces the result row the feasibility claim rests on. Each condition trains
the same connector with the same loss on the same folds, differing only in what
the encoder sees:

    trivial   seven surface statistics of the scanpath
    random    an untrained frozen encoder over the full fixation sequence
    scanpath  whatever the configured encoder is

Reading the row means reading the distances between neighbours. Beating the
permutation null says only that something is there; beating the trivial
features is what says it is not merely text length.

Examples (from the repo root):

    uv run python scripts/run_binding_eval.py
    uv run python scripts/run_binding_eval.py training.max_steps=400 training.batch_size=256
    uv run python scripts/run_binding_eval.py eval.n_permutations=200 eval.folds=[0,1]
"""

from __future__ import annotations

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import cogfm.anchor  # noqa: F401
import cogfm.connectors  # noqa: F401
import cogfm.encoders  # noqa: F401
import cogfm.losses  # noqa: F401
from cogfm.binding.model import BindingModel
from cogfm.data.adapters.zuco import ZuCoReadingDataset
from cogfm.data.splits import make_folds
from cogfm.eval.report import aggregate, format_gaps, format_result_row
from cogfm.eval.runner import embed_fold, evaluate_similarity
from cogfm.registry import ANCHORS, CONNECTORS, ENCODERS, LOSSES
from cogfm.seed import set_seed
from torch.utils.data import Subset

from cogfm.binding.train import train_connector

log = logging.getLogger(__name__)


def build_anchor(cfg: DictConfig):
    """Load the anchor once, to be shared by every condition and fold.

    The anchor is frozen and evaluated without dropout, so it returns the same
    vector for a sentence no matter which run asks. Rebuilding it per condition
    would reload the weights and re-encode the corpus for nothing, and would
    throw away the sentence cache each time.
    """
    params = {k: v for k, v in cfg.anchor.items() if k not in ("name", "dim")}
    anchor = ANCHORS.build(cfg.anchor.name, dim=cfg.anchor.dim, **params)
    anchor.requires_grad_(False)
    return anchor


def build_model(cfg: DictConfig, encoder_name: str, encoder_dim: int, anchor) -> BindingModel:
    """Assemble the two towers with one condition's encoder in place."""
    encoder = ENCODERS.build(encoder_name, embed_dim=encoder_dim)
    connector_params = {k: v for k, v in cfg.connector.items() if k != "name"}
    connector = CONNECTORS.build(
        cfg.connector.name, in_dim=encoder_dim, out_dim=cfg.anchor.dim, **connector_params
    )
    return BindingModel(encoder, connector, anchor)


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    dataset = ZuCoReadingDataset(root=cfg.data.root, task=cfg.data.task)
    folds = make_folds(
        dataset.subject_ids,
        dataset.sentence_ids,
        dataset.sentences,
        n_folds=cfg.data.n_folds,
        seed=cfg.data.split_seed,
    )
    wanted = list(cfg.eval.folds) if cfg.eval.folds else list(range(len(folds)))
    log.info("%d trials, %d sentences, folds %s", len(dataset), len(dataset.sentences), wanted)

    anchor = build_anchor(cfg)

    summaries = []
    for condition in cfg.conditions:
        records = []
        for index in wanted:
            fold = folds[index]
            set_seed(cfg.seed + index)

            model = build_model(cfg, condition.encoder, condition.embed_dim, anchor)
            model.encoder.requires_grad_(False)
            optimizer = torch.optim.Adam(
                model.connector.parameters(),
                lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
            )
            first, last = train_connector(
                model,
                LOSSES.build(cfg.loss.name, temperature=cfg.loss.temperature),
                optimizer,
                Subset(dataset, fold.train.tolist()),
                max_steps=cfg.training.max_steps,
                batch_size=cfg.training.batch_size,
                seed=cfg.seed + index,
                log_every=cfg.training.log_every,
            )
            log.info("[%s] fold %d: loss %.4f -> %.4f", condition.name, index, first, last)

            similarity, order, query_sentences = embed_fold(
                model,
                dataset,
                fold.test.tolist(),
                fold.test_sentences,
                batch_size=cfg.eval.batch_size,
            )
            record = evaluate_similarity(
                similarity,
                order,
                query_sentences,
                dataset.sentences,
                condition=condition.name,
                fold=index,
                pool_size=cfg.eval.pool_size,
                tolerance=cfg.eval.tolerance,
                pool_seed=cfg.eval.pool_seed,
                n_permutations=cfg.eval.n_permutations,
                permutation_seed=cfg.eval.permutation_seed,
            )
            log.info("%s", record)
            records.append(record)
        summaries.append(aggregate(records))

    print("\n" + format_result_row(summaries))
    print("\n" + format_gaps(summaries, metric=cfg.eval.headline_metric))
    print("\nKonfiguration:\n" + OmegaConf.to_yaml(cfg.eval))


if __name__ == "__main__":
    main()
