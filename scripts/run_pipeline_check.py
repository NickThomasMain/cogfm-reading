"""Run the end-to-end pipeline check from a Hydra config.

Examples (from the repo root):

    # default composition
    uv run python scripts/run_pipeline_check.py

    # swap components / override values on the command line
    uv run python scripts/run_pipeline_check.py connector=linear optimizer.lr=0.0005

    # a small sweep: one run per connector
    uv run python scripts/run_pipeline_check.py --multirun connector=linear,mlp
"""

import logging

import hydra
from omegaconf import DictConfig

from cogfm.binding.train import run_pipeline_check

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    metrics = run_pipeline_check(cfg)
    log.info("final metrics: %s", metrics)


if __name__ == "__main__":
    main()
