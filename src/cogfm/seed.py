"""Reproducibility helper: seed every random-number generator from one place.

ML results depend on randomness (weight init, data shuffling, dropout). To make
a run repeatable, every source of randomness must start from the same seed.
Call `set_seed(cfg.seed)` once at the start of a run.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs.

    Args:
        seed: the integer seed, typically ``cfg.seed``.
        deterministic: if True, also force deterministic CuDNN behaviour on GPU.
            Slightly slower, but makes GPU runs bit-for-bit reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # torch is imported lazily so this helper also works in a torch-free context.
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
