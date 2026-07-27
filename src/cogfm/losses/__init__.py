"""Binding losses.

Importing this package registers the built-in losses as a side effect.
"""

from cogfm.losses import infonce  # noqa: F401  (registers InfoNCELoss)
from cogfm.losses.base import BindingLoss

__all__ = ["BindingLoss"]
