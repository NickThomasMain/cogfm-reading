"""Anchor.

Importing this package registers the built-in anchor(s) as a side effect.
"""

from cogfm.anchor import standin  # noqa: F401  (registers StandInAnchor)
from cogfm.anchor.base import AnchorEncoder

__all__ = ["AnchorEncoder"]
