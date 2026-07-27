"""Connectors.

Importing this package registers the built-in connectors as a side effect.
"""

from cogfm.connectors import linear, mlp  # noqa: F401  (register connectors)
from cogfm.connectors.base import Connector

__all__ = ["Connector"]
