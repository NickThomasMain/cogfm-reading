"""Modality encoders.

Importing this package registers the built-in encoders as a side effect, so the
registry knows about them (e.g. ``ENCODERS.build("scanpath")``).
"""

from cogfm.encoders import scanpath  # noqa: F401  (registers ScanpathEncoder)
from cogfm.encoders.base import ModalityEncoder

__all__ = ["ModalityEncoder"]
