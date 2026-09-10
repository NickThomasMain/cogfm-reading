"""Modality encoders.

Importing this package registers the built-in encoders as a side effect, so the
registry knows about them (e.g. ``ENCODERS.build("scanpath")``).
"""

from cogfm.encoders import labram  # noqa: F401  (registers LaBraMEncoder)
from cogfm.encoders import noise  # noqa: F401  (registers NoiseEncoder)
from cogfm.encoders import precomputed  # noqa: F401  (registers PrecomputedEncoder)
from cogfm.encoders import scanez  # noqa: F401  (registers ScanEZEncoder)
from cogfm.encoders import scanpath  # noqa: F401  (registers ScanpathEncoder)
from cogfm.encoders import trivial  # noqa: F401  (registers TrivialFeatureEncoder)
from cogfm.encoders.base import ModalityEncoder

__all__ = ["ModalityEncoder"]
