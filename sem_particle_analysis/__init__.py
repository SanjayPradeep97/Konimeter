"""Segmentation and measurement of particles in SEM/TEM micrographs,
built on Meta's Segment Anything Model."""

from . import modality, region
from .model import SAMModel, discover_checkpoints, infer_model_type
from .scale_detection import OCRUnavailableError, ScaleDetector
from .segmentation import ParticleSegmenter
from .analysis import ParticleAnalyzer
from .data_manager import ResultsManager

# Single source of truth is the version in pyproject.toml; read it back from the
# installed distribution so the two can't drift apart.
try:
    from importlib.metadata import PackageNotFoundError, version as _version

    __version__ = _version("sem-particle-analysis")
except (ImportError, PackageNotFoundError):  # not installed (e.g. run from source tree)
    __version__ = "unknown"
__all__ = [
    "SAMModel",
    "discover_checkpoints",
    "infer_model_type",
    "ScaleDetector",
    "OCRUnavailableError",
    "modality",
    "region",
    "ParticleSegmenter",
    "ParticleAnalyzer",
    "ResultsManager",
]
