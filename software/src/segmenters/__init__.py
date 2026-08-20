"""Segmentadores intercambiables del pipeline de toppers."""

from .base import SegmentationResult
from .classical import ClassicalPromptLocalizer
from .sam2_segmenter import SAM2Segmenter, SAM2UnavailableError

__all__ = [
    "SegmentationResult",
    "ClassicalPromptLocalizer",
    "SAM2Segmenter",
    "SAM2UnavailableError",
]
