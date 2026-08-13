"""Visión clásica empleada como localizador espacial y línea base."""

from __future__ import annotations

import numpy as np

from segment_toppers import segment_toppers

from .base import SegmentationResult
from .prompts import boxes_from_mask


class ClassicalPromptLocalizer:
    """Localiza aproximadamente los toppers para construir prompts de SAM 2."""

    method_name = "classical_prompt_localizer"

    def segment(
        self,
        image: np.ndarray,
        *,
        scale: float = 10.0,
        wcs_info: dict | None = None,
        debug_dir: str | None = None,
        image_name: str = "image",
    ) -> SegmentationResult:
        mask = segment_toppers(
            image,
            debug_dir=debug_dir,
            img_name=image_name,
            scale=scale,
            wcs_info=wcs_info,
        )
        boxes = boxes_from_mask(mask)
        if not boxes:
            raise RuntimeError("El localizador clásico no produjo ningún prompt válido.")
        return SegmentationResult(
            mask=mask,
            method=self.method_name,
            prompt_boxes=boxes,
            metadata={"role": "prompt_localization", "component_count": len(boxes)},
        )
