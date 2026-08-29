"""Pipeline híbrido cuya salida operativa de segmentación procede de SAM 2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from extract_contours import extract_contours
from configuration import load_app_config
from segmenters import ClassicalPromptLocalizer, SAM2Segmenter, SegmentationResult


class HybridAISegmentationPipeline:
    """Encadena localización clásica y segmentación final con SAM 2."""

    def __init__(
        self,
        sam2_segmenter: SAM2Segmenter,
        prompt_localizer: ClassicalPromptLocalizer | None = None,
    ) -> None:
        self.prompt_localizer = prompt_localizer or ClassicalPromptLocalizer()
        self.sam2_segmenter = sam2_segmenter

    def segment(
        self,
        image: np.ndarray,
        *,
        scale: float = 10.0,
        wcs_info: dict | None = None,
        debug_dir: str | None = None,
        image_name: str = "image",
    ) -> SegmentationResult:
        prompt_result = self.prompt_localizer.segment(
            image,
            scale=scale,
            wcs_info=wcs_info,
            debug_dir=None,
            image_name=image_name,
        )
        result = self.sam2_segmenter.segment(
            image,
            prompt_boxes=prompt_result.prompt_boxes,
        )
        result.metadata.update(
            {
                "architecture": "hybrid_classical_localization_sam2_segmentation",
                "prompt_source": prompt_result.method,
                "prompt_count": len(prompt_result.prompt_boxes),
                "prompt_mask": prompt_result.mask,
            }
        )
        if debug_dir:
            self.save_debug(result, image, Path(debug_dir), image_name)
        return result

    @staticmethod
    def save_debug(
        result: SegmentationResult,
        image: np.ndarray,
        debug_dir: Path,
        image_name: str,
    ) -> None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"{image_name}_sam2_mask.png"), result.mask)
        prompt_mask = result.metadata.get("prompt_mask")
        if isinstance(prompt_mask, np.ndarray):
            cv2.imwrite(str(debug_dir / f"{image_name}_classical_prompt_mask.png"), prompt_mask)
        overlay = image.copy()
        for index, (x1, y1, x2, y2) in enumerate(result.prompt_boxes, start=1):
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 80, 0), 3)
            cv2.putText(
                overlay,
                str(index),
                (x1 + 5, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 80, 0),
                2,
            )
        cv2.imwrite(str(debug_dir / f"{image_name}_sam2_prompts.png"), overlay)


def default_checkpoint() -> Path:
    configured = os.environ.get("TFM_SAM2_CHECKPOINT")
    if configured:
        return Path(configured).expanduser().resolve()
    return load_app_config().sam2.checkpoint


def build_operational_pipeline(
    *,
    checkpoint: str | Path | None = None,
    model_config: str | None = None,
    device: str | None = None,
    box_margin_fraction: float | None = None,
    multimask_output: bool | None = None,
    min_component_area_px: int | None = None,
) -> HybridAISegmentationPipeline:
    """Construye el pipeline desde ``default.yaml`` salvo overrides explícitos."""

    config = load_app_config()

    sam2 = SAM2Segmenter(
        checkpoint or default_checkpoint(),
        model_config=model_config or config.sam2.model_config,
        device=device or config.device,
        box_margin_fraction=(
            config.sam2.box_margin_fraction
            if box_margin_fraction is None
            else box_margin_fraction
        ),
        multimask_output=(
            config.sam2.multimask_output
            if multimask_output is None
            else multimask_output
        ),
        min_component_area_px=(
            config.sam2.min_component_area_px
            if min_component_area_px is None
            else min_component_area_px
        ),
    )
    return HybridAISegmentationPipeline(sam2)


def segment_and_extract_with_ai(
    pipeline: HybridAISegmentationPipeline,
    image: np.ndarray,
    *,
    scale: float,
    wcs_info: dict | None,
    debug_dir: str | None,
    image_name: str,
    contour_extractor: Callable = extract_contours,
) -> tuple[SegmentationResult, list[np.ndarray], dict]:
    """Segmenta con SAM 2 y entrega esa misma máscara a los contornos."""

    result = pipeline.segment(
        image,
        scale=scale,
        wcs_info=wcs_info,
        debug_dir=debug_dir,
        image_name=image_name,
    )
    contours, report = contour_extractor(
        result.mask,
        image,
        debug_dir,
        image_name,
    )
    report["segmentation_method"] = result.method
    report["prompt_source"] = result.metadata.get("prompt_source")
    report["prompt_count"] = result.metadata.get("prompt_count")
    return result, contours, report
