"""Adaptador de SAM 2 para segmentación de imágenes rectificadas."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .base import SegmentationResult
from .prompts import pad_box


class SAM2UnavailableError(RuntimeError):
    """Indica que SAM 2 no puede cargarse con la instalación actual."""


class SAM2Segmenter:
    """Ejecuta SAM 2 con cajas y carga el modelo solo en el primer uso."""

    method_name = "sam2_hiera_tiny_prompted"

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        model_config: str = "sam2_hiera_t.yaml",
        device: str = "auto",
        box_margin_fraction: float = 0.05,
        min_component_area_px: int = 300,
        predictor: Any | None = None,
        model_builder: Callable[..., Any] | None = None,
        predictor_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.model_config = model_config
        self.device = device
        self.box_margin_fraction = box_margin_fraction
        self.min_component_area_px = min_component_area_px
        self._predictor = predictor
        self._model_builder = model_builder
        self._predictor_factory = predictor_factory

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError as exc:
            raise SAM2UnavailableError(
                "PyTorch no está instalado. Instale requirements-ai.txt."
            ) from exc

    def _ensure_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        if not self.checkpoint.exists():
            raise SAM2UnavailableError(
                f"No se encontró el checkpoint de SAM 2 en {self.checkpoint}."
            )
        try:
            if self._model_builder is None or self._predictor_factory is None:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                self._model_builder = build_sam2
                self._predictor_factory = SAM2ImagePredictor
            model = self._model_builder(
                self.model_config,
                str(self.checkpoint),
                device=self._resolve_device(),
            )
            self._predictor = self._predictor_factory(model)
        except SAM2UnavailableError:
            raise
        except Exception as exc:
            raise SAM2UnavailableError(
                "SAM 2 no pudo inicializarse. Revise dependencias, configuración y checkpoint."
            ) from exc
        return self._predictor

    @staticmethod
    def _select_mask(
        masks: np.ndarray,
        scores: np.ndarray,
        box_area: int,
    ) -> tuple[np.ndarray, float]:
        order = np.argsort(np.asarray(scores))[::-1]
        for index in order:
            area = int(np.count_nonzero(masks[index]))
            if 0.10 * box_area <= area <= 1.20 * box_area:
                return masks[index], float(scores[index])
        index = int(order[0])
        return masks[index], float(scores[index])

    def _postprocess(self, mask: np.ndarray) -> np.ndarray:
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        cleaned = np.zeros_like(binary)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= self.min_component_area_px:
                cleaned[labels == label] = 255
        return cleaned

    def segment(
        self,
        image: np.ndarray,
        *,
        prompt_boxes: list[tuple[int, int, int, int]],
    ) -> SegmentationResult:
        if not prompt_boxes:
            raise ValueError("SAM 2 necesita al menos una caja de prompt.")
        predictor = self._ensure_predictor()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)
        accumulated = np.zeros(image.shape[:2], dtype=np.uint8)
        padded_boxes: list[tuple[int, int, int, int]] = []
        selected_scores: list[float] = []
        for box in prompt_boxes:
            padded = pad_box(box, image.shape, self.box_margin_fraction)
            x1, y1, x2, y2 = padded
            box_area = max(1, (x2 - x1) * (y2 - y1))
            masks, scores, _ = predictor.predict(
                box=np.asarray(padded, dtype=np.float32),
                multimask_output=True,
            )
            selected, score = self._select_mask(masks, scores, box_area)
            accumulated[selected.astype(bool)] = 255
            padded_boxes.append(padded)
            selected_scores.append(score)

        return SegmentationResult(
            mask=self._postprocess(accumulated),
            method=self.method_name,
            prompt_boxes=padded_boxes,
            scores=selected_scores,
            metadata={
                "model_family": "SAM 2",
                "model_variant": "Hiera Tiny",
                "pretrained": True,
                "device": self._resolve_device(),
                "box_margin_fraction": self.box_margin_fraction,
            },
        )
