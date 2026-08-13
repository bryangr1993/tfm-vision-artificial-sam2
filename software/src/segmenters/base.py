"""Contratos comunes para los métodos de segmentación."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SegmentationResult:
    """Salida normalizada de un segmentador.

    La máscara siempre es binaria, con fondo 0 y primer plano 255. Los cuadros
    usan el convenio ``(x_min, y_min, x_max, y_max)`` con límites exclusivos.
    """

    mask: np.ndarray
    method: str
    prompt_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError("La máscara de segmentación debe tener dos dimensiones.")
        self.mask = np.where(self.mask > 0, 255, 0).astype(np.uint8)
