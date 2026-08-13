"""Construcción de prompts espaciales a partir de una máscara aproximada."""

from __future__ import annotations

import cv2
import numpy as np


def boxes_from_mask(
    mask: np.ndarray,
    *,
    min_area_px: int = 20_000,
    max_area_px: int = 600_000,
    expected_instances: int = 8,
) -> list[tuple[int, int, int, int]]:
    """Obtiene cajas de componentes válidas sin usar su contorno como salida."""

    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    candidates: list[tuple[int, int, int, int, int, float, float]] = []
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area_px <= area <= max_area_px:
            cx, cy = centroids[label]
            candidates.append((x, y, x + width, y + height, area, float(cx), float(cy)))

    if len(candidates) > expected_instances:
        candidates = sorted(candidates, key=lambda item: item[4], reverse=True)[:expected_instances]

    candidates.sort(key=lambda item: (item[6], item[5]))
    return [(x1, y1, x2, y2) for x1, y1, x2, y2, *_ in candidates]


def pad_box(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
    fraction: float,
) -> tuple[int, int, int, int]:
    """Amplía una caja de forma proporcional y la recorta a la imagen."""

    x1, y1, x2, y2 = box
    height, width = image_shape[:2]
    pad_x = round((x2 - x1) * fraction)
    pad_y = round((y2 - y1) * fraction)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )
