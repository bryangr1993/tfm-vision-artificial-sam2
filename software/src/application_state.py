"""Reglas de estado de la aplicación independientes de Tkinter."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any


def has_successful_wcs(wcs_info: dict[str, Any] | None) -> bool:
    """Indica si hay un sistema de coordenadas apto para intentar exportar."""

    if not wcs_info or wcs_info.get("status") != "SUCCESS":
        return False
    try:
        origin = tuple(float(value) for value in wcs_info["origin"])
        unit_x = tuple(float(value) for value in wcs_info["uX"])
        unit_y = tuple(float(value) for value in wcs_info["uY"])
    except (KeyError, TypeError, ValueError):
        return False
    if any(len(vector) != 2 for vector in (origin, unit_x, unit_y)):
        return False
    if not all(math.isfinite(value) for vector in (origin, unit_x, unit_y) for value in vector):
        return False
    norm_x = math.hypot(*unit_x)
    norm_y = math.hypot(*unit_y)
    if not (0.95 <= norm_x <= 1.05 and 0.95 <= norm_y <= 1.05):
        return False
    dot = (unit_x[0] * unit_y[0] + unit_x[1] * unit_y[1]) / (norm_x * norm_y)
    return abs(dot) <= 0.05


def can_export_dxf(
    *,
    processed: bool,
    wcs_info: dict[str, Any] | None,
    contours: Sequence[Any] | None,
) -> bool:
    """Centraliza la habilitación del botón y el bloqueo de la exportación."""

    return bool(processed and has_successful_wcs(wcs_info) and contours)
