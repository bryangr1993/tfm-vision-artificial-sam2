"""Carga y valida la configuración operativa única de la aplicación."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_PIXELS_PER_MM = 10.0
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    sheet_width_mm: float
    sheet_height_mm: float
    pixels_per_mm: float
    marker_margin_mm: float


@dataclass(frozen=True, slots=True)
class SAM2Config:
    model_config: str
    checkpoint: Path
    box_margin_fraction: float
    multimask_output: bool
    min_component_area_px: int


@dataclass(frozen=True, slots=True)
class ExportConfig:
    offset_mm: float
    wcs_y_direction: str
    include_wcs_reference: bool
    bounds_tolerance_mm: float


@dataclass(frozen=True, slots=True)
class AppConfig:
    seed: int
    device: str
    operational_method: str
    geometry: GeometryConfig
    sam2: SAM2Config
    export: ExportConfig
    source_path: Path


def _mapping(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"La sección '{key}' debe ser un mapa YAML.")
    return value


def _positive_float(value: Any, key: str, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{key}' debe ser numérico.") from exc
    invalid = not math.isfinite(number) or (number < 0 if allow_zero else number <= 0)
    if invalid:
        qualifier = "mayor o igual que cero" if allow_zero else "mayor que cero"
        raise ValueError(f"'{key}' debe ser {qualifier}.")
    return number


def _boolean(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"'{key}' debe ser true o false.")
    return value


def validate_operational_scale(scale: float) -> float:
    """Acepta solo la escala para la que se validaron todos los umbrales."""

    numeric = _positive_float(scale, "geometry.pixels_per_mm")
    if abs(numeric - SUPPORTED_PIXELS_PER_MM) > 1e-9:
        raise ValueError(
            "La aplicación está validada únicamente a "
            f"{SUPPORTED_PIXELS_PER_MM:g} px/mm; se recibió {numeric:g} px/mm."
        )
    return numeric


def _resolve_repository_path(value: Any, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' debe contener una ruta.")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def load_app_config(path: str | Path | None = None) -> AppConfig:
    """Lee ``default.yaml`` (u otro archivo explícito) y valida su contrato."""

    source_path = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG_PATH.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"No se encontró la configuración: {source_path}")
    with source_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    root = _mapping(raw, "raíz")
    project = _mapping(root.get("project"), "project")
    geometry = _mapping(root.get("geometry"), "geometry")
    segmentation = _mapping(root.get("segmentation"), "segmentation")
    sam2 = _mapping(segmentation.get("sam2"), "segmentation.sam2")
    export = _mapping(root.get("export"), "export")

    pixels_per_mm = validate_operational_scale(geometry.get("pixels_per_mm"))
    device = str(project.get("device", "")).lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("'project.device' debe ser auto, cpu o cuda.")
    operational_method = str(segmentation.get("operational_method", "")).lower()
    if operational_method != "sam2":
        raise ValueError("El método operativo configurado debe ser 'sam2'.")
    y_direction = str(export.get("wcs_y_direction", "")).lower()
    if y_direction not in {"down", "up"}:
        raise ValueError("'export.wcs_y_direction' debe ser down o up.")
    box_margin = _positive_float(
        sam2.get("box_margin_fraction"),
        "segmentation.sam2.box_margin_fraction",
        allow_zero=True,
    )
    if box_margin > 0.5:
        raise ValueError("El margen de caja de SAM 2 no puede superar 0.5.")
    min_area = int(sam2.get("min_component_area_px"))
    if min_area <= 0:
        raise ValueError("'segmentation.sam2.min_component_area_px' debe ser positivo.")

    return AppConfig(
        seed=int(project.get("seed")),
        device=device,
        operational_method=operational_method,
        geometry=GeometryConfig(
            sheet_width_mm=_positive_float(
                geometry.get("sheet_width_mm"), "geometry.sheet_width_mm"
            ),
            sheet_height_mm=_positive_float(
                geometry.get("sheet_height_mm"), "geometry.sheet_height_mm"
            ),
            pixels_per_mm=pixels_per_mm,
            marker_margin_mm=_positive_float(
                geometry.get("marker_margin_mm"), "geometry.marker_margin_mm"
            ),
        ),
        sam2=SAM2Config(
            model_config=str(sam2.get("model_config")),
            checkpoint=_resolve_repository_path(
                sam2.get("checkpoint"), "segmentation.sam2.checkpoint"
            ),
            box_margin_fraction=box_margin,
            multimask_output=_boolean(
                sam2.get("multimask_output"),
                "segmentation.sam2.multimask_output",
            ),
            min_component_area_px=min_area,
        ),
        export=ExportConfig(
            offset_mm=_positive_float(
                export.get("offset_mm"), "export.offset_mm", allow_zero=True
            ),
            wcs_y_direction=y_direction,
            include_wcs_reference=_boolean(
                export.get("include_wcs_reference"),
                "export.include_wcs_reference",
            ),
            bounds_tolerance_mm=_positive_float(
                export.get("bounds_tolerance_mm"), "export.bounds_tolerance_mm"
            ),
        ),
        source_path=source_path,
    )
