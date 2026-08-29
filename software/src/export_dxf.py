"""Transformación y exportación DXF con verificación de lectura de retorno."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import uuid

import cv2
import ezdxf
import numpy as np

from configuration import validate_operational_scale

try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid

    SHAPELY_AVAILABLE = True
except ImportError:  # El offset cero sigue disponible en un entorno incompleto.
    Polygon = None
    make_valid = None
    SHAPELY_AVAILABLE = False


class DxfExportError(RuntimeError):
    """Error que impide entregar un DXF estructuralmente confiable."""


class OffsetDependencyError(DxfExportError):
    """El usuario solicitó offset sin disponer de Shapely."""


class DxfValidationError(DxfExportError):
    """El archivo guardado no supera las comprobaciones de retorno."""


@dataclass(frozen=True, slots=True)
class DxfValidationReport:
    path: str
    outer_paths: int
    hole_paths: int
    wcs_entities: int
    bounds_mm: tuple[float, float, float, float]

    def as_dict(self) -> dict:
        return asdict(self)


def _validate_wcs_info(wcs_info: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(wcs_info, dict) or wcs_info.get("status") != "SUCCESS":
        raise DxfExportError("El WCS no está en estado SUCCESS.")
    try:
        origin = np.asarray(wcs_info["origin"], dtype=float)
        unit_x = np.asarray(wcs_info["uX"], dtype=float)
        unit_y = np.asarray(wcs_info["uY"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise DxfExportError("El WCS no contiene origen y ejes numéricos completos.") from exc
    if any(vector.shape != (2,) for vector in (origin, unit_x, unit_y)):
        raise DxfExportError("El origen y los ejes del WCS deben tener dos componentes.")
    if not np.all(np.isfinite(np.concatenate((origin, unit_x, unit_y)))):
        raise DxfExportError("El WCS contiene valores no finitos.")
    norm_x = float(np.linalg.norm(unit_x))
    norm_y = float(np.linalg.norm(unit_y))
    if not (0.95 <= norm_x <= 1.05 and 0.95 <= norm_y <= 1.05):
        raise DxfExportError("Los ejes del WCS deben ser unitarios.")
    unit_x = unit_x / norm_x
    unit_y = unit_y / norm_y
    if abs(float(np.dot(unit_x, unit_y))) > 0.05:
        raise DxfExportError("Los ejes del WCS deben ser ortogonales.")
    return origin, unit_x, unit_y


def _validate_point_sequence(points) -> None:
    if len(points) < 3:
        raise DxfExportError("Una trayectoria cerrada necesita al menos tres vértices.")
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or not np.all(np.isfinite(array)):
        raise DxfExportError("La trayectoria contiene coordenadas inválidas.")
    if len({(round(float(x), 9), round(float(y), 9)) for x, y in array}) < 3:
        raise DxfExportError("La trayectoria no contiene tres vértices distintos.")
    x_values = array[:, 0]
    y_values = array[:, 1]
    signed_double_area = np.dot(x_values, np.roll(y_values, -1)) - np.dot(
        y_values, np.roll(x_values, -1)
    )
    if abs(float(signed_double_area)) <= 1e-9:
        raise DxfExportError("La trayectoria cerrada tiene área nula.")


def simplify_and_transform_contour(cnt, wcs_info, scale=10.0, y_direction="down"):
    """Simplifica y proyecta un contorno a milímetros relativos al WCS."""

    try:
        scale = validate_operational_scale(scale)
    except ValueError as exc:
        raise DxfExportError(str(exc)) from exc
    if y_direction not in {"down", "up"}:
        raise DxfExportError("La dirección Y debe ser 'down' o 'up'.")
    contour = np.asarray(cnt)
    if contour.ndim not in {2, 3} or contour.size < 6:
        raise DxfExportError("Cada contorno debe contener al menos tres puntos.")
    origin, unit_x, unit_y = _validate_wcs_info(wcs_info)

    simplified = cv2.approxPolyDP(contour, 0.15 * float(scale), closed=True)
    sign_y = 1.0 if y_direction == "down" else -1.0
    transformed_points = []
    for point in simplified.reshape(-1, 2):
        vector = np.asarray(point, dtype=float) - origin
        x_mm = float(np.dot(vector, unit_x) / scale)
        y_mm = float(np.dot(vector, unit_y) / scale) * sign_y
        transformed_points.append((x_mm, y_mm))
    _validate_point_sequence(transformed_points)
    return transformed_points


def apply_offset_shapely(points, offset_mm, *, role="outer"):
    """Aplica offset exterior a siluetas e interior a huecos, sin fallback oculto."""

    if not math.isfinite(float(offset_mm)) or offset_mm < 0:
        raise DxfExportError("El offset debe ser un número finito mayor o igual que cero.")
    if offset_mm == 0:
        return list(points)
    if role not in {"outer", "hole"}:
        raise DxfExportError(f"Rol de trayectoria desconocido: {role}")
    if not SHAPELY_AVAILABLE:
        raise OffsetDependencyError(
            "El offset requiere Shapely. Instale requirements-core.txt o use offset 0 mm."
        )
    _validate_point_sequence(points)
    try:
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = make_valid(polygon)
        if polygon.geom_type == "MultiPolygon":
            polygon = max(polygon.geoms, key=lambda item: item.area)
        if polygon.geom_type != "Polygon" or polygon.is_empty:
            raise DxfExportError("El contorno no forma un polígono válido para offset.")
        distance = float(offset_mm) if role == "outer" else -float(offset_mm)
        buffered = polygon.buffer(distance, join_style=1)
        if buffered.is_empty:
            raise DxfExportError(
                "El offset elimina por completo un hueco; reduzca la compensación."
            )
        if buffered.geom_type == "MultiPolygon":
            buffered = max(buffered.geoms, key=lambda item: item.area)
        if buffered.geom_type != "Polygon":
            raise DxfExportError("El offset produjo una geometría no exportable.")
        coordinates = list(buffered.exterior.coords)
        if len(coordinates) > 1 and np.allclose(coordinates[0], coordinates[-1]):
            coordinates = coordinates[:-1]
        result = [(float(x), float(y)) for x, y in coordinates]
        _validate_point_sequence(result)
        return result
    except DxfExportError:
        raise
    except Exception as exc:
        raise DxfExportError(f"No se pudo aplicar el offset: {exc}") from exc


def _add_wcs_reference(modelspace, y_direction: str) -> None:
    sign_y = 1.0 if y_direction == "down" else -1.0
    attributes = {"layer": "WCS_REF"}
    modelspace.add_line((0, 0), (20, 0), dxfattribs=attributes)
    modelspace.add_lwpolyline(
        [(18, 0.5), (20, 0), (18, -0.5)], close=False, dxfattribs=attributes
    )
    modelspace.add_line((0, 0), (0, 20 * sign_y), dxfattribs=attributes)
    modelspace.add_lwpolyline(
        [(0.5, 18 * sign_y), (0, 20 * sign_y), (-0.5, 18 * sign_y)],
        close=False,
        dxfattribs=attributes,
    )


def validate_dxf_file(
    path,
    *,
    expected_outer_count: int,
    expected_hole_count: int = 0,
    include_wcs_reference: bool = False,
    sheet_size_mm=(210.0, 297.0),
    bounds_tolerance_mm=50.0,
) -> DxfValidationReport:
    """Reabre el DXF y comprueba capas, cierre, conteos, finitud y límites."""

    try:
        document = ezdxf.readfile(str(path))
    except Exception as exc:
        raise DxfValidationError(f"El DXF no puede leerse de retorno: {exc}") from exc
    layer_names = {layer.dxf.name for layer in document.layers}
    if "TOPPERS_CUT" not in layer_names:
        raise DxfValidationError("Falta la capa TOPPERS_CUT.")
    if expected_hole_count and "TOPPERS_HOLES" not in layer_names:
        raise DxfValidationError("Falta la capa TOPPERS_HOLES.")
    if include_wcs_reference and "WCS_REF" not in layer_names:
        raise DxfValidationError("Falta la capa WCS_REF solicitada.")

    modelspace = document.modelspace()
    outer_entities = []
    hole_entities = []
    wcs_entities = []
    for entity in modelspace:
        layer = entity.dxf.layer
        if layer == "TOPPERS_CUT":
            outer_entities.append(entity)
        elif layer == "TOPPERS_HOLES":
            hole_entities.append(entity)
        elif layer == "WCS_REF":
            wcs_entities.append(entity)
    if len(outer_entities) != expected_outer_count:
        raise DxfValidationError(
            f"Se esperaban {expected_outer_count} siluetas y se leyeron {len(outer_entities)}."
        )
    if len(hole_entities) != expected_hole_count:
        raise DxfValidationError(
            f"Se esperaban {expected_hole_count} huecos y se leyeron {len(hole_entities)}."
        )
    if include_wcs_reference and len(wcs_entities) != 4:
        raise DxfValidationError("La referencia WCS debe contener cuatro entidades.")

    all_points = []
    for entity in outer_entities + hole_entities:
        if entity.dxftype() != "LWPOLYLINE":
            raise DxfValidationError("Las rutas de corte deben ser LWPOLYLINE.")
        if not entity.closed:
            raise DxfValidationError("Todas las rutas de corte deben estar cerradas.")
        points = [(float(item[0]), float(item[1])) for item in entity.get_points("xy")]
        try:
            _validate_point_sequence(points)
        except DxfExportError as exc:
            raise DxfValidationError(str(exc)) from exc
        all_points.extend(points)
    if not all_points:
        raise DxfValidationError("El DXF no contiene rutas de corte.")

    coordinates = np.asarray(all_points, dtype=float)
    min_x, min_y = np.min(coordinates, axis=0)
    max_x, max_y = np.max(coordinates, axis=0)
    width_mm, height_mm = map(float, sheet_size_mm)
    tolerance = float(bounds_tolerance_mm)
    if width_mm <= 0 or height_mm <= 0 or tolerance < 0:
        raise DxfValidationError("Los límites físicos de validación son inválidos.")
    if max(abs(min_x), abs(max_x)) > width_mm + tolerance:
        raise DxfValidationError("Una coordenada X excede el entorno físico permitido.")
    if max(abs(min_y), abs(max_y)) > height_mm + tolerance:
        raise DxfValidationError("Una coordenada Y excede el entorno físico permitido.")
    return DxfValidationReport(
        path=str(Path(path).resolve()),
        outer_paths=len(outer_entities),
        hole_paths=len(hole_entities),
        wcs_entities=len(wcs_entities),
        bounds_mm=(float(min_x), float(min_y), float(max_x), float(max_y)),
    )


def export_to_dxf(
    contours,
    wcs_info,
    output_path,
    scale=10.0,
    offset_mm=0.0,
    y_direction="down",
    *,
    contour_roles=None,
    include_wcs_reference=False,
    sheet_size_mm=(210.0, 297.0),
    bounds_tolerance_mm=50.0,
):
    """Exporta de forma atómica y solo entrega un archivo que supera el retorno."""

    destination = Path(output_path)
    temporary_path = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.tmp.dxf"
    )
    try:
        _validate_wcs_info(wcs_info)
        if not contours:
            raise DxfExportError("No hay contornos para exportar.")
        roles = list(contour_roles) if contour_roles is not None else ["outer"] * len(contours)
        if len(roles) != len(contours) or any(role not in {"outer", "hole"} for role in roles):
            raise DxfExportError("La lista de roles no corresponde con los contornos.")
        if offset_mm > 0 and not SHAPELY_AVAILABLE:
            raise OffsetDependencyError(
                "El offset requiere Shapely. Instale requirements-core.txt o use offset 0 mm."
            )

        document = ezdxf.new(dxfversion="R2010")
        document.layers.new(name="TOPPERS_CUT", dxfattribs={"color": 4})
        if "hole" in roles:
            document.layers.new(name="TOPPERS_HOLES", dxfattribs={"color": 3})
        if include_wcs_reference:
            document.layers.new(name="WCS_REF", dxfattribs={"color": 1})
        modelspace = document.modelspace()
        for contour, role in zip(contours, roles):
            points_mm = simplify_and_transform_contour(
                contour, wcs_info, scale, y_direction
            )
            if offset_mm > 0:
                points_mm = apply_offset_shapely(
                    points_mm, offset_mm, role=role
                )
            layer = "TOPPERS_HOLES" if role == "hole" else "TOPPERS_CUT"
            modelspace.add_lwpolyline(
                points_mm, close=True, dxfattribs={"layer": layer}
            )
        if include_wcs_reference:
            _add_wcs_reference(modelspace, y_direction)

        destination.parent.mkdir(parents=True, exist_ok=True)
        document.saveas(str(temporary_path))
        report = validate_dxf_file(
            temporary_path,
            expected_outer_count=roles.count("outer"),
            expected_hole_count=roles.count("hole"),
            include_wcs_reference=include_wcs_reference,
            sheet_size_mm=sheet_size_mm,
            bounds_tolerance_mm=bounds_tolerance_mm,
        )
        os.replace(temporary_path, destination)
        print(
            "  [export_dxf] DXF verificado y guardado: "
            f"{destination} ({report.outer_paths} siluetas, {report.hole_paths} huecos)"
        )
        return True
    except Exception as exc:
        if temporary_path.exists():
            temporary_path.unlink()
        print(f"  [export_dxf] Exportación bloqueada: {exc}")
        return False


def generate_validation_dxf(output_path, y_direction="down"):
    """Genera una pieza patrón para una validación física posterior en la máquina."""

    destination = Path(output_path)
    temporary_path = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.tmp.dxf"
    )
    try:
        if y_direction not in {"down", "up"}:
            raise DxfExportError("La dirección Y debe ser 'down' o 'up'.")
        document = ezdxf.new(dxfversion="R2010")
        document.layers.new(name="VALIDATION_AXES", dxfattribs={"color": 1})
        modelspace = document.modelspace()
        sign_y = 1.0 if y_direction == "down" else -1.0
        attributes = {"layer": "VALIDATION_AXES"}
        modelspace.add_line((0, 0), (20, 0), dxfattribs=attributes)
        modelspace.add_lwpolyline(
            [(18, 0.5), (20, 0), (18, -0.5)], close=False, dxfattribs=attributes
        )
        modelspace.add_line((0, 0), (0, 20 * sign_y), dxfattribs=attributes)
        modelspace.add_lwpolyline(
            [(0.5, 18 * sign_y), (0, 20 * sign_y), (-0.5, 18 * sign_y)],
            close=False,
            dxfattribs=attributes,
        )
        modelspace.add_lwpolyline(
            [(2, 2 * sign_y), (12, 2 * sign_y), (12, 12 * sign_y), (2, 12 * sign_y)],
            close=True,
            dxfattribs=attributes,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        document.saveas(str(temporary_path))
        reopened = ezdxf.readfile(str(temporary_path))
        entities = list(reopened.modelspace())
        if len(entities) != 5 or "VALIDATION_AXES" not in {
            layer.dxf.name for layer in reopened.layers
        }:
            raise DxfValidationError("El patrón de validación no supera la lectura de retorno.")
        os.replace(temporary_path, destination)
        print(f"  [export_dxf] Patrón para validación física guardado: {destination}")
        return True
    except Exception as exc:
        if temporary_path.exists():
            temporary_path.unlink()
        print(f"  [export_dxf] No se generó el patrón de validación: {exc}")
        return False
