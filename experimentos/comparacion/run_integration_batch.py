"""Ejecuta el flujo operativo completo sobre las trece capturas reales.

Las diez capturas con marca WCS deben producir un DXF legible y estructuralmente
coherente. Las tres capturas sin marca deben permanecer en modo de análisis y no
deben producir DXF. La validación es geométrica y de estructura del archivo; no
equivale a una prueba física de corte ni a una importación en RDWorks.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import cv2
import ezdxf
import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SAM_DIR = ROOT / "experimentos" / "sam2"
OUTPUT = ROOT / "resultados" / "integracion_lote_v8"
METRICS = ROOT / "resultados" / "metricas"
FIGURES = ROOT / "resultados" / "figuras_protocolo_v8"
SAM_SELECTION = ROOT / "resultados" / "metricas" / "sam2_idv2_prompt_selection.json"

sys.path.insert(0, str(SAM_DIR))
sys.path.insert(0, str(ROOT / "software" / "src"))
from ai_pipeline import build_operational_pipeline, segment_and_extract_with_ai
from detect_markers import detect_markers
from detect_wcs_l import detect_wcs_l
from export_dxf import export_to_dxf
from load_images import load_image
from protocol_common import load_manifest, read_required_image, sample_metrics, write_csv
from rectify_sheet import rectify_sheet


EXPECTED_WCS_IDS = {f"real_{number}" for number in range(20, 40, 2)}


def validate_dxf(path: Path, expected_outer: int, expected_holes: int) -> dict:
    try:
        document = ezdxf.readfile(path)
        outer_entities = [
            entity
            for entity in document.modelspace()
            if entity.dxftype() == "LWPOLYLINE" and entity.dxf.layer == "TOPPERS_CUT"
        ]
        hole_entities = [
            entity
            for entity in document.modelspace()
            if entity.dxftype() == "LWPOLYLINE" and entity.dxf.layer == "TOPPERS_HOLES"
        ]
        entities = outer_entities + hole_entities
        points = [point for entity in entities for point in entity.get_points("xy")]
        finite = all(math.isfinite(float(x)) and math.isfinite(float(y)) for x, y in points)
        closed = all(bool(entity.closed) for entity in entities)
        if points:
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            bounds = [min(xs), min(ys), max(xs), max(ys)]
            plausible_bounds = (
                -50.0 <= bounds[0] <= 250.0
                and -50.0 <= bounds[1] <= 350.0
                and -50.0 <= bounds[2] <= 250.0
                and -50.0 <= bounds[3] <= 350.0
            )
        else:
            bounds = []
            plausible_bounds = False
        return {
            "roundtrip_readable": True,
            "outer_polyline_count": len(outer_entities),
            "hole_polyline_count": len(hole_entities),
            "polyline_count": len(entities),
            "polyline_count_matches_contours": (
                len(outer_entities) == expected_outer and len(hole_entities) == expected_holes
            ),
            "all_polylines_closed": closed,
            "all_coordinates_finite": finite,
            "bounds_mm": bounds,
            "bounds_plausible_for_a4_context": plausible_bounds,
            "structurally_valid": (
                len(outer_entities) == expected_outer
                and len(hole_entities) == expected_holes
                and expected_outer > 0
                and closed
                and finite
                and plausible_bounds
            ),
            "error": "",
        }
    except Exception as exc:
        return {
            "roundtrip_readable": False,
            "outer_polyline_count": 0,
            "hole_polyline_count": 0,
            "polyline_count": 0,
            "polyline_count_matches_contours": False,
            "all_polylines_closed": False,
            "all_coordinates_finite": False,
            "bounds_mm": [],
            "bounds_plausible_for_a4_context": False,
            "structurally_valid": False,
            "error": repr(exc),
        }


def plot_status(rows: list[dict]) -> None:
    checks = [
        ("Marcadores 4/4", "markers_ok"),
        ("Clasificación WCS", "wcs_classification_correct"),
        ("8 siluetas externas", "external_silhouette_policy_ok"),
        ("Política DXF", "dxf_policy_correct"),
        ("DXF estructural", "dxf_structurally_valid_or_not_applicable"),
    ]
    values = np.asarray([[1.0 if bool(row[key]) else 0.0 for _, key in checks] for row in rows])
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    cmap = plt.matplotlib.colors.ListedColormap(["#D5822A", "#2F6B9A"])
    ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(checks)), [label for label, _ in checks], rotation=24, ha="right")
    ax.set_yticks(range(len(rows)), [row["sample_id"] for row in rows])
    fig.suptitle("Comprobaciones del lote operativo de trece capturas", fontsize=10.5)
    ax.set_title(
        "Azul: conforme. En capturas sin WCS, no exportar DXF es conforme.",
        fontsize=8.5,
        color="#555555",
        pad=9,
    )
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            ax.text(
                column_index,
                row_index,
                "OK" if values[row_index, column_index] == 1 else "NO",
                ha="center",
                va="center",
                color="white" if values[row_index, column_index] == 1 else "#262626",
                fontsize=8.5,
                fontweight="bold",
            )
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "integration_batch_status_v8.png", dpi=300, facecolor="white")
    fig.savefig(FIGURES / "integration_batch_status_v8.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero no está disponible.")

    sam_selection = json.loads(SAM_SELECTION.read_text(encoding="utf-8"))
    sam_margin = float(sam_selection["selected_box_margin_fraction"])
    pipeline = build_operational_pipeline(device=device, box_margin_fraction=sam_margin)
    rows = sorted(
        [row for row in load_manifest() if row["domain"] == "real"],
        key=lambda row: int(row["sample_id"].rsplit("_", 1)[-1]),
    )
    mask_dir = OUTPUT / "masks"
    dxf_dir = OUTPUT / "dxf"
    contour_dir = OUTPUT / "contours"
    mask_dir.mkdir(parents=True, exist_ok=True)
    dxf_dir.mkdir(parents=True, exist_ok=True)
    contour_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    dxf_details: dict[str, dict] = {}
    for index, row in enumerate(rows, start=1):
        total_start = time.perf_counter()
        sample_id = row["sample_id"]
        expected_wcs = sample_id in EXPECTED_WCS_IDS

        start = time.perf_counter()
        raw = load_image(str(ROOT / row["raw_image"]), 0)
        load_seconds = time.perf_counter() - start

        start = time.perf_counter()
        markers = detect_markers(raw, None, sample_id)
        marker_seconds = time.perf_counter() - start
        if markers is None or len(markers) != 4:
            results.append(
                {
                    "sample_id": sample_id,
                    "expected_wcs": expected_wcs,
                    "markers_detected": 0 if markers is None else len(markers),
                    "markers_ok": False,
                    "wcs_status": "NOT_RUN",
                    "wcs_classification_correct": False,
                    "segmentation_method": "NOT_RUN",
                    "prompt_count": 0,
                    "topper_count": 0,
                    "outer_contour_count": 0,
                    "cut_path_count": 0,
                    "hole_path_count": 0,
                    "eight_toppers": False,
                    "external_silhouette_policy_ok": False,
                    "dxf_created": False,
                    "dxf_policy_correct": not expected_wcs,
                    "dxf_structurally_valid": False,
                    "dxf_structurally_valid_or_not_applicable": not expected_wcs,
                    "iou_vs_assisted_reference": "",
                    "dice_vs_assisted_reference": "",
                    "boundary_f1_vs_assisted_reference": "",
                    "load_seconds": load_seconds,
                    "marker_seconds": marker_seconds,
                    "rectification_seconds": 0.0,
                    "wcs_seconds": 0.0,
                    "segmentation_and_contours_seconds": 0.0,
                    "dxf_seconds": 0.0,
                    "total_seconds": time.perf_counter() - total_start,
                    "error": "four_markers_not_found",
                }
            )
            continue

        start = time.perf_counter()
        rectified, _ = rectify_sheet(
            raw,
            markers,
            None,
            sample_id,
            sheet_size=(210.0, 297.0),
            scale=10.0,
            marker_margin=10.0,
        )
        rectification_seconds = time.perf_counter() - start

        start = time.perf_counter()
        wcs = detect_wcs_l(
            rectified,
            debug_dir=None,
            img_name=sample_id,
            marker_margin=10.0,
            scale=10.0,
        )
        wcs_seconds = time.perf_counter() - start
        wcs_detected = wcs["status"] == "SUCCESS"

        start = time.perf_counter()
        segmentation, contours, contour_report = segment_and_extract_with_ai(
            pipeline,
            rectified,
            scale=10.0,
            wcs_info=wcs,
            debug_dir=None,
            image_name=sample_id,
        )
        segmentation_seconds = time.perf_counter() - start
        cv2.imwrite(str(mask_dir / f"{sample_id}_sam2_operational.png"), segmentation.mask)
        overlay = rectified.copy()
        cv2.drawContours(overlay, contours, -1, (0, 170, 0), 4)
        cv2.imwrite(str(contour_dir / f"{sample_id}_contours.png"), overlay)

        dxf_created = False
        dxf_validation = {
            "structurally_valid": False,
            "roundtrip_readable": False,
            "error": "not_applicable_without_wcs",
        }
        start = time.perf_counter()
        if wcs_detected:
            dxf_path = dxf_dir / f"{sample_id}_sam2_operational.dxf"
            dxf_created = export_to_dxf(
                contours,
                wcs,
                str(dxf_path),
                10.0,
                0.0,
                "down",
                contour_roles=contour_report["path_roles"],
            )
            if dxf_created:
                dxf_validation = validate_dxf(
                    dxf_path,
                    int(contour_report["outer_contours_count"]),
                    int(contour_report["holes_preserved_count"]),
                )
                dxf_details[sample_id] = dxf_validation
        dxf_seconds = time.perf_counter() - start

        reference = read_required_image(ROOT / row["ground_truth"], cv2.IMREAD_GRAYSCALE)
        quality = sample_metrics(reference, segmentation.mask)
        outer_count = int(contour_report["outer_contours_count"])
        hole_count = int(contour_report["holes_preserved_count"])
        cut_path_count = len(contours)
        external_silhouette_policy_ok = (
            outer_count == 8 and hole_count == 0 and cut_path_count == 8
        )
        dxf_policy_correct = (expected_wcs and dxf_created) or (not expected_wcs and not dxf_created)
        structural_or_na = (
            bool(dxf_validation.get("structurally_valid")) if expected_wcs else not dxf_created
        )
        results.append(
            {
                "sample_id": sample_id,
                "expected_wcs": expected_wcs,
                "markers_detected": len(markers),
                "markers_ok": len(markers) == 4,
                "wcs_status": wcs["status"],
                "wcs_classification_correct": wcs_detected == expected_wcs,
                "segmentation_method": segmentation.method,
                "prompt_count": len(segmentation.prompt_boxes),
                "topper_count": int(contour_report["toppers_detected"]),
                "outer_contour_count": outer_count,
                "cut_path_count": cut_path_count,
                "hole_path_count": hole_count,
                "eight_toppers": int(contour_report["toppers_detected"]) == 8,
                "external_silhouette_policy_ok": external_silhouette_policy_ok,
                "dxf_created": dxf_created,
                "dxf_policy_correct": dxf_policy_correct,
                "dxf_structurally_valid": bool(dxf_validation.get("structurally_valid")),
                "dxf_structurally_valid_or_not_applicable": structural_or_na,
                "iou_vs_assisted_reference": quality["iou"],
                "dice_vs_assisted_reference": quality["dice"],
                "boundary_f1_vs_assisted_reference": quality["boundary_f1"],
                "load_seconds": load_seconds,
                "marker_seconds": marker_seconds,
                "rectification_seconds": rectification_seconds,
                "wcs_seconds": wcs_seconds,
                "segmentation_and_contours_seconds": segmentation_seconds,
                "dxf_seconds": dxf_seconds,
                "total_seconds": time.perf_counter() - total_start,
                "error": dxf_validation.get("error", "") if expected_wcs else "",
            }
        )
        print(f"Integración [{index}/{len(rows)}] {sample_id}")

    METRICS.mkdir(parents=True, exist_ok=True)
    write_csv(METRICS / "integration_batch_v8.csv", results)
    summary = {
        "protocol_version": "integration_batch_v8",
        "n_captures": len(results),
        "expected_wcs_captures": sum(bool(row["expected_wcs"]) for row in results),
        "expected_no_wcs_captures": sum(not bool(row["expected_wcs"]) for row in results),
        "markers_4_of_4": sum(bool(row["markers_ok"]) for row in results),
        "correct_wcs_classifications": sum(
            bool(row["wcs_classification_correct"]) for row in results
        ),
        "captures_with_8_toppers": sum(bool(row["eight_toppers"]) for row in results),
        "captures_with_8_external_paths_and_no_holes": sum(
            bool(row["external_silhouette_policy_ok"]) for row in results
        ),
        "hole_paths_per_capture": {
            row["sample_id"]: int(row["hole_path_count"]) for row in results
        },
        "correct_dxf_policy": sum(bool(row["dxf_policy_correct"]) for row in results),
        "structurally_valid_dxf_expected_or_not_applicable": sum(
            bool(row["dxf_structurally_valid_or_not_applicable"]) for row in results
        ),
        "all_checks_pass": all(
            row["markers_ok"]
            and row["wcs_classification_correct"]
            and row["external_silhouette_policy_ok"]
            and row["dxf_policy_correct"]
            and row["dxf_structurally_valid_or_not_applicable"]
            for row in results
        ),
        "mean_total_seconds": float(np.mean([float(row["total_seconds"]) for row in results])),
        "device": device,
        "sam2_box_margin_fraction": sam_margin,
        "sam2_margin_selection": str(SAM_SELECTION.relative_to(ROOT)).replace("\\", "/"),
        "contour_export_policy": (
            "external silhouettes only: 8 outer paths, internal printed details excluded"
        ),
        "hole_support": "available as an explicit opt-in, disabled for this product",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "dxf_validation_scope": (
            "ezdxf round-trip, layer/polyline count, closure, finite coordinates and plausible bounds"
        ),
        "dxf_validation_limit": (
            "No physical laser cut and no RDWorks import were performed; the result is structural, not physical."
        ),
        "wcs_validation_limit": (
            "SUCCESS denotes geometric detection and is not treated as independent metrological validation."
        ),
        "reference_limit": (
            "Quality metrics use an algorithmically assisted canonical reference, not an independent manual annotation."
        ),
        "dxf_details": dxf_details,
    }
    (METRICS / "integration_batch_summary_v8.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_status(results)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
