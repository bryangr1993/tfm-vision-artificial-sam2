"""Evalua la linea base clasica fija sobre TEST de ``asset_identity_v2``.

La rutina no realiza busqueda ni ajuste. Lee exclusivamente las doce laminas
de prueba, aplica la misma funcion ``segment_toppers`` utilizada por el
producto y calcula las definiciones de IoU, Dice, Boundary F1 y componentes
empleadas en los protocolos de RF y SAM 2.

Alcance temporal
----------------
Las imagenes se cargan antes de cronometrar. Se ejecuta un calentamiento no
registrado y cinco repeticiones por lamina. Cada repeticion comprende la
llamada completa a ``segment_toppers``. Se excluyen lectura de disco,
rectificacion y deteccion del WCS porque las laminas sinteticas ya estan en el
plano rectificado y no contienen la marca fisica.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = ROOT / "datos" / "manifiesto" / "datasets_asset_identity_v2.csv"
QUALITY_REPORT = ROOT / "resultados" / "metricas" / "dataset_identity_v2_quality.json"
METRICS = ROOT / "resultados" / "metricas"
MASKS = ROOT / "resultados" / "classical_idv2_test_masks"
DETAILS_CSV = METRICS / "classical_idv2_test_metrics.csv"
TIMING_CSV = METRICS / "classical_idv2_test_runtime_runs.csv"
SUMMARY_JSON = METRICS / "classical_idv2_test_summary.json"

WARMUP_RUNS = 1
MEASURED_REPETITIONS = 5
BOUNDARY_TOLERANCE_PX = 3
COMPONENT_MIN_AREA_PX = 20_000
OPERATIONAL_SCALE_PX_PER_MM = 10.0
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260827

sys.path.insert(0, str(ROOT / "software" / "src"))
from segment_toppers import segment_toppers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No hay filas para escribir en {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def binary_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def dice_iou(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    ref = reference > 0
    pred = prediction > 0
    intersection = int(np.count_nonzero(ref & pred))
    union = int(np.count_nonzero(ref | pred))
    denominator = int(np.count_nonzero(ref) + np.count_nonzero(pred))
    dice = 2.0 * intersection / denominator if denominator else 1.0
    iou = intersection / union if union else 1.0
    return float(dice), float(iou)


def boundary_f1(
    reference: np.ndarray,
    prediction: np.ndarray,
    tolerance: int = BOUNDARY_TOLERANCE_PX,
) -> float:
    kernel = np.ones((3, 3), np.uint8)
    ref = binary_mask(reference)
    pred = binary_mask(prediction)
    ref_edge = cv2.morphologyEx(ref, cv2.MORPH_GRADIENT, kernel) > 0
    pred_edge = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, kernel) > 0
    zone_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1,) * 2)
    ref_zone = cv2.dilate(ref_edge.astype(np.uint8), zone_kernel) > 0
    pred_zone = cv2.dilate(pred_edge.astype(np.uint8), zone_kernel) > 0
    precision = np.count_nonzero(pred_edge & ref_zone) / max(1, np.count_nonzero(pred_edge))
    recall = np.count_nonzero(ref_edge & pred_zone) / max(1, np.count_nonzero(ref_edge))
    return float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0


def component_count(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask(mask))
    return sum(
        int(stats[label, cv2.CC_STAT_AREA]) >= COMPONENT_MIN_AREA_PX
        for label in range(1, count)
    )


def bootstrap_mean_ci(values: list[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(data, size=(BOOTSTRAP_RESAMPLES, len(data)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def read_validated_test_rows() -> list[dict[str, str]]:
    quality = json.loads(QUALITY_REPORT.read_text(encoding="utf-8"))
    if not quality.get("passed") or int(quality.get("critical_failure_count", 1)) != 0:
        raise RuntimeError("La puerta de calidad de asset_identity_v2 no esta aprobada.")

    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    if Counter(row["split"] for row in all_rows) != Counter({"train": 24, "val": 12, "test": 12}):
        raise RuntimeError("Conteos de particion inesperados en asset_identity_v2.")

    rows = [row for row in all_rows if row["domain"] == "synthetic" and row["split"] == "test"]
    if len(rows) != 12 or len({row["sample_id"] for row in rows}) != 12:
        raise RuntimeError("TEST debe contener doce sample_id unicos.")
    if {row["layout"] for row in rows} != {"L1", "L2"}:
        raise RuntimeError("TEST no contiene ambos layouts.")
    if Counter(row["condition"] for row in rows) != Counter(
        {condition: 2 for condition in ("C1", "C2", "C3", "C4", "C5", "C6")}
    ):
        raise RuntimeError("TEST no contiene dos laminas por condicion C1-C6.")

    for row in rows:
        image_path = ROOT / row["image"]
        mask_path = ROOT / row["ground_truth"]
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(row["sample_id"])
        if sha256_file(image_path) != row["image_sha256"]:
            raise RuntimeError(f"Hash de imagen inesperado: {row['sample_id']}")
        if sha256_file(mask_path) != row["ground_truth_sha256"]:
            raise RuntimeError(f"Hash de mascara inesperado: {row['sample_id']}")
    return sorted(rows, key=lambda row: row["sample_id"])


def main() -> None:
    rows = read_validated_test_rows()
    loaded: list[tuple[dict[str, str], np.ndarray, np.ndarray]] = []
    for row in rows:
        image = cv2.imread(str(ROOT / row["image"]), cv2.IMREAD_COLOR)
        reference = cv2.imread(str(ROOT / row["ground_truth"]), cv2.IMREAD_GRAYSCALE)
        if image is None or reference is None:
            raise FileNotFoundError(row["sample_id"])
        if image.shape[:2] != (2970, 2100) or reference.shape != image.shape[:2]:
            raise RuntimeError(f"Dimensiones inesperadas: {row['sample_id']}")
        loaded.append((row, image, binary_mask(reference)))

    # Calentamiento no registrado. No se usa la salida para las metricas.
    segment_toppers(
        loaded[0][1],
        scale=OPERATIONAL_SCALE_PX_PER_MM,
        wcs_info=None,
    )

    details: list[dict] = []
    timing_rows: list[dict] = []
    MASKS.mkdir(parents=True, exist_ok=True)
    for index, (row, image, reference) in enumerate(loaded, start=1):
        predictions: list[np.ndarray] = []
        for repetition in range(1, MEASURED_REPETITIONS + 1):
            start = time.perf_counter()
            prediction = segment_toppers(
                image,
                scale=OPERATIONAL_SCALE_PX_PER_MM,
                wcs_info=None,
            )
            elapsed = time.perf_counter() - start
            predictions.append(prediction)
            timing_rows.append(
                {
                    "dataset_version": "asset_identity_v2",
                    "split": "test",
                    "sample_id": row["sample_id"],
                    "layout": row["layout"],
                    "condition": row["condition"],
                    "method": "vision_clasica_fija",
                    "repetition": repetition,
                    "segmentation_seconds": elapsed,
                    "input_width_px": image.shape[1],
                    "input_height_px": image.shape[0],
                    "device": "cpu",
                }
            )

        reference_prediction = predictions[0]
        if any(not np.array_equal(reference_prediction, other) for other in predictions[1:]):
            raise RuntimeError(f"La linea clasica no fue determinista en {row['sample_id']}")
        output_path = MASKS / f"{row['sample_id']}_vision_clasica_fija.png"
        if not cv2.imwrite(str(output_path), reference_prediction):
            raise OSError(f"No se pudo guardar {output_path}")

        dice, iou = dice_iou(reference, reference_prediction)
        expected = component_count(reference)
        predicted = component_count(reference_prediction)
        sample_times = [
            float(item["segmentation_seconds"])
            for item in timing_rows
            if item["sample_id"] == row["sample_id"]
        ]
        details.append(
            {
                "dataset_version": "asset_identity_v2",
                "split": "test",
                "sample_id": row["sample_id"],
                "layout": row["layout"],
                "condition": row["condition"],
                "method": "vision_clasica_fija",
                "dice": dice,
                "iou": iou,
                "boundary_f1": boundary_f1(reference, reference_prediction),
                "boundary_tolerance_px": BOUNDARY_TOLERANCE_PX,
                "expected_components": expected,
                "predicted_components": predicted,
                "component_error": abs(predicted - expected),
                "mean_segmentation_seconds": float(np.mean(sample_times)),
                "median_segmentation_seconds": float(np.median(sample_times)),
                "measured_repetitions": MEASURED_REPETITIONS,
                "prediction_path": output_path.relative_to(ROOT).as_posix(),
            }
        )
        print(
            f"Clasica TEST [{index:02d}/{len(loaded)}] {row['sample_id']}: "
            f"IoU={iou:.6f}, BF1={details[-1]['boundary_f1']:.6f}"
        )

    write_csv(DETAILS_CSV, details)
    write_csv(TIMING_CSV, timing_rows)
    ious = [float(row["iou"]) for row in details]
    ci_low, ci_high = bootstrap_mean_ci(ious)
    times = np.asarray([float(row["segmentation_seconds"]) for row in timing_rows])
    implementation = ROOT / "software" / "src" / "segment_toppers.py"
    summary = {
        "protocol_version": "classical_asset_identity_v2_test_v1",
        "status": "fixed_baseline_evaluated_on_locked_test",
        "method": "vision_clasica_fija",
        "selection": "fixed baseline; no training, validation tuning or test-time adjustment",
        "test_set_used_for_selection": False,
        "dataset_version": "asset_identity_v2",
        "split_protocol": "asset_identity_disjoint",
        "manifest": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST),
        "quality_report": QUALITY_REPORT.relative_to(ROOT).as_posix(),
        "quality_gate_passed": True,
        "sheet_count": len(details),
        "sample_ids": [row["sample_id"] for row in details],
        "layouts": sorted({row["layout"] for row in details}),
        "conditions": sorted({row["condition"] for row in details}),
        "metric_definitions": {
            "iou": "pixel intersection divided by pixel union",
            "dice": "twice pixel intersection divided by sum of foreground pixels",
            "boundary_f1": "symmetric boundary precision/recall with elliptical dilation tolerance",
            "boundary_tolerance_px": BOUNDARY_TOLERANCE_PX,
            "component_error": "absolute difference in connected components with area at least 20000 px",
            "component_min_area_px": COMPONENT_MIN_AREA_PX,
        },
        "test_mean_iou": float(np.mean(ious)),
        "test_std_iou": float(np.std(ious, ddof=1)),
        "test_iou_bootstrap_95_ci_low": ci_low,
        "test_iou_bootstrap_95_ci_high": ci_high,
        "test_mean_dice": float(np.mean([float(row["dice"]) for row in details])),
        "test_mean_boundary_f1": float(
            np.mean([float(row["boundary_f1"]) for row in details])
        ),
        "test_mean_component_error": float(
            np.mean([float(row["component_error"]) for row in details])
        ),
        "timing": {
            "status": "method_only_diagnostic_run",
            "scope": "complete segment_toppers call on preloaded 2100x2970 rectified images",
            "image_loading_excluded": True,
            "rectification_excluded": True,
            "wcs_detection_excluded": True,
            "model_loading": "not applicable",
            "warmup_runs": WARMUP_RUNS,
            "measured_repetitions_per_sheet": MEASURED_REPETITIONS,
            "measured_run_count": len(timing_rows),
            "mean_seconds": float(times.mean()),
            "std_seconds": float(times.std(ddof=1)),
            "median_seconds": float(np.median(times)),
            "p25_seconds": float(np.quantile(times, 0.25)),
            "p75_seconds": float(np.quantile(times, 0.75)),
            "min_seconds": float(times.min()),
            "max_seconds": float(times.max()),
            "device": "cpu",
            "cross_method_comparison_source": (
                "resultados/metricas/segmentation_runtime_idv2_final_summary.json"
            ),
        },
        "fixed_operational_parameters": {
            "scale_px_per_mm": OPERATIONAL_SCALE_PX_PER_MM,
            "wcs_info": None,
            "reason_wcs_absent": "synthetic canonical sheets contain no physical WCS mark",
        },
        "implementation": implementation.relative_to(ROOT).as_posix(),
        "implementation_sha256": sha256_file(implementation),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "details_csv": DETAILS_CSV.relative_to(ROOT).as_posix(),
        "timing_runs_csv": TIMING_CSV.relative_to(ROOT).as_posix(),
        "mask_directory": MASKS.relative_to(ROOT).as_posix(),
    }
    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
