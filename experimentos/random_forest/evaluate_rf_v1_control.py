"""Evalúa el RF v1 no aumentado como control del ajuste agrupado.

El modelo fue entrenado solo con las 36 láminas de entrenamiento. Este script
vuelve a seleccionar su umbral sobre las seis láminas de validación completas y
procesa las seis láminas de prueba una sola vez. De este modo puede compararse
con la búsqueda de hiperparámetros y con RF v2 bajo el mismo protocolo.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = ROOT / "datos" / "manifiesto" / "datasets.csv"
MODEL_PATH = ROOT / "resultados" / "modelos" / "random_forest_legacy_v1.joblib"
METRICS = ROOT / "resultados" / "metricas"
MASKS = ROOT / "resultados" / "rf_v1_selected_masks"
THRESHOLDS = np.round(np.arange(0.30, 0.76, 0.05), 2)

sys.path.insert(0, str(HERE))
from feature_extraction import extract_pixel_features
from optimize_rf import boundary_f1, component_count, dice_iou, write_csv
from postprocess_masks import postprocess_rf_mask


def rows_for(split: str) -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if row["domain"] == "synthetic" and row["split"] == split]


def probability(model, image: np.ndarray) -> np.ndarray:
    features, _ = extract_pixel_features(image, include_coords=False)
    return model.predict_proba(features)[:, 1].reshape(image.shape[:2])


def validation(model) -> tuple[list[dict], list[dict]]:
    details: list[dict] = []
    rows = rows_for("val")
    for index, row in enumerate(rows, start=1):
        image = cv2.imread(str(ROOT / row["image"]))
        reference = cv2.imread(str(ROOT / row["ground_truth"]), cv2.IMREAD_GRAYSCALE)
        probs = probability(model, image)
        expected = component_count(reference)
        for threshold in THRESHOLDS:
            prediction, _ = postprocess_rf_mask(
                (probs >= threshold).astype(np.uint8) * 255
            )
            dice, iou = dice_iou(reference, prediction)
            predicted = component_count(prediction)
            details.append(
                {
                    "model": "RF_v1_control_no_augmentation",
                    "threshold": float(threshold),
                    "sample_id": row["sample_id"],
                    "dice": dice,
                    "iou": iou,
                    "expected_components": expected,
                    "predicted_components": predicted,
                    "component_error": abs(predicted - expected),
                }
            )
        print(f"RF v1 validación [{index}/{len(rows)}] {row['sample_id']}")

    grouped: defaultdict[float, list[dict]] = defaultdict(list)
    for row in details:
        grouped[float(row["threshold"])].append(row)
    summary: list[dict] = []
    for threshold, values in grouped.items():
        mean_iou = float(np.mean([row["iou"] for row in values]))
        component_error = float(np.mean([row["component_error"] for row in values]))
        summary.append(
            {
                "model": "RF_v1_control_no_augmentation",
                "threshold": threshold,
                "mean_validation_iou": mean_iou,
                "std_validation_iou": float(np.std([row["iou"] for row in values], ddof=1)),
                "mean_validation_dice": float(np.mean([row["dice"] for row in values])),
                "mean_component_error": component_error,
                "selection_score": mean_iou - 0.005 * component_error,
            }
        )
    return details, sorted(summary, key=lambda row: row["selection_score"], reverse=True)


def test(model, threshold: float) -> list[dict]:
    MASKS.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    rows = rows_for("test")
    for index, row in enumerate(rows, start=1):
        image = cv2.imread(str(ROOT / row["image"]))
        reference = cv2.imread(str(ROOT / row["ground_truth"]), cv2.IMREAD_GRAYSCALE)
        start = time.perf_counter()
        probs = probability(model, image)
        prediction, _ = postprocess_rf_mask((probs >= threshold).astype(np.uint8) * 255)
        elapsed = time.perf_counter() - start
        dice, iou = dice_iou(reference, prediction)
        expected = component_count(reference)
        predicted = component_count(prediction)
        results.append(
            {
                "model": "RF_v1_control_no_augmentation",
                "sample_id": row["sample_id"],
                "split": "test",
                "threshold": threshold,
                "dice": dice,
                "iou": iou,
                "boundary_f1": boundary_f1(reference, prediction),
                "expected_components": expected,
                "predicted_components": predicted,
                "component_error": abs(predicted - expected),
                "inference_seconds": elapsed,
            }
        )
        cv2.imwrite(str(MASKS / f"{row['sample_id']}_rf_v1.png"), prediction)
        print(f"RF v1 prueba [{index}/{len(rows)}] {row['sample_id']}: IoU={iou:.5f}")
    return results


def main() -> None:
    package = joblib.load(MODEL_PATH)
    model = package["model"]
    details, summary = validation(model)
    write_csv(METRICS / "rf_v1_validation_threshold_per_sheet.csv", details)
    write_csv(METRICS / "rf_v1_validation_threshold_summary.csv", summary)
    winner = summary[0]
    test_rows = test(model, float(winner["threshold"]))
    write_csv(METRICS / "rf_v1_test_metrics.csv", test_rows)
    output = {
        "model": "RF_v1_control_no_augmentation",
        "origin": "pre-existing configuration retained as a protocol control",
        "parameters": package["parameters"],
        "selected_threshold": winner["threshold"],
        "selection_rule": "mean_validation_iou - 0.005 * mean_component_error",
        "validation": winner,
        "test_mean_iou": float(np.mean([row["iou"] for row in test_rows])),
        "test_std_iou": float(np.std([row["iou"] for row in test_rows], ddof=1)),
        "test_mean_dice": float(np.mean([row["dice"] for row in test_rows])),
        "test_mean_boundary_f1": float(np.mean([row["boundary_f1"] for row in test_rows])),
        "test_mean_component_error": float(np.mean([row["component_error"] for row in test_rows])),
    }
    (METRICS / "rf_v1_control_selection.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
