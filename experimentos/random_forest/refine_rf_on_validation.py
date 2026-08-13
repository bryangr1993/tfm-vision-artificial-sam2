"""Selecciona modelo y umbral con máscaras completas de validación.

Esta etapa corrige la diferencia entre el objetivo de búsqueda por píxeles
muestreados y el uso final sobre una lámina completa. Compara el ganador de la
búsqueda agrupada con la configuración RF v2 previa. Solo después de bloquear
modelo y umbral evalúa el conjunto de prueba.
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
MODELS = ROOT / "resultados" / "modelos"
METRICS = ROOT / "resultados" / "metricas"
MASKS = ROOT / "resultados" / "rf_selected_masks"

sys.path.insert(0, str(HERE))
from feature_extraction import extract_pixel_features
from postprocess_masks import postprocess_rf_mask
from optimize_rf_v7 import boundary_f1, component_count, dice_iou, write_csv


THRESHOLDS = np.round(np.arange(0.30, 0.76, 0.05), 2)


def manifest_rows() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_models() -> dict[str, object]:
    optimized = joblib.load(MODELS / "random_forest_optimized.joblib")
    legacy = joblib.load(MODELS / "random_forest_legacy_v2.joblib")
    return {
        "RF_busqueda_agrupada": optimized["model"],
        "RF_v2_control": legacy["model"],
    }


def evaluate_validation(models: dict[str, object], rows: list[dict[str, str]]):
    validation = [row for row in rows if row["domain"] == "synthetic" and row["split"] == "val"]
    details: list[dict] = []
    for index, row in enumerate(validation, start=1):
        image = cv2.imread(str(ROOT / row["image"]))
        reference = cv2.imread(str(ROOT / row["ground_truth"]), cv2.IMREAD_GRAYSCALE)
        features, _ = extract_pixel_features(image, include_coords=False)
        expected = component_count(reference)
        for model_name, model in models.items():
            probability = model.predict_proba(features)[:, 1].reshape(reference.shape)
            for threshold in THRESHOLDS:
                prediction, _ = postprocess_rf_mask(
                    (probability >= threshold).astype(np.uint8) * 255
                )
                dice, iou = dice_iou(reference, prediction)
                predicted_components = component_count(prediction)
                details.append(
                    {
                        "model": model_name,
                        "threshold": float(threshold),
                        "sample_id": row["sample_id"],
                        "dice": dice,
                        "iou": iou,
                        "expected_components": expected,
                        "predicted_components": predicted_components,
                        "component_error": abs(predicted_components - expected),
                    }
                )
        print(f"Validación completa [{index}/{len(validation)}] {row['sample_id']}")
    return details


def summarize(details: list[dict]) -> list[dict]:
    grouped: defaultdict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in details:
        grouped[(row["model"], row["threshold"])].append(row)
    summary = []
    for (model, threshold), values in grouped.items():
        mean_iou = float(np.mean([row["iou"] for row in values]))
        mean_dice = float(np.mean([row["dice"] for row in values]))
        std_iou = float(np.std([row["iou"] for row in values], ddof=1))
        component_error = float(np.mean([row["component_error"] for row in values]))
        # IoU domina la decisión. El término secundario evita máscaras fragmentadas.
        score = mean_iou - 0.005 * component_error
        summary.append(
            {
                "model": model,
                "threshold": threshold,
                "mean_validation_iou": mean_iou,
                "std_validation_iou": std_iou,
                "mean_validation_dice": mean_dice,
                "mean_component_error": component_error,
                "selection_score": score,
            }
        )
    return sorted(summary, key=lambda row: row["selection_score"], reverse=True)


def evaluate_test(model_name, model, threshold, rows):
    test = [row for row in rows if row["domain"] == "synthetic" and row["split"] == "test"]
    MASKS.mkdir(parents=True, exist_ok=True)
    results = []
    for index, row in enumerate(test, start=1):
        image = cv2.imread(str(ROOT / row["image"]))
        reference = cv2.imread(str(ROOT / row["ground_truth"]), cv2.IMREAD_GRAYSCALE)
        start = time.perf_counter()
        features, _ = extract_pixel_features(image, include_coords=False)
        probability = model.predict_proba(features)[:, 1].reshape(reference.shape)
        prediction, _ = postprocess_rf_mask((probability >= threshold).astype(np.uint8) * 255)
        elapsed = time.perf_counter() - start
        dice, iou = dice_iou(reference, prediction)
        expected = component_count(reference)
        predicted = component_count(prediction)
        results.append(
            {
                "model": model_name,
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
        cv2.imwrite(str(MASKS / f"{row['sample_id']}_{model_name}.png"), prediction)
        print(f"Prueba final [{index}/{len(test)}] {row['sample_id']}: IoU={iou:.5f}")
    return results


def main() -> None:
    rows = manifest_rows()
    models = load_models()
    details = evaluate_validation(models, rows)
    summary = summarize(details)
    write_csv(METRICS / "rf_validation_model_threshold_per_sheet.csv", details)
    write_csv(METRICS / "rf_validation_model_threshold_summary.csv", summary)
    winner = summary[0]
    model_name = winner["model"]
    threshold = float(winner["threshold"])
    test_results = evaluate_test(model_name, models[model_name], threshold, rows)
    write_csv(METRICS / "rf_selected_test_metrics.csv", test_results)
    selection = {
        "selected_model": model_name,
        "selected_threshold": threshold,
        "selection_rule": "mean_validation_iou - 0.005 * mean_component_error",
        "validation": winner,
        "test_mean_iou": float(np.mean([row["iou"] for row in test_results])),
        "test_std_iou": float(np.std([row["iou"] for row in test_results], ddof=1)),
        "test_mean_dice": float(np.mean([row["dice"] for row in test_results])),
        "test_mean_boundary_f1": float(np.mean([row["boundary_f1"] for row in test_results])),
        "test_mean_component_error": float(np.mean([row["component_error"] for row in test_results])),
    }
    (METRICS / "rf_final_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(selection, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
