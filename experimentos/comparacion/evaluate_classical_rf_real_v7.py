"""Evalúa la línea clásica y el RF seleccionado en el dominio real."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RF_DIR = ROOT / "experimentos" / "random_forest"
MANIFEST = ROOT / "datos" / "manifiesto" / "datasets.csv"
MODEL_PATH = ROOT / "resultados" / "modelos" / "random_forest_legacy_v1.joblib"
RF_SELECTION = ROOT / "resultados" / "metricas" / "rf_v1_control_selection.json"
METRICS = ROOT / "resultados" / "metricas"
RF_MASKS = ROOT / "resultados" / "real_predictions" / "random_forest"
CLASSICAL_MASKS = ROOT / "resultados" / "real_predictions" / "clasico"

sys.path.insert(0, str(RF_DIR))
from feature_extraction import extract_pixel_features
from optimize_rf_v7 import boundary_f1, component_count, dice_iou, write_csv
from postprocess_masks import postprocess_rf_mask


def real_rows() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["domain"] == "real"]


def bootstrap_mean_ci(values: list[float], seed: int) -> list[float]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(10_000, len(data)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def summarize(method: str, results: list[dict], timing_note: str) -> dict:
    ious = [float(row["iou"]) for row in results]
    dices = [float(row["dice"]) for row in results]
    boundaries = [float(row["boundary_f1"]) for row in results]
    summary = {
        "method": method,
        "evaluation_domain": "13 rectified captures of one physical sheet",
        "mean_iou": float(np.mean(ious)),
        "std_iou": float(np.std(ious, ddof=1)),
        "iou_capture_bootstrap_95_ci": bootstrap_mean_ci(ious, 51),
        "mean_dice": float(np.mean(dices)),
        "std_dice": float(np.std(dices, ddof=1)),
        "dice_capture_bootstrap_95_ci": bootstrap_mean_ci(dices, 52),
        "mean_boundary_f1": float(np.mean(boundaries)),
        "mean_component_error": float(np.mean([row["component_error"] for row in results])),
        "timing_note": timing_note,
        "interpretation_limit": (
            "Capture-level variation refers to repeated views of one sheet, not new designs."
        ),
    }
    times = [float(row["inference_seconds"]) for row in results if row["inference_seconds"] != ""]
    if times:
        summary["mean_inference_seconds"] = float(np.mean(times))
    return summary


def main() -> None:
    selection = json.loads(RF_SELECTION.read_text(encoding="utf-8"))
    threshold = float(selection["selected_threshold"])
    model = joblib.load(MODEL_PATH)["model"]
    RF_MASKS.mkdir(parents=True, exist_ok=True)
    CLASSICAL_MASKS.mkdir(parents=True, exist_ok=True)

    classical_results: list[dict] = []
    rf_results: list[dict] = []
    rows = real_rows()
    for index, row in enumerate(rows, start=1):
        image = cv2.imread(str(ROOT / row["image"]))
        reference = cv2.imread(str(ROOT / row["ground_truth"]), cv2.IMREAD_GRAYSCALE)
        classical = cv2.imread(str(ROOT / row["legacy_reference"]), cv2.IMREAD_GRAYSCALE)
        classical_dice, classical_iou = dice_iou(reference, classical)
        expected = component_count(reference)
        classical_components = component_count(classical)
        classical_results.append(
            {
                "domain": "real",
                "sample_id": row["sample_id"],
                "method": "vision_clasica",
                "dice": classical_dice,
                "iou": classical_iou,
                "boundary_f1": boundary_f1(reference, classical),
                "expected_components": expected,
                "predicted_components": classical_components,
                "component_error": abs(classical_components - expected),
                "inference_seconds": "",
            }
        )
        cv2.imwrite(str(CLASSICAL_MASKS / f"{row['sample_id']}_clasico.png"), classical)

        start = time.perf_counter()
        features, _ = extract_pixel_features(image, include_coords=False)
        probability = model.predict_proba(features)[:, 1].reshape(reference.shape)
        prediction, _ = postprocess_rf_mask(
            (probability >= threshold).astype(np.uint8) * 255
        )
        elapsed = time.perf_counter() - start
        rf_dice, rf_iou = dice_iou(reference, prediction)
        rf_components = component_count(prediction)
        rf_results.append(
            {
                "domain": "real",
                "sample_id": row["sample_id"],
                "method": "Random Forest v1 seleccionado",
                "threshold": threshold,
                "dice": rf_dice,
                "iou": rf_iou,
                "boundary_f1": boundary_f1(reference, prediction),
                "expected_components": expected,
                "predicted_components": rf_components,
                "component_error": abs(rf_components - expected),
                "inference_seconds": elapsed,
            }
        )
        cv2.imwrite(str(RF_MASKS / f"{row['sample_id']}_rf_v1.png"), prediction)
        print(
            f"Real [{index}/{len(rows)}] {row['sample_id']}: "
            f"clásico IoU={classical_iou:.5f}; RF IoU={rf_iou:.5f}"
        )

    write_csv(METRICS / "classical_real_metrics.csv", classical_results)
    write_csv(METRICS / "rf_selected_real_metrics.csv", rf_results)
    summaries = {
        "classical": summarize(
            "vision_clasica",
            classical_results,
            "Masks were generated by the pre-existing baseline execution; time is not recomputed here.",
        ),
        "random_forest": summarize(
            "Random Forest v1 selected on synthetic validation",
            rf_results,
            "Feature extraction, probability prediction and postprocessing measured together.",
        ),
    }
    (METRICS / "classical_rf_real_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
