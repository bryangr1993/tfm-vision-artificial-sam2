"""Evalúa la línea clásica sobre el conjunto sintético de prueba bloqueado."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = ROOT / "datos" / "manifiesto" / "datasets.csv"
METRICS = ROOT / "resultados" / "metricas"
MASKS = ROOT / "resultados" / "classical_test_masks"

sys.path.insert(0, str(ROOT / "software" / "src"))
sys.path.insert(0, str(ROOT / "experimentos" / "random_forest"))
from optimize_rf import boundary_f1, component_count, dice_iou, write_csv
from segment_toppers import segment_toppers


def main() -> None:
    with MANIFEST.open("r", encoding="utf-8", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["domain"] == "synthetic" and row["split"] == "test"
        ]
    MASKS.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for index, row in enumerate(rows, start=1):
        image = cv2.imread(str(ROOT / row["image"]))
        reference = cv2.imread(str(ROOT / row["ground_truth"]), cv2.IMREAD_GRAYSCALE)
        start = time.perf_counter()
        prediction = segment_toppers(image, scale=10.0, wcs_info=None)
        elapsed = time.perf_counter() - start
        dice, iou = dice_iou(reference, prediction)
        expected = component_count(reference)
        predicted = component_count(prediction)
        results.append(
            {
                "method": "vision_clasica",
                "sample_id": row["sample_id"],
                "split": "test",
                "dice": dice,
                "iou": iou,
                "boundary_f1": boundary_f1(reference, prediction),
                "expected_components": expected,
                "predicted_components": predicted,
                "component_error": abs(predicted - expected),
                "inference_seconds": elapsed,
            }
        )
        cv2.imwrite(str(MASKS / f"{row['sample_id']}_clasico.png"), prediction)
        print(f"Clásico prueba [{index}/{len(rows)}] {row['sample_id']}: IoU={iou:.5f}")
    write_csv(METRICS / "classical_test_metrics.csv", results)
    summary = {
        "method": "vision_clasica",
        "selection": "fixed baseline; no training or validation selection",
        "test_mean_iou": float(np.mean([row["iou"] for row in results])),
        "test_std_iou": float(np.std([row["iou"] for row in results], ddof=1)),
        "test_mean_dice": float(np.mean([row["dice"] for row in results])),
        "test_mean_boundary_f1": float(np.mean([row["boundary_f1"] for row in results])),
        "test_mean_component_error": float(np.mean([row["component_error"] for row in results])),
        "test_mean_inference_seconds": float(np.mean([row["inference_seconds"] for row in results])),
    }
    (METRICS / "classical_test_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
