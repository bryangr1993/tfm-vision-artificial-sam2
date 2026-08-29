"""Selección de prompts de SAM 2 en validación y evaluación final en prueba."""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = ROOT / "datos" / "manifiesto" / "datasets.csv"
PROMPTS = ROOT / "datos" / "sinteticos" / "prompts"
CHECKPOINT = ROOT / "resultados" / "modelos" / "sam2_hiera_tiny.pt"
METRICS = ROOT / "resultados" / "metricas"
MASKS = ROOT / "resultados" / "sam2_selected_masks"

sys.path.insert(0, str(ROOT / "software" / "src"))
from segmenters.sam2_segmenter import SAM2Segmenter

STRATEGIES = {
    "caja": {"margin": 0.00, "points": "none"},
    "caja_margen_5": {"margin": 0.05, "points": "none"},
    "caja_margen_10": {"margin": 0.10, "points": "none"},
    "caja_punto_positivo": {"margin": 0.00, "points": "positive"},
    "caja_puntos_positivos_negativos": {"margin": 0.00, "points": "positive_negative"},
}


def dice_iou(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    ref = reference > 0
    pred = prediction > 0
    intersection = int(np.count_nonzero(ref & pred))
    union = int(np.count_nonzero(ref | pred))
    denominator = int(np.count_nonzero(ref) + np.count_nonzero(pred))
    return (
        2.0 * intersection / denominator if denominator else 1.0,
        intersection / union if union else 1.0,
    )


def boundary_f1(reference: np.ndarray, prediction: np.ndarray, tolerance: int = 3) -> float:
    kernel = np.ones((3, 3), np.uint8)
    ref = np.where(reference > 0, 255, 0).astype(np.uint8)
    pred = np.where(prediction > 0, 255, 0).astype(np.uint8)
    ref_edge = cv2.morphologyEx(ref, cv2.MORPH_GRADIENT, kernel) > 0
    pred_edge = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, kernel) > 0
    zone_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1,) * 2)
    ref_zone = cv2.dilate(ref_edge.astype(np.uint8), zone_kernel) > 0
    pred_zone = cv2.dilate(pred_edge.astype(np.uint8), zone_kernel) > 0
    precision = np.count_nonzero(pred_edge & ref_zone) / max(1, np.count_nonzero(pred_edge))
    recall = np.count_nonzero(ref_edge & pred_zone) / max(1, np.count_nonzero(ref_edge))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def component_count(mask: np.ndarray, min_area: int = 20_000) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return sum(stats[label, cv2.CC_STAT_AREA] >= min_area for label in range(1, count))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def manifest_rows() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def pad_box(box, shape, fraction):
    x1, y1, x2, y2 = map(int, box)
    height, width = shape[:2]
    pad_x = round((x2 - x1) * fraction)
    pad_y = round((y2 - y1) * fraction)
    return np.asarray(
        [max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)],
        dtype=np.float32,
    )


def postprocess(mask: np.ndarray) -> np.ndarray:
    result = np.where(mask > 0, 255, 0).astype(np.uint8)
    result = cv2.morphologyEx(
        result, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(result)
    cleaned = np.zeros_like(result)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= 300:
            cleaned[labels == label] = 255
    return cleaned


def predict_strategy(predictor, image, prompts, strategy):
    settings = STRATEGIES[strategy]
    start = time.perf_counter()
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    accumulated = np.zeros(image.shape[:2], dtype=np.uint8)
    scores = []
    for prompt in prompts:
        original_box = np.asarray(prompt["bbox"], dtype=np.float32)
        box = pad_box(original_box, image.shape, settings["margin"])
        kwargs = {"box": box, "multimask_output": True}
        if settings["points"] == "positive":
            kwargs["point_coords"] = np.asarray([prompt["positive_point"]], dtype=np.float32)
            kwargs["point_labels"] = np.asarray([1], dtype=np.int32)
        elif settings["points"] == "positive_negative":
            kwargs["point_coords"] = np.asarray(
                [prompt["positive_point"], *prompt["negative_points"]], dtype=np.float32
            )
            kwargs["point_labels"] = np.asarray([1, 0, 0, 0, 0], dtype=np.int32)
        masks, predicted_scores, _ = predictor.predict(**kwargs)
        # El producto selecciona la multimáscara con respecto al área de la caja
        # ya ampliada. Mantener la misma base evita una divergencia silenciosa
        # entre el experimento y la aplicación.
        box_area = max(1, int((box[2] - box[0]) * (box[3] - box[1])))
        selected, score = SAM2Segmenter._select_mask(masks, predicted_scores, box_area)
        accumulated[selected.astype(bool)] = 255
        scores.append(score)
    return postprocess(accumulated), scores, time.perf_counter() - start


def evaluate_partition(predictor, rows, partition, strategies):
    selected_rows = [row for row in rows if row["domain"] == "synthetic" and row["split"] == partition]
    results = []
    for index, row in enumerate(selected_rows, start=1):
        image = cv2.imread(str(ROOT / row["image"]))
        reference = cv2.imread(str(ROOT / row["ground_truth"]), cv2.IMREAD_GRAYSCALE)
        prompt_path = PROMPTS / f"synthetic_{row['sample_id']}_prompts.json"
        prompt_data = json.loads(prompt_path.read_text(encoding="utf-8"))
        for strategy in strategies:
            prediction, scores, elapsed = predict_strategy(
                predictor, image, prompt_data["prompts"], strategy
            )
            dice, iou = dice_iou(reference, prediction)
            expected = component_count(reference)
            predicted = component_count(prediction)
            results.append(
                {
                    "partition": partition,
                    "sample_id": row["sample_id"],
                    "strategy": strategy,
                    "dice": dice,
                    "iou": iou,
                    "boundary_f1": boundary_f1(reference, prediction),
                    "expected_components": expected,
                    "predicted_components": predicted,
                    "component_error": abs(predicted - expected),
                    "mean_predicted_score": float(np.mean(scores)),
                    "inference_seconds": elapsed,
                }
            )
            if partition == "test":
                MASKS.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(MASKS / f"{row['sample_id']}_{strategy}.png"), prediction)
        print(f"SAM 2 {partition} [{index}/{len(selected_rows)}] {row['sample_id']}")
    return results


def summarize_validation(results):
    grouped = defaultdict(list)
    for row in results:
        grouped[row["strategy"]].append(row)
    summary = []
    for strategy, values in grouped.items():
        mean_iou = float(np.mean([row["iou"] for row in values]))
        component_error = float(np.mean([row["component_error"] for row in values]))
        summary.append(
            {
                "strategy": strategy,
                "mean_validation_iou": mean_iou,
                "std_validation_iou": float(np.std([row["iou"] for row in values], ddof=1)),
                "mean_validation_dice": float(np.mean([row["dice"] for row in values])),
                "mean_validation_boundary_f1": float(np.mean([row["boundary_f1"] for row in values])),
                "mean_component_error": component_error,
                "selection_score": mean_iou - 0.005 * component_error,
                "mean_inference_seconds": float(np.mean([row["inference_seconds"] for row in values])),
            }
        )
    return sorted(summary, key=lambda row: row["selection_score"], reverse=True)


def main() -> None:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2("configs/sam2/sam2_hiera_t.yaml", str(CHECKPOINT), device=device)
    predictor = SAM2ImagePredictor(model)
    rows = manifest_rows()
    validation = evaluate_partition(predictor, rows, "val", list(STRATEGIES))
    summary = summarize_validation(validation)
    write_csv(METRICS / "sam2_validation_prompt_metrics.csv", validation)
    write_csv(METRICS / "sam2_validation_prompt_summary.csv", summary)
    winner = summary[0]["strategy"]
    test = evaluate_partition(predictor, rows, "test", [winner])
    write_csv(METRICS / "sam2_test_metrics.csv", test)
    final = {
        "model_family": "SAM 2",
        "model_variant": "Hiera Tiny",
        "pretrained": True,
        "device": device,
        "prompt_source": "generator metadata",
        "selected_strategy": winner,
        "validation": summary[0],
        "test_mean_iou": float(np.mean([row["iou"] for row in test])),
        "test_std_iou": float(np.std([row["iou"] for row in test], ddof=1)),
        "test_mean_dice": float(np.mean([row["dice"] for row in test])),
        "test_mean_boundary_f1": float(np.mean([row["boundary_f1"] for row in test])),
        "test_mean_component_error": float(np.mean([row["component_error"] for row in test])),
        "checkpoint_sha256": "65B50056E05BCB13694174F51BB6DA89C894B57B75CCDF0BA6352C597C5D1125",
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    (METRICS / "sam2_final_selection.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
