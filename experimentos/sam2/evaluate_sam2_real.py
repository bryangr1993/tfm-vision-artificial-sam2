"""Evalúa el SAM 2 seleccionado sobre las trece capturas reales.

La estrategia de prompts ya está bloqueada con el conjunto sintético de
validación. Las cajas reales proceden del localizador clásico aproximado, pero
las fronteras de sus máscaras no intervienen en el cálculo de la salida de SAM 2.
La referencia se define en el plano canónico común de la lámina rectificada.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = ROOT / "datos" / "manifiesto" / "datasets.csv"
CHECKPOINT = ROOT / "resultados" / "modelos" / "sam2_hiera_tiny.pt"
SELECTION = ROOT / "resultados" / "metricas" / "sam2_final_selection.json"
METRICS = ROOT / "resultados" / "metricas"
MASKS = ROOT / "resultados" / "real_predictions" / "sam2"

sys.path.insert(0, str(HERE))
from evaluate_sam2 import (
    boundary_f1,
    component_count,
    dice_iou,
    predict_strategy,
    write_csv,
)


def real_rows() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["domain"] == "real"]


def bootstrap_mean_ci(values: list[float], seed: int = 42) -> list[float]:
    """Intervalo percentil descriptivo por captura, con semilla fija."""

    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(10_000, len(data)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> None:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    strategy = selection["selected_strategy"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2("configs/sam2/sam2_hiera_t.yaml", str(CHECKPOINT), device=device)
    predictor = SAM2ImagePredictor(model)
    MASKS.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    rows = real_rows()
    for index, row in enumerate(rows, start=1):
        image = cv2.imread(str(ROOT / row["image"]))
        reference = cv2.imread(str(ROOT / row["ground_truth"]), cv2.IMREAD_GRAYSCALE)
        prompt_data = json.loads((ROOT / row["prompt"]).read_text(encoding="utf-8"))
        prediction, scores, elapsed = predict_strategy(
            predictor, image, prompt_data["prompts"], strategy
        )
        dice, iou = dice_iou(reference, prediction)
        expected = component_count(reference)
        predicted = component_count(prediction)
        results.append(
            {
                "domain": "real",
                "sample_id": row["sample_id"],
                "method": "SAM 2 Hiera Tiny",
                "strategy": strategy,
                "prompt_source": "localizador_clasico_solo_cajas",
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
        cv2.imwrite(str(MASKS / f"{row['sample_id']}_sam2.png"), prediction)
        print(
            f"SAM 2 real [{index}/{len(rows)}] {row['sample_id']}: "
            f"IoU={iou:.5f}, componentes={predicted}, tiempo={elapsed:.3f} s"
        )

    write_csv(METRICS / "sam2_real_metrics.csv", results)
    iou_values = [float(row["iou"]) for row in results]
    dice_values = [float(row["dice"]) for row in results]
    boundary_values = [float(row["boundary_f1"]) for row in results]
    summary = {
        "model_family": "SAM 2",
        "model_variant": "Hiera Tiny",
        "pretrained": True,
        "selected_strategy": strategy,
        "selection_partition": "synthetic_validation",
        "evaluation_domain": "13 rectified captures of one physical sheet",
        "prompt_source": "classical coarse localizer; boxes only",
        "reference": "assisted canonical reference independent of compared masks",
        "mean_iou": float(np.mean(iou_values)),
        "std_iou": float(np.std(iou_values, ddof=1)),
        "iou_capture_bootstrap_95_ci": bootstrap_mean_ci(iou_values, seed=42),
        "mean_dice": float(np.mean(dice_values)),
        "std_dice": float(np.std(dice_values, ddof=1)),
        "dice_capture_bootstrap_95_ci": bootstrap_mean_ci(dice_values, seed=43),
        "mean_boundary_f1": float(np.mean(boundary_values)),
        "mean_component_error": float(np.mean([row["component_error"] for row in results])),
        "mean_inference_seconds": float(np.mean([row["inference_seconds"] for row in results])),
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "interpretation_limit": (
            "The interval quantifies variation between captures of the same sheet and must not "
            "be interpreted as generalization to new sheet designs."
        ),
    }
    (METRICS / "sam2_real_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
