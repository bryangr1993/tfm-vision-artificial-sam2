"""Evalúa el RF IDV2 bloqueado sobre las 13 capturas reales.

La referencia real es canónica y asistida. En consecuencia, Dice, IoU y F1 de
contorno se interpretan como concordancia con esa referencia, no como exactitud
frente a una verdad terreno independiente. La evaluación ocurre después del
bloqueo de modelo y umbral y no modifica la selección.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REAL_MANIFEST = ROOT / "datos" / "manifiesto" / "datasets.csv"
IDV2_MANIFEST = ROOT / "datos" / "manifiesto" / "datasets_asset_identity_v2.csv"
LOCK = ROOT / "resultados" / "metricas" / "rf_idv2_selection_locked.json"
MODEL_PATH = ROOT / "resultados" / "modelos" / "random_forest_identity_v2_selected.joblib"
METRICS_CSV = ROOT / "resultados" / "metricas" / "rf_idv2_real_agreement_metrics.csv"
SUMMARY_JSON = ROOT / "resultados" / "metricas" / "rf_idv2_real_agreement_summary.json"
MASK_DIR = ROOT / "resultados" / "rf_idv2_real_agreement_masks"
FIGURE = ROOT / "resultados" / "figuras" / "rf_idv2_real_agreement_examples.png"

sys.path.insert(0, str(HERE))
from feature_extraction import BASE_FEATURE_NAMES, extract_pixel_features
from postprocess_masks import postprocess_rf_mask
from train_rf_identity_v2 import (
    boundary_f1,
    component_count,
    dice_iou,
    file_sha256,
    predict_probability,
)


def read_real_rows() -> list[dict[str, str]]:
    with REAL_MANIFEST.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["domain"] == "real"]
    if len(rows) != 13:
        raise RuntimeError(f"Se esperaban 13 capturas reales; se encontraron {len(rows)}.")
    return rows


def evaluate() -> tuple[list[dict], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    if METRICS_CSV.exists() or SUMMARY_JSON.exists():
        raise RuntimeError("La evaluación real post-lock ya existe; no se sobrescribe automáticamente.")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "locked_before_test" or lock["test_set_used_for_selection"] is not False:
        raise RuntimeError("El modelo no dispone de un bloqueo válido anterior a prueba.")
    if file_sha256(IDV2_MANIFEST) != lock["dataset_manifest_sha256"]:
        raise RuntimeError("Cambió el manifiesto de selección IDV2.")
    if file_sha256(MODEL_PATH) != lock["selected_model_sha256"]:
        raise RuntimeError("Cambió el modelo seleccionado después del bloqueo.")
    package = joblib.load(MODEL_PATH)
    if tuple(package["feature_names"]) != BASE_FEATURE_NAMES:
        raise RuntimeError("Contrato de características incompatible.")
    model = package["model"]
    threshold = float(package["threshold"])
    rows = read_real_rows()
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    output: list[dict] = []
    images: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for index, row in enumerate(rows, start=1):
        image = cv2.imread(str(ROOT / row["image"]))
        reference = cv2.imread(str(ROOT / row["ground_truth"]), cv2.IMREAD_GRAYSCALE)
        if image is None or reference is None:
            raise FileNotFoundError(row["sample_id"])
        started = time.perf_counter()
        features, names = extract_pixel_features(image, include_coords=False)
        if tuple(names) != BASE_FEATURE_NAMES:
            raise RuntimeError("Extracción incompatible con el modelo bloqueado.")
        probability = predict_probability(model, features, reference.shape)
        prediction, _ = postprocess_rf_mask((probability >= threshold).astype(np.uint8) * 255)
        elapsed = time.perf_counter() - started
        dice, iou = dice_iou(reference, prediction)
        expected = component_count(reference)
        predicted = component_count(prediction)
        mask_path = MASK_DIR / f"{row['sample_id']}_rf_idv2.png"
        cv2.imwrite(str(mask_path), prediction)
        output.append(
            {
                "model": "RF_selected_IDV2_locked",
                "sample_id": row["sample_id"],
                "evaluation_domain": "real_external_post_lock",
                "threshold_locked_on_synthetic_validation": threshold,
                "agreement_dice": dice,
                "agreement_iou": iou,
                "agreement_boundary_f1": boundary_f1(reference, prediction),
                "reference_components": expected,
                "predicted_components": predicted,
                "component_count_difference": predicted - expected,
                "absolute_component_error": abs(predicted - expected),
                "inference_seconds_including_feature_extraction": elapsed,
                "reference_status": "assisted_canonical_not_independent_ground_truth",
                "metric_interpretation": "agreement_not_accuracy",
                "prediction_mask": mask_path.relative_to(ROOT).as_posix(),
            }
        )
        images[row["sample_id"]] = (image, reference, prediction)
        print(f"Real [{index:02d}/{len(rows)}] {row['sample_id']}: concordancia IoU={iou:.5f}")
    return output, images


def write_csv(rows: list[dict]) -> None:
    with METRICS_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict], images: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    ordered = sorted(rows, key=lambda row: float(row["agreement_iou"]))
    representatives = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    fig, axes = plt.subplots(4, 3, figsize=(5.55, 6.85))
    row_labels = ("Captura rectificada", "Referencia asistida", "Predicción RF", "Acuerdo y desacuerdo")
    for column, metric_row in enumerate(representatives):
        sample_id = str(metric_row["sample_id"])
        image, reference, prediction = images[sample_id]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ref = reference > 0
        pred = prediction > 0
        overlay = np.full((*reference.shape, 3), 245, dtype=np.uint8)
        overlay[ref & pred] = (62, 132, 84)      # acuerdo positivo
        overlay[~ref & pred] = (180, 55, 145)    # falso positivo respecto a referencia
        overlay[ref & ~pred] = (230, 135, 45)    # falso negativo respecto a referencia
        panels = (rgb, reference, prediction, overlay)
        for row_index, panel in enumerate(panels):
            axes[row_index, column].imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            axes[row_index, column].axis("off")
            if column == 0:
                axes[row_index, column].set_ylabel(row_labels[row_index], fontsize=8.5)
        axes[0, column].set_title(
            f"{sample_id}\nIoU de concordancia = {float(metric_row['agreement_iou']):.4f}",
            fontsize=8.5,
        )
    fig.suptitle("Transferencia del RF IDV2 a capturas reales (mínima, mediana y máxima)", fontsize=10.5)
    fig.text(
        0.5,
        0.015,
        "Verde: acuerdo positivo · Magenta: predicción adicional · Naranja: región omitida. "
        "La referencia es canónica y asistida.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=300)
    fig.savefig(FIGURE.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    rows, images = evaluate()
    write_csv(rows)
    make_figure(rows, images)
    summary = {
        "status": "real_external_evaluation_after_synthetic_selection_lock",
        "model": "RF_selected_IDV2_locked",
        "model_sha256": file_sha256(MODEL_PATH),
        "selection_lock_sha256": file_sha256(LOCK),
        "synthetic_test_used_for_selection": False,
        "real_data_used_for_selection_or_tuning": False,
        "capture_count": len(rows),
        "physical_sheet_count": 1,
        "reference_status": "assisted_canonical_not_independent_ground_truth",
        "metric_interpretation": "agreement_with_assisted_reference_not_accuracy",
        "mean_agreement_iou": float(np.mean([row["agreement_iou"] for row in rows])),
        "std_agreement_iou_across_captures": float(np.std([row["agreement_iou"] for row in rows], ddof=1)),
        "mean_agreement_dice": float(np.mean([row["agreement_dice"] for row in rows])),
        "mean_agreement_boundary_f1": float(np.mean([row["agreement_boundary_f1"] for row in rows])),
        "mean_absolute_component_error": float(np.mean([row["absolute_component_error"] for row in rows])),
        "captures_with_eight_components": sum(int(row["predicted_components"]) == 8 for row in rows),
        "mean_inference_seconds_including_feature_extraction": float(
            np.mean([row["inference_seconds_including_feature_extraction"] for row in rows])
        ),
        "timing_context": (
            "Observed wall-clock time on a shared workstation with concurrent project processes; "
            "descriptive only, not the controlled final runtime benchmark."
        ),
        "scope": (
            "Trece adquisiciones de una sola lámina física; describe estabilidad de transferencia "
            "en esas capturas y no generalización poblacional a nuevos diseños físicos."
        ),
        "metrics_csv": METRICS_CSV.relative_to(ROOT).as_posix(),
        "figure": FIGURE.relative_to(ROOT).as_posix(),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
