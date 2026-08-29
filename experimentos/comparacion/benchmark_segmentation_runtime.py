"""Mide tiempos comparables de las tres líneas sobre imágenes rectificadas.

El modelo y los archivos se cargan antes de cronometrar. Cada método recibe la
misma imagen rectificada, en el mismo proceso y hardware. El protocolo hace un
calentamiento por método y cinco repeticiones medidas por captura. Para SAM 2,
el total operativo incluye localización clásica, ``set_image``/codificador,
decodificación de las cajas y postproceso.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SAM_DIR = ROOT / "experimentos" / "sam2"
RF_DIR = ROOT / "experimentos" / "random_forest"
METRICS = ROOT / "resultados" / "metricas"

sys.path.insert(0, str(SAM_DIR))
sys.path.insert(0, str(RF_DIR))
sys.path.insert(0, str(ROOT / "software" / "src"))

from feature_extraction import BASE_FEATURE_NAMES, extract_pixel_features
from postprocess_masks import postprocess_rf_mask
from protocol_common import (
    ProductAlignedSAM2,
    build_predictor,
    detect_wcs_for_rectified,
    load_manifest,
    operational_boxes,
    read_required_image,
    resolve_device,
    write_csv,
)
from segment_toppers import segment_toppers


RF_MODEL = ROOT / "resultados" / "modelos" / "random_forest_identity_v2_selected.joblib"
RF_SELECTION = ROOT / "resultados" / "metricas" / "rf_idv2_selection_locked.json"
SAM_SELECTION = ROOT / "resultados" / "metricas" / "sam2_idv2_prompt_selection.json"


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    summaries: list[dict] = []
    for method, values in grouped.items():
        totals = np.asarray([float(row["total_seconds"]) for row in values])
        summaries.append(
            {
                "method": method,
                "n_images": len({row["sample_id"] for row in values}),
                "n_measured_runs": len(values),
                "mean_seconds": float(totals.mean()),
                "std_seconds": float(totals.std(ddof=1)),
                "median_seconds": float(np.median(totals)),
                "p25_seconds": float(np.quantile(totals, 0.25)),
                "p75_seconds": float(np.quantile(totals, 0.75)),
                "min_seconds": float(totals.min()),
                "max_seconds": float(totals.max()),
            }
        )
    return sorted(summaries, key=lambda row: row["mean_seconds"])


def time_classical(image: np.ndarray, wcs_info: dict) -> tuple[np.ndarray, dict[str, float]]:
    start = time.perf_counter()
    mask = segment_toppers(image, scale=10.0, wcs_info=wcs_info)
    elapsed = time.perf_counter() - start
    return mask, {"total_seconds": elapsed}


def time_rf(image: np.ndarray, model, threshold: float) -> tuple[np.ndarray, dict[str, float]]:
    start = time.perf_counter()
    features, names = extract_pixel_features(image, include_coords=False)
    if tuple(names) != BASE_FEATURE_NAMES:
        raise RuntimeError("El extractor no coincide con el contrato de 19 características.")
    probability = model.predict_proba(features)[:, 1].reshape(image.shape[:2])
    mask, _ = postprocess_rf_mask((probability >= threshold).astype(np.uint8) * 255)
    elapsed = time.perf_counter() - start
    del features, probability
    return mask, {"total_seconds": elapsed}


def time_sam(
    image: np.ndarray,
    wcs_info: dict,
    runner: ProductAlignedSAM2,
) -> tuple[np.ndarray, dict[str, float]]:
    boxes, _, prompt_seconds = operational_boxes(image, wcs_info=wcs_info)
    prediction, _, timing, _ = runner.predict(image, boxes)
    timing = dict(timing)
    timing["prompt_localization_seconds"] = prompt_seconds
    timing["total_seconds"] = prompt_seconds + timing["sam_total_seconds"]
    return prediction, timing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=["real_13", "real_20", "real_28", "real_38"],
        help="Capturas representativas usadas en el banco temporal.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.warmup < 1 or args.repetitions < 2:
        raise ValueError("Se requiere al menos un calentamiento y dos repeticiones.")

    rows_by_id = {
        row["sample_id"]: row
        for row in load_manifest()
        if row["domain"] == "real" and row["sample_id"] in args.sample_ids
    }
    missing = sorted(set(args.sample_ids) - set(rows_by_id))
    if missing:
        raise KeyError(f"Muestras ausentes del manifiesto: {missing}")

    device = resolve_device(args.device)
    sam_selection = json.loads(SAM_SELECTION.read_text(encoding="utf-8"))
    sam_margin = float(sam_selection["selected_box_margin_fraction"])
    sam_runner = ProductAlignedSAM2(
        build_predictor(device),
        device=device,
        box_margin_fraction=sam_margin,
    )
    rf_bundle = joblib.load(RF_MODEL)
    rf_model = rf_bundle["model"]
    rf_selection = json.loads(RF_SELECTION.read_text(encoding="utf-8"))
    rf_threshold = float(rf_bundle["threshold"])
    if tuple(rf_bundle["feature_names"]) != BASE_FEATURE_NAMES:
        raise RuntimeError("El modelo RF seleccionado no cumple el contrato de 19 características.")
    if int(rf_bundle["feature_count"]) != 19:
        raise RuntimeError("El modelo RF seleccionado no declara 19 características.")
    if not np.isclose(rf_threshold, float(rf_selection["selected_threshold"])):
        raise RuntimeError("El umbral del modelo RF y el bloqueo de validación no coinciden.")

    inputs: list[tuple[str, np.ndarray, dict]] = []
    for sample_id in args.sample_ids:
        row = rows_by_id[sample_id]
        image = read_required_image(ROOT / row["image"])
        wcs = detect_wcs_for_rectified(image, sample_id)
        inputs.append((sample_id, image, wcs))

    methods = {
        "vision_clasica": lambda image, wcs: time_classical(image, wcs),
        "random_forest_identity_v2": lambda image, wcs: time_rf(
            image, rf_model, rf_threshold
        ),
        "sam2_operativo_hibrido": lambda image, wcs: time_sam(image, wcs, sam_runner),
    }

    # Calentamiento explícito por método con la primera captura; no se registra.
    first_id, first_image, first_wcs = inputs[0]
    for method, function in methods.items():
        for _ in range(args.warmup):
            function(first_image, first_wcs)
        gc.collect()
        print(f"Calentamiento completado: {method} ({first_id})")

    measured: list[dict] = []
    for sample_id, image, wcs in inputs:
        for method, function in methods.items():
            for repetition in range(1, args.repetitions + 1):
                _, timing = function(image, wcs)
                measured.append(
                    {
                        "sample_id": sample_id,
                        "method": method,
                        "repetition": repetition,
                        "total_seconds": timing["total_seconds"],
                        "prompt_localization_seconds": timing.get(
                            "prompt_localization_seconds", ""
                        ),
                        "encoder_seconds": timing.get("encoder_seconds", ""),
                        "decoder_seconds": timing.get("decoder_seconds", ""),
                        "postprocess_seconds": timing.get("postprocess_seconds", ""),
                        "sam_total_seconds": timing.get("sam_total_seconds", ""),
                        "input_width_px": image.shape[1],
                        "input_height_px": image.shape[0],
                        "device": device if method == "sam2_operativo_hibrido" else "cpu",
                    }
                )
                print(f"Tiempo {sample_id} {method} [{repetition}/{args.repetitions}]")
            gc.collect()

    summary = summarize(measured)
    METRICS.mkdir(parents=True, exist_ok=True)
    write_csv(METRICS / "segmentation_runtime_runs_v8.csv", measured)
    write_csv(METRICS / "segmentation_runtime_summary_v8.csv", summary)
    payload = {
        "protocol_version": "segmentation_runtime_v8",
        "scope": "segmentation on preloaded 2100x2970 rectified images",
        "sample_ids": args.sample_ids,
        "warmup_runs_per_method": args.warmup,
        "measured_repetitions_per_image": args.repetitions,
        "same_process_and_hardware": True,
        "model_loading_excluded": True,
        "disk_image_loading_excluded": True,
        "rectification_and_wcs_detection_excluded": True,
        "sam2_total_definition": (
            "classical prompt localization + set_image/encoder + box decoder + postprocess"
        ),
        "sam2_box_margin_fraction": sam_margin,
        "sam2_margin_selection": str(SAM_SELECTION.relative_to(ROOT)).replace("\\", "/"),
        "sam2_margin_selection_partition": "asset_identity_v2 validation only",
        "classical_total_definition": "complete segment_toppers call",
        "rf_total_definition": "19-feature extraction + predict_proba + RF postprocess",
        "rf_model": str(RF_MODEL.relative_to(ROOT)).replace("\\", "/"),
        "rf_model_sha256": hashlib.sha256(RF_MODEL.read_bytes()).hexdigest(),
        "rf_selection": str(RF_SELECTION.relative_to(ROOT)).replace("\\", "/"),
        "rf_selection_sha256": hashlib.sha256(RF_SELECTION.read_bytes()).hexdigest(),
        "rf_dataset_partition": "asset_identity_v2 with identity-disjoint train/validation/test",
        "rf_threshold": rf_threshold,
        "rf_feature_count": int(rf_bundle["feature_count"]),
        "rf_candidate_id": rf_bundle["candidate_id"],
        "device_sam2": device,
        "device_other_methods": "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "processor": platform.processor(),
        "summary": summary,
    }
    (METRICS / "segmentation_runtime_summary_v8.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
