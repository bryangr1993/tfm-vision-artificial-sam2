"""Banco temporal coherente de las tres lineas finales sobre IDV2 TEST.

Los tres metodos se ejecutan en el mismo proceso, hardware y entorno de
Python. Las doce imagenes de TEST se cargan antes de medir. Tambien se cargan
previamente los modelos de RF y SAM 2. El alcance comienza con la segmentacion
de una imagen rectificada de 2100 x 2970 px y termina con la mascara binaria.

* Clasica: llamada completa a ``segment_toppers``.
* RF seleccionado: 19 caracteristicas, ``predict_proba`` y postproceso.
* SAM 2 operativo: localizador clasico de cajas, codificador, decodificador de
  cajas y postproceso.

La lectura de disco, la rectificacion y la deteccion del WCS quedan fuera.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
import sklearn
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RF_DIR = ROOT / "experimentos" / "random_forest"
SAM_DIR = ROOT / "experimentos" / "sam2"
METRICS = ROOT / "resultados" / "metricas"
MANIFEST = ROOT / "datos" / "manifiesto" / "datasets_asset_identity_v2.csv"
QUALITY_REPORT = METRICS / "dataset_identity_v2_quality.json"
RF_MODEL = ROOT / "resultados" / "modelos" / "random_forest_identity_v2_selected.joblib"
RF_LOCK = METRICS / "rf_idv2_selection_locked.json"
SAM_LOCK = METRICS / "sam2_idv2_prompt_selection.json"
SAM_CHECKPOINT = ROOT / "resultados" / "modelos" / "sam2_hiera_tiny.pt"
RUNS_CSV = METRICS / "segmentation_runtime_idv2_final_runs.csv"
SUMMARY_CSV = METRICS / "segmentation_runtime_idv2_final_summary.csv"
SUMMARY_JSON = METRICS / "segmentation_runtime_idv2_final_summary.json"

WARMUP_RUNS_PER_METHOD = 1
MEASURED_REPETITIONS_PER_SHEET = 3
EXPECTED_FEATURE_COUNT = 19
EXPECTED_TEST_SHEETS = 12

sys.path.insert(0, str(ROOT / "software" / "src"))
sys.path.insert(0, str(RF_DIR))
sys.path.insert(0, str(SAM_DIR))

from feature_extraction import BASE_FEATURE_NAMES, extract_pixel_features
from postprocess_masks import postprocess_rf_mask
from protocol_common import ProductAlignedSAM2, build_predictor, operational_boxes
from segment_toppers import segment_toppers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No hay filas para {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validated_test_images() -> list[tuple[str, np.ndarray]]:
    quality = json.loads(QUALITY_REPORT.read_text(encoding="utf-8"))
    if not quality.get("passed") or int(quality.get("critical_failure_count", 1)) != 0:
        raise RuntimeError("La puerta de calidad de IDV2 no esta aprobada.")
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    if Counter(row["split"] for row in all_rows) != Counter({"train": 24, "val": 12, "test": 12}):
        raise RuntimeError("Conteos de particion inesperados.")
    rows = sorted(
        (row for row in all_rows if row["domain"] == "synthetic" and row["split"] == "test"),
        key=lambda row: row["sample_id"],
    )
    if len(rows) != EXPECTED_TEST_SHEETS:
        raise RuntimeError("TEST debe contener doce laminas.")
    loaded: list[tuple[str, np.ndarray]] = []
    for row in rows:
        path = ROOT / row["image"]
        if sha256_file(path) != row["image_sha256"]:
            raise RuntimeError(f"Hash de imagen inesperado: {row['sample_id']}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (2970, 2100):
            raise RuntimeError(f"Imagen invalida: {row['sample_id']}")
        loaded.append((row["sample_id"], image))
    return loaded


def predict_rf(image: np.ndarray, model, threshold: float) -> np.ndarray:
    features, names = extract_pixel_features(image, include_coords=False)
    if tuple(names) != tuple(BASE_FEATURE_NAMES) or len(names) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError("El contrato de caracteristicas RF no contiene 19 variables.")
    probabilities = np.empty(len(features), dtype=np.float32)
    chunk = 250_000
    for start in range(0, len(features), chunk):
        stop = min(start + chunk, len(features))
        probabilities[start:stop] = model.predict_proba(features[start:stop])[:, 1]
    raw = (probabilities.reshape(image.shape[:2]) >= threshold).astype(np.uint8) * 255
    prediction, _ = postprocess_rf_mask(raw)
    del features, probabilities, raw
    return prediction


def predict_sam(image: np.ndarray, runner: ProductAlignedSAM2) -> np.ndarray:
    boxes, _, _ = operational_boxes(image, wcs_info=None, scale=10.0)
    prediction, _, _, _ = runner.predict(image, boxes)
    return prediction


def validate_prediction(method: str, sample_id: str, image: np.ndarray, mask: np.ndarray) -> None:
    if mask.shape != image.shape[:2] or mask.dtype != np.uint8:
        raise RuntimeError(f"Salida invalida de {method} en {sample_id}")
    if set(np.unique(mask).tolist()) - {0, 255}:
        raise RuntimeError(f"Salida no binaria de {method} en {sample_id}")


def summarize(rows: list[dict]) -> list[dict]:
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    result: list[dict] = []
    for method, values in grouped.items():
        times = np.asarray([float(row["segmentation_seconds"]) for row in values])
        result.append(
            {
                "method": method,
                "n_images": len({row["sample_id"] for row in values}),
                "n_measured_runs": len(values),
                "mean_seconds": float(times.mean()),
                "std_seconds": float(times.std(ddof=1)),
                "median_seconds": float(np.median(times)),
                "p25_seconds": float(np.quantile(times, 0.25)),
                "p75_seconds": float(np.quantile(times, 0.75)),
                "min_seconds": float(times.min()),
                "max_seconds": float(times.max()),
            }
        )
    return sorted(result, key=lambda row: float(row["mean_seconds"]))


def main() -> None:
    images = validated_test_images()
    rf_lock = json.loads(RF_LOCK.read_text(encoding="utf-8"))
    sam_lock = json.loads(SAM_LOCK.read_text(encoding="utf-8"))
    if rf_lock.get("test_set_used_for_selection") is not False:
        raise RuntimeError("El RF no tiene un bloqueo de seleccion valido.")
    if sam_lock.get("selection_partition") != "val only":
        raise RuntimeError("El margen de SAM 2 no fue seleccionado solo en validacion.")
    if sha256_file(RF_MODEL) != rf_lock["selected_model_sha256"]:
        raise RuntimeError("El hash del RF seleccionado no coincide con su bloqueo.")
    if sha256_file(SAM_CHECKPOINT) != str(sam_lock["checkpoint_sha256"]).lower():
        raise RuntimeError("El hash del checkpoint SAM 2 no coincide con su bloqueo.")
    if sha256_file(MANIFEST) != rf_lock["dataset_manifest_sha256"]:
        raise RuntimeError("El manifiesto no coincide con el bloqueo RF.")
    if sha256_file(MANIFEST) != sam_lock["manifest_sha256"]:
        raise RuntimeError("El manifiesto no coincide con el bloqueo SAM 2.")

    rf_package = joblib.load(RF_MODEL)
    rf_model = rf_package["model"]
    rf_threshold = float(rf_lock["selected_threshold"])
    sam_margin = float(sam_lock["selected_box_margin_fraction"])
    sam_runner = ProductAlignedSAM2(
        build_predictor("cpu"),
        device="cpu",
        box_margin_fraction=sam_margin,
    )

    methods = {
        "vision_clasica_fija": lambda image: segment_toppers(image, scale=10.0, wcs_info=None),
        "random_forest_seleccionado": lambda image: predict_rf(image, rf_model, rf_threshold),
        "sam2_operativo_hibrido": lambda image: predict_sam(image, sam_runner),
    }
    method_names = list(methods)

    # Un calentamiento por metodo sobre la misma primera lamina. No se registra.
    first_id, first_image = images[0]
    for method, function in methods.items():
        prediction = function(first_image)
        validate_prediction(method, first_id, first_image, prediction)
        del prediction
        gc.collect()
        print(f"Calentamiento completado: {method}")

    measured: list[dict] = []
    for sample_index, (sample_id, image) in enumerate(images):
        for repetition in range(1, MEASURED_REPETITIONS_PER_SHEET + 1):
            # Orden rotatorio para no favorecer sistematicamente al primer metodo.
            rotation = (sample_index + repetition - 1) % len(method_names)
            order = method_names[rotation:] + method_names[:rotation]
            for order_position, method in enumerate(order, start=1):
                start = time.perf_counter()
                prediction = methods[method](image)
                elapsed = time.perf_counter() - start
                validate_prediction(method, sample_id, image, prediction)
                measured.append(
                    {
                        "dataset_version": "asset_identity_v2",
                        "split": "test",
                        "sample_id": sample_id,
                        "method": method,
                        "repetition": repetition,
                        "rotating_order_position": order_position,
                        "segmentation_seconds": elapsed,
                        "input_width_px": image.shape[1],
                        "input_height_px": image.shape[0],
                        "device": "cpu",
                    }
                )
                del prediction
                gc.collect()
                print(
                    f"Tiempo [{sample_index + 1:02d}/{len(images)}] {sample_id} "
                    f"rep={repetition} {method}: {elapsed:.4f} s"
                )
        write_csv(RUNS_CSV, measured)

    summary_rows = summarize(measured)
    write_csv(RUNS_CSV, measured)
    write_csv(SUMMARY_CSV, summary_rows)
    payload = {
        "protocol_version": "segmentation_runtime_asset_identity_v2_final_v1",
        "status": "comparable_final_methods_same_process_hardware_and_inputs",
        "dataset_version": "asset_identity_v2",
        "split": "test",
        "sample_ids": [sample_id for sample_id, _ in images],
        "n_images": len(images),
        "scope": "segmentation from preloaded 2100x2970 rectified image to binary mask",
        "same_process_and_hardware": True,
        "rotating_method_order": True,
        "warmup_runs_per_method": WARMUP_RUNS_PER_METHOD,
        "measured_repetitions_per_image": MEASURED_REPETITIONS_PER_SHEET,
        "image_loading_excluded": True,
        "model_loading_excluded": True,
        "rectification_excluded": True,
        "wcs_detection_excluded": True,
        "classical_total_definition": "complete segment_toppers call",
        "rf_total_definition": "19-feature extraction + predict_proba + selected-threshold postprocess",
        "sam2_total_definition": (
            "classical prompt localization + set_image/encoder + box decoder + postprocess"
        ),
        "rf_threshold": rf_threshold,
        "rf_model": RF_MODEL.relative_to(ROOT).as_posix(),
        "rf_model_sha256": sha256_file(RF_MODEL),
        "rf_selection_lock": RF_LOCK.relative_to(ROOT).as_posix(),
        "sam2_margin_fraction": sam_margin,
        "sam2_checkpoint": SAM_CHECKPOINT.relative_to(ROOT).as_posix(),
        "sam2_checkpoint_sha256": sha256_file(SAM_CHECKPOINT),
        "sam2_selection_lock": SAM_LOCK.relative_to(ROOT).as_posix(),
        "device_all_methods": "cpu",
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "processor": platform.processor(),
        "summary": summary_rows,
        "runs_csv": RUNS_CSV.relative_to(ROOT).as_posix(),
        "summary_csv": SUMMARY_CSV.relative_to(ROOT).as_posix(),
    }
    SUMMARY_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
