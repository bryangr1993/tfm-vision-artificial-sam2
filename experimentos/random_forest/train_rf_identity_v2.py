"""Entrenamiento y evaluación reproducibles de RF sobre ``asset_identity_v2``.

El protocolo se separa en comandos para impedir que la prueba participe en la
selección:

1. ``select``: búsqueda agrupada en píxeles de entrenamiento y selección final
   sobre las doce láminas completas de validación.
2. ``ablate``: diagnóstico de familias de características, siempre después de
   bloquear el modelo y sin consultar prueba.
3. ``test``: una sola evaluación del modelo bloqueado y del control previamente
   registrado sobre las doce láminas de prueba.

La configuración de control reconstruye el RF histórico (200 árboles,
profundidad 25, hoja mínima 2) mediante código, en lugar de cargar un artefacto
de origen no reproducible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import joblib
import matplotlib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import jaccard_score
from sklearn.model_selection import GroupKFold, ParameterSampler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = ROOT / "datos" / "manifiesto" / "datasets_asset_identity_v2.csv"
QUALITY_REPORT = ROOT / "resultados" / "metricas" / "dataset_identity_v2_quality.json"
METRICS = ROOT / "resultados" / "metricas"
MODELS = ROOT / "resultados" / "modelos"
FIGURES = ROOT / "resultados" / "figuras"
MASKS = ROOT / "resultados" / "rf_idv2_test_masks"

SEARCH_FOLDS = METRICS / "rf_idv2_search_folds.csv"
SEARCH_SUMMARY = METRICS / "rf_idv2_search_summary.csv"
VALIDATION_DETAILS = METRICS / "rf_idv2_validation_per_sheet.csv"
VALIDATION_SUMMARY = METRICS / "rf_idv2_validation_summary.csv"
SELECTION_LOCK = METRICS / "rf_idv2_selection_locked.json"
TEST_DETAILS = METRICS / "rf_idv2_test_metrics.csv"
TEST_SUMMARY = METRICS / "rf_idv2_test_summary.json"
ABLATION_CSV = METRICS / "rf_idv2_feature_ablation.csv"
IMPORTANCE_CSV = METRICS / "rf_idv2_feature_importance.csv"
RUN_CONFIG = METRICS / "rf_idv2_run_configuration.json"
SELECTED_MODEL = MODELS / "random_forest_identity_v2_selected.joblib"
CONTROL_MODEL = MODELS / "random_forest_identity_v2_control.joblib"

SEED = 20260827
SAMPLES_PER_CLASS_PER_SHEET = 1_500
VALIDATION_ABLATION_PER_CLASS = 1_000
SEARCH_CANDIDATE_COUNT = 10
STAGE2_TOP_COUNT = 2
THRESHOLDS = tuple(float(value) for value in np.round(np.arange(0.35, 0.66, 0.05), 2))
COMPONENT_AREA = 20_000
SELECTION_COMPONENT_PENALTY = 0.005
PREDICTION_CHUNK = 250_000

sys.path.insert(0, str(HERE))
from feature_extraction import BASE_FEATURE_NAMES, FEATURE_GROUPS, extract_pixel_features
from postprocess_masks import postprocess_rf_mask


CONTROL_PARAMS: dict[str, object] = {
    "n_estimators": 200,
    "max_depth": 25,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "bootstrap": True,
}

SEARCH_SPACE: dict[str, list[object]] = {
    "n_estimators": [100, 160, 240],
    "max_depth": [12, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", 0.5, 0.8],
    "class_weight": ["balanced", "balanced_subsample"],
    "bootstrap": [True],
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No hay filas para {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_manifest() -> list[dict[str, str]]:
    quality = json.loads(QUALITY_REPORT.read_text(encoding="utf-8"))
    if not quality.get("passed"):
        raise RuntimeError("La puerta de calidad del dataset no está aprobada.")
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if Counter(row["split"] for row in rows) != Counter({"train": 24, "val": 12, "test": 12}):
        raise RuntimeError("Conteos de split inesperados en asset_identity_v2.")
    return rows


def load_image_mask(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(ROOT / row["image"]))
    mask = cv2.imread(str(ROOT / row["ground_truth"]), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise FileNotFoundError(row["sample_id"])
    return image, mask


def balanced_coordinates(mask: np.ndarray, per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    foreground = np.argwhere(mask == 255)
    dilated = cv2.dilate(mask, np.ones((35, 35), np.uint8), iterations=1)
    near_background = np.argwhere((dilated == 255) & (mask == 0))
    far_background = np.argwhere(dilated == 0)
    rng = np.random.default_rng(seed)

    def choose(values: np.ndarray, count: int) -> np.ndarray:
        count = min(len(values), count)
        if not count:
            return np.empty((0, 2), dtype=np.int64)
        return values[rng.choice(len(values), count, replace=False)]

    fg = choose(foreground, per_class)
    near = choose(near_background, per_class // 2)
    far = choose(far_background, per_class - len(near))
    coordinates = np.vstack((fg, near, far))
    labels = np.concatenate(
        (
            np.ones(len(fg), dtype=np.uint8),
            np.zeros(len(near) + len(far), dtype=np.uint8),
        )
    )
    stats = {"foreground": len(fg), "near_background": len(near), "far_background": len(far)}
    return coordinates, labels, stats


def sample_rows(
    rows: list[dict[str, str]],
    per_class: int,
    seed_offset: int,
    include_coords: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    sampling_rows: list[dict] = []
    for index, row in enumerate(rows, start=1):
        image, mask = load_image_mask(row)
        coords, target, stats = balanced_coordinates(mask, per_class, SEED + seed_offset + index)
        features, names = extract_pixel_features(image, include_coords=include_coords)
        flat = coords[:, 0] * mask.shape[1] + coords[:, 1]
        matrices.append(features[flat])
        labels.append(target)
        group = row.get("cv_group") or row["sample_id"]
        groups.append(np.full(len(target), group, dtype=object))
        sampling_rows.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "cv_group": group,
                **stats,
                "total": len(target),
                "feature_count": len(names),
            }
        )
        print(f"Muestreo [{index:02d}/{len(rows)}] {row['sample_id']}: {len(target)} píxeles")
    return np.vstack(matrices), np.concatenate(labels), np.concatenate(groups), sampling_rows


def estimator(params: dict[str, object], seed: int = SEED) -> RandomForestClassifier:
    return RandomForestClassifier(**params, random_state=seed, n_jobs=-1)


def parameter_candidates() -> list[tuple[str, dict[str, object], str]]:
    candidates: list[tuple[str, dict[str, object], str]] = [
        ("C00_control", dict(CONTROL_PARAMS), "pre_registered_control")
    ]
    sampled = list(
        ParameterSampler(
            SEARCH_SPACE,
            n_iter=SEARCH_CANDIDATE_COUNT - 1,
            random_state=SEED,
        )
    )
    for index, params in enumerate(sampled, start=1):
        candidates.append((f"C{index:02d}_search", dict(params), "seeded_parameter_sample"))
    return candidates


def grouped_screening(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[list[dict], list[dict]]:
    unique_groups = sorted(set(groups.tolist()))
    if unique_groups != ["G1", "G2", "G3", "G4"]:
        raise RuntimeError(f"Grupos internos inesperados: {unique_groups}")
    fold_rows: list[dict] = []
    candidates = parameter_candidates()
    splitter = GroupKFold(n_splits=4)
    for candidate_index, (candidate_id, params, origin) in enumerate(candidates, start=1):
        start = time.perf_counter()
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y, groups), start=1):
            model = estimator(params)
            model.fit(x[train_idx], y[train_idx])
            train_pred = model.predict(x[train_idx])
            valid_pred = model.predict(x[valid_idx])
            fold_rows.append(
                {
                    "candidate_id": candidate_id,
                    "origin": origin,
                    "fold": fold,
                    "held_out_asset_group": sorted(set(groups[valid_idx].tolist()))[0],
                    "train_jaccard": jaccard_score(y[train_idx], train_pred),
                    "validation_jaccard": jaccard_score(y[valid_idx], valid_pred),
                    "train_pixels": len(train_idx),
                    "validation_pixels": len(valid_idx),
                    "parameters": json.dumps(json_safe(params), sort_keys=True),
                }
            )
        elapsed = time.perf_counter() - start
        print(f"Cribado [{candidate_index:02d}/{len(candidates)}] {candidate_id}: {elapsed:.1f} s")
        write_csv(SEARCH_FOLDS, fold_rows)

    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for row in fold_rows:
        grouped[str(row["candidate_id"])].append(row)
    summary: list[dict] = []
    for candidate_id, values in grouped.items():
        validation = [float(row["validation_jaccard"]) for row in values]
        training = [float(row["train_jaccard"]) for row in values]
        summary.append(
            {
                "candidate_id": candidate_id,
                "origin": values[0]["origin"],
                "mean_group_cv_jaccard": float(np.mean(validation)),
                "std_group_cv_jaccard": float(np.std(validation, ddof=1)),
                "mean_train_jaccard": float(np.mean(training)),
                "generalization_gap": float(np.mean(training) - np.mean(validation)),
                "parameters": values[0]["parameters"],
            }
        )
    summary.sort(key=lambda item: (-float(item["mean_group_cv_jaccard"]), float(item["generalization_gap"])))
    for rank, row in enumerate(summary, start=1):
        row["screening_rank"] = rank
    write_csv(SEARCH_SUMMARY, summary)
    return fold_rows, summary


def predict_probability(model: RandomForestClassifier, features: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    probabilities = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), PREDICTION_CHUNK):
        stop = min(len(features), start + PREDICTION_CHUNK)
        probabilities[start:stop] = model.predict_proba(features[start:stop])[:, 1]
    return probabilities.reshape(shape)


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
    dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1,) * 2)
    ref_zone = cv2.dilate(ref_edge.astype(np.uint8), dilation) > 0
    pred_zone = cv2.dilate(pred_edge.astype(np.uint8), dilation) > 0
    precision = np.count_nonzero(pred_edge & ref_zone) / max(1, np.count_nonzero(pred_edge))
    recall = np.count_nonzero(ref_edge & pred_zone) / max(1, np.count_nonzero(ref_edge))
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def component_count(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return sum(int(stats[label, cv2.CC_STAT_AREA]) >= COMPONENT_AREA for label in range(1, count))


def stage2_candidate_ids(screening_summary: list[dict]) -> list[str]:
    top = [str(row["candidate_id"]) for row in screening_summary[:STAGE2_TOP_COUNT]]
    if "C00_control" not in top:
        top.append("C00_control")
    return top


def full_sheet_validation(
    models: dict[str, RandomForestClassifier],
    rows: list[dict[str, str]],
) -> tuple[list[dict], list[dict]]:
    details: list[dict] = []
    for row_index, row in enumerate(rows, start=1):
        image, reference = load_image_mask(row)
        features, names = extract_pixel_features(image, include_coords=False)
        if tuple(names) != BASE_FEATURE_NAMES:
            raise RuntimeError("Las características del modelo no coinciden con el contrato de 19 variables.")
        expected = component_count(reference)
        for candidate_id, model in models.items():
            probability = predict_probability(model, features, reference.shape)
            for threshold in THRESHOLDS:
                raw = (probability >= threshold).astype(np.uint8) * 255
                prediction, _ = postprocess_rf_mask(raw)
                dice, iou = dice_iou(reference, prediction)
                predicted = component_count(prediction)
                details.append(
                    {
                        "candidate_id": candidate_id,
                        "sample_id": row["sample_id"],
                        "split": "val",
                        "layout": row["layout"],
                        "condition": row["condition"],
                        "threshold": threshold,
                        "dice": dice,
                        "iou": iou,
                        "expected_components": expected,
                        "predicted_components": predicted,
                        "component_error": abs(predicted - expected),
                    }
                )
        print(f"Validación completa [{row_index:02d}/{len(rows)}] {row['sample_id']}")
        write_csv(VALIDATION_DETAILS, details)

    grouped: defaultdict[tuple[str, float], list[dict]] = defaultdict(list)
    for item in details:
        grouped[(str(item["candidate_id"]), float(item["threshold"]))].append(item)
    summary: list[dict] = []
    for (candidate_id, threshold), values in grouped.items():
        mean_iou = float(np.mean([item["iou"] for item in values]))
        mean_error = float(np.mean([item["component_error"] for item in values]))
        summary.append(
            {
                "candidate_id": candidate_id,
                "threshold": threshold,
                "mean_validation_iou": mean_iou,
                "std_validation_iou": float(np.std([item["iou"] for item in values], ddof=1)),
                "mean_validation_dice": float(np.mean([item["dice"] for item in values])),
                "mean_component_error": mean_error,
                "selection_score": mean_iou - SELECTION_COMPONENT_PENALTY * mean_error,
                "sheet_count": len(values),
            }
        )
    summary.sort(
        key=lambda item: (
            -float(item["selection_score"]),
            -float(item["mean_validation_iou"]),
            float(item["mean_component_error"]),
        )
    )
    for rank, item in enumerate(summary, start=1):
        item["selection_rank"] = rank
    write_csv(VALIDATION_SUMMARY, summary)
    return details, summary


def model_package(
    model: RandomForestClassifier,
    candidate_id: str,
    threshold: float,
    params: dict[str, object],
    train_rows: list[dict[str, str]],
) -> dict:
    return {
        "model": model,
        "candidate_id": candidate_id,
        "feature_names": list(BASE_FEATURE_NAMES),
        "feature_count": 19,
        "feature_contract": {
            "color": 9,
            "sobel_gradients": 3,
            "multiscale": 3,
            "local_statistics": 4,
            "laplacian_included": False,
            "coordinate_features_included": False,
        },
        "threshold": threshold,
        "parameters": params,
        "random_seed": SEED,
        "dataset_manifest": MANIFEST.relative_to(ROOT).as_posix(),
        "dataset_manifest_sha256": file_sha256(MANIFEST),
        "training_sheets": [row["sample_id"] for row in train_rows],
        "training_asset_groups": sorted({row["cv_group"] for row in train_rows}),
        "split_protocol": "asset_identity_disjoint",
    }


def select() -> None:
    if SELECTION_LOCK.exists() or TEST_SUMMARY.exists():
        raise RuntimeError(
            "Ya existe una selección bloqueada o una evaluación de prueba. "
            "No se sobrescribe automáticamente un resultado experimental."
        )
    METRICS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = read_manifest()
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "val"]
    started = time.perf_counter()
    x_train, y_train, groups, sampling_rows = sample_rows(
        train_rows, SAMPLES_PER_CLASS_PER_SHEET, seed_offset=1_000
    )
    write_csv(METRICS / "rf_idv2_training_sample_summary.csv", sampling_rows)
    _, screening = grouped_screening(x_train, y_train, groups)
    candidate_lookup = {candidate_id: (params, origin) for candidate_id, params, origin in parameter_candidates()}
    selected_ids = stage2_candidate_ids(screening)
    models: dict[str, RandomForestClassifier] = {}
    for candidate_id in selected_ids:
        params, _ = candidate_lookup[candidate_id]
        model = estimator(params)
        model.fit(x_train, y_train)
        models[candidate_id] = model
        print(f"Ajustado para etapa 2: {candidate_id}")
    _, validation_summary = full_sheet_validation(models, validation_rows)
    winner = validation_summary[0]
    winner_id = str(winner["candidate_id"])
    winner_threshold = float(winner["threshold"])
    winner_params, winner_origin = candidate_lookup[winner_id]

    selected_package = model_package(
        models[winner_id], winner_id, winner_threshold, winner_params, train_rows
    )
    joblib.dump(selected_package, SELECTED_MODEL, compress=3)
    control_best = next(
        item for item in validation_summary if item["candidate_id"] == "C00_control"
    )
    control_package = model_package(
        models["C00_control"],
        "C00_control",
        float(control_best["threshold"]),
        CONTROL_PARAMS,
        train_rows,
    )
    joblib.dump(control_package, CONTROL_MODEL, compress=3)
    elapsed = time.perf_counter() - started
    lock = {
        "status": "locked_before_test",
        "selected_candidate": winner_id,
        "selected_candidate_origin": winner_origin,
        "selected_threshold": winner_threshold,
        "selection_rule": "mean_full_sheet_validation_iou - 0.005 * mean_absolute_component_count_error",
        "pixel_screening_rule": "mean Jaccard across four asset-identity-disjoint GroupKFold folds",
        "test_set_used_for_selection": False,
        "stage2_candidates": selected_ids,
        "validation": winner,
        "selected_hyperparameters": winner_params,
        "control_hyperparameters": CONTROL_PARAMS,
        "search_space": SEARCH_SPACE,
        "search_candidate_count_including_control": SEARCH_CANDIDATE_COUNT,
        "threshold_grid": THRESHOLDS,
        "component_area_px": COMPONENT_AREA,
        "samples_per_class_per_training_sheet": SAMPLES_PER_CLASS_PER_SHEET,
        "seed": SEED,
        "dataset_manifest_sha256": file_sha256(MANIFEST),
        "selected_model": SELECTED_MODEL.relative_to(ROOT).as_posix(),
        "selected_model_sha256": file_sha256(SELECTED_MODEL),
        "control_model": CONTROL_MODEL.relative_to(ROOT).as_posix(),
        "control_model_sha256": file_sha256(CONTROL_MODEL),
        "elapsed_seconds": elapsed,
    }
    write_json(SELECTION_LOCK, lock)
    write_json(
        RUN_CONFIG,
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "feature_names": BASE_FEATURE_NAMES,
            "feature_groups": FEATURE_GROUPS,
            "script": Path(__file__).relative_to(ROOT).as_posix(),
            "script_sha256": file_sha256(Path(__file__)),
        },
    )
    plot_results()
    print(json.dumps(json_safe(lock), indent=2, ensure_ascii=False))


def evaluate_model_on_test(
    name: str,
    package: dict,
    rows: list[dict[str, str]],
    write_masks: bool,
) -> list[dict]:
    model = package["model"]
    threshold = float(package["threshold"])
    if tuple(package["feature_names"]) != BASE_FEATURE_NAMES:
        raise RuntimeError(f"Contrato de características no válido en {name}.")
    results: list[dict] = []
    if write_masks:
        MASKS.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        image, reference = load_image_mask(row)
        started = time.perf_counter()
        features, _ = extract_pixel_features(image, include_coords=False)
        probability = predict_probability(model, features, reference.shape)
        prediction, _ = postprocess_rf_mask((probability >= threshold).astype(np.uint8) * 255)
        elapsed = time.perf_counter() - started
        dice, iou = dice_iou(reference, prediction)
        expected = component_count(reference)
        predicted = component_count(prediction)
        results.append(
            {
                "model": name,
                "sample_id": row["sample_id"],
                "split": "test",
                "layout": row["layout"],
                "condition": row["condition"],
                "threshold": threshold,
                "dice": dice,
                "iou": iou,
                "boundary_f1": boundary_f1(reference, prediction),
                "expected_components": expected,
                "predicted_components": predicted,
                "component_error": abs(predicted - expected),
                "inference_seconds_including_features": elapsed,
            }
        )
        if write_masks:
            cv2.imwrite(str(MASKS / f"{row['sample_id']}_{name}.png"), prediction)
        print(f"Prueba {name} [{index:02d}/{len(rows)}] {row['sample_id']}: IoU={iou:.5f}")
    return results


def summarize_test(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    summary: dict[str, dict[str, float | int]] = {}
    for model, values in grouped.items():
        summary[model] = {
            "sheet_count": len(values),
            "mean_iou": float(np.mean([item["iou"] for item in values])),
            "std_iou": float(np.std([item["iou"] for item in values], ddof=1)),
            "mean_dice": float(np.mean([item["dice"] for item in values])),
            "mean_boundary_f1": float(np.mean([item["boundary_f1"] for item in values])),
            "mean_component_error": float(np.mean([item["component_error"] for item in values])),
            "mean_inference_seconds_including_features": float(
                np.mean([item["inference_seconds_including_features"] for item in values])
            ),
        }
    return summary


def test_locked() -> None:
    if TEST_SUMMARY.exists() or TEST_DETAILS.exists():
        raise RuntimeError(
            "La prueba bloqueada ya fue evaluada. El script se detiene para evitar iteraciones sobre test."
        )
    lock = json.loads(SELECTION_LOCK.read_text(encoding="utf-8"))
    if lock.get("status") != "locked_before_test" or lock.get("test_set_used_for_selection") is not False:
        raise RuntimeError("El artefacto de selección no certifica bloqueo previo a prueba.")
    if file_sha256(MANIFEST) != lock["dataset_manifest_sha256"]:
        raise RuntimeError("El manifiesto cambió después de bloquear la selección.")
    if file_sha256(SELECTED_MODEL) != lock["selected_model_sha256"]:
        raise RuntimeError("El modelo seleccionado cambió después del bloqueo.")
    if file_sha256(CONTROL_MODEL) != lock["control_model_sha256"]:
        raise RuntimeError("El modelo de control cambió después del bloqueo.")
    rows = [row for row in read_manifest() if row["split"] == "test"]
    selected = joblib.load(SELECTED_MODEL)
    control = joblib.load(CONTROL_MODEL)
    details = evaluate_model_on_test("RF_selected", selected, rows, write_masks=True)
    if selected["candidate_id"] != "C00_control":
        details.extend(evaluate_model_on_test("RF_control", control, rows, write_masks=False))
    write_csv(TEST_DETAILS, details)
    summary = {
        "status": "test_evaluated_once_after_lock",
        "selection_lock_sha256": file_sha256(SELECTION_LOCK),
        "test_set_used_for_selection": False,
        "dataset_manifest_sha256": file_sha256(MANIFEST),
        "models": summarize_test(details),
    }
    write_json(TEST_SUMMARY, summary)
    plot_results()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def ablate() -> None:
    if ABLATION_CSV.exists():
        raise RuntimeError("La ablación ya existe; no se sobrescribe automáticamente.")
    lock = json.loads(SELECTION_LOCK.read_text(encoding="utf-8"))
    selected_package = joblib.load(SELECTED_MODEL)
    params = dict(lock["selected_hyperparameters"])
    rows = read_manifest()
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    x_train, y_train, _, _ = sample_rows(train_rows, SAMPLES_PER_CLASS_PER_SHEET, seed_offset=1_000)
    x_val, y_val, _, _ = sample_rows(val_rows, VALIDATION_ABLATION_PER_CLASS, seed_offset=9_000, include_coords=True)
    feature_sets: dict[str, list[int]] = {
        "color_9": list(range(9)),
        "color_gradients_12": list(range(12)),
        "full_19": list(range(19)),
        "full_19_plus_coordinates_21": list(range(21)),
    }
    rows_out: list[dict] = []
    for index, (name, columns) in enumerate(feature_sets.items(), start=1):
        train_matrix = x_train if len(columns) == 19 else (
            np.column_stack((x_train, np.zeros((len(x_train), 2), dtype=x_train.dtype)))[:, columns]
            if len(columns) == 21
            else x_train[:, columns]
        )
        if len(columns) == 21:
            # Las coordenadas de entrenamiento se extraen de nuevo para no usar ceros.
            train_matrix, y_train_coords, _, _ = sample_rows(
                train_rows,
                SAMPLES_PER_CLASS_PER_SHEET,
                seed_offset=1_000,
                include_coords=True,
            )
            if not np.array_equal(y_train_coords, y_train):
                raise RuntimeError("El muestreo con coordenadas no conservó las etiquetas.")
        model = estimator(params, seed=SEED + index)
        model.fit(train_matrix[:, columns] if train_matrix.shape[1] > len(columns) else train_matrix, y_train)
        prediction = model.predict(x_val[:, columns])
        rows_out.append(
            {
                "ablation": name,
                "feature_count": len(columns),
                "validation_sampled_pixel_jaccard": jaccard_score(y_val, prediction),
                "validation_pixel_count": len(y_val),
                "seed": SEED + index,
                "role": "diagnostic_only_not_used_for_model_selection",
            }
        )
        print(f"Ablación [{index}/{len(feature_sets)}] {name}")
    write_csv(ABLATION_CSV, rows_out)

    importance = selected_package["model"].feature_importances_
    importance_rows = [
        {"feature": name, "importance": float(value), "rank": 0}
        for name, value in zip(BASE_FEATURE_NAMES, importance)
    ]
    importance_rows.sort(key=lambda item: -float(item["importance"]))
    for rank, item in enumerate(importance_rows, start=1):
        item["rank"] = rank
    write_csv(IMPORTANCE_CSV, importance_rows)
    plot_results()
    print(json.dumps(rows_out, indent=2, ensure_ascii=False))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plot_results() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "pdf.fonttype": 42,
    })
    screening = read_csv_rows(SEARCH_SUMMARY)
    if screening:
        ordered = sorted(screening, key=lambda row: int(row["screening_rank"]))
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        x = np.arange(len(ordered))
        means = [float(row["mean_group_cv_jaccard"]) for row in ordered]
        errors = [float(row["std_group_cv_jaccard"]) for row in ordered]
        colors = ["#a51c30" if row["candidate_id"] == "C00_control" else "#2d6a9f" for row in ordered]
        ax.bar(x, means, yerr=errors, capsize=3, color=colors)
        ax.set_xticks(x, [row["candidate_id"] for row in ordered], rotation=45, ha="right")
        ax.set_ylabel("Jaccard medio (GroupKFold)")
        ax.set_title("Cribado de hiperparámetros por cohortes de identidad")
        lower = max(0.0, min(mean - error for mean, error in zip(means, errors)) - 0.01)
        ax.set_ylim(lower, 1.005)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES / "rf_idv2_hyperparameter_screening.png", dpi=180)
        plt.close(fig)

    validation = read_csv_rows(VALIDATION_SUMMARY)
    if validation:
        best_by_candidate: dict[str, dict[str, str]] = {}
        for row in validation:
            best_by_candidate.setdefault(row["candidate_id"], row)
        ordered = sorted(best_by_candidate.values(), key=lambda row: int(row["selection_rank"]))
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        x = np.arange(len(ordered))
        scores = [float(row["selection_score"]) for row in ordered]
        ax.scatter(x, scores, color="#3c7d5a", s=70, zorder=3)
        ax.vlines(x, min(scores) - 0.0004, scores, color="#3c7d5a", alpha=0.45)
        ax.set_xticks(x, [row["candidate_id"] for row in ordered])
        ax.set_ylabel("IoU − 0,005 × error de componentes")
        ax.set_title("Selección sobre láminas completas de validación")
        ax.set_ylim(min(scores) - 0.0004, max(scores) + 0.0004)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES / "rf_idv2_full_sheet_validation.png", dpi=180)
        fig.savefig(FIGURES / "rf_idv2_full_sheet_validation.pdf")
        plt.close(fig)

    ablation = read_csv_rows(ABLATION_CSV)
    if ablation:
        fig, ax = plt.subplots(figsize=(5.1, 3.5))
        labels = [row["ablation"].replace("_", " ") for row in ablation]
        values = [float(row["validation_sampled_pixel_jaccard"]) for row in ablation]
        positions = np.arange(len(labels))
        ax.scatter(values, positions, color="#7d5794", s=75, zorder=3)
        ax.hlines(positions, min(values) - 0.003, values, color="#7d5794", alpha=0.45)
        ax.set_yticks(positions, labels)
        ax.set_xlabel("Jaccard en píxeles de validación (diagnóstico)")
        ax.set_title("Ablación de familias de características")
        ax.set_xlim(min(values) - 0.003, max(values) + 0.003)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES / "rf_idv2_feature_ablation.png", dpi=180)
        fig.savefig(FIGURES / "rf_idv2_feature_ablation.pdf")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("select", "ablate", "test", "plot"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "select":
        select()
    elif args.command == "ablate":
        ablate()
    elif args.command == "test":
        test_locked()
    else:
        plot_results()


if __name__ == "__main__":
    main()
