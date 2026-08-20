"""Optimización reproducible de Random Forest sin fuga entre láminas.

La búsqueda utiliza solo las 36 láminas de entrenamiento y validación cruzada
agrupada. El umbral se elige en las seis láminas de validación. Las seis láminas
de prueba se procesan una sola vez al final.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "datos"
MANIFEST = DATA / "manifiesto" / "datasets.csv"
OUTPUT = ROOT / "resultados" / "metricas"
MODELS = ROOT / "resultados" / "modelos"
MASK_OUTPUT = ROOT / "resultados" / "rf_optimized_masks"

sys.path.insert(0, str(HERE))
from feature_extraction import extract_pixel_features
from postprocess_masks import postprocess_rf_mask


SEED = 42
SAMPLES_PER_CLASS = 2_000
N_ITER = 10
CV_FOLDS = 4
THRESHOLDS = np.round(np.arange(0.35, 0.71, 0.05), 2)


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def realistic_augmentation(image: np.ndarray, mask: np.ndarray, seed: int) -> np.ndarray:
    """Introduce variaciones físicas dentro de los rangos de las capturas reales."""

    rng = np.random.default_rng(seed)
    augmented = image.astype(np.float32).copy()
    if rng.uniform() > 0.30:
        background = mask == 0
        targets = (rng.uniform(220, 255), rng.uniform(218, 255), rng.uniform(215, 255))
        for channel, target in enumerate(targets):
            augmented[..., channel][background] *= target / 255.0
    for channel in range(3):
        augmented[..., channel] *= rng.uniform(0.97, 1.03)
    augmented *= rng.uniform(0.88, 1.02)
    augmented += rng.normal(0, rng.uniform(1.2, 3.2), augmented.shape)
    return np.clip(augmented, 0, 255).astype(np.uint8)


def balanced_sample(
    image: np.ndarray,
    mask: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    features, _ = extract_pixel_features(image, include_coords=False)
    foreground = np.argwhere(mask == 255)
    dilated = cv2.dilate(mask, np.ones((35, 35), np.uint8), iterations=1)
    near_background = np.argwhere((dilated == 255) & (mask == 0))
    far_background = np.argwhere(dilated == 0)
    rng = np.random.default_rng(seed)

    def choose(values: np.ndarray, count: int) -> np.ndarray:
        actual = min(len(values), count)
        indexes = rng.choice(len(values), actual, replace=False)
        return values[indexes]

    foreground_sample = choose(foreground, SAMPLES_PER_CLASS)
    near_sample = choose(near_background, SAMPLES_PER_CLASS // 2)
    far_sample = choose(far_background, SAMPLES_PER_CLASS // 2)
    coordinates = np.vstack([foreground_sample, near_sample, far_sample])
    labels = np.concatenate(
        [
            np.ones(len(foreground_sample), dtype=np.uint8),
            np.zeros(len(near_sample) + len(far_sample), dtype=np.uint8),
        ]
    )
    flat_indexes = coordinates[:, 0] * mask.shape[1] + coordinates[:, 1]
    return features[flat_indexes], labels


def training_matrix(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    train_rows = [row for row in rows if row["domain"] == "synthetic" and row["split"] == "train"]
    for index, row in enumerate(train_rows, start=1):
        image = cv2.imread(str(ROOT / row["image"]))
        mask = cv2.imread(str(ROOT / row["ground_truth"]), cv2.IMREAD_GRAYSCALE)
        image = realistic_augmentation(image, mask, SEED + index)
        sampled_features, sampled_labels = balanced_sample(image, mask, SEED * 10 + index)
        features.append(sampled_features)
        labels.append(sampled_labels)
        groups.append(np.full(len(sampled_labels), row["sample_id"], dtype=object))
        print(f"[{index:02d}/{len(train_rows)}] {row['sample_id']}: {len(sampled_labels)} píxeles")
    return np.vstack(features), np.concatenate(labels), np.concatenate(groups)


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
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def predict_probability(model: RandomForestClassifier, image: np.ndarray) -> np.ndarray:
    features, _ = extract_pixel_features(image, include_coords=False)
    return model.predict_proba(features)[:, 1].reshape(image.shape[:2]).astype(np.float32)


def component_count(mask: np.ndarray, min_area: int = 20_000) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return sum(stats[label, cv2.CC_STAT_AREA] >= min_area for label in range(1, count))


def select_threshold(
    model: RandomForestClassifier,
    rows: list[dict[str, str]],
) -> tuple[float, list[dict[str, float | str]]]:
    validation = [row for row in rows if row["domain"] == "synthetic" and row["split"] == "val"]
    aggregates: defaultdict[float, list[dict[str, float]]] = defaultdict(list)
    for index, row in enumerate(validation, start=1):
        image = cv2.imread(str(ROOT / row["image"]))
        reference = cv2.imread(str(ROOT / row["ground_truth"]), cv2.IMREAD_GRAYSCALE)
        probability = predict_probability(model, image)
        for threshold in THRESHOLDS:
            raw = (probability >= threshold).astype(np.uint8) * 255
            prediction, _ = postprocess_rf_mask(raw)
            dice, iou = dice_iou(reference, prediction)
            aggregates[float(threshold)].append(
                {
                    "dice": dice,
                    "iou": iou,
                    "component_error": abs(component_count(prediction) - component_count(reference)),
                }
            )
        print(f"Validación [{index}/{len(validation)}] {row['sample_id']}")

    summary: list[dict[str, float | str]] = []
    for threshold, values in sorted(aggregates.items()):
        mean_iou = float(np.mean([item["iou"] for item in values]))
        mean_dice = float(np.mean([item["dice"] for item in values]))
        mean_component_error = float(np.mean([item["component_error"] for item in values]))
        selection_score = mean_iou - 0.005 * mean_component_error
        summary.append(
            {
                "threshold": threshold,
                "mean_validation_iou": mean_iou,
                "mean_validation_dice": mean_dice,
                "mean_component_error": mean_component_error,
                "selection_score": selection_score,
            }
        )
    winner = max(summary, key=lambda row: float(row["selection_score"]))
    return float(winner["threshold"]), summary


def evaluate_test(
    model: RandomForestClassifier,
    rows: list[dict[str, str]],
    threshold: float,
) -> list[dict[str, float | int | str]]:
    test_rows = [row for row in rows if row["domain"] == "synthetic" and row["split"] == "test"]
    MASK_OUTPUT.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, float | int | str]] = []
    for index, row in enumerate(test_rows, start=1):
        image = cv2.imread(str(ROOT / row["image"]))
        reference = cv2.imread(str(ROOT / row["ground_truth"]), cv2.IMREAD_GRAYSCALE)
        start = time.perf_counter()
        probability = predict_probability(model, image)
        raw = (probability >= threshold).astype(np.uint8) * 255
        prediction, _ = postprocess_rf_mask(raw)
        elapsed = time.perf_counter() - start
        dice, iou = dice_iou(reference, prediction)
        result = {
            "sample_id": row["sample_id"],
            "split": "test",
            "threshold": threshold,
            "dice": dice,
            "iou": iou,
            "boundary_f1": boundary_f1(reference, prediction),
            "component_count": component_count(prediction),
            "expected_components": component_count(reference),
            "component_error": abs(component_count(prediction) - component_count(reference)),
            "inference_seconds": elapsed,
        }
        results.append(result)
        cv2.imwrite(str(MASK_OUTPUT / f"{row['sample_id']}_rf_optimized.png"), prediction)
        print(f"Prueba [{index}/{len(test_rows)}] {row['sample_id']}: IoU={iou:.5f}")
    return results


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    rows = read_manifest()
    x_train, y_train, groups = training_matrix(rows)
    print(f"Matriz de búsqueda: {x_train.shape}; clases={np.bincount(y_train)}")

    estimator = RandomForestClassifier(random_state=SEED, n_jobs=-1, bootstrap=True)
    parameters = {
        "n_estimators": [100, 150, 200, 300],
        "max_depth": [12, 18, 25, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", 0.5, 0.8],
        "class_weight": ["balanced", "balanced_subsample"],
    }
    search = RandomizedSearchCV(
        estimator,
        parameters,
        n_iter=N_ITER,
        scoring="jaccard",
        n_jobs=-1,
        cv=GroupKFold(n_splits=CV_FOLDS),
        refit=True,
        random_state=SEED,
        verbose=2,
        return_train_score=True,
    )
    search_start = time.perf_counter()
    search.fit(x_train, y_train, groups=groups)
    search_seconds = time.perf_counter() - search_start

    search_rows: list[dict] = []
    for rank_index in np.argsort(search.cv_results_["rank_test_score"]):
        search_rows.append(
            {
                "rank": int(search.cv_results_["rank_test_score"][rank_index]),
                "mean_cv_jaccard": float(search.cv_results_["mean_test_score"][rank_index]),
                "std_cv_jaccard": float(search.cv_results_["std_test_score"][rank_index]),
                "mean_train_jaccard": float(search.cv_results_["mean_train_score"][rank_index]),
                "mean_fit_seconds": float(search.cv_results_["mean_fit_time"][rank_index]),
                "parameters": json.dumps(json_safe(search.cv_results_["params"][rank_index]), sort_keys=True),
            }
        )
    write_csv(OUTPUT / "rf_hyperparameter_search.csv", search_rows)

    threshold, threshold_rows = select_threshold(search.best_estimator_, rows)
    write_csv(OUTPUT / "rf_validation_thresholds.csv", threshold_rows)
    test_rows = evaluate_test(search.best_estimator_, rows, threshold)
    write_csv(OUTPUT / "rf_test_metrics.csv", test_rows)

    model_path = MODELS / "random_forest_optimized.joblib"
    joblib.dump(
        {
            "model": search.best_estimator_,
            "feature_names": extract_pixel_features(np.zeros((8, 8, 3), np.uint8))[1],
            "include_coords": False,
            "training_groups": sorted(set(groups.tolist())),
            "hyperparameters": search.best_params_,
            "threshold": threshold,
        },
        model_path,
        compress=3,
    )
    configuration = {
        "seed": SEED,
        "samples_per_class_per_sheet": SAMPLES_PER_CLASS,
        "search_iterations": N_ITER,
        "group_folds": CV_FOLDS,
        "scoring": "jaccard",
        "best_cv_jaccard": search.best_score_,
        "best_hyperparameters": search.best_params_,
        "selected_validation_threshold": threshold,
        "search_seconds": search_seconds,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scikit_learn": sklearn.__version__,
        "model_path": model_path.relative_to(ROOT).as_posix(),
    }
    (OUTPUT / "rf_optimized_configuration.json").write_text(
        json.dumps(json_safe(configuration), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(json_safe(configuration), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
