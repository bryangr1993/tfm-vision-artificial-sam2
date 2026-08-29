"""Utilidades compartidas para el protocolo reproducible de SAM 2.

La implementación reproduce las decisiones del producto sin importar la GUI:
margen de caja del 5 %, selección por área de la caja ya ampliada, salida
multimáscara y limpieza morfológica con componentes de al menos 300 píxeles.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = ROOT / "datos" / "manifiesto" / "datasets.csv"
CHECKPOINT = ROOT / "resultados" / "modelos" / "sam2_hiera_tiny.pt"
MODEL_CONFIG = "configs/sam2/sam2_hiera_t.yaml"

sys.path.insert(0, str(ROOT / "software" / "src"))
from detect_wcs_l import detect_wcs_l
from segmenters.classical import ClassicalPromptLocalizer
from segmenters.prompts import boxes_from_mask, pad_box
from segmenters.sam2_segmenter import SAM2Segmenter


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No hay filas para escribir en {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_required_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer {path}")
    return image


def binary_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def dice_iou(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    ref = reference > 0
    pred = prediction > 0
    intersection = int(np.count_nonzero(ref & pred))
    union = int(np.count_nonzero(ref | pred))
    denominator = int(np.count_nonzero(ref) + np.count_nonzero(pred))
    dice = 2.0 * intersection / denominator if denominator else 1.0
    iou = intersection / union if union else 1.0
    return dice, iou


def boundary_f1(reference: np.ndarray, prediction: np.ndarray, tolerance: int = 3) -> float:
    kernel = np.ones((3, 3), np.uint8)
    ref = binary_mask(reference)
    pred = binary_mask(prediction)
    ref_edge = cv2.morphologyEx(ref, cv2.MORPH_GRADIENT, kernel) > 0
    pred_edge = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, kernel) > 0
    zone_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1,) * 2)
    ref_zone = cv2.dilate(ref_edge.astype(np.uint8), zone_kernel) > 0
    pred_zone = cv2.dilate(pred_edge.astype(np.uint8), zone_kernel) > 0
    precision = np.count_nonzero(pred_edge & ref_zone) / max(1, np.count_nonzero(pred_edge))
    recall = np.count_nonzero(ref_edge & pred_zone) / max(1, np.count_nonzero(ref_edge))
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def component_count(mask: np.ndarray, min_area: int = 20_000) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask(mask))
    return sum(int(stats[label, cv2.CC_STAT_AREA]) >= min_area for label in range(1, count))


def bootstrap_mean_ci(values: Iterable[float], seed: int) -> tuple[float, float]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return math.nan, math.nan
    if data.size == 1:
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(10_000, data.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def synchronize(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero torch.cuda.is_available() es False.")
    return requested


def build_predictor(device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint de SAM 2: {CHECKPOINT}")
    model = build_sam2(MODEL_CONFIG, str(CHECKPOINT), device=device)
    return SAM2ImagePredictor(model)


class ProductAlignedSAM2:
    """Ejecuta SAM 2 con exactamente las decisiones del segmentador operativo."""

    def __init__(
        self,
        predictor,
        *,
        device: str,
        box_margin_fraction: float = 0.05,
        min_component_area_px: int = 300,
    ) -> None:
        self.predictor = predictor
        self.device = device
        self.box_margin_fraction = box_margin_fraction
        self.segmenter = SAM2Segmenter(
            CHECKPOINT,
            device=device,
            box_margin_fraction=box_margin_fraction,
            min_component_area_px=min_component_area_px,
            predictor=predictor,
        )

    def predict(
        self,
        image: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
    ) -> tuple[np.ndarray, list[float], dict[str, float], list[tuple[int, int, int, int]]]:
        if not boxes:
            raise ValueError("SAM 2 necesita al menos una caja.")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        synchronize(self.device)
        total_start = time.perf_counter()
        encoder_start = total_start
        self.predictor.set_image(rgb)
        synchronize(self.device)
        encoder_seconds = time.perf_counter() - encoder_start

        accumulated = np.zeros(image.shape[:2], dtype=np.uint8)
        scores: list[float] = []
        padded_boxes: list[tuple[int, int, int, int]] = []
        synchronize(self.device)
        decoder_start = time.perf_counter()
        for box in boxes:
            padded = pad_box(box, image.shape, self.box_margin_fraction)
            x1, y1, x2, y2 = padded
            padded_area = max(1, (x2 - x1) * (y2 - y1))
            masks, predicted_scores, _ = self.predictor.predict(
                box=np.asarray(padded, dtype=np.float32),
                multimask_output=True,
            )
            selected, score = self.segmenter._select_mask(
                masks,
                predicted_scores,
                padded_area,
            )
            accumulated[selected.astype(bool)] = 255
            scores.append(score)
            padded_boxes.append(padded)
        synchronize(self.device)
        decoder_seconds = time.perf_counter() - decoder_start

        post_start = time.perf_counter()
        prediction = self.segmenter._postprocess(accumulated)
        postprocess_seconds = time.perf_counter() - post_start
        total_seconds = time.perf_counter() - total_start
        timing = {
            "encoder_seconds": encoder_seconds,
            "decoder_seconds": decoder_seconds,
            "postprocess_seconds": postprocess_seconds,
            "sam_total_seconds": total_seconds,
        }
        return prediction, scores, timing, padded_boxes


def ideal_boxes(row: dict[str, str], reference: np.ndarray) -> tuple[list[tuple[int, int, int, int]], str]:
    if row["domain"] == "synthetic":
        prompt_path = ROOT / "datos" / "sinteticos" / "prompts" / (
            f"synthetic_{row['sample_id']}_prompts.json"
        )
        prompt_data = json.loads(prompt_path.read_text(encoding="utf-8"))
        boxes = [tuple(map(int, prompt["bbox"])) for prompt in prompt_data["prompts"]]
        return boxes, "generator_metadata_oracle"
    boxes = boxes_from_mask(reference, expected_instances=8)
    return boxes, "assisted_reference_oracle_diagnostic"


def operational_boxes(
    image: np.ndarray,
    *,
    wcs_info: dict | None,
    scale: float = 10.0,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray, float]:
    localizer = ClassicalPromptLocalizer()
    start = time.perf_counter()
    result = localizer.segment(image, scale=scale, wcs_info=wcs_info)
    elapsed = time.perf_counter() - start
    return result.prompt_boxes, result.mask, elapsed


def detect_wcs_for_rectified(image: np.ndarray, sample_id: str) -> dict:
    return detect_wcs_l(image, debug_dir=None, img_name=sample_id, marker_margin=10.0, scale=10.0)


def sample_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    dice, iou = dice_iou(reference, prediction)
    expected = component_count(reference)
    predicted = component_count(prediction)
    return {
        "dice": dice,
        "iou": iou,
        "boundary_f1": boundary_f1(reference, prediction),
        "expected_components": expected,
        "predicted_components": predicted,
        "component_error": abs(predicted - expected),
        "foreground_fraction": float(np.count_nonzero(prediction) / prediction.size),
    }
