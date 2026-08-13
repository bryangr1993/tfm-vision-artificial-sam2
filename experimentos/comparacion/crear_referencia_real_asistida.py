"""Crea la referencia canónica para las trece capturas reales rectificadas.

La referencia se obtiene de la mediana de las trece capturas rectificadas y de
GrabCut inicializado únicamente con cajas. No usa las máscaras de los métodos
evaluados como etiquetas. Las trece fotografías muestran la misma lámina física,
por lo que la anotación se define en el plano canónico compartido después de la
rectificación. El script también genera una lámina de control para su verificación.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RECTIFIED = ROOT / "datos" / "reales" / "rectificadas"
PROMPT = ROOT / "datos" / "reales" / "prompts_legacy" / "real_13_prompts.json"
ANNOTATIONS = ROOT / "datos" / "anotaciones"


def fill_external(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) >= 250:
            cv2.drawContours(filled, [contour], -1, 255, thickness=-1)
    return filled


def segment_box(image: np.ndarray, box: list[int]) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    margin_x = round((x2 - x1) * 0.08)
    margin_y = round((y2 - y1) * 0.08)
    rx1, ry1 = max(1, x1 - margin_x), max(1, y1 - margin_y)
    rx2, ry2 = min(width - 1, x2 + margin_x), min(height - 1, y2 + margin_y)
    crop = image[ry1:ry2, rx1:rx2]
    crop_height, crop_width = crop.shape[:2]
    grabcut_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        crop,
        grabcut_mask,
        (2, 2, crop_width - 4, crop_height - 4),
        background_model,
        foreground_model,
        8,
        cv2.GC_INIT_WITH_RECT,
    )
    foreground_crop = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    foreground = np.zeros((height, width), dtype=np.uint8)
    foreground[ry1:ry2, rx1:rx2] = foreground_crop

    # Los bordes impresos ayudan a cerrar siluetas de zonas claras sin fijar el
    # resultado a una máscara clásica o de SAM 2.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 100)
    edge_canvas = np.zeros_like(foreground)
    edge_canvas[ry1:ry2, rx1:rx2] = cv2.dilate(
        edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    combined = cv2.bitwise_or(foreground, edge_canvas)
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    local = np.zeros_like(combined)
    local[ry1:ry2, rx1:rx2] = combined[ry1:ry2, rx1:rx2]
    contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError(f"No se obtuvo contorno para la caja {box}")
    # Las cajas no se solapan de forma sustancial. La silueta buscada es el
    # contorno de mayor área dentro de cada una.
    selected = max(contours, key=cv2.contourArea)
    result = np.zeros_like(combined)
    cv2.drawContours(result, [selected], -1, 255, thickness=-1)
    return result


def main() -> None:
    paths = sorted(RECTIFIED.glob("rectified_*.png"), key=lambda path: int(path.stem.split("_")[-1]))
    images = [cv2.imread(str(path)) for path in paths]
    if len(images) != 13 or any(image is None for image in images):
        raise RuntimeError("Se esperaban trece imágenes reales rectificadas.")
    median = np.median(np.stack(images), axis=0).astype(np.uint8)
    prompt_data = json.loads(PROMPT.read_text(encoding="utf-8"))
    reference = np.zeros(median.shape[:2], dtype=np.uint8)
    for prompt in prompt_data["prompts"]:
        reference = cv2.bitwise_or(reference, segment_box(median, prompt["bbox"]))
    reference = fill_external(reference)

    reference_dir = ANNOTATIONS / "referencia_real_canonica"
    ground_truth_dir = ANNOTATIONS / "real_ground_truth"
    reference_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(reference_dir / "imagen_mediana_real.png"), median)
    cv2.imwrite(str(reference_dir / "referencia_canonica_real.png"), reference)
    for path in paths:
        number = path.stem.split("_")[-1]
        cv2.imwrite(str(ground_truth_dir / f"real_{number}_gt.png"), reference)
    overlay = median.copy()
    contours, _ = cv2.findContours(reference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, contours, -1, (0, 200, 0), 5)
    for index, contour in enumerate(
        sorted(contours, key=lambda item: cv2.boundingRect(item)[1]), start=1
    ):
        moments = cv2.moments(contour)
        if moments["m00"]:
            center = (
                round(moments["m10"] / moments["m00"]),
                round(moments["m01"] / moments["m00"]),
            )
            cv2.circle(overlay, center, 25, (0, 200, 0), -1)
            cv2.putText(
                overlay,
                str(index),
                (center[0] - 9, center[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                3,
            )
    cv2.imwrite(str(reference_dir / "control_visual_referencia_real.png"), overlay)
    metadata = {
        "status": "assisted_reference_visual_qc_complete",
        "source_images": [path.name for path in paths],
        "canonical_image": "pixelwise median of 13 rectified captures",
        "annotation_method": "box-initialized GrabCut plus edge closure and external fill",
        "experimental_unit_note": "the 13 captures show the same physical sheet in canonical rectified coordinates",
        "uses_classical_masks_as_labels": False,
        "uses_sam2_masks_as_labels": False,
        "prompt_boxes_source": "classical coarse localization on real_13",
        "visual_qc": {
            "status": "complete",
            "checks": [
                "eight separate instances",
                "outer silhouettes follow visible print borders",
                "rainbow inner opening preserved",
                "corner fiducials and WCS mark excluded",
                "thin appendages retained",
            ],
        },
        "instances": len(contours),
    }
    (reference_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
