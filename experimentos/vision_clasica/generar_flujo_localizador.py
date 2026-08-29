"""Genera una figura auditable con las etapas del localizador clásico.

La visualización reproduce los parámetros de ``segment_toppers.py`` sobre la
captura real 20. Su salida es explicativa. La máscara clásica solo suministra
las cajas que condicionan a SAM 2 en la aplicación.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "software" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detect_wcs_l import detect_wcs_l
from segmenters.prompts import boxes_from_mask, pad_box


def build_stages(image: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    height, width = image.shape[:2]
    scale = 10.0
    wcs = detect_wcs_l(image, scale=scale)

    keep = np.full((height, width), 255, dtype=np.uint8)
    margin = round(8.0 * scale)
    marker = round(18.0 * scale)
    keep[:margin, :] = 0
    keep[-margin:, :] = 0
    keep[:, :margin] = 0
    keep[:, -margin:] = 0
    keep[:marker, :marker] = 0
    keep[:marker, -marker:] = 0
    keep[-marker:, :marker] = 0
    keep[-marker:, -marker:] = 0
    if wcs["status"] == "SUCCESS":
        origin = tuple(round(value) for value in wcs["origin"])
        end_x = tuple(
            round(origin[index] + wcs["uX"][index] * 230) for index in range(2)
        )
        end_y = tuple(
            round(origin[index] + wcs["uY"][index] * 230) for index in range(2)
        )
        cv2.line(keep, origin, end_x, 0, thickness=35)
        cv2.line(keep, origin, end_y, 0, thickness=35)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hsv_union = np.where((saturation > 45) | (value < 115), 255, 0).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    combined = cv2.bitwise_or(hsv_union, edges)
    combined = cv2.bitwise_and(combined, keep)
    closed = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    flood = closed.copy()
    cv2.floodFill(flood, np.zeros((height + 2, width + 2), dtype=np.uint8), (0, 0), 255)
    filled = cv2.bitwise_or(closed, cv2.bitwise_not(flood))
    opened = cv2.morphologyEx(
        filled,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    final_mask = np.zeros_like(opened)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= 5_000:
            final_mask[labels == label] = 255

    boxes = [pad_box(box, image.shape, 0.05) for box in boxes_from_mask(final_mask)]
    box_view = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for index, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(box_view, (x1, y1), (x2, y2), (46, 126, 185), 10)
        cv2.putText(
            box_view,
            str(index),
            (x1 + 8, max(35, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (21, 50, 75),
            4,
            cv2.LINE_AA,
        )
    stages = [
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        hsv_union,
        edges,
        combined,
        final_mask,
        box_view,
    ]
    return stages, boxes


def main() -> None:
    source = ROOT / "datos" / "reales" / "rectificadas" / "rectified_20.png"
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(source)
    stages, boxes = build_stages(image)
    if len(boxes) != 8:
        raise RuntimeError(f"Se esperaban ocho cajas y se obtuvieron {len(boxes)}")

    titles = [
        "Hoja rectificada",
        "Saturación u\noscuridad",
        "Bordes Canny",
        "Unión y\nexclusiones",
        "Morfología y\ncomponentes",
        "Ocho cajas\npara SAM 2",
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    figure, axes = plt.subplots(2, 3, figsize=(5.8, 4.7), constrained_layout=True)
    for axis, stage, title in zip(axes.flat, stages, titles):
        axis.imshow(stage, cmap="gray" if stage.ndim == 2 else None)
        axis.set_title(title, color="#15324B", fontweight="bold", pad=5, fontsize=8.5)
        axis.axis("off")
    figure.suptitle(
        "Etapas del localizador clásico en la captura 20",
        color="#15324B",
        fontweight="bold",
        fontsize=10.5,
    )
    output = ROOT / "resultados" / "figuras"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "flujo_localizador_clasico_real20.png", dpi=220)
    figure.savefig(output / "flujo_localizador_clasico_real20.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
