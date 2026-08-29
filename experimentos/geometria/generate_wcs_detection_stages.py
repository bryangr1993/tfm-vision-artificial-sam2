"""Genera una figura auditable con las etapas del detector WCS actual.

El script no altera el detector. Reproduce su preprocesado para visualizarlo y
obtiene el origen y los ejes finales llamando a ``detect_wcs_l``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SAM_DIR = ROOT / "experimentos" / "sam2"
SOFTWARE_SRC = ROOT / "software" / "src"
FIGURES = ROOT / "resultados" / "figuras_protocolo_v8"
METRICS = ROOT / "resultados" / "metricas"

sys.path.insert(0, str(SAM_DIR))
sys.path.insert(0, str(SOFTWARE_SRC))

from detect_markers import detect_markers
from detect_wcs_l import detect_wcs_l
from load_images import load_image
from protocol_common import load_manifest
from rectify_sheet import rectify_sheet


SAMPLE_ID = "real_20"
SCALE = 10.0
MARKER_MARGIN_MM = 10.0
ROI_SIZE_MM = 45.0
MARKER_EXCLUSION_HALF_SIZE_MM = 8.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess_like_detector(rectified: np.ndarray) -> dict:
    height, width = rectified.shape[:2]
    roi_x_end = min(width, int(ROI_SIZE_MM * SCALE))
    roi_y_end = min(height, int(ROI_SIZE_MM * SCALE))
    roi = rectified[:roi_y_end, :roi_x_end]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    threshold = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        15,
    )

    margin_px = int(MARKER_MARGIN_MM * SCALE)
    exclusion_px = int(MARKER_EXCLUSION_HALF_SIZE_MM * SCALE)
    marker_box = (
        max(0, margin_px - exclusion_px),
        max(0, margin_px - exclusion_px),
        min(roi_x_end, margin_px + exclusion_px),
        min(roi_y_end, margin_px + exclusion_px),
    )
    x1, y1, x2, y2 = marker_box
    threshold[y1:y2, x1:x2] = 0

    lines = cv2.HoughLinesP(
        threshold,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=30,
        maxLineGap=10,
    )
    horizontal: list[tuple[float, int, int, int, int]] = []
    vertical: list[tuple[float, int, int, int, int]] = []
    for line in [] if lines is None else lines:
        lx1, ly1, lx2, ly2 = map(int, line[0])
        dx, dy = lx2 - lx1, ly2 - ly1
        length = float(np.hypot(dx, dy))
        if length < 2.5 * SCALE or length > 25.0 * SCALE:
            continue
        angle = abs(float(np.degrees(np.arctan2(dy, dx))))
        if angle > 90:
            angle = 180 - angle
        item = (length, lx1, ly1, lx2, ly2)
        if angle <= 20:
            horizontal.append(item)
        elif angle >= 70:
            vertical.append(item)
    horizontal.sort(key=lambda item: item[0], reverse=True)
    vertical.sort(key=lambda item: item[0], reverse=True)
    return {
        "roi": roi,
        "enhanced": enhanced,
        "threshold": threshold,
        "marker_box": marker_box,
        "hough_line_count": 0 if lines is None else int(len(lines)),
        "horizontal": horizontal,
        "vertical": vertical,
    }


def add_marker_exclusion(axis, marker_box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = marker_box
    axis.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="#D65F5F",
            linewidth=1.8,
            linestyle="--",
        )
    )


def make_figure(stages: dict, wcs: dict) -> None:
    roi_rgb = cv2.cvtColor(stages["roi"], cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(2, 2, figsize=(5.8, 5.25))

    axes[0, 0].imshow(roi_rgb)
    add_marker_exclusion(axes[0, 0], stages["marker_box"])
    axes[0, 0].set_title("a) ROI de búsqueda 45 × 45 mm", fontsize=8.5)

    axes[0, 1].imshow(stages["threshold"], cmap="gray", vmin=0, vmax=255)
    add_marker_exclusion(axes[0, 1], stages["marker_box"])
    axes[0, 1].set_title("b) Binarización y exclusión de M1", fontsize=8.5)

    axes[1, 0].imshow(roi_rgb)
    for _, x1, y1, x2, y2 in stages["horizontal"][:8]:
        axes[1, 0].plot([x1, x2], [y1, y2], color="#2F6B9A", linewidth=2.1)
    for _, x1, y1, x2, y2 in stages["vertical"][:8]:
        axes[1, 0].plot([x1, x2], [y1, y2], color="#D5822A", linewidth=2.1)
    axes[1, 0].set_title(
        "c) Segmentos Hough\ntras los filtros geométricos",
        fontsize=8.5,
    )
    axes[1, 0].legend(
        handles=[
            Line2D([0], [0], color="#2F6B9A", lw=2.1, label="Horizontales"),
            Line2D([0], [0], color="#D5822A", lw=2.1, label="Verticales"),
        ],
        loc="lower right",
        frameon=True,
        fontsize=8.2,
    )

    axes[1, 1].imshow(roi_rgb)
    ox, oy = map(float, wcs["origin"])
    ux = np.asarray(wcs["uX"], dtype=float)
    uy = np.asarray(wcs["uY"], dtype=float)
    arrow_length = 145.0
    axes[1, 1].scatter([ox], [oy], s=80, color="#C53E3E", edgecolor="white", zorder=5)
    axes[1, 1].arrow(
        ox,
        oy,
        arrow_length * ux[0],
        arrow_length * ux[1],
        width=2.5,
        head_width=13,
        head_length=16,
        color="#2F6B9A",
        length_includes_head=True,
    )
    axes[1, 1].arrow(
        ox,
        oy,
        arrow_length * uy[0],
        arrow_length * uy[1],
        width=2.5,
        head_width=13,
        head_length=16,
        color="#3A913F",
        length_includes_head=True,
    )
    axes[1, 1].annotate("Origen", (ox, oy), xytext=(7, -13), textcoords="offset points", fontsize=8.2)
    axes[1, 1].annotate(
        "X",
        (ox + arrow_length * ux[0], oy + arrow_length * ux[1]),
        xytext=(4, -3),
        textcoords="offset points",
        color="#2F6B9A",
        fontweight="bold",
    )
    axes[1, 1].annotate(
        "Y",
        (ox + arrow_length * uy[0], oy + arrow_length * uy[1]),
        xytext=(4, -3),
        textcoords="offset points",
        color="#3A913F",
        fontweight="bold",
    )
    axes[1, 1].set_title("d) Origen y ejes\nvalidados", fontsize=8.5)

    for axis in axes.flat:
        axis.set_xlim(0, stages["roi"].shape[1])
        axis.set_ylim(stages["roi"].shape[0], 0)
        axis.axis("off")

    fig.suptitle("Etapas de detección WCS en la captura real_20", fontsize=10.5)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.89, bottom=0.08, hspace=0.30, wspace=0.13)
    fig.text(
        0.5,
        0.01,
        "CLAHE → umbral adaptativo → HoughP → filtros de longitud, orientación, ortogonalidad y proximidad",
        ha="center",
        fontsize=8.2,
        color="#555555",
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "wcs_detection_stages_v8.png", dpi=300, facecolor="white")
    fig.savefig(FIGURES / "wcs_detection_stages_v8.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    rows = {row["sample_id"]: row for row in load_manifest() if row["domain"] == "real"}
    row = rows[SAMPLE_ID]
    raw_path = ROOT / row["raw_image"]
    raw = load_image(str(raw_path), 0)
    markers = detect_markers(raw, None, SAMPLE_ID)
    if markers is None or len(markers) != 4:
        raise RuntimeError("No se obtuvieron cuatro marcadores para real_20.")
    rectified, _ = rectify_sheet(
        raw,
        markers,
        None,
        SAMPLE_ID,
        sheet_size=(210.0, 297.0),
        scale=SCALE,
        marker_margin=MARKER_MARGIN_MM,
    )
    stages = preprocess_like_detector(rectified)
    wcs = detect_wcs_l(
        rectified,
        debug_dir=None,
        img_name=SAMPLE_ID,
        marker_margin=MARKER_MARGIN_MM,
        scale=SCALE,
    )
    if wcs["status"] != "SUCCESS":
        raise RuntimeError(f"El detector actual no aceptó real_20: {wcs['status']}")
    make_figure(stages, wcs)

    detector_path = SOFTWARE_SRC / "detect_wcs_l.py"
    payload = {
        "sample_id": SAMPLE_ID,
        "source_raw_image": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
        "roi_mm": [0.0, 0.0, ROI_SIZE_MM, ROI_SIZE_MM],
        "scale_px_per_mm": SCALE,
        "preprocessing": {
            "clahe_clip_limit": 3.0,
            "clahe_grid": [8, 8],
            "adaptive_threshold_block_size": 51,
            "adaptive_threshold_c": 15,
            "hough_threshold": 40,
            "hough_min_line_length_px": 30,
            "hough_max_line_gap_px": 10,
        },
        "hough_segments_total": stages["hough_line_count"],
        "horizontal_segments_after_length_and_angle_filters": len(stages["horizontal"]),
        "vertical_segments_after_length_and_angle_filters": len(stages["vertical"]),
        "segments_shown_per_orientation_max": 8,
        "detector_result": wcs,
        "figure": "resultados/figuras_protocolo_v8/wcs_detection_stages_v8.png",
        "provenance": {
            "raw_image_sha256": sha256_file(raw_path),
            "detect_wcs_l_sha256": sha256_file(detector_path),
            "detect_markers_sha256": sha256_file(SOFTWARE_SRC / "detect_markers.py"),
            "rectify_sheet_sha256": sha256_file(SOFTWARE_SRC / "rectify_sheet.py"),
        },
        "interpretation_limit": (
            "The figure visualizes the current geometric detector and does not constitute "
            "independent metrological validation."
        ),
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "wcs_detection_stages_v8.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
