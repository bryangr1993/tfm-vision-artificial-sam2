"""Evalúa por separado el registro WCS y la estabilidad de los contornos.

El registro se recalcula desde las capturas sin rectificar. Su dispersión se
expresa en el plano rectificado, sin presentarla como una calibración física
independiente. La estabilidad de contorno se calcula entre pares de capturas de
la misma lámina y se informa por método, sin mezclarla con el origen WCS.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SAM_DIR = ROOT / "experimentos" / "sam2"
METRICS = ROOT / "resultados" / "metricas"
OUTPUT = ROOT / "resultados" / "wcs_repeatability_v8"
FIGURES = ROOT / "resultados" / "figuras_protocolo_v8"

sys.path.insert(0, str(SAM_DIR))
sys.path.insert(0, str(ROOT / "software" / "src"))
from detect_markers import detect_markers
from detect_wcs_l import detect_wcs_l
from load_images import load_image
from rectify_sheet import rectify_sheet
from protocol_common import boundary_f1, component_count, dice_iou, load_manifest, write_csv


EXPECTED_WCS_IDS = {f"real_{number}" for number in range(20, 40, 2)}
EXPECTED_NO_WCS_IDS = {"real_13", "real_15", "real_17"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_number(sample_id: str) -> int:
    return int(sample_id.rsplit("_", 1)[-1])


def prediction_paths(sample_id: str) -> dict[str, Path]:
    return {
        "Visión clásica": ROOT / "resultados" / "real_predictions" / "clasico" / f"{sample_id}_clasico.png",
        "Random Forest IDV2": ROOT
        / "resultados"
        / "rf_idv2_real_agreement_masks"
        / f"{sample_id}_rf_idv2.png",
        "SAM 2 operativo": ROOT / "resultados" / "sam2_identity_v2" / "masks" / "real" / "operational" / f"{sample_id}_sam2.png",
    }


def recompute_registration(rows: list[dict[str, str]]) -> list[dict]:
    rectified_dir = OUTPUT / "rectified_recomputed"
    rectified_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        expected = sample_id in EXPECTED_WCS_IDS
        raw_path = ROOT / row["raw_image"]
        image = load_image(str(raw_path), 0)
        markers = detect_markers(image, None, sample_id)
        record = {
            "sample_id": sample_id,
            "expected_wcs": expected,
            "markers_detected": 0 if markers is None else len(markers),
            "wcs_status": "NOT_RUN",
            "classification_correct": False,
            "origin_x_px": "",
            "origin_y_px": "",
            "origin_x_mm_from_sheet": "",
            "origin_y_mm_from_sheet": "",
            "axis_x_angle_deg": "",
            "orthogonality_error_deg": "",
            "origin_deviation_from_mean_mm": "",
        }
        if markers is None or len(markers) != 4:
            record["wcs_status"] = "MARKERS_NOT_FOUND"
            results.append(record)
            continue
        rectified, _ = rectify_sheet(
            image,
            markers,
            None,
            sample_id,
            sheet_size=(210.0, 297.0),
            scale=10.0,
            marker_margin=10.0,
        )
        cv2.imwrite(str(rectified_dir / f"{sample_id}.png"), rectified)
        wcs = detect_wcs_l(
            rectified,
            debug_dir=None,
            img_name=sample_id,
            marker_margin=10.0,
            scale=10.0,
        )
        detected = wcs["status"] == "SUCCESS"
        record["wcs_status"] = wcs["status"]
        record["classification_correct"] = detected == expected
        if detected:
            origin = np.asarray(wcs["origin"], dtype=float)
            ux = np.asarray(wcs["uX"], dtype=float)
            uy = np.asarray(wcs["uY"], dtype=float)
            angle_x = math.degrees(math.atan2(ux[1], ux[0]))
            angle_between = math.degrees(
                math.acos(float(np.clip(np.dot(ux, uy), -1.0, 1.0)))
            )
            record.update(
                {
                    "origin_x_px": float(origin[0]),
                    "origin_y_px": float(origin[1]),
                    "origin_x_mm_from_sheet": float(origin[0] / 10.0),
                    "origin_y_mm_from_sheet": float(origin[1] / 10.0),
                    "axis_x_angle_deg": angle_x,
                    "orthogonality_error_deg": abs(90.0 - angle_between),
                }
            )
        results.append(record)
        print(f"WCS [{index}/{len(rows)}] {sample_id}: {record['wcs_status']}")

    valid = [row for row in results if row["expected_wcs"] and row["wcs_status"] == "SUCCESS"]
    if valid:
        mean_x = float(np.mean([float(row["origin_x_px"]) for row in valid]))
        mean_y = float(np.mean([float(row["origin_y_px"]) for row in valid]))
        for row in valid:
            dx = float(row["origin_x_px"]) - mean_x
            dy = float(row["origin_y_px"]) - mean_y
            row["origin_deviation_from_mean_mm"] = float(math.hypot(dx, dy) / 10.0)
    return results


def contour_stability(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    wcs_ids = [row["sample_id"] for row in rows if row["sample_id"] in EXPECTED_WCS_IDS]
    capture_rows: list[dict] = []
    pairwise_rows: list[dict] = []
    for method in ["Visión clásica", "Random Forest IDV2", "SAM 2 operativo"]:
        masks: dict[str, np.ndarray] = {}
        for sample_id in wcs_ids:
            path = prediction_paths(sample_id)[method]
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(
                    f"Falta {path}. Ejecute primero evaluate_sam2_protocol.py para SAM 2."
                )
            masks[sample_id] = np.where(mask > 0, 255, 0).astype(np.uint8)
            moments = cv2.moments(masks[sample_id], binaryImage=True)
            cx = moments["m10"] / moments["m00"] if moments["m00"] else math.nan
            cy = moments["m01"] / moments["m00"] if moments["m00"] else math.nan
            capture_rows.append(
                {
                    "method": method,
                    "sample_id": sample_id,
                    "foreground_area_px": int(np.count_nonzero(masks[sample_id])),
                    "foreground_fraction": float(np.count_nonzero(masks[sample_id]) / masks[sample_id].size),
                    "mask_centroid_x_px": float(cx),
                    "mask_centroid_y_px": float(cy),
                    "components_ge_20000_px": component_count(masks[sample_id]),
                }
            )
        for left, right in combinations(wcs_ids, 2):
            dice, iou = dice_iou(masks[left], masks[right])
            pairwise_rows.append(
                {
                    "method": method,
                    "sample_a": left,
                    "sample_b": right,
                    "dice": dice,
                    "iou": iou,
                    "boundary_f1_tolerance_3px": boundary_f1(masks[left], masks[right]),
                    "absolute_area_difference_px": abs(
                        int(np.count_nonzero(masks[left])) - int(np.count_nonzero(masks[right]))
                    ),
                }
            )
    return capture_rows, pairwise_rows


def summarize(registration: list[dict], captures: list[dict], pairwise: list[dict]) -> dict:
    expected_positive = [row for row in registration if row["expected_wcs"]]
    expected_negative = [row for row in registration if not row["expected_wcs"]]
    valid = [row for row in expected_positive if row["wcs_status"] == "SUCCESS"]
    origin_deviations = [float(row["origin_deviation_from_mean_mm"]) for row in valid]
    angles = [float(row["axis_x_angle_deg"]) for row in valid]
    contour_summary: list[dict] = []
    for method in sorted({row["method"] for row in pairwise}):
        method_pairs = [row for row in pairwise if row["method"] == method]
        method_captures = [row for row in captures if row["method"] == method]
        contour_summary.append(
            {
                "method": method,
                "n_captures": len(method_captures),
                "n_capture_pairs": len(method_pairs),
                "mean_pairwise_iou": float(np.mean([row["iou"] for row in method_pairs])),
                "std_pairwise_iou": float(np.std([row["iou"] for row in method_pairs], ddof=1)),
                "mean_pairwise_boundary_f1": float(
                    np.mean([row["boundary_f1_tolerance_3px"] for row in method_pairs])
                ),
                "std_foreground_area_px": float(
                    np.std([row["foreground_area_px"] for row in method_captures], ddof=1)
                ),
                "mean_foreground_fraction": float(
                    np.mean([row["foreground_fraction"] for row in method_captures])
                ),
                "fraction_captures_with_8_components": float(
                    np.mean([row["components_ge_20000_px"] == 8 for row in method_captures])
                ),
                "all_captures_have_8_components": all(
                    row["components_ge_20000_px"] == 8 for row in method_captures
                ),
            }
        )
    return {
        "protocol_version": "wcs_and_contour_repeatability_v8",
        "registration": {
            "expected_wcs_captures": len(expected_positive),
            "detected_expected_wcs": len(valid),
            "expected_no_wcs_captures": len(expected_negative),
            "correctly_rejected_no_wcs": sum(
                row["wcs_status"] != "SUCCESS" for row in expected_negative
            ),
            "all_classifications_correct": all(row["classification_correct"] for row in registration),
            "mean_origin_x_mm_from_sheet": float(
                np.mean([float(row["origin_x_mm_from_sheet"]) for row in valid])
            ) if valid else math.nan,
            "mean_origin_y_mm_from_sheet": float(
                np.mean([float(row["origin_y_mm_from_sheet"]) for row in valid])
            ) if valid else math.nan,
            "origin_radial_deviation_mean_mm": float(np.mean(origin_deviations)) if valid else math.nan,
            "origin_radial_deviation_max_mm": float(np.max(origin_deviations)) if valid else math.nan,
            "axis_x_angle_mean_deg": float(np.mean(angles)) if valid else math.nan,
            "axis_x_angle_std_deg": float(np.std(angles, ddof=1)) if len(angles) > 1 else 0.0,
            "interpretation": (
                "Capture-to-capture geometric dispersion after recalculating markers and homography. "
                "No independent metrological ground truth was available."
            ),
        },
        "contour_stability": contour_summary,
        "separation_note": (
            "WCS origin/axis dispersion and segmentation contour stability are reported as "
            "different phenomena; neither is used as a proxy for the other."
        ),
        "provenance": {
            "manifest_sha256": sha256_file(ROOT / "datos" / "manifiesto" / "datasets.csv"),
            "detect_markers_sha256": sha256_file(ROOT / "software" / "src" / "detect_markers.py"),
            "rectify_sheet_sha256": sha256_file(ROOT / "software" / "src" / "rectify_sheet.py"),
            "detect_wcs_l_sha256": sha256_file(ROOT / "software" / "src" / "detect_wcs_l.py"),
            "sam2_selection_sha256": sha256_file(
                ROOT / "resultados" / "metricas" / "sam2_idv2_prompt_selection.json"
            ),
            "rf_model_sha256": sha256_file(
                ROOT / "resultados" / "modelos" / "random_forest_identity_v2_selected.joblib"
            ),
            "rf_selection_sha256": sha256_file(
                ROOT / "resultados" / "metricas" / "rf_idv2_selection_locked.json"
            ),
        },
    }


def plot_registration(registration: list[dict]) -> None:
    valid = [row for row in registration if row["expected_wcs"] and row["wcs_status"] == "SUCCESS"]
    x = np.asarray([float(row["origin_x_mm_from_sheet"]) for row in valid])
    y = np.asarray([float(row["origin_y_mm_from_sheet"]) for row in valid])
    mean_x, mean_y = float(x.mean()), float(y.mean())
    fig, ax = plt.subplots(figsize=(4.75, 3.8), constrained_layout=True)
    ax.scatter(x, y, s=58, color="#2F6B9A", edgecolor="#173A53", linewidth=0.8)
    for row, xv, yv in zip(valid, x, y):
        ax.annotate(row["sample_id"].replace("real_", ""), (xv, yv), xytext=(4, 4), textcoords="offset points", fontsize=8.5)
    ax.axvline(mean_x, color="#373737", linestyle="--", linewidth=1.0)
    ax.axhline(mean_y, color="#373737", linestyle="--", linewidth=1.0)
    fig.suptitle("Origen WCS detectado en diez capturas con marca", fontsize=10.5)
    ax.set_title(
        "Coordenadas desde la esquina de la hoja rectificada a 10 px/mm",
        fontsize=8.5,
        color="#555555",
        pad=9,
    )
    ax.set_xlabel("Origen X (mm)")
    ax.set_ylabel("Origen Y (mm)")
    ax.grid(True, color="#D9D9D9", linewidth=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "wcs_origin_repeatability_v8.png", dpi=300, facecolor="white")
    fig.savefig(FIGURES / "wcs_origin_repeatability_v8.pdf", facecolor="white")
    plt.close(fig)


def plot_contour_stability(pairwise: list[dict]) -> None:
    methods = ["Visión clásica", "Random Forest IDV2", "SAM 2 operativo"]
    colors = ["#2F6B9A", "#D5822A", "#7A8B3A"]
    data = [[row["iou"] for row in pairwise if row["method"] == method] for method in methods]
    capture_csv = METRICS / "contour_stability_per_capture_v8.csv"
    with capture_csv.open("r", encoding="utf-8", newline="") as stream:
        capture_rows = list(csv.DictReader(stream))
    fractions = [
        np.mean(
            [
                int(row["components_ge_20000_px"]) == 8
                for row in capture_rows
                if row["method"] == method
            ]
        )
        for method in methods
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4), constrained_layout=True)
    boxes = axes[0].boxplot(data, patch_artist=True, tick_labels=methods, showfliers=True)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
    for median in boxes["medians"]:
        median.set_color("#262626")
        median.set_linewidth(1.6)
    axes[0].set_title(
        "IoU entre pares de capturas\n45 pares por método sobre una lámina rectificada",
        fontsize=10,
        pad=9,
    )
    axes[0].set_ylabel("IoU entre capturas")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    bars = axes[1].bar(methods, fractions, color=colors, edgecolor="#24445C", linewidth=0.8)
    axes[1].bar_label(bars, labels=[f"{100 * value:.0f} %" for value in fractions], padding=3)
    axes[1].set_title(
        "Capturas con ocho componentes válidos\nDiez capturas con WCS por método",
        fontsize=10,
        pad=9,
    )
    axes[1].set_ylabel("Proporción de capturas")
    axes[1].set_ylim(0.0, 1.08)
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    fig.suptitle("Estabilidad de segmentación en una misma lámina", fontsize=13)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "contour_pairwise_stability_v8.png", dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-recompute", action="store_true")
    args = parser.parse_args()
    rows = sorted(
        [row for row in load_manifest() if row["domain"] == "real"],
        key=lambda row: sample_number(row["sample_id"]),
    )
    METRICS.mkdir(parents=True, exist_ok=True)
    registration_csv = METRICS / "wcs_registration_per_capture_v8.csv"
    if args.skip_recompute:
        with registration_csv.open("r", encoding="utf-8", newline="") as stream:
            registration = list(csv.DictReader(stream))
        for row in registration:
            row["expected_wcs"] = row["expected_wcs"].lower() == "true"
            row["classification_correct"] = row["classification_correct"].lower() == "true"
    else:
        registration = recompute_registration(rows)
        write_csv(registration_csv, registration)
    captures, pairwise = contour_stability(rows)
    write_csv(METRICS / "contour_stability_per_capture_v8.csv", captures)
    write_csv(METRICS / "contour_pairwise_stability_v8.csv", pairwise)
    payload = summarize(registration, captures, pairwise)
    (METRICS / "wcs_contour_repeatability_summary_v8.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_registration(registration)
    plot_contour_stability(pairwise)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
