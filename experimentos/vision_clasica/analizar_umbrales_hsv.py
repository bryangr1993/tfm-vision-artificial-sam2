"""Genera evidencia descriptiva para los umbrales HSV de la línea base.

El análisis utiliza únicamente el subconjunto sintético de validación del
protocolo ``asset_identity_disjoint``. Las máscaras proceden del canal alfa del
generador, por lo que la figura no depende de la predicción clásica. El script
no optimiza los umbrales. Describe el comportamiento de los valores históricos
S > 45 y V < 115 sobre datos que no intervienen en la prueba final.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "datos" / "manifiesto" / "datasets_asset_identity_v2.csv"
OUTPUT_DIR = ROOT / "resultados" / "figuras"
METRICS_DIR = ROOT / "resultados" / "metricas"
SATURATION_THRESHOLD = 45
VALUE_THRESHOLD = 115


def read_validation_rows() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["domain"] == "synthetic" and row["split"] == "val"
    ]
    if not selected:
        raise RuntimeError("El manifiesto no contiene láminas sintéticas de validación.")
    return selected


def relative_histogram(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bin_edges)
    return counts / max(int(counts.sum()), 1)


def main() -> None:
    rows = read_validation_rows()
    foreground_s: list[np.ndarray] = []
    background_s: list[np.ndarray] = []
    foreground_v: list[np.ndarray] = []
    background_v: list[np.ndarray] = []
    sheet_rates: list[dict[str, float | str]] = []

    for row in rows:
        image = cv2.imread(str(ROOT / row["image"]), cv2.IMREAD_COLOR)
        ground_truth = cv2.imread(
            str(ROOT / row["ground_truth"]), cv2.IMREAD_GRAYSCALE
        )
        if image is None or ground_truth is None:
            raise FileNotFoundError(row["sample_id"])
        foreground = ground_truth > 0
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        foreground_s.append(saturation[foreground])
        background_s.append(saturation[~foreground])
        foreground_v.append(value[foreground])
        background_v.append(value[~foreground])

        color_cue = saturation > SATURATION_THRESHOLD
        dark_cue = value < VALUE_THRESHOLD
        union = color_cue | dark_cue
        sheet_rates.append(
            {
                "sample_id": row["sample_id"],
                "foreground_recall_s": float(color_cue[foreground].mean()),
                "background_rate_s": float(color_cue[~foreground].mean()),
                "foreground_recall_v": float(dark_cue[foreground].mean()),
                "background_rate_v": float(dark_cue[~foreground].mean()),
                "foreground_recall_union": float(union[foreground].mean()),
                "background_rate_union": float(union[~foreground].mean()),
            }
        )

    fg_s = np.concatenate(foreground_s)
    bg_s = np.concatenate(background_s)
    fg_v = np.concatenate(foreground_v)
    bg_v = np.concatenate(background_v)
    bins = np.arange(0, 260, 4)
    centers = (bins[:-1] + bins[1:]) / 2

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
            # Los exponentes de la escala logarítmica se componen como
            # superíndices; 11,5 pt mantiene esos glifos por encima de 8 pt.
            "ytick.labelsize": 11.5,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(5.0, 3.8))
    specifications = [
        (
            axes[0],
            relative_histogram(fg_s, bins),
            relative_histogram(bg_s, bins),
            SATURATION_THRESHOLD,
            "Saturación S",
            r"Regla: $S>45$",
        ),
        (
            axes[1],
            relative_histogram(fg_v, bins),
            relative_histogram(bg_v, bins),
            VALUE_THRESHOLD,
            "Valor V",
            r"Regla: $V<115$",
        ),
    ]
    for axis, foreground_hist, background_hist, threshold, xlabel, rule in specifications:
        axis.step(
            centers,
            np.maximum(background_hist, 1e-7),
            where="mid",
            color="#526777",
            linewidth=1.7,
            label="Fondo",
        )
        axis.step(
            centers,
            np.maximum(foreground_hist, 1e-7),
            where="mid",
            color="#E8873A",
            linewidth=2.0,
            label="Topper",
        )
        axis.axvline(threshold, color="#2F80C1", linewidth=2.0, linestyle="--")
        axis.text(
            threshold + 5,
            0.23,
            rule,
            color="#15324B",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#E7F2F8", "edgecolor": "#2F80C1"},
        )
        axis.set_yscale("log")
        axis.set_xlim(0, 255)
        axis.set_ylim(1e-7, 0.5)
        axis.set_xlabel(xlabel)
        axis.grid(axis="y", color="#D9E0E5", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Frecuencia relativa (escala logarítmica)")
    axes[0].legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.02, 0.02),
        borderaxespad=0.0,
    )

    recall_union = np.array([row["foreground_recall_union"] for row in sheet_rates])
    background_union = np.array([row["background_rate_union"] for row in sheet_rates])
    figure.suptitle(
        "Distribución de las señales HSV\nen el conjunto sintético de validación",
        color="#15324B",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.035,
        (
            f"Unión de ambas reglas: recuperación media del objeto "
            f"{recall_union.mean() * 100:.1f}%\n"
            f"Admisión media del fondo {background_union.mean() * 100:.3f}% "
            f"(12 láminas)."
        ),
        ha="center",
        color="#15324B",
        fontsize=9,
    )

    figure.subplots_adjust(left=0.14, right=0.98, top=0.74, bottom=0.25, wspace=0.34)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / "evidencia_umbrales_hsv.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / "evidencia_umbrales_hsv.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "protocol": "asset_identity_disjoint",
        "domain": "synthetic",
        "split": "val",
        "sheets": len(rows),
        "thresholds": {
            "saturation_strictly_greater_than": SATURATION_THRESHOLD,
            "value_strictly_less_than": VALUE_THRESHOLD,
        },
        "interpretation": "descriptive evidence; thresholds were not optimized by this script",
        "foreground_recall_union_mean": float(recall_union.mean()),
        "foreground_recall_union_std": float(recall_union.std(ddof=1)),
        "background_rate_union_mean": float(background_union.mean()),
        "background_rate_union_std": float(background_union.std(ddof=1)),
        "per_sheet": sheet_rates,
    }
    (METRICS_DIR / "hsv_threshold_evidence.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "per_sheet"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
