"""Genera figuras técnicas a partir de las evidencias del protocolo experimental."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "resultados" / "metricas"
FIGURES = ROOT / "resultados" / "figuras_protocolo_v8"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def prompt_source_figure() -> None:
    rows = read_csv(METRICS / "sam2_idv2_locked_scenario_summary.csv")
    key_order = [
        ("test", "IDV2 prueba\n12 láminas"),
        ("real", "Real\n13 capturas, una lámina"),
    ]
    scenarios = [("ideal", "Cajas ideales"), ("operational", "Cajas operativas")]
    colors = {"ideal": "#86AED0", "operational": "#2F6B9A"}
    by_key = {
        (row["split"], row["prompt_scenario"]): row for row in rows
    }
    x = np.arange(len(key_order), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.3, 5.4), constrained_layout=True)
    for scenario_index, (scenario, label) in enumerate(scenarios):
        values = []
        errors_low = []
        errors_high = []
        for split, _ in key_order:
            row = by_key[(split, scenario)]
            value = float(row["mean_iou"])
            values.append(value)
            errors_low.append(value - float(row["iou_bootstrap_95_ci_low"]))
            errors_high.append(float(row["iou_bootstrap_95_ci_high"]) - value)
        positions = x + (scenario_index - 0.5) * width
        bars = ax.bar(
            positions,
            values,
            width=width,
            color=colors[scenario],
            edgecolor="#24445C",
            linewidth=0.8,
            label=label,
            yerr=np.asarray([errors_low, errors_high]),
            capsize=3,
        )
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)
    fig.suptitle("Rendimiento de SAM 2 según el origen de las cajas", fontsize=13)
    ax.set_title(
        "IoU media e intervalo bootstrap descriptivo del 95 % por captura",
        fontsize=9,
        color="#555555",
        pad=9,
    )
    ax.set_xticks(x, [label for _, label in key_order])
    ax.set_ylabel("IoU")
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False, loc="lower left")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "sam2_prompt_source_comparison_v8.png", dpi=220, facecolor="white")
    plt.close(fig)


def runtime_figure() -> None:
    rows = read_csv(METRICS / "segmentation_runtime_runs_v8.csv")
    methods = [
        "vision_clasica",
        "random_forest_identity_v2",
        "sam2_operativo_hibrido",
    ]
    labels = ["Visión clásica", "Random Forest", "SAM 2 operativo"]
    colors = ["#2F6B9A", "#D5822A", "#7A8B3A"]
    values = [
        [float(row["total_seconds"]) for row in rows if row["method"] == method]
        for method in methods
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    boxes = ax.boxplot(values, patch_artist=True, tick_labels=labels, showfliers=True)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.30)
        patch.set_edgecolor(color)
    for median in boxes["medians"]:
        median.set_color("#262626")
        median.set_linewidth(1.6)
    fig.suptitle("Tiempo de segmentación por método", fontsize=13)
    ax.set_title(
        "Cuatro capturas × cinco repeticiones. Imágenes de 2100 × 2970 px en el mismo equipo",
        fontsize=9,
        color="#555555",
        pad=9,
    )
    ax.set_ylabel("Tiempo (s)")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "segmentation_runtime_comparison_v8.png", dpi=220, facecolor="white")
    plt.close(fig)


def sam_stage_figure() -> None:
    rows = [
        row
        for row in read_csv(METRICS / "segmentation_runtime_runs_v8.csv")
        if row["method"] == "sam2_operativo_hibrido"
    ]
    sample_ids = sorted({row["sample_id"] for row in rows}, key=lambda value: int(value.split("_")[1]))
    stages = [
        ("prompt_localization_seconds", "Localización", "#B9CFE1"),
        ("encoder_seconds", "Codificador", "#6F9DC1"),
        ("decoder_seconds", "Decodificador", "#2F6B9A"),
        ("postprocess_seconds", "Postproceso", "#173A53"),
    ]
    x = np.arange(len(sample_ids))
    bottom = np.zeros(len(sample_ids), dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    for field, label, color in stages:
        values = np.asarray(
            [
                np.mean([float(row[field]) for row in rows if row["sample_id"] == sample_id])
                for sample_id in sample_ids
            ]
        )
        ax.bar(x, values, bottom=bottom, color=color, edgecolor="#24445C", linewidth=0.6, label=label)
        bottom += values
    fig.suptitle("Descomposición del tiempo de SAM 2 operativo", fontsize=13)
    ax.set_title(
        "Media de cinco repeticiones por captura. El total incluye set_image y codificador",
        fontsize=9,
        color="#555555",
        pad=9,
    )
    ax.set_xticks(x, sample_ids)
    ax.set_ylabel("Tiempo (s)")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "sam2_runtime_stages_v8.png", dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    prompt_source_figure()
    runtime_figure()
    sam_stage_figure()
    print(f"Figuras generadas en {FIGURES}")


if __name__ == "__main__":
    main()
