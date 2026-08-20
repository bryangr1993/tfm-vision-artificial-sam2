"""Genera figuras de resultados directamente desde los artefactos validados."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METRICS = ROOT / "resultados" / "metricas"
FIGURES = ROOT / "memoria" / "figuras"
REAL_PRED = ROOT / "resultados" / "real_predictions"

COLORS = {
    "Clásico": "#24506F",
    "Random Forest": "#D97706",
    "SAM 2": "#6B7D3E",
}
HATCHES = {"Clásico": "//", "Random Forest": "..", "SAM 2": "xx"}


def read_json(name: str):
    return json.loads((METRICS / name).read_text(encoding="utf-8"))


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D7DCE0", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#263238")


def save(fig, name: str):
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def comparison_data():
    classical = read_json("classical_test_summary.json")
    rf = read_json("rf_v1_control_selection.json")
    sam = read_json("sam2_final_selection.json")
    real_pair = read_json("classical_rf_real_summary.json")
    sam_real = read_json("sam2_real_summary.json")
    synthetic = {
        "Clásico": {
            "iou": classical["test_mean_iou"],
            "std_iou": classical["test_std_iou"],
            "dice": classical["test_mean_dice"],
            "boundary_f1": classical["test_mean_boundary_f1"],
        },
        "Random Forest": {
            "iou": rf["test_mean_iou"],
            "std_iou": rf["test_std_iou"],
            "dice": rf["test_mean_dice"],
            "boundary_f1": rf["test_mean_boundary_f1"],
        },
        "SAM 2": {
            "iou": sam["test_mean_iou"],
            "std_iou": sam["test_std_iou"],
            "dice": sam["test_mean_dice"],
            "boundary_f1": sam["test_mean_boundary_f1"],
        },
    }
    real = {
        "Clásico": {
            "iou": real_pair["classical"]["mean_iou"],
            "std_iou": real_pair["classical"]["std_iou"],
        },
        "Random Forest": {
            "iou": real_pair["random_forest"]["mean_iou"],
            "std_iou": real_pair["random_forest"]["std_iou"],
        },
        "SAM 2": {"iou": sam_real["mean_iou"], "std_iou": sam_real["std_iou"]},
    }
    return synthetic, real


def figure_domain_gap(synthetic, real):
    methods = list(synthetic)
    x = np.arange(2)
    width = 0.22
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    for index, method in enumerate(methods):
        values = [synthetic[method]["iou"], real[method]["iou"]]
        errors = [synthetic[method]["std_iou"], real[method]["std_iou"]]
        positions = x + (index - 1) * width
        bars = ax.bar(
            positions,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=method,
            color=COLORS[method],
            edgecolor="#263238",
            linewidth=0.7,
            hatch=HATCHES[method],
        )
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4, fontsize=9)
    fig.suptitle(
        "IoU por método y dominio", x=0.10, y=0.98, ha="left", fontsize=14, weight="bold"
    )
    fig.text(
        0.10,
        0.91,
        "Media por lámina ± desviación estándar; sintético n=6, real n=13 capturas de una hoja",
        fontsize=9,
        color="#4F5B62",
    )
    ax.set_xticks(x, ["Prueba sintética", "Capturas reales"])
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style_axes(ax)
    fig.subplots_adjust(top=0.83, bottom=0.22)
    save(fig, "fig_brecha_dominios")


def figure_synthetic_metrics(synthetic):
    methods = list(synthetic)
    metrics = [("iou", "IoU"), ("dice", "Dice"), ("boundary_f1", "Boundary F1")]
    x = np.arange(len(metrics))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    for index, method in enumerate(methods):
        values = [synthetic[method][key] for key, _ in metrics]
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=method,
            color=COLORS[method],
            edgecolor="#263238",
            linewidth=0.7,
            hatch=HATCHES[method],
        )
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)
    fig.suptitle(
        "Métricas en el conjunto sintético de prueba",
        x=0.10,
        y=0.98,
        ha="left",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.10,
        0.91,
        "Media por lámina; seis muestras bloqueadas hasta cerrar modelo, umbral y prompts",
        fontsize=9,
        color="#4F5B62",
    )
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("Puntuación")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style_axes(ax)
    fig.subplots_adjust(top=0.83, bottom=0.22)
    save(fig, "fig_metricas_sinteticas")


def figure_prompt_selection():
    with (METRICS / "sam2_validation_prompt_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    labels = {
        "caja": "Caja",
        "caja_margen_5": "Caja + 5%",
        "caja_margen_10": "Caja + 10%",
        "caja_punto_positivo": "Caja + punto positivo",
        "caja_puntos_positivos_negativos": "Caja + puntos positivos/negativos",
    }
    rows.sort(key=lambda row: float(row["mean_validation_iou"]))
    y = np.arange(len(rows))
    values = [float(row["mean_validation_iou"]) for row in rows]
    colors = ["#6B7D3E" if row["strategy"] == "caja_margen_5" else "#AAB2B7" for row in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.hlines(y, 0.86, values, color=colors, linewidth=3)
    ax.scatter(values, y, s=80, color=colors, edgecolor="#263238", linewidth=0.7, zorder=3)
    for value, ypos in zip(values, y):
        ax.text(value + 0.0008, ypos, f"{value:.4f}", va="center", fontsize=9)
    fig.suptitle(
        "IoU de SAM 2 por estrategia de prompt",
        x=0.28,
        y=0.98,
        ha="left",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.28,
        0.91,
        "Validación sintética, n=6 por estrategia; eje enfocado para mostrar diferencias pequeñas",
        fontsize=9,
        color="#4F5B62",
    )
    ax.set_yticks(y, [labels[row["strategy"]] for row in rows])
    ax.set_xlabel("IoU medio")
    ax.set_xlim(0.86, 0.92)
    ax.grid(axis="x", color="#D7DCE0", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(top=0.83, left=0.28)
    save(fig, "fig_sam_prompts_validacion")


def rgb(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def gray(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def figure_real20_pipeline():
    source = ROOT / "resultados" / "integracion_real20"
    panels = [
        (rgb(ROOT / "datos" / "reales" / "rectificadas" / "rectified_20.png"), "Hoja rectificada", None),
        (rgb(source / "real_20_sam2_prompts.png"), "Cajas para SAM 2", None),
        (gray(source / "real_20_classical_prompt_mask.png"), "Localización aproximada", "gray"),
        (gray(source / "real_20_sam2_mask.png"), "Máscara final de SAM 2", "gray"),
        (rgb(source / "real_20_contours.png"), "Contornos extraídos", None),
        (rgb(source / "real_20_wcs_axes.png"), "Registro WCS", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 11.8))
    for ax, (image, title, cmap) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle("Flujo operativo de IA en la captura real 20", fontsize=15, weight="bold")
    fig.subplots_adjust(top=0.91, bottom=0.03, hspace=0.16, wspace=0.06)
    save(fig, "real_20_pipeline_sam2")


def figure_real13_comparison():
    sample = "real_13"
    original = rgb(ROOT / "datos" / "reales" / "rectificadas" / "rectified_13.png")
    reference = gray(ROOT / "datos" / "anotaciones" / "real_ground_truth" / "real_13_gt.png")
    classical = gray(REAL_PRED / "clasico" / f"{sample}_clasico.png")
    rf = gray(REAL_PRED / "random_forest" / f"{sample}_rf_v1.png")
    sam = gray(REAL_PRED / "sam2" / f"{sample}_sam2.png")
    overlay = original.copy()
    for mask, color in [(classical, (36, 80, 111)), (rf, (217, 119, 6)), (sam, (107, 125, 62))]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, contours, -1, color, 4)
    panels = [
        (original, "Captura rectificada", None),
        (reference, "Referencia canónica", "gray"),
        (classical, "Visión clásica", "gray"),
        (rf, "Random Forest", "gray"),
        (sam, "SAM 2", "gray"),
        (overlay, "Contornos superpuestos", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 11.8))
    for ax, (image, title, cmap) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle("Comparación cualitativa en la captura real 13", fontsize=15, weight="bold")
    fig.text(
        0.5,
        0.015,
        "Superposición: clásico azul, Random Forest naranja y SAM 2 oliva",
        ha="center",
        fontsize=9,
        color="#4F5B62",
    )
    fig.subplots_adjust(top=0.91, bottom=0.06, hspace=0.16, wspace=0.06)
    save(fig, "real_13_comparison")


def figure_real_samples():
    """Presenta pocas capturas a tamaño legible sin ocultar el tamaño real del conjunto."""
    identifiers = [13, 17, 20, 24, 30, 38]
    notes = {
        13: "sin WCS",
        17: "sombra inferior",
        20: "WCS visible",
        24: "sombra inferior y cabezal",
        30: "recolocación y sombra",
        38: "MDF desplazado",
    }
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 8.1))
    for ax, identifier in zip(axes.flat, identifiers):
        image = rgb(
            ROOT
            / "datos"
            / "reales"
            / "rectificadas"
            / f"rectified_{identifier}.png"
        )
        ax.imshow(image)
        ax.set_title(f"Captura {identifier} · {notes[identifier]}", fontsize=10)
        ax.axis("off")
    fig.suptitle(
        "Muestras representativas del conjunto real",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.5,
        0.025,
        "Se muestran 6 de las 13 capturas; todas corresponden a una misma hoja física rectificada.",
        ha="center",
        fontsize=9,
        color="#4F5B62",
    )
    fig.subplots_adjust(top=0.91, bottom=0.07, hspace=0.14, wspace=0.08)
    save(fig, "muestra_6_capturas_reales")


def figure_synthetic_samples():
    """Resume las cuatro familias con ocho hojas que siguen siendo reconocibles."""
    identifiers = [
        "F1_L1_C1",
        "F2_L1_C2",
        "F3_L1_C3",
        "F4_L1_C4",
        "F1_L2_C5",
        "F2_L2_C6",
        "F3_L2_C1",
        "F4_L2_C2",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 7.4))
    for ax, identifier in zip(axes.flat, identifiers):
        image = rgb(
            ROOT
            / "datos"
            / "sinteticos"
            / "imagenes"
            / f"sheet_{identifier}_rgb.png"
        )
        ax.imshow(image)
        ax.set_title(identifier.replace("_", " · "), fontsize=9)
        ax.axis("off")
    fig.suptitle(
        "Ejemplos de familias y condiciones sintéticas",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.5,
        0.025,
        "Se muestran 8 de las 48 hojas; F identifica la familia, L la distribución y C la condición visual.",
        ha="center",
        fontsize=9,
        color="#4F5B62",
    )
    fig.subplots_adjust(top=0.91, bottom=0.07, hspace=0.14, wspace=0.07)
    save(fig, "muestra_8_hojas_sinteticas")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#263238",
            "text.color": "#263238",
            "pdf.fonttype": 42,
        }
    )
    synthetic, real = comparison_data()
    figure_domain_gap(synthetic, real)
    figure_synthetic_metrics(synthetic)
    figure_prompt_selection()
    figure_real_samples()
    figure_synthetic_samples()
    figure_real20_pipeline()
    figure_real13_comparison()
    print("Figuras generadas en", FIGURES)


if __name__ == "__main__":
    main()
