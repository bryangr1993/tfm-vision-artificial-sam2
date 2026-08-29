"""Regenera figuras compuestas al tamaño real de impresión del TFM.

El script consume únicamente artefactos ya validados. No entrena modelos ni
recalcula métricas; cambia la composición y la tipografía de las figuras.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "resultados" / "metricas"
FIGURES = ROOT / "resultados" / "figuras"
PROTOCOL_FIGURES = ROOT / "resultados" / "figuras_protocolo_v8"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
        }
    )


def dataset_partitions() -> None:
    rows = [
        row
        for row in read_csv(ROOT / "datos" / "manifiesto" / "datasets_asset_identity_v2.csv")
        if row["domain"] == "synthetic"
    ]
    labels = {
        "train": "Entrenamiento · 24 hojas",
        "val": "Validación · 12 hojas",
        "test": "Prueba · 12 hojas",
    }
    selected: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == split]
        indices = np.linspace(0, len(subset) - 1, 4, dtype=int)
        selected[split] = [subset[int(index)] for index in indices]

    fig, axes = plt.subplots(3, 4, figsize=(5.55, 4.75))
    for row_index, split in enumerate(("train", "val", "test")):
        for column_index, row in enumerate(selected[split]):
            axis = axes[row_index, column_index]
            axis.imshow(rgb(ROOT / row["image"]))
            group = row.get("cv_group") or split.upper()
            axis.set_title(
                f"{group} · {row['layout']} · {row['condition']}",
                fontsize=8.5,
                pad=3,
            )
            axis.axis("off")
        axes[row_index, 0].text(
            -0.25,
            0.5,
            labels[split],
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#15324B",
        )
    fig.suptitle("Muestras de las particiones sintéticas", fontsize=10.5, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        "Cuatro ejemplos por partición; las identidades de activo son disjuntas entre filas.",
        ha="center",
        fontsize=8.5,
        color="#425466",
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.91, bottom=0.07, wspace=0.08, hspace=0.22)
    save(fig, FIGURES / "dataset_idv2_split_contact")


def rf_real_examples() -> None:
    rows = sorted(
        read_csv(METRICS / "rf_idv2_real_agreement_metrics.csv"),
        key=lambda row: float(row["agreement_iou"]),
    )
    representatives = [rows[0], rows[len(rows) // 2], rows[-1]]
    reference = gray(
        ROOT
        / "datos"
        / "anotaciones"
        / "referencia_real_canonica"
        / "referencia_canonica_real.png"
    )
    row_labels = ("Captura", "Referencia asistida", "Predicción RF", "Acuerdo y error")
    fig, axes = plt.subplots(4, 3, figsize=(5.55, 6.85))
    for column, metric_row in enumerate(representatives):
        sample_id = metric_row["sample_id"]
        number = sample_id.rsplit("_", 1)[-1]
        image = rgb(ROOT / "datos" / "reales" / "rectificadas" / f"rectified_{number}.png")
        prediction = gray(ROOT / metric_row["prediction_mask"])
        ref = reference > 0
        pred = prediction > 0
        overlay = np.full((*reference.shape, 3), 245, dtype=np.uint8)
        overlay[ref & pred] = (62, 132, 84)
        overlay[~ref & pred] = (180, 55, 145)
        overlay[ref & ~pred] = (230, 135, 45)
        for row_index, panel in enumerate((image, reference, prediction, overlay)):
            axes[row_index, column].imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            axes[row_index, column].axis("off")
        axes[0, column].set_title(
            f"{sample_id}\nIoU = {float(metric_row['agreement_iou']):.4f}".replace(".", ","),
            fontsize=8.5,
        )
    fig.suptitle("Transferencia del Random Forest a capturas reales", y=0.995, fontsize=10.5, fontweight="bold")
    for label, ypos in zip(row_labels, (0.80, 0.59, 0.38, 0.17)):
        fig.text(0.018, ypos, label, rotation=90, ha="center", va="center", fontsize=8.5, color="#15324B")
    fig.text(
        0.5,
        0.012,
        "Verde: acuerdo · magenta: predicción adicional · naranja: región omitida.",
        ha="center",
        fontsize=8.5,
        color="#425466",
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.045, hspace=0.12, wspace=0.05)
    save(fig, FIGURES / "rf_idv2_real_agreement_examples")


def sam_margin() -> None:
    rows = sorted(
        read_csv(METRICS / "sam2_idv2_margin_validation_summary.csv"),
        key=lambda row: float(row["box_margin_fraction"]),
    )
    margins = [100 * float(row["box_margin_fraction"]) for row in rows]
    means = np.array([float(row["mean_validation_iou"]) for row in rows])
    low = means - np.array([float(row["iou_bootstrap_95_ci_low"]) for row in rows])
    high = np.array([float(row["iou_bootstrap_95_ci_high"]) for row in rows]) - means
    colors = ["#2F6B9A" if margin == 5 else "#D8E5EF" for margin in margins]
    fig, ax = plt.subplots(figsize=(4.85, 3.55), constrained_layout=True)
    ax.errorbar(
        margins,
        means,
        yerr=np.array([low, high]),
        fmt="none",
        ecolor="#333333",
        elinewidth=1.1,
        capsize=4,
        zorder=2,
    )
    ax.scatter(margins, means, s=58, c=colors, edgecolor="#24445C", linewidth=0.9, zorder=3)
    for margin, mean, upper in zip(margins, means, means + high):
        ax.annotate(
            f"{mean:.4f}".replace(".", ","),
            (margin, upper),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    fig.suptitle("Selección del margen de caja de SAM 2", fontsize=10.5, fontweight="bold")
    ax.set_title(
        "Detalle vertical · intervalos bootstrap del 95 % · TEST no participa",
        fontsize=8.5,
        color="#555555",
        pad=9,
    )
    ax.set_xlabel("Margen por lado respecto a la caja (%)")
    ax.set_ylabel("IoU media de validación")
    ax.set_ylim(0.9185, 0.9265)
    ax.set_xlim(-1.5, 11.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}".replace(".", ",")))
    ax.set_xticks(margins)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    save(fig, PROTOCOL_FIGURES / "sam2_margin_selection_idv2")


def wcs_repeatability() -> None:
    rows = [
        row
        for row in read_csv(METRICS / "wcs_registration_per_capture_v8.csv")
        if row["expected_wcs"].lower() == "true" and row["wcs_status"] == "SUCCESS"
    ]
    x = np.array([float(row["origin_x_mm_from_sheet"]) for row in rows])
    y = np.array([float(row["origin_y_mm_from_sheet"]) for row in rows])
    fig, ax = plt.subplots(figsize=(4.75, 3.8), constrained_layout=True)
    ax.scatter(x, y, s=58, color="#2F6B9A", edgecolor="#173A53", linewidth=0.8)
    for row, xv, yv in zip(rows, x, y):
        ax.annotate(row["sample_id"].replace("real_", ""), (xv, yv), xytext=(4, 4), textcoords="offset points", fontsize=8.5)
    ax.axvline(float(x.mean()), color="#373737", linestyle="--", linewidth=1.0)
    ax.axhline(float(y.mean()), color="#373737", linestyle="--", linewidth=1.0)
    fig.suptitle("Origen WCS en diez capturas con marca", fontsize=10.5, fontweight="bold")
    ax.set_title("Coordenadas desde la esquina de la hoja a 10 px/mm", fontsize=8.5, color="#555555", pad=9)
    ax.set_xlabel("Origen X (mm)")
    ax.set_ylabel("Origen Y (mm)")
    ax.grid(True, color="#D9D9D9", linewidth=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    save(fig, PROTOCOL_FIGURES / "wcs_origin_repeatability_v8")


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def integration_status() -> None:
    rows = read_csv(METRICS / "integration_batch_v8.csv")
    checks = [
        ("Marcadores\n4/4", "markers_ok"),
        ("Clasificación\nWCS", "wcs_classification_correct"),
        ("8 siluetas\nexternas", "external_silhouette_policy_ok"),
        ("Política\nDXF", "dxf_policy_correct"),
        ("DXF\nestructural", "dxf_structurally_valid_or_not_applicable"),
    ]
    values = np.array([[1.0 if as_bool(row[key]) else 0.0 for _, key in checks] for row in rows])
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    cmap = matplotlib.colors.ListedColormap(["#D5822A", "#2F6B9A"])
    ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(checks)), [label for label, _ in checks])
    ax.set_yticks(range(len(rows)), [row["sample_id"] for row in rows])
    fig.suptitle("Comprobaciones del lote operativo", fontsize=10.5, fontweight="bold")
    ax.set_title("Azul: conforme; no exportar sin WCS también es conforme", fontsize=8.5, color="#555555", pad=9)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            ax.text(
                column_index,
                row_index,
                "OK" if values[row_index, column_index] == 1 else "NO",
                ha="center",
                va="center",
                color="white" if values[row_index, column_index] == 1 else "#262626",
                fontsize=8.5,
                fontweight="bold",
            )
    save(fig, PROTOCOL_FIGURES / "integration_batch_status_v8")


def main() -> None:
    configure()
    dataset_partitions()
    rf_real_examples()
    sam_margin()
    wcs_repeatability()
    integration_status()
    print("Figuras legibles regeneradas desde artefactos existentes.")


if __name__ == "__main__":
    main()
