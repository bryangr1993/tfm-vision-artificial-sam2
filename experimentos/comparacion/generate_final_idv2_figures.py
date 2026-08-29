"""Valida y genera las figuras comparativas finales del protocolo IDV2.

Entradas principales:

* TEST comun de ``asset_identity_v2`` para clasica, RF seleccionado y SAM 2
  operativo.
* Concordancia externa sobre 13 capturas de una sola lamina fisica y una
  referencia canonica asistida.
* Banco temporal coherente, ejecutado en el mismo proceso y hardware.

El script falla antes de dibujar si las poblaciones, claves, definiciones o
bloqueos de seleccion son incompatibles. Las tablas que alimentan las figuras
se guardan junto con un informe de validacion y hashes de sus fuentes.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METRICS = ROOT / "resultados" / "metricas"
FIGURES = ROOT / "resultados" / "figuras"

MANIFEST = ROOT / "datos" / "manifiesto" / "datasets_asset_identity_v2.csv"
CLASSICAL_TEST = METRICS / "classical_idv2_test_metrics.csv"
CLASSICAL_TEST_SUMMARY = METRICS / "classical_idv2_test_summary.json"
RF_TEST = METRICS / "rf_idv2_test_metrics.csv"
RF_TEST_SUMMARY = METRICS / "rf_idv2_test_summary.json"
RF_LOCK = METRICS / "rf_idv2_selection_locked.json"
SAM_TEST = METRICS / "sam2_idv2_locked_test_operational_metrics.csv"
SAM_LOCK = METRICS / "sam2_idv2_prompt_selection.json"

CLASSICAL_REAL = METRICS / "classical_real_metrics.csv"
CLASSICAL_REAL_SUMMARY = METRICS / "classical_rf_real_summary.json"
RF_REAL = METRICS / "rf_idv2_real_agreement_metrics.csv"
RF_REAL_SUMMARY = METRICS / "rf_idv2_real_agreement_summary.json"
SAM_SCENARIOS = METRICS / "sam2_idv2_locked_scenario_summary.csv"
SAM_REAL_SCENARIOS = METRICS / "sam2_idv2_locked_real_prompt_scenarios.csv"
REFERENCE_AUDIT = METRICS / "real_reference_validation_v8.json"

RUNTIME_RUNS = METRICS / "segmentation_runtime_idv2_final_runs.csv"
RUNTIME_SUMMARY = METRICS / "segmentation_runtime_idv2_final_summary.json"

TEST_SOURCE_CSV = METRICS / "comparison_idv2_final_per_sheet.csv"
TEST_SUMMARY_CSV = METRICS / "comparison_idv2_final_summary.csv"
GAP_SOURCE_CSV = METRICS / "synthetic_real_gap_final.csv"
RUNTIME_SOURCE_CSV = METRICS / "runtime_comparison_idv2_final.csv"
VALIDATION_JSON = METRICS / "final_comparison_validation.json"

FIG_TEST = FIGURES / "comparison_test_idv2_final"
FIG_GAP = FIGURES / "synthetic_real_gap_final"
FIG_RUNTIME = FIGURES / "runtime_comparison_idv2_final"

METHODS = (
    "Visión clásica fija",
    "Random Forest seleccionado",
    "SAM 2 operativo",
)
METHOD_COLORS = {
    "Visión clásica fija": "#245B78",
    "Random Forest seleccionado": "#C77700",
    "SAM 2 operativo": "#3A7D6D",
}
METHOD_MARKERS = {
    "Visión clásica fija": "o",
    "Random Forest seleccionado": "s",
    "SAM 2 operativo": "^",
}
RUNTIME_METHOD_MAP = {
    "vision_clasica_fija": "Visión clásica fija",
    "random_forest_seleccionado": "Random Forest seleccionado",
    "sam2_operativo_hibrido": "SAM 2 operativo",
}
METRIC_LABELS = {
    "iou": "IoU",
    "dice": "Dice",
    "boundary_f1": "Boundary F1 (3 px)",
}
SEED = 20260827


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No hay filas para {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_unit(value: str | float, label: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise RuntimeError(f"{label} fuera de [0, 1]: {value}")
    return parsed


def finite_nonnegative(value: str | float, label: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise RuntimeError(f"{label} debe ser finito y no negativo: {value}")
    return parsed


def assert_close(actual: float, expected: float, label: str, atol: float = 1e-10) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise RuntimeError(f"{label}: {actual} no coincide con {expected}")


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "text.color": "#252A2E",
            "axes.labelcolor": "#252A2E",
            "axes.edgecolor": "#555D63",
            "xtick.color": "#3F474D",
            "ytick.color": "#3F474D",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def validate_test_comparison() -> tuple[list[dict], list[dict], list[str]]:
    manifest_rows = read_csv(MANIFEST)
    test_manifest = [row for row in manifest_rows if row["domain"] == "synthetic" and row["split"] == "test"]
    expected_ids = sorted(row["sample_id"] for row in test_manifest)
    if len(expected_ids) != 12 or len(set(expected_ids)) != 12:
        raise RuntimeError("El manifiesto no contiene 12 claves TEST unicas.")

    classical_summary = read_json(CLASSICAL_TEST_SUMMARY)
    rf_summary = read_json(RF_TEST_SUMMARY)
    rf_lock = read_json(RF_LOCK)
    sam_lock = read_json(SAM_LOCK)
    if classical_summary.get("test_set_used_for_selection") is not False:
        raise RuntimeError("La linea clasica no esta declarada como fija.")
    if rf_lock.get("test_set_used_for_selection") is not False:
        raise RuntimeError("El RF no fue bloqueado antes de TEST.")
    if sam_lock.get("selection_partition") != "val only":
        raise RuntimeError("El margen de SAM 2 no fue seleccionado solo en validacion.")
    manifest_hash = sha256_file(MANIFEST)
    for label, value in (
        ("clasica", classical_summary["manifest_sha256"]),
        ("RF", rf_lock["dataset_manifest_sha256"]),
        ("SAM 2", sam_lock["manifest_sha256"]),
    ):
        if str(value).lower() != manifest_hash:
            raise RuntimeError(f"El manifiesto de {label} no coincide con el archivo actual.")

    raw_by_method: dict[str, list[dict[str, str]]] = {
        "Visión clásica fija": read_csv(CLASSICAL_TEST),
        "Random Forest seleccionado": [
            row for row in read_csv(RF_TEST) if row["model"] == "RF_selected"
        ],
        "SAM 2 operativo": [
            row
            for row in read_csv(SAM_TEST)
            if row["split"] == "test" and row["prompt_scenario"] == "operational"
        ],
    }
    normalized: list[dict] = []
    for method in METHODS:
        rows = raw_by_method[method]
        ids = [row["sample_id"] for row in rows]
        if len(rows) != 12 or len(set(ids)) != 12 or sorted(ids) != expected_ids:
            raise RuntimeError(f"Poblacion TEST incompatible para {method}.")
        for row in rows:
            if row.get("split") != "test":
                raise RuntimeError(f"Particion inesperada en {method}: {row['sample_id']}")
            normalized.append(
                {
                    "dataset_version": "asset_identity_v2",
                    "split": "test",
                    "sample_id": row["sample_id"],
                    "layout": row.get("layout") or row["sample_id"].split("_")[2],
                    "condition": row.get("condition") or row["sample_id"].split("_")[3],
                    "method": method,
                    "iou": finite_unit(row["iou"], f"IoU {method}"),
                    "dice": finite_unit(row["dice"], f"Dice {method}"),
                    "boundary_f1": finite_unit(row["boundary_f1"], f"BF1 {method}"),
                    "boundary_tolerance_px": 3,
                    "expected_components": int(row["expected_components"]),
                    "predicted_components": int(row["predicted_components"]),
                    "component_error": finite_nonnegative(row["component_error"], "error de componentes"),
                }
            )

    if any(row["expected_components"] != 8 for row in normalized):
        raise RuntimeError("Alguna lamina TEST no tiene ocho componentes de referencia.")

    summaries: list[dict] = []
    for method in METHODS:
        rows = [row for row in normalized if row["method"] == method]
        summary: dict[str, object] = {
            "dataset_version": "asset_identity_v2",
            "split": "test",
            "method": method,
            "n_sheets": len(rows),
            "mean_component_error": float(np.mean([row["component_error"] for row in rows])),
        }
        for metric in METRIC_LABELS:
            values = np.asarray([float(row[metric]) for row in rows])
            summary[f"mean_{metric}"] = float(values.mean())
            summary[f"std_{metric}"] = float(values.std(ddof=1))
            summary[f"min_{metric}"] = float(values.min())
            summary[f"max_{metric}"] = float(values.max())
        summaries.append(summary)

    calculated = {row["method"]: row for row in summaries}
    assert_close(
        float(calculated["Visión clásica fija"]["mean_iou"]),
        float(classical_summary["test_mean_iou"]),
        "IoU clasica",
    )
    assert_close(
        float(calculated["Random Forest seleccionado"]["mean_iou"]),
        float(rf_summary["models"]["RF_selected"]["mean_iou"]),
        "IoU RF",
    )
    locked_test = next(
        row
        for row in sam_lock["locked_scenario_summary"]
        if row["split"] == "test" and row["prompt_scenario"] == "operational"
    )
    assert_close(
        float(calculated["SAM 2 operativo"]["mean_iou"]),
        float(locked_test["mean_iou"]),
        "IoU SAM 2",
    )
    write_csv(TEST_SOURCE_CSV, normalized)
    write_csv(TEST_SUMMARY_CSV, summaries)
    return normalized, summaries, expected_ids


def validate_domain_gap(test_summaries: list[dict]) -> list[dict]:
    classical_summary = read_json(CLASSICAL_REAL_SUMMARY)["classical"]
    rf_summary = read_json(RF_REAL_SUMMARY)
    sam_lock = read_json(SAM_LOCK)
    reference_audit = read_json(REFERENCE_AUDIT)
    classical_rows = read_csv(CLASSICAL_REAL)
    rf_rows = read_csv(RF_REAL)
    sam_rows = [
        row
        for row in read_csv(SAM_REAL_SCENARIOS)
        if row["split"] == "real" and row["prompt_scenario"] == "operational"
    ]
    expected_real_ids = sorted(row["sample_id"] for row in classical_rows)
    if len(expected_real_ids) != 13 or len(set(expected_real_ids)) != 13:
        raise RuntimeError("La linea clasica real no contiene 13 capturas unicas.")
    if sorted(row["sample_id"] for row in rf_rows) != expected_real_ids:
        raise RuntimeError("El RF real no usa las mismas 13 capturas.")
    if sorted(row["sample_id"] for row in sam_rows) != expected_real_ids:
        raise RuntimeError("SAM 2 real no usa las mismas 13 capturas.")
    if rf_summary.get("real_data_used_for_selection_or_tuning") is not False:
        raise RuntimeError("El RF uso el dominio real para seleccion o ajuste.")
    if rf_summary.get("reference_status") != "assisted_canonical_not_independent_ground_truth":
        raise RuntimeError("El RF real no declara la naturaleza asistida de la referencia.")
    if int(rf_summary.get("physical_sheet_count", 0)) != 1:
        raise RuntimeError("La evaluacion real no declara una sola lamina fisica.")
    if reference_audit.get("experimental_unit") != "one physical sheet observed in 13 captures":
        raise RuntimeError("La auditoria de referencia real no coincide con la unidad experimental.")
    assert_close(
        float(classical_summary["mean_iou"]),
        float(reference_audit["mean_iou_with_classical_capture_masks"]),
        "Concordancia clasica con referencia asistida",
    )

    sam_real = next(
        row
        for row in sam_lock["locked_scenario_summary"]
        if row["split"] == "real" and row["prompt_scenario"] == "operational"
    )
    real_values = {
        "Visión clásica fija": {
            "iou": float(classical_summary["mean_iou"]),
            "dice": float(classical_summary["mean_dice"]),
            "boundary_f1": float(classical_summary["mean_boundary_f1"]),
            "component_error": float(classical_summary["mean_component_error"]),
        },
        "Random Forest seleccionado": {
            "iou": float(rf_summary["mean_agreement_iou"]),
            "dice": float(rf_summary["mean_agreement_dice"]),
            "boundary_f1": float(rf_summary["mean_agreement_boundary_f1"]),
            "component_error": float(rf_summary["mean_absolute_component_error"]),
        },
        "SAM 2 operativo": {
            "iou": float(sam_real["mean_iou"]),
            "dice": float(sam_real["mean_dice"]),
            "boundary_f1": float(sam_real["mean_boundary_f1"]),
            "component_error": float(sam_real["mean_component_error"]),
        },
    }
    test_by_method = {row["method"]: row for row in test_summaries}
    rows_out: list[dict] = []
    for method in METHODS:
        synthetic_iou = float(test_by_method[method]["mean_iou"])
        real = real_values[method]
        for metric in ("iou", "dice", "boundary_f1"):
            finite_unit(real[metric], f"{metric} real {method}")
        rows_out.append(
            {
                "method": method,
                "synthetic_dataset": "asset_identity_v2 TEST",
                "synthetic_sheet_count": 12,
                "synthetic_mean_iou": synthetic_iou,
                "real_evaluation": "external agreement with assisted canonical reference",
                "real_capture_count": 13,
                "real_physical_sheet_count": 1,
                "real_mean_agreement_iou": real["iou"],
                "real_mean_agreement_dice": real["dice"],
                "real_mean_agreement_boundary_f1": real["boundary_f1"],
                "real_mean_absolute_component_error": real["component_error"],
                "real_minus_synthetic_iou": real["iou"] - synthetic_iou,
                "reference_status": "assisted_canonical_not_independent_ground_truth",
                "interpretation": "descriptive_domain_contrast_not_population_generalization",
            }
        )
    write_csv(GAP_SOURCE_CSV, rows_out)
    return rows_out


def validate_runtime() -> tuple[list[dict], list[dict]]:
    payload = read_json(RUNTIME_SUMMARY)
    runs = read_csv(RUNTIME_RUNS)
    if payload.get("status") != "comparable_final_methods_same_process_hardware_and_inputs":
        raise RuntimeError("El banco temporal final no declara comparabilidad completa.")
    required_true = ("same_process_and_hardware", "image_loading_excluded", "model_loading_excluded")
    if any(payload.get(field) is not True for field in required_true):
        raise RuntimeError("El alcance temporal no es coherente entre metodos.")
    expected_ids = sorted(payload["sample_ids"])
    if len(expected_ids) != 12 or len(set(expected_ids)) != 12:
        raise RuntimeError("El banco temporal no contiene 12 imagenes TEST unicas.")
    repetitions = int(payload["measured_repetitions_per_image"])
    keys = [(row["sample_id"], row["method"], int(row["repetition"])) for row in runs]
    if len(keys) != 12 * 3 * repetitions or len(set(keys)) != len(keys):
        raise RuntimeError("El banco temporal contiene claves duplicadas o faltantes.")
    if set(row["method"] for row in runs) != set(RUNTIME_METHOD_MAP):
        raise RuntimeError("El banco temporal no contiene los tres metodos finales.")
    if any(row["device"] != "cpu" for row in runs):
        raise RuntimeError("Los metodos no usan el mismo tipo de dispositivo.")
    if any((int(row["input_width_px"]), int(row["input_height_px"])) != (2100, 2970) for row in runs):
        raise RuntimeError("El banco temporal mezcla dimensiones de entrada.")
    if any(row["sample_id"] not in expected_ids for row in runs):
        raise RuntimeError("El banco temporal contiene muestras ajenas a TEST.")
    for row in runs:
        finite_nonnegative(row["segmentation_seconds"], "tiempo de segmentacion")

    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in runs:
        grouped[RUNTIME_METHOD_MAP[row["method"]]].append(float(row["segmentation_seconds"]))
    summaries: list[dict] = []
    payload_summary = {RUNTIME_METHOD_MAP[row["method"]]: row for row in payload["summary"]}
    for method in METHODS:
        values = np.asarray(grouped[method])
        summary = {
            "method": method,
            "n_images": len(expected_ids),
            "repetitions_per_image": repetitions,
            "n_measured_runs": len(values),
            "mean_seconds": float(values.mean()),
            "std_seconds": float(values.std(ddof=1)),
            "median_seconds": float(np.median(values)),
            "p25_seconds": float(np.quantile(values, 0.25)),
            "p75_seconds": float(np.quantile(values, 0.75)),
            "min_seconds": float(values.min()),
            "max_seconds": float(values.max()),
            "device": "cpu",
            "scope": payload["scope"],
        }
        assert_close(
            float(summary["mean_seconds"]),
            float(payload_summary[method]["mean_seconds"]),
            f"Tiempo medio {method}",
        )
        summaries.append(summary)
    write_csv(RUNTIME_SOURCE_CSV, summaries)
    return runs, summaries


def figure_test(rows: list[dict], summaries: list[dict]) -> None:
    configure_plotting()
    fig, ax = plt.subplots(figsize=(5.7, 4.5))
    rng = np.random.default_rng(SEED)
    metric_names = list(METRIC_LABELS)
    centers = np.arange(len(metric_names), dtype=float)
    offsets = dict(zip(METHODS, (-0.25, 0.0, 0.25)))
    summary_map = {row["method"]: row for row in summaries}
    for method in METHODS:
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        for metric_index, metric in enumerate(metric_names):
            values = np.asarray([float(row[metric]) for row in rows if row["method"] == method])
            x = np.full(len(values), centers[metric_index] + offsets[method])
            x += rng.uniform(-0.035, 0.035, size=len(values))
            ax.scatter(
                x,
                values,
                s=18,
                facecolors="none",
                edgecolors=color,
                marker=marker,
                linewidths=0.8,
                alpha=0.42,
                zorder=2,
            )
            mean = float(summary_map[method][f"mean_{metric}"])
            ax.scatter(
                centers[metric_index] + offsets[method],
                mean,
                s=74,
                color=color,
                edgecolor="#202428",
                linewidth=0.7,
                marker=marker,
                zorder=4,
                label={
                    "Visión clásica fija": "Visión clásica",
                    "Random Forest seleccionado": "Random Forest",
                    "SAM 2 operativo": "SAM 2",
                }[method] if metric_index == 0 else None,
            )
            ax.annotate(
                f"{mean:.3f}".replace(".", ","),
                (centers[metric_index] + offsets[method], mean),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=color,
                fontweight="bold",
            )
    fig.suptitle(
        "Comparación en TEST con identidades separadas",
        x=0.5,
        y=0.965,
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.895,
        "asset_identity_v2 · 12 láminas · escala enfocada 0,70-1,00\n"
        "Puntos abiertos: cada lámina · sólido: media",
        ha="center",
        va="top",
        fontsize=8.7,
        color="#596168",
    )
    ax.set_xticks(centers, [METRIC_LABELS[name] for name in metric_names])
    ax.set_ylabel("Puntuación")
    ax.set_ylim(0.70, 1.005)
    ax.set_yticks(np.arange(0.70, 1.01, 0.05))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}".replace(".", ",")))
    ax.grid(axis="y", color="#D8DDE1", linewidth=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.23), ncol=3, frameon=False)
    fig.text(
        0.01,
        0.01,
        "Boundary F1 usa tolerancia de 3 px.\n"
        "Las tres salidas obtuvieron error medio de componentes igual a 0.",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#596168",
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.735, bottom=0.25)
    save_figure(fig, FIG_TEST)


def figure_gap(rows: list[dict]) -> None:
    configure_plotting()
    fig, ax = plt.subplots(figsize=(5.55, 4.75))
    positions = np.arange(len(METHODS))
    width = 0.34
    synthetic = [float(next(row for row in rows if row["method"] == method)["synthetic_mean_iou"]) for method in METHODS]
    real = [
        float(next(row for row in rows if row["method"] == method)["real_mean_agreement_iou"])
        for method in METHODS
    ]
    bars_synthetic = ax.bar(
        positions - width / 2,
        synthetic,
        width,
        color="#245B78",
        edgecolor="#173A4E",
        linewidth=0.8,
        label="TEST sintético",
    )
    bars_real = ax.bar(
        positions + width / 2,
        real,
        width,
        color="#E7B66E",
        edgecolor="#8C5707",
        linewidth=0.8,
        hatch="///",
        label="Capturas reales asistidas",
    )
    for bars in (bars_synthetic, bars_real):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.3f}".replace(".", ","),
                ha="center",
                va="bottom",
                fontsize=9,
                color="#252A2E",
                fontweight="bold",
            )
    for index, method in enumerate(METHODS):
        row = next(row for row in rows if row["method"] == method)
        gap = float(row["real_minus_synthetic_iou"])
        sign = "+" if gap >= 0 else "−"
        label = f"Δ {sign}{abs(gap):.3f}".replace(".", ",")
        y = max(synthetic[index], real[index]) + 0.10
        ax.text(index, y, label, ha="center", va="bottom", fontsize=9, color="#596168")
    ax.set_title("Contraste descriptivo entre TEST sintético\ny capturas reales", loc="center", pad=24)
    ax.text(
        0.0,
        1.015,
        "IoU media · barras desde cero\n"
        "Δ = concordancia real asistida − TEST sintético",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#596168",
    )
    ax.set_xticks(positions, METHODS)
    ax.set_ylabel("IoU / concordancia IoU")
    ax.set_ylim(0.0, 1.13)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}".replace(".", ",")))
    ax.grid(axis="y", color="#D8DDE1", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False)
    fig.text(
        0.05,
        0.025,
        "La referencia real fue construida con asistencia algorítmica; las 13 capturas corresponden a una sola lámina.\n"
        "El contraste no estima generalización poblacional ni equivale a exactitud frente a una referencia independiente.",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#596168",
        wrap=True,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.82, bottom=0.34)
    save_figure(fig, FIG_GAP)


def figure_runtime(rows: list[dict]) -> None:
    configure_plotting()
    fig, ax = plt.subplots(figsize=(5.35, 3.65))
    y_positions = np.arange(len(METHODS))[::-1]
    for y, method in zip(y_positions, METHODS):
        row = next(row for row in rows if row["method"] == method)
        p25 = float(row["p25_seconds"])
        p75 = float(row["p75_seconds"])
        median = float(row["median_seconds"])
        mean = float(row["mean_seconds"])
        color = METHOD_COLORS[method]
        ax.hlines(y, p25, p75, color=color, linewidth=6, alpha=0.35, zorder=2)
        ax.scatter(
            median,
            y,
            s=95,
            color=color,
            marker=METHOD_MARKERS[method],
            edgecolor="#202428",
            linewidth=0.8,
            zorder=4,
        )
        ax.text(
            p75 * 1.12,
            y,
            f"mediana {median:.3f} s · media {mean:.3f} s".replace(".", ","),
            va="center",
            ha="left",
            fontsize=9,
            color="#3F474D",
        )
    ax.set_title("Tiempo de segmentación de los tres métodos finales", loc="left", pad=24)
    ax.text(
        0.0,
        1.015,
        "12 láminas TEST · 3 repeticiones por lámina · CPU · mismo proceso y hardware · escala horizontal logarítmica",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#596168",
    )
    ax.set_yticks(y_positions, METHODS)
    ax.set_xscale("log")
    tick_values = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30]
    ax.xaxis.set_major_locator(FixedLocator(tick_values))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
    ax.set_xlim(0.09, 45)
    ax.set_xlabel("Segundos por lámina rectificada")
    ax.grid(axis="x", which="major", color="#D8DDE1", linewidth=0.75)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(
        0.01,
        0.01,
        "Símbolo: mediana. Segmento grueso: intervalo intercuartílico. Se excluyeron carga de archivos, carga de modelos, "
        "rectificación y detección del WCS. SAM 2 incluye el localizador clásico de cajas y el codificador.",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#596168",
        wrap=True,
    )
    fig.subplots_adjust(left=0.26, right=0.95, top=0.77, bottom=0.23)
    save_figure(fig, FIG_RUNTIME)


def main() -> None:
    test_rows, test_summaries, test_ids = validate_test_comparison()
    gap_rows = validate_domain_gap(test_summaries)
    _, runtime_summaries = validate_runtime()
    figure_test(test_rows, test_summaries)
    figure_gap(gap_rows)
    figure_runtime(runtime_summaries)

    source_paths = (
        MANIFEST,
        CLASSICAL_TEST,
        CLASSICAL_TEST_SUMMARY,
        RF_TEST,
        RF_TEST_SUMMARY,
        RF_LOCK,
        SAM_TEST,
        SAM_LOCK,
        CLASSICAL_REAL,
        CLASSICAL_REAL_SUMMARY,
        RF_REAL,
        RF_REAL_SUMMARY,
        SAM_SCENARIOS,
        SAM_REAL_SCENARIOS,
        REFERENCE_AUDIT,
        RUNTIME_RUNS,
        RUNTIME_SUMMARY,
    )
    report = {
        "status": "passed",
        "assessment": "ready_to_use_with_explicit_real_reference_caveat",
        "validated_test_sample_ids": test_ids,
        "checks": {
            "same_test_population_all_methods": True,
            "test_sheet_count_per_method": 12,
            "test_sample_ids_unique": True,
            "metric_ranges_valid": True,
            "boundary_f1_tolerance_px": 3,
            "selection_locks_exclude_test": True,
            "dataset_manifest_hash_aligned": True,
            "real_capture_population_aligned": True,
            "real_capture_count": 13,
            "real_physical_sheet_count": 1,
            "real_reference_is_assisted_not_independent": True,
            "runtime_same_process_hardware_inputs": True,
            "runtime_scope_aligned": True,
        },
        "required_caveat": (
            "Real-domain metrics are agreement with an algorithmically assisted canonical reference "
            "over 13 captures of one physical sheet, not accuracy against independent ground truth "
            "or population-level generalization."
        ),
        "chart_contracts": [
            {
                "figure": "comparison_test_idv2_final",
                "question": "How do the three final methods compare on the same locked IDV2 TEST sheets?",
                "family": "Comparison and uncertainty",
                "variant": "grouped strip and mean plot",
                "palette": "three explicit method colors plus distinct marker shapes",
                "scale": "focused score scale 0.70-1.00, explicitly disclosed",
            },
            {
                "figure": "synthetic_real_gap_final",
                "question": "How do synthetic TEST IoU and assisted real-domain agreement differ descriptively?",
                "family": "Comparison",
                "variant": "paired bars from zero",
                "palette": "hard two-root domain comparison with hatch",
                "scale": "zero-based 0-1.13",
            },
            {
                "figure": "runtime_comparison_idv2_final",
                "question": "What is the segmentation time of each final method under one coherent protocol?",
                "family": "Distribution and comparison",
                "variant": "median and interquartile dot-interval",
                "palette": "three explicit method colors plus distinct marker shapes",
                "scale": "logarithmic seconds, explicitly disclosed",
            },
        ],
        "source_hashes": {
            path.relative_to(ROOT).as_posix(): sha256_file(path) for path in source_paths
        },
        "derived_tables": [
            TEST_SOURCE_CSV.relative_to(ROOT).as_posix(),
            TEST_SUMMARY_CSV.relative_to(ROOT).as_posix(),
            GAP_SOURCE_CSV.relative_to(ROOT).as_posix(),
            RUNTIME_SOURCE_CSV.relative_to(ROOT).as_posix(),
        ],
        "figures": [
            FIG_TEST.with_suffix(".png").relative_to(ROOT).as_posix(),
            FIG_TEST.with_suffix(".pdf").relative_to(ROOT).as_posix(),
            FIG_GAP.with_suffix(".png").relative_to(ROOT).as_posix(),
            FIG_GAP.with_suffix(".pdf").relative_to(ROOT).as_posix(),
            FIG_RUNTIME.with_suffix(".png").relative_to(ROOT).as_posix(),
            FIG_RUNTIME.with_suffix(".pdf").relative_to(ROOT).as_posix(),
        ],
        "generator": Path(__file__).resolve().relative_to(ROOT).as_posix(),
    }
    VALIDATION_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
