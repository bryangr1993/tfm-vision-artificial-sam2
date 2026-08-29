"""Genera análisis descriptivo y figuras del protocolo RF asset_identity_v2."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "resultados" / "metricas"
FIGURES = ROOT / "resultados" / "figuras"
REPORT = ROOT / "documentacion" / "INFORME_DATOS_Y_RF_IDV2.md"


def read_csv(name: str) -> list[dict[str, str]]:
    with (METRICS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name: str, rows: list[dict]) -> None:
    with (METRICS / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_test(test_rows: list[dict[str, str]]) -> tuple[list[dict], dict]:
    grouped: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_model: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in test_rows:
        grouped[(row["model"], row["condition"])].append(row)
        by_model[row["model"]].append(row)
    condition_rows: list[dict] = []
    for (model, condition), values in sorted(grouped.items()):
        condition_rows.append(
            {
                "model": model,
                "condition": condition,
                "sheet_count": len(values),
                "mean_iou": float(np.mean([float(item["iou"]) for item in values])),
                "mean_dice": float(np.mean([float(item["dice"]) for item in values])),
                "mean_boundary_f1": float(np.mean([float(item["boundary_f1"]) for item in values])),
                "mean_component_error": float(np.mean([float(item["component_error"]) for item in values])),
                "mean_inference_seconds": float(
                    np.mean([float(item["inference_seconds_including_features"]) for item in values])
                ),
            }
        )
    summaries: dict[str, dict[str, float | int]] = {}
    for model, values in by_model.items():
        summaries[model] = {
            "sheet_count": len(values),
            "mean_iou": float(np.mean([float(item["iou"]) for item in values])),
            "std_iou": float(np.std([float(item["iou"]) for item in values], ddof=1)),
            "mean_dice": float(np.mean([float(item["dice"]) for item in values])),
            "mean_boundary_f1": float(np.mean([float(item["boundary_f1"]) for item in values])),
            "mean_component_error": float(np.mean([float(item["component_error"]) for item in values])),
            "mean_inference_seconds": float(
                np.mean([float(item["inference_seconds_including_features"]) for item in values])
            ),
        }
    selected = summaries["RF_selected"]
    control = summaries["RF_control"]
    delta = {
        "iou": float(selected["mean_iou"] - control["mean_iou"]),
        "dice": float(selected["mean_dice"] - control["mean_dice"]),
        "boundary_f1": float(selected["mean_boundary_f1"] - control["mean_boundary_f1"]),
        "inference_seconds": float(selected["mean_inference_seconds"] - control["mean_inference_seconds"]),
    }
    return condition_rows, {"models": summaries, "selected_minus_control": delta}


def hyperparameter_sensitivity(search_rows: list[dict[str, str]]) -> list[dict]:
    expanded: list[dict] = []
    for row in search_rows:
        params = json.loads(row["parameters"])
        expanded.append({**row, **params})
    parameters = (
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "class_weight",
    )
    output: list[dict] = []
    for parameter in parameters:
        groups: defaultdict[str, list[float]] = defaultdict(list)
        for row in expanded:
            label = "None" if row[parameter] is None else str(row[parameter])
            groups[label].append(float(row["mean_group_cv_jaccard"]))
        for level, values in sorted(groups.items()):
            output.append(
                {
                    "parameter": parameter,
                    "level": level,
                    "candidate_count": len(values),
                    "mean_group_cv_jaccard": float(np.mean(values)),
                    "min_group_cv_jaccard": min(values),
                    "max_group_cv_jaccard": max(values),
                    "interpretation": "exploratory_marginal_not_causal",
                }
            )
    return output


def level_sort_key(value: str) -> tuple[int, float | str]:
    if value == "None":
        return (2, value)
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def plot_condition_sensitivity(rows: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for model, color, marker in (
        ("RF_selected", "#1f6f8b", "o"),
        ("RF_control", "#a54b4b", "s"),
    ):
        model_rows = sorted((row for row in rows if row["model"] == model), key=lambda row: row["condition"])
        ax.plot(
            [row["condition"] for row in model_rows],
            [row["mean_iou"] for row in model_rows],
            marker=marker,
            linewidth=1.8,
            label=model.replace("RF_", "RF "),
            color=color,
        )
    ax.set_ylabel("IoU medio (dos disposiciones)")
    ax.set_xlabel("Condición sintética")
    ax.set_title("Sensibilidad del RF a las condiciones visuales de prueba")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "rf_idv2_test_condition_sensitivity.png", dpi=180)
    plt.close(fig)


def plot_hyperparameter_sensitivity(rows: list[dict]) -> None:
    parameters = []
    for row in rows:
        if row["parameter"] not in parameters:
            parameters.append(row["parameter"])
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0))
    for ax, parameter in zip(axes.flat, parameters):
        values = sorted(
            (row for row in rows if row["parameter"] == parameter),
            key=lambda row: level_sort_key(str(row["level"])),
        )
        ax.bar(
            [row["level"] for row in values],
            [row["mean_group_cv_jaccard"] for row in values],
            color="#527a58",
        )
        ax.set_title(parameter.replace("_", " "))
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(0.965, 0.976)
    fig.suptitle("Sensibilidad marginal exploratoria del cribado RF", y=1.01)
    fig.text(0.01, 0.01, "Cada nivel resume solo los candidatos muestreados; no es un efecto causal aislado.", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "rf_idv2_hyperparameter_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(posthoc: dict, conditions: list[dict]) -> None:
    quality = json.loads((METRICS / "dataset_identity_v2_quality.json").read_text(encoding="utf-8"))
    lock = json.loads((METRICS / "rf_idv2_selection_locked.json").read_text(encoding="utf-8"))
    ablations = read_csv("rf_idv2_feature_ablation.csv")
    importance = read_csv("rf_idv2_feature_importance.csv")[:5]
    selected = posthoc["models"]["RF_selected"]
    control = posthoc["models"]["RF_control"]
    delta = posthoc["selected_minus_control"]
    condition_selected = {row["condition"]: row for row in conditions if row["model"] == "RF_selected"}
    worst_condition = min(condition_selected.values(), key=lambda row: row["mean_iou"])
    lines = [
        "# Reconstrucción del corpus y del Random Forest",
        "",
        "## Resultado ejecutivo",
        "",
        f"La puerta de calidad del corpus `asset_identity_v2` fue {'aprobada' if quality['passed'] else 'rechazada'} con {quality['critical_failure_count']} fallos críticos. Se generaron 48 láminas: 24 de entrenamiento, 12 de validación y 12 de prueba.",
        "",
        "La partición se realizó antes de generar las láminas. Cada una de las cuatro familias aporta cuatro identidades a entrenamiento, dos a validación y dos a prueba. Las familias son estratos compartidos; las identidades, nombres de archivo y hashes SHA-256 son disjuntos.",
        "",
        f"El modelo seleccionado usa {lock['selected_hyperparameters']['n_estimators']} árboles, profundidad máxima {lock['selected_hyperparameters']['max_depth']}, hoja mínima {lock['selected_hyperparameters']['min_samples_leaf']}, `max_features={lock['selected_hyperparameters']['max_features']}` y umbral {lock['selected_threshold']:.2f}. En validación completa obtuvo IoU {lock['validation']['mean_validation_iou']:.6f} ± {lock['validation']['std_validation_iou']:.6f}, sin error en el número de componentes.",
        "",
        f"En prueba bloqueada el RF seleccionado obtuvo IoU {selected['mean_iou']:.6f} ± {selected['std_iou']:.6f}, Dice {selected['mean_dice']:.6f}, F1 de contorno {selected['mean_boundary_f1']:.6f} y error medio de componentes {selected['mean_component_error']:.1f}. El control alcanzó IoU {control['mean_iou']:.6f}. La diferencia de IoU fue {delta['iou']:+.6f}; es una mejora modesta y no justifica afirmar superioridad estadística.",
        "",
        f"La condición más exigente fue {worst_condition['condition']}, con IoU medio {worst_condition['mean_iou']:.6f}. Corresponde al desenfoque gaussiano combinado con ruido, lo que identifica una debilidad concreta del clasificador.",
        "",
        "## Ablación de características",
        "",
        "| Variante | Variables | Jaccard diagnóstico |",
        "|---|---:|---:|",
    ]
    for row in ablations:
        lines.append(f"| {row['ablation'].replace('_', ' ')} | {row['feature_count']} | {float(row['validation_sampled_pixel_jaccard']):.6f} |")
    lines.extend(
        [
            "",
            "El conjunto completo de 19 variables mejora las variantes de color y gradientes. Añadir coordenadas reduce ligeramente el resultado, por lo que se excluyen del modelo principal. Esta ablación es diagnóstica y no intervino en la selección.",
            "",
            "Las cinco mayores importancias por disminución de impureza fueron: "
            + ", ".join(f"{row['feature']} ({100*float(row['importance']):.1f} %)" for row in importance)
            + ". Estas importancias son descriptivas y pueden favorecer variables continuas correlacionadas.",
            "",
            "## Integridad experimental",
            "",
            "- La búsqueda usó cuatro pliegues disjuntos por cohortes de identidad.",
            "- La selección final se realizó en láminas completas de validación.",
            "- Modelo, umbral y hashes se bloquearon antes de consultar prueba.",
            "- El script de prueba se niega a sobrescribir una evaluación existente.",
            "- El tiempo incluye extracción de características y predicción, pero no describe todavía el tiempo extremo a extremo del software con adquisición, rectificación y exportación.",
            "",
            "## Limitaciones",
            "",
            "Las doce láminas de prueba reutilizan el mismo banco de ocho identidades bajo seis condiciones y dos disposiciones. Por ello no son doce observaciones poblacionales independientes. Los promedios sirven para describir sensibilidad controlada, pero no para construir una afirmación inferencial fuerte. La transferencia a fotografías reales debe informarse como una evaluación externa separada y no debe emplearse para reajustar el modelo.",
            "",
        ]
    )
    real_summary_path = METRICS / "rf_idv2_real_agreement_summary.json"
    if real_summary_path.exists():
        real = json.loads(real_summary_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "## Evaluación externa real posterior al bloqueo",
                "",
                f"Sin modificar modelo ni umbral, se procesaron {real['capture_count']} capturas de una sola lámina física. La concordancia media con la referencia canónica asistida fue IoU {real['mean_agreement_iou']:.6f} ± {real['std_agreement_iou_across_captures']:.6f}, Dice {real['mean_agreement_dice']:.6f} y F1 de contorno {real['mean_agreement_boundary_f1']:.6f}. El error absoluto medio fue de {real['mean_absolute_component_error']:.1f} componentes y ninguna de las trece capturas produjo las ocho componentes esperadas.",
                "",
                f"El tiempo observado fue {real['mean_inference_seconds_including_feature_extraction']:.3f} s por lámina e incluye extracción de las 19 características, clasificación y postprocesamiento. Se obtuvo con otros procesos del proyecto activos y, por ello, es descriptivo: no sustituye al banco temporal final controlado. La salida suele colapsar el papel en una gran región, lo que confirma una brecha de dominio marcada entre síntesis y captura real.",
                "",
                "Estas cifras son **concordancia con una referencia asistida**, no exactitud frente a una anotación independiente. Tampoco representan trece diseños reales distintos. Constituyen evidencia de estabilidad entre adquisiciones de una única hoja y documentan honestamente el fallo de transferencia del RF.",
                "",
            ]
        )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    test_rows = read_csv("rf_idv2_test_metrics.csv")
    conditions, posthoc = aggregate_test(test_rows)
    sensitivity = hyperparameter_sensitivity(read_csv("rf_idv2_search_summary.csv"))
    write_csv("rf_idv2_test_by_condition.csv", conditions)
    write_csv("rf_idv2_hyperparameter_sensitivity.csv", sensitivity)
    (METRICS / "rf_idv2_posthoc_analysis.json").write_text(
        json.dumps(posthoc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot_condition_sensitivity(conditions)
    plot_hyperparameter_sensitivity(sensitivity)
    write_report(posthoc, conditions)
    print(json.dumps(posthoc, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
