"""Consolida la seleccion de Random Forest sin consultar el conjunto de prueba.

La regla compara exclusivamente el rendimiento por hoja completa en validacion.
Los resultados de prueba se incorporan despues de fijar modelo y umbral.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "resultados" / "metricas"


def read_json(name: str) -> dict:
    return json.loads((METRICS / name).read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict]:
    with (METRICS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values)


def main() -> None:
    v1 = read_json("rf_v1_control_selection.json")
    v2 = read_json("rf_final_selection.json")
    summary_rows = read_csv("rf_validation_model_threshold_summary.csv")
    optimized_val = max(
        (row for row in summary_rows if row["model"] == "RF_busqueda_agrupada"),
        key=lambda row: float(row["selection_score"]),
    )
    optimized_test = read_csv("rf_test_metrics.csv")

    candidates = [
        {
            "model": v1["model"],
            "origin": "control previo sin aumento",
            "threshold": v1["selected_threshold"],
            "validation_iou": v1["validation"]["mean_validation_iou"],
            "validation_component_error": v1["validation"]["mean_component_error"],
            "selection_score": v1["validation"]["selection_score"],
            "test_iou": v1["test_mean_iou"],
            "test_dice": v1["test_mean_dice"],
            "test_boundary_f1": v1["test_mean_boundary_f1"],
            "test_component_error": v1["test_mean_component_error"],
        },
        {
            "model": v2["selected_model"],
            "origin": "control previo con aumento",
            "threshold": v2["selected_threshold"],
            "validation_iou": v2["validation"]["mean_validation_iou"],
            "validation_component_error": v2["validation"]["mean_component_error"],
            "selection_score": v2["validation"]["selection_score"],
            "test_iou": v2["test_mean_iou"],
            "test_dice": v2["test_mean_dice"],
            "test_boundary_f1": v2["test_mean_boundary_f1"],
            "test_component_error": v2["test_mean_component_error"],
        },
        {
            "model": "RF_busqueda_agrupada",
            "origin": "busqueda aleatoria con GroupKFold",
            "threshold": float(optimized_val["threshold"]),
            "validation_iou": float(optimized_val["mean_validation_iou"]),
            "validation_component_error": float(optimized_val["mean_component_error"]),
            "selection_score": float(optimized_val["selection_score"]),
            "test_iou": mean(optimized_test, "iou"),
            "test_dice": mean(optimized_test, "dice"),
            "test_boundary_f1": mean(optimized_test, "boundary_f1"),
            "test_component_error": mean(optimized_test, "component_error"),
        },
    ]

    selected = max(candidates, key=lambda row: row["selection_score"])
    fieldnames = list(candidates[0])
    with (METRICS / "rf_consolidated_model_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    artifact = {
        "selection_protocol": (
            "maximum mean full-sheet validation IoU minus 0.005 times "
            "the mean absolute component-count error"
        ),
        "test_set_used_for_selection": False,
        "selected": selected,
        "candidates": candidates,
    }
    (METRICS / "rf_consolidated_selection.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["selected"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
