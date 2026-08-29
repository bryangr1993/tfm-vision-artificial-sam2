"""Selecciona el margen de caja de SAM 2 solo con validación IDV2.

La búsqueda usa las doce láminas de validación con identidades de activo
disjuntas. Las cajas proceden del localizador operativo. Tras bloquear el
margen, el conjunto de prueba se abre una sola vez. El escenario de cajas
ideales se calcula después y se conserva como diagnóstico separado.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import torch

from protocol_common import (
    CHECKPOINT,
    MODEL_CONFIG,
    ProductAlignedSAM2,
    bootstrap_mean_ci,
    build_predictor,
    detect_wcs_for_rectified,
    operational_boxes,
    read_required_image,
    resolve_device,
    sample_metrics,
    sha256_file,
    write_csv,
    ROOT,
)


MANIFEST_IDV2 = ROOT / "datos" / "manifiesto" / "datasets_asset_identity_v2.csv"
QUALITY_IDV2 = ROOT / "resultados" / "metricas" / "dataset_identity_v2_quality.json"
CONFIG_IDV2 = ROOT / "datos" / "manifiesto" / "dataset_identity_v2_config.json"
LEGACY_MANIFEST = ROOT / "datos" / "manifiesto" / "datasets.csv"
METRICS = ROOT / "resultados" / "metricas"
OUTPUT = ROOT / "resultados" / "sam2_identity_v2"
FIGURES = ROOT / "resultados" / "figuras_protocolo_v8"
MARGINS = [0.00, 0.03, 0.05, 0.10]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_dataset(rows: list[dict[str, str]]) -> dict:
    quality = json.loads(QUALITY_IDV2.read_text(encoding="utf-8"))
    if not quality.get("passed") or quality.get("critical_failure_count") != 0:
        raise RuntimeError("La auditoría de integridad IDV2 no está aprobada.")
    if {row["split_protocol"] for row in rows} != {"asset_identity_disjoint"}:
        raise RuntimeError("El manifiesto no declara asset_identity_disjoint de forma uniforme.")
    split_counts = {
        split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")
    }
    if split_counts != {"train": 24, "val": 12, "test": 12}:
        raise RuntimeError(f"Conteos IDV2 inesperados: {split_counts}")
    assets = {
        split: {
            asset
            for row in rows
            if row["split"] == split
            for asset in row["asset_ids"].split(";")
        }
        for split in ("train", "val", "test")
    }
    if assets["train"] & assets["val"] or assets["train"] & assets["test"] or assets["val"] & assets["test"]:
        raise RuntimeError("Se detectó solapamiento de identidades entre particiones IDV2.")
    return {"split_counts": split_counts, "asset_sets": {key: sorted(value) for key, value in assets.items()}}


def ideal_boxes_from_instances(path: Path) -> list[tuple[int, int, int, int]]:
    instances = read_required_image(path, cv2.IMREAD_UNCHANGED)
    if instances.ndim == 3:
        instances = instances[:, :, 0]
    boxes: list[tuple[int, int, int, int]] = []
    for label in sorted(int(value) for value in np.unique(instances) if int(value) > 0):
        ys, xs = np.where(instances == label)
        if xs.size == 0:
            continue
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)))
    if len(boxes) != 8:
        raise RuntimeError(f"Se esperaban 8 cajas ideales en {path}, se obtuvieron {len(boxes)}.")
    return boxes


def margin_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[float(row["box_margin_fraction"])].append(row)
    summaries: list[dict] = []
    for index, margin in enumerate(MARGINS):
        values = grouped[margin]
        ious = [float(row["iou"]) for row in values]
        ci_low, ci_high = bootstrap_mean_ci(ious, seed=12_000 + index)
        component_error = float(np.mean([row["component_error"] for row in values]))
        mean_iou = float(np.mean(ious))
        summaries.append(
            {
                "box_margin_fraction": margin,
                "n_validation_sheets": len(values),
                "mean_validation_iou": mean_iou,
                "std_validation_iou": float(np.std(ious, ddof=1)),
                "iou_bootstrap_95_ci_low": ci_low,
                "iou_bootstrap_95_ci_high": ci_high,
                "mean_validation_dice": float(np.mean([row["dice"] for row in values])),
                "mean_validation_boundary_f1": float(
                    np.mean([row["boundary_f1"] for row in values])
                ),
                "mean_component_error": component_error,
                "selection_score": mean_iou - 0.005 * component_error,
                "mean_prompt_count": float(np.mean([row["prompt_count"] for row in values])),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            -row["selection_score"],
            -row["mean_validation_boundary_f1"],
            row["box_margin_fraction"],
        ),
    )


def evaluate_locked_scenarios(
    rows: list[dict[str, str]],
    runner: ProductAlignedSAM2,
    *,
    split_label: str,
    save_masks: bool,
) -> list[dict]:
    results: list[dict] = []
    for index, row in enumerate(rows, start=1):
        image = read_required_image(ROOT / row["image"])
        reference = read_required_image(ROOT / row["ground_truth"], cv2.IMREAD_GRAYSCALE)
        boxes_operational, _, prompt_seconds = operational_boxes(image, wcs_info=None)
        boxes_ideal = ideal_boxes_from_instances(ROOT / row["instance_mask"])
        for scenario, boxes, source, localization_seconds in [
            ("operational", boxes_operational, "classical_prompt_localizer", prompt_seconds),
            ("ideal", boxes_ideal, "instance_mask_oracle", 0.0),
        ]:
            prediction, scores, timing, _ = runner.predict(image, boxes)
            result = {
                "dataset_version": "asset_identity_v2",
                "split": split_label,
                "sample_id": row["sample_id"],
                "prompt_scenario": scenario,
                "prompt_source": source,
                "box_margin_fraction": runner.box_margin_fraction,
                "prompt_count": len(boxes),
                **sample_metrics(reference, prediction),
                "mean_predicted_score": float(np.mean(scores)),
                "prompt_localization_seconds": localization_seconds,
                **timing,
                "full_hybrid_seconds": localization_seconds + timing["sam_total_seconds"],
            }
            results.append(result)
            if save_masks:
                directory = OUTPUT / "masks" / split_label / scenario
                directory.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(directory / f"{row['sample_id']}_sam2.png"), prediction)
        print(f"SAM 2 IDV2 {split_label} [{index}/{len(rows)}] {row['sample_id']}")
    return results


def evaluate_real(runner: ProductAlignedSAM2) -> list[dict]:
    rows = [row for row in read_csv(LEGACY_MANIFEST) if row["domain"] == "real"]
    results: list[dict] = []
    for index, row in enumerate(rows, start=1):
        image = read_required_image(ROOT / row["image"])
        reference = read_required_image(ROOT / row["ground_truth"], cv2.IMREAD_GRAYSCALE)
        wcs = detect_wcs_for_rectified(image, row["sample_id"])
        boxes_operational, _, prompt_seconds = operational_boxes(image, wcs_info=wcs)
        # En real no existe una máscara de instancias independiente. Las cajas
        # ideales se extraen de la preanotación asistida y solo son diagnósticas.
        count, _, stats, _ = cv2.connectedComponentsWithStats((reference > 0).astype(np.uint8))
        boxes_ideal = []
        for label in range(1, count):
            x, y, width, height, area = map(int, stats[label])
            if area >= 20_000:
                boxes_ideal.append((x, y, x + width, y + height))
        for scenario, boxes, source, localization_seconds in [
            ("operational", boxes_operational, "classical_prompt_localizer", prompt_seconds),
            ("ideal", boxes_ideal, "assisted_reference_oracle_diagnostic", 0.0),
        ]:
            prediction, scores, timing, _ = runner.predict(image, boxes)
            results.append(
                {
                    "dataset_version": "real_repeated_capture",
                    "split": "real",
                    "sample_id": row["sample_id"],
                    "prompt_scenario": scenario,
                    "prompt_source": source,
                    "box_margin_fraction": runner.box_margin_fraction,
                    "prompt_count": len(boxes),
                    "wcs_status": wcs["status"],
                    **sample_metrics(reference, prediction),
                    "mean_predicted_score": float(np.mean(scores)),
                    "prompt_localization_seconds": localization_seconds,
                    **timing,
                    "full_hybrid_seconds": localization_seconds + timing["sam_total_seconds"],
                }
            )
            directory = OUTPUT / "masks" / "real" / scenario
            directory.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(directory / f"{row['sample_id']}_sam2.png"), prediction)
        print(f"SAM 2 real margen bloqueado [{index}/{len(rows)}] {row['sample_id']}")
    return results


def scenario_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["prompt_scenario"])].append(row)
    output = []
    for index, ((split, scenario), values) in enumerate(sorted(grouped.items())):
        ious = [float(row["iou"]) for row in values]
        ci_low, ci_high = bootstrap_mean_ci(ious, seed=14_000 + index)
        output.append(
            {
                "split": split,
                "prompt_scenario": scenario,
                "n_images": len(values),
                "mean_iou": float(np.mean(ious)),
                "std_iou": float(np.std(ious, ddof=1)),
                "iou_bootstrap_95_ci_low": ci_low,
                "iou_bootstrap_95_ci_high": ci_high,
                "mean_dice": float(np.mean([row["dice"] for row in values])),
                "mean_boundary_f1": float(np.mean([row["boundary_f1"] for row in values])),
                "mean_component_error": float(
                    np.mean([row["component_error"] for row in values])
                ),
                "mean_prompt_count": float(np.mean([row["prompt_count"] for row in values])),
            }
        )
    return output


def plot_margin_search(summary: list[dict], winner: float) -> None:
    ordered = sorted(summary, key=lambda row: row["box_margin_fraction"])
    margins = [100 * float(row["box_margin_fraction"]) for row in ordered]
    means = np.asarray([float(row["mean_validation_iou"]) for row in ordered])
    low = means - np.asarray([float(row["iou_bootstrap_95_ci_low"]) for row in ordered])
    high = np.asarray([float(row["iou_bootstrap_95_ci_high"]) for row in ordered]) - means
    colors = ["#D8E5EF" if margin != 100 * winner else "#2F6B9A" for margin in margins]
    fig, ax = plt.subplots(figsize=(4.85, 3.55), constrained_layout=True)
    ax.errorbar(
        margins,
        means,
        yerr=np.asarray([low, high]),
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
    fig.suptitle("Selección del margen de caja de SAM 2 en IDV2", fontsize=10.5)
    ax.set_title(
        "Detalle vertical · intervalos bootstrap del 95 % · el conjunto de prueba no participa",
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
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "sam2_margin_selection_idv2.png", dpi=300, facecolor="white")
    fig.savefig(FIGURES / "sam2_margin_selection_idv2.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()

    all_rows = read_csv(MANIFEST_IDV2)
    integrity = verify_dataset(all_rows)
    val_rows = [row for row in all_rows if row["split"] == "val"]
    test_rows = [row for row in all_rows if row["split"] == "test"]
    device = resolve_device(args.device)
    predictor = build_predictor(device)
    runners = {
        margin: ProductAlignedSAM2(predictor, device=device, box_margin_fraction=margin)
        for margin in MARGINS
    }

    validation: list[dict] = []
    for index, row in enumerate(val_rows, start=1):
        image = read_required_image(ROOT / row["image"])
        reference = read_required_image(ROOT / row["ground_truth"], cv2.IMREAD_GRAYSCALE)
        boxes, _, prompt_seconds = operational_boxes(image, wcs_info=None)
        for margin in MARGINS:
            prediction, scores, timing, _ = runners[margin].predict(image, boxes)
            validation.append(
                {
                    "dataset_version": "asset_identity_v2",
                    "split": "val",
                    "sample_id": row["sample_id"],
                    "condition": row["condition"],
                    "layout": row["layout"],
                    "prompt_scenario": "operational",
                    "prompt_source": "classical_prompt_localizer",
                    "box_margin_fraction": margin,
                    "prompt_count": len(boxes),
                    **sample_metrics(reference, prediction),
                    "mean_predicted_score": float(np.mean(scores)),
                    "prompt_localization_seconds": prompt_seconds,
                    **timing,
                }
            )
        print(f"Búsqueda margen IDV2 [{index}/{len(val_rows)}] {row['sample_id']}")

    summary = margin_summary(validation)
    winner = float(summary[0]["box_margin_fraction"])
    locked_runner = runners[winner]
    test_scenarios = evaluate_locked_scenarios(
        test_rows,
        locked_runner,
        split_label="test",
        save_masks=True,
    )
    real_scenarios = [] if args.skip_real else evaluate_real(locked_runner)
    diagnostic_scenarios = test_scenarios + real_scenarios
    summaries = scenario_summary(diagnostic_scenarios)

    METRICS.mkdir(parents=True, exist_ok=True)
    write_csv(METRICS / "sam2_idv2_margin_validation_metrics.csv", validation)
    write_csv(METRICS / "sam2_idv2_margin_validation_summary.csv", summary)
    write_csv(METRICS / "sam2_idv2_locked_test_prompt_scenarios.csv", test_scenarios)
    write_csv(
        METRICS / "sam2_idv2_locked_test_operational_metrics.csv",
        [row for row in test_scenarios if row["prompt_scenario"] == "operational"],
    )
    if real_scenarios:
        write_csv(METRICS / "sam2_idv2_locked_real_prompt_scenarios.csv", real_scenarios)
    write_csv(METRICS / "sam2_idv2_locked_scenario_summary.csv", summaries)
    selection = {
        "protocol_version": "sam2_prompt_margin_identity_v2",
        "dataset_version": "asset_identity_v2",
        "split_protocol": "asset_identity_disjoint",
        "dataset_integrity": integrity,
        "candidate_margins": MARGINS,
        "selection_partition": "val only",
        "selection_prompt_source": "classical_prompt_localizer",
        "selection_rule": (
            "maximize mean IoU minus 0.005 times mean absolute component-count error; "
            "tie-break by boundary F1 and then smaller margin"
        ),
        "selected_box_margin_fraction": winner,
        "validation_winner": summary[0],
        "test_access_policy": (
            "The test partition was evaluated only after locking the margin on validation; "
            "no alternative margin was evaluated on test."
        ),
        "ideal_prompt_policy": (
            "Ideal boxes were evaluated only after selection as an oracle diagnostic and did not affect the winner."
        ),
        "real_reference_policy": (
            "Real results are agreement with an algorithmically assisted canonical preannotation, not accuracy against independent ground truth."
        ),
        "manifest": str(MANIFEST_IDV2.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": sha256_file(MANIFEST_IDV2),
        "dataset_config_sha256": sha256_file(CONFIG_IDV2),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "model_config": MODEL_CONFIG,
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "validation_table": summary,
        "locked_scenario_summary": summaries,
    }
    (METRICS / "sam2_idv2_prompt_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_margin_search(summary, winner)
    print(json.dumps(selection, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
