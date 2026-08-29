"""Compara SAM 2 con cajas ideales y operativas usando un único protocolo.

Las cajas ideales constituyen un escenario diagnóstico optimista. Las cajas
operativas se recalculan mediante el localizador clásico que alimenta el
producto. En ambos casos, SAM 2 usa el margen, selector multimáscara y
postproceso del software. El tiempo comunicado incluye el codificador
(`set_image`) y el decodificador, pero las comparaciones repetidas se generan
con ``benchmark_segmentation_runtime.py``.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from protocol_common import (
    CHECKPOINT,
    MODEL_CONFIG,
    ProductAlignedSAM2,
    bootstrap_mean_ci,
    build_predictor,
    detect_wcs_for_rectified,
    ideal_boxes,
    load_manifest,
    operational_boxes,
    read_required_image,
    resolve_device,
    sample_metrics,
    sha256_file,
    write_csv,
    ROOT,
)


OUTPUT = ROOT / "resultados" / "sam2_protocol_v8"
METRICS = ROOT / "resultados" / "metricas"


def selected_rows(partitions: set[str]) -> list[dict[str, str]]:
    rows = []
    for row in load_manifest():
        partition = row["split"] if row["domain"] == "synthetic" else "real"
        if partition in partitions:
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["partition"], row["prompt_scenario"])].append(row)
    result: list[dict] = []
    for group_index, ((domain, partition, scenario), values) in enumerate(sorted(grouped.items())):
        ious = [float(row["iou"]) for row in values]
        dices = [float(row["dice"]) for row in values]
        boundaries = [float(row["boundary_f1"]) for row in values]
        ci_low, ci_high = bootstrap_mean_ci(ious, seed=8_000 + group_index)
        result.append(
            {
                "domain": domain,
                "partition": partition,
                "prompt_scenario": scenario,
                "n_images": len(values),
                "mean_iou": float(np.mean(ious)),
                "std_iou": float(np.std(ious, ddof=1)) if len(ious) > 1 else 0.0,
                "iou_bootstrap_95_ci_low": ci_low,
                "iou_bootstrap_95_ci_high": ci_high,
                "mean_dice": float(np.mean(dices)),
                "std_dice": float(np.std(dices, ddof=1)) if len(dices) > 1 else 0.0,
                "mean_boundary_f1": float(np.mean(boundaries)),
                "mean_component_error": float(np.mean([row["component_error"] for row in values])),
                "mean_prompt_count": float(np.mean([row["prompt_count"] for row in values])),
                "mean_encoder_seconds": float(np.mean([row["encoder_seconds"] for row in values])),
                "mean_decoder_seconds": float(np.mean([row["decoder_seconds"] for row in values])),
                "mean_sam_total_seconds": float(np.mean([row["sam_total_seconds"] for row in values])),
                "mean_full_hybrid_seconds": float(np.mean([row["full_hybrid_seconds"] for row in values])),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partitions",
        nargs="+",
        default=["val", "test", "real"],
        choices=["train", "val", "test", "real"],
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-masks", action="store_true", default=True)
    args = parser.parse_args()

    device = resolve_device(args.device)
    predictor = build_predictor(device)
    runner = ProductAlignedSAM2(predictor, device=device)
    rows = selected_rows(set(args.partitions))
    if not rows:
        raise RuntimeError("El manifiesto no contiene filas para las particiones solicitadas.")

    results: list[dict] = []
    failures: list[dict] = []
    for index, row in enumerate(rows, start=1):
        partition = row["split"] if row["domain"] == "synthetic" else "real"
        image = read_required_image(ROOT / row["image"])
        reference = read_required_image(ROOT / row["ground_truth"], cv2.IMREAD_GRAYSCALE)
        wcs_info = (
            detect_wcs_for_rectified(image, row["sample_id"])
            if row["domain"] == "real"
            else {"status": "NOT_APPLICABLE"}
        )
        scenarios: list[tuple[str, list[tuple[int, int, int, int]], str, float]] = []
        oracle_boxes, oracle_source = ideal_boxes(row, reference)
        scenarios.append(("ideal", oracle_boxes, oracle_source, 0.0))
        try:
            boxes, _, prompt_seconds = operational_boxes(image, wcs_info=wcs_info)
            scenarios.append(
                ("operational", boxes, "classical_prompt_localizer", prompt_seconds)
            )
        except Exception as exc:
            failures.append(
                {
                    "sample_id": row["sample_id"],
                    "partition": partition,
                    "scenario": "operational",
                    "error": repr(exc),
                }
            )

        for scenario, boxes, source, prompt_seconds in scenarios:
            try:
                prediction, scores, timing, padded = runner.predict(image, boxes)
                metrics = sample_metrics(reference, prediction)
                result = {
                    "domain": row["domain"],
                    "partition": partition,
                    "sample_id": row["sample_id"],
                    "prompt_scenario": scenario,
                    "prompt_source": source,
                    "prompt_count": len(boxes),
                    "padded_prompt_count": len(padded),
                    "box_margin_fraction": 0.05,
                    "mask_selector_box_area": "padded_box",
                    "postprocess_min_component_area_px": 300,
                    "wcs_status": wcs_info.get("status", "NOT_APPLICABLE"),
                    **metrics,
                    "mean_predicted_score": float(np.mean(scores)),
                    "prompt_localization_seconds": prompt_seconds,
                    **timing,
                    "full_hybrid_seconds": prompt_seconds + timing["sam_total_seconds"],
                    "device": device,
                }
                results.append(result)
                if args.save_masks:
                    mask_dir = OUTPUT / "masks" / partition / scenario
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(mask_dir / f"{row['sample_id']}_sam2.png"), prediction)
            except Exception as exc:
                failures.append(
                    {
                        "sample_id": row["sample_id"],
                        "partition": partition,
                        "scenario": scenario,
                        "error": repr(exc),
                    }
                )
        print(f"SAM 2 protocolo [{index}/{len(rows)}] {row['sample_id']}")

    summary = summarize(results)
    METRICS.mkdir(parents=True, exist_ok=True)
    write_csv(METRICS / "sam2_prompt_source_metrics_v8.csv", results)
    write_csv(METRICS / "sam2_prompt_source_summary_v8.csv", summary)
    protocol = {
        "protocol_version": "sam2_prompt_source_v8",
        "model_family": "SAM 2",
        "model_variant": "Hiera Tiny",
        "pretrained": True,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "model_config": MODEL_CONFIG,
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "scenarios": {
            "ideal": (
                "Synthetic boxes come from generator metadata. Real boxes are extracted from "
                "the assisted reference and are an optimistic diagnostic, not an operational result."
            ),
            "operational": (
                "Boxes are recomputed for every image with the classical prompt localizer used "
                "by the software; SAM 2 provides the final mask."
            ),
        },
        "product_alignment": {
            "box_margin_fraction": 0.05,
            "mask_selection_area_basis": "padded box",
            "multimask_output": True,
            "postprocess": "3x3 elliptical closing and removal of components below 300 px",
        },
        "timing_scope": (
            "Single-run diagnostic timings include set_image/encoder, decoder and postprocess. "
            "Use segmentation_runtime_summary_v8.json for repeated comparable timings."
        ),
        "real_reference": (
            "Algorithmically assisted canonical mask: median of 13 rectified views, classical-"
            "derived prompt boxes, GrabCut, Canny and morphology. It is not an independent manual annotation."
        ),
        "n_results": len(results),
        "failures": failures,
        "interpretation_limits": [
            "The 13 real images are repeated captures of one physical sheet.",
            "Real ideal prompts use the same assisted reference later used for scoring.",
            "The current synthetic split must only be interpreted according to its separately audited identity integrity.",
        ],
    }
    (METRICS / "sam2_prompt_source_protocol_v8.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "failures": failures}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
