"""Verifica la coherencia interna y calcula huellas del paquete experimental.

La salida distingue comprobaciones fallidas de limitaciones declaradas. Una
limitación documentada no invalida por sí sola un resultado reproducible.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "resultados" / "metricas"
OUTPUT = METRICS / "protocol_package_validation.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def read_csv(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    checks: list[dict] = []

    def check(name: str, passed: bool, evidence) -> None:
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})

    quality = read_json("resultados/metricas/dataset_identity_v2_quality.json")
    selection = read_json("resultados/metricas/sam2_idv2_prompt_selection.json")
    reference = read_json("resultados/metricas/real_reference_validation_v8.json")
    runtime = read_json("resultados/metricas/segmentation_runtime_summary_v8.json")
    wcs = read_json("resultados/metricas/wcs_contour_repeatability_summary_v8.json")
    wcs_stages = read_json("resultados/metricas/wcs_detection_stages_v8.json")
    integration = read_json("resultados/metricas/integration_batch_summary_v8.json")

    check("identity_v2_quality_passed", quality["passed"], quality["checks"])
    check(
        "identity_v2_split_counts",
        quality["split_counts"] == {"train": 24, "val": 12, "test": 12},
        quality["split_counts"],
    )
    check(
        "identity_v2_no_critical_failures",
        int(quality["critical_failure_count"]) == 0,
        quality["critical_failure_count"],
    )

    validation_table = selection["validation_table"]
    winner_by_score = max(validation_table, key=lambda row: float(row["selection_score"]))
    check(
        "sam_margin_candidates_complete",
        sorted(float(value) for value in selection["candidate_margins"])
        == [0.0, 0.03, 0.05, 0.1],
        selection["candidate_margins"],
    )
    check(
        "sam_margin_selected_on_validation_only",
        selection["selection_partition"] == "val only"
        and int(selection["validation_winner"]["n_validation_sheets"]) == 12,
        {
            "partition": selection["selection_partition"],
            "n_validation_sheets": selection["validation_winner"]["n_validation_sheets"],
        },
    )
    check(
        "sam_selected_margin_is_validation_winner",
        float(selection["selected_box_margin_fraction"]) == 0.05
        and float(winner_by_score["box_margin_fraction"]) == 0.05,
        {
            "selected": selection["selected_box_margin_fraction"],
            "winner_by_score": winner_by_score,
        },
    )
    scenarios = {
        (row["split"], row["prompt_scenario"]): row
        for row in selection["locked_scenario_summary"]
    }
    check(
        "sam_locked_test_evaluated_once_at_selected_margin",
        int(scenarios[("test", "operational")]["n_images"]) == 12
        and float(selection["selected_box_margin_fraction"]) == 0.05
        and "only after locking" in selection["test_access_policy"],
        {
            "n_test": scenarios[("test", "operational")]["n_images"],
            "policy": selection["test_access_policy"],
        },
    )
    manifest = ROOT / selection["manifest"]
    check(
        "sam_manifest_hash_matches",
        manifest.is_file() and sha256_file(manifest) == selection["manifest_sha256"],
        {"path": selection["manifest"], "recorded_sha256": selection["manifest_sha256"]},
    )

    check(
        "reference_is_explicitly_assisted",
        reference["algorithmic_assistance_from_classical_localizer"]
        and reference["overall_assessment"] == "needs_independent_manual_review"
        and reference["all_13_ground_truth_files_byte_identical_to_canonical"],
        {
            "assessment": reference["overall_assessment"],
            "copies": reference["ground_truth_copies"],
            "safe_description": reference["safe_description"],
        },
    )
    check(
        "rainbow_opening_supported_by_mask",
        bool(reference["rainbow_opening_evidence"]["supported"]),
        reference["rainbow_opening_evidence"],
    )

    runtime_rows = read_csv("resultados/metricas/segmentation_runtime_runs_v8.csv")
    counts = Counter(row["method"] for row in runtime_rows)
    expected_methods = {
        "vision_clasica",
        "random_forest_identity_v2",
        "sam2_operativo_hibrido",
    }
    check(
        "runtime_repeated_balanced_design",
        set(counts) == expected_methods and all(counts[method] == 20 for method in expected_methods),
        dict(counts),
    )
    check(
        "runtime_sam_includes_encoder",
        runtime["same_process_and_hardware"]
        and "set_image/encoder" in runtime["sam2_total_definition"],
        runtime["sam2_total_definition"],
    )
    rf_model = ROOT / runtime["rf_model"]
    rf_selection = ROOT / runtime["rf_selection"]
    check(
        "runtime_rf_identity_v2_contract",
        runtime["rf_feature_count"] == 19
        and float(runtime["rf_threshold"]) == 0.55
        and rf_model.is_file()
        and rf_selection.is_file()
        and sha256_file(rf_model) == runtime["rf_model_sha256"]
        and sha256_file(rf_selection) == runtime["rf_selection_sha256"],
        {
            "model": runtime["rf_model"],
            "feature_count": runtime["rf_feature_count"],
            "threshold": runtime["rf_threshold"],
        },
    )

    registration = wcs["registration"]
    check(
        "wcs_positive_and_negative_controls",
        registration["all_classifications_correct"]
        and registration["detected_expected_wcs"] == 10
        and registration["correctly_rejected_no_wcs"] == 3,
        registration,
    )
    stability_methods = {row["method"]: row for row in wcs["contour_stability"]}
    expected_stability_methods = {
        "Visión clásica",
        "Random Forest IDV2",
        "SAM 2 operativo",
    }
    check(
        "contour_stability_kept_separate_and_complete",
        set(stability_methods) == expected_stability_methods
        and all(
            int(row["n_captures"]) == 10 and int(row["n_capture_pairs"]) == 45
            for row in stability_methods.values()
        )
        and "different phenomena" in wcs["separation_note"],
        stability_methods,
    )
    current_detector = ROOT / "software" / "src" / "detect_wcs_l.py"
    check(
        "wcs_detection_stages_match_current_detector",
        wcs_stages["sample_id"] == "real_20"
        and wcs_stages["detector_result"]["status"] == "SUCCESS"
        and wcs_stages["provenance"]["detect_wcs_l_sha256"]
        == sha256_file(current_detector)
        and (ROOT / wcs_stages["figure"]).is_file(),
        {
            "sample_id": wcs_stages["sample_id"],
            "status": wcs_stages["detector_result"]["status"],
            "figure": wcs_stages["figure"],
            "detector_sha256": wcs_stages["provenance"]["detect_wcs_l_sha256"],
        },
    )

    integration_rows = read_csv("resultados/metricas/integration_batch_v8.csv")
    check(
        "integration_all_thirteen_captures_pass",
        integration["all_checks_pass"]
        and integration["n_captures"] == 13
        and len(integration_rows) == 13,
        {
            "all_checks_pass": integration["all_checks_pass"],
            "n_captures": integration["n_captures"],
        },
    )
    check(
        "integration_external_silhouette_policy",
        integration["captures_with_8_external_paths_and_no_holes"] == 13
        and all(
            int(row["outer_contour_count"]) == 8
            and int(row["cut_path_count"]) == 8
            and int(row["hole_path_count"]) == 0
            for row in integration_rows
        ),
        {
            "policy": integration["contour_export_policy"],
            "holes": integration["hole_paths_per_capture"],
        },
    )
    dxf_details = integration["dxf_details"]
    check(
        "integration_ten_structural_dxf_and_three_blocked",
        len(dxf_details) == 10
        and integration["correct_dxf_policy"] == 13
        and all(
            detail["structurally_valid"]
            and detail["outer_polyline_count"] == 8
            and detail["hole_polyline_count"] == 0
            and detail["polyline_count"] == 8
            for detail in dxf_details.values()
        )
        and sum(row["dxf_created"].lower() == "true" for row in integration_rows) == 10,
        dxf_details,
    )

    core_artifacts = [
        "datos/manifiesto/datasets_asset_identity_v2.csv",
        "resultados/metricas/dataset_identity_v2_quality.json",
        "resultados/metricas/sam2_idv2_prompt_selection.json",
        "resultados/metricas/sam2_idv2_margin_validation_metrics.csv",
        "resultados/metricas/sam2_idv2_margin_validation_summary.csv",
        "resultados/metricas/sam2_idv2_locked_test_prompt_scenarios.csv",
        "resultados/metricas/sam2_idv2_locked_test_operational_metrics.csv",
        "resultados/metricas/sam2_idv2_locked_real_prompt_scenarios.csv",
        "resultados/metricas/sam2_idv2_locked_scenario_summary.csv",
        "resultados/metricas/real_reference_validation_v8.json",
        "resultados/metricas/segmentation_runtime_runs_v8.csv",
        "resultados/metricas/segmentation_runtime_summary_v8.csv",
        "resultados/metricas/segmentation_runtime_summary_v8.json",
        "resultados/metricas/wcs_registration_per_capture_v8.csv",
        "resultados/metricas/contour_stability_per_capture_v8.csv",
        "resultados/metricas/contour_pairwise_stability_v8.csv",
        "resultados/metricas/wcs_contour_repeatability_summary_v8.json",
        "resultados/metricas/wcs_detection_stages_v8.json",
        "resultados/metricas/integration_batch_v8.csv",
        "resultados/metricas/integration_batch_summary_v8.json",
        "resultados/figuras_protocolo_v8/sam2_margin_selection_idv2.png",
        "resultados/figuras_protocolo_v8/sam2_prompt_source_comparison_v8.png",
        "resultados/figuras_protocolo_v8/segmentation_runtime_comparison_v8.png",
        "resultados/figuras_protocolo_v8/sam2_runtime_stages_v8.png",
        "resultados/figuras_protocolo_v8/wcs_origin_repeatability_v8.png",
        "resultados/figuras_protocolo_v8/wcs_detection_stages_v8.png",
        "resultados/figuras_protocolo_v8/contour_pairwise_stability_v8.png",
        "resultados/figuras_protocolo_v8/integration_batch_status_v8.png",
        "resultados/figuras_protocolo_v8/real_reference_provenance_v8.png",
        "resultados/referencia_real_validacion_v8/real_reference_manual_review.png",
        "resultados/referencia_real_validacion_v8/manual_review_checklist.md",
        "resultados/INFORME_PROTOCOLO_SAM2_WCS.md",
        "experimentos/PROTOCOLO_SAM2_WCS.md",
        "experimentos/sam2/protocol_common.py",
        "experimentos/sam2/select_prompt_margin_identity_v2.py",
        "experimentos/comparacion/benchmark_segmentation_runtime.py",
        "experimentos/geometria/evaluate_wcs_repeatability.py",
        "experimentos/geometria/generate_wcs_detection_stages.py",
        "experimentos/comparacion/run_integration_batch.py",
        "experimentos/comparacion/validate_real_reference.py",
        "experimentos/comparacion/generate_protocol_figures.py",
        "experimentos/comparacion/verify_protocol_package.py",
    ]
    dxf_paths = sorted((ROOT / "resultados" / "integracion_lote_v8" / "dxf").glob("*.dxf"))
    core_artifacts.extend(str(path.relative_to(ROOT)).replace("\\", "/") for path in dxf_paths)
    missing = [relative for relative in core_artifacts if not (ROOT / relative).is_file()]
    check("all_core_artifacts_present", not missing and len(dxf_paths) == 10, missing)

    artifacts = {}
    for relative in core_artifacts:
        path = ROOT / relative
        if path.is_file():
            artifacts[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    payload = {
        "status": "passed" if all(row["passed"] for row in checks) else "failed",
        "checks": checks,
        "limitations": [
            "SAM 2 and the other methods were benchmarked on CPU because CUDA was unavailable.",
            "The real reference is an assisted canonical preannotation and still requires independent manual review.",
            "Real agreement covers thirteen captures of one physical sheet, not thirteen independent designs.",
            "WCS values measure capture-to-capture geometric dispersion without independent metrological ground truth.",
            "DXF validation is a structural round-trip and does not replace RDWorks import or a physical laser cut.",
        ],
        "artifact_count": len(artifacts),
        "artifacts": dict(sorted(artifacts.items())),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
