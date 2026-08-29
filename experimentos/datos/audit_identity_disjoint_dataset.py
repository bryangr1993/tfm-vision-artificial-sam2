"""Audita integridad y ausencia de contaminación del corpus asset_identity_v2."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "datos" / "manifiesto" / "datasets_asset_identity_v2.csv"
ASSET_SPLIT = ROOT / "datos" / "manifiesto" / "asset_split_v2.csv"
REPORT_JSON = ROOT / "resultados" / "metricas" / "dataset_identity_v2_quality.json"
REPORT_MD = ROOT / "documentacion" / "INFORME_CALIDAD_DATOS_IDV2.md"
EXPECTED_SPLITS = {"train": 24, "val": 12, "test": 12}
EXPECTED_FAMILIES = {"F1", "F2", "F3", "F4"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def intersections(values: dict[str, set[str]]) -> dict[str, list[str]]:
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    return {f"{left}__{right}": sorted(values[left] & values[right]) for left, right in pairs}


def audit() -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, evidence: object, severity: str = "critical") -> None:
        checks.append({"check": name, "passed": bool(passed), "severity": severity, "evidence": evidence})

    rows = read_csv(MANIFEST)
    assets = read_csv(ASSET_SPLIT)
    split_counts = Counter(row["split"] for row in rows)
    add("sheet_count_by_split", dict(split_counts) == EXPECTED_SPLITS, dict(split_counts))
    add("unique_sample_id", len({row["sample_id"] for row in rows}) == len(rows), len(rows))
    add("split_protocol_label", {row["split_protocol"] for row in rows} == {"asset_identity_disjoint"}, sorted({row["split_protocol"] for row in rows}))

    asset_ids_by_split: dict[str, set[str]] = defaultdict(set)
    asset_hashes_by_split: dict[str, set[str]] = defaultdict(set)
    source_files_by_split: dict[str, set[str]] = defaultdict(set)
    families_by_split: dict[str, set[str]] = defaultdict(set)
    for row in assets:
        split = row["split"]
        asset_ids_by_split[split].add(row["asset_id"])
        asset_hashes_by_split[split].add(row["asset_sha256"])
        source_files_by_split[split].add(row["source_file"].casefold())
        families_by_split[split].add(row["family"])
    id_cross = intersections(asset_ids_by_split)
    hash_cross = intersections(asset_hashes_by_split)
    file_cross = intersections(source_files_by_split)
    add("asset_id_cross_split", not any(id_cross.values()), id_cross)
    add("asset_hash_cross_split", not any(hash_cross.values()), hash_cross)
    add("source_filename_cross_split", not any(file_cross.values()), file_cross)
    add(
        "semantic_family_present_in_each_split",
        all(families_by_split[split] == EXPECTED_FAMILIES for split in EXPECTED_SPLITS),
        {key: sorted(value) for key, value in families_by_split.items()},
    )
    add(
        "semantic_families_intentionally_not_disjoint",
        all(families_by_split[split] == EXPECTED_FAMILIES for split in EXPECTED_SPLITS),
        "Las familias son estratos compartidos; la identidad de activo es la unidad disjunta.",
        severity="info",
    )
    add("asset_registry_unique_hash", len({row["asset_sha256"] for row in assets}) == 32, len({row["asset_sha256"] for row in assets}))

    image_hashes: dict[str, set[str]] = defaultdict(set)
    mask_hashes: dict[str, set[str]] = defaultdict(set)
    path_failures: list[str] = []
    dimension_failures: list[str] = []
    mask_failures: list[str] = []
    metadata_failures: list[str] = []
    usage_by_split: dict[str, set[str]] = defaultdict(set)
    cv_groups: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        split = row["split"]
        image_path = ROOT / row["image"]
        mask_path = ROOT / row["ground_truth"]
        instance_path = ROOT / row["instance_mask"]
        metadata_path = ROOT / row["metadata"]
        if not all(path.exists() for path in (image_path, mask_path, instance_path, metadata_path)):
            path_failures.append(row["sample_id"])
            continue
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        instances = cv2.imread(str(instance_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or instances is None or image.shape[:2] != (2970, 2100) or mask.shape != (2970, 2100) or instances.shape != (2970, 2100):
            dimension_failures.append(row["sample_id"])
        if set(np.unique(mask).tolist()) != {0, 255}:
            mask_failures.append(f"{row['sample_id']}:binary_values={np.unique(mask).tolist()}")
        if set(np.unique(instances).tolist()) != set(range(9)):
            mask_failures.append(f"{row['sample_id']}:instance_values={np.unique(instances).tolist()}")
        actual_image_hash = sha256(image_path)
        actual_mask_hash = sha256(mask_path)
        if actual_image_hash != row["image_sha256"] or actual_mask_hash != row["ground_truth_sha256"]:
            mask_failures.append(f"{row['sample_id']}:manifest_hash_mismatch")
        image_hashes[split].add(actual_image_hash)
        mask_hashes[split].add(actual_mask_hash)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        meta_ids = [instance["asset_id"] for instance in metadata["instances"]]
        manifest_ids = row["asset_ids"].split(";")
        if len(meta_ids) != 8 or meta_ids != manifest_ids:
            metadata_failures.append(f"{row['sample_id']}:instance_manifest_mismatch")
        unauthorized = sorted(set(meta_ids) - asset_ids_by_split[split])
        if unauthorized:
            metadata_failures.append(f"{row['sample_id']}:unauthorized={unauthorized}")
        usage_by_split[split].update(meta_ids)
        if split == "train":
            cv_groups[row["cv_group"]].update(meta_ids)

    add("all_paths_exist", not path_failures, path_failures)
    add("dimensions_2100x2970", not dimension_failures, dimension_failures)
    add("binary_and_instance_mask_validity", not mask_failures, mask_failures)
    add("metadata_asset_authorization", not metadata_failures, metadata_failures)
    image_cross = intersections(image_hashes)
    mask_cross = intersections(mask_hashes)
    add("image_hash_cross_split", not any(image_cross.values()), image_cross)
    add("mask_hash_cross_split", not any(mask_cross.values()), mask_cross)
    add(
        "all_allocated_assets_used",
        all(usage_by_split[split] == asset_ids_by_split[split] for split in EXPECTED_SPLITS),
        {split: sorted(usage_by_split[split]) for split in EXPECTED_SPLITS},
    )
    cv_intersections = intersections({
        "train": cv_groups.get("G1", set()),
        "val": cv_groups.get("G2", set()),
        "test": cv_groups.get("G3", set()) | cv_groups.get("G4", set()),
    })
    add(
        "train_cv_groups_asset_disjoint",
        not any(cv_intersections.values()) and set(cv_groups) == {"G1", "G2", "G3", "G4"},
        {group: sorted(ids) for group, ids in sorted(cv_groups.items())},
    )
    condition_coverage = {
        split: sorted({row["condition"] for row in rows if row["split"] == split})
        for split in EXPECTED_SPLITS
    }
    layout_coverage = {
        split: sorted({row["layout"] for row in rows if row["split"] == split})
        for split in EXPECTED_SPLITS
    }
    add("condition_coverage", all(set(values) == {"C1", "C2", "C3", "C4", "C5", "C6"} for values in condition_coverage.values()), condition_coverage)
    add("layout_coverage", all(set(values) == {"L1", "L2"} for values in layout_coverage.values()), layout_coverage)

    critical_failures = [item for item in checks if item["severity"] == "critical" and not item["passed"]]
    return {
        "dataset": "asset_identity_v2",
        "grain": "one full synthetic A4 sheet per manifest row; eight instances per sheet",
        "intended_use": "RF hyperparameter selection and locked synthetic evaluation",
        "passed": not critical_failures,
        "sheet_count": len(rows),
        "asset_count": len(assets),
        "split_counts": dict(split_counts),
        "checks": checks,
        "critical_failure_count": len(critical_failures),
    }


def write_reports(report: dict[str, object]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checks = report["checks"]
    lines = [
        "# Informe de calidad del dataset `asset_identity_v2`",
        "",
        f"**Resultado global:** {'APROBADO' if report['passed'] else 'NO APROBADO'}",
        "",
        "La unidad del manifiesto es una lámina A4 sintética completa con ocho instancias. El conjunto se usa para seleccionar hiperparámetros del Random Forest y realizar una evaluación sintética bloqueada.",
        "",
        "La familia semántica no es disjunta: F1--F4 se mantienen en todos los subconjuntos como estratos. La disyunción se exige sobre la identidad del diseño, el nombre del archivo y su SHA-256.",
        "",
        "| Comprobación | Estado | Evidencia |",
        "|---|---:|---|",
    ]
    for check in checks:
        evidence = json.dumps(check["evidence"], ensure_ascii=False)
        if len(evidence) > 240:
            evidence = evidence[:237] + "..."
        lines.append(f"| `{check['check']}` | {'OK' if check['passed'] else 'FALLO'} | `{evidence}` |")
    lines.extend(
        [
            "",
            "## Riesgo residual",
            "",
            "La independencia por identidad elimina la contaminación detectada en la versión anterior, pero no convierte 32 diseños en una muestra poblacional amplia. Los resultados describen generalización a ocho identidades sintéticas no vistas y deben contrastarse por separado con las capturas reales.",
            "",
            "## Repetición",
            "",
            "```powershell",
            "python experimentos/datos/generate_identity_disjoint_dataset.py --overwrite",
            "python experimentos/datos/audit_identity_disjoint_dataset.py",
            "```",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    report = audit()
    write_reports(report)
    print(json.dumps({key: report[key] for key in ("dataset", "passed", "sheet_count", "asset_count", "split_counts", "critical_failure_count")}, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
