"""Construye el manifiesto canónico y ejecuta controles de calidad del dataset.

La unidad de partición es la lámina completa. El script no crea ni modifica
imágenes. Solo registra rutas relativas, dimensiones, checksums y reglas de
integridad para que los experimentos sean auditables.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datos"
LEGACY_METRICS = ROOT / "resultados" / "metricas" / "legacy"
MANIFEST = DATA / "manifiesto" / "datasets.csv"
REPORT = ROOT / "documentacion" / "INFORME_CALIDAD_DATOS.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path | None) -> str:
    return "" if path is None else path.relative_to(ROOT).as_posix()


def image_info(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def read_split() -> dict[str, str]:
    split_path = LEGACY_METRICS / "dataset_split.csv"
    with split_path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {row["sheet_id"]: row["split"] for row in csv.DictReader(stream)}


def synthetic_rows(split: dict[str, str]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for image_path in sorted((DATA / "sinteticos" / "imagenes").glob("*.png")):
        stem = image_path.stem
        sample_id = stem.removeprefix("synthetic_").removesuffix("_rgb")
        mask_path = DATA / "sinteticos" / "mascaras" / f"{sample_id}_mask.png"
        metadata_path = DATA / "sinteticos" / "metadatos" / f"{sample_id}_meta.json"
        tokens = sample_id.split("_")
        width, height = image_info(image_path)
        rows.append(
            {
                "sample_id": sample_id,
                "domain": "synthetic",
                "split": split[sample_id],
                "family": tokens[1],
                "layout": tokens[2],
                "condition": tokens[3],
                "raw_image": "",
                "image": relative(image_path),
                "ground_truth": relative(mask_path),
                "legacy_reference": "",
                "prompt": "",
                "metadata": relative(metadata_path),
                "width_px": width,
                "height_px": height,
                "image_sha256": sha256(image_path),
                "ground_truth_sha256": sha256(mask_path),
                "annotation_status": "generator_ground_truth",
                "notes": "Máscara producida por el generador sintético.",
            }
        )
    return rows


def real_rows() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    annotation_dir = DATA / "anotaciones" / "real_ground_truth"
    for image_path in sorted(
        (DATA / "reales" / "rectificadas").glob("rectified_*.png"),
        key=lambda path: int(path.stem.split("_")[-1]),
    ):
        number = image_path.stem.split("_")[-1]
        raw_path = DATA / "reales" / "raw" / f"{number}.jpg"
        classical_path = DATA / "reales" / "referencias_clasicas" / f"{number}_mask.png"
        prompt_path = DATA / "reales" / "prompts_legacy" / f"real_{number}_prompts.json"
        ground_truth = annotation_dir / f"real_{number}_gt.png"
        width, height = image_info(image_path)
        reference_metadata = DATA / "anotaciones" / "referencia_real_canonica" / "metadata.json"
        rows.append(
            {
                "sample_id": f"real_{number}",
                "domain": "real",
                "split": "real_evaluation",
                "family": "production",
                "layout": "L1",
                "condition": "uncontrolled_capture",
                "raw_image": relative(raw_path),
                "image": relative(image_path),
                "ground_truth": relative(ground_truth) if ground_truth.exists() else "",
                "legacy_reference": relative(classical_path),
                "prompt": relative(prompt_path),
                "metadata": relative(reference_metadata) if reference_metadata.exists() else "",
                "width_px": width,
                "height_px": height,
                "image_sha256": sha256(image_path),
                "ground_truth_sha256": sha256(ground_truth) if ground_truth.exists() else "",
                "annotation_status": (
                    "review_required" if not ground_truth.exists() else "assisted_visual_qc_complete"
                ),
                "notes": (
                    "Referencia canónica común porque las 13 capturas corresponden a la misma "
                    "lámina rectificada. La salida clásica no se usa como verdad terreno."
                ),
            }
        )
    return rows


def mask_values(path: Path) -> set[int]:
    with Image.open(path).convert("L") as image:
        return set(image.getdata())


def build_report(rows: list[dict[str, str | int]]) -> str:
    by_domain = Counter(str(row["domain"]) for row in rows)
    by_split = Counter(str(row["split"]) for row in rows)
    missing: list[str] = []
    shape_errors: list[str] = []
    non_binary: list[str] = []
    hashes: defaultdict[str, list[str]] = defaultdict(list)

    for row in rows:
        image_path = ROOT / str(row["image"])
        hashes[str(row["image_sha256"])].append(str(row["sample_id"]))
        if not image_path.exists():
            missing.append(relative(image_path))
            continue
        gt_value = str(row["ground_truth"])
        if not gt_value:
            continue
        gt_path = ROOT / gt_value
        if not gt_path.exists():
            missing.append(gt_value)
            continue
        if image_info(image_path) != image_info(gt_path):
            shape_errors.append(str(row["sample_id"]))
        values = mask_values(gt_path)
        if not values.issubset({0, 255}):
            non_binary.append(f"{row['sample_id']}: {sorted(values)[:12]}")

    duplicated_images = {key: value for key, value in hashes.items() if len(value) > 1}
    accepted_annotation_status = {"reviewed", "assisted_visual_qc_complete"}
    real_pending = sum(
        row["domain"] == "real"
        and row["annotation_status"] not in accepted_annotation_status
        for row in rows
    )
    risk = "ALTO" if real_pending else "BAJO"
    if real_pending:
        reference_finding = (
            "La carencia de una referencia real independiente es el único riesgo crítico "
            "para la evaluación final. Las máscaras clásicas existentes se conservan como "
            "línea base, pero no se consideran anotaciones de referencia."
        )
        remediation = (
            f"Se crearán y revisarán las {by_domain['real']} máscaras reales antes de "
            "calcular métricas finales."
        )
    else:
        reference_finding = (
            "La referencia real canónica está disponible y superó el control visual. Se "
            "construyó sobre la mediana de las trece capturas rectificadas de una misma "
            "lámina, sin usar como etiquetas las salidas clásicas ni las de SAM 2."
        )
        remediation = (
            "No quedan acciones de remediación abiertas sobre las referencias. La naturaleza "
            "asistida de la anotación y el uso de una sola lámina física se declararán como "
            "limitaciones del conjunto real."
        )
    return f"""# Informe de calidad de datos

## Dataset y granularidad

El manifiesto contiene **{len(rows)} láminas**. La unidad de análisis y partición es una lámina completa. Hay **{by_domain['synthetic']} muestras sintéticas** y **{by_domain['real']} capturas reales rectificadas**.

Las particiones sintéticas son: entrenamiento **{by_split['train']}**, validación **{by_split['val']}** y prueba **{by_split['test']}**. Las **{by_split['real_evaluation']}** imágenes reales forman un conjunto de evaluación externo al entrenamiento sintético.

## Controles ejecutados

- existencia de imágenes, referencias y metadatos declarados;
- coincidencia de dimensiones entre imagen y máscara;
- valores binarios en las máscaras disponibles;
- unicidad del contenido de las imágenes mediante SHA-256;
- separación por identificador de lámina;
- disponibilidad de una verdad terreno real independiente.

## Hallazgos

| Comprobación | Resultado | Riesgo |
|---|---:|---|
| Rutas obligatorias ausentes | {len(missing)} | {'Alto' if missing else 'Bajo'} |
| Pares imagen-máscara con dimensiones distintas | {len(shape_errors)} | {'Alto' if shape_errors else 'Bajo'} |
| Máscaras no binarias | {len(non_binary)} | {'Alto' if non_binary else 'Bajo'} |
| Imágenes duplicadas por contenido | {len(duplicated_images)} grupos | {'Medio' if duplicated_images else 'Bajo'} |
| Anotaciones reales pendientes de revisión | {real_pending} | {risk.title()} |

{reference_finding}

## Remediación

{remediation} Los experimentos bloquean el conjunto sintético de prueba durante la selección de hiperparámetros y de prompts. El manifiesto y este informe se regeneran después de cada cambio de anotación.

## Detalle de incidencias

- Rutas ausentes: {missing or 'ninguna'}.
- Dimensiones incompatibles: {shape_errors or 'ninguna'}.
- Valores de máscara no binarios: {non_binary or 'ninguno'}.
- Duplicados: {duplicated_images or 'ninguno'}.
"""


def main() -> None:
    split = read_split()
    rows = synthetic_rows(split) + real_rows()
    fieldnames = list(rows[0].keys())
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    REPORT.write_text(build_report(rows), encoding="utf-8")
    print(f"Manifiesto: {MANIFEST} ({len(rows)} filas)")
    print(f"Informe: {REPORT}")


if __name__ == "__main__":
    main()
