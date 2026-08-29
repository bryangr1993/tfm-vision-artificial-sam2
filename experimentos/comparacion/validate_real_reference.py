"""Audita la procedencia y la topología de la referencia canónica real.

El script no modifica la anotación. Produce una validación reproducible y una
lámina para revisión humana. La anotación existente es asistida por algoritmos y
por cajas procedentes del localizador clásico; por ello no se describe como una
referencia manual independiente.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "datos" / "anotaciones" / "referencia_real_canonica"
GT_DIR = ROOT / "datos" / "anotaciones" / "real_ground_truth"
REAL_DIR = ROOT / "datos" / "reales"
METRICS = ROOT / "resultados" / "metricas"
OUTPUT = ROOT / "resultados" / "referencia_real_validacion_v8"
FIGURES = ROOT / "resultados" / "figuras_protocolo_v8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contour_topology(mask: np.ndarray) -> tuple[int, int]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0, 0
    parents = hierarchy[0, :, 3]
    external = int(np.count_nonzero(parents == -1))
    holes = 0
    for index in range(len(contours)):
        depth = 0
        parent = int(parents[index])
        while parent >= 0:
            depth += 1
            parent = int(parents[parent])
        if depth % 2 == 1:
            holes += 1
    return external, holes


def rainbow_opening_evidence(mask: np.ndarray) -> dict:
    """Comprueba la concavidad abierta del arcoíris sin confundirla con un hueco cerrado."""

    count, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    candidates = []
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        cx, cy = map(float, centroids[label])
        if cx > mask.shape[1] / 2 and cy < mask.shape[0] / 3:
            candidates.append((area, label, x, y, width, height))
    if not candidates:
        return {"supported": False, "reason": "rainbow_component_not_found"}
    _, label, x, y, width, height = max(candidates)
    # La abertura es una concavidad conectada al fondo exterior por abajo, no
    # un agujero topológico cerrado. Se inspecciona la franja central inferior.
    x1, x2 = int(x + 0.35 * width), int(x + 0.65 * width)
    y1, y2 = int(y + 0.35 * height), y + height
    background = (mask == 0).astype(np.uint8)
    _, background_labels = cv2.connectedComponents(background)
    exterior_label = int(background_labels[0, 0])
    exterior_background = background_labels == exterior_label
    fraction = float(np.mean(exterior_background[y1:y2, x1:x2]))
    return {
        "supported": fraction >= 0.25,
        "interpretation": "border-connected concavity, not a closed topological hole",
        "exterior_background_fraction_in_central_lower_roi": fraction,
        "component_label": int(label),
        "component_bbox_px": [x, y, width, height],
        "inspection_roi_px": [x1, y1, x2, y2],
    }


def dice_iou(reference: np.ndarray, comparison: np.ndarray) -> tuple[float, float]:
    left = reference > 0
    right = comparison > 0
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    denominator = int(np.count_nonzero(left) + np.count_nonzero(right))
    return (
        2.0 * intersection / denominator if denominator else 1.0,
        intersection / union if union else 1.0,
    )


def make_review_figure(median: np.ndarray, reference: np.ndarray) -> None:
    overlay = median.copy()
    contours, _ = cv2.findContours(reference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, contours, -1, (0, 170, 0), 5)
    rgb_median = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)
    rgb_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 6.2))
    axes[0].imshow(rgb_median)
    axes[0].set_title("Mediana de 13 capturas")
    axes[1].imshow(reference, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Máscara canónica asistida")
    axes[2].imshow(rgb_overlay)
    axes[2].set_title("Contornos exteriores")
    for axis in axes:
        axis.axis("off")
    fig.suptitle("Referencia real sometida a revisión humana", fontsize=13)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.88, bottom=0.09, wspace=0.06)
    fig.text(
        0.5,
        0.025,
        "Cajas del localizador clásico + GrabCut + Canny + morfología. No es una anotación manual independiente.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "real_reference_manual_review.png", dpi=220, facecolor="white")
    fig.savefig(FIGURES / "real_reference_provenance_v8.png", dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    canonical_path = REFERENCE_DIR / "referencia_canonica_real.png"
    median_path = REFERENCE_DIR / "imagen_mediana_real.png"
    metadata_path = REFERENCE_DIR / "metadata.json"
    reference = cv2.imread(str(canonical_path), cv2.IMREAD_GRAYSCALE)
    median = cv2.imread(str(median_path), cv2.IMREAD_COLOR)
    if reference is None or median is None:
        raise FileNotFoundError("No se encontró la referencia canónica o la imagen mediana.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    canonical_hash = sha256_file(canonical_path)
    gt_paths = sorted(GT_DIR.glob("real_*_gt.png"), key=lambda path: int(path.stem.split("_")[1]))
    gt_hashes = {path.name: sha256_file(path) for path in gt_paths}
    all_gt_identical = len(gt_paths) == 13 and set(gt_hashes.values()) == {canonical_hash}
    external_components, holes = contour_topology(reference)
    rainbow_opening = rainbow_opening_evidence(reference)

    prompt_path = REAL_DIR / "prompts_legacy" / "real_13_prompts.json"
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))["prompts"]
    prompt_sources = sorted({prompt.get("source", "unknown") for prompt in prompts})
    classical_comparisons = []
    for path in sorted((REAL_DIR / "referencias_clasicas").glob("*_mask.png"), key=lambda item: int(item.stem.split("_")[0])):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        dice, iou = dice_iou(reference, mask)
        classical_comparisons.append({"capture": path.stem.split("_")[0], "dice": dice, "iou": iou})

    contradictions = []
    qc_checks = metadata.get("visual_qc", {}).get("checks", [])
    if "rainbow inner opening preserved" in qc_checks and not rainbow_opening["supported"]:
        contradictions.append(
            {
                "severity": "high",
                "claim": "rainbow inner opening preserved",
                "evidence": "The expected border-connected concavity could not be demonstrated in the binary mask.",
            }
        )
    if metadata.get("uses_classical_masks_as_labels") is False and "classic_bbox" in prompt_sources:
        contradictions.append(
            {
                "severity": "medium",
                "claim": "uses_classical_masks_as_labels: false",
                "evidence": (
                    "No classical mask pixels are copied as labels, but classical-derived boxes delimit all eight annotation regions. "
                    "The statement is technically narrow and must not be paraphrased as methodological independence."
                ),
            }
        )

    payload = {
        "protocol_version": "real_reference_validation_v8",
        "overall_assessment": "needs_independent_manual_review",
        "canonical_reference": str(canonical_path.relative_to(ROOT)).replace("\\", "/"),
        "canonical_sha256": canonical_hash,
        "ground_truth_copies": len(gt_paths),
        "all_13_ground_truth_files_byte_identical_to_canonical": all_gt_identical,
        "experimental_unit": "one physical sheet observed in 13 captures",
        "annotation_pipeline": [
            "pixelwise median of 13 rectified captures",
            "eight prompt boxes derived from the classical coarse localizer on real_13",
            "box-initialized GrabCut",
            "Canny edge closure and morphology",
            "external-contour filling",
            "recorded visual quality check without a saved manual pixel-edit audit trail",
        ],
        "direct_use_of_classical_mask_pixels_as_labels": False,
        "algorithmic_assistance_from_classical_localizer": True,
        "prompt_sources": prompt_sources,
        "external_components": external_components,
        "closed_holes_in_binary_reference": holes,
        "rainbow_opening_evidence": rainbow_opening,
        "foreground_fraction": float(np.count_nonzero(reference) / reference.size),
        "mean_iou_with_classical_capture_masks": float(
            np.mean([row["iou"] for row in classical_comparisons])
        ),
        "std_iou_with_classical_capture_masks": float(
            np.std([row["iou"] for row in classical_comparisons], ddof=1)
        ),
        "metadata_contradictions": contradictions,
        "safe_description": (
            "Algorithmically assisted canonical reference based on 13 rectified views, "
            "classical-derived boxes, GrabCut, edge detection and morphology; reviewed visually by one author."
        ),
        "unsafe_descriptions": [
            "independent manual ground truth",
            "reference independent of classical vision",
            "gold-standard annotation",
        ],
        "required_human_action": (
            "Inspect and, if necessary, edit the binary mask against the median image, with special attention to the rainbow opening, "
            "thin butterfly appendages, the dinosaur mouth and the vehicle wheels. Save an audit note identifying the reviewer and changes."
        ),
        "ground_truth_hashes": gt_hashes,
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (METRICS / "real_reference_validation_v8.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT / "manual_review_checklist.md").write_text(
        "# Revisión manual de la referencia real\n\n"
        "La máscara actual es una preanotación algorítmica, no una verdad terreno manual independiente.\n\n"
        "- Comparar la máscara binaria con la imagen mediana a escala 100 %.\n"
        "- Revisar la abertura interior del arcoíris.\n"
        "- Revisar antenas y apéndices finos de la mariposa.\n"
        "- Revisar la boca y las extremidades del dinosaurio.\n"
        "- Revisar ruedas y huecos del camión.\n"
        "- Confirmar que marcadores y marca WCS queden excluidos.\n"
        "- Registrar revisor, fecha y cambios manuales realizados.\n",
        encoding="utf-8",
    )
    make_review_figure(median, reference)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
