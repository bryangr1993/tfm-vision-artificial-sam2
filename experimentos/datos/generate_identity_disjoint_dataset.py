"""Genera 48 láminas sintéticas con partición previa por identidad de activo.

Protocolo ``asset_identity_disjoint``:

* cada familia semántica F1--F4 aporta cuatro activos a entrenamiento, dos a
  validación y dos a prueba;
* un activo (nombre y SHA-256) solo puede pertenecer a un subconjunto;
* las familias semánticas se conservan en los tres subconjuntos y, por tanto,
  deliberadamente no son disjuntas;
* los 16 activos de entrenamiento se organizan en cuatro cohortes internas.
  Cada cohorte contiene un activo por familia y sirve como grupo de GroupKFold.

El script nunca consulta el manifiesto histórico por lámina. Primero bloquea
la asignación de los 32 activos y después compone las imágenes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "datos" / "fuentes_toppers" / "alpha"
ASSET_REGISTRY = ROOT / "datos" / "manifiesto" / "asset_registry_v2.csv"
ASSET_SPLIT = ROOT / "datos" / "manifiesto" / "asset_split_v2.csv"
DATASET_MANIFEST = ROOT / "datos" / "manifiesto" / "datasets_asset_identity_v2.csv"
OUTPUT = ROOT / "datos" / "sinteticos_identidad_v2"
CONFIG_PATH = ROOT / "datos" / "manifiesto" / "dataset_identity_v2_config.json"
QC_PATH = ROOT / "resultados" / "figuras" / "dataset_idv2_split_contact.png"

WIDTH = 2100
HEIGHT = 2970
SPLIT_SEED = 20260827
GENERATION_SEED = 42027
SPLIT_PROTOCOL = "asset_identity_disjoint"
CONDITIONS = ("C1", "C2", "C3", "C4", "C5", "C6")
LAYOUTS = ("L1", "L2")
GRID_CENTERS = (
    (600, 450),
    (1500, 450),
    (600, 1100),
    (1500, 1100),
    (600, 1750),
    (1500, 1750),
    (600, 2400),
    (1500, 2400),
)


@dataclass(frozen=True)
class Asset:
    asset_id: str
    source_file: str
    family: str
    sha256: str
    split: str = ""
    cv_group: str = ""

    @property
    def path(self) -> Path:
        return ASSET_DIR / self.source_file


@dataclass(frozen=True)
class Recipe:
    sample_id: str
    split: str
    layout: str
    condition: str
    cv_group: str
    seed: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_registry() -> list[Asset]:
    if not ASSET_REGISTRY.exists():
        raise FileNotFoundError(
            f"Falta {ASSET_REGISTRY}. Ejecute primero prepare_source_assets.py."
        )
    with ASSET_REGISTRY.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assets = [
        Asset(
            asset_id=row["asset_id"],
            source_file=row["source_file"],
            family=row["family"],
            sha256=row["sha256"],
        )
        for row in rows
    ]
    if len(assets) != 32 or len({item.asset_id for item in assets}) != 32:
        raise RuntimeError("El registro debe contener 32 identidades únicas.")
    for asset in assets:
        if not asset.path.exists():
            raise FileNotFoundError(asset.path)
        if file_sha256(asset.path) != asset.sha256:
            raise RuntimeError(f"Hash distinto del registro: {asset.path}")
    return assets


def split_assets(assets: list[Asset]) -> list[Asset]:
    """Asigna 4/2/2 activos por familia con una semilla fija."""

    assigned: list[Asset] = []
    for family_index, family in enumerate(("F1", "F2", "F3", "F4"), start=1):
        members = sorted((a for a in assets if a.family == family), key=lambda a: a.asset_id)
        if len(members) != 8:
            raise RuntimeError(f"{family} debe contener ocho activos; contiene {len(members)}.")
        random.Random(SPLIT_SEED + family_index).shuffle(members)
        for position, asset in enumerate(members):
            if position < 4:
                split = "train"
                cv_group = f"G{position + 1}"
            elif position < 6:
                split = "val"
                cv_group = ""
            else:
                split = "test"
                cv_group = ""
            assigned.append(
                Asset(
                    asset_id=asset.asset_id,
                    source_file=asset.source_file,
                    family=asset.family,
                    sha256=asset.sha256,
                    split=split,
                    cv_group=cv_group,
                )
            )
    return sorted(assigned, key=lambda a: a.asset_id)


def write_asset_split(assets: list[Asset]) -> None:
    ASSET_SPLIT.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "asset_id",
        "source_file",
        "family",
        "split",
        "cv_group",
        "asset_sha256",
        "split_seed",
        "split_protocol",
    )
    with ASSET_SPLIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for asset in assets:
            writer.writerow(
                {
                    "asset_id": asset.asset_id,
                    "source_file": asset.source_file,
                    "family": asset.family,
                    "split": asset.split,
                    "cv_group": asset.cv_group,
                    "asset_sha256": asset.sha256,
                    "split_seed": SPLIT_SEED,
                    "split_protocol": SPLIT_PROTOCOL,
                }
            )


def build_recipes() -> list[Recipe]:
    recipes: list[Recipe] = []
    ordinal = 0
    for group_index in range(1, 5):
        for condition_index, condition in enumerate(CONDITIONS, start=1):
            ordinal += 1
            layout = LAYOUTS[(group_index + condition_index) % 2]
            recipes.append(
                Recipe(
                    sample_id=f"idv2_train_G{group_index}_{layout}_{condition}",
                    split="train",
                    layout=layout,
                    condition=condition,
                    cv_group=f"G{group_index}",
                    seed=GENERATION_SEED + ordinal,
                )
            )
    for split in ("val", "test"):
        for layout in LAYOUTS:
            for condition in CONDITIONS:
                ordinal += 1
                recipes.append(
                    Recipe(
                        sample_id=f"idv2_{split}_{layout}_{condition}",
                        split=split,
                        layout=layout,
                        condition=condition,
                        cv_group="",
                        seed=GENERATION_SEED + ordinal,
                    )
                )
    assert len(recipes) == 48
    return recipes


def assets_for_recipe(recipe: Recipe, assets: list[Asset]) -> list[Asset]:
    if recipe.split == "train":
        base = [
            item
            for item in assets
            if item.split == "train" and item.cv_group == recipe.cv_group
        ]
        if len(base) != 4 or {item.family for item in base} != {"F1", "F2", "F3", "F4"}:
            raise RuntimeError(f"Cohorte inválida: {recipe.cv_group}")
        selected = base * 2
    else:
        selected = [item for item in assets if item.split == recipe.split]
        if len(selected) != 8:
            raise RuntimeError(f"El split {recipe.split} no contiene ocho activos.")
    random.Random(recipe.seed).shuffle(selected)
    return selected


def threshold_alpha(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    red, green, blue, alpha = rgba.split()
    binary_alpha = alpha.point(lambda value: 255 if value >= 35 else 0)
    return Image.merge("RGBA", (red, green, blue, binary_alpha))


def transform_asset(
    asset: Asset,
    layout: str,
    center: tuple[int, int],
    rng: random.Random,
) -> tuple[Image.Image, dict[str, float | int | str]]:
    image = threshold_alpha(Image.open(asset.path))
    source_width, source_height = image.size
    if layout == "L1":
        scale_multiplier = 1.0
        angle = 0.0
        dx = 0.0
        dy = 0.0
    else:
        scale_multiplier = rng.uniform(0.88, 1.12)
        angle = rng.uniform(-18.0, 18.0)
        dx = rng.uniform(-35.0, 35.0)
        dy = rng.uniform(-35.0, 35.0)
    scale = (430.0 * scale_multiplier) / max(source_width, source_height)
    resized = image.resize(
        (max(1, round(source_width * scale)), max(1, round(source_height * scale))),
        Image.Resampling.LANCZOS,
    )
    resized = threshold_alpha(resized)
    if angle:
        resized = resized.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        resized = threshold_alpha(resized)
    center_x = round(center[0] + dx)
    center_y = round(center[1] + dy)
    left = center_x - resized.width // 2
    top = center_y - resized.height // 2
    return resized, {
        "asset_id": asset.asset_id,
        "source_file": asset.source_file,
        "asset_sha256": asset.sha256,
        "family": asset.family,
        "center_x": center_x,
        "center_y": center_y,
        "scale": scale,
        "rotation_degrees": angle,
        "translation_x_px": dx,
        "translation_y_px": dy,
        "paste_left": left,
        "paste_top": top,
        "rendered_width": resized.width,
        "rendered_height": resized.height,
    }


def deformation_maps() -> tuple[np.ndarray, np.ndarray]:
    y_coords, x_coords = np.indices((HEIGHT, WIDTH), dtype=np.float32)
    map_x = x_coords + 12.0 * np.sin(2.0 * np.pi * y_coords / 600.0)
    map_y = y_coords + 12.0 * np.cos(2.0 * np.pi * x_coords / 600.0)
    return map_x, map_y


def apply_condition(
    image_bgr: np.ndarray,
    binary_mask: np.ndarray,
    instance_mask: np.ndarray,
    condition: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if condition == "C1":
        return image_bgr.copy(), binary_mask.copy(), instance_mask.copy()
    if condition in {"C2", "C3"}:
        start, stop = (0.72, 1.0) if condition == "C2" else (1.0, 0.72)
        gradient = np.linspace(start, stop, HEIGHT, dtype=np.float32)[:, None, None]
        output = np.clip(image_bgr.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
        return output, binary_mask.copy(), instance_mask.copy()
    if condition == "C4":
        output = np.clip(image_bgr.astype(np.float32) * 0.7 + 35.0, 0, 255).astype(np.uint8)
        return output, binary_mask.copy(), instance_mask.copy()
    if condition == "C5":
        rng = np.random.default_rng(seed)
        blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
        noise = rng.normal(0.0, 7.5, image_bgr.shape).astype(np.float32)
        output = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return output, binary_mask.copy(), instance_mask.copy()
    if condition == "C6":
        map_x, map_y = deformation_maps()
        output = cv2.remap(
            image_bgr,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        warped_binary = cv2.remap(
            binary_mask,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_instances = cv2.remap(
            instance_mask,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return output, warped_binary, warped_instances
    raise ValueError(condition)


def instance_geometry(mask: np.ndarray, instance_id: int) -> tuple[list[int], list[list[int]]]:
    binary = np.where(mask == instance_id, 255, 0).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if not len(xs):
        raise RuntimeError(f"La instancia {instance_id} desapareció durante la composición.")
    bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(int).tolist()
    return bbox, contour


def generate_sheet(recipe: Recipe, assets: list[Asset], overwrite: bool) -> dict[str, str | int]:
    image_dir = OUTPUT / "imagenes"
    mask_dir = OUTPUT / "mascaras"
    instance_dir = OUTPUT / "instancias"
    metadata_dir = OUTPUT / "metadatos"
    contour_dir = OUTPUT / "contornos"
    for directory in (image_dir, mask_dir, instance_dir, metadata_dir, contour_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / f"{recipe.sample_id}_rgb.png"
    mask_path = mask_dir / f"{recipe.sample_id}_mask.png"
    instance_path = instance_dir / f"{recipe.sample_id}_instances.png"
    metadata_path = metadata_dir / f"{recipe.sample_id}_meta.json"
    contour_path = contour_dir / f"{recipe.sample_id}_contours.json"
    targets = (image_path, mask_path, instance_path, metadata_path, contour_path)
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError(
            f"Ya existen salidas de {recipe.sample_id}; use --overwrite para una regeneración determinista."
        )

    canvas = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    binary_canvas = Image.new("L", (WIDTH, HEIGHT), 0)
    instance_canvas = Image.new("L", (WIDTH, HEIGHT), 0)
    rng = random.Random(recipe.seed)
    placement_rows: list[dict[str, float | int | str]] = []
    for instance_id, (asset, center) in enumerate(zip(assets, GRID_CENTERS), start=1):
        transformed, placement = transform_asset(asset, recipe.layout, center, rng)
        alpha = transformed.getchannel("A")
        left = int(placement["paste_left"])
        top = int(placement["paste_top"])
        canvas.paste(transformed, (left, top), alpha)
        binary_canvas.paste(255, (left, top), alpha)
        instance_canvas.paste(instance_id, (left, top), alpha)
        placement_rows.append({"instance_id": instance_id, **placement})

    image_bgr = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    binary = np.asarray(binary_canvas)
    instances = np.asarray(instance_canvas)
    image_bgr, binary, instances = apply_condition(
        image_bgr, binary, instances, recipe.condition, recipe.seed
    )

    contours: list[dict[str, object]] = []
    for placement in placement_rows:
        instance_id = int(placement["instance_id"])
        bbox, contour = instance_geometry(instances, instance_id)
        placement["bbox_xywh"] = bbox
        contours.append({"instance_id": instance_id, "contour_xy": contour})

    if not cv2.imwrite(str(image_path), image_bgr):
        raise IOError(image_path)
    if not cv2.imwrite(str(mask_path), binary):
        raise IOError(mask_path)
    if not cv2.imwrite(str(instance_path), instances):
        raise IOError(instance_path)
    metadata = {
        "sample_id": recipe.sample_id,
        "split": recipe.split,
        "split_protocol": SPLIT_PROTOCOL,
        "cv_group": recipe.cv_group,
        "layout": recipe.layout,
        "condition": recipe.condition,
        "generator_seed": recipe.seed,
        "canvas": {"width_px": WIDTH, "height_px": HEIGHT, "px_per_mm": 10.0},
        "condition_parameters": {
            "C2_C3_shadow_factor_range": [0.72, 1.0],
            "C4_contrast_multiplier": 0.7,
            "C4_brightness_offset": 35.0,
            "C5_gaussian_kernel": [5, 5],
            "C5_noise_sigma": 7.5,
            "C6_amplitude_px": 12.0,
            "C6_period_px": 600.0,
        },
        "instances": placement_rows,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    contour_path.write_text(
        json.dumps({"sample_id": recipe.sample_id, "contours": contours}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "sample_id": recipe.sample_id,
        "domain": "synthetic",
        "split": recipe.split,
        "split_protocol": SPLIT_PROTOCOL,
        "family": "mixed_F1_F2_F3_F4",
        "cv_group": recipe.cv_group,
        "layout": recipe.layout,
        "condition": recipe.condition,
        "generator_seed": recipe.seed,
        "raw_image": "",
        "image": image_path.relative_to(ROOT).as_posix(),
        "ground_truth": mask_path.relative_to(ROOT).as_posix(),
        "instance_mask": instance_path.relative_to(ROOT).as_posix(),
        "legacy_reference": "",
        "prompt": "",
        "metadata": metadata_path.relative_to(ROOT).as_posix(),
        "width_px": WIDTH,
        "height_px": HEIGHT,
        "image_sha256": file_sha256(image_path),
        "ground_truth_sha256": file_sha256(mask_path),
        "asset_ids": ";".join(item.asset_id for item in assets),
        "asset_sha256s": ";".join(item.sha256 for item in assets),
        "annotation_status": "generator_ground_truth_asset_identity_disjoint",
        "notes": "Familias semánticas compartidas; identidades, archivos y hashes de activo disjuntos entre splits.",
    }


def write_manifest(rows: list[dict[str, str | int]]) -> None:
    with DATASET_MANIFEST.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(rows: list[dict[str, str | int]]) -> None:
    thumb_width, thumb_height = 210, 297
    columns = 6
    selected: list[dict[str, str | int]] = []
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        selected.extend(split_rows[:6])
    output = Image.new("RGB", (columns * thumb_width, 3 * (thumb_height + 28)), "white")
    draw = ImageDraw.Draw(output)
    for index, row in enumerate(selected):
        image = Image.open(ROOT / str(row["image"])).convert("RGB")
        image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        col = index % columns
        display_row = index // columns
        x = col * thumb_width
        y = display_row * (thumb_height + 28)
        output.paste(image, (x, y))
        draw.text((x + 4, y + thumb_height + 4), str(row["sample_id"]), fill="black")
    QC_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.save(QC_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribe exclusivamente las 48 salidas deterministas declaradas; no borra directorios.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registered = read_registry()
    assets = split_assets(registered)
    write_asset_split(assets)
    recipes = build_recipes()
    rows: list[dict[str, str | int]] = []
    for index, recipe in enumerate(recipes, start=1):
        selected = assets_for_recipe(recipe, assets)
        rows.append(generate_sheet(recipe, selected, args.overwrite))
        print(f"[{index:02d}/{len(recipes)}] {recipe.sample_id}")
    write_manifest(rows)
    make_contact_sheet(rows)
    config = {
        "dataset_version": "asset_identity_v2",
        "split_protocol": SPLIT_PROTOCOL,
        "split_seed": SPLIT_SEED,
        "generation_seed_base": GENERATION_SEED,
        "canvas": {"width_px": WIDTH, "height_px": HEIGHT, "px_per_mm": 10.0},
        "asset_allocation_per_family": {"train": 4, "val": 2, "test": 2},
        "sheet_allocation": {"train": 24, "val": 12, "test": 12},
        "semantic_families_disjoint": False,
        "asset_identity_disjoint": True,
        "train_cv_groups": 4,
        "instances_per_sheet": 8,
        "generator": Path(__file__).relative_to(ROOT).as_posix(),
        "asset_registry": ASSET_REGISTRY.relative_to(ROOT).as_posix(),
        "asset_split": ASSET_SPLIT.relative_to(ROOT).as_posix(),
        "dataset_manifest": DATASET_MANIFEST.relative_to(ROOT).as_posix(),
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Dataset generado: {len(rows)} láminas")
    print(f"Manifiesto: {DATASET_MANIFEST}")
    print(f"Control visual: {QC_PATH}")


if __name__ == "__main__":
    main()
