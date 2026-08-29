"""Importa y registra el banco privado de 32 toppers con canal alfa.

La copia local se excluye de Git. El CSV de procedencia sí se versiona y permite
verificar nombres, dimensiones y hashes sin afirmar derechos no documentados.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "datos" / "fuentes_toppers" / "alpha"
REGISTRY = ROOT / "datos" / "manifiesto" / "asset_registry_v2.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_number(path: Path) -> int:
    try:
        return int(path.stem.split("_")[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Nombre de activo no reconocido: {path.name}") from exc


def family_for(number: int) -> str:
    if not 1 <= number <= 32:
        raise ValueError(f"Índice fuera del banco esperado: {number}")
    return f"F{(number - 1) // 8 + 1}"


def import_assets(source_dir: Path) -> list[dict[str, str | int]]:
    candidates = sorted(source_dir.glob("IMG_*_alpha.png"), key=asset_number)
    numbers = [asset_number(path) for path in candidates]
    if numbers != list(range(1, 33)):
        raise RuntimeError(
            "Se requieren exactamente IMG_1_alpha.png ... IMG_32_alpha.png; "
            f"se encontraron índices {numbers}."
        )

    TARGET.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    for source in candidates:
        number = asset_number(source)
        target = TARGET / source.name
        shutil.copy2(source, target)
        with Image.open(target) as image:
            width, height = image.size
            mode = image.mode
            extrema = image.getchannel("A").getextrema() if "A" in image.getbands() else None
        if mode != "RGBA" or extrema is None or extrema[0] == extrema[1]:
            raise RuntimeError(f"{target.name} no contiene un canal alfa útil (modo={mode}).")
        rows.append(
            {
                "asset_id": f"A{number:02d}",
                "source_file": source.name,
                "family": family_for(number),
                "width_px": width,
                "height_px": height,
                "mode": mode,
                "sha256": sha256(target),
                "source_package": "author_provided_asset_bank",
                "creation_method_status": "creation_method_not_documented",
                "redistribution_status": "private_academic_use_no_open_license_asserted",
            }
        )

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directorio que contiene IMG_1_alpha.png ... IMG_32_alpha.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    rows = import_assets(source_dir)
    print(f"Importados y verificados {len(rows)} activos en {TARGET}")
    print(f"Registro reproducible: {REGISTRY}")


if __name__ == "__main__":
    main()
