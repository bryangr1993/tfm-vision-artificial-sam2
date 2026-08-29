"""Genera la figura de variabilidad visual a partir de capturas reales del estudio."""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datos" / "reales" / "raw"
OUTPUT = ROOT / "memoria" / "figuras" / "dificultades_visuales_toppers.pdf"

PANELS = [
    ("28.jpg", "Iluminación desigual", "Sombra horizontal sobre la hoja y el soporte"),
    ("17.jpg", "Presencia del cabezal", "Elementos de la máquina dentro del campo visual"),
    ("34.jpg", "Rotación del soporte", "Cambio simultáneo de orientación y perspectiva"),
    ("38.jpg", "Desplazamiento en la mesa", "La hoja aparece lejos de la posición nominal"),
]


def load_for_panel(path: Path, target_ratio: float = 1.72) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    width, height = image.size
    ratio = width / height
    if ratio > target_ratio:
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        image = image.crop((left, 0, left + new_width, height))
    else:
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        image = image.crop((0, top, width, top + new_height))
    return image


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 9,
    })
    # La leyenda de la memoria aporta el contexto. Dentro de la figura se
    # conservan solo las fotografías y sus encabezados para maximizar el área
    # útil y evitar repetir texto a una escala menor.
    fig, axes = plt.subplots(2, 2, figsize=(5.35, 3.55))
    fig.patch.set_facecolor("white")

    for index, (axis, (filename, title, subtitle)) in enumerate(zip(axes.flat, PANELS), start=1):
        axis.imshow(load_for_panel(RAW / filename))
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#0b2d57")
            spine.set_linewidth(1.2)
        axis.set_title(f"{index}. {title}", loc="left", color="#0b2d57", pad=7)
        # El detalle se conserva en la leyenda larga de la memoria. Repetirlo
        # dentro del panel reducía el área útil de la fotografía.

    fig.subplots_adjust(left=0.035, right=0.985, top=0.965, bottom=0.02, wspace=0.08, hspace=0.20)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
