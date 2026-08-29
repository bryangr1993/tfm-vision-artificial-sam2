"""Captura las cuatro pestañas de la GUI a partir de una ejecución ya validada.

El script no vuelve a inferir ni modifica resultados. Carga los artefactos de la
captura 20 guardados por la aplicación, los presenta en la interfaz actual y
registra las cuatro vistas con un estado operativo coherente de SAM 2 + WCS.
"""

from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

import cv2
from PIL import Image, ImageGrab


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "software" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detect_wcs_l import detect_wcs_l  # noqa: E402
from gui import InnokeyApp  # noqa: E402


RAW_IMAGE = ROOT / "datos" / "reales" / "raw" / "20.jpg"
RESULTS = ROOT / "software" / "outputs_gui" / "20_results"
OUTPUT = ROOT / "memoria" / "figuras" / "interfaz_final"
CAPTURE_WIDTH = 1425
CAPTURE_HEIGHT = 970
CAPTURE_FONT_PT = 16


def checked_read(path: Path, mode: int) -> object:
    image = cv2.imread(str(path), mode)
    if image is None:
        raise FileNotFoundError(path)
    return image


def prepare_recorded_result(app: InnokeyApp) -> None:
    app.img_path = str(RAW_IMAGE)
    app.lbl_filename.config(text=RAW_IMAGE.name)
    # La captura marcada supera los cien megapíxeles. La decodificación reducida
    # conserva mucha más resolución que la ventana y evita un coste innecesario.
    original_marked = checked_read(
        RESULTS / "original_markers.png", cv2.IMREAD_REDUCED_COLOR_4
    )
    app.original_img = original_marked
    rectified = checked_read(RESULTS / "rectified_20.png", cv2.IMREAD_COLOR)
    rectified_wcs = checked_read(RESULTS / "rectified_with_wcs_20.png", cv2.IMREAD_COLOR)
    segmentation = checked_read(RESULTS / "20_contours.png", cv2.IMREAD_COLOR)
    mask = checked_read(RESULTS / "sam2_mask_20.png", cv2.IMREAD_GRAYSCALE)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= app.config.sam2.min_component_area_px
    ]
    contours.sort(key=lambda contour: (cv2.boundingRect(contour)[1], cv2.boundingRect(contour)[0]))
    if len(contours) != 8:
        raise RuntimeError(f"Se esperaban 8 contornos exteriores y se obtuvieron {len(contours)}")

    app.rectified = rectified
    app.mask = mask
    app.contours = contours
    app.wcs_info = detect_wcs_l(
        rectified,
        debug_dir=None,
        scale=app.config.geometry.pixels_per_mm,
        marker_margin=app.config.geometry.marker_margin_mm,
    )
    if app.wcs_info.get("status") != "SUCCESS":
        raise RuntimeError(f"El WCS registrado no es válido: {app.wcs_info}")

    app.report = {
        "toppers_detected": 8,
        "outer_contours_count": 8,
        "holes_preserved_count": 0,
        "cut_paths_total": 8,
        "path_roles": ["outer"] * 8,
    }
    app.processed = True
    app.last_dxf_path = str(RESULTS / "toppers_20.dxf")
    app.last_dxf_validated = True
    app.btn_process.config(state=tk.NORMAL)
    app.btn_export.config(state=tk.NORMAL)
    app.update_status_ui(
        hoja="SÍ",
        marcadores="4/4",
        wcs="SÍ",
        toppers=8,
        modo="SAM 2 + WCS",
    )
    app.lbl_checkpoint.config(
        text="sam2_hiera_tiny.pt: verificado en la ejecución registrada",
        foreground=app.accent_green,
    )

    app.root.update_idletasks()
    app.root.update()
    app.display_image_on_canvas(original_marked, app.canvas_orig, is_original=True)
    app.display_image_on_canvas(rectified_wcs, app.canvas_rect)
    app.display_image_on_canvas(segmentation, app.canvas_seg)
    app.render_vectors_tab("down", 0.0)


def capture_tab(app: InnokeyApp, tab: object, filename: str) -> Path:
    app.notebook.select(tab)
    app.root.update_idletasks()
    app.root.update()
    time.sleep(0.3)

    left = app.root.winfo_rootx()
    top = app.root.winfo_rooty()
    right = left + app.root.winfo_width()
    bottom = top + app.root.winfo_height()
    capture = ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True)
    destination = OUTPUT / filename
    capture.save(destination)
    return destination


def hide_footer_for_capture(widget: tk.Misc) -> None:
    """Oculta únicamente el pie institucional cuando la pantalla no da altura."""

    for child in widget.winfo_children():
        try:
            text = str(child.cget("text"))
        except tk.TclError:
            text = ""
        if text.startswith("TFM: Visión Artificial aplicada al corte CNC"):
            child.pack_forget()
        hide_footer_for_capture(child)


def configure_document_capture(app: InnokeyApp) -> None:
    """Aumenta la escala visual para que la captura siga siendo legible en A4.

    No cambia el procesamiento ni el estado de la aplicación. Solo aplica una
    presentación equivalente a usar escalado de accesibilidad en pantalla.
    """

    app.root.geometry(f"{CAPTURE_WIDTH}x{CAPTURE_HEIGHT}+20+20")
    app.style.configure("TButton", padding=6, font=("Segoe UI", CAPTURE_FONT_PT))
    app.style.configure(
        "Action.TButton", padding=8, font=("Segoe UI", CAPTURE_FONT_PT, "bold")
    )
    app.style.configure("TLabel", font=("Segoe UI", CAPTURE_FONT_PT))
    app.style.configure("Heading.TLabel", font=("Segoe UI", 16, "bold"))
    app.style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
    app.style.configure("StatusVal.TLabel", font=("Segoe UI", CAPTURE_FONT_PT, "bold"))
    app.style.configure("TLabelframe.Label", font=("Segoe UI", CAPTURE_FONT_PT, "bold"))
    app.style.configure("TNotebook.Tab", font=("Segoe UI", CAPTURE_FONT_PT))
    app.style.configure("TCheckbutton", font=("Segoe UI", CAPTURE_FONT_PT))
    app.style.configure("TEntry", font=("Segoe UI", CAPTURE_FONT_PT))
    app.style.configure("TCombobox", font=("Segoe UI", CAPTURE_FONT_PT))

    # Dos etiquetas tienen una fuente explícita y no heredan el estilo de ttk.
    # Se elevan al mismo mínimo empleado en el resto de la captura.
    def raise_explicit_fonts(widget: tk.Misc) -> None:
        try:
            current = widget.cget("font")
        except tk.TclError:
            current = ""
        if current:
            try:
                actual = tkfont.Font(root=app.root, font=current).actual()
                size = abs(int(actual.get("size", 0)))
                if 0 < size < CAPTURE_FONT_PT:
                    style_bits = []
                    if actual.get("weight") == "bold":
                        style_bits.append("bold")
                    if actual.get("slant") == "italic":
                        style_bits.append("italic")
                    widget.configure(
                        font=(
                            actual.get("family", "Segoe UI"),
                            CAPTURE_FONT_PT,
                            " ".join(style_bits) or "normal",
                        )
                    )
            except (tk.TclError, ValueError):
                pass
        for child in widget.winfo_children():
            raise_explicit_fonts(child)

    raise_explicit_fonts(app.root)

    # La barra lateral gana anchura para conservar las etiquetas sin cortes.
    main_frame = app.content_frame.master
    for child in main_frame.winfo_children():
        if child is not app.content_frame:
            child.configure(width=470)
            sidebar = child
            break
    else:
        raise RuntimeError("No se encontró la barra lateral de la interfaz")

    # Se reducen solo los márgenes verticales. El contenido y todos los controles
    # permanecen visibles, incluido el botón final de exportación.
    for child in sidebar.winfo_children():
        try:
            label = str(child.cget("text"))
        except tk.TclError:
            label = ""
        if label == "INNOKEY CNC VISION":
            child.pack_configure(pady=(14, 2))
        elif label == "Segmentación operativa con SAM 2":
            child.pack_configure(pady=(0, 9))
        elif child is app.btn_process:
            child.pack_configure(pady=(8, 6))
        elif child is app.grp_status:
            child.configure(padding=7)
            child.pack_configure(pady=3)
        elif child is app.btn_export:
            child.pack_configure(pady=(8, 6))
        elif isinstance(child, ttk.LabelFrame):
            child.configure(padding=7)
            child.pack_configure(pady=3)
        elif isinstance(child, ttk.Separator):
            child.pack_configure(pady=(0, 8))
    app.lbl_checkpoint.configure(wraplength=205)


def widget_box(
    app: InnokeyApp, widgets: tuple[tk.Misc, ...], padding: int = 10
) -> tuple[int, int, int, int]:
    """Devuelve el rectángulo de varios widgets en coordenadas de la captura."""

    root_x = app.root.winfo_rootx()
    root_y = app.root.winfo_rooty()
    left = min(widget.winfo_rootx() for widget in widgets) - root_x - padding
    top = min(widget.winfo_rooty() for widget in widgets) - root_y - padding
    right = max(
        widget.winfo_rootx() + widget.winfo_width() for widget in widgets
    ) - root_x + padding
    bottom = max(
        widget.winfo_rooty() + widget.winfo_height() for widget in widgets
    ) - root_y + padding
    return (
        max(0, left),
        max(0, top),
        min(app.root.winfo_width(), right),
        min(app.root.winfo_height(), bottom),
    )


def canvas_content_box(
    app: InnokeyApp, canvas: tk.Canvas, padding: int = 12
) -> tuple[int, int, int, int]:
    """Recorta el contenido dibujado, sin conservar el vacío del lienzo."""

    bounds = canvas.bbox("all")
    if bounds is None:
        return widget_box(app, (canvas,), padding)
    root_x = app.root.winfo_rootx()
    root_y = app.root.winfo_rooty()
    left = canvas.winfo_rootx() - root_x + bounds[0] - padding
    top = canvas.winfo_rooty() - root_y + bounds[1] - padding
    right = canvas.winfo_rootx() - root_x + bounds[2] + padding
    bottom = canvas.winfo_rooty() - root_y + bounds[3] + padding
    return (
        max(0, left),
        max(0, top),
        min(app.root.winfo_width(), right),
        min(app.root.winfo_height(), bottom),
    )


def main() -> None:
    for required in (
        RAW_IMAGE,
        RESULTS / "original_markers.png",
        RESULTS / "rectified_20.png",
        RESULTS / "rectified_with_wcs_20.png",
        RESULTS / "20_contours.png",
        RESULTS / "sam2_mask_20.png",
        RESULTS / "toppers_20.dxf",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Evita que Windows mezcle coordenadas lógicas de Tk con píxeles físicos
    # al calcular el rectángulo de captura en pantallas con escalado.
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    root.tk.call("tk", "scaling", 1.2)
    root.attributes("-topmost", True)
    app = InnokeyApp(root)
    root.unbind("<Configure>")
    hide_footer_for_capture(root)
    configure_document_capture(app)
    prepare_recorded_result(app)
    root.lift()
    root.focus_force()

    captures = [
        capture_tab(app, app.tab_orig, "gui_final_01_original.png"),
        capture_tab(app, app.tab_rect, "gui_final_02_rectificada.png"),
        capture_tab(app, app.tab_seg, "gui_final_03_sam2.png"),
        capture_tab(app, app.tab_vec, "gui_final_04_vectores.png"),
    ]
    status_box = widget_box(app, (app.btn_process, app.grp_status, app.btn_export), 14)
    sam_box = canvas_content_box(app, app.canvas_seg, 14)
    vector_box = (
        round(CAPTURE_WIDTH * 0.43),
        round(CAPTURE_HEIGHT * 0.105),
        round(CAPTURE_WIDTH * 0.86),
        round(CAPTURE_HEIGHT * 0.945),
    )
    with Image.open(captures[2]) as sam_view:
        sam_view.crop((0, 0, sam_view.width, sam_view.height - 40)).save(
            OUTPUT / "gui_final_03_sam2_document.png"
        )
        sam_view.crop(status_box).save(OUTPUT / "gui_final_detalle_estado.png")
        sam_view.crop(sam_box).save(OUTPUT / "gui_final_detalle_sam2.png")
    with Image.open(captures[3]) as vector_view:
        vector_view.crop(vector_box).save(OUTPUT / "gui_final_detalle_vectores.png")

    root.destroy()

    for capture in captures:
        print(capture)


if __name__ == "__main__":
    main()
