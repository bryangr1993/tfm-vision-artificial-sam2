import os
import sys
import time
import datetime
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Asegurar que el directorio src está en el path
src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.append(src_path)

# Importaciones del pipeline original
from load_images import load_image
from detect_markers import detect_markers
from rectify_sheet import rectify_sheet
from detect_wcs_l import detect_wcs_l
from export_dxf import simplify_and_transform_contour, apply_offset_shapely, SHAPELY_AVAILABLE
from ai_pipeline import build_operational_pipeline, segment_and_extract_with_ai
from segmenters import SAM2UnavailableError
import ezdxf

class InnokeyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Innokey - Registro Geométrico y Vectorización (TFM)")
        self.root.geometry("1300x820")
        self.root.minsize(1100, 700)
        
        # Color Palette - Sobrio y Académico
        self.bg_color = "#f5f5f7"
        self.sidebar_color = "#ffffff"
        self.text_color = "#1d1d1f"
        self.accent_green = "#2e7d32" # Green for SUCCESS WCS
        self.accent_amber = "#e65100" # Orange for Modo Análisis
        self.accent_gray = "#757575"  # Gray for Pending
        
        # Configurar estilos ttk
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        self.style.configure(".", background=self.bg_color, foreground=self.text_color)
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("Sidebar.TFrame", background=self.sidebar_color, relief="solid", borderwidth=1)
        self.style.configure("TButton", padding=6, font=("Segoe UI", 10))
        self.style.configure("Action.TButton", padding=8, font=("Segoe UI", 10, "bold"))
        self.style.configure("TLabel", background=self.bg_color, font=("Segoe UI", 10))
        self.style.configure("Heading.TLabel", font=("Segoe UI", 12, "bold"))
        self.style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"), background=self.sidebar_color)
        self.style.configure("StatusVal.TLabel", font=("Segoe UI", 10, "bold"))
        
        # Variables de estado
        self.img_path = None
        self.original_img = None
        self.processed = False
        
        # Resultados del procesamiento
        self.markers = None
        self.rectified = None
        self.wcs_info = None
        self.mask = None
        self.segmentation_result = None
        self.ai_pipeline = None
        self.contours = None
        self.report = None
        self.elapsed_time = 0.0
        self.base_output_dir = os.path.join(os.path.dirname(src_path), "outputs_gui")
        
        # Referencias de imágenes para Tkinter
        self.tk_orig = None
        self.tk_rect = None
        self.tk_seg = None
        
        # Crear la estructura de la interfaz
        self.create_layout()
        
    def create_layout(self):
        # Frame Principal de Layout (Sidebar a la izquierda, Contenido a la derecha)
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 1. SIDEBAR (Panel de Control)
        sidebar = ttk.Frame(main_frame, style="Sidebar.TFrame", width=340)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        sidebar.pack_propagate(False)
        
        # Título y Subtítulo
        lbl_title = ttk.Label(sidebar, text="INNOKEY CNC VISION", style="Title.TLabel")
        lbl_title.pack(anchor=tk.W, padx=20, pady=(25, 2))
        lbl_sub = ttk.Label(sidebar, text="Segmentación operativa con SAM 2", font=("Segoe UI", 9, "italic"), foreground="#86868b", background=self.sidebar_color)
        lbl_sub.pack(anchor=tk.W, padx=20, pady=(0, 20))
        
        # Separador
        sep1 = ttk.Separator(sidebar, orient="horizontal")
        sep1.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        # --- Grupo 1: Carga de Archivo ---
        grp_load = ttk.LabelFrame(sidebar, text=" Entrada de Datos ", labelanchor="nw", padding=10)
        grp_load.pack(fill=tk.X, padx=15, pady=5)
        
        self.btn_load = ttk.Button(grp_load, text="Cargar Imagen...", command=self.load_file)
        self.btn_load.pack(fill=tk.X, pady=2)
        
        self.lbl_filename = ttk.Label(grp_load, text="Ninguna imagen cargada", font=("Segoe UI", 9, "italic"), foreground="#86868b")
        self.lbl_filename.pack(anchor=tk.W, pady=(5, 0))
        
        # --- Grupo 2: Parámetros del Proceso ---
        grp_params = ttk.LabelFrame(sidebar, text=" Parámetros Operativos ", labelanchor="nw", padding=10)
        grp_params.pack(fill=tk.X, padx=15, pady=5)
        
        # Escala px/mm
        lbl_scale = ttk.Label(grp_params, text="Escala (px/mm):")
        lbl_scale.grid(row=0, column=0, sticky=tk.W, pady=4)
        self.var_scale = tk.DoubleVar(value=10.0)
        self.ent_scale = ttk.Spinbox(grp_params, from_=1.0, to=50.0, increment=1.0, textvariable=self.var_scale, width=10)
        self.ent_scale.grid(row=0, column=1, sticky=tk.E, pady=4, padx=5)
        
        # Offset de corte (mm)
        lbl_offset = ttk.Label(grp_params, text="Offset corte (mm):")
        lbl_offset.grid(row=1, column=0, sticky=tk.W, pady=4)
        self.var_offset = tk.DoubleVar(value=0.0)
        self.ent_offset = ttk.Spinbox(grp_params, from_=0.0, to=10.0, increment=0.1, textvariable=self.var_offset, width=10)
        self.ent_offset.grid(row=1, column=1, sticky=tk.E, pady=4, padx=5)
        
        # Rotación
        lbl_rot = ttk.Label(grp_params, text="Rotación:")
        lbl_rot.grid(row=2, column=0, sticky=tk.W, pady=4)
        self.var_rotate = tk.StringVar(value="0°")
        self.cb_rotate = ttk.Combobox(grp_params, values=["0°", "90°", "180°", "270°"], textvariable=self.var_rotate, state="readonly", width=12)
        self.cb_rotate.grid(row=2, column=1, sticky=tk.E, pady=4, padx=5)
        self.cb_rotate.bind("<<ComboboxSelected>>", self.on_rotate_changed)
        
        # Dirección eje Y
        lbl_dir = ttk.Label(grp_params, text="Sentido Eje Y:")
        lbl_dir.grid(row=3, column=0, sticky=tk.W, pady=4)
        self.var_ydir = tk.StringVar(value="Abajo (down)")
        self.cb_ydir = ttk.Combobox(grp_params, values=["Abajo (down)", "Arriba (up)"], textvariable=self.var_ydir, state="readonly", width=12)
        self.cb_ydir.grid(row=3, column=1, sticky=tk.E, pady=4, padx=5)
        
        # Checkbox Marca WCS auxiliar
        self.var_inc_wcs = tk.BooleanVar(value=True)
        self.chk_wcs = ttk.Checkbutton(grp_params, text="Incluir marca WCS auxiliar", variable=self.var_inc_wcs)
        self.chk_wcs.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(8, 2))
        
        # --- Botón Ejecutar Procesamiento ---
        self.btn_process = ttk.Button(sidebar, text="PROCESAR IMAGEN", style="Action.TButton", command=self.process_image, state=tk.DISABLED)
        self.btn_process.pack(fill=tk.X, padx=15, pady=(15, 10))
        
        # --- Grupo 3: Estado del Procesamiento ---
        self.grp_status = ttk.LabelFrame(sidebar, text=" Estado del Procesamiento ", labelanchor="nw", padding=10)
        self.grp_status.pack(fill=tk.X, padx=15, pady=5)
        
        # Grid para etiquetas de estado
        lbl_status_hoja = ttk.Label(self.grp_status, text="Hoja detectada:")
        lbl_status_hoja.grid(row=0, column=0, sticky=tk.W, pady=3)
        self.val_hoja = ttk.Label(self.grp_status, text="PENDIENTE", foreground=self.accent_gray, style="StatusVal.TLabel")
        self.val_hoja.grid(row=0, column=1, sticky=tk.E, pady=3, padx=10)
        
        lbl_status_markers = ttk.Label(self.grp_status, text="Marcadores:")
        lbl_status_markers.grid(row=1, column=0, sticky=tk.W, pady=3)
        self.val_markers = ttk.Label(self.grp_status, text="PENDIENTE", foreground=self.accent_gray, style="StatusVal.TLabel")
        self.val_markers.grid(row=1, column=1, sticky=tk.E, pady=3, padx=10)
        
        lbl_status_wcs = ttk.Label(self.grp_status, text="WCS detectado:")
        lbl_status_wcs.grid(row=2, column=0, sticky=tk.W, pady=3)
        self.val_wcs = ttk.Label(self.grp_status, text="PENDIENTE", foreground=self.accent_gray, style="StatusVal.TLabel")
        self.val_wcs.grid(row=2, column=1, sticky=tk.E, pady=3, padx=10)
        
        lbl_status_toppers = ttk.Label(self.grp_status, text="Toppers detectados:")
        lbl_status_toppers.grid(row=3, column=0, sticky=tk.W, pady=3)
        self.val_toppers = ttk.Label(self.grp_status, text="PENDIENTE", foreground=self.accent_gray, style="StatusVal.TLabel")
        self.val_toppers.grid(row=3, column=1, sticky=tk.E, pady=3, padx=10)
        
        lbl_status_modo = ttk.Label(self.grp_status, text="Modo de Operación:")
        lbl_status_modo.grid(row=4, column=0, sticky=tk.W, pady=3)
        self.val_modo = ttk.Label(self.grp_status, text="PENDIENTE", foreground=self.accent_gray, style="StatusVal.TLabel")
        self.val_modo.grid(row=4, column=1, sticky=tk.E, pady=3, padx=10)
        
        # --- Botón Exportar Vector ---
        self.btn_export = ttk.Button(sidebar, text="EXPORTAR VECTOR (DXF)", style="Action.TButton", command=self.export_vector, state=tk.DISABLED)
        self.btn_export.pack(fill=tk.X, padx=15, pady=(15, 10))
        
        # Mensaje de ayuda / información
        lbl_info = ttk.Label(sidebar, text="TFM: Visión Artificial aplicada al corte CNC\nUniversidad Internacional de La Rioja (UNIR)", 
                             font=("Segoe UI", 8), justify=tk.CENTER, foreground="#86868b", background=self.sidebar_color)
        lbl_info.pack(side=tk.BOTTOM, fill=tk.X, pady=15)
        
        # 2. CONTENIDO (Panel Derecho con Pestañas)
        self.content_frame = ttk.Frame(main_frame, padding=15)
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Notebook de Pestañas
        self.notebook = ttk.Notebook(self.content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Crear las pestañas
        self.tab_orig = ttk.Frame(self.notebook)
        self.tab_rect = ttk.Frame(self.notebook)
        self.tab_seg = ttk.Frame(self.notebook)
        self.tab_vec = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_orig, text="   Original   ")
        self.notebook.add(self.tab_rect, text="   Rectificada   ")
        self.notebook.add(self.tab_seg, text="   Segmentación IA (SAM 2)   ")
        self.notebook.add(self.tab_vec, text="   Vectores (mm)   ")
        
        # Configurar las vistas de imagen con canvas
        self.canvas_orig = tk.Canvas(self.tab_orig, bg="#e8e8ed")
        self.canvas_orig.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_rect = tk.Canvas(self.tab_rect, bg="#e8e8ed")
        self.canvas_rect.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_seg = tk.Canvas(self.tab_seg, bg="#e8e8ed")
        self.canvas_seg.pack(fill=tk.BOTH, expand=True)
        
        # Vinculación para redimensionamiento automático
        self.root.bind("<Configure>", self.on_window_resize)
        
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar fotografía real de toppers",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*")]
        )
        if not file_path:
            return

        self.load_path(file_path)

    def load_path(self, file_path):
        """Carga una imagen desde una ruta conocida, sin abrir el selector."""
        if not file_path or not os.path.isfile(file_path):
            messagebox.showerror("Error de Carga", f"No existe la imagen:\n{file_path}")
            return False

        self.img_path = file_path
        self.lbl_filename.config(text=os.path.basename(file_path))
        self.var_rotate.set("0°")
        
        # Cargar vista previa original con corrección de orientación EXIF automática
        self.original_img = load_image(self.img_path, manual_rotate=0)
        if self.original_img is None:
            messagebox.showerror("Error de Carga", "No se pudo leer la imagen seleccionada.")
            return False
            
        self.processed = False
        self.btn_process.config(state=tk.NORMAL)
        self.btn_export.config(state=tk.DISABLED)
        
        # Limpiar estados
        self.val_hoja.config(text="PENDIENTE", foreground=self.accent_gray)
        self.val_markers.config(text="PENDIENTE", foreground=self.accent_gray)
        self.val_wcs.config(text="PENDIENTE", foreground=self.accent_gray)
        self.val_toppers.config(text="PENDIENTE", foreground=self.accent_gray)
        self.val_modo.config(text="PENDIENTE", foreground=self.accent_gray)
        
        # Limpiar pestaña de vectores
        for widget in self.tab_vec.winfo_children():
            widget.destroy()
            
        # Limpiar canvas rectified y segment
        self.canvas_rect.delete("all")
        self.canvas_seg.delete("all")
        
        # Mostrar imagen original en pestaña 1
        self.display_image_on_canvas(self.original_img, self.canvas_orig, is_original=True)
        self.notebook.select(0)
        return True
        
    def get_rotated_image(self, img, angle_str):
        if angle_str == "90°":
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle_str == "180°":
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle_str == "270°":
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def on_rotate_changed(self, event):
        if self.original_img is not None:
            rotated = self.get_rotated_image(self.original_img, self.var_rotate.get())
            self.display_image_on_canvas(rotated, self.canvas_orig, is_original=True)

    def display_image_on_canvas(self, cv_img, canvas, is_original=False):
        canvas.delete("all")
        h_canvas = canvas.winfo_height()
        w_canvas = canvas.winfo_width()
        
        # Si el canvas no está renderizado (tamaño <= 1), usar el tamaño por defecto
        if h_canvas <= 1 or w_canvas <= 1:
            h_canvas = 700
            w_canvas = 900
            
        h, w = cv_img.shape[:2]
        scale = min((w_canvas - 40) / w, (h_canvas - 40) / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        # Mantener referencia
        if is_original:
            self.tk_orig = tk_img
        elif canvas == self.canvas_rect:
            self.tk_rect = tk_img
        else:
            self.tk_seg = tk_img
            
        x_center = w_canvas // 2
        y_center = h_canvas // 2
        canvas.create_image(x_center, y_center, image=tk_img)
        
    def on_window_resize(self, event):
        # Manejar rediseño de imágenes cargadas ante cambio de ventana
        if self.img_path is None:
            return
            
        # Redibujar imagen original
        if not self.processed:
            rotated = self.get_rotated_image(self.original_img, self.var_rotate.get())
            self.display_image_on_canvas(rotated, self.canvas_orig, is_original=True)
        else:
            # Si ya procesamos, recargar las imágenes del output folder
            base_name = os.path.splitext(os.path.basename(self.img_path))[0]
            out_dir = os.path.join(self.base_output_dir, f"{base_name}_results")
            
            orig_path = os.path.join(out_dir, "original_markers.png")
            if os.path.exists(orig_path):
                img_orig = cv2.imread(orig_path)
                self.display_image_on_canvas(img_orig, self.canvas_orig, is_original=True)
                
            rect_path = os.path.join(out_dir, f"rectified_with_wcs_{base_name}.png")
            if not os.path.exists(rect_path):
                rect_path = os.path.join(out_dir, f"rectified_{base_name}.png")
            if os.path.exists(rect_path):
                img_rect = cv2.imread(rect_path)
                self.display_image_on_canvas(img_rect, self.canvas_rect)
                
            seg_path = os.path.join(out_dir, f"{base_name}_contours.png")
            if os.path.exists(seg_path):
                img_seg = cv2.imread(seg_path)
                self.display_image_on_canvas(img_seg, self.canvas_seg)
                
    def process_image(self):
        if not self.img_path:
            return
            
        self.root.config(cursor="watch")
        self.btn_process.config(text="PROCESANDO...", state=tk.DISABLED)
        self.btn_load.config(state=tk.DISABLED)
        self.root.update()
        
        t_start = time.time()
        
        # Configurar carpeta de salida
        base_name = os.path.splitext(os.path.basename(self.img_path))[0]
        out_dir = os.path.join(self.base_output_dir, f"{base_name}_results")
        os.makedirs(out_dir, exist_ok=True)
        
        scale_val = self.var_scale.get()
        offset_val = self.var_offset.get()
        ydir_val = "down" if "down" in self.var_ydir.get() else "up"
        
        try:
            # 1. Cargar imagen original con rotación manual elegida
            rot_str = self.var_rotate.get()
            rot_val = 0
            if "90" in rot_str: rot_val = 90
            elif "180" in rot_str: rot_val = 180
            elif "270" in rot_str: rot_val = 270
            
            img = load_image(self.img_path, manual_rotate=rot_val)
            
            # 2. Detección de marcadores
            self.markers = detect_markers(img, debug_dir=None)
            
            # Dibujar marcadores en original y guardar
            orig_marked = img.copy()
            if self.markers is not None:
                for idx, pt in enumerate(self.markers):
                    cv2.circle(orig_marked, (int(pt[0]), int(pt[1])), 45, (0, 255, 0), 10)
                    cv2.putText(orig_marked, f"M{idx+1}", (int(pt[0])-55, int(pt[1])-55),
                                cv2.FONT_HERSHEY_SIMPLEX, 3.2, (0, 255, 0), 8)
            cv2.imwrite(os.path.join(out_dir, "original_markers.png"), orig_marked)
            
            # Si no detectó 4 marcadores, abortar
            if self.markers is None or len(self.markers) < 4:
                self.update_status_ui(hoja="NO", marcadores=f"{len(self.markers) if self.markers else 0}/4", wcs="NO", toppers=0, modo="PENDIENTE")
                messagebox.showerror("Error de Procesamiento", "No se detectaron los 4 marcadores de la hoja A4.")
                self.reset_cursor_buttons()
                return
                
            # 3. Rectificación por homografía
            sheet_w, sheet_h = 210.0, 297.0
            self.rectified, M = rectify_sheet(img, self.markers, out_dir, base_name, sheet_size=(sheet_w, sheet_h), scale=scale_val, marker_margin=10.0)
            
            # 4. Detección de la marca WCS en L
            self.wcs_info = detect_wcs_l(self.rectified, debug_dir=None, scale=scale_val)
            
            # Guardar versión rectificada con ejes WCS si corresponde
            rectified_w_wcs = self.rectified.copy()
            if self.wcs_info["status"] == "SUCCESS":
                ox, oy = self.wcs_info["origin"]
                ux_x, ux_y = self.wcs_info["uX"]
                uy_x, uy_y = self.wcs_info["uY"]
                # Dibujar origen
                cv2.circle(rectified_w_wcs, (int(ox), int(oy)), 12, (0, 0, 255), -1)
                # Eje X
                x_end = (int(ox + ux_x * 150), int(oy + ux_y * 150))
                cv2.arrowedLine(rectified_w_wcs, (int(ox), int(oy)), x_end, (255, 0, 0), 4, tipLength=0.2)
                cv2.putText(rectified_w_wcs, "X (WCS)", (x_end[0] + 10, x_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                # Eje Y
                y_end = (int(ox + uy_x * 150), int(oy + uy_y * 150))
                cv2.arrowedLine(rectified_w_wcs, (int(ox), int(oy)), y_end, (0, 255, 0), 4, tipLength=0.2)
                cv2.putText(rectified_w_wcs, "Y (WCS)", (y_end[0], y_end[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(out_dir, f"rectified_with_wcs_{base_name}.png"), rectified_w_wcs)
            
            # 5 y 6. Localización clásica, segmentación SAM 2 y contornos
            if self.ai_pipeline is None:
                self.ai_pipeline = build_operational_pipeline()
            self.segmentation_result, self.contours, self.report = segment_and_extract_with_ai(
                self.ai_pipeline,
                self.rectified,
                scale=scale_val,
                wcs_info=self.wcs_info,
                debug_dir=out_dir,
                image_name=base_name,
            )
            self.mask = self.segmentation_result.mask
            cv2.imwrite(os.path.join(out_dir, f"sam2_mask_{base_name}.png"), self.mask)
            
            self.elapsed_time = (time.time() - t_start)
            self.processed = True
            
            # 7. Actualizar indicadores de la interfaz
            wcs_success = self.wcs_info["status"] == "SUCCESS"
            self.update_status_ui(
                hoja="SÍ",
                marcadores="4/4",
                wcs="SÍ" if wcs_success else "NO",
                toppers=self.report["toppers_detected"],
                modo="SAM 2 + WCS" if wcs_success else "SAM 2 · ANÁLISIS"
            )
            
            # Activar botones
            self.btn_export.config(state=tk.NORMAL)
            
            # Renderizar vistas en las pestañas
            # Original marcada
            self.display_image_on_canvas(orig_marked, self.canvas_orig, is_original=True)
            # Rectificada con WCS
            if wcs_success:
                self.display_image_on_canvas(rectified_w_wcs, self.canvas_rect)
            else:
                self.display_image_on_canvas(self.rectified, self.canvas_rect)
            # Segmentación y contornos
            img_contours = cv2.imread(os.path.join(out_dir, f"{base_name}_contours.png"))
            self.display_image_on_canvas(img_contours, self.canvas_seg)
            
            # 8. Renderizar pestaña de Vectores (Matplotlib interactivo)
            self.render_vectors_tab(ydir_val, offset_val)
            
            # 9. Guardar reporte de procesamiento y vector por defecto en la carpeta
            default_dxf_path = os.path.join(out_dir, f"toppers_{base_name}.dxf")
            self.save_dxf_logic(default_dxf_path, scale_val, offset_val, ydir_val, self.var_inc_wcs.get())
            self.save_json_report(out_dir, base_name, scale_val, offset_val, ydir_val)
            
            # Cambiar pestaña a Rectificada
            self.notebook.select(1)
            
        except SAM2UnavailableError as e:
            messagebox.showerror(
                "SAM 2 no disponible",
                f"La segmentación de IA no pudo iniciarse:\n\n{e}\n\n"
                "La aplicación no sustituirá SAM 2 por la línea base clásica de forma silenciosa."
            )
        except Exception as e:
            messagebox.showerror("Error Grave", f"Ocurrió un error en el pipeline: {e}")
            import traceback
            traceback.print_exc()
            
        self.reset_cursor_buttons()
        
    def reset_cursor_buttons(self):
        self.root.config(cursor="")
        self.btn_process.config(text="PROCESAR IMAGEN", state=tk.NORMAL)
        self.btn_load.config(state=tk.NORMAL)
        
    def update_status_ui(self, hoja, marcadores, wcs, toppers, modo):
        # Helper para cambiar colores según estado
        self.val_hoja.config(text=hoja, foreground=self.accent_green if hoja == "SÍ" else self.accent_amber)
        self.val_markers.config(text=marcadores, foreground=self.accent_green if "4" in marcadores else self.accent_amber)
        self.val_wcs.config(text=wcs, foreground=self.accent_green if wcs == "SÍ" else self.accent_amber)
        self.val_toppers.config(text=str(toppers), foreground=self.accent_green if toppers > 0 else self.accent_amber)
        self.val_modo.config(text=modo, foreground=self.accent_green if "SAM 2" in modo and "ANÁLISIS" not in modo else self.accent_amber)
        
    def render_vectors_tab(self, y_direction, offset_mm):
        # Limpiar pestaña de vectores
        for widget in self.tab_vec.winfo_children():
            widget.destroy()
            
        fig = Figure(figsize=(7, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'box')
        
        # Límites A4 físicos
        ax.set_xlim(-10, 220)
        ax.set_ylim(-10, 310)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlabel("Eje X (mm)", fontweight="bold")
        ax.set_ylabel("Eje Y (mm)", fontweight="bold")
        ax.set_title("Previsualización Vectorial de Trayectorias de Corte", fontsize=12, fontweight="bold", pad=12)
        
        scale_val = self.var_scale.get()
        wcs_success = self.wcs_info["status"] == "SUCCESS"
        
        # Si no hay WCS, dibujar los contornos en coordenadas de píxeles divididos por escala
        # (Modo Análisis)
        wcs_dummy = self.wcs_info
        if not wcs_success:
            wcs_dummy = {
                "status": "SUCCESS",
                "origin": (0, 0),
                "uX": (1.0, 0.0),
                "uY": (0.0, 1.0)
            }
            ax.set_title("Trayectorias en MODO ANÁLISIS (Sin origen WCS)", fontsize=12, fontweight="bold", color=self.accent_amber, pad=12)
            
        # Graficar contornos
        for idx, cnt in enumerate(self.contours):
            pts_mm = simplify_and_transform_contour(cnt, wcs_dummy, scale_val, y_direction)
            if offset_mm > 0.0:
                pts_mm = apply_offset_shapely(pts_mm, offset_mm)
                
            x_mm = [p[0] for p in pts_mm] + [pts_mm[0][0]]
            y_mm = [p[1] for p in pts_mm] + [pts_mm[0][1]]
            ax.plot(x_mm, y_mm, color="cyan", linewidth=2.0, label="TOPPERS_CUT" if idx == 0 else "")
            # Dibujar el ID del topper en su centroide para trazabilidad
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
                # Proyectar centroide
                v = np.array([cx - wcs_dummy["origin"][0], cy - wcs_dummy["origin"][1]])
                S_Y = 1.0 if y_direction == "down" else -1.0
                cx_mm = np.dot(v, wcs_dummy["uX"]) / scale_val
                cy_mm = (np.dot(v, wcs_dummy["uY"]) / scale_val) * S_Y
                ax.text(cx_mm, cy_mm, str(idx+1), color="#2e7d32", fontsize=10, fontweight="bold",
                        bbox=dict(boxstyle="circle,pad=0.2", fc="yellow", ec="green", lw=1))
        
        # Graficar marca en L WCS si está presente y se seleccionó
        if wcs_success and self.var_inc_wcs.get():
            S_Y = 1.0 if y_direction == "down" else -1.0
            # Brazo X
            ax.plot([0, 20], [0, 0], color="red", linewidth=3.0, label="WCS_REF")
            ax.plot([18, 20, 18], [0.5, 0, -0.5], color="red", linewidth=2.0) # Flecha X
            # Brazo Y
            ax.plot([0, 0], [0, 20 * S_Y], color="red", linewidth=3.0)
            ax.plot([0.5, 0, -0.5], [18*S_Y, 20*S_Y, 18*S_Y], color="red", linewidth=2.0) # Flecha Y
            # Origen (0,0)
            ax.plot(0, 0, "ro", markersize=8)
            ax.text(23, 0, "X", color="red", fontweight="bold", fontsize=11)
            ax.text(2, 22 * S_Y, "Y", color="red", fontweight="bold", fontsize=11)
            ax.text(-8, -12 * S_Y, "(0,0) WCS", color="red", fontsize=9, fontweight="bold")
            
        if y_direction == "down":
            ax.invert_yaxis()
            
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="gray")
        
        # Dibujar Canvas en Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.tab_vec)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Guardar preview vectorial en la carpeta de resultados
        base_name = os.path.splitext(os.path.basename(self.img_path))[0]
        out_dir = os.path.join(self.base_output_dir, f"{base_name}_results")
        fig.savefig(os.path.join(out_dir, f"vector_preview_{base_name}.png"), dpi=200, bbox_inches="tight")
        
    def export_vector(self):
        if not self.processed:
            return
            
        wcs_success = self.wcs_info["status"] == "SUCCESS"
        
        # Si no hay WCS, el sistema se encuentra en modo análisis. Advertir y bloquear
        if not wcs_success:
            messagebox.showwarning(
                "Modo Análisis Activo",
                "El software se encuentra operando en Modo Análisis debido a la ausencia de una marca física WCS (marca en L).\n\n"
                "La exportación de vectores de corte está bloqueada temporalmente para evitar incidentes y colisiones en la máquina láser."
            )
            return
            
        # Diálogo para guardar archivo
        base_name = os.path.splitext(os.path.basename(self.img_path))[0]
        file_path = filedialog.asksaveasfilename(
            title="Exportar archivo vectorial DXF a la máquina láser",
            initialfile=f"toppers_{base_name}.dxf",
            filetypes=[("Vector DXF (AutoCAD R2010)", "*.dxf")]
        )
        if not file_path:
            return
            
        scale_val = self.var_scale.get()
        offset_val = self.var_offset.get()
        ydir_val = "down" if "down" in self.var_ydir.get() else "up"
        include_wcs_ref = self.var_inc_wcs.get()
        
        success = self.save_dxf_logic(file_path, scale_val, offset_val, ydir_val, include_wcs_ref)
        if success:
            messagebox.showinfo("Exportación Exitosa", f"Archivo vectorial guardado con éxito:\n{os.path.basename(file_path)}")
        else:
            messagebox.showerror("Error", "Fallo al generar el archivo DXF.")
            
    def save_dxf_logic(self, file_path, scale_val, offset_val, ydir_val, include_wcs_ref):
        if self.wcs_info["status"] != "SUCCESS":
            return False
            
        try:
            doc = ezdxf.new(dxfversion='R2010')
            doc.layers.new(name='TOPPERS_CUT', dxfattribs={'color': 4}) # Cyan
            msp = doc.modelspace()
            
            # Exportar toppers
            for cnt in self.contours:
                pts_mm = simplify_and_transform_contour(cnt, self.wcs_info, scale_val, ydir_val)
                if offset_val > 0.0:
                    pts_mm = apply_offset_shapely(pts_mm, offset_val)
                msp.add_lwpolyline(pts_mm, close=True, dxfattribs={'layer': 'TOPPERS_CUT'})
                
            # Exportar WCS auxiliar
            if include_wcs_ref:
                doc.layers.new(name='WCS_REF', dxfattribs={'color': 1}) # Rojo
                S_Y = 1.0 if ydir_val == "down" else -1.0
                
                # Eje X
                msp.add_line((0, 0), (20, 0), dxfattribs={'layer': 'WCS_REF'})
                # Flecha Eje X
                msp.add_lwpolyline([(18, 0.5), (20, 0), (18, -0.5)], close=False, dxfattribs={'layer': 'WCS_REF'})
                
                # Eje Y
                msp.add_line((0, 0), (0, 20 * S_Y), dxfattribs={'layer': 'WCS_REF'})
                # Flecha Eje Y
                msp.add_lwpolyline([(0.5, 18 * S_Y), (0, 20 * S_Y), (-0.5, 18 * S_Y)], close=False, dxfattribs={'layer': 'WCS_REF'})
                
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            doc.saveas(file_path)
            return True
        except Exception as e:
            print(f"Error exportando DXF: {e}")
            return False
            
    def save_json_report(self, out_dir, base_name, scale, offset, ydir):
        report_data = {
            "image_used": os.path.basename(self.img_path),
            "processing_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "wcs_detected": self.wcs_info["status"] == "SUCCESS",
            "toppers_count": self.report["toppers_detected"],
            "scale_px_mm": scale,
            "offset_mm": offset,
            "y_axis_advance": ydir,
            "processing_time_s": round(self.elapsed_time, 3),
            "segmentation_method": self.segmentation_result.method if self.segmentation_result else None,
            "segmentation_model_family": "SAM 2",
            "prompt_source": self.segmentation_result.metadata.get("prompt_source") if self.segmentation_result else None,
            "prompt_count": len(self.segmentation_result.prompt_boxes) if self.segmentation_result else 0,
            "output_dxf_path": os.path.join(out_dir, f"toppers_{base_name}.dxf"),
            "mode": "SAM 2 + WCS EXPORT" if self.wcs_info["status"] == "SUCCESS" else "SAM 2 + MODO ANÁLISIS",
            "observations": "Procesamiento exitoso." if self.wcs_info["status"] == "SUCCESS" else "WCS no encontrado. Modo análisis de seguridad activo."
        }
        report_path = os.path.join(out_dir, "summary_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
            
        txt_path = os.path.join(out_dir, "summary_report.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write("           REPORTE DE PROCESAMIENTO VECTORIAL DE TOPPERS\n")
            f.write("="*70 + "\n")
            f.write(f"Imagen de entrada:  {report_data['image_used']}\n")
            f.write(f"Fecha de proceso:   {report_data['processing_date']}\n")
            f.write(f"WCS Detectado:      {report_data['wcs_detected']}\n")
            f.write(f"Modo de operación:  {report_data['mode']}\n")
            f.write(f"Toppers detectados: {report_data['toppers_count']}\n")
            f.write(f"Segmentación:       {report_data['segmentation_model_family']}\n")
            f.write(f"Origen de prompts:  {report_data['prompt_source']}\n")
            f.write(f"Escala de trabajo:  {report_data['scale_px_mm']} px/mm\n")
            f.write(f"Offset de corte:    {report_data['offset_mm']} mm\n")
            f.write(f"Avance de eje Y:    WCS_{report_data['y_axis_advance'].upper()}\n")
            f.write(f"Tiempo de ejecución: {report_data['processing_time_s']} s\n")
            f.write(f"Ruta DXF generada:  {report_data['output_dxf_path']}\n")
            f.write(f"Observaciones:      {report_data['observations']}\n")
            f.write("="*70 + "\n")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Interfaz Innokey CNC Vision")
    parser.add_argument("--image", help="Imagen que se precargará al iniciar")
    parser.add_argument(
        "--process",
        action="store_true",
        help="Procesar automáticamente la imagen indicada con --image",
    )
    args = parser.parse_args()

    root = tk.Tk()
    app = InnokeyApp(root)
    if args.image:
        def load_demo_image():
            if app.load_path(os.path.abspath(args.image)) and args.process:
                root.after(350, app.process_image)

        root.after(250, load_demo_image)
    root.mainloop()

if __name__ == "__main__":
    main()
