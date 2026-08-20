import argparse
import os
import sys
import time
import traceback

import cv2

# Aseguramos que la carpeta src esté en el path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_images import load_image
from detect_markers import detect_markers
from rectify_sheet import rectify_sheet
from detect_wcs_l import detect_wcs_l
from extract_contours import extract_contours
from segment_toppers import segment_toppers
from ai_pipeline import build_operational_pipeline, segment_and_extract_with_ai
from export_dxf import export_to_dxf, generate_validation_dxf
from metrics_report import save_individual_report, generate_lot_summary
from visualization import generate_scientific_figures

# Mapeo oficial de las 13 imágenes prioritarias a sus condiciones del TFM
IMAGE_CONDITIONS = {
    "13.jpg": "Caso base normal, sin L",
    "15.jpg": "Sombra superior horizontal, sin L",
    "17.jpg": "Sombra inferior, cabezal, sin L",
    "20.jpg": "Caso base normal, con L",
    "22.jpg": "Sombra superior, con L",
    "24.jpg": "Sombra inferior, cabezal visible, con L",
    "26.jpg": "Cámara recolocada, con L",
    "28.jpg": "Cámara recolocada con sombra superior, con L",
    "30.jpg": "Cámara recolocada con sombra inferior, con L",
    "32.jpg": "MDF rotado ligeramente horario, con L",
    "34.jpg": "MDF rotado ligeramente antihorario, con L",
    "36.jpg": "MDF desplazado izquierda, con L",
    "38.jpg": "MDF desplazado derecha, con L",
}

def get_image_condition(img_name):
    return IMAGE_CONDITIONS.get(img_name, "Captura experimental de toppers")

def process_single_image(img_path, args, is_batch=False):
    """
    Orquesta el procesamiento de una sola imagen a través de todo el pipeline.
    """
    img_name = os.path.basename(img_path)
    print(f"\n" + "-"*70)
    print(f" PROCESANDO IMAGEN: {img_name}")
    print(f" Condición del Experimento: {get_image_condition(img_name)}")
    print("-"*70)
    
    t_start = time.time()
    
    # 1. Rutas de salida específicas
    out_debug_dir = os.path.join(args.output, "debug")
    out_rect_dir = os.path.join(args.output, "rectified")
    out_masks_dir = os.path.join(args.output, "masks")
    out_contours_dir = os.path.join(args.output, "contours")
    out_dxf_dir = os.path.join(args.output, "dxf")
    out_reports_dir = os.path.join(args.output, "reports")
    out_figures_dir = os.path.join(args.output, "figures_tfm")
    
    # 2. Parseo de parámetros manuales del WCS si existen
    manual_origin = None
    if args.manual_wcs_origin:
        try:
            parts = [float(x) for x in args.manual_wcs_origin.split(",")]
            if len(parts) == 2:
                manual_origin = (parts[0], parts[1])
        except Exception:
            print("  [main] Error: Formato incorrecto para --manual-wcs-origin. Debe ser 'x,y'.")
            
    manual_axes = None
    if args.manual_wcs_axes:
        try:
            parts = [float(x) for x in args.manual_wcs_axes.split(",")]
            if len(parts) == 4:
                manual_axes = (parts[0], parts[1], parts[2], parts[3])
        except Exception:
            print("  [main] Error: Formato incorrecto para --manual-wcs-axes. Debe ser 'ux_x,ux_y,uy_x,uy_y'.")
            
    # Inicialización de métricas
    metrics = {
        "image_name": img_name,
        "condition": get_image_condition(img_name),
        "markers_detected": 0,
        "markers_coordinates": None,
        "wcs_status": "PENDING",
        "wcs_origin": None,
        "wcs_uX": None,
        "wcs_uY": None,
        "toppers_detected": 0,
        "segmentation_method": None,
        "discarded_count": 0,
        "time_load_ms": 0.0,
        "time_markers_ms": 0.0,
        "time_rectify_ms": 0.0,
        "time_wcs_ms": 0.0,
        "time_segment_ms": 0.0,
        "time_contours_ms": 0.0,
        "time_export_dxf_ms": 0.0,
        "time_total_ms": 0.0,
        "dxf_path": None,
        "observations": ""
    }
    
    try:
        # STEP 1: Cargar imagen y rotar
        t0 = time.time()
        img = load_image(img_path, args.rotate)
        if img is None:
            metrics["observations"] = "Fallo al cargar imagen"
            return metrics
        metrics["time_load_ms"] = (time.time() - t0) * 1000
        
        # STEP 2: Detección de marcadores
        t0 = time.time()
        markers = detect_markers(img, out_debug_dir, img_name.replace('.jpg', ''))
        if markers is None:
            metrics["observations"] = "Marcadores insuficientes (< 4)"
            return metrics
        metrics["markers_detected"] = len(markers)
        metrics["markers_coordinates"] = markers
        metrics["time_markers_ms"] = (time.time() - t0) * 1000
        
        # STEP 3: Rectificación por homografía
        t0 = time.time()
        
        # Obtener dimensiones físicas de la hoja de forma dinámica
        sheet_w, sheet_h = 210.0, 297.0
        if hasattr(args, "sheet_size") and args.sheet_size:
            try:
                parts = args.sheet_size.split(",")
                if len(parts) == 2:
                    sheet_w = float(parts[0])
                    sheet_h = float(parts[1])
            except Exception:
                print("  [main] Error parseando --sheet-size. Usando A4 (210x297 mm) por defecto.")
                
        rectified, M = rectify_sheet(img, markers, out_rect_dir, img_name.replace('.jpg', ''),
                                     sheet_size=(sheet_w, sheet_h), scale=args.scale,
                                     marker_margin=args.marker_margin)
        metrics["time_rectify_ms"] = (time.time() - t0) * 1000
        
        # STEP 4: Detección del WCS
        t0 = time.time()
        wcs = detect_wcs_l(rectified, out_debug_dir, img_name.replace('.jpg', ''), 
                           manual_origin, manual_axes,
                           marker_margin=args.marker_margin, scale=args.scale)
        metrics["wcs_status"] = wcs["status"]
        metrics["wcs_origin"] = wcs["origin"]
        metrics["wcs_uX"] = wcs["uX"]
        metrics["wcs_uY"] = wcs["uY"]
        metrics["time_wcs_ms"] = (time.time() - t0) * 1000
        
        # STEP 5 y 6: Segmentación operativa y extracción de contornos
        t0 = time.time()
        if args.segmenter == "sam2":
            if not hasattr(args, "_ai_pipeline"):
                args._ai_pipeline = build_operational_pipeline(
                    checkpoint=args.sam_checkpoint,
                    model_config=args.sam_config,
                    device=args.device,
                    box_margin_fraction=args.sam_box_margin,
                )
            result, contours, report = segment_and_extract_with_ai(
                args._ai_pipeline,
                rectified,
                scale=args.scale,
                wcs_info=wcs,
                debug_dir=out_contours_dir,
                image_name=img_name.replace('.jpg', ''),
            )
            mask = result.mask
            metrics["segmentation_method"] = result.method
        else:
            mask = segment_toppers(
                rectified,
                out_masks_dir,
                img_name.replace('.jpg', ''),
                scale=args.scale,
                wcs_info=wcs,
            )
            contours, report = extract_contours(
                mask,
                rectified,
                out_contours_dir,
                img_name.replace('.jpg', ''),
            )
            metrics["segmentation_method"] = "classical_baseline"
        os.makedirs(out_masks_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_masks_dir, f"{img_name.rsplit('.', 1)[0]}_{args.segmenter}_mask.png"), mask)
        metrics["time_segment_ms"] = (time.time() - t0) * 1000
        metrics["toppers_detected"] = report["toppers_detected"]
        metrics["discarded_count"] = report["discarded_components_count"]
        metrics["time_contours_ms"] = 0.0
        
        # STEP 7: Exportación DXF
        t0 = time.time()
        if wcs["status"] == "SUCCESS":
            dxf_filename = f"toppers_{img_name.replace('.jpg', '.dxf')}"
            dxf_path = os.path.join(out_dxf_dir, dxf_filename)
            success_dxf = export_to_dxf(contours, wcs, dxf_path, args.scale, args.offset, args.wcs_y_direction)
            if success_dxf:
                metrics["dxf_path"] = dxf_path
                metrics["observations"] = "Procesamiento y DXF exitoso"
            else:
                metrics["observations"] = "Fallo en exportación DXF"
            
            # Generar DXF de calibración de ejes si es necesario
            val_dxf_path = os.path.join(out_dxf_dir, "validation_wcs_axes.dxf")
            if not os.path.exists(val_dxf_path):
                generate_validation_dxf(val_dxf_path, args.wcs_y_direction)
        else:
            # Modo Análisis
            metrics["observations"] = "WCS_NOT_FOUND (Modo Análisis - Sin DXF WCS)"
            print(f"  [main] Advertencia: {wcs['message']}. Ejecutando en Modo Análisis.")
            
        metrics["time_export_dxf_ms"] = (time.time() - t0) * 1000
        
        # STEP 8: Renderizado de Figuras Académicas para LaTeX
        generate_scientific_figures(img, markers, rectified, wcs, mask, contours, 
                                     report, out_figures_dir, img_name.replace('.jpg', ''),
                                     args.wcs_y_direction, args.offset)
        
        metrics["time_total_ms"] = (time.time() - t_start) * 1000
        print(f"  [main] Tiempo total: {metrics['time_total_ms']:.2f} ms")
        
        # STEP 9: Reporte Individual JSON
        save_individual_report(img_name, metrics, out_reports_dir)
        
        return metrics
        
    except Exception as e:
        print(f"  [main] Error catastrófico procesando {img_name}: {e}")
        traceback.print_exc()
        metrics["observations"] = f"Error: {str(e)}"
        metrics["time_total_ms"] = (time.time() - t_start) * 1000
        save_individual_report(img_name, metrics, out_reports_dir)
        return metrics

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Visión Artificial Industrial para Toppers Innokey")
    
    parser.add_argument("--input", type=str, default="data/raw",
                        help="Directorio con imágenes de entrada (jpg/png)")
    parser.add_argument("--image", type=str, default=None,
                        help="Ruta a una imagen específica de entrada")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Directorio de salida para los resultados")
    parser.add_argument("--scale", type=float, default=10.0,
                        help="Escala del lienzo rectificado en px/mm (defecto: 10.0)")
    parser.add_argument("--sheet-size", type=str, default="210,297",
                        help="Dimensiones físicas de la hoja en mm como 'ancho,alto' (defecto: '210,297' para A4)")
    parser.add_argument("--marker-margin", type=float, default=10.0,
                        help="Margen físico del centro del marcador al borde de la hoja en mm (defecto: 10.0)")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Compensación exterior de corte en mm (defecto: 0.0)")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                        help="Forzar rotación horaria de la imagen de entrada (grados)")
    parser.add_argument("--wcs-y-direction", type=str, default="down", choices=["down", "up"],
                        help="Sentido de avance del eje Y de la cortadora (defecto: 'down')")
    parser.add_argument("--segmenter", type=str, default="sam2", choices=["sam2", "classical"],
                        help="Segmentador. SAM 2 es el flujo operativo; classical solo reproduce la línea base.")
    parser.add_argument("--sam-checkpoint", type=str, default=None,
                        help="Ruta opcional al checkpoint de SAM 2.")
    parser.add_argument("--sam-config", type=str, default="configs/sam2/sam2_hiera_t.yaml",
                        help="Configuración de la variante SAM 2.")
    parser.add_argument("--sam-box-margin", type=float, default=0.05,
                        help="Margen proporcional aplicado a las cajas de prompt.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Dispositivo de inferencia para SAM 2.")
    parser.add_argument("--manual-wcs-origin", type=str, default=None,
                        help="Origen forzado manual del WCS en píxeles rectificados 'x,y'")
    parser.add_argument("--manual-wcs-axes", type=str, default=None,
                        help="Vectores base manuales 'ux_x,ux_y,uy_x,uy_y'")
                        
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" INICIALIZANDO PIPELINE DE VISIÓN ARTIFICIAL - TFM INNOKEY")
    print("="*80)
    print(f"  • Directorio Input: {args.input}")
    print(f"  • Directorio Output: {args.output}")
    print(f"  • Dimensiones Hoja: {args.sheet_size} mm")
    print(f"  • Margen Marcador: {args.marker_margin} mm")
    print(f"  • Escala de Trabajo: {args.scale} px/mm")
    print(f"  • Offset de Corte: {args.offset} mm")
    print(f"  • Dirección del Eje Y: WCS_{args.wcs_y_direction.upper()}")
    print(f"  • Segmentación: {'SAM 2 (operativa)' if args.segmenter == 'sam2' else 'Visión clásica (línea base)'}")
    print("="*80 + "\n")
    
    # Si se procesa una imagen individual
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: No se encontró la imagen en {args.image}")
            sys.exit(1)
        results = [process_single_image(args.image, args)]
        
    # Si se procesa un directorio completo por lotes
    else:
        if not os.path.exists(args.input):
            print(f"Error: El directorio de entrada {args.input} no existe.")
            sys.exit(1)
            
        # Buscar todas las imágenes válidas jpg/jpeg/png
        extensions = [".jpg", ".jpeg", ".png"]
        all_files = os.listdir(args.input)
        img_files = [f for f in all_files if os.path.splitext(f.lower())[1] in extensions]
        
        # Filtramos para priorizar procesar primero las que pertenecen a la base de datos oficial
        official_images = ["13.jpg", "15.jpg", "17.jpg", "20.jpg", "22.jpg", "24.jpg", "26.jpg", "28.jpg", "30.jpg", "32.jpg", "34.jpg", "36.jpg", "38.jpg"]
        other_images = [f for f in img_files if f not in official_images]
        
        # Procesamos en el orden oficial estructurado para el TFM
        img_processing_list = [f for f in official_images if f in img_files] + other_images
        
        if len(img_processing_list) == 0:
            print(f"Advertencia: No se encontraron imágenes válidas en {args.input}")
            sys.exit(0)
            
        print(f"Detectadas {len(img_processing_list)} imágenes para procesamiento por lotes.")
        
        results = []
        for img_name in img_processing_list:
            img_path = os.path.join(args.input, img_name)
            res = process_single_image(img_path, args, is_batch=True)
            results.append(res)
            
        # Generar reporte de resumen de lote
        reports_dir = os.path.join(args.output, "reports")
        generate_lot_summary(results, reports_dir)

if __name__ == "__main__":
    main()
