import cv2
import numpy as np
import os

from configuration import validate_operational_scale

def segment_toppers(rectified_img, debug_dir=None, img_name="img", scale=10.0, wcs_info=None):
    """
    Segmenta de forma robusta las siluetas exteriores de los toppers de color sobre la hoja rectificada.
    
    Parámetros:
        rectified_img (np.ndarray): Imagen rectificada.
        debug_dir (str): Directorio para guardar depuración visual.
        img_name (str): Nombre del archivo original.
        scale (float): Escala de trabajo en px/mm (defecto: 10.0).
        wcs_info (dict): Información del WCS para excluir la marca quirúrgicamente.
        
    Retorna:
        np.ndarray: Máscara binaria (255 para toppers, 0 para fondo).
    """
    scale = validate_operational_scale(scale)
    h, w = rectified_img.shape[:2]
    
    # 1. Definición de la máscara de exclusión (protección de zonas que no son toppers)
    keep_mask = np.ones((h, w), dtype=np.uint8) * 255
    
    # Calcular márgenes dinámicos en base a la escala física (px/mm)
    margin_px = int(8.0 * scale)    # Margen perimetral estrecho de 8 mm
    marker_px = int(18.0 * scale)   # Zona de marcadores de esquina de 18 mm
    
    # Evitar recortar toppers exteriores aplicando los márgenes perimetrales
    keep_mask[0:margin_px, :] = 0
    keep_mask[h-margin_px:h, :] = 0
    keep_mask[:, 0:margin_px] = 0
    keep_mask[:, w-margin_px:w] = 0
    
    # Marcadores de esquina
    keep_mask[0:marker_px, 0:marker_px] = 0
    keep_mask[0:marker_px, w-marker_px:w] = 0
    keep_mask[h-marker_px:h, w-marker_px:w] = 0
    keep_mask[h-marker_px:h, 0:marker_px] = 0
    
    # Exclusión quirúrgica de la marca en L del WCS para no obstruir los toppers vecinos (ej. la estrella)
    if wcs_info is not None and wcs_info.get("status") == "SUCCESS":
        ox, oy = wcs_info["origin"]
        ux_x, ux_y = wcs_info["uX"]
        uy_x, uy_y = wcs_info["uY"]
        
        # Excluir brazo horizontal (longitud 23 mm = 230 px, espesor de seguridad de 3.5 mm = 35 px)
        p_x_end = (int(ox + ux_x * 230), int(oy + ux_y * 230))
        cv2.line(keep_mask, (int(ox), int(oy)), p_x_end, 0, thickness=int(3.5 * scale))
        
        # Excluir brazo vertical
        p_y_end = (int(ox + uy_x * 230), int(oy + uy_y * 230))
        cv2.line(keep_mask, (int(ox), int(oy)), p_y_end, 0, thickness=int(3.5 * scale))
    
    # 2. Segmentación Multi-Criterio basada en HSV (Generosa y Robusta a Sombras)
    hsv = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    
    # A. Filtro de Saturación (S > 45): Captura colores vivos y pasteles (amarillo claro, rosa, tonos piel)
    sat_mask = (S > 45).astype(np.uint8) * 255
    
    # B. Filtro de Luminosidad (V < 115): Captura textos, sombras finas y contornos oscuros
    dark_mask = (V < 115).astype(np.uint8) * 255
    
    # C. Detector de Transiciones Canny (Garantiza contornos de siluetas blancas o claras)
    gray = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    # Dilatación de bordes para sellar y engrosar las fronteras de corte
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_edges = cv2.dilate(edges, kernel_dilate)
    
    # 3. Fusión de Criterios y Enmascarado de Protección Quirúrgico
    combined_mask = cv2.bitwise_or(sat_mask, dark_mask)
    combined_mask = cv2.bitwise_or(combined_mask, dilated_edges)
    combined_mask = cv2.bitwise_and(combined_mask, keep_mask)
    
    # 4. Operaciones Morfológicas de Consolidación
    
    # Cierre de escala estrecha para unir trazos finos sin fusionar toppers adyacentes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Inundación de fondo (Flood Fill) para rellenar de forma compacta los interiores
    # de toppers blancos/claros (ej. el unicornio y las nubes del arcoíris)
    mask_flood = mask_closed.copy()
    floodfill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(mask_flood, floodfill_mask, (0, 0), 255)
    mask_inv = cv2.bitwise_not(mask_flood)
    mask_filled = cv2.bitwise_or(mask_closed, mask_inv)
    
    # Apertura morfológica para purificar y alisar los bordes de corte vectoriales
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_opened = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel_open)
    
    # 5. Filtrado por Componentes Conectados (Eliminación de ruido pequeño residual)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_opened)
    final_mask = np.zeros_like(mask_opened)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Conservar solo formas con área física relevante (mayor a 5000 píxeles cuadrados)
        if area >= 5000:
            final_mask[labels_im == i] = 255
            
    # 6. Guardar depuración visual
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        
        # Guardar máscara binaria
        mask_path = os.path.join(debug_dir, f"{img_name}_mask.png")
        cv2.imwrite(mask_path, final_mask)
        
        # Guardar visualización sobre la imagen rectificada (color morado translúcido)
        overlay = rectified_img.copy()
        overlay[final_mask == 255] = [255, 0, 255] # Pintar toppers morados
        debug_overlay = cv2.addWeighted(rectified_img, 0.7, overlay, 0.3, 0)
        
        # Dibujar líneas de exclusión (márgenes y zonas especiales) en rojo traslúcido
        excl_overlay = debug_overlay.copy()
        excl_overlay[keep_mask == 0] = [0, 0, 255]
        cv2.addWeighted(debug_overlay, 0.8, excl_overlay, 0.2, 0, dst=debug_overlay)
        
        overlay_path = os.path.join(debug_dir, f"{img_name}_segmentation_overlay.png")
        cv2.imwrite(overlay_path, debug_overlay)
        print(f"  [segment_toppers] Máscara y overlay guardados en: {debug_dir}")
        
    return final_mask
