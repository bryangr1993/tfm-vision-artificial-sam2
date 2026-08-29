import cv2
import numpy as np
import os

from configuration import validate_operational_scale

def rectify_sheet(img, src_pts, out_dir=None, img_name="img", sheet_size=(210.0, 297.0), scale=10.0, marker_margin=10.0):
    """
    Rectifica la hoja de papel aplicando una transformación homográfica a resolución original.
    Soporta cualquier tamaño de hoja (A3, A4, A5, cuadrados, etc.) sin distorsión.
    
    Parámetros:
        img (np.ndarray): Imagen original en formato BGR.
        src_pts (list): Lista de 4 tuplas [(x, y), ...] ordenadas [TL, TR, BR, BL] en resolución original.
        out_dir (str): Directorio para guardar la hoja rectificada.
        img_name (str): Nombre de la imagen de origen.
        sheet_size (tuple): Dimensiones físicas de la hoja (ancho, alto) en mm (defecto: (210.0, 297.0) para A4).
        scale (float): Escala de salida en px/mm (defecto: 10.0).
        marker_margin (float): Margen físico en mm desde el centro del marcador de esquina al borde de la hoja (defecto: 10.0).
        
    Retorna:
        tuple: (warped_img, homography_matrix)
               warped_img: Imagen rectificada con dimensiones dinámicas según la escala y la hoja.
               homography_matrix: Matriz de homografía de 3x3.
    """
    scale = validate_operational_scale(scale)
    if len(src_pts) != 4:
        raise ValueError("Se requieren exactamente 4 puntos de origen para la homografía.")
        
    # Puntos de origen en resolución original
    src_arr = np.array(src_pts, dtype=np.float32)
    
    # Calcular dimensiones de lienzo dinámicamente según la hoja física
    sheet_w, sheet_h = sheet_size
    dst_w = int(sheet_w * scale)
    dst_h = int(sheet_h * scale)
    
    # El margen físico en píxeles
    margin_px = int(marker_margin * scale)
    
    # Mapeamos los centroides de los marcadores detectados a sus posiciones físicas reales dentro del lienzo
    # Esto preserva el área de la hoja que se encuentra por fuera de los marcadores (márgenes),
    # evitando que la marca láser L u otros elementos queden recortados o incompletos.
    dst_arr = np.array([
        [margin_px, margin_px],                          # TL (Superior Izquierda)
        [dst_w - 1 - margin_px, margin_px],              # TR (Superior Derecha)
        [dst_w - 1 - margin_px, dst_h - 1 - margin_px],  # BR (Inferior Derecha)
        [margin_px, dst_h - 1 - margin_px]               # BL (Inferior Izquierda)
    ], dtype=np.float32)
    
    # Calcular matriz de homografía
    M = cv2.getPerspectiveTransform(src_arr, dst_arr)
    
    # Aplicar warping de perspectiva en resolución original para máxima nitidez
    warped = cv2.warpPerspective(img, M, (dst_w, dst_h), flags=cv2.INTER_LANCZOS4)
    
    # Guardar imagen rectificada y matriz si se proporciona un directorio de salida
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # Guardar imagen rectificada
        img_path = os.path.join(out_dir, f"rectified_{img_name}.png")
        cv2.imwrite(img_path, warped)
        
        # Guardar matriz de homografía
        matrix_path = os.path.join(out_dir, f"homography_{img_name}.npy")
        np.save(matrix_path, M)
        print(f"  [rectify_sheet] Hoja rectificada guardada en: {img_path}")
        print(f"  [rectify_sheet] Matriz de homografía guardada en: {matrix_path}")
        
    return warped, M
