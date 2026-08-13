import cv2
import numpy as np
import os

def detect_wcs_l(rectified_img, debug_dir=None, img_name="img", 
                 manual_origin=None, manual_axes=None, marker_margin=10.0, scale=10.0):
    """
    Detecta la marca láser en forma de L para definir el origen del WCS y la dirección de los ejes.
    
    Parámetros:
        rectified_img (np.ndarray): Imagen A4 rectificada (2100x2970 px).
        debug_dir (str): Directorio de depuración.
        img_name (str): Nombre de la imagen.
        manual_origin (tuple): Origen manual (x, y) en píxeles si se fuerza.
        manual_axes (tuple): Vectores base manuales (ux_x, ux_y, uy_x, uy_y) si se fuerzan.
        
    Retorna:
        dict: Diccionario con la información del WCS:
              {
                "status": "SUCCESS" o "WCS_NOT_FOUND",
                "origin": (x, y) en píxeles rectificados,
                "uX": (ux_x, ux_y) vector unitario del eje X,
                "uY": (uy_x, uy_y) vector unitario del eje Y,
                "message": Mensaje descriptivo
              }
    """
    # 1. Modo Manual Forzado
    if manual_origin is not None:
        uX = (1.0, 0.0)
        uY = (0.0, 1.0)
        if manual_axes is not None and len(manual_axes) == 4:
            uX = (manual_axes[0], manual_axes[1])
            uY = (manual_axes[2], manual_axes[3])
            
        print(f"  [detect_wcs_l] Usando WCS configurado manualmente: Origen={manual_origin}, uX={uX}, uY={uY}")
        
        # Guardar visualización del WCS manual
        if debug_dir:
            save_wcs_debug(rectified_img, manual_origin, uX, uY, debug_dir, img_name, manual=True)
            
        return {
            "status": "SUCCESS",
            "origin": manual_origin,
            "uX": uX,
            "uY": uY,
            "message": "Manual override applied"
        }
        
    H, W = rectified_img.shape[:2]
    # Calcular el margen físico en píxeles donde está mapeado el centroide del marcador TL
    margin_px = int(marker_margin * scale)
    
    # ROI dinámica y generalizada de gran amplitud para buscar la marca WCS L
    # en toda la esquina superior izquierda de la hoja (lienzo virtual).
    # Esto tolera variaciones de posición "al ojo" del operario y permite que la L
    # quede grabada por dentro o por fuera de los marcadores de registro de esquina.
    # El rango X e Y abarca un área de búsqueda acotada de 45 mm x 45 mm (450x450 px)
    # cubriendo de forma segura el cuadrante de calibración y evitando falsos positivos con los toppers.
    roi_x_start = 0
    roi_x_end = min(W, 450)
    roi_y_start = 0
    roi_y_end = min(H, 450)
    
    roi = rectified_img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Mejorar contraste local para resaltar las marcas del grabado láser tenue
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_roi = clahe.apply(gray_roi)
    
    # Binarización adaptativa para separar trazos oscuros finos bajo sombras cambiantes
    thresh = cv2.adaptiveThreshold(enhanced_roi, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 51, 15)

    # Filtrar el área del marcador de esquina M1 para evitar falsos positivos con sus bordes cuadrados
    m_start_x = max(0, margin_px - 85)
    m_end_x = min(roi_x_end, margin_px + 85)
    m_start_y = max(0, margin_px - 85)
    m_end_y = min(roi_y_end, margin_px + 85)
    thresh[m_start_y:m_end_y, m_start_x:m_end_x] = 0
    
    # Detectar segmentos de línea usando Hough lineal probabilístico
    lines = cv2.HoughLinesP(thresh, rho=1, theta=np.pi/180, 
                            threshold=40, minLineLength=30, maxLineGap=10)
    
    if lines is None or len(lines) == 0:
        print("  [detect_wcs_l] No se detectaron segmentos de líneas en la ROI de la L.")
        return {
            "status": "WCS_NOT_FOUND",
            "origin": None,
            "uX": None,
            "uY": None,
            "message": "No lines detected in ROI"
        }
        
    # Clasificación de segmentos de líneas en candidatos horizontales y verticales
    horiz_segments = []
    vert_segments = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        length = np.hypot(dx, dy)
        
        # Filtro físico: la marca grabada por el láser tiene brazos de 15-20 mm (150-200 px).
        # Cualquier segmento mayor a 25 mm (250 px) pertenece a bordes impresos de la hoja o al chasis.
        # Cualquier segmento menor a 2.5 mm (25 px) es ruido.
        if length < 25 or length > 250:
            continue
            
        angle = np.abs(np.degrees(np.arctan2(dy, dx)))
        # Mapeamos ángulos al rango [0, 90]
        if angle > 90:
            angle = 180 - angle
            
        # Clasificar según ángulo
        if angle <= 20:
            horiz_segments.append((length, x1, y1, x2, y2))
        elif angle >= 70:
            vert_segments.append((length, x1, y1, x2, y2))
            
    # Ordenamos por longitud de forma descendente para evaluar los segmentos más dominantes
    horiz_segments.sort(key=lambda x: x[0], reverse=True)
    vert_segments.sort(key=lambda x: x[0], reverse=True)
    
    found_l = False
    best_wcs = None
    
    # Búsqueda combinatoria robusta RANSAC-like evaluando los segmentos más largos
    # (revisando las mejores 8 combinaciones horizontales y verticales)
    for h_idx, (h_len, h_x1, h_y1, h_x2, h_y2) in enumerate(horiz_segments[:8]):
        for v_idx, (v_len, v_x1, v_y1, v_x2, v_y2) in enumerate(vert_segments[:8]):
            
            # --- EXTRACCIÓN DE NUBE DE PÍXELES DEL BRAZO HORIZONTAL (Promediado de Ruido por Smoke) ---
            # Definimos un corredor de holgura alrededor del segmento Hough en la máscara adaptativa
            h_ymin = max(0, min(h_y1, h_y2) - 15)
            h_ymax = min(roi_y_end - roi_y_start, max(h_y1, h_y2) + 15)
            h_xmin = max(0, min(h_x1, h_x2) - 10)
            h_xmax = min(roi_x_end - roi_x_start, max(h_x1, h_x2) + 10)
            
            # Recuperar coordenadas de píxeles activos (255) y trasladarlas a rectified space
            h_pix_y, h_pix_x = np.where(thresh[h_ymin:h_ymax, h_xmin:h_xmax] == 255)
            h_pix_x = h_pix_x + h_xmin + roi_x_start
            h_pix_y = h_pix_y + h_ymin + roi_y_start
            
            # Si hay suficientes píxeles en el corredor (ancho real), hacemos ajuste robusto de mínimos cuadrados,
            # de lo contrario recurrimos a los extremos como fallback seguro.
            if len(h_pix_x) > 40:
                h_pts = np.column_stack((h_pix_x, h_pix_y)).astype(np.float32)
            else:
                h_pts = np.array([[h_x1 + roi_x_start, h_y1 + roi_y_start], [h_x2 + roi_x_start, h_y2 + roi_y_start]], dtype=np.float32)
                
            # --- EXTRACCIÓN DE NUBE DE PÍXELES DEL BRAZO VERTICAL (Promediado de Ruido por Smoke) ---
            v_ymin = max(0, min(v_y1, v_y2) - 10)
            v_ymax = min(roi_y_end - roi_y_start, max(v_y1, v_y2) + 10)
            v_xmin = max(0, min(v_x1, v_x2) - 15)
            v_xmax = min(roi_x_end - roi_x_start, max(v_x1, v_x2) + 15)
            
            v_pix_y, v_pix_x = np.where(thresh[v_ymin:v_ymax, v_xmin:v_xmax] == 255)
            v_pix_x = v_pix_x + v_xmin + roi_x_start
            v_pix_y = v_pix_y + v_ymin + roi_y_start
            
            if len(v_pix_x) > 40:
                v_pts = np.column_stack((v_pix_x, v_pix_y)).astype(np.float32)
            else:
                v_pts = np.array([[v_x1 + roi_x_start, v_y1 + roi_y_start], [v_x2 + roi_x_start, v_y2 + roi_y_start]], dtype=np.float32)
                
            # Ajustar rectas mediante estimación robusta Huber sobre la nube completa de píxeles
            # (reducción estadística del error angular y filtrado de ruido de smoke)
            vx_h, vy_h, x_h, y_h = cv2.fitLine(h_pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
            vx_v, vy_v, x_v, y_v = cv2.fitLine(v_pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
            
            # Normalizar sentidos hacia los cuadrantes de trabajo (+X, +Y)
            if vx_h[0] < 0:
                vx_h, vy_h = -vx_h, -vy_h
            if vy_v[0] < 0:
                vx_v, vy_v = -vx_v, -vy_v
                
            # Calcular vectores unitarios crudos para el filtro de validación ortogonal
            raw_uX = np.array([float(vx_h[0]), float(vy_h[0])])
            raw_uX /= np.linalg.norm(raw_uX)
            raw_uY = np.array([float(vx_v[0]), float(vy_v[0])])
            raw_uY /= np.linalg.norm(raw_uY)
            
            # Filtrar alineaciones crudas no ortogonales (p. ej. si no es una forma en L)
            raw_dot_product = np.abs(np.dot(raw_uX, raw_uY))
            if raw_dot_product > 0.45: # Tolerancia ortogonal cruda de cos(63°)
                continue
                
            # ORTOGONALIZACIÓN MATEMÁTICA ESTRICTA (90.00°):
            # Promediamos la desviación angular de ambos brazos para regularizar y mitigar
            # el efecto del humo difuminado (smoke blur).
            theta_h = np.arctan2(raw_uX[1], raw_uX[0])
            theta_v = np.arctan2(raw_uY[1], raw_uY[0])
            theta_v_est = theta_v - np.pi/2
            
            # Ángulo de rotación promedio corregido
            avg_theta = (theta_h + theta_v_est) / 2.0
            
            # Construir vectores base ortonormales garantizados
            uX = np.array([np.cos(avg_theta), np.sin(avg_theta)])
            uY = np.array([-np.sin(avg_theta), np.cos(avg_theta)])
            
            # Resolver intersección exacta para el origen del WCS (vértice)
            # t = uY[1]*(x_v - x_h) - uY[0]*(y_v - y_h)
            t = uY[1] * (x_v[0] - x_h[0]) - uY[0] * (y_v[0] - y_h[0])
            orig_x = float(x_h[0] + t * uX[0])
            orig_y = float(y_h[0] + t * uX[1])
            origin = (orig_x, orig_y)
            
            try:
                # --- FILTRO 1: MARGEN DE SEGURIDAD FÍSICA CONTRA MARCOS IMPRESOS ---
                # Exigimos una holgura mínima absoluta de 6.0 mm (60 px) desde el borde físico del lienzo (0,0).
                # Esto descarta de forma absoluta la esquina del marco impreso a (40, 30).
                if origin[0] < 60 or origin[1] < 60:
                    continue
                    
                # --- FILTRO 2: PROXIMIDAD DE EXTREMOS (SHAPE MATCHING) ---
                # Para que sea un verdadero grabado en L, las rectas ajustadas deben intersectarse
                # en o muy cerca de los extremos físicos de los segmentos reales detectados.
                # Calculamos la distancia mínima del vértice 'origin' a las puntas de los brazos.
                min_dist_h = min(np.hypot(orig_x - (h_x1 + roi_x_start), orig_y - (h_y1 + roi_y_start)),
                                 np.hypot(orig_x - (h_x2 + roi_x_start), orig_y - (h_y2 + roi_y_start)))
                min_dist_v = min(np.hypot(orig_x - (v_x1 + roi_x_start), orig_y - (v_y1 + roi_y_start)),
                                 np.hypot(orig_x - (v_x2 + roi_x_start), orig_y - (v_y2 + roi_y_start)))
                
                # Los extremos deben coincidir con la intersección a menos de 4.5 mm (45 px)
                if min_dist_h > 45 or min_dist_v > 45:
                    continue
                    
                # --- FILTRO 3: PROXIMIDAD GENERAL AL MARCADOR TL ---
                # El operario colocará por diseño el WCS "cercano al TL".
                # Aumentamos la tolerancia máxima a 75 mm (750 px) para dar amplio margen a la colocación manual "al ojo".
                dist_to_tl = np.hypot(orig_x - margin_px, orig_y - margin_px)
                if dist_to_tl > 750:
                    continue
                    
                # Si pasa todos los filtros, hemos hallado un L-mark candidato robusto.
                # Favorecemos la combinación que maximice la longitud acumulada de los brazos.
                combined_len = h_len + v_len
                if best_wcs is None or combined_len > best_wcs["score"]:
                    best_wcs = {
                        "score": combined_len,
                        "origin": origin,
                        "uX": tuple(uX),
                        "uY": tuple(uY),
                        "dot": raw_dot_product,
                        "angle": np.degrees(np.arccos(np.clip(raw_dot_product, -1.0, 1.0))),
                        "dist_to_tl": dist_to_tl
                    }
                    found_l = True
            except np.linalg.LinAlgError:
                continue
                
    if not found_l:
        print("  [detect_wcs_l] No se detectó ninguna marca en L geométricamente válida y cercana al TL.")
        return {
            "status": "WCS_NOT_FOUND",
            "origin": None,
            "uX": None,
            "uY": None,
            "message": "No valid physical L-shape matching in ROI"
        }
        
    origin = best_wcs["origin"]
    uX = best_wcs["uX"]
    uY = best_wcs["uY"]
    actual_angle = best_wcs["angle"]
    
    print(f"  [detect_wcs_l] ¡Marca L Detectada Exitosamente y validada físicamente!")
    print(f"    Vértice (Origen): {origin}")
    print(f"    Eje X Unitario (uX): {uX}")
    print(f"    Eje Y Unitario (uY): {uY}")
    print(f"    Ángulo entre ejes: {actual_angle:.1f}° (dot={best_wcs['dot']:.3f})")
    print(f"    Distancia al TL: {best_wcs['dist_to_tl']:.1f} px ({best_wcs['dist_to_tl']/10.0:.1f} mm)")
    
    # Guardar depuración visual
    if debug_dir:
        save_wcs_debug(rectified_img, origin, uX, uY, debug_dir, img_name, manual=False)
        
    return {
        "status": "SUCCESS",
        "origin": origin,
        "uX": uX,
        "uY": uY,
        "message": "Automatic WCS L-mark detection successful and physically validated"
    }

def save_wcs_debug(img, origin, uX, uY, debug_dir, img_name, manual=False):
    """Dibuja y guarda los ejes WCS sobre la imagen rectificada."""
    debug_img = img.copy()
    ox, oy = int(origin[0]), int(origin[1])
    
    # Dibujar origen
    cv2.circle(debug_img, (ox, oy), 12, (0, 0, 255), -1) # Círculo rojo en origen
    
    # Dibujar brazo X (azul) - longitud visual 150px
    x_end = (int(ox + uX[0] * 150), int(oy + uX[1] * 150))
    cv2.arrowedLine(debug_img, (ox, oy), x_end, (255, 0, 0), 4, tipLength=0.2)
    cv2.putText(debug_img, "X (WCS)", (x_end[0] + 10, x_end[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Dibujar brazo Y (verde) - longitud visual 150px
    y_end = (int(ox + uY[0] * 150), int(oy + uY[1] * 150))
    cv2.arrowedLine(debug_img, (ox, oy), y_end, (0, 255, 0), 4, tipLength=0.2)
    cv2.putText(debug_img, "Y (WCS)", (y_end[0], y_end[1] + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Título indicativo
    label = "WCS MANUAL" if manual else "WCS AUTOMATICO (L-Mark)"
    cv2.putText(debug_img, label, (50, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255) if manual else (0, 200, 0), 3)
    
    # Guardar en la carpeta debug
    os.makedirs(debug_dir, exist_ok=True)
    out_path = os.path.join(debug_dir, f"{img_name}_wcs_axes.png")
    cv2.imwrite(out_path, debug_img)
    print(f"  [detect_wcs_l] Visualización del WCS guardada en: {out_path}")
