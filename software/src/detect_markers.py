import cv2
import numpy as np
import os
import itertools

def is_valid_quad(pts):
    """
    Verifica si el cuadrilátero formado por las 4 esquinas es geométricamente válido:
    1. Es estrictamente convexo.
    2. Sus 4 ángulos internos se encuentran entre 50 y 130 grados (cercanos a 90°).
    3. Sus ratios de lados opuestos están en el rango [0.6, 1.66], evitando trapecios exagerados
       o cuñas colineales degeneradas bajo perspectiva real, mientras mantiene invarianza dimensional
       (soportando hojas A3, A4, A5, cuadradas o rectangulares de cualquier proporción).
    
    Parámetros:
        pts (list): Lista de 4 tuplas (x, y) de las esquinas.
        
    Retorna:
        bool: True si el cuadrilátero es geométricamente válido, False en caso contrario.
    """
    if len(pts) != 4:
        return False
        
    pts_arr = np.array(pts, dtype=np.float32)
    
    # 1. Chequeo de convexidad
    # Ordenamos temporalmente los puntos de forma horaria usando ángulo polar desde el centroide
    cx = np.mean(pts_arr[:, 0])
    cy = np.mean(pts_arr[:, 1])
    angles = np.arctan2(pts_arr[:, 1] - cy, pts_arr[:, 0] - cx)
    sort_idx = np.argsort(angles)
    sorted_pts = pts_arr[sort_idx]
    
    if not cv2.isContourConvex(sorted_pts.astype(np.int32)):
        return False
        
    # 2. Chequeo de ángulos internos en los 4 vértices
    for i in range(4):
        p1 = sorted_pts[i - 1]          # Vértice anterior
        p2 = sorted_pts[i]              # Vértice actual
        p3 = sorted_pts[(i + 1) % 4]    # Vértice siguiente
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return False
            
        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        
        # El ángulo debe ser relativamente cercano a un ángulo recto (50 a 130 grados)
        if angle < 50 or angle > 130:
            return False
            
    # 3. Chequeo de ratios de lados opuestos (evita trapecios altamente deformados o cuñas)
    # Lados en sorted_pts (orden horario):
    # Lado 0: sorted_pts[0] -> sorted_pts[1] (Arriba)
    # Lado 1: sorted_pts[1] -> sorted_pts[2] (Derecha)
    # Lado 2: sorted_pts[2] -> sorted_pts[3] (Abajo)
    # Lado 3: sorted_pts[3] -> sorted_pts[0] (Izquierda)
    len0 = np.linalg.norm(sorted_pts[1] - sorted_pts[0])
    len1 = np.linalg.norm(sorted_pts[2] - sorted_pts[1])
    len2 = np.linalg.norm(sorted_pts[3] - sorted_pts[2])
    len3 = np.linalg.norm(sorted_pts[0] - sorted_pts[3])
    
    if len0 == 0 or len1 == 0 or len2 == 0 or len3 == 0:
        return False
        
    ratio_top_bottom = len0 / len2
    ratio_left_right = len3 / len1
    
    # Tolerancia amplia para perspectiva y distorsión de cámara, pero excluyendo formas no rectangulares
    if ratio_top_bottom < 0.6 or ratio_top_bottom > 1.66:
        return False
    if ratio_left_right < 0.6 or ratio_left_right > 1.66:
        return False
        
    return True

def detect_markers(img, debug_dir=None, img_name="img"):
    """
    Detecta los 4 marcadores negros de esquina de la hoja de forma robusta.
    Aplica filtros de vecindario blanco, busca combinaciones óptimas de candidatos
    y valida la geometría cuadrangular general del conjunto.
    
    Parámetros:
        img (np.ndarray): Imagen de entrada en formato BGR.
        debug_dir (str): Directorio para guardar imágenes de depuración.
        img_name (str): Nombre de la imagen para nombrar el archivo de depuración.
        
    Retorna:
        list: Lista de 4 tuplas [(x, y), ...] ordenadas como [TL, TR, BR, BL] en resolución original,
              o None si falla la detección.
    """
    orig_h, orig_w = img.shape[:2]
    
    # Redimensionamiento temporal para acelerar y estabilizar la detección
    target_w = 2000
    scale = target_w / orig_w
    target_h = int(orig_h * scale)
    resized = cv2.resize(img, (target_w, target_h))
    
    # Conversión a escala de grises
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Método 1: Umbral binario simple (los marcadores negros son muy oscuros)
    _, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    def filter_contours(cnts):
        cands = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            # Filtro de área ajustado para marcadores cuadrados en escala reducida (80 a 1500 px^2)
            if area < 80 or area > 1500:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            extent = float(area) / (w * h)
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Filtro geométrico relajado para soportar cuadriláteros deformados por perspectiva
            if 0.60 <= aspect_ratio <= 1.60 and extent >= 0.50 and solidity >= 0.70:
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_val = cv2.mean(gray, mask=mask)[0]
                
                # El marcador negro debe ser oscuro
                if mean_val < 95:
                    # --- FILTRO DE VECINDARIO SOBRE FONDO BLANCO ---
                    # Expandimos la caja delimitadora por 25 píxeles en todas direcciones
                    x_ext = max(0, x - 25)
                    y_ext = max(0, y - 25)
                    w_ext = min(gray.shape[1] - x_ext, w + 50)
                    h_ext = min(gray.shape[0] - y_ext, h + 50)
                    
                    # Crear máscara de vecindario expandido
                    neigh_mask = np.zeros(gray.shape, dtype=np.uint8)
                    neigh_mask[y_ext:y_ext+h_ext, x_ext:x_ext+w_ext] = 255
                    # Excluimos el propio contorno del marcador negro restando su máscara
                    neigh_mask = cv2.subtract(neigh_mask, mask)
                    
                    # Calcular el promedio de intensidad del vecindario
                    neigh_mean = cv2.mean(gray, mask=neigh_mask)[0]
                    
                    # Si el vecindario es brillante (hoja blanca de papel >= 170)
                    # Esto descarta falsos positivos sobre chasis metálicos o cabezal rodeados de fondo oscuro (< 120)
                    if neigh_mean >= 165:
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cx_orig = int(cx / scale)
                            cy_orig = int(cy / scale)
                            # Guardamos: x_orig, y_orig, x_resized, y_resized, area, neigh_mean, cnt
                            cands.append((cx_orig, cy_orig, cx, cy, area, neigh_mean, cnt))
        return cands

    candidates = filter_contours(contours)
    
    # Fallback: Si no se encuentran suficientes candidatos, probar umbral adaptativo
    if len(candidates) < 4:
        print(f"  [detect_markers] Solo {len(candidates)} candidatos con umbral simple. Probando umbral adaptativo...")
        thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
        contours_adapt, _ = cv2.findContours(thresh_adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = filter_contours(contours_adapt)
        
    if len(candidates) < 4:
        print(f"  [detect_markers] Error: Solo se detectaron {len(candidates)} candidatos de marcadores.")
        return None
        
    # --- BÚSQUEDA EXHAUSTIVA DE LA MEJOR COMBINACIÓN DE 4 CANDIDATOS ---
    # Evaluamos todas las combinaciones de 4 candidatos para encontrar la que forma el mejor cuadrilátero rectangular
    best_comb = None
    best_score = -1.0
    
    valid_combinations = []
    for comb in itertools.combinations(candidates, 4):
        pts_comb = [(c[0], c[1]) for c in comb]
        if is_valid_quad(pts_comb):
            # Formar array de puntos
            pts_arr = np.array(pts_comb, dtype=np.float32)
            cx_c = np.mean(pts_arr[:, 0])
            cy_c = np.mean(pts_arr[:, 1])
            angles_c = np.arctan2(pts_arr[:, 1] - cy_c, pts_arr[:, 0] - cx_c)
            sort_c = np.argsort(angles_c)
            sorted_comb_pts = pts_arr[sort_c]
            
            # Calcular área del cuadrilátero formado en resolución reducida (para escala de puntuación)
            resized_pts = np.array([[c[2], c[3]] for c in comb], dtype=np.float32)
            resized_cx = np.mean(resized_pts[:, 0])
            resized_cy = np.mean(resized_pts[:, 1])
            resized_angles = np.arctan2(resized_pts[:, 1] - resized_cy, resized_pts[:, 0] - resized_cx)
            resized_sorted = resized_pts[np.argsort(resized_angles)]
            
            area_quad = cv2.contourArea(resized_sorted.astype(np.int32))
            
            # Calcular desviación de los ángulos internos respecto a 90 grados
            angle_dev = 0.0
            for i in range(4):
                p1 = sorted_comb_pts[i - 1]
                p2 = sorted_comb_pts[i]
                p3 = sorted_comb_pts[(i + 1) % 4]
                v1 = p1 - p2
                v2 = p3 - p2
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 > 0 and n2 > 0:
                    cos_t = np.dot(v1, v2) / (n1 * n2)
                    angle = np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))
                    angle_dev += np.abs(angle - 90.0)
            
            # Calcular desviación de ratios de lados opuestos
            len0 = np.linalg.norm(sorted_comb_pts[1] - sorted_comb_pts[0])
            len1 = np.linalg.norm(sorted_comb_pts[2] - sorted_comb_pts[1])
            len2 = np.linalg.norm(sorted_comb_pts[3] - sorted_comb_pts[2])
            len3 = np.linalg.norm(sorted_comb_pts[0] - sorted_comb_pts[3])
            
            ratio_top_bottom = len0 / len2 if len2 > 0 else 0.0
            ratio_left_right = len3 / len1 if len1 > 0 else 0.0
            side_ratio_dev = np.abs(ratio_top_bottom - 1.0) + np.abs(ratio_left_right - 1.0)
            
            # Promedio del brillo del vecindario de fondo blanco
            avg_neigh = np.mean([c[5] for c in comb])
            
            # Puntuación combinada:
            # 1. Favorece áreas grandes (los 4 extremos cubren la hoja, descartando combinaciones locales)
            # 2. Penaliza desviaciones de perpendicularidad (ángulos distintos a 90)
            # 3. Penaliza asimetrías de lados opuestos (trapecios extraños)
            # 4. Favorece vecindarios blancos uniformes
            score = area_quad * (1.0 / (1.0 + angle_dev * 0.01)) * (1.0 / (1.0 + side_ratio_dev)) * (avg_neigh / 255.0)
            valid_combinations.append((comb, score, area_quad))
            
    if len(valid_combinations) > 0:
        # Ordenar por puntuación descendente
        valid_combinations.sort(key=lambda x: x[1], reverse=True)
        best_comb, best_score, best_area = valid_combinations[0]
        selected = list(best_comb)
        print(f"  [detect_markers] Combinación geométrica óptima seleccionada (Score: {best_score:.2f}, Área: {best_area:.1f} px^2).")
    else:
        # Fallback si no hay ninguna combinación geométricamente perfecta: usar selección inicial por Convex Hull
        print("  [detect_markers] ADVERTENCIA: No se encontró combinación geométricamente perfecta. Aplicando Convex Hull Fallback.")
        points_c = np.array([[c[2], c[3]] for c in candidates], dtype=np.float32)
        hull_c = cv2.convexHull(points_c)
        epsilon = 0.02 * cv2.arcLength(hull_c, True)
        approx = cv2.approxPolyDP(hull_c, epsilon, True)
        
        selected = []
        if len(approx) == 4:
            for p in approx:
                px, py = p[0]
                closest = min(candidates, key=lambda c: (c[2] - px)**2 + (c[3] - py)**2)
                selected.append(closest)
        else:
            tl = min(candidates, key=lambda c: c[0]**2 + c[1]**2)
            tr = min(candidates, key=lambda c: (c[0] - orig_w)**2 + c[1]**2)
            br = min(candidates, key=lambda c: (c[0] - orig_w)**2 + (c[1] - orig_h)**2)
            bl = min(candidates, key=lambda c: c[0]**2 + (c[1] - orig_h)**2)
            
            unique_sel = []
            for cand in [tl, tr, br, bl]:
                if cand not in unique_sel:
                    unique_sel.append(cand)
            while len(unique_sel) < 4:
                rem = [c for c in candidates if c not in unique_sel]
                if rem:
                    unique_sel.append(max(rem, key=lambda c: c[4]))
                else:
                    break
            selected = unique_sel[:4]

    # Ordenación estrictamente horaria TL, TR, BR, BL usando polar angles respecto al centro
    pts_s = np.array([[s[0], s[1]] for s in selected], dtype=np.float32)
    cx_s = np.mean(pts_s[:, 0])
    cy_s = np.mean(pts_s[:, 1])
    
    angles = np.arctan2(pts_s[:, 1] - cy_s, pts_s[:, 0] - cx_s)
    sort_idx = np.argsort(angles)
    selected = [selected[idx] for idx in sort_idx]
    
    # Extraer coordenadas finales en escala original
    final_pts = [(s[0], s[1]) for s in selected]
    
    # Guardar imagen de depuración si se especifica
    if debug_dir:
        debug_img = resized.copy()
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # TL=R, TR=G, BR=B, BL=Y
        labels = ["TL", "TR", "BR", "BL"]
        for idx, s in enumerate(selected):
            cx, cy = s[2], s[3]
            
            # Para hacer que la marca sea discreta y no tape la L del WCS ni otros elementos,
            # dibujamos su tamaño real (sin agrandarlo) con un borde fino de color (grosor 3)
            # y un relleno semi-transparente muy sutil (25% opacidad) que sigue exactamente
            # la inclinación física y escala del marcador de esquina detectado.
            rect = cv2.minAreaRect(s[6])
            center, size, angle = rect
            
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # Dibujar relleno semi-transparente
            overlay = debug_img.copy()
            cv2.drawContours(overlay, [box], 0, colors[idx], -1)
            cv2.addWeighted(overlay, 0.25, debug_img, 0.75, 0, debug_img)
            
            # Dibujar contorno exterior nítido
            cv2.drawContours(debug_img, [box], 0, colors[idx], 3)
            
            # Posicionamiento inteligente del texto de etiqueta para evitar solapar el WCS
            # Colocamos las etiquetas en dirección exterior opuesta al centro de la hoja
            dx = 30 if cx > (cx_s * scale) else -55
            dy = -25 if cy < (cy_s * scale) else 45
            
            cv2.putText(debug_img, labels[idx], (cx + dx, cy + dy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[idx], 3)
            
        # Dibujar centroide de referencia del cuadrilátero
        cv2.circle(debug_img, (int(cx_s * scale), int(cy_s * scale)), 10, (255, 255, 255), -1)
        
        os.makedirs(debug_dir, exist_ok=True)
        out_path = os.path.join(debug_dir, f"{img_name}_markers.png")
        cv2.imwrite(out_path, debug_img)
        print(f"  [detect_markers] Imagen de depuración de marcadores guardada en: {out_path}")
        
    return final_pts

