import cv2
import numpy as np
import os

def extract_contours(mask, rectified_img=None, debug_dir=None, img_name="img"):
    """
    Extrae, filtra y une espacialmente los contornos exteriores de los toppers.
    
    Parámetros:
        mask (np.ndarray): Máscara binaria segmentada (2100x2970 px).
        rectified_img (np.ndarray): Imagen rectificada original para depuración.
        debug_dir (str): Directorio de depuración.
        img_name (str): Nombre de la imagen.
        
    Retorna:
        tuple: (accepted_contours, report)
               accepted_contours: Lista de arrays de contornos aceptados por el sistema.
               report: Diccionario con la bitácora cuantitativa de toppers y descartes.
    """
    # 1. Extracción de contornos externos
    raw_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 2. Algoritmo de Unión por Proximidad Espacial basado en Distancia Euclidiana Exacta de Contornos
    # Si algunos toppers están fragmentados (ej. estrellas flotantes, acentos de letras),
    # los agrupamos si la distancia mínima exacta entre sus píxeles de contorno es menor a 5.0 px (0.5 mm).
    num_raw = len(raw_contours)
    grouped_indices = list(range(num_raw))
    
    def exact_contour_distance(cnt1, cnt2):
        pts1 = cnt1.reshape(-1, 2)
        pts2 = cnt2.reshape(-1, 2)
        # Submuestrear contornos gigantes para optimizar el rendimiento y garantizar cálculo instantáneo (< 5ms)
        if len(pts1) > 200:
            pts1 = pts1[::len(pts1)//100 or 1]
        if len(pts2) > 200:
            pts2 = pts2[::len(pts2)//100 or 1]
        diff = pts1[:, None, :] - pts2[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        return np.min(dists)
    
    # Agrupación por transitividad de cercanía (DSU / Componentes Conectadas usando distancia de contorno exacta)
    for i in range(num_raw):
        for j in range(i + 1, num_raw):
            dist = exact_contour_distance(raw_contours[i], raw_contours[j])
            if dist < 5.0: # Umbral estricto de 0.5 mm (5 px) para evitar unir toppers independientes adyacentes
                # Unimos los grupos de i y j
                root_i = grouped_indices[i]
                root_j = grouped_indices[j]
                if root_i != root_j:
                    for k in range(num_raw):
                        if grouped_indices[k] == root_j:
                            grouped_indices[k] = root_i
                            
    # Re-generar contornos fusionados basados en los grupos
    unique_groups = set(grouped_indices)
    merged_contours = []
    
    for group in unique_groups:
        group_cnts = [raw_contours[idx] for idx in range(num_raw) if grouped_indices[idx] == group]
        if len(group_cnts) == 1:
            merged_contours.append(group_cnts[0])
        else:
            # Creamos una máscara local temporal, dibujamos el grupo, lo dilatamos y extraemos contorno
            temp_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(temp_mask, group_cnts, -1, 255, -1)
            # Dilatación y cierre ligero para sellar puentes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, kernel)
            cnts_temp, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts_temp:
                # Tomamos el contorno de mayor área resultante
                merged_contours.append(max(cnts_temp, key=cv2.contourArea))
                
    # 3. Filtrado Flexible e Informe de Descartes
    accepted_contours = []
    discard_log = []
    
    for idx, cnt in enumerate(merged_contours):
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        # Calcular centroide
        M = cv2.moments(cnt)
        cx, cy = 0, 0
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
        # Filtro 1: Área demasiado pequeña (ruido perimetral, motas o boquilla láser)
        if area < 20000:
            discard_log.append({
                "index": idx,
                "area": area,
                "centroid": (cx, cy),
                "cause": f"Area too small ({area:.0f} px^2 < 20000 px^2)"
            })
            continue
            
        # Filtro 2: Área excesivamente grande (fusión anómala de fondo)
        if area > 600000:
            discard_log.append({
                "index": idx,
                "area": area,
                "centroid": (cx, cy),
                "cause": f"Area too large ({area:.0f} px^2 > 600000 px^2)"
            })
            continue
            
        # Filtro 3: Aspecto geométrico altamente anómalo (línea o mancha perimetral)
        if aspect_ratio < 0.15 or aspect_ratio > 6.0:
            discard_log.append({
                "index": idx,
                "area": area,
                "centroid": (cx, cy),
                "cause": f"Anomalous aspect ratio ({aspect_ratio:.2f})"
            })
            continue
            
        # Si aprueba los filtros, se añade a la lista de toppers aceptados
        accepted_contours.append(cnt)
        
    # Ordenar toppers aceptados de arriba a abajo, izquierda a derecha por posición de centroide
    accepted_contours = sorted(accepted_contours, key=lambda c: (cv2.moments(c)["m01"]/cv2.moments(c)["m00"] if cv2.moments(c)["m00"] > 0 else 0))
    
    # 4. Generación de la Bitácora de Reporte
    num_accepted = len(accepted_contours)
    report = {
        "toppers_expected": 8,
        "toppers_detected": num_accepted,
        "raw_components_extracted": num_raw,
        "components_merged": num_raw - len(merged_contours),
        "discarded_components_count": len(discard_log),
        "discards": discard_log,
        "accepted_details": []
    }
    
    for i, cnt in enumerate(accepted_contours):
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
        report["accepted_details"].append({
            "topper_id": i + 1,
            "area_px": area,
            "centroid": (cx, cy),
            "bbox": (x, y, w, h)
        })
        
    print(f"  [extract_contours] Contornos listos. Esperados: 8 | Detectados: {num_accepted} | Descartes: {len(discard_log)}")
    
    # 5. Generación de visualización de depuración
    if debug_dir and rectified_img is not None:
        debug_img = rectified_img.copy()
        
        # Dibujar descartados en rojo con una cruz
        for d in discard_log:
            cx, cy = d["centroid"]
            cv2.putText(debug_img, "X", (cx - 15, cy + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            # Dibujar contorno de descarte
            # Buscamos el contorno original
            cnt_disc = merged_contours[d["index"]]
            cv2.drawContours(debug_img, [cnt_disc], -1, (0, 0, 255), 2)
            
        # Dibujar aceptados en verde con numeración
        for i, cnt in enumerate(accepted_contours):
            cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 4)
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
            # Dibujar píldora verde de fondo para el texto
            cv2.circle(debug_img, (cx, cy), 25, (0, 255, 0), -1)
            cv2.putText(debug_img, str(i + 1), (cx - 10, cy + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
            
        os.makedirs(debug_dir, exist_ok=True)
        out_path = os.path.join(debug_dir, f"{img_name}_contours.png")
        cv2.imwrite(out_path, debug_img)
        print(f"  [extract_contours] Imagen de contornos guardada en: {out_path}")
        
    return accepted_contours, report
