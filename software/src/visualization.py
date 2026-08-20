import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # Forzar backend no interactivo para entornos headless
import matplotlib.pyplot as plt

def generate_scientific_figures(img_raw, markers, rectified_img, wcs_info, 
                                 mask, contours, report, output_dir, img_name="img",
                                 y_direction="down", offset_mm=0.0):
    """
    Genera y guarda figuras técnicas individuales de alta resolución y un panel científico
    integrado de 3x3 para ilustrar el pipeline metodológico en el TFM.
    
    Parámetros:
        img_raw (np.ndarray): Imagen original de entrada.
        markers (list): Coordenadas de los marcadores en la original.
        rectified_img (np.ndarray): Imagen rectificada A4 (2100x2970 px).
        wcs_info (dict): Información del WCS.
        mask (np.ndarray): Máscara binaria de segmentación.
        contours (list): Contornos aceptados.
        report (dict): Detalles cuantitativos de la segmentación.
        output_dir (str): Directorio de salida.
        img_name (str): Nombre de la imagen base.
        y_direction (str): Dirección del eje Y ('down' o 'up').
        offset_mm (float): Offset aplicado.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # ==========================================
        # 1. FIGURAS INDIVIDUALES DE ALTA CALIDAD
        # ==========================================
        
        # Figura 1: Imagen Original Redimensionada para Thesis
        h_orig, w_orig = img_raw.shape[:2]
        resized_raw = cv2.resize(img_raw, (800, int(800 * h_orig / w_orig)))
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_01_raw.png"), resized_raw)
        
        # Figura 2: Marcadores Detectados (Overlay Rotado y Discreto de alta estética)
        markers_img = resized_raw.copy()
        scale_raw = 800 / w_orig
        if markers and len(markers) == 4:
            # Calcular inclinación física real y escala dinámica del papel
            dx = markers[1][0] - markers[0][0]
            dy = markers[1][1] - markers[0][1]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Estimación del scale dinámico: la distancia de centroides TL-TR es 180mm en una A4 estándar
            width_orig = np.linalg.norm(np.array(markers[1]) - np.array(markers[0]))
            S_px_mm = width_orig / 180.0
            marker_size_scaled = 15.0 * S_px_mm * scale_raw # 15 mm físico real
            
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # BGR: TL=R, TR=G, BR=B, BL=Y
            labels = ["M1 (TL)", "M2 (TR)", "M3 (BR)", "M4 (BL)"]
            
            # Centroide de referencia para el posicionamiento inteligente de etiquetas
            pts_s = np.array([[pt[0] * scale_raw, pt[1] * scale_raw] for pt in markers], dtype=np.float32)
            cx_s = np.mean(pts_s[:, 0])
            cy_s = np.mean(pts_s[:, 1])
            
            for idx, pt in enumerate(markers):
                cx = int(pt[0] * scale_raw)
                cy = int(pt[1] * scale_raw)
                
                # Crear caja rotada del marcador con tamaño físico real escalado
                rect = ((cx, cy), (marker_size_scaled, marker_size_scaled), angle)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                
                # Dibujar relleno semi-transparente (25% opacidad)
                overlay = markers_img.copy()
                cv2.drawContours(overlay, [box], 0, colors[idx], -1)
                cv2.addWeighted(overlay, 0.25, markers_img, 0.75, 0, markers_img)
                
                # Dibujar contorno exterior nítido (grosor 2 para escala reducida)
                cv2.drawContours(markers_img, [box], 0, colors[idx], 2)
                
                # Posicionamiento inteligente exterior para evitar solapamientos del WCS
                dx_lbl = 20 if cx > cx_s else -50
                dy_lbl = -20 if cy < cy_s else 35
                cv2.putText(markers_img, labels[idx], (cx + dx_lbl, cy + dy_lbl), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[idx], 2)
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_02_markers.png"), markers_img)
        
        # Figura 3: Hoja Rectificada (Versión pequeña para fácil importación)
        h_rect, w_rect = rectified_img.shape[:2]
        rect_w = 600
        rect_h = int(600 * h_rect / w_rect)
        
        rect_small = cv2.resize(rectified_img, (rect_w, rect_h))
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_03_rectified.png"), rect_small)
        
        # Figura 4: Máscara de Segmentación
        mask_small = cv2.resize(mask, (rect_w, rect_h))
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_05_mask.png"), mask_small)
        
        # Figura 5: Contornos Detectados y Enumerados
        contours_img = rectified_img.copy()
        for idx, cnt in enumerate(contours):
            cv2.drawContours(contours_img, [cnt], -1, (0, 255, 0), 4)
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
            cv2.circle(contours_img, (cx, cy), 20, (0, 255, 0), -1)
            cv2.putText(contours_img, str(idx + 1), (cx - 8, cy + 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        contours_small = cv2.resize(contours_img, (rect_w, rect_h))
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_06_contours.png"), contours_small)
        
        # Figura 6: Detalle del WCS (ROI de la marca L)
        roi_img = np.zeros((300, 300, 3), dtype=np.uint8)
        if wcs_info["status"] == "SUCCESS":
            ox, oy = wcs_info["origin"]
            # Tomamos una ROI de 500x500px centrada en el origen y escalamos
            x_start = max(0, int(ox - 250))
            x_end = min(rectified_img.shape[1], int(ox + 250))
            y_start = max(0, int(oy - 250))
            y_end = min(rectified_img.shape[0], int(oy + 250))
            
            roi_raw = rectified_img[y_start:y_end, x_start:x_end].copy()
            # Dibujar ejes en el fragmento
            ox_roi = int(ox - x_start)
            oy_roi = int(oy - y_start)
            uX, uY = wcs_info["uX"], wcs_info["uY"]
            
            cv2.circle(roi_raw, (ox_roi, oy_roi), 8, (0, 0, 255), -1)
            cv2.arrowedLine(roi_raw, (ox_roi, oy_roi), 
                            (int(ox_roi + uX[0]*120), int(oy_roi + uX[1]*120)), (255, 0, 0), 3)
            cv2.arrowedLine(roi_raw, (ox_roi, oy_roi), 
                            (int(ox_roi + uY[0]*120), int(oy_roi + uY[1]*120)), (0, 255, 0), 3)
            roi_img = cv2.resize(roi_raw, (300, 300))
        else:
            cv2.putText(roi_img, "WCS NOT FOUND", (30, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_04_wcs_roi.png"), roi_img)

        # ==========================================
        # 2. PANEL CIENTÍFICO MULTI-FOTO 3X3 (TFM)
        # ==========================================
        fig, axes = plt.subplots(3, 3, figsize=(15, 18), dpi=150)
        fig.suptitle(f"Pipeline de Procesamiento y Registro WCS - Caso: {img_name}\n(Trabajo Fin de Máster - Innokey)", 
                     fontsize=18, fontweight='bold', color='#1E293B', y=0.96)
        
        # Panel 1: Original
        axes[0, 0].imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("1. Captura Fotográfica Original", fontsize=11, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Panel 2: Marcadores Detectados
        axes[0, 1].imshow(cv2.cvtColor(markers_img, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("2. Localización de Marcadores", fontsize=11, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Panel 3: Hoja Rectificada
        axes[0, 2].imshow(cv2.cvtColor(rect_small, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("3. Rectificación A4 (Homografía)", fontsize=11, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Panel 4: ROI de la Marca L
        axes[1, 0].imshow(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("4. Región de la Marca Láser L", fontsize=11, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Panel 5: Origen WCS y Ejes
        wcs_overlay = rectified_img.copy()
        if wcs_info["status"] == "SUCCESS":
            ox, oy = int(wcs_info["origin"][0]), int(wcs_info["origin"][1])
            uX, uY = wcs_info["uX"], wcs_info["uY"]
            cv2.circle(wcs_overlay, (ox, oy), 25, (0, 0, 255), -1)
            cv2.arrowedLine(wcs_overlay, (ox, oy), (int(ox + uX[0]*250), int(oy + uX[1]*250)), (255, 0, 0), 10, tipLength=0.2)
            cv2.arrowedLine(wcs_overlay, (ox, oy), (int(ox + uY[0]*250), int(oy + uY[1]*250)), (0, 255, 0), 10, tipLength=0.2)
        wcs_overlay_small = cv2.resize(wcs_overlay, (rect_w, rect_h))
        axes[1, 1].imshow(cv2.cvtColor(wcs_overlay_small, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("5. Sistema WCS Estimado (Ejes)", fontsize=11, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Panel 6: Máscara Binaria
        axes[1, 2].imshow(mask_small, cmap='gray')
        axes[1, 2].set_title("6. Máscara de Segmentación", fontsize=11, fontweight='bold')
        axes[1, 2].axis('off')
        
        # Panel 7: Contornos
        axes[2, 0].imshow(cv2.cvtColor(contours_small, cv2.COLOR_BGR2RGB))
        axes[2, 0].set_title("7. Contornos de Toppers", fontsize=11, fontweight='bold')
        axes[2, 0].axis('off')
        
        # Panel 8: Vista Vectorial de Simulación en mm WCS
        axes[2, 1].set_facecolor('#FAFAFA')
        axes[2, 1].set_title("8. Polilíneas Vectoriales (mm WCS)", fontsize=11, fontweight='bold')
        if wcs_info["status"] == "SUCCESS":
            # Dibujar los contornos en coordenadas de mm relativas al WCS
            origin = wcs_info["origin"]
            uX_a = np.array(wcs_info["uX"])
            uY_a = np.array(wcs_info["uY"])
            S_Y = 1.0 if y_direction == "down" else -1.0
            
            for cnt in contours:
                pts_mm = []
                for pt in cnt:
                    px, py = pt[0]
                    v = np.array([px - origin[0], py - origin[1]])
                    x_mm = np.dot(v, uX_a) / 10.0
                    y_mm = (np.dot(v, uY_a) / 10.0) * S_Y
                    pts_mm.append((x_mm, y_mm))
                pts_mm = np.array(pts_mm)
                # Cerrar curva
                pts_mm = np.vstack([pts_mm, pts_mm[0]])
                axes[2, 1].plot(pts_mm[:, 0], pts_mm[:, 1], 'c-', linewidth=1.5)
                
            # Ejes WCS (20mm)
            axes[2, 1].arrow(0, 0, 20, 0, head_width=1.5, head_length=2, fc='r', ec='r', label='Eje X')
            axes[2, 1].arrow(0, 0, 0, 20 * S_Y, head_width=1.5, head_length=2, fc='g', ec='g', label='Eje Y')
            axes[2, 1].plot(0, 0, 'ro', markersize=6, label='Origen WCS')
            axes[2, 1].grid(True, linestyle='--', alpha=0.5)
            axes[2, 1].set_xlabel("X (mm)", fontsize=8)
            axes[2, 1].set_ylabel("Y (mm)", fontsize=8)
            axes[2, 1].set_aspect('equal', adjustable='box')
        else:
            axes[2, 1].text(0.5, 0.5, "Requiere WCS\npara Simulación Vectorial", 
                            ha='center', va='center', transform=axes[2, 1].transAxes, 
                            fontsize=10, color='red', fontweight='bold')
            axes[2, 1].axis('off')
            
        # Panel 9: Cuadro de Métricas / Observaciones del TFM
        axes[2, 2].set_facecolor('#F1F5F9')
        axes[2, 2].axis('off')
        axes[2, 2].set_title("9. Resumen Cuantitativo TFM", fontsize=11, fontweight='bold')
        
        status_l = "DETECTADA (AUTOMÁTICO)" if wcs_info["status"] == "SUCCESS" else "WCS NO DETECTADO"
        metrics_text = (
            f"  • Imagen: {img_name}.jpg\n"
            f"  • Marcadores de Hoja: {len(markers) if markers else 0} / 4\n"
            f"  • Estado Marca L: {status_l}\n"
            f"  • Origen WCS: " + (f"({wcs_info['origin'][0]:.1f}, {wcs_info['origin'][1]:.1f}) px\n" if wcs_info["origin"] else "N/A\n") +
            f"  • Ángulo de Rotación MDF: " + (f"{np.degrees(np.arctan2(wcs_info['uX'][1], wcs_info['uX'][0])):.2f}°\n" if wcs_info["status"] == "SUCCESS" else "N/A\n") +
            f"  • Toppers Aceptados: {report['toppers_detected']} / 8\n"
            f"  • Descartes de Ruido: {report['discarded_components_count']}\n"
            f"  • Offset Aplicado: {offset_mm} mm\n"
            f"  • Dirección del Eje Y: WCS_{y_direction.upper()}\n"
            f"  • Resolución Lienzo: {w_rect} x {h_rect} px\n"
            f"  • Escala de Trabajo: {rectified_img.shape[1]/w_rect*10.0 if w_rect>0 else 10.0:.1f} px / mm\n"
        )
        axes[2, 2].text(0.05, 0.9, "Métricas del Experimento:", fontsize=10, fontweight='bold', color='#0F172A')
        axes[2, 2].text(0.05, 0.25, metrics_text, fontsize=9.5, color='#334155', family='monospace', va='top')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        panel_path = os.path.join(output_dir, f"{img_name}_panel_tfm.png")
        plt.savefig(panel_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"  [visualization] Panel científico 3x3 guardado exitosamente en: {panel_path}")
        return True
    except Exception as e:
        print(f"  [visualization] Error generando figuras científicas: {e}")
        return False
