import cv2
import numpy as np

def postprocess_rf_mask(raw_mask, min_component_area=300):
    """
    Aplica postprocesamiento morfológico leve a la máscara binarizada del Random Forest
    para limpiar ruido de fondo sin degradar detalles finos.
    
    raw_mask: np.array de tipo uint8, (H, W), con valores 0 y 255
    min_component_area: área mínima para descartar ruido aislado de fondo
    
    Retorna:
    processed_mask: máscara limpia de tipo uint8, (H, W)
    contours: lista de contornos extraídos del topper
    """
    # 1. Eliminar componentes de ruido muy pequeños
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(raw_mask)
    cleaned = raw_mask.copy()
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_component_area:
            cleaned[labels_im == label] = 0
            
    # 2. Cierre morfológico leve (kernel elíptico 3x3) para rellenar microporos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed_mask = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # 3. Extraer contornos externos de las formas resultantes
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return processed_mask, contours
