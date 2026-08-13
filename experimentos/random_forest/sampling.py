import os
import cv2
import numpy as np
import random
from feature_extraction import extract_pixel_features

def sample_sheet_pixels(img_path, mask_path, include_coords=False, num_topper_pixels=5000, num_bg_pixels=5000, seed=42):
    """
    Realiza un muestreo balanceado de píxeles (topper/fondo) en una hoja sintética.
    El fondo se divide 50% lejano y 50% cercano (negativos difíciles).
    
    Retorna:
    sampled_features: np.array de (N, num_features)
    sampled_labels: np.array de (N,) con valores 1 (topper) y 0 (fondo)
    stats: dict con recuentos reales de píxeles muestreados
    """
    # Cargar imágenes
    img_bgr = cv2.imread(img_path)
    mask_bin = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    H, W = mask_bin.shape
    
    # 1. Identificar regiones de píxeles
    topper_coords = np.argwhere(mask_bin == 255)
    
    # Dilatación para definir fondo cercano y lejano
    # Usamos una ventana de 35x35 para considerar fondo en la vecindad inmediata
    kernel = np.ones((35, 35), np.uint8)
    mask_dilated = cv2.dilate(mask_bin, kernel, iterations=1)
    
    far_bg_coords = np.argwhere(mask_dilated == 0)
    near_bg_coords = np.argwhere((mask_dilated == 255) & (mask_bin == 0))
    
    # Configurar semilla local para consistencia
    rng = np.random.default_rng(seed)
    
    # 2. Muestrear Topper
    n_topper = min(len(topper_coords), num_topper_pixels)
    if n_topper > 0:
        idx_topper = rng.choice(len(topper_coords), n_topper, replace=False)
        topper_sampled = topper_coords[idx_topper]
    else:
        topper_sampled = np.empty((0, 2), dtype=np.int64)
        
    # 3. Muestrear Fondo
    n_far = num_bg_pixels // 2
    n_near = num_bg_pixels - n_far
    
    n_far_actual = min(len(far_bg_coords), n_far)
    n_near_actual = min(len(near_bg_coords), n_near)
    
    if n_far_actual > 0:
        idx_far = rng.choice(len(far_bg_coords), n_far_actual, replace=False)
        far_sampled = far_bg_coords[idx_far]
    else:
        far_sampled = np.empty((0, 2), dtype=np.int64)
        
    if n_near_actual > 0:
        idx_near = rng.choice(len(near_bg_coords), n_near_actual, replace=False)
        near_sampled = near_bg_coords[idx_near]
    else:
        near_sampled = np.empty((0, 2), dtype=np.int64)
        
    # Combinar todas las coordenadas de píxeles muestreados
    all_coords = np.vstack([topper_sampled, far_sampled, near_sampled])
    
    # Crear etiquetas: 1 para Topper, 0 para Fondo
    labels = np.concatenate([
        np.ones(len(topper_sampled), dtype=np.uint8),
        np.zeros(len(far_sampled) + len(near_sampled), dtype=np.uint8)
    ])
    
    # 4. Extraer características locales de la hoja completa
    features, feature_names = extract_pixel_features(img_bgr, include_coords=include_coords)
    
    # Obtener el índice 1D correspondiente para cada píxel (y * W + x)
    flat_indices = all_coords[:, 0] * W + all_coords[:, 1]
    
    sampled_features = features[flat_indices]
    
    stats = {
        "topper_pixels": len(topper_sampled),
        "far_bg_pixels": len(far_sampled),
        "near_bg_pixels": len(near_sampled),
        "total_pixels": len(all_coords)
    }
    
    return sampled_features, labels, stats, feature_names
