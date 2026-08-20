import cv2
import numpy as np

def extract_pixel_features(img_bgr, include_coords=False):
    """
    Extrae un tensor de características locales por píxel a partir de una imagen BGR.
    
    img_bgr: np.array de dimensiones (H, W, 3)
    include_coords: si es True, incluye las coordenadas (x, y) normalizadas como características (para ablación).
    
    Retorna:
    features: np.array de dimensiones (H * W, num_features)
    feature_names: lista con los nombres de las características correspondientes.
    """
    H, W, C = img_bgr.shape
    
    # 1. RGB normalizado
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r_ch, g_ch, b_ch = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    
    # 2. HSV normalizado
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch = img_hsv[:,:,0] / 180.0
    s_ch = img_hsv[:,:,1] / 255.0
    v_ch = img_hsv[:,:,2] / 255.0
    
    # 3. Lab normalizado
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L_ch = img_lab[:,:,0] / 255.0
    a_ch = img_lab[:,:,1] / 255.0
    b_lab_ch = img_lab[:,:,2] / 255.0
    
    # 4. Escala de grises
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # 5. Gradientes (Sobel)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalizar gradientes
    sobel_x_norm = (sobel_x - sobel_x.min()) / (sobel_x.max() - sobel_x.min() + 1e-6)
    sobel_y_norm = (sobel_y - sobel_y.min()) / (sobel_y.max() - sobel_y.min() + 1e-6)
    grad_magnitude_norm = grad_magnitude / (grad_magnitude.max() + 1e-6)
    
    # 6. Suavizado Gaussiano y DoG
    blur_1 = cv2.GaussianBlur(gray, (5, 5), 2.0)
    blur_2 = cv2.GaussianBlur(gray, (11, 11), 5.0)
    dog = blur_1 - blur_2
    
    # 7. Media y desviación estándar local en ventanas
    # Ventana 7x7
    mean_7 = cv2.blur(gray, (7, 7))
    mean_sq_7 = cv2.blur(gray**2, (7, 7))
    var_7 = np.maximum(0.0, mean_sq_7 - mean_7**2)
    std_7 = np.sqrt(var_7)
    
    # Ventana 15x15
    mean_15 = cv2.blur(gray, (15, 15))
    mean_sq_15 = cv2.blur(gray**2, (15, 15))
    var_15 = np.maximum(0.0, mean_sq_15 - mean_15**2)
    std_15 = np.sqrt(var_15)
    
    # Lista de canales de características básicas
    feature_maps = [
        r_ch, g_ch, b_ch,             # RGB
        h_ch, s_ch, v_ch,             # HSV
        L_ch, a_ch, b_lab_ch,         # Lab
        sobel_x_norm, sobel_y_norm, grad_magnitude_norm, # Gradientes
        blur_1, blur_2, dog,          # Suavizado y DoG
        mean_7, std_7,                # Textura 7x7
        mean_15, std_15               # Textura 15x15
    ]
    
    feature_names = [
        "red", "green", "blue",
        "hue", "saturation", "value",
        "L_lightness", "a_redgreen", "b_blueyellow",
        "sobel_x", "sobel_y", "grad_magnitude",
        "gaussian_blur_s2", "gaussian_blur_s5", "difference_of_gaussians",
        "local_mean_7", "local_std_7",
        "local_mean_15", "local_std_15"
    ]
    
    # 8. Coordenadas normalizadas (Opcional - Ablación)
    if include_coords:
        y_indices, x_indices = np.indices((H, W), dtype=np.float32)
        y_norm = y_indices / float(H - 1)
        x_norm = x_indices / float(W - 1)
        
        feature_maps.extend([x_norm, y_norm])
        feature_names.extend(["coord_x", "coord_y"])
        
    # Stack y aplanar
    stacked = np.stack(feature_maps, axis=-1)  # (H, W, num_features)
    flat_features = stacked.reshape(-1, len(feature_names))
    
    return flat_features, feature_names
