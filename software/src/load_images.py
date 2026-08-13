import cv2
import numpy as np
from PIL import Image, ImageOps
import os

def load_image(img_path, manual_rotate=0):
    """
    Carga una imagen corrigiendo su orientación EXIF y aplicando rotación manual si es necesario.
    
    Parámetros:
        img_path (str): Ruta del archivo de imagen.
        manual_rotate (int): Grados de rotación horaria (0, 90, 180, 270).
        
    Retorna:
        np.ndarray: Imagen cargada en formato BGR de OpenCV, o None si ocurre un error.
    """
    if not os.path.exists(img_path):
        print(f"Error: El archivo no existe en {img_path}")
        return None
        
    try:
        # Abrimos con PIL para leer y procesar la orientación EXIF de forma robusta
        pil_img = Image.open(img_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        
        # Convertimos de RGB (PIL) a BGR (OpenCV)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Aplicamos rotación manual si es requerida
        if manual_rotate == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            print("  [load_images] Rotación manual aplicada: 90 grados horaria.")
        elif manual_rotate == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
            print("  [load_images] Rotación manual aplicada: 180 grados.")
        elif manual_rotate == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            print("  [load_images] Rotación manual aplicada: 270 grados horaria (90 antihoraria).")
            
        return img
    except Exception as e:
        print(f"Error cargando la imagen {img_path}: {e}")
        return None
