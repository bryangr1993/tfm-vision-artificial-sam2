import cv2
import numpy as np
import os
import ezdxf

# Intentar importar shapely de manera opcional para offset robusto
try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

def simplify_and_transform_contour(cnt, wcs_info, scale=10.0, y_direction="down"):
    """
    Simplifica el contorno y lo transforma a coordenadas físicas en milímetros relativas al WCS.
    
    Parámetros:
        cnt (np.ndarray): Contorno en coordenadas de píxeles de la hoja rectificada.
        wcs_info (dict): Información del WCS conteniendo origin, uX, uY y status.
        scale (float): Escala en px/mm (por defecto 10.0).
        y_direction (str): Avance del eje Y ('down' o 'up').
        
    Retorna:
        list: Lista de tuplas (x_mm, y_mm) del contorno transformado.
    """
    # 1. Simplificación mediante Douglas-Peucker (RDP)
    # Epsilon de 1.5 píxeles (~0.15 mm) remueve ruido del pixelado y preserva curvas suaves
    epsilon = 1.5
    simplified = cv2.approxPolyDP(cnt, epsilon, closed=True)
    
    # 2. Transformación de Coordenadas
    origin = wcs_info["origin"]
    uX = np.array(wcs_info["uX"])
    uY = np.array(wcs_info["uY"])
    
    S_Y = 1.0 if y_direction == "down" else -1.0
    
    transformed_points = []
    for pt in simplified:
        px, py = pt[0]
        # Vector v desde el origen del WCS al punto
        vx = px - origin[0]
        vy = py - origin[1]
        v = np.array([vx, vy])
        
        # Proyecciones sobre los vectores base reales (producto punto) en mm
        x_mm = np.dot(v, uX) / scale
        y_mm = (np.dot(v, uY) / scale) * S_Y
        
        transformed_points.append((float(x_mm), float(y_mm)))
        
    return transformed_points

def apply_offset_shapely(points, offset_mm):
    """
    Aplica una compensación de corte (offset) exterior robusta usando Shapely.
    
    Parámetros:
        points (list): Lista de tuplas (x_mm, y_mm).
        offset_mm (float): Distancia de offset en milímetros.
        
    Retorna:
        list: Puntos compensados exteriormente.
    """
    if not SHAPELY_AVAILABLE:
        print("  [export_dxf] Advertencia: Shapely no está instalado. Omitiendo offset.")
        return points
        
    if offset_mm <= 0:
        return points
        
    try:
        poly = Polygon(points)
        if not poly.is_valid:
            poly = make_valid(poly)
            
        # join_style=1 (ROUND) crea esquinas redondeadas ideales para corte CNC
        buffered = poly.buffer(offset_mm, join_style=1)
        
        # Extraer coordenadas exteriores del polígono resultante
        if buffered.geom_type == 'Polygon':
            coords = list(buffered.exterior.coords)
        elif buffered.geom_type == 'MultiPolygon':
            # Si se subdivide, tomamos el polígono más grande
            largest_poly = max(buffered.geoms, key=lambda p: p.area)
            coords = list(largest_poly.exterior.coords)
        else:
            coords = points
            
        # Remover el último punto duplicado devuelto por Shapely
        if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
            
        return [(float(c[0]), float(c[1])) for c in coords]
    except Exception as e:
        print(f"  [export_dxf] Error aplicando offset con Shapely: {e}. Usando contorno original.")
        return points

def export_to_dxf(contours, wcs_info, output_path, scale=10.0, offset_mm=0.0, y_direction="down"):
    """
    Simplifica, proyecta y exporta todos los contornos a un archivo DXF vectorial alineado con el WCS.
    
    Parámetros:
        contours (list): Lista de contornos en píxeles.
        wcs_info (dict): Información del WCS detectado.
        output_path (str): Ruta de destino para guardar el archivo DXF.
        scale (float): Escala px/mm.
        offset_mm (float): Offset exterior en mm.
        y_direction (str): Dirección de Y ('down' o 'up').
        
    Retorna:
        bool: True si la exportación fue exitosa, False en caso contrario.
    """
    if wcs_info["status"] != "SUCCESS":
        print("  [export_dxf] Error: WCS no está en estado SUCCESS. No se puede exportar DXF alineado.")
        return False
        
    try:
        # Crear documento ezdxf
        doc = ezdxf.new(dxfversion='R2010')
        doc.layers.new(name='TOPPERS_CUT', dxfattribs={'color': 4}) # Color Cyan
        msp = doc.modelspace()
        
        for idx, cnt in enumerate(contours):
            # 1. Simplificar y Proyectar a mm WCS
            pts_mm = simplify_and_transform_contour(cnt, wcs_info, scale, y_direction)
            
            # 2. Aplicar offset si corresponde
            if offset_mm > 0.0:
                pts_mm = apply_offset_shapely(pts_mm, offset_mm)
                
            # 3. Escribir polilínea cerrada en DXF
            msp.add_lwpolyline(pts_mm, close=True, dxfattribs={'layer': 'TOPPERS_CUT'})
            
        # Guardar archivo
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        doc.saveas(output_path)
        print(f"  [export_dxf] Archivo vectorial DXF guardado con éxito en: {output_path}")
        return True
    except Exception as e:
        print(f"  [export_dxf] Error exportando a DXF: {e}")
        return False

def generate_validation_dxf(output_path, y_direction="down"):
    """
    Genera un archivo DXF de calibración conteniendo los ejes WCS y un cuadrado de 10x10 mm.
    Ayuda al operario a validar físicamente la alineación y sentidos de los ejes en RDWorks.
    """
    try:
        doc = ezdxf.new(dxfversion='R2010')
        doc.layers.new(name='VALIDATION_AXES', dxfattribs={'color': 1}) # Color Rojo
        msp = doc.modelspace()
        
        S_Y = 1.0 if y_direction == "down" else -1.0
        
        # Eje X: Línea de (0,0) a (20, 0) mm
        msp.add_line((0, 0), (20, 0), dxfattribs={'layer': 'VALIDATION_AXES'})
        # Flecha Eje X (flecha de 2mm)
        msp.add_lwpolyline([(18, 0.5), (20, 0), (18, -0.5)], close=False, dxfattribs={'layer': 'VALIDATION_AXES'})
        
        # Eje Y: Línea de (0,0) a (0, 20 * S_Y) mm
        msp.add_line((0, 0), (0, 20 * S_Y), dxfattribs={'layer': 'VALIDATION_AXES'})
        # Flecha Eje Y (flecha de 2mm)
        msp.add_lwpolyline([(0.5, 18 * S_Y), (0, 20 * S_Y), (-0.5, 18 * S_Y)], close=False, dxfattribs={'layer': 'VALIDATION_AXES'})
        
        # Cuadrado de prueba de 10x10 mm para verificar escala y avance en cuadrante positivo
        # En RDWorks (X+ derecha, Y+ abajo), el cuadrado avanza en X+ (derecha) e Y+ (abajo)
        msp.add_lwpolyline([
            (2, 2 * S_Y),
            (12, 2 * S_Y),
            (12, 12 * S_Y),
            (2, 12 * S_Y)
        ], close=True, dxfattribs={'layer': 'VALIDATION_AXES'})
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        doc.saveas(output_path)
        print(f"  [export_dxf] Ejes de validación DXF guardados en: {output_path}")
        return True
    except Exception as e:
        print(f"  [export_dxf] Error generando DXF de validación: {e}")
        return False
