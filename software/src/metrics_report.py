import json
import csv
import os
import pandas as pd

def save_individual_report(image_name, metrics, output_dir):
    """
    Guarda las métricas cuantitativas de procesamiento de una imagen individual en un archivo JSON.
    
    Parámetros:
        image_name (str): Nombre de la imagen procesada.
        metrics (dict): Diccionario conteniendo los resultados cuantitativos.
        output_dir (str): Directorio de salida.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"report_{image_name.replace('.jpg', '')}.json")
    
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"  [metrics_report] Reporte JSON guardado en: {report_path}")
        return True
    except Exception as e:
        print(f"  [metrics_report] Error guardando reporte JSON: {e}")
        return False

def generate_lot_summary(accumulated_results, output_dir):
    """
    Consolida las métricas de un lote completo de imágenes y las exporta a un CSV resumen
    y genera una visualización de tabla en consola.
    
    Parámetros:
        accumulated_results (list): Lista de diccionarios de métricas individuales.
        output_dir (str): Directorio de salida.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "lot_summary_report.csv")
    
    # Preparar datos para la tabla formal de resultados requerida por el TFM
    summary_data = []
    for r in accumulated_results:
        summary_data.append({
            "Imagen": r.get("image_name"),
            "Condición": r.get("condition", "N/A"),
            "Éxito Detección Hoja": "SÍ" if r.get("markers_detected") == 4 else "NO",
            "Éxito Detección L": "SÍ" if r.get("wcs_status") == "SUCCESS" else "NO",
            "Toppers Detectados": r.get("toppers_detected", 0),
            "Descartes": r.get("discarded_count", 0),
            "Origen WCS (px)": str(r.get("wcs_origin")) if r.get("wcs_origin") else "N/A",
            "Tiempo Total (ms)": f"{r.get('time_total_ms', 0.0):.1f}",
            "Observaciones": r.get("observations", "")
        })
        
    try:
        # Escribir a CSV usando pandas para formateo profesional
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n  [metrics_report] Reporte consolidado de lote guardado en: {csv_path}")
        
        # Imprimir una hermosa tabla en consola para feedback instantáneo
        print("\n" + "="*95)
        print("                        TABLA DE RESULTADOS PRELIMINARES DEL PIPELINE")
        print("="*95)
        print(df.to_string(index=False))
        print("="*95 + "\n")
        return True
    except Exception as e:
        print(f"  [metrics_report] Error generando reporte consolidado: {e}")
        return False
