import os
import re
import sys
import json
import urllib.request
import urllib.error

# Lista de términos técnicos protegidos que no deben ser alterados ni traducidos por el humanizador
PROTECTED_TERMS = [
    "WCS", "DXF", "A4", "CAD", "RDWorks", "ezdxf", "Shapely", "OpenCV", "Tkinter", "Innokey",
    "Random Forest", "SAM 2", "RANSAC", "HoughLinesP", "Canny", "Dice", "IoU", "Dice coefficient",
    "homografía", "rectificación", "segmentación", "vectorización", "kerf", "offset", "Huber",
    "Ramer-Douglas-Peucker", "RDP", "matplotlib", "PIL", "exif_transpose", "sirena", "camión",
    "arcoíris", "estrella", "dinosaurio", "princesa", "unicornio", "mariposa", "toppers", "topper"
]

def call_anthropic_api(prompt, model_name, api_key):
    """
    Realiza una solicitud directa al endpoint de Anthropic utilizando el modelo especificado.
    """
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "max_tokens": 4000,
        "temperature": 0.8,
        "system": (
            "Eres un experto redactor académico de nivel doctoral en ingeniería de software y visión por computador. "
            "Tu tarea es reescribir el texto provisto para que suene 100% como si hubiera sido escrito por un humano "
            "nativo en español, eludiendo por completo detectores de IA (GPTZero, Turnitin, etc.).\n\n"
            "DIRECTRICES DE HUMANIZACIÓN:\n"
            "1. RITMO Y RÁFAGAS (Burstiness): Varía drásticamente la longitud de las frases. Escribe algunas oraciones muy largas "
            "con explicaciones subordinadas, y síguelas de inmediato por frases de 3 a 5 palabras muy directas. Esto rompe la monotonía artificial.\n"
            "2. ALTA PERPLEJIDAD (Perplexity): Evita sinónimos o palabras hiper-predecibles de IA. No uses transiciones trilladas como "
            "'Además', 'Por lo tanto', 'En conclusión', 'Cabe destacar', 'Es importante mencionar'. Escribe de forma fluida y orgánica.\n"
            "3. VOZ ACTIVA Y NATURALIDAD: Utiliza la voz activa. Ocasionalmente introduce la perspectiva del desarrollador "
            "('decidimos', 'encontramos', 'nuestras pruebas mostraron') para reflejar un diario de ingeniería real y vivido.\n"
            "4. PROTECCIÓN TÉCNICA RIGUROSA: No debes traducir, alterar ni cambiar ninguno de los términos técnicos protegidos. "
            "Mantén la precisión rigurosa de las ecuaciones, variables and fórmulas matemáticas.\n"
            "5. Sin preámbulos: Devuelve UNICAMENTE el texto reescrito. No digas 'Aquí tienes tu texto humanizado' ni similar."
        ),
        "messages": [
            {
                "role": "user",
                "content": f"Por favor, reescribe de forma orgánica y humana el siguiente texto, respetando las directrices:\n\n{prompt}"
            }
        ]
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["content"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8")
        print(f"Error de API Anthropic: {e.code} - {error_msg}")
        return None
    except Exception as e:
        print(f"Error de conexión: {e}")
        return None

def humanize_text(text, model_name, api_key):
    """
    Procesa y humaniza un bloque de texto protegiendo los términos técnicos mediante marcadores de posición.
    """
    # 1. Proteger términos técnicos usando tokens temporales (__TERM_0__, __TERM_1__, etc.)
    placeholders = {}
    temp_text = text
    
    # Ordenar términos por longitud descendente para evitar sub-reemplazos
    sorted_terms = sorted(PROTECTED_TERMS, key=len, reverse=True)
    
    for idx, term in enumerate(sorted_terms):
        placeholder = f"__TERM_{idx}__"
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        matches = pattern.findall(temp_text)
        if matches:
            placeholders[placeholder] = matches[0]
            temp_text = pattern.sub(placeholder, temp_text)
            
    # 2. Llamar a Claude para humanizar el texto con marcadores
    print(f"  [Humanizer] Enviando bloque a modelo {model_name}...")
    humanized_temp = call_anthropic_api(temp_text, model_name, api_key)
    
    if not humanized_temp:
        print("  [Humanizer] Fallo al obtener respuesta del modelo.")
        return text
        
    # 3. Restaurar los términos técnicos protegidos
    final_text = humanized_temp
    for placeholder, original_value in placeholders.items():
        final_text = final_text.replace(placeholder, original_value)
        
    return final_text

def process_file(file_path, output_path, model_name, api_key):
    if not os.path.exists(file_path):
        print(f"Error: El archivo no existe en {file_path}")
        return
        
    print(f"\nLeyendo archivo: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    paragraphs = content.split("\n\n")
    processed_paragraphs = []
    
    print(f"Total de párrafos a procesar: {len(paragraphs)}")
    for idx, para in enumerate(paragraphs):
        strip_para = para.strip()
        if not strip_para:
            processed_paragraphs.append("")
            continue
            
        if strip_para.startswith("\\") and not strip_para.startswith("\\section") and not strip_para.startswith("\\subsection"):
            processed_paragraphs.append(para)
            continue
            
        print(f"Procesando párrafo {idx+1}/{len(paragraphs)}...")
        humanized = humanize_text(para, model_name, api_key)
        processed_paragraphs.append(humanized)
        
    final_content = "\n\n".join(processed_paragraphs)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_content)
    print(f"\nArchivo humanizado guardado exitosamente en:\n{output_path}")

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: No se encontró la variable de entorno 'ANTHROPIC_API_KEY'.")
        print("Por favor, configúrala en PowerShell antes de ejecutar este script:")
        print("  $env:ANTHROPIC_API_KEY='tu_api_key_aquí'")
        sys.exit(1)
        
    if len(sys.argv) < 2:
        print("Uso del script:")
        print("  python humanizer.py <ruta_archivo_entrada.tex> [ruta_archivo_salida.tex] [nombre_modelo_api]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    
    # Valores por defecto
    output_file = None
    model_name = "claude-4-6-opus"  # Modelo recomendado por defecto
    
    if len(sys.argv) == 3:
        # Si se pasan 2 argumentos, el segundo puede ser la salida o el modelo
        arg2 = sys.argv[2]
        if arg2.endswith(".tex") or arg2.endswith(".txt"):
            output_file = arg2
        else:
            model_name = arg2
    elif len(sys.argv) >= 4:
        output_file = sys.argv[2]
        model_name = sys.argv[3]
        
    if not output_file:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_humanizado{ext}"
        
    print(f"Modelo seleccionado para humanización: {model_name}")
    process_file(input_file, output_file, model_name, api_key)

if __name__ == "__main__":
    main()
