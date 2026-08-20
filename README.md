# Registro geométrico y vectorización de toppers mediante visión artificial e IA

Trabajo Fin de Máster y aplicación para convertir fotografías de láminas impresas en contornos vectoriales aptos para corte láser CNC. El sistema detecta la referencia geométrica, rectifica la imagen, segmenta los objetos con SAM 2 y genera un archivo DXF que conserva la escala física del montaje.

Repositorio público: <https://github.com/bryangr1993/tfm-vision-artificial-sam2>

## Arquitectura

La solución combina visión clásica e inteligencia artificial:

1. Detección de marcadores y estimación de la homografía.
2. Rectificación y recuperación del sistema de coordenadas de trabajo (WCS).
3. Generación de indicaciones espaciales para SAM 2.
4. Segmentación de los objetos.
5. Extracción, simplificación y vectorización de contornos.
6. Exportación DXF en capas separadas para corte y referencia geométrica.

El método clásico y Random Forest se incluyen como líneas base experimentales. La aplicación utiliza SAM 2 como segmentador operativo y comunica cualquier fallo de inicialización sin sustituir silenciosamente el método seleccionado.

## Resultados principales

| Método | IoU sintético | IoU real | Error de componentes real |
|---|---:|---:|---:|
| Visión clásica | 0,9016 | 0,9787 | 0,00 |
| Random Forest | 0,9902 | 0,2877 | 7,00 |
| SAM 2 Hiera Tiny | 0,9153 | 0,9536 | 0,00 |

Aunque la visión clásica alcanza el mayor IoU en las capturas reales, SAM 2 mantiene las ocho instancias esperadas y produce una salida vectorizable en todas las imágenes evaluadas.

## Instalación

Se recomienda Python 3.12 y un entorno virtual independiente:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-core.txt
pip install -r requirements-ai.txt
```

Los checkpoints y modelos entrenados no se almacenan en Git por su tamaño. Sus nombres, procedencia y sumas SHA-256 se documentan en `resultados/modelos/README.md`.

## Ejecución

```powershell
python software/src/gui.py
```

Si se dispone de las capturas descritas en `datos/README.md`, puede iniciarse una demostración con:

```powershell
python software/src/gui.py --image datos/reales/raw/20.jpg --process
```

## Reproducibilidad

El protocolo completo, las particiones y el orden de ejecución están documentados en `documentacion/PROTOCOLO_EXPERIMENTAL.md`. Las comprobaciones técnicas del entregable se resumen en `documentacion/VALIDACION_TECNICA.md`.

```powershell
python experimentos/construir_manifiesto.py
python experimentos/random_forest/optimize_rf.py
python experimentos/sam2/evaluate_sam2.py
python experimentos/sam2/evaluate_sam2_real.py
python -m unittest discover software/tests -v
```

## Estructura

- `memoria/`: fuentes LaTeX, bibliografía y figuras.
- `software/`: aplicación, segmentadores, configuración y pruebas.
- `datos/`: manifiesto, anotaciones y documentación de acceso.
- `experimentos/`: optimización y evaluación de los métodos.
- `resultados/`: métricas, predicciones y evidencias de integración.
- `documentacion/`: protocolo experimental y controles técnicos.
- `output/pdf/`: memoria compilada.

La memoria completa está disponible en `output/pdf/TFM_Bryan_Guananga.pdf`.
