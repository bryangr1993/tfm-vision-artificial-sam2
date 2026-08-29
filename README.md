# Registro geométrico y vectorización de toppers mediante visión artificial e IA

Trabajo Fin de Máster y aplicación para convertir fotografías de láminas impresas en contornos vectoriales aptos para corte láser CNC. El flujo detecta la referencia geométrica, rectifica la imagen, segmenta los objetos con SAM 2 y genera un archivo DXF que conserva la escala física del montaje.

Repositorio público: <https://github.com/bryangr1993/tfm-vision-artificial-sam2>

## Arquitectura

La solución operativa combina visión clásica e inteligencia artificial:

1. Detección de marcadores y estimación de la homografía.
2. Rectificación y recuperación del sistema de coordenadas de trabajo (WCS).
3. Localización clásica de ocho cajas de indicación.
4. Segmentación de los objetos con SAM 2 Hiera Tiny.
5. Extracción y simplificación de ocho siluetas exteriores.
6. Exportación DXF en capas separadas para corte y referencia geométrica.

La visión clásica y Random Forest se utilizan como líneas base experimentales. La aplicación emplea la ruta operativa de SAM 2 y comunica cualquier fallo de inicialización sin sustituir silenciosamente el método seleccionado.

## Protocolo experimental vigente

La comparación sintética utiliza exclusivamente `asset_identity_v2`, formado por 32 activos gráficos y 48 hojas A4 de ocho instancias. La asignación se fija antes de generar las hojas:

- Entrenamiento: 16 identidades de activo y 24 hojas.
- Validación: 8 identidades de activo y 12 hojas.
- Prueba bloqueada: 8 identidades de activo y 12 hojas.

Los identificadores, nombres y hashes de los activos son disjuntos entre particiones. Las cuatro familias semánticas F1-F4 son estratos compartidos y están representadas en cada partición. No deben describirse como familias disjuntas. La auditoría también descarta cruces por hash de imagen o máscara.

Random Forest compara diez configuraciones mediante cuatro pliegues que mantienen intacta cada cohorte de identidades gráficas. Después evalúa tres finalistas y siete umbrales sobre las doce hojas completas de validación. El candidato C07, con 19 variables y umbral 0,55, queda bloqueado antes de abrir la prueba. SAM 2 permanece preentrenado y selecciona un margen de caja del 5 % entre cuatro alternativas usando solo las doce hojas de validación.

Los 32 PNG fuente no forman un conjunto de datos público. Su procedencia, hashes y restricción de redistribución se documentan en `datos/fuentes_toppers/README.md` y `datos/manifiesto/asset_registry_v2.csv`.

## Resultados en prueba sintética bloqueada

Los tres métodos se comparan sobre las mismas doce hojas de prueba. IoU, Dice y Boundary F1 se calculan por hoja. Boundary F1 usa una tolerancia de tres píxeles.

| Método | IoU | Dice | Boundary F1 | Error abs. de componentes |
|---|---:|---:|---:|---:|
| Visión clásica fija | 0,9098 | 0,9528 | 0,8039 | 0,00 |
| Random Forest seleccionado | 0,9879 | 0,9939 | 0,9412 | 0,00 |
| SAM 2 operativo | 0,9231 | 0,9600 | 0,7376 | 0,00 |

Random Forest obtuvo el mayor solapamiento en el banco sintético. La diferencia de IoU entre el modelo seleccionado y su control fue de 0,0004, por lo que no se interpreta como evidencia de superioridad estadística.

## Concordancia en capturas reales

La evaluación externa comprende trece adquisiciones de una sola lámina física. La referencia canónica se construyó con asistencia algorítmica a partir de la mediana de las capturas, cajas propuestas por el localizador clásico, GrabCut, Canny y morfología. No es una verdad de terreno manual independiente.

| Método | IoU de concordancia | Dice de concordancia | Boundary F1 de concordancia | Error abs. de componentes |
|---|---:|---:|---:|---:|
| Visión clásica fija | 0,9787 | 0,9892 | 0,9069 | 0,00 |
| Random Forest seleccionado | 0,2881 | 0,4473 | 0,2366 | 7,00 |
| SAM 2 operativo | 0,9536 | 0,9763 | 0,7200 | 0,00 |

Estas cifras describen concordancia y sensibilidad a la adquisición, no exactitud ni generalización a diseños físicos nuevos. La referencia usa cajas del localizador clásico, aunque no copia sus máscaras. Esta dependencia exige cautela adicional al interpretar la concordancia de la línea clásica. Random Forest no transfirió adecuadamente al dominio real. SAM 2 mantuvo las ocho instancias en las trece capturas y proporciona la máscara utilizada por la aplicación.

## Tiempo de segmentación

El banco operativo principal usa cuatro capturas reales rectificadas y precargadas, cinco repeticiones por captura y un calentamiento por método. Se ejecuta en CPU, dentro del mismo proceso y hardware. Incluye la ruta completa de cada segmentador desde la imagen rectificada hasta la máscara, pero excluye carga de archivos y modelos, rectificación y detección WCS.

| Método | Tiempo medio ± DE (s) | Corridas medidas |
|---|---:|---:|
| Visión clásica fija | 0,1367 ± 0,0198 | 20 |
| SAM 2 operativo | 2,1736 ± 0,2161 | 20 |
| Random Forest seleccionado | 12,7163 ± 1,6652 | 20 |

Un banco auxiliar sobre las doce hojas de prueba, con tres repeticiones por hoja, confirmó el mismo orden relativo: 0,1442 s para visión clásica, 2,2367 s para SAM 2 y 14,5438 s para Random Forest. Sus cifras se mantienen separadas porque corresponden a otra población de imágenes.

## Geometría e integración

El WCS se detectó en las diez capturas que contenían la marca y se rechazó correctamente en las tres que no la contenían. La desviación radial media del origen entre capturas fue de 0,1953 mm, con un máximo de 0,3899 mm, y la desviación estándar del ángulo del eje X fue de 0,0851°. Estos valores describen repetibilidad entre capturas y no exactitud metrológica.

La integración por lote produjo ocho siluetas exteriores y ocho trayectorias cerradas en cada una de las diez capturas con WCS. Las tres capturas sin WCS no generaron DXF. Los archivos esperados superaron una reapertura estructural con `ezdxf`. No se realizó un corte físico ni una importación en RDWorks.

## Instalación

Se recomienda Python 3.12 y un entorno virtual independiente:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-core.txt
pip install -r requirements-ai.txt
```

Los checkpoints y modelos entrenados no se almacenan en el historial ordinario por su tamaño. Sus nombres, procedencia y sumas SHA-256 se documentan en `resultados/modelos/README.md`.

## Ejecución de la aplicación

```powershell
python software/src/gui.py
```

Tras incorporar una copia autorizada de las capturas descritas en `datos/README.md`, puede iniciarse una demostración mediante:

```powershell
python software/src/gui.py --image datos/reales/raw/20.jpg --process
```

## Reproducción experimental

El orden completo, las reglas de bloqueo y el alcance de cada medición están en `documentacion/PROTOCOLO_EXPERIMENTAL.md` y `experimentos/PROTOCOLO_SAM2_WCS.md`. Una secuencia resumida es:

```powershell
python experimentos/datos/generate_identity_disjoint_dataset.py --overwrite
python experimentos/datos/audit_identity_disjoint_dataset.py
python experimentos/datos/test_identity_disjoint_dataset.py
python experimentos/random_forest/train_rf_identity_v2.py select
python experimentos/random_forest/train_rf_identity_v2.py ablate
python experimentos/random_forest/train_rf_identity_v2.py test
python experimentos/random_forest/evaluate_rf_identity_v2_real.py
python experimentos/comparacion/evaluate_classical_identity_v2.py
python experimentos/sam2/select_prompt_margin_identity_v2.py --device auto
python experimentos/comparacion/benchmark_segmentation_runtime.py --device auto
python experimentos/geometria/evaluate_wcs_repeatability.py
python experimentos/comparacion/run_integration_batch.py --device auto
python experimentos/comparacion/generate_protocol_figures.py
python experimentos/comparacion/generate_final_idv2_figures.py
python -m unittest discover software/tests -v
```

La evaluación de prueba de Random Forest se protege contra una segunda ejecución con el mismo bloqueo. Para regenerar todo desde cero debe utilizarse un directorio de resultados limpio y conservar los hashes producidos por la selección.

## Estructura

- `memoria/`: fuentes LaTeX, bibliografía y figuras.
- `software/`: aplicación, segmentadores, configuración y pruebas.
- `datos/`: manifiestos, anotaciones y documentación de acceso.
- `experimentos/`: generación, selección y evaluación reproducibles.
- `resultados/`: métricas, modelos, predicciones y evidencias de integración.
- `documentacion/`: protocolo experimental y controles técnicos.
- `output/pdf/`: memoria compilada.

## Memoria final

El documento completo puede consultarse en [`output/pdf/TFM_Bryan_Guananga_FInal.pdf`](output/pdf/TFM_Bryan_Guananga_FInal.pdf).
