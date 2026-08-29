# Validación técnica del entregable

Corte de evidencia: 27 de agosto de 2026.

## Evaluación general

La evidencia sintética es reproducible y está lista para compartirse. La comparación usa una población común, mantiene bloqueadas las decisiones antes de prueba y conserva hashes de los artefactos principales. Los resultados reales deben compartirse con una salvedad obligatoria: cuantifican concordancia con una referencia canónica asistida sobre trece capturas de una sola lámina física. No representan exactitud frente a una verdad terreno independiente ni generalización poblacional.

## Calidad e independencia de los datos

- Corpus sintético: `asset_identity_v2`.
- Unidad de análisis: una hoja A4 completa con ocho instancias.
- Activos fuente: 32 identidades.
- Hojas: 24 de entrenamiento, 12 de validación y 12 de prueba.
- Regla de partición: `asset_identity_disjoint`.
- Cruces entre particiones por identidad, nombre de archivo o hash de activo: 0.
- Cruces por hash de imagen o máscara: 0.
- Rutas ausentes, dimensiones incompatibles o máscaras inválidas: 0.
- Fallos críticos de la puerta de calidad: 0.
- Pruebas automáticas del corpus: 3 superadas y 0 fallidas.

Las familias F1-F4 son estratos semánticos presentes en entrenamiento, validación y prueba. No son disjuntas. La disyunción se aplica a la identidad del activo y queda fijada antes de generar las hojas.

La regeneración completa produjo el mismo hash del manifiesto y volvió a superar la puerta de calidad. El resultado está registrado en `resultados/metricas/dataset_identity_v2_reproducibility.json`.

## Selección y bloqueo de modelos

### Random Forest

- Contrato de entrada: 19 variables por píxel, sin coordenadas espaciales.
- Etapa 1: diez configuraciones y cuatro pliegues `GroupKFold` disjuntos por identidad.
- Etapa 2: tres candidatos sobre las doce hojas completas de validación y siete umbrales entre 0,35 y 0,65.
- Función objetivo: IoU medio menos 0,005 por el error medio absoluto de componentes.
- Semilla: 20260827.
- Ganador: C07 con 240 árboles, profundidad máxima 20, mínimo de dos muestras por hoja terminal, `max_features=0.5`, ponderación balanceada y umbral 0,55.
- Validación del ganador: IoU 0,9918 ± 0,0097, Dice 0,9959 y error de componentes 0,00.
- Estado: selección bloqueada antes de prueba. Los datos reales no participaron en el ajuste.

### SAM 2

- Modelo: SAM 2 Hiera Tiny preentrenado, sin ajuste fino.
- Indicaciones operativas: ocho cajas generadas por el localizador clásico.
- Alternativas de margen: 0 %, 3 %, 5 % y 10 %.
- Población de selección: doce hojas de validación.
- Margen seleccionado: 5 %.
- Validación del ganador: IoU 0,9232 ± 0,0032, Dice 0,9600, Boundary F1 0,7663 y error de componentes 0,00.
- Estado: margen bloqueado antes de prueba. Las cajas ideales se usan después como diagnóstico y no intervienen en la selección.

## Prueba sintética bloqueada

Los tres métodos usan las mismas doce hojas de `asset_identity_v2` TEST. Las desviaciones estándar se calculan entre hojas.

| Método | IoU, media ± DE | Dice, media ± DE | Boundary F1, media ± DE | Error abs. de componentes |
|---|---:|---:|---:|---:|
| Visión clásica fija | 0,9098 ± 0,0037 | 0,9528 ± 0,0020 | 0,8039 ± 0,0060 | 0,00 |
| Random Forest seleccionado | 0,9879 ± 0,0089 | 0,9939 ± 0,0045 | 0,9412 ± 0,0467 | 0,00 |
| SAM 2 operativo | 0,9231 ± 0,0072 | 0,9600 ± 0,0039 | 0,7376 ± 0,0178 | 0,00 |

La tabla consolidada y sus 36 filas de origen se validaron con `experimentos/comparacion/generate_final_idv2_figures.py`. La comprobación confirma población común, claves únicas, métricas dentro de rango, tolerancia de contorno de tres píxeles, bloqueos previos a prueba y alineación del hash del manifiesto.

Random Forest supera a su control por 0,0004 IoU, 0,0002 Dice y 0,0040 Boundary F1. La magnitud es pequeña y no sustenta por sí sola una afirmación de superioridad estadística.

## Concordancia en el dominio real

Las trece observaciones corresponden a una única lámina física. La referencia canónica se deriva de la mediana de trece capturas rectificadas y usa cajas del localizador clásico para inicializar GrabCut. Después aplica cierre de bordes y morfología. No copia directamente los píxeles de las máscaras clásicas, pero tampoco es metodológicamente independiente de ese localizador.

| Método | IoU de concordancia | Dice de concordancia | Boundary F1 de concordancia | Error abs. de componentes |
|---|---:|---:|---:|---:|
| Visión clásica fija | 0,9787 | 0,9892 | 0,9069 | 0,00 |
| Random Forest seleccionado | 0,2881 | 0,4473 | 0,2366 | 7,00 |
| SAM 2 operativo | 0,9536 | 0,9763 | 0,7200 | 0,00 |

La concordancia alta de la visión clásica debe leerse con cautela adicional por el uso de cajas clásicas durante la construcción de la referencia. Random Forest colapsó hacia una región dominante y no produjo ocho componentes en ninguna captura. SAM 2 y la línea clásica conservaron las ocho instancias en las trece adquisiciones. Estas observaciones describen transferencia y sensibilidad a la captura para una sola lámina, no desempeño esperado sobre nuevos diseños físicos.

La auditoría de la referencia asigna el estado `needs_independent_manual_review`. Antes de presentarla como verdad terreno independiente sería necesario realizar una anotación manual separada y conservar el registro de quién anotó, qué cambió y con qué criterio.

## Tiempo de segmentación

### Banco operativo principal

El banco usa cuatro capturas reales rectificadas de 2100 × 2970 píxeles, un calentamiento por método y cinco repeticiones por captura. Las veinte corridas de cada método se ejecutan en CPU, dentro del mismo proceso y hardware, con las imágenes y los modelos precargados.

| Método | Media ± DE (s) | Mediana (s) | Corridas |
|---|---:|---:|---:|
| Visión clásica fija | 0,1367 ± 0,0198 | 0,1330 | 20 |
| SAM 2 operativo | 2,1736 ± 0,2161 | 2,1430 | 20 |
| Random Forest seleccionado | 12,7163 ± 1,6652 | 12,1190 | 20 |

SAM 2 incluye localización de cajas, `set_image` con el codificador, decodificación y postproceso. Random Forest incluye extracción de las 19 variables, `predict_proba` y postproceso. La línea clásica incluye su llamada completa. Se excluyen lectura desde disco, carga de modelos, rectificación y WCS.

### Banco auxiliar en prueba sintética

Una comprobación adicional usa las doce hojas de TEST, tres repeticiones por hoja y orden rotatorio de métodos. Produjo 36 corridas por método: 0,1442 ± 0,0248 s para visión clásica, 2,2367 ± 0,3076 s para SAM 2 y 14,5438 ± 4,1488 s para Random Forest. Este banco confirma el orden relativo, pero no sustituye la latencia operativa principal porque usa otra población de imágenes.

## WCS e integración

- Capturas con WCS esperado y detectado: 10 de 10.
- Capturas sin WCS esperado y correctamente rechazado: 3 de 3.
- Desviación radial media del origen: 0,1953 mm.
- Desviación radial máxima del origen: 0,3899 mm.
- Desviación estándar del ángulo del eje X: 0,0851°.
- Capturas con ocho siluetas exteriores y sin huecos exportables: 13 de 13.
- DXF esperados, reabiertos y estructuralmente válidos: 10 de 10.
- Capturas sin WCS que no generaron DXF: 3 de 3.

La dispersión WCS compara capturas después de recalcular marcadores y homografía. No mide exactitud metrológica porque no existe una referencia física independiente. La validación DXF comprueba lectura de retorno, capas, número de polilíneas, cierre, finitud de coordenadas y límites plausibles. No incluye importación en RDWorks ni corte láser físico.

## Software

La batería ejecutada con Python 3.12 superó 23 de 23 pruebas. Cubre inicialización de SAM 2 sin sustitución silenciosa, paso de la máscara neuronal a la vectorización, configuración, estado de la aplicación, jerarquía de contornos, política de siluetas exteriores, bloqueos de exportación, reapertura DXF, integración sin interfaz y regresión WCS sobre controles positivos, negativos y ambiguos.

La comprobación integral del paquete superó 20 de 20 controles y registró 52 artefactos con SHA-256. Incluye calidad IDV2, bloqueo de SAM 2, referencia asistida, diseño temporal equilibrado, contrato RF de 19 variables, controles WCS, correspondencia de las etapas del detector vigente, separación entre estabilidad y geometría, política de siluetas exteriores y validación estructural del lote DXF. El estado almacenado en `resultados/metricas/protocol_package_validation.json` es `passed`.

## Trazabilidad principal

| Artefacto | SHA-256 |
|---|---|
| Manifiesto `asset_identity_v2` | `d850a74911feb592c33c7fc66e686d3093d34a33036742f981d1c2ab232c10dc` |
| Bloqueo de Random Forest | `6d6f14050b5a87de8400954dc304b35a9508a8f6333eca82675163948f3d1731` |
| Modelo Random Forest seleccionado | `4d2b085da6cd81d91d83ddb2a451a49c7d54e3ea829e856df0480cdd4ef06a5a` |
| Selección de margen de SAM 2 | `9b26689a00003378db4ba60de94f5b114b6ffd1bf1d9df91add9a05f5f0fcaf5` |
| Checkpoint SAM 2 Hiera Tiny | `65b50056e05bcb13694174f51bb6da89c894b57b75ccdf0ba6352c597c5d1125` |
| Referencia real canónica | `325d6b318809ae98e6b8e859063fa1f154788e520099fb2fa9be551edfa1d614` |
| Resumen de prueba de Random Forest | `b1cd10e69054ebb786311ef6ff180be4ee10e0d879fa6bdc260ccde42cb09721` |
| Resumen de concordancia real de Random Forest | `24e8638a6cf5d98d68962feafa5dcc99e2ed4caf8e6f8c40a40830993116df08` |
| Banco auxiliar de tiempos IDV2 | `39eff595d63a17ff41ebd38b65f98f997ae419c067ce361d3630c973e55458d7` |
| Validación integral del paquete | `3e6e363ded0ee9e662072fbd52e28221e1e29ee4626530c80a5d775814b546be` |
| Informe de SAM 2, WCS e integración | `61b951fc0e6ac888555dc44c33705cf91b26a798e4aa5aca38ba8813f107b8f1` |

## Fuentes de evidencia

- Calidad del corpus: `resultados/metricas/dataset_identity_v2_quality.json`.
- Reproducibilidad del corpus: `resultados/metricas/dataset_identity_v2_reproducibility.json`.
- Comparación sintética: `resultados/metricas/comparison_idv2_final_summary.csv`.
- Contraste sintético-real: `resultados/metricas/synthetic_real_gap_final.csv`.
- Validación de la comparación: `resultados/metricas/final_comparison_validation.json`.
- Auditoría de la referencia real: `resultados/metricas/real_reference_validation_v8.json`.
- Banco operativo de tiempos: `resultados/metricas/segmentation_runtime_summary_v8.json`.
- Banco auxiliar de tiempos: `resultados/metricas/segmentation_runtime_idv2_final_summary.json`.
- Repetibilidad WCS y estabilidad de contornos: `resultados/metricas/wcs_contour_repeatability_summary_v8.json`.
- Integración por lote: `resultados/metricas/integration_batch_summary_v8.json`.
- Validación integral del paquete: `resultados/metricas/protocol_package_validation.json`.
- Informe de SAM 2, WCS e integración: `resultados/INFORME_PROTOCOLO_SAM2_WCS.md`.
