# Mapa de figuras cuantitativas

Este inventario corresponde al protocolo `asset_identity_v2` y a la evaluación operativa asociada.

## Comparación final

| Figura | Pregunta | Población y fuente | Lectura permitida | Salida |
|---|---|---|---|---|
| Comparación común en TEST | ¿Cómo se comparan los tres métodos sobre exactamente las mismas hojas de prueba? | 12 hojas IDV2 TEST por método. `comparison_idv2_final_per_sheet.csv` y `comparison_idv2_final_summary.csv` | RF obtiene el mayor IoU, Dice y Boundary F1 sintéticos. Los puntos muestran la dispersión por hoja. El eje enfocado se declara en la figura | `resultados/figuras/comparison_test_idv2_final.png` y `.pdf` |
| Contraste sintético-real | ¿Cómo cambia descriptivamente el IoU entre TEST sintético y las capturas reales? | 12 hojas sintéticas y 13 capturas de una sola lámina física. `synthetic_real_gap_final.csv` | RF presenta una brecha marcada. Las barras reales son concordancia con una referencia asistida, no exactitud ni generalización poblacional | `resultados/figuras/synthetic_real_gap_final.png` y `.pdf` |
| Tiempo auxiliar sobre TEST | ¿Se conserva el orden temporal al usar las doce hojas sintéticas? | 12 hojas, 3 repeticiones por hoja y 36 corridas por método. `runtime_comparison_idv2_final.csv` | La escala logarítmica se declara. Es evidencia auxiliar y no reemplaza el banco operativo sobre capturas reales | `resultados/figuras/runtime_comparison_idv2_final.png` y `.pdf` |
| Tiempo operativo principal | ¿Cuánto tarda cada ruta sobre capturas reales rectificadas y precargadas? | 4 capturas, 5 repeticiones y 20 corridas por método. `segmentation_runtime_runs_v8.csv` y `segmentation_runtime_summary_v8.json` | Visión clásica es la ruta más rápida, seguida de SAM 2 y RF. SAM 2 incluye localizador, codificador, decodificador y postproceso | `resultados/figuras_protocolo_v8/segmentation_runtime_comparison_v8.png` |
| Etapas temporales de SAM 2 | ¿Qué parte del tiempo de SAM 2 corresponde a localización, codificación, decodificación y postproceso? | Las mismas 20 corridas del banco operativo | El codificador forma parte del tiempo de segmentación y no debe omitirse al comparar métodos | `resultados/figuras_protocolo_v8/sam2_runtime_stages_v8.png` |

Las tres figuras comparativas finales se generan con `experimentos/comparacion/generate_final_idv2_figures.py`. El script valida población, claves, bloqueos, definiciones de métricas, hashes y alcance temporal antes de dibujar. El informe resultante es `resultados/metricas/final_comparison_validation.json`.

## Datos, líneas base y Random Forest

| Figura | Pregunta | Población y fuente | Lectura permitida | Salida |
|---|---|---|---|---|
| Contacto de particiones | ¿Qué identidades y condiciones aparecen en cada partición? | 48 hojas y 32 activos. Manifiesto IDV2 y registro de activos | Las identidades son disjuntas. Las familias F1-F4 son estratos compartidos presentes en las tres particiones | `resultados/figuras/dataset_idv2_split_contact.png` |
| Evidencia de umbrales HSV | ¿Qué proporción de objeto y fondo cubren los umbrales fijos de la línea clásica? | 12 hojas de validación. `hsv_threshold_evidence.json` | Es evidencia descriptiva de parámetros fijos, no una optimización realizada por el gráfico | `resultados/figuras/evidencia_umbrales_hsv.png` y `.pdf` |
| Cribado de hiperparámetros RF | ¿Cómo rinden las diez configuraciones en los cuatro pliegues agrupados por identidad? | Píxeles muestreados de entrenamiento. `rf_idv2_search_folds.csv` y `rf_idv2_search_summary.csv` | Sirve para reducir candidatos. No es el resultado final por hoja ni usa prueba | `resultados/figuras/rf_idv2_hyperparameter_screening.png` |
| Validación por hoja completa RF | ¿Qué candidato y umbral maximizan la función objetivo en validación? | 3 candidatos, 7 umbrales y 12 hojas. `rf_idv2_validation_per_sheet.csv` y `rf_idv2_validation_summary.csv` | El ganador C07 con umbral 0,55 se fija antes de abrir TEST | `resultados/figuras/rf_idv2_full_sheet_validation.png` |
| Ablación de variables RF | ¿Qué aportan color, gradientes, multiescala, estadísticas locales y coordenadas? | Validación IDV2. `rf_idv2_feature_ablation.csv` | Las 19 variables mejoran frente a subconjuntos menores. Añadir coordenadas no mejora el IoU medio | `resultados/figuras/rf_idv2_feature_ablation.png` |
| Sensibilidad de hiperparámetros RF | ¿Qué patrones aparecen al agrupar candidatos por cada hiperparámetro? | Etapa de cribado. `rf_idv2_hyperparameter_sensitivity.csv` | Es un análisis descriptivo marginal. No identifica efectos causales aislados | `resultados/figuras/rf_idv2_hyperparameter_sensitivity.png` |
| Sensibilidad por condición en TEST | ¿Cómo varía RF entre las seis condiciones sintéticas? | 12 hojas de prueba. `rf_idv2_test_by_condition.csv` | Describe robustez dentro del corpus bloqueado. No autoriza reajustar el modelo | `resultados/figuras/rf_idv2_test_condition_sensitivity.png` |
| Transferencia real de RF | ¿Qué patrón produce el RF bloqueado al pasar a capturas reales? | 13 capturas de una lámina. Métricas y máscaras de concordancia RF | Hace visible el colapso hacia una región dominante. La referencia mostrada es asistida | `resultados/figuras/rf_idv2_real_agreement_examples.png` |

Las figuras de datos y Random Forest se generan con `experimentos/random_forest/analyze_rf_identity_v2.py`, salvo la evidencia HSV, que procede del análisis de la línea clásica.

## SAM 2, referencia real, WCS e integración

| Figura | Pregunta | Población y fuente | Lectura permitida | Salida |
|---|---|---|---|---|
| Selección del margen de caja | ¿Qué margen operativo se elige sin consultar TEST? | 12 hojas de validación y márgenes 0 %, 3 %, 5 % y 10 %. `sam2_idv2_margin_validation_summary.csv` | El margen del 5 % maximiza la función objetivo en validación. Las diferencias son pequeñas | `resultados/figuras_protocolo_v8/sam2_margin_selection_idv2.png` |
| Cajas ideales y operativas | ¿Cuánto cambia SAM 2 al sustituir cajas de referencia por cajas del localizador? | TEST IDV2 y capturas reales. `sam2_idv2_locked_scenario_summary.csv` | Las cajas ideales son un diagnóstico optimista. La cifra de producto es la del escenario operativo | `resultados/figuras_protocolo_v8/sam2_prompt_source_comparison_v8.png` |
| Procedencia de la referencia real | ¿Cómo se construyó y qué limitación conserva la preanotación canónica? | Mediana de 13 capturas, cajas clásicas, GrabCut, bordes y morfología. `real_reference_validation_v8.json` | Debe describirse como referencia asistida y pendiente de anotación manual independiente | `resultados/figuras_protocolo_v8/real_reference_provenance_v8.png` |
| Repetibilidad del origen WCS | ¿Cuánta dispersión entre capturas presenta el origen reconstruido? | 10 capturas con WCS. `wcs_registration_per_capture_v8.csv` | Mide dispersión entre adquisiciones. No es exactitud metrológica | `resultados/figuras_protocolo_v8/wcs_origin_repeatability_v8.png` |
| Estabilidad entre pares de contornos | ¿Qué tan parecidas son las máscaras entre capturas de la misma lámina? | 10 capturas y 45 pares por método. `contour_pairwise_stability_v8.csv` | Se informa separada de la geometría WCS. Un IoU alto de RF entre capturas no implica segmentación correcta | `resultados/figuras_protocolo_v8/contour_pairwise_stability_v8.png` |
| Estado de integración por lote | ¿Se cumplen la política WCS, las ocho siluetas y la exportación estructural? | 13 capturas. `integration_batch_v8.csv` y `integration_batch_summary_v8.json` | Diez DXF pasan la reapertura estructural y tres exportaciones se bloquean por ausencia de WCS. No demuestra importación en RDWorks ni corte físico | `resultados/figuras_protocolo_v8/integration_batch_status_v8.png` |

Estas figuras se generan con `experimentos/comparacion/generate_protocol_figures.py`. El mapa conserva los nombres físicos de los artefactos para que las rutas sean comprobables.

## Evidencia técnica del localizador

`resultados/figuras/flujo_localizador_clasico_real20.png` y su equivalente PDF muestran, sobre la captura `real_20`, la secuencia de rectificación, respuesta cromática, limpieza morfológica y obtención de las ocho cajas que alimentan a SAM 2. La figura se genera con `experimentos/vision_clasica/generar_flujo_localizador.py`.

Esta pieza explica el origen de los prompts operativos. No es una métrica de desempeño y no debe utilizarse para sostener una comparación cuantitativa entre métodos.

`resultados/figuras_protocolo_v8/wcs_detection_stages_v8.png` documenta sobre la misma captura la ROI de 45 × 45 mm, la binarización adaptativa, los segmentos de Hough y el origen con sus ejes finales. Se genera con `experimentos/geometria/generate_wcs_detection_stages.py` y conserva sus parámetros en `resultados/metricas/wcs_detection_stages_v8.json`. Explica el detector vigente, pero tampoco demuestra exactitud metrológica.

## Criterios de publicación

- Toda mención al dominio real debe usar el término **concordancia**.
- Debe indicarse junto a la figura que las trece capturas pertenecen a una sola lámina física.
- La referencia real debe describirse como canónica y asistida, nunca como verdad terreno independiente.
- La comparación temporal principal corresponde al banco de cuatro capturas reales y 20 corridas por método.
- El banco temporal sobre doce hojas sintéticas es auxiliar y debe mantener su población identificada.
- La repetibilidad WCS no debe presentarse como precisión o exactitud metrológica.
- La validación DXF es estructural. No sustituye una importación en RDWorks ni un corte físico.
