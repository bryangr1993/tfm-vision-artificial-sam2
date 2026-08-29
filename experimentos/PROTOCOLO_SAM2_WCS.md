# Protocolo reproducible de SAM 2, WCS e integración

## Alcance

Este protocolo parte del corpus `asset_identity_v2` ya generado y auditado. Cubre la selección del margen de caja de SAM 2, la prueba bloqueada, la concordancia en capturas reales, el banco temporal operativo, la repetibilidad WCS y la integración hasta un DXF estructuralmente verificable.

Random Forest utiliza el modelo seleccionado con identidades disjuntas, 19 variables y umbral 0,55. SAM 2 Hiera Tiny permanece preentrenado y usa el margen de caja del 5 % elegido únicamente con las doce hojas de validación.

## Orden de ejecución

Los comandos se ejecutan desde la raíz del repositorio con el entorno virtual del proyecto:

```powershell
.\.venv\Scripts\python.exe experimentos\datos\audit_identity_disjoint_dataset.py
.\.venv\Scripts\python.exe experimentos\comparacion\validate_real_reference.py
.\.venv\Scripts\python.exe experimentos\sam2\select_prompt_margin_identity_v2.py --device auto
.\.venv\Scripts\python.exe experimentos\comparacion\benchmark_segmentation_runtime.py --device auto
.\.venv\Scripts\python.exe experimentos\geometria\evaluate_wcs_repeatability.py
.\.venv\Scripts\python.exe experimentos\geometria\generate_wcs_detection_stages.py
.\.venv\Scripts\python.exe experimentos\comparacion\run_integration_batch.py --device auto
.\.venv\Scripts\python.exe experimentos\comparacion\generate_protocol_figures.py
.\.venv\Scripts\python.exe experimentos\comparacion\verify_protocol_package.py
```

La comparación auxiliar sobre las doce hojas de TEST y sus tres figuras consolidadas se reproducen después mediante:

```powershell
.\.venv\Scripts\python.exe experimentos\comparacion\benchmark_idv2_final_runtime.py
.\.venv\Scripts\python.exe experimentos\comparacion\generate_final_idv2_figures.py
```

## Decisiones experimentales

### Prompts de SAM 2

- El escenario operativo recalcula ocho cajas con el mismo localizador clásico que usa la aplicación.
- La validación compara márgenes de 0 %, 3 %, 5 % y 10 % sobre doce hojas.
- La regla maximiza IoU medio menos 0,005 por el error medio absoluto de componentes. Los desempates usan Boundary F1 y después el margen menor.
- El margen del 5 % se bloquea antes de abrir TEST.
- El escenario con cajas ideales se calcula después del bloqueo y funciona únicamente como límite diagnóstico optimista.
- Los escenarios usan selección multimáscara respecto al área de la caja ampliada y el mismo postproceso del producto.

### Tiempo de segmentación

El banco operativo principal usa cuatro capturas reales rectificadas y precargadas. Ejecuta un calentamiento por método y cinco repeticiones por captura, para un total de veinte corridas medidas por método. Todas las rutas se ejecutan en CPU, en el mismo proceso y hardware.

- Visión clásica: llamada completa a `segment_toppers`.
- Random Forest: extracción de 19 variables, `predict_proba` y postproceso con umbral 0,55.
- SAM 2: localización de cajas, `set_image` con el codificador, decodificación de cajas y postproceso.

La carga de archivos y modelos, la rectificación y la detección WCS quedan fuera de este banco. La integración por lote mide el flujo completo por separado.

El banco auxiliar usa las doce hojas sintéticas de TEST, tres repeticiones por hoja y orden rotatorio. Sus resultados no se mezclan con la latencia operativa principal.

### WCS y estabilidad de contornos

La repetibilidad WCS parte de imágenes sin rectificar. Cada captura vuelve a pasar por detección de marcadores, homografía y reconstrucción del origen y del eje. Los resultados describen dispersión entre capturas y no exactitud metrológica, porque no existe una referencia física independiente.

La estabilidad de contornos se calcula por separado mediante comparaciones por pares de máscaras rectificadas. Un método puede ser estable entre capturas y, al mismo tiempo, presentar baja concordancia con la referencia.

### Integración y exportación

La política operativa conserva ocho siluetas exteriores. Los detalles impresos internos no se convierten en trayectorias de corte. El soporte de huecos existe como opción explícita y permanece desactivado para este producto.

La integración exige WCS válido y un DXF estructuralmente correcto en las diez capturas con la marca. Las tres capturas sin WCS deben bloquear la exportación. La comprobación abre de nuevo cada DXF con `ezdxf` y valida capas, cantidad de polilíneas, cierre, finitud de coordenadas y límites plausibles. No incluye importación en RDWorks ni corte físico.

### Referencia real

Las trece capturas corresponden a una sola lámina física. La referencia canónica se construye con asistencia algorítmica mediante la mediana de las capturas rectificadas, cajas del localizador clásico, GrabCut, Canny y morfología. No es una verdad terreno manual independiente.

Por esta razón, IoU, Dice y Boundary F1 reales se denominan métricas de concordancia. Los intervalos entre capturas describen sensibilidad a la adquisición y no generalización a diseños físicos nuevos. La dependencia de cajas clásicas exige cautela adicional al interpretar la concordancia de la línea clásica.

## Resultados bloqueados

| Dominio | Método | IoU | Dice | Boundary F1 | Error abs. de componentes |
|---|---|---:|---:|---:|---:|
| IDV2 TEST, 12 hojas | Visión clásica fija | 0,9098 | 0,9528 | 0,8039 | 0,00 |
| IDV2 TEST, 12 hojas | Random Forest seleccionado | 0,9879 | 0,9939 | 0,9412 | 0,00 |
| IDV2 TEST, 12 hojas | SAM 2 operativo | 0,9231 | 0,9600 | 0,7376 | 0,00 |
| Concordancia real, 13 capturas de 1 lámina | Visión clásica fija | 0,9787 | 0,9892 | 0,9069 | 0,00 |
| Concordancia real, 13 capturas de 1 lámina | Random Forest seleccionado | 0,2881 | 0,4473 | 0,2366 | 7,00 |
| Concordancia real, 13 capturas de 1 lámina | SAM 2 operativo | 0,9536 | 0,9763 | 0,7200 | 0,00 |

El banco temporal operativo produjo 0,1367 ± 0,0198 s para visión clásica, 2,1736 ± 0,2161 s para SAM 2 y 12,7163 ± 1,6652 s para Random Forest. La repetibilidad WCS clasificó correctamente las diez capturas positivas y las tres negativas. La integración produjo diez DXF estructuralmente válidos y bloqueó las tres exportaciones sin WCS.

## Salidas principales

### Selección y métricas

- `resultados/metricas/sam2_idv2_prompt_selection.json`
- `resultados/metricas/sam2_idv2_locked_test_operational_metrics.csv`
- `resultados/metricas/sam2_idv2_locked_scenario_summary.csv`
- `resultados/metricas/comparison_idv2_final_summary.csv`
- `resultados/metricas/synthetic_real_gap_final.csv`
- `resultados/metricas/final_comparison_validation.json`

### Tiempos

- `resultados/metricas/segmentation_runtime_runs_v8.csv`
- `resultados/metricas/segmentation_runtime_summary_v8.json`
- `resultados/metricas/segmentation_runtime_idv2_final_runs.csv`
- `resultados/metricas/segmentation_runtime_idv2_final_summary.json`

### Referencia, geometría e integración

- `resultados/metricas/real_reference_validation_v8.json`
- `resultados/metricas/wcs_registration_per_capture_v8.csv`
- `resultados/metricas/wcs_contour_repeatability_summary_v8.json`
- `resultados/metricas/wcs_detection_stages_v8.json`
- `resultados/metricas/integration_batch_v8.csv`
- `resultados/metricas/integration_batch_summary_v8.json`
- `resultados/metricas/protocol_package_validation.json`
- `resultados/INFORME_PROTOCOLO_SAM2_WCS.md`

### Figuras

- `resultados/figuras/comparison_test_idv2_final.png`
- `resultados/figuras/synthetic_real_gap_final.png`
- `resultados/figuras/runtime_comparison_idv2_final.png`
- `resultados/figuras_protocolo_v8/`

La correspondencia entre cada gráfico, su pregunta, población, fuente y límite interpretativo se documenta en `documentacion/MAPA_FIGURAS_CUANTITATIVAS.md`.
