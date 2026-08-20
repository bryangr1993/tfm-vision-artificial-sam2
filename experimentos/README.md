# Experimentos reproducibles

Este directorio reúne los scripts utilizados para construir la evidencia experimental del TFM. Cada etapa parte del manifiesto canónico y escribe sus métricas o artefactos en las rutas documentadas.

- `construir_manifiesto.py`: genera el manifiesto canónico y ejecuta los controles de calidad de datos.
- `random_forest/`: optimización agrupada, validación por lámina y consolidación de la selección de Random Forest.
- `sam2/`: selección de prompts y evaluación final de SAM 2 en los dominios sintético y real.
- `comparacion/`: referencia real asistida, líneas base, comparación consolidada y figuras cuantitativas.

Los comandos exactos y el orden de ejecución se documentan en `documentacion/PROTOCOLO_EXPERIMENTAL.md` y en el Anexo A de la memoria.
