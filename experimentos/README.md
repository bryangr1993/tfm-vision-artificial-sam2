# Experimentos reproducibles de la versión 7

Este directorio conserva únicamente las rutas utilizadas para construir la evidencia final del TFM. Los scripts exploratorios de versiones anteriores se retiraron porque dependían de directorios locales ya inexistentes y no formaban parte del protocolo bloqueado.

- `construir_manifiesto.py`: genera el manifiesto canónico y ejecuta los controles de calidad de datos.
- `random_forest/`: optimización agrupada, validación por lámina y consolidación de la selección de Random Forest.
- `sam2/`: selección de prompts y evaluación final de SAM 2 en los dominios sintético y real.
- `comparacion/`: referencia real asistida, líneas base, comparación consolidada y figuras cuantitativas.

Los comandos exactos y el orden de ejecución se documentan en `documentacion/PROTOCOLO_EXPERIMENTAL.md` y en el Anexo A de la memoria.
