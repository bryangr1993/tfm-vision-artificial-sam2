# Experimentos reproducibles

Este directorio contiene los scripts que generan la evidencia experimental vigente. La comparación sintética usa exclusivamente el corpus `asset_identity_v2` y su regla `asset_identity_disjoint`.

## Organización

- `datos/`: importa los 32 activos, genera las 48 hojas, audita la separación por identidad y ejecuta las pruebas del corpus.
- `random_forest/`: extrae las 19 variables, realiza la búsqueda agrupada de hiperparámetros, selecciona el umbral con hojas completas, ejecuta la prueba bloqueada y cuantifica la transferencia real.
- `sam2/`: selecciona el margen de caja solo en validación y evalúa SAM 2 con prompts ideales y operativos.
- `vision_clasica/`: conserva la evidencia de la línea base fija y del localizador que genera las cajas operativas.
- `comparacion/`: evalúa la línea clásica, audita la referencia real asistida, mide tiempos, valida la integración y genera las figuras comparativas.
- `geometria/`: mide por separado la repetibilidad WCS y la estabilidad de los contornos entre capturas.

## Reglas del protocolo

- Las identidades, nombres y hashes de activo son disjuntos entre entrenamiento, validación y prueba.
- Las familias F1-F4 son estratos compartidos y están presentes en cada partición.
- Las decisiones de Random Forest y SAM 2 se bloquean antes de consultar TEST.
- La línea clásica permanece fija y no se ajusta con TEST.
- Las métricas reales se denominan concordancia con una referencia canónica asistida.
- Las trece capturas reales representan una sola lámina física.
- La dispersión WCS no equivale a exactitud metrológica.
- La reapertura DXF es una validación estructural y no sustituye RDWorks ni un corte físico.

## Documentos de ejecución

- Protocolo completo: `documentacion/PROTOCOLO_EXPERIMENTAL.md`.
- SAM 2, tiempos, WCS e integración: `experimentos/PROTOCOLO_SAM2_WCS.md`.
- Controles técnicos: `documentacion/VALIDACION_TECNICA.md`.
- Mapa de figuras: `documentacion/MAPA_FIGURAS_CUANTITATIVAS.md`.

Todos los comandos se ejecutan desde la raíz del repositorio. Los archivos de selección y los resúmenes guardan hashes del manifiesto, los modelos y el checkpoint para mantener la trazabilidad.
