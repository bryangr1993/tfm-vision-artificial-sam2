# Registro geométrico y vectorización de toppers mediante visión artificial e IA

Este repositorio contiene la versión 7 del Trabajo Fin de Máster de Bryan Guillermo Guananga Rodríguez. El sistema rectifica una fotografía de una lámina de toppers, localiza los objetos, obtiene una máscara con SAM 2, extrae sus contornos y genera un archivo DXF apto para el flujo de corte láser.

Repositorio público: <https://github.com/bryangr1993/tfm-vision-artificial-sam2>

La arquitectura es híbrida. La visión clásica se ocupa de los marcadores, la homografía y la localización aproximada. SAM 2 produce la máscara operativa que alimenta la vectorización. Random Forest y el método clásico se mantienen como métodos de comparación experimental.

## Estado

La versión 7 integra SAM 2 como motor de segmentación de la aplicación y conserva los métodos clásico y Random Forest como comparaciones experimentales. Las decisiones, fases y criterios de aceptación se registran en `documentacion/PLAN_MAESTRO_CORRECCION_TFM_V7.md`.

## Estructura

- `memoria/`: fuentes LaTeX, bibliografía y figuras.
- `software/`: aplicación, segmentadores, configuración y pruebas.
- `datos/`: manifiestos, anotaciones y documentación de acceso.
- `experimentos/`: optimización y evaluación de Random Forest, SAM 2 y líneas base.
- `resultados/`: métricas, figuras generadas y metadatos de modelos.
- `documentacion/`: protocolo, trazabilidad y matriz de observaciones.
- `output/pdf/`: versión compilada y validada de la memoria.

## Reproducibilidad

El repositorio separa las dependencias del núcleo geométrico de las dependencias de IA. Los checkpoints y modelos pesados no se almacenan en el historial ordinario de Git. `resultados/modelos/README.md` registra nombre, tamaño, origen y checksum.

La GUI puede abrirse con `python software/src/gui.py`. Para una demostración reproducible puede utilizarse `python software/src/gui.py --image datos/reales/raw/20.jpg --process`.
