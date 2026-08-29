# Informe de calidad de datos

## Corpus y granularidad

El corpus sintético vigente es `asset_identity_v2`. La unidad de análisis y partición es una hoja A4 completa con ocho instancias. Contiene 32 identidades de activo y 48 hojas: 24 de entrenamiento, 12 de validación y 12 de prueba.

Las trece capturas reales forman una evaluación externa separada. Corresponden a adquisiciones repetidas de una sola lámina física y no se cuentan como trece diseños independientes.

## Regla de independencia

La partición se fija antes de generar las hojas y aplica la regla `asset_identity_disjoint`:

- cruces por identificador de activo: 0.
- cruces por nombre de archivo fuente: 0.
- cruces por hash de activo: 0.
- cruces por hash de imagen: 0.
- cruces por hash de máscara: 0.

Las familias F1-F4 no son disjuntas. Funcionan como estratos compartidos y están representadas en entrenamiento, validación y prueba.

## Controles ejecutados

| Comprobación | Resultado |
|---|---:|
| Hojas con rutas obligatorias ausentes | 0 |
| Pares imagen-máscara con dimensiones incompatibles | 0 |
| Máscaras binarias o de instancia inválidas | 0 |
| Activos sin asignación o sin uso | 0 |
| Fallos críticos de la puerta de calidad | 0 |
| Pruebas automáticas superadas | 3 de 3 |

La regeneración completa produjo el mismo SHA-256 del manifiesto antes y después: `d850a74911feb592c33c7fc66e686d3093d34a33036742f981d1c2ab232c10dc`.

## Referencia real

La referencia canónica se construyó con asistencia algorítmica mediante la mediana de trece capturas rectificadas, cajas del localizador clásico, GrabCut, detección de bordes y morfología. No copia directamente las máscaras clásicas, pero depende de sus cajas y no dispone de una anotación manual independiente documentada.

Por ello, las métricas reales describen concordancia. No deben denominarse exactitud ni usarse para afirmar generalización a nuevos diseños físicos. La auditoría conserva como acción pendiente una anotación manual separada con registro de autor y cambios.

## Evidencia reproducible

- Manifiesto: `datos/manifiesto/datasets_asset_identity_v2.csv`.
- Puerta de calidad: `resultados/metricas/dataset_identity_v2_quality.json`.
- Regeneración y pruebas: `resultados/metricas/dataset_identity_v2_reproducibility.json`.
- Auditoría de la referencia real: `resultados/metricas/real_reference_validation_v8.json`.
- Informe ampliado: `documentacion/INFORME_CALIDAD_DATOS_IDV2.md`.

Las cifras del corpus sintético que no procedan de `asset_identity_v2` quedan fuera del protocolo vigente.
