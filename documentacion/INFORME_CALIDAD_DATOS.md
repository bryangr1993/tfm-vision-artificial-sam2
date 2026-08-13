# Informe de calidad de datos

## Dataset y granularidad

El manifiesto contiene **61 láminas**. La unidad de análisis y partición es una lámina completa. Hay **48 muestras sintéticas** y **13 capturas reales rectificadas**.

Las particiones sintéticas son: entrenamiento **36**, validación **6** y prueba **6**. Las **13** imágenes reales forman un conjunto de evaluación externo al entrenamiento sintético.

## Controles ejecutados

- existencia de imágenes, referencias y metadatos declarados;
- coincidencia de dimensiones entre imagen y máscara;
- valores binarios en las máscaras disponibles;
- unicidad del contenido de las imágenes mediante SHA-256;
- separación por identificador de lámina;
- disponibilidad de una verdad terreno real independiente.

## Hallazgos

| Comprobación | Resultado | Riesgo |
|---|---:|---|
| Rutas obligatorias ausentes | 0 | Bajo |
| Pares imagen-máscara con dimensiones distintas | 0 | Bajo |
| Máscaras no binarias | 0 | Bajo |
| Imágenes duplicadas por contenido | 0 grupos | Bajo |
| Anotaciones reales pendientes de revisión | 0 | Bajo |

La referencia real canónica está disponible y superó el control visual. Se construyó sobre la mediana de las trece capturas rectificadas de una misma lámina, sin usar como etiquetas las salidas clásicas ni las de SAM 2.

## Remediación

No quedan acciones de remediación abiertas sobre las referencias. La naturaleza asistida de la anotación y el uso de una sola lámina física se declararán como limitaciones del conjunto real. Los experimentos bloquean el conjunto sintético de prueba durante la selección de hiperparámetros y de prompts. El manifiesto y este informe se regeneran después de cada cambio de anotación.

## Detalle de incidencias

- Rutas ausentes: ninguna.
- Dimensiones incompatibles: ninguna.
- Valores de máscara no binarios: ninguno.
- Duplicados: ninguno.
