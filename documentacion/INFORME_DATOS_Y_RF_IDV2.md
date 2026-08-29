# Reconstrucción del corpus y del Random Forest

## Resultado ejecutivo

La puerta de calidad del corpus `asset_identity_v2` fue aprobada con 0 fallos críticos. Se generaron 48 láminas: 24 de entrenamiento, 12 de validación y 12 de prueba.

La partición se realizó antes de generar las láminas. Cada una de las cuatro familias aporta cuatro identidades a entrenamiento, dos a validación y dos a prueba. Las familias son estratos compartidos; las identidades, nombres de archivo y hashes SHA-256 son disjuntos.

El modelo seleccionado usa 240 árboles, profundidad máxima 20, hoja mínima 2, `max_features=0.5` y umbral 0.55. En validación completa obtuvo IoU 0.991814 ± 0.009651, sin error en el número de componentes.

En prueba bloqueada el RF seleccionado obtuvo IoU 0.987928 ± 0.008860, Dice 0.993909, F1 de contorno 0.941233 y error medio de componentes 0.0. El control alcanzó IoU 0.987517. La diferencia de IoU fue +0.000411; es una mejora modesta y no justifica afirmar superioridad estadística.

La condición más exigente fue C5, con IoU medio 0.970706. Corresponde al desenfoque gaussiano combinado con ruido, lo que identifica una debilidad concreta del clasificador.

## Ablación de características

| Variante | Variables | Jaccard diagnóstico |
|---|---:|---:|
| color 9 | 9 | 0.952334 |
| color gradients 12 | 12 | 0.963558 |
| full 19 | 19 | 0.981016 |
| full 19 plus coordinates 21 | 21 | 0.980028 |

El conjunto completo de 19 variables mejora las variantes de color y gradientes. Añadir coordenadas reduce ligeramente el resultado, por lo que se excluyen del modelo principal. Esta ablación es diagnóstica y no intervino en la selección.

Las cinco mayores importancias por disminución de impureza fueron: saturation (41.1 %), hue (18.8 %), grad_magnitude (11.8 %), blue (4.4 %), green (4.0 %). Estas importancias son descriptivas y pueden favorecer variables continuas correlacionadas.

## Integridad experimental

- La búsqueda usó cuatro pliegues disjuntos por cohortes de identidad.
- La selección final se realizó en láminas completas de validación.
- Modelo, umbral y hashes se bloquearon antes de consultar prueba.
- El script de prueba se niega a sobrescribir una evaluación existente.
- El tiempo incluye extracción de características y predicción, pero no describe todavía el tiempo extremo a extremo del software con adquisición, rectificación y exportación.

## Limitaciones

Las doce láminas de prueba reutilizan el mismo banco de ocho identidades bajo seis condiciones y dos disposiciones. Por ello no son doce observaciones poblacionales independientes. Los promedios sirven para describir sensibilidad controlada, pero no para construir una afirmación inferencial fuerte. La transferencia a fotografías reales debe informarse como una evaluación externa separada y no debe emplearse para reajustar el modelo.

## Evaluación externa real posterior al bloqueo

Sin modificar modelo ni umbral, se procesaron 13 capturas de una sola lámina física. La concordancia media con la referencia canónica asistida fue IoU 0.288068 ± 0.003660, Dice 0.447275 y F1 de contorno 0.236577. El error absoluto medio fue de 7.0 componentes y ninguna de las trece capturas produjo las ocho componentes esperadas.

El tiempo observado fue 19.111 s por lámina e incluye extracción de las 19 características, clasificación y postprocesamiento. Se obtuvo con otros procesos del proyecto activos y, por ello, es descriptivo: no sustituye al banco temporal final controlado. La salida suele colapsar el papel en una gran región, lo que confirma una brecha de dominio marcada entre síntesis y captura real.

Estas cifras son **concordancia con una referencia asistida**, no exactitud frente a una anotación independiente. Tampoco representan trece diseños reales distintos. Constituyen evidencia de estabilidad entre adquisiciones de una única hoja y documentan honestamente el fallo de transferencia del RF.
