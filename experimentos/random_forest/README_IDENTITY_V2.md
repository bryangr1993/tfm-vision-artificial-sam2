# Protocolo Random Forest sobre `asset_identity_v2`

## Pregunta experimental

Se evalúa si un clasificador de píxeles Random Forest puede segmentar diseños sintéticos cuya identidad no apareció en entrenamiento. La unidad final de evaluación es la lámina completa. La familia semántica F1--F4 se usa como estrato compartido y no como unidad disjunta.

## Variables de entrada

El modelo principal utiliza exactamente 19 características por píxel:

- nueve canales de color: RGB, HSV y Lab;
- tres gradientes: Sobel x, Sobel y y magnitud;
- tres respuestas multiescala: dos suavizados gaussianos y su diferencia;
- cuatro estadísticas locales: media y desviación estándar en ventanas 7×7 y 15×15.

No se calcula un Laplaciano. Las coordenadas normalizadas x/y solo aparecen en la ablación de 21 variables y nunca en el modelo principal.

## Optimización en dos etapas

1. **Cribado por píxeles.** Se muestrean por lámina 1.500 píxeles de objeto y 1.500 de fondo, repartidos entre fondo cercano y lejano. Diez configuraciones, incluido el control previamente registrado, se comparan mediante `GroupKFold` de cuatro pliegues. Cada grupo reúne cuatro identidades fuente, una por familia. Ninguna identidad cruza entre pliegues.
2. **Validación por lámina completa.** Las dos mejores configuraciones del cribado y el control se reajustan con todo el entrenamiento. Sobre las doce láminas completas de validación se exploran siete umbrales entre 0,35 y 0,65. La regla fijada antes de prueba es:

   `IoU medio − 0,005 × error absoluto medio en el número de componentes`.

   IoU domina la decisión. La penalización equivale a medio punto porcentual por componente omitido o espurio y evita seleccionar máscaras muy fragmentadas cuando dos candidatos tienen solapamiento parecido.

El espacio explora 100--240 árboles, profundidades 12, 20, 30 o ilimitada, hojas mínimas de 1, 2, 4 u 8 píxeles muestreados, divisiones mínimas de 2, 5 o 10, tres reglas de selección de variables y dos ponderaciones de clase. La semilla `20260827` y el presupuesto de diez candidatos se registran en JSON.

## Control reproducible

El control reconstruye la configuración histórica con 200 árboles, profundidad máxima 25, `min_samples_leaf=2`, `max_features=sqrt` y ponderación `balanced`. `min_samples_leaf` es el mínimo de observaciones muestreadas en una hoja terminal del árbol; no significa “por lámina”.

## Bloqueo de prueba

Los comandos están separados deliberadamente:

```powershell
python experimentos/random_forest/train_rf_identity_v2.py select
python experimentos/random_forest/train_rf_identity_v2.py ablate
python experimentos/random_forest/train_rf_identity_v2.py test
```

`select` no abre imágenes de prueba y crea `rf_idv2_selection_locked.json`, que fija modelo, umbral y hashes. `test` comprueba esos hashes y se niega a ejecutarse si ya existe una evaluación. La ablación es diagnóstica y no interviene en la selección.

## Límites de inferencia

El conjunto contiene 32 identidades fuente y no representa toda la diversidad comercial. La prueba sintética mide transferencia a ocho identidades retenidas bajo transformaciones controladas. Las capturas reales constituyen una evaluación externa distinta y no deben usarse para cambiar hiperparámetros ni umbral.
