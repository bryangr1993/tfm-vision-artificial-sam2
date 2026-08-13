# Protocolo experimental de la versión 7

## Pregunta y unidad experimental

El estudio comprueba si Random Forest y SAM 2 pueden segmentar los toppers con suficiente fidelidad geométrica para alimentar la vectorización y cómo cambia su desempeño al pasar del banco sintético a capturas reales. La unidad experimental es la lámina completa, no un píxel aislado.

## Datos y bloqueo de decisiones

El manifiesto contiene 48 hojas sintéticas y 13 capturas reales rectificadas de una misma hoja física. Las hojas sintéticas se dividieron en 36 de entrenamiento, 6 de validación y 6 de prueba. Ningún píxel de una lámina aparece en más de una partición.

El conjunto sintético de prueba permaneció bloqueado hasta elegir el candidato y el umbral de Random Forest y la estrategia de prompts de SAM 2. Las capturas reales se utilizaron después como evaluación externa de transferencia y no intervinieron en la selección.

## Referencias

Las máscaras sintéticas proceden del canal alfa utilizado por el generador. La referencia real es una máscara canónica asistida construida sobre la mediana de las trece capturas alineadas. Se inicializó cada región con GrabCut, se reforzaron bordes y se inspeccionó visualmente el resultado. Las predicciones de visión clásica, Random Forest y SAM 2 no se utilizaron como etiquetas de píxel.

La referencia real representa ocho diseños de una sola hoja. Los intervalos entre capturas cuantifican variabilidad de adquisición, no generalización a productos nuevos.

## Optimización y selección de Random Forest

La búsqueda aleatoria evaluó diez candidatos mediante cuatro particiones `GroupKFold`, con la lámina como grupo. Se ajustaron número de árboles, profundidad máxima, mínimos por hoja y división, selección de características y balanceo de clases. El ajuste utilizó 144 000 píxeles muestreados de las 36 hojas de entrenamiento.

La selección final no se basó únicamente en el Jaccard de píxeles muestreados. El ganador de la búsqueda, el control con aumento RF v2 y el control sin aumento RF v1 se evaluaron sobre las seis hojas completas de validación. La puntuación utilizada fue:

`IoU medio - 0,005 × error medio absoluto de componentes`.

RF v1 con umbral 0,70 obtuvo la mayor puntuación de validación y quedó bloqueado para la prueba. El archivo `resultados/metricas/rf_consolidated_selection.json` declara de forma explícita que la prueba no intervino en la selección.

## Selección de prompts de SAM 2

SAM 2 Hiera Tiny se mantuvo preentrenado y sin ajuste fino. En validación se compararon cinco estrategias: caja ajustada, márgenes de 5 % y 10 %, caja con punto positivo y caja con puntos positivos y negativos. La caja con margen de 5 % obtuvo el mayor IoU medio y se aplicó sin cambios a prueba y a las capturas reales.

En sintético, las cajas proceden de metadatos conocidos. En real, un localizador clásico genera cajas aproximadas. SAM 2 produce la máscara final. Esta dependencia se declara en la memoria y no se presenta el modelo como detector autónomo.

## Métricas

Las métricas se calcularon por lámina:

- Dice e IoU para solapamiento.
- Boundary F1 con tolerancia de tres píxeles.
- Error absoluto en el número de componentes.
- Tiempo de inferencia y posprocesamiento.

En las trece capturas reales también se calcularon intervalos bootstrap del 95 %. Debido a que todas corresponden a una misma hoja, esos intervalos describen variación entre adquisiciones.

## Trazabilidad

Las decisiones finales se registran en archivos JSON y las métricas por hoja en CSV bajo `resultados/metricas/`. Las figuras cuantitativas se generan desde esos archivos mediante `experimentos/comparacion/generar_figuras_v7.py`. El manifiesto, los checksums de modelos, el informe de calidad y las pruebas de integración permiten auditar la cadena desde los datos hasta el PDF.
