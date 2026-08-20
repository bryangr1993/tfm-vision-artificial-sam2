# Mapa de figuras cuantitativas

| Figura | Pregunta | Forma | Datos | Mensaje respaldado | Paleta | Salida |
|---|---|---|---|---|---|---|
| Comparación entre dominios | ¿Cómo cambia el IoU de cada método entre sintético y real? | Barras agrupadas con desviación estándar | JSON finales, 3 métodos x 2 dominios | RF presenta la mayor brecha; SAM 2 se mantiene próximo a la línea clásica en real | Azul, naranja y oliva con tramas | `fig_brecha_dominios.pdf/png` |
| Métricas sintéticas | ¿Qué calidad geométrica logra cada método en prueba bloqueada? | Barras agrupadas | CSV de prueba, n=6 láminas | RF lidera IoU y Dice; los tres conservan las componentes | Mismos colores y tramas | `fig_metricas_sinteticas.pdf/png` |
| Selección de prompts | ¿Qué estrategia de SAM 2 se selecciona en validación? | Gráfico de puntos con eje enfocado | CSV de resumen, n=6 láminas por estrategia | La caja con 5% de margen obtiene el mayor IoU, aunque las diferencias son pequeñas | Oliva focal y grises | `fig_sam_prompts_validacion.pdf/png` |

Todas las figuras se construyen con `experimentos/comparacion/generar_figuras.py`. La revisión final se realiza sobre los PDF integrados en la memoria y sobre sus PNG de control.
