# Protocolo experimental

## Preguntas y unidades de análisis

El estudio comprueba si Random Forest y SAM 2 segmentan los toppers con una fidelidad geométrica útil para la vectorización y cómo cambia su comportamiento al pasar del banco sintético a capturas reales. La unidad principal en sintético es la hoja completa. Los píxeles solo se muestrean dentro del entrenamiento de Random Forest.

Las trece capturas reales corresponden a una única hoja física con ocho diseños. Sirven para estudiar sensibilidad a la adquisición y concordancia con una referencia canónica asistida. No se tratan como trece productos independientes.

## Datos e independencia

El conjunto asset_identity_v2 contiene 32 activos gráficos y 48 hojas A4 de ocho instancias. Las identidades se asignan antes de generar las hojas:

- Entrenamiento: 16 activos y 24 hojas.
- Validación: 8 activos y 12 hojas.
- Prueba: 8 activos y 12 hojas.

Las cuatro familias visuales son estratos compartidos, mientras que los identificadores, nombres y hashes de activo son disjuntos. La auditoría también exige que no coincidan hashes de imagen o máscara entre particiones. Entrenamiento se divide en cuatro grupos de identidades para la validación cruzada.

## Referencias de segmentación

Las máscaras sintéticas proceden del canal alfa durante la composición y se deforman junto con la imagen cuando corresponde. La referencia real se construye sobre la mediana píxel a píxel de las trece capturas rectificadas. Ocho cajas propuestas por el localizador clásico inicializan GrabCut y el resultado se completa con bordes y morfología.

La referencia real no copia máscaras clásicas, pero depende de cajas clásicas y no dispone de una edición manual independiente documentada. Por ello, las métricas reales se denominan concordancia con una preanotación canónica asistida.

## Random Forest

Cada píxel se representa mediante 19 variables: nueve de color, tres de gradiente, tres multiescala y cuatro estadísticas locales. Por cada hoja de entrenamiento se muestrean 1500 píxeles de objeto y 1500 de fondo.

La primera etapa compara diez configuraciones, incluido un control, mediante cuatro pliegues GroupKFold disjuntos por identidad. El espacio explora número de árboles, profundidad, mínimos de división y hoja terminal, cantidad de variables por división y ponderación de clases.

Los dos mejores candidatos y el control se reconstruyen después sobre las doce hojas completas de validación. Los umbrales 0,35 a 0,65 se ordenan mediante:

IoU medio - 0,005 × error medio absoluto de componentes.

El candidato y su umbral quedan bloqueados en resultados/metricas/rf_idv2_selection_locked.json. La prueba se ejecuta una sola vez después de ese bloqueo. La evaluación real posterior no modifica el modelo.

## SAM 2

SAM 2 Hiera Tiny se utiliza preentrenado y sin ajuste fino. La aplicación obtiene cajas con el localizador clásico y usa la máscara neuronal como entrada de la vectorización.

La validación compara márgenes de caja de 0 %, 3 %, 5 % y 10 % sobre las doce hojas de validación. La selección usa la misma puntuación de IoU y componentes que Random Forest, con desempate por Boundary F1 y margen menor. El escenario con cajas ideales se calcula solo después del bloqueo como diagnóstico del localizador.

## Métricas y tiempos

Las métricas por hoja son IoU, Dice, F1 de contorno con tolerancia de tres píxeles y error absoluto de componentes. La medición temporal operativa usa cuatro capturas reales rectificadas y precargadas, un calentamiento y cinco repeticiones por captura. Excluye lectura desde disco, carga del modelo, rectificación y WCS. Un banco auxiliar aplica el mismo alcance a las doce hojas sintéticas de TEST con tres repeticiones por hoja y mantiene sus resultados separados.

La ruta de SAM 2 incluye localización de cajas, set_image con el codificador, decodificación y postprocesamiento. Random Forest incluye las 19 variables, la probabilidad y su postprocesamiento. La línea clásica incluye su llamada completa.

## Geometría e integración

La evaluación WCS parte de las imágenes sin rectificar y vuelve a detectar los marcadores y la homografía. Se informa la dispersión del origen y del ángulo sin presentarla como calibración metrológica.

La integración exige cuatro marcadores en las trece capturas, WCS válido en diez, rechazo de las tres capturas sin marca y exactamente ocho siluetas exteriores. El modo operativo descarta huecos internos de la máscara porque corresponden a detalles impresos, no a perforaciones deseadas. Cada DXF esperado se reabre y valida. El alcance es estructural y geométrico, no físico.

## Orden de reproducción

1. Importar los 32 activos autorizados con experimentos/datos/prepare_source_assets.py.
2. Generar y auditar el conjunto mediante generate_identity_disjoint_dataset.py y audit_identity_disjoint_dataset.py.
3. Entrenar, seleccionar y probar Random Forest con train_rf_identity_v2.py.
4. Evaluar el Random Forest bloqueado en real con evaluate_rf_identity_v2_real.py.
5. Seleccionar el margen y evaluar SAM 2 con select_prompt_margin_identity_v2.py.
6. Evaluar la línea clásica con el script IDV2 del directorio experimentos/comparacion/.
7. Validar la referencia real, el banco temporal operativo, WCS y la integración por lote.
8. Ejecutar el banco temporal auxiliar IDV2 y regenerar las figuras desde los CSV y JSON.
9. Ejecutar python -m unittest discover software/tests -v.

Los archivos de bloqueo guardan hashes del manifiesto, modelos y checkpoint. Las tablas de la memoria se derivan de las salidas conservadas en resultados/metricas/.
