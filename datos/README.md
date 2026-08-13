# Datos y disponibilidad

La unidad experimental es una lámina completa. Las particiones de entrenamiento, validación y prueba se definen por lámina para evitar que los píxeles de una misma imagen aparezcan en más de un subconjunto.

El manifiesto `manifiesto/datasets.csv` contiene 61 filas: 48 hojas sintéticas y 13 capturas reales rectificadas. La partición sintética comprende 36 hojas de entrenamiento, 6 de validación y 6 de prueba. Las 13 capturas reales corresponden a distintas adquisiciones de una misma hoja física y se utilizan exclusivamente como evaluación externa de transferencia.

`anotaciones/real_ground_truth/` contiene la referencia binaria alineada con cada captura. `anotaciones/referencia_real_canonica/` documenta la construcción asistida, la imagen mediana, el control visual y los metadatos de la referencia. Las salidas de los métodos comparados no se emplearon como etiquetas de píxel.

Las imágenes crudas, las rectificadas y las hojas sintéticas de gran tamaño no se incluyen en el historial público ordinario. Sus rutas relativas, dimensiones y checksums quedan registrados en el manifiesto. Para repetir los experimentos con los mismos datos se debe colocar el paquete de datos autorizado bajo las rutas declaradas y ejecutar:

```powershell
python experimentos/construir_manifiesto.py
```

El informe regenerado en `documentacion/INFORME_CALIDAD_DATOS.md` debe mostrar cero rutas ausentes, dimensiones incompatibles, máscaras no binarias y duplicados antes de ejecutar las evaluaciones.
