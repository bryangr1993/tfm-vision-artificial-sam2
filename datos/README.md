# Datos y disponibilidad

## Corpus sintético vigente

La unidad experimental es una hoja A4 completa con ocho instancias. El corpus `asset_identity_v2` contiene 32 identidades de activo y 48 hojas:

- Entrenamiento: 16 activos y 24 hojas.
- Validación: 8 activos y 12 hojas.
- Prueba: 8 activos y 12 hojas.

La asignación se fija antes de generar las hojas. Los identificadores, nombres y hashes de activo son disjuntos entre particiones. Las familias F1-F4 funcionan como estratos semánticos compartidos y están presentes en los tres subconjuntos. La auditoría también exige disyunción por hash de imagen y máscara.

Los archivos principales son:

- `manifiesto/asset_registry_v2.csv`: identidad, familia, dimensiones, hash y procedencia de los 32 activos.
- `manifiesto/asset_split_v2.csv`: asignación previa de cada activo.
- `manifiesto/dataset_identity_v2_config.json`: semillas, condiciones y parámetros de generación.
- `manifiesto/datasets_asset_identity_v2.csv`: una fila por hoja con rutas, identidades y hashes.
- `sinteticos_identidad_v2/`: imágenes, máscaras binarias, máscaras de instancia y metadatos del corpus.

## Disponibilidad de los activos

Los 32 PNG fuente no se publican ni se presentan como un conjunto abierto. El registro disponible no permite establecer una licencia de redistribución de cada diseño. Su uso queda limitado al entorno académico autorizado y su política se detalla en `fuentes_toppers/README.md`.

Quien disponga legítimamente del paquete puede importarlo mediante:

```powershell
python experimentos/datos/prepare_source_assets.py --source-dir "RUTA_AL_PAQUETE/32_TOPPERS_SIN_FONDO"
```

## Capturas reales y referencia

El dominio real comprende trece adquisiciones de una sola lámina física. Sirve para estudiar sensibilidad a la captura y transferencia externa, no para representar trece diseños independientes.

La referencia canónica real se construyó con asistencia algorítmica a partir de la mediana de las capturas rectificadas, cajas del localizador clásico, GrabCut, Canny y morfología. No es una verdad terreno manual independiente. Las cifras obtenidas frente a ella se denominan concordancia.

## Generación y controles

Desde la raíz del repositorio:

```powershell
python experimentos/datos/generate_identity_disjoint_dataset.py --overwrite
python experimentos/datos/audit_identity_disjoint_dataset.py
python experimentos/datos/test_identity_disjoint_dataset.py
```

La regeneración completa conserva el hash del manifiesto `d850a74911feb592c33c7fc66e686d3093d34a33036742f981d1c2ab232c10dc`. La puerta de calidad registra cero fallos críticos y las tres pruebas automáticas deben finalizar correctamente antes de entrenar o evaluar modelos.

Los resultados se conservan en `resultados/metricas/dataset_identity_v2_quality.json` y `resultados/metricas/dataset_identity_v2_reproducibility.json`. El informe explicativo está en `documentacion/INFORME_CALIDAD_DATOS_IDV2.md`.
