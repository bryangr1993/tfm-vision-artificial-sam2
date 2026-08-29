# Dataset sintético `asset_identity_v2`

Este directorio aloja la segunda versión del corpus sintético. Las imágenes pesadas, máscaras binarias, máscaras de instancias y contornos se regeneran localmente y quedan fuera del historial Git. Los 48 metadatos de composición sí se conservan para documentar qué identidad aparece en cada posición.

La partición se realiza sobre los 32 activos fuente antes de componer las láminas:

- entrenamiento: 16 identidades y 24 láminas;
- validación: 8 identidades y 12 láminas;
- prueba: 8 identidades y 12 láminas.

Las familias F1--F4 aparecen en los tres subconjuntos como estratos semánticos. La regla `asset_identity_disjoint` exige que ningún identificador, nombre de archivo o hash de activo cruce entre entrenamiento, validación y prueba.

Para regenerar y auditar:

```powershell
python experimentos/datos/generate_identity_disjoint_dataset.py --overwrite
python experimentos/datos/audit_identity_disjoint_dataset.py
python -m unittest discover -s experimentos/datos -p "test_identity_disjoint_dataset.py" -v
```

El manifiesto canónico de esta versión es `datos/manifiesto/datasets_asset_identity_v2.csv`. La versión anterior se conserva para trazabilidad, pero sus métricas no deben mezclarse con las obtenidas bajo este protocolo.
