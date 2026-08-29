# Activos fuente de toppers

Este directorio contiene, únicamente en el entorno local autorizado, los 32 PNG con canal alfa usados para construir el conjunto sintético con separación por identidad. Los archivos se importan en `alpha/` mediante:

```powershell
python experimentos/datos/prepare_source_assets.py --source-dir "RUTA_AL_PAQUETE/32_TOPPERS_SIN_FONDO"
```

`datos/manifiesto/asset_registry_v2.csv` conserva nombre, familia, dimensiones, hash SHA-256 y estado de procedencia. El registro permite comprobar que se utilizó exactamente el mismo banco sin publicar los archivos pesados.

## Procedencia y uso permitido

Los archivos proceden de un banco de activos aportado por el autor. No existe un registro primario suficiente para establecer cómo se creó cada transparencia, por lo que el manifiesto identifica el método de creación como **no documentado**. Esta limitación se mantiene separada de la integridad técnica, que sí puede comprobarse mediante dimensiones, canal alfa y hashes.

No se presupone ni se concede una licencia de redistribución de los diseños representados. Los PNG se mantienen fuera del historial Git y su uso queda limitado al entorno académico autorizado del proyecto. Antes de publicar el corpus o entregarlo a terceros debe confirmarse por escrito la titularidad o la licencia de cada diseño.

Esta limitación no impide reproducir localmente los experimentos cuando el usuario dispone legítimamente del paquete de activos. Sí impide presentar los 32 PNG como un conjunto de datos público o con licencia abierta.
