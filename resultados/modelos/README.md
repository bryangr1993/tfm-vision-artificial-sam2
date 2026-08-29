# Modelos y checkpoint

Los binarios se mantienen fuera del historial ordinario de Git. La aplicación y
los experimentos esperan los siguientes archivos en este directorio:

| Archivo | Tamaño (bytes) | SHA-256 | Origen |
|---|---:|---|---|
| `sam2_hiera_tiny.pt` | 155906050 | `65B50056E05BCB13694174F51BB6DA89C894B57B75CCDF0BA6352C597C5D1125` | Checkpoint oficial SAM 2 Hiera Tiny |
| `random_forest_identity_v2_selected.joblib` | 3580636 | `4D2B085DA6CD81D91D83DDB2A451A49C7D54E3EA829E856DF0480CDD4EF06A5A` | Modelo seleccionado en validación |
| `random_forest_identity_v2_control.joblib` | 3564870 | `9BB195DF4A0338BBDB58EAAD1C83B89B7634F3E7EEAF3750E9420924B9CB1122` | Configuración de control reconstruida sobre IDV2 |

SAM 2 se obtiene desde <https://github.com/facebookresearch/sam2> y queda
sujeto a la licencia publicada por Meta. Los dos modelos Random Forest se
regeneran mediante `experimentos/random_forest/train_rf_identity_v2.py`.

Para comprobar un archivo en PowerShell:

```powershell
Get-FileHash resultados/modelos/sam2_hiera_tiny.pt -Algorithm SHA256
```

La ausencia o alteración del checkpoint de SAM 2 provoca un error explícito. La
aplicación no sustituye su salida por la segmentación clásica.
