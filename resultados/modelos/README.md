# Modelos y checkpoints

Los modelos de gran tamaño no se publican en el historial ordinario de Git. La
configuración espera estos nombres dentro de `resultados/modelos/`.

| Archivo | Tamaño (bytes) | SHA-256 | Origen |
|---|---:|---|---|
| `sam2_hiera_tiny.pt` | 155906050 | `65B50056E05BCB13694174F51BB6DA89C894B57B75CCDF0BA6352C597C5D1125` | Checkpoint oficial de SAM 2 Hiera Tiny, Meta |
| `random_forest_legacy_v1.joblib` | 60111985 | `B6141FB7DB8320D9A33155F38BF0EFE1AC6FAE15195462548B1D50DDA5627836` | Control RF v1 generado por el proyecto |
| `random_forest_legacy_v2.joblib` | 233889745 | `4B8D8ED8EE2D7E51C7F93AC50A8E9035AFE1D0BE58462A81E9DB18902E7C3911` | Control RF v2 generado por el proyecto |

SAM 2 se obtiene desde <https://github.com/facebookresearch/sam2>. Deben
respetarse la licencia y las condiciones publicadas por Meta. Los Random Forest
pueden reconstruirse con los scripts de `experimentos/random_forest/`.

Para verificar un archivo en PowerShell:

```powershell
Get-FileHash resultados/modelos/sam2_hiera_tiny.pt -Algorithm SHA256
```

La ausencia o alteración del checkpoint de SAM 2 provoca un error explícito en
la aplicación. No se sustituye su salida por la segmentación clásica.
