# Informe de calidad del dataset `asset_identity_v2`

**Resultado global:** APROBADO

La unidad del manifiesto es una lámina A4 sintética completa con ocho instancias. El conjunto se usa para seleccionar hiperparámetros del Random Forest y realizar una evaluación sintética bloqueada.

La familia semántica no es disjunta: F1--F4 se mantienen en todos los subconjuntos como estratos. La disyunción se exige sobre la identidad del diseño, el nombre del archivo y su SHA-256.

| Comprobación | Estado | Evidencia |
|---|---:|---|
| `sheet_count_by_split` | OK | `{"train": 24, "val": 12, "test": 12}` |
| `unique_sample_id` | OK | `48` |
| `split_protocol_label` | OK | `["asset_identity_disjoint"]` |
| `asset_id_cross_split` | OK | `{"train__val": [], "train__test": [], "val__test": []}` |
| `asset_hash_cross_split` | OK | `{"train__val": [], "train__test": [], "val__test": []}` |
| `source_filename_cross_split` | OK | `{"train__val": [], "train__test": [], "val__test": []}` |
| `semantic_family_present_in_each_split` | OK | `{"train": ["F1", "F2", "F3", "F4"], "val": ["F1", "F2", "F3", "F4"], "test": ["F1", "F2", "F3", "F4"]}` |
| `semantic_families_intentionally_not_disjoint` | OK | `"Las familias son estratos compartidos; la identidad de activo es la unidad disjunta."` |
| `asset_registry_unique_hash` | OK | `32` |
| `all_paths_exist` | OK | `[]` |
| `dimensions_2100x2970` | OK | `[]` |
| `binary_and_instance_mask_validity` | OK | `[]` |
| `metadata_asset_authorization` | OK | `[]` |
| `image_hash_cross_split` | OK | `{"train__val": [], "train__test": [], "val__test": []}` |
| `mask_hash_cross_split` | OK | `{"train__val": [], "train__test": [], "val__test": []}` |
| `all_allocated_assets_used` | OK | `{"train": ["A01", "A04", "A05", "A08", "A10", "A12", "A13", "A16", "A19", "A21", "A22", "A23", "A25", "A30", "A31", "A32"], "val": ["A02", "A07", "A09", "A14", "A20", "A24", "A27", "A29"], "test": ["A03", "A06", "A11", "A15", "A17", "A18...` |
| `train_cv_groups_asset_disjoint` | OK | `{"G1": ["A08", "A10", "A22", "A31"], "G2": ["A01", "A16", "A23", "A25"], "G3": ["A05", "A12", "A19", "A30"], "G4": ["A04", "A13", "A21", "A32"]}` |
| `condition_coverage` | OK | `{"train": ["C1", "C2", "C3", "C4", "C5", "C6"], "val": ["C1", "C2", "C3", "C4", "C5", "C6"], "test": ["C1", "C2", "C3", "C4", "C5", "C6"]}` |
| `layout_coverage` | OK | `{"train": ["L1", "L2"], "val": ["L1", "L2"], "test": ["L1", "L2"]}` |

## Riesgo residual

La independencia por identidad elimina la contaminación detectada en la versión anterior, pero no convierte 32 diseños en una muestra poblacional amplia. Los resultados describen generalización a ocho identidades sintéticas no vistas y deben contrastarse por separado con las capturas reales.

## Repetición

```powershell
python experimentos/datos/generate_identity_disjoint_dataset.py --overwrite
python experimentos/datos/audit_identity_disjoint_dataset.py
```
