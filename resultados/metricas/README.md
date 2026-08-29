# Guía de las métricas del protocolo final

La memoria utiliza como evidencia principal los artefactos del protocolo `asset_identity_v2` y los archivos identificados con `idv2`, `v8` o `final`. Esta guía evita mezclar esos resultados con ensayos exploratorios conservados por trazabilidad.

## Artefactos canónicos

- Calidad y reproducibilidad del conjunto: `dataset_identity_v2_quality.json` y `dataset_identity_v2_reproducibility.json`.
- Evidencia de los umbrales clásicos: `hsv_threshold_evidence.json`.
- Línea clásica sintética: `classical_idv2_test_metrics.csv`, `classical_idv2_test_summary.json` y `classical_idv2_test_runtime_runs.csv`.
- Random Forest IDV2: todos los archivos cuyo nombre comienza por `rf_idv2_`.
- SAM 2 IDV2: todos los archivos cuyo nombre comienza por `sam2_idv2_`.
- Comparación final: `comparison_idv2_final_per_sheet.csv`, `comparison_idv2_final_summary.csv`, `synthetic_real_gap_final.csv` y `final_comparison_validation.json`.
- Referencia real: `real_reference_validation_v8.json` y `classical_real_metrics.csv`.
- Banco temporal principal sobre cuatro capturas reales: `segmentation_runtime_runs_v8.csv`, `segmentation_runtime_summary_v8.csv` y `segmentation_runtime_summary_v8.json`.
- Banco temporal auxiliar sobre prueba sintética: `segmentation_runtime_idv2_final_runs.csv`, `segmentation_runtime_idv2_final_summary.csv`, `segmentation_runtime_idv2_final_summary.json` y `runtime_comparison_idv2_final.csv`.
- Registro e integración: archivos que comienzan por `wcs_`, `contour_` o `integration_batch_` y terminan en `v8`.
- Comprobación del paquete: `protocol_package_validation.json`.

`classical_rf_real_summary.json` es un archivo mixto. Su bloque `classical` sigue siendo una entrada de los generadores finales, pero su bloque de Random Forest pertenece al protocolo anterior y no debe citarse. Los resultados reales vigentes de Random Forest se encuentran en `rf_idv2_real_agreement_metrics.csv` y `rf_idv2_real_agreement_summary.json`.

## Artefactos históricos

Los archivos siguientes pertenecen a iteraciones anteriores. Se conservan para rastrear el desarrollo, pero no respaldan las cifras de la memoria actual:

- `classical_test_metrics.csv`
- `classical_test_summary.json`
- `integration_real20.json`
- `rf_consolidated_model_comparison.csv`
- `rf_consolidated_selection.json`
- `rf_final_selection.json`
- `rf_hyperparameter_search.csv`
- `rf_optimized_configuration.json`
- `rf_selected_real_metrics.csv`
- `rf_selected_test_metrics.csv`
- `rf_test_metrics.csv`
- `rf_v1_control_selection.json`
- `rf_v1_test_metrics.csv`
- `rf_v1_validation_threshold_per_sheet.csv`
- `rf_v1_validation_threshold_summary.csv`
- `rf_validation_model_threshold_per_sheet.csv`
- `rf_validation_model_threshold_summary.csv`
- `rf_validation_thresholds.csv`
- `sam2_final_selection.json`
- `sam2_real_metrics.csv`
- `sam2_real_summary.json`
- `sam2_test_metrics.csv`
- `sam2_validation_prompt_metrics.csv`
- `sam2_validation_prompt_summary.csv`

Ante cualquier discrepancia, deben prevalecer los artefactos canónicos y sus hashes registrados en `final_comparison_validation.json` y `protocol_package_validation.json`.
