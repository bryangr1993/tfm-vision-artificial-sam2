# Informe reproducible del protocolo SAM 2, WCS y tiempos

## Resultado principal

El protocolo deja a SAM 2 como método operativo del flujo de software y evita seleccionar decisiones con el conjunto de prueba. El margen de caja del 5 % se eligió exclusivamente sobre las doce láminas de validación IDV2. Una vez bloqueado, se aplicó al conjunto de prueba sin comparar alternativas. Con cajas operativas alcanzó IoU 0,923078, Dice 0,959987 y Boundary F1 0,737595 en las doce láminas de prueba. El error medio en el número de componentes fue cero.

La integración completa también quedó comprobada con las trece capturas reales disponibles. Las trece produjeron ocho siluetas exteriores y ninguna trayectoria de hueco. Las diez capturas con marca WCS generaron un DXF estructuralmente válido. Las tres capturas sin WCS se detuvieron sin exportar. Esta validación confirma la estructura de los archivos mediante lectura de retorno, pero no sustituye una importación en RDWorks ni un corte físico.

## Separación de datos y selección del margen

El conjunto IDV2 contiene 48 láminas sintéticas completas y 32 identidades gráficas. La partición comprende 24 láminas de entrenamiento, 12 de validación y 12 de prueba. Los controles de calidad no encontraron identidades, huellas de archivo ni nombres de archivo fuente compartidos entre particiones.

La optimización del prompt comparó cuatro márgenes con las cajas producidas por el localizador operativo. La regla de selección priorizó la IoU media y penalizó el error en el número de componentes.

| Margen por lado | Láminas de validación | IoU media | Dice medio | Boundary F1 medio | Error medio de componentes |
|---:|---:|---:|---:|---:|---:|
| 0 % | 12 | 0,921400 | 0,959089 | 0,767908 | 0,000 |
| 3 % | 12 | 0,922847 | 0,959873 | 0,768492 | 0,000 |
| **5 %** | **12** | **0,923156** | **0,960040** | 0,766333 | **0,000** |
| 10 % | 12 | 0,922812 | 0,959854 | 0,757871 | 0,000 |

La ventaja del 5 % es pequeña, por lo que no debe interpretarse como una diferencia sustantiva entre configuraciones. Su elección es reproducible porque aplica una regla declarada sobre validación. El conjunto de prueba no participó en esa decisión.

## Evaluación bloqueada de SAM 2

| Partición | Escenario de cajas | N | IoU media | Dice medio | Boundary F1 medio | Error medio de componentes |
|---|---|---:|---:|---:|---:|---:|
| Prueba IDV2 | Operativas | 12 | 0,923078 | 0,959987 | 0,737595 | 0,000 |
| Prueba IDV2 | Ideales | 12 | 0,923176 | 0,960041 | 0,737646 | 0,000 |
| Real | Operativas | 13 capturas | 0,953640 | 0,976253 | 0,720002 | 0,000 |
| Real | Ideales | 13 capturas | 0,953720 | 0,976295 | 0,720568 | 0,000 |

Las cajas ideales son un diagnóstico optimista. Las cajas operativas se obtienen con el mismo localizador utilizado por el producto. La diferencia mínima observada entre ambos escenarios indica que, en estas muestras, la localización operativa no es el principal cuello de botella.

Los resultados reales son concordancia con una preanotación canónica asistida. No representan exactitud frente a una verdad terreno independiente. Además, las trece capturas proceden de una sola lámina física, por lo que describen sensibilidad a la adquisición y no generalización a diseños nuevos.

## Referencia real

Los trece archivos de referencia real son copias byte a byte de una máscara canónica. La máscara se construyó a partir de la mediana de las trece capturas, cajas del localizador clásico, GrabCut, detección de bordes y operaciones morfológicas. Esta procedencia impide describirla como anotación manual independiente.

La inspección reproducible identifica ocho componentes exteriores. La abertura del arcoíris aparece como una concavidad conectada al fondo exterior y no como un hueco topológico cerrado. La evidencia geométrica calculada en su región central es compatible con que esa abertura esté presente. Aun así, la máscara requiere una revisión humana independiente con registro de autor, fecha y cambios antes de elevar su categoría de referencia.

La IoU media entre esta preanotación y las máscaras clásicas de las capturas es 0,978732. Ese valor documenta una dependencia metodológica fuerte y refuerza la necesidad de usar el término concordancia cuando se informan resultados reales.

## Tiempo de segmentación

El banco temporal usa cuatro capturas rectificadas de 2100 × 2970 píxeles. Los tres métodos se ejecutaron en el mismo proceso y equipo, con un calentamiento por método y cinco repeticiones por captura. Esto proporciona veinte mediciones por método. La lectura desde disco, la carga de modelos, la rectificación y la detección WCS quedaron fuera de esta comparación. SAM 2 incluye la localización de cajas, `set_image`, el codificador, la decodificación y el postproceso. Random Forest incluye la extracción de 19 características, `predict_proba` y su postproceso.

| Método | Mediciones | Media (s) | Desviación estándar (s) | Mediana (s) |
|---|---:|---:|---:|---:|
| Visión clásica | 20 | 0,1367 | 0,0198 | 0,1330 |
| SAM 2 operativo | 20 | 2,1736 | 0,2161 | 2,1430 |
| Random Forest IDV2 | 20 | 12,7163 | 1,6652 | 12,1190 |

Todas las mediciones se realizaron en CPU porque CUDA no estaba disponible. El checkpoint de SAM 2 sí estaba disponible y quedó identificado mediante SHA-256. Los tiempos describen este equipo y este tamaño de imagen. No deben generalizarse a una GPU ni a resoluciones distintas sin una nueva medición.

## Repetibilidad WCS y estabilidad de contorno

La detección WCS se recalculó desde las imágenes sin rectificar. Las cuatro marcas de esquina, la homografía y la marca en L se estimaron de nuevo para cada captura. El detector clasificó correctamente las diez capturas con WCS y rechazó las tres capturas sin WCS.

Entre las diez detecciones positivas, la desviación radial media del origen respecto a su centro fue 0,1953 mm y la máxima fue 0,3899 mm. El ángulo medio del eje X fue −0,6429° y su desviación estándar fue 0,0851°. Estas cifras expresan dispersión entre capturas. No son una medida de exactitud metrológica porque no se dispuso de un patrón físico independiente.

La figura `resultados/figuras_protocolo_v8/wcs_detection_stages_v8.png` documenta el procesamiento de la captura `real_20`. Muestra la ROI de 45 × 45 mm, la binarización adaptativa, los segmentos Hough que alimentan la búsqueda geométrica y el origen con sus ejes finales. El resultado procede del detector actual y su archivo complementario conserva los umbrales y las huellas de los módulos utilizados.

La estabilidad de contorno se calculó por separado sobre los 45 pares posibles de las diez capturas con WCS.

| Método | IoU media entre pares | Boundary F1 medio entre pares | Fracción media de primer plano | Capturas con 8 componentes |
|---|---:|---:|---:|---:|
| Visión clásica | 0,964445 | 0,769908 | 0,288623 | 10/10 |
| SAM 2 operativo | 0,963066 | 0,731955 | 0,287290 | 10/10 |
| Random Forest IDV2 | 0,988347 | 0,252860 | 0,993910 | 0/10 |

La IoU entre pares del Random Forest es alta porque repite una predicción degenerada que cubre casi toda la hoja. No constituye evidencia de buen rendimiento. Su concordancia media con la preanotación real es 0,288068 y ninguna captura conserva ocho componentes. El resultado demuestra por qué la repetibilidad y la corrección deben analizarse como propiedades distintas.

## Integración y política de exportación

El lote recorre carga de imagen, detección de cuatro marcadores, rectificación, detección WCS, localización de ocho cajas, segmentación SAM 2, extracción de contornos y exportación DXF.

- Las 13 capturas detectaron cuatro marcadores.
- Las 10 capturas con WCS fueron aceptadas y produjeron DXF.
- Las 3 capturas sin WCS fueron rechazadas y no produjeron DXF.
- Las 13 capturas produjeron 8 contornos exteriores, 0 huecos y 8 trayectorias totales.
- Los 10 DXF se reabrieron correctamente con `ezdxf`.
- Cada DXF contiene 8 polilíneas cerradas en la capa de siluetas exteriores, coordenadas finitas y límites plausibles para el contexto A4.

La exclusión de huecos es intencional. Ojos, ruedas y otros detalles internos pertenecen a la impresión y no deben convertirse en trayectorias del láser. La concavidad abierta del arcoíris permanece en el perímetro exterior. El soporte de huecos existe como opción explícita, pero está desactivado para este producto.

## Límites que deben acompañar los resultados

- La referencia real sigue siendo una preanotación asistida pendiente de revisión manual independiente.
- Las trece capturas reales corresponden a una única lámina física.
- El banco temporal se ejecutó en CPU y sobre cuatro capturas representativas.
- La dispersión WCS no equivale a exactitud metrológica absoluta.
- La validación DXF es estructural. Falta comprobar importación en RDWorks y corte físico.

## Formulaciones defendibles para la memoria

1. «El margen del 5 % se seleccionó exclusivamente en las doce láminas de validación IDV2 entre cuatro candidatos. El conjunto de prueba se evaluó una sola vez después de bloquear esta decisión».
2. «Con cajas operativas, SAM 2 obtuvo una IoU media de 0,9231 en las doce láminas de prueba y conservó ocho componentes en todos los casos».
3. «En datos reales se informa concordancia con una preanotación canónica asistida, no exactitud frente a una verdad terreno independiente».
4. «La dispersión radial media del origen WCS fue 0,195 mm entre diez capturas de una misma lámina. Este valor describe repetibilidad geométrica y no exactitud metrológica absoluta».
5. «La integración por lote generó diez DXF estructuralmente válidos en las capturas con WCS y bloqueó la exportación en las tres capturas sin WCS».
6. «La exportación utilizó ocho siluetas exteriores y excluyó detalles impresos internos. La validación por lectura de retorno no sustituye una prueba física de corte».
7. «El tiempo medio de SAM 2 fue 2,174 s en CPU e incluyó la localización de prompts y el codificador de imagen».
8. «La alta estabilidad entre capturas del Random Forest no implicó corrección, ya que su máscara ocupó en promedio el 99,4 % de la hoja y no recuperó las ocho componentes».

## Archivos de soporte

- Selección de margen y escenarios: `resultados/metricas/sam2_idv2_prompt_selection.json`
- Tabla de validación: `resultados/metricas/sam2_idv2_margin_validation_summary.csv`
- Evaluación bloqueada: `resultados/metricas/sam2_idv2_locked_scenario_summary.csv`
- Banco temporal: `resultados/metricas/segmentation_runtime_summary_v8.json`
- Repetibilidad WCS y estabilidad: `resultados/metricas/wcs_contour_repeatability_summary_v8.json`
- Etapas del detector WCS: `resultados/metricas/wcs_detection_stages_v8.json`
- Integración: `resultados/metricas/integration_batch_summary_v8.json`
- Procedencia de la referencia: `resultados/metricas/real_reference_validation_v8.json`
- Lista de comprobación humana: `resultados/referencia_real_validacion_v8/manual_review_checklist.md`
- Figuras técnicas: `resultados/figuras_protocolo_v8/`
- Verificación y huellas: `resultados/metricas/protocol_package_validation.json`

Los comandos para regenerar el paquete están documentados en `experimentos/PROTOCOLO_SAM2_WCS.md`.
