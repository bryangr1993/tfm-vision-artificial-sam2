# Validación técnica del entregable

Fecha de cierre: 20 de agosto de 2026.

## Memoria compilada

- Entregable: `output/pdf/TFM_Bryan_Guananga.pdf`.
- Páginas: 65.
- Tamaño: 4 253 484 bytes.
- SHA-256: `2BC6CFDE1075F2405DC5465A99522D2ECFC5B672CB76A654DF71857F6B7CC5F2`.
- Compilación final: 0 cajas desbordadas, 0 referencias indefinidas y 0 advertencias de LaTeX.
- Inspección visual: portada, índices, 17 figuras incorporadas, tablas, capítulos, Anexo A y referencias revisados sobre páginas renderizadas.
- Auditoría editorial: 0 puntos y coma en las diez fuentes que forman la memoria, 5 palabras clave en español, 5 en inglés, 0 referencias a arXiv y 0 claves bibliográficas ausentes.

## Datos y protocolo

- Manifiesto canónico: 61 filas, con 48 hojas sintéticas y 13 capturas reales.
- Partición sintética: 36 hojas de entrenamiento, 6 de validación y 6 de prueba.
- Controles de calidad: 0 rutas ausentes, 0 dimensiones incompatibles, 0 máscaras no binarias y 0 duplicados.
- La referencia real canónica contiene ocho regiones y no utiliza como etiquetas las salidas comparadas de visión clásica, Random Forest o SAM 2.
- La selección de Random Forest y de prompts de SAM 2 se realizó con validación. El conjunto de prueba no intervino en esas decisiones.

## Resultados consolidados

| Método | IoU sintético | IoU real | Error de componentes real |
|---|---:|---:|---:|
| Visión clásica | 0,9016 | 0,9787 | 0,00 |
| Random Forest seleccionado | 0,9902 | 0,2877 | 7,00 |
| SAM 2 Hiera Tiny | 0,9153 | 0,9536 | 0,00 |

La memoria declara de forma expresa que la visión clásica obtuvo el mejor IoU real. SAM 2 se mantiene como motor de segmentación de la aplicación porque generó ocho instancias estables y una salida vectorizable en todas las capturas evaluadas.

## Software

- Pruebas unitarias y de integración: 4 de 4 superadas.
- Compilación de módulos Python: superada.
- Regeneración del manifiesto: 61 filas, 0 rutas ausentes, 0 incompatibilidades de dimensión, 0 máscaras no binarias y 0 duplicados.
- La aplicación falla de forma explícita si no puede inicializar SAM 2 y no sustituye su máscara silenciosamente por la clásica.
- Prueba científica registrada sobre la captura 20: WCS válido, 8 prompts, 8 componentes, 8 contornos y DXF de 65 399 bytes.
- Reproducción independiente en un entorno limpio con Python 3.12 y PyTorch 2.13.0: 8 contornos y DXF válido de 65 397 bytes. La diferencia de dos bytes corresponde a la serialización del archivo y no altera el número de trayectorias.
- Exportación directa mediante la lógica de la GUI: 8 polilíneas cerradas en `TOPPERS_CUT` y 4 entidades auxiliares en `WCS_REF`.
- La interfaz validada muestra el modo `SAM 2 + WCS` y habilita la exportación solo con referencia geométrica válida.

## Publicación

- Repositorio público: <https://github.com/bryangr1993/tfm-vision-artificial-sam2>.
- Los modelos pesados se excluyen del historial ordinario. Sus nombres, tamaños, orígenes y checksums se documentan en `resultados/modelos/README.md` y en el Anexo A.
