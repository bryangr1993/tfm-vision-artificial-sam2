# Plan maestro de corrección del TFM — versión 7

## 1. Propósito y criterio rector

Este documento convierte las observaciones de María y la auditoría del paquete consolidado en un programa de trabajo único. El objetivo no es maquillar la versión 6, sino conservar lo que ya funciona y corregir con evidencia aquello que debilita el carácter de TFM de un Máster en Inteligencia Artificial.

La decisión central es la siguiente:

> La aplicación final utilizará SAM 2 para generar la máscara de segmentación que se entrega al módulo de extracción de contornos y a la generación del archivo DXF.

La visión clásica no desaparece. Se mantiene para detectar marcadores, estimar la homografía, rectificar la imagen, localizar aproximadamente los objetos y construir los prompts que recibe SAM 2. También se conserva como línea base experimental. El resultado será, por tanto, una arquitectura híbrida con una salida operativa basada en IA.

Que el método clásico alcance mejores métricas no invalidará el trabajo. Si ocurre después de aplicar una optimización razonable y una evaluación justa, se documentará como resultado académico. Lo que no se mantendrá es una GUI cuya segmentación final dependa exclusivamente del método clásico.

## 2. Elementos que se conservarán

- La versión 6 enviada a María quedará congelada como referencia. No se editará su archivo LaTeX ni se reemplazarán sus resultados.
- Se reutilizará la mayor parte posible de la redacción existente. El 4 % obtenido en Turnitin confirma que la voz del documento funciona.
- Se conservarán las partes correctas del procesamiento geométrico: orientación, detección de marcadores, homografía, rectificación, extracción de contornos, simplificación y exportación DXF.
- Se mantendrán Random Forest, SAM 2 y la visión clásica como métodos comparados.
- Se reaprovecharán las imágenes, tablas, scripts y resultados que tengan procedencia clara y sean metodológicamente válidos.
- Las modificaciones de estilo serán selectivas. No se reescribirá un párrafo correcto solo para cambiar su forma.

## 3. Diagnóstico verificado

### 3.1 Memoria

- La versión 6 dedica más protagonismo operativo a la visión clásica que a la IA.
- La estructura actual coloca Requisitos antes de Objetivos y no presenta Metodología como capítulo independiente.
- El estado del arte explica con poco detalle la arquitectura y el funcionamiento de los modelos de IA.
- Faltan representaciones visuales de Random Forest, SAM 2, calibración, homografía y arquitectura completa del sistema.
- Varias figuras son pequeñas, algunas agrupan demasiada información y no todas están citadas explícitamente en el texto.
- La comparación bibliográfica en forma de tabla debe sustituirse por una síntesis argumentada y visual.
- Parte de los resultados aparece dentro del capítulo de desarrollo del método.
- Hay tablas redundantes y detalles internos, como nombres de archivos de modelos, que no aportan a la memoria.
- Se localizaron 40 usos de punto y coma. Cada uno se revisará y solo se conservará cuando sea realmente necesario.
- La bibliografía contiene referencias de arXiv que deben sustituirse por publicaciones revisadas por pares cuando exista una versión formal.
- Las conclusiones presentan el método clásico como núcleo operativo, algo incompatible con la orientación solicitada por María.

### 3.2 Experimentos

- Random Forest utiliza parámetros fijos. El barrido del umbral de decisión no equivale a una optimización formal de hiperparámetros.
- La búsqueda de la estrategia de prompts de SAM 2 promedia las 48 láminas sintéticas. Esto usa datos de prueba durante la selección de la estrategia.
- La cifra sintética de SAM 2 incluida en la versión 6 corresponde al promedio de esas 48 láminas, no a una evaluación final aislada sobre el conjunto de prueba.
- La evaluación real usa como referencia una máscara producida por el método clásico. Por eso mide concordancia con la línea base y no exactitud frente a una anotación independiente.
- Las cajas que sirven como prompts para SAM 2 también proceden del método clásico. Este diseño es válido para una arquitectura híbrida, pero obliga a evitar cualquier afirmación de independencia entre ambos métodos.
- Los tiempos incluidos en algunas comparaciones no proceden de un protocolo unificado. Se mezclan valores constantes y mediciones de alcances diferentes.
- Los experimentos tienen rutas absolutas ligadas al equipo original. El paquete consolidado no es reproducible sin corregirlas.

### 3.3 Software

- La GUI actual llama directamente a la segmentación clásica.
- La máscara clásica alimenta la extracción de contornos y la generación del DXF.
- El módulo de extracción de contornos ya acepta una máscara binaria genérica. Esto permite integrar SAM 2 sin rehacer la etapa geométrica.
- No hay una interfaz común entre segmentadores ni pruebas automatizadas.
- El archivo de dependencias omite varias bibliotecas necesarias para los experimentos de IA.
- Los modelos y el checkpoint superan o se aproximan a los límites habituales de GitHub. Deben distribuirse mediante descarga, versión de lanzamiento o almacenamiento externo con checksum.

## 4. Organización de trabajo propuesta

El paquete `TFM_CONSOLIDADO_CHATGPT` quedará como archivo histórico de la versión 6. La versión activa se construirá en una carpeta independiente con una sola copia canónica de cada elemento:

```text
TFM_V7/
├── memoria/
│   ├── plantilla_latex_7.tex
│   ├── bibliografia.bib
│   └── figuras/
├── software/
│   ├── src/
│   │   ├── pipeline.py
│   │   ├── gui.py
│   │   ├── geometry/
│   │   └── segmenters/
│   ├── tests/
│   └── config/
├── datos/
│   ├── manifiesto/
│   ├── reales/
│   ├── sinteticos/
│   └── anotaciones/
├── experimentos/
│   ├── random_forest/
│   ├── sam2/
│   └── comparacion/
├── resultados/
│   ├── metricas/
│   ├── figuras/
│   └── modelos/
├── documentacion/
│   ├── matriz_observaciones_maria.md
│   ├── trazabilidad_resultados.md
│   └── protocolo_experimental.md
├── requirements-core.txt
├── requirements-ai.txt
└── README.md
```

No se duplicará innecesariamente el paquete de 1,5 GB. Los datos pesados tendrán una ubicación canónica y un manifiesto con identificadores, partición, procedencia y checksum.

## 5. Fases de ejecución

### Fase 0. Congelación y trazabilidad de la versión 6

**Acciones**

1. Registrar el hash del archivo LaTeX enviado a María y de los resultados principales.
2. Crear la estructura `TFM_V7` sin modificar la versión entregada.
3. Copiar únicamente los archivos canónicos que se continuarán editando.
4. Crear un repositorio Git local para que cada cambio sea reversible.
5. Elaborar un inventario que distinga fuentes, datos, modelos, resultados y material descartado.

**Criterio de cierre**

La versión 6 puede reconstruirse y compararse con la versión 7. Ningún archivo histórico ha sido sobrescrito.

### Fase 1. Matriz de las 106 observaciones de María

**Acciones**

1. Convertir los 106 comentarios del PDF en una tabla de seguimiento.
2. Agruparlos en estructura, redacción, IA, metodología, figuras, tablas, resultados, conclusiones, bibliografía y formato.
3. Asignar a cada comentario una acción, un archivo, una evidencia de corrección y un estado.
4. Marcar los comentarios repetidos sin perder su trazabilidad individual.

**Criterio de cierre**

Cada comentario tiene una respuesta verificable. La revisión final no depende de recordar anotaciones dispersas en el PDF.

### Fase 2. Reconstrucción reproducible de datos y particiones

**Acciones**

1. Crear un manifiesto único para las 48 láminas sintéticas y las 13 imágenes reales rectificadas utilizadas en la evaluación.
2. Fijar las particiones de entrenamiento, validación y prueba por lámina completa.
3. Prohibir la mezcla de píxeles de una misma lámina entre particiones.
4. Sustituir rutas absolutas por rutas relativas y configuración externa.
5. Registrar la semilla aleatoria y las versiones de las dependencias.
6. Conservar intacto el conjunto de prueba hasta cerrar las decisiones de modelo y prompts.

**Criterio de cierre**

Un script puede reconstruir el inventario y las particiones desde una instalación limpia.

### Fase 3. Verdad terreno independiente para imágenes reales

**Acciones**

1. Generar máscaras iniciales asistidas para las 13 imágenes reales rectificadas.
2. Corregir manualmente bordes, huecos, uniones y falsos componentes sin usar la máscara clásica como referencia final automática.
3. Guardar cada anotación como máscara binaria independiente y documentar el criterio de inclusión.
4. Realizar una segunda revisión visual de todas las máscaras. Bryan validará los casos dudosos por su conocimiento del proceso productivo.
5. Registrar qué zonas no pueden resolverse con certeza y excluirlas o tratarlas de forma explícita.

**Criterio de cierre**

Los tres métodos se comparan contra la misma referencia independiente. La métrica real deja de ser una medida circular de concordancia con el método clásico.

### Fase 4. Optimización formal de Random Forest

**Acciones**

1. Encapsular la extracción de características y el muestreo de píxeles en un pipeline reproducible.
2. Aplicar validación cruzada agrupada por lámina con `GroupKFold`, o una partición entrenamiento-validación agrupada si el coste computacional lo exige.
3. Explorar de forma aleatoria y controlada `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight` y tamaño de muestra.
4. Optimizar el umbral de probabilidad después de seleccionar el modelo, usando solo validación.
5. Elegir la configuración con una función objetivo que priorice IoU y penalice componentes espurios y tiempo excesivo.
6. Evaluar una sola vez la configuración seleccionada sobre prueba.
7. Guardar todas las ejecuciones, semillas, tiempos y configuraciones en CSV o JSON.

**Criterio de cierre**

La memoria puede explicar qué se optimizó, cómo se evitó la fuga de información, qué configuración ganó y cuánto aportó frente a la configuración inicial.

### Fase 5. Optimización y evaluación justa de SAM 2

**Acciones**

1. Documentar la variante exacta de SAM 2, su checkpoint, arquitectura, preentrenamiento y motivo de selección.
2. Justificar por qué no se entrena desde cero y diferenciar inferencia con prompts de entrenamiento supervisado.
3. Convertir margen de caja, combinación de cajas y puntos, número de puntos, prompts negativos y preprocesamiento en parámetros explícitos.
4. Seleccionar la estrategia de prompts solo con el conjunto de validación.
5. Bloquear esa estrategia y evaluarla una única vez en prueba.
6. Medir por separado el efecto de la caja clásica y el comportamiento de SAM 2. Se incluirá una pequeña ablación con cajas de verdad terreno o perturbadas cuando sea viable.
7. Si la optimización de prompts sigue siendo insuficiente, evaluar como extensión condicionada un ajuste ligero del decodificador o una técnica eficiente en parámetros. Esta etapa solo se ejecutará si el hardware, el tiempo y el tamaño de los datos permiten una comparación fiable.

**Criterio de cierre**

No se utilizan láminas de prueba para escoger prompts. La cifra final de SAM 2 procede de una partición retenida y su dependencia del localizador clásico está declarada.

### Fase 6. Comparación experimental común

**Acciones**

1. Evaluar visión clásica, Random Forest y SAM 2 sobre las mismas referencias sintéticas y reales siempre que la entrada sea comparable.
2. Calcular Dice e IoU por imagen y reportar media, desviación y distribución.
3. Añadir una métrica de borde, como Boundary F1, porque el producto final depende de la geometría del contorno.
4. Medir componentes espurios, contornos válidos, área, tiempo y validez del DXF.
5. Usar un banco de tiempos único con calentamiento, repeticiones, mismo equipo y alcance claramente definido.
6. Aplicar intervalos de confianza mediante bootstrap por imagen y una comparación pareada cuando el número de casos lo permita.
7. Separar resultados sintéticos, resultados reales y validación del flujo de corte.
8. Si es posible, ejecutar una prueba física con un DXF generado desde la máscara de SAM 2 y documentar el resultado.

**Criterio de cierre**

La tabla consolidada se genera directamente desde archivos de resultados. Ningún tiempo o promedio se introduce manualmente en el texto.

### Fase 7. Integración de IA en la aplicación

**Arquitectura funcional**

```text
Imagen capturada
      ↓
Orientación y marcadores
      ↓
Homografía y rectificación
      ↓
Localización clásica aproximada
      ↓
Prompts espaciales
      ↓
SAM 2 → máscara de IA
      ↓
Contornos, simplificación y validación
      ↓
DXF para corte
```

**Acciones**

1. Crear una interfaz común para los segmentadores.
2. Implementar un adaptador de SAM 2 con carga diferida del modelo y configuración reproducible.
3. Modificar el pipeline para que la máscara de SAM 2 alimente obligatoriamente la extracción de contornos en el flujo final.
4. Mantener la visión clásica como localizador de prompts y como modo de comparación diagnóstica.
5. Mostrar en la GUI el estado de carga, el progreso, la máscara de IA, los prompts y los errores recuperables.
6. Identificar claramente el resultado como “Segmentación IA — SAM 2”.
7. Añadir pruebas unitarias para contratos de entrada y salida, máscaras vacías, dimensiones, contornos y exportación.
8. Añadir una prueba de integración que demuestre que el DXF procede de la máscara devuelta por SAM 2 y no de la máscara clásica.

**Criterio de cierre**

El flujo que se presenta y valida en la memoria genera el DXF a partir de la segmentación de SAM 2. Esto se comprueba con una prueba automatizada y con una captura de la GUI.

### Fase 8. Reestructuración de la memoria

La versión 7 seguirá esta organización:

1. **Introducción**
   - Contexto global de la IA aplicada a visión por computador.
   - Problema industrial y breve explicación de Innokey.
   - Motivación, contribución y estructura de la memoria.
   - Una o dos figuras introductorias, sin saturar el capítulo con detalles técnicos.
2. **Contexto y estado del arte**
   - Flujo CAD/CAM, calibración, homografía, segmentación y vectorización.
   - Métodos de aprendizaje automático y aprendizaje profundo relevantes.
   - Random Forest, redes de segmentación y modelos fundacionales.
   - Arquitectura y funcionamiento de SAM 2.
   - Motivo de la selección de Random Forest y SAM 2 frente a otras alternativas.
3. **Objetivos**
   - Objetivo general y objetivos específicos medibles.
   - Relación de cada objetivo con su evidencia de cumplimiento.
4. **Requisitos**
   - Requisitos funcionales y no funcionales explicados y verificables.
5. **Metodología**
   - Diseño de la investigación.
   - Datos, anotación, particiones y prevención de fugas.
   - Modelos, parámetros, hiperparámetros y optimización.
   - Métricas y protocolo experimental.
6. **Descripción de la herramienta software desarrollada**
   - Arquitectura híbrida.
   - Implementación de SAM 2 en el flujo.
   - GUI, vectorización, DXF y manejo de errores.
7. **Evaluación y discusión**
   - Resultados sintéticos y reales.
   - Comparación justa de métodos.
   - Ablaciones, tiempos, limitaciones y repercusión productiva.
8. **Conclusiones y trabajo futuro**
   - Cumplimiento explícito del objetivo general y de cada objetivo específico.
   - Papel central de la IA y explicación honesta de sus resultados.
   - Limitaciones y mejoras futuras.
9. **Apéndice A. Reproducibilidad y repositorio**
   - Estructura, instalación, modelos, datos de muestra y comandos de reproducción.

La mayor parte de la prosa válida se moverá a su capítulo correcto. Solo se reescribirán las afirmaciones que cambien por los nuevos experimentos o por la integración de SAM 2.

### Fase 9. Sistema visual y tablas

**Figuras nuevas o reconstruidas**

1. Problema industrial y flujo de registro, segmentación, vectorización y corte.
2. Relación entre CAD y CAM en el caso de uso.
3. Calibración, homografía y rectificación de perspectiva.
4. Funcionamiento de un conjunto Random Forest aplicado a píxeles.
5. Panorama de familias de segmentación consideradas.
6. Arquitectura de SAM 2 con codificador, memoria, prompts y decodificador de máscara.
7. Diseño de particiones y prevención de fuga de información.
8. Flujo de optimización de hiperparámetros.
9. Arquitectura híbrida definitiva del software.
10. Comparaciones cualitativas legibles y capturas de la GUI final.

Se priorizarán diagramas vectoriales hechos en TikZ, SVG o código reproducible. Las capturas y fotografías permanecerán como imágenes ráster de alta resolución. No se generarán diagramas con texto deformado mediante modelos de imagen.

**Reglas editoriales**

- Cada figura y tabla se menciona antes o inmediatamente después de aparecer.
- Cada figura tiene un pie explicativo de cuatro o cinco oraciones cuando el contenido lo justifique.
- Toda subfigura se explica de forma individual.
- Las imágenes compuestas se dividen cuando la legibilidad lo requiere.
- Las tablas no repiten información que puede expresarse con una frase.
- Los índices de figuras y tablas seguirán el formato solicitado por María.

### Fase 10. Bibliografía y redacción

**Acciones**

1. Sustituir preprints por artículos o actas revisadas por pares cuando exista la publicación formal.
2. Usar fuentes originales para arquitecturas, algoritmos y métricas.
3. Eliminar referencias no utilizadas y comprobar que toda cita respalda la afirmación asociada.
4. Expandir cada sigla inglesa en su primera aparición.
5. Revisar comillas, comas, mayúsculas, términos en inglés y traducciones técnicas.
6. Reemplazar “grilla” por la forma técnica adecuada al contexto.
7. Eliminar frases informales y detalles internos que no pertenecen a una memoria académica.
8. Revisar uno por uno los puntos y coma. Se preferirán puntos o comas cuando la oración resulte más natural.
9. Mantener cinco palabras clave: Visión artificial, Segmentación, Vectorización, Random Forest y SAM.
10. Actualizar resumen y abstract después de cerrar los resultados, no antes.

**Criterio de estilo**

La redacción final conservará el ritmo y la voz de la versión 6. No se realizará una “humanización” automática general. Si un apartado nuevo queda demasiado uniforme, se revisará de forma manual y, solo como control adicional, podrá compararse con una versión refinada en Antigravity sin copiar cambios de manera ciega.

### Fase 11. Repositorio abierto y reproducibilidad

**Acciones**

1. Preparar un README con instalación, datos de ejemplo y comandos de entrenamiento, evaluación y ejecución.
2. Separar dependencias centrales y dependencias de IA.
3. Añadir scripts de descarga de checkpoints con checksum.
4. Evitar subir directamente archivos de modelos que excedan los límites de GitHub.
5. Incluir una muestra redistribuible de datos y explicar cómo obtener el conjunto completo.
6. Publicar el repositorio cuando el contenido esté saneado y Bryan autorice la acción externa.
7. Insertar el enlace permanente en el Apéndice A.

**Criterio de cierre**

Una instalación limpia puede ejecutar una demostración, reconstruir las tablas principales y comprobar la procedencia de cada resultado.

### Fase 12. Control final y entrega a María

**Acciones**

1. Compilar la memoria desde cero y resolver todas las referencias, citas y advertencias relevantes.
2. Renderizar el PDF completo y revisar visualmente cada página.
3. Comprobar tamaño y legibilidad de figuras, tablas, pies y ecuaciones.
4. Ejecutar la matriz de 106 comentarios y adjuntar evidencia de cierre.
5. Ejecutar pruebas del software y guardar un informe de resultados.
6. Verificar que todas las métricas del texto coincidan con los CSV generados.
7. Revisar resumen, abstract, conclusiones e índices al final.
8. Preparar una nota breve para María con los cambios principales y la respuesta a sus observaciones.

## 6. Resultados que dejarán de presentarse de la misma forma

- La concordancia de SAM 2 con la máscara clásica en imágenes reales no se presentará como exactitud real. Puede conservarse como análisis secundario si queda claramente etiquetada.
- El promedio de las 48 láminas usado para elegir prompts no será la cifra final de prueba.
- Los tiempos no reproducibles o introducidos como constantes se sustituirán por mediciones comunes.
- La visión clásica dejará de aparecer como motor final de la aplicación.
- Random Forest no se describirá como modelo optimizado hasta completar la búsqueda formal.
- No se afirmará que el sistema es completamente de IA. Se explicará su naturaleza híbrida y la función concreta de cada componente.

## 7. Criterios de aceptación de la versión 7

La versión se considerará lista para enviar cuando se cumpla todo lo siguiente:

- [ ] La versión 6 permanece intacta y su hash está registrado.
- [ ] Las 106 observaciones de María tienen evidencia de cierre.
- [ ] La estructura de capítulos sigue el orden acordado.
- [ ] El estado del arte concede prioridad a la IA y explica las arquitecturas seleccionadas.
- [ ] Existe una optimización formal y reproducible de hiperparámetros de Random Forest.
- [ ] La estrategia de prompts de SAM 2 se selecciona sin tocar el conjunto de prueba.
- [ ] Las 13 imágenes reales tienen verdad terreno independiente revisada.
- [ ] Visión clásica, Random Forest y SAM 2 se comparan mediante un protocolo común.
- [ ] Las tablas proceden de resultados calculados y no de cifras manuales.
- [ ] La máscara de SAM 2 alimenta contornos y DXF en la GUI final.
- [ ] Una prueba de integración demuestra la procedencia de la máscara.
- [ ] Las figuras son legibles, están citadas y explican todos sus elementos.
- [ ] La bibliografía prioriza fuentes originales revisadas por pares.
- [ ] No quedan rutas absolutas, referencias rotas ni marcadores pendientes.
- [ ] El PDF ha sido revisado página por página.
- [ ] El repositorio está listo para publicación y el Apéndice A lo documenta.
- [ ] Las conclusiones responden de forma explícita al objetivo general y a los objetivos específicos.

## 8. Intervenciones que requerirán a Bryan

La mayor parte del trabajo puede ejecutarse de forma autónoma. Se solicitará apoyo solo en tres puntos:

1. Validar los casos dudosos de las 13 máscaras reales, porque requieren conocimiento del objeto físico y del criterio de corte.
2. Realizar, si es viable, una prueba de corte con un DXF generado desde SAM 2.
3. Autorizar la publicación final del repositorio en GitHub y facilitar cualquier ejemplo de TFM que María haya adjuntado y que no esté ya en el paquete.

## 9. Orden inmediato de ejecución

1. Crear `TFM_V7` y congelar formalmente la versión 6.
2. Construir la matriz de comentarios.
3. Normalizar datos, rutas y particiones.
4. Crear la verdad terreno real.
5. Rehacer los experimentos de Random Forest y SAM 2.
6. Integrar SAM 2 en la GUI y verificar el DXF.
7. Generar tablas y figuras desde resultados definitivos.
8. Reestructurar y actualizar la memoria.
9. Preparar repositorio, apéndice y control final.

Este orden evita redactar conclusiones o fijar cifras antes de disponer de resultados válidos. También permite conservar la prosa existente hasta saber exactamente qué afirmaciones deben cambiar.
