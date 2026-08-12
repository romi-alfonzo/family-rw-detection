# Qué falta escribir en la tesis

_Actualizado el 2026-08-12. Inventario de todo lo que ya está medido pero todavía no
redactado, más las correcciones que cada capítulo necesita._

Los resultados están en `4_resultados/` y se regeneran con el código de `2_codigo/`.
**Ninguna cifra se transcribe a mano.**

---

## A. Resultados obtenidos que NO están todavía en el documento

### A1. Hiperparámetros de las características estadísticas → §4.3.1

Cierra la única limitación de optimización declarada en la tesis.

| Configuración | Exactitud | macro-F1 | Cómputo |
|---|---|---|---|
| 19 características + Random Forest | 0,601 | 0,584 | 11 min |
| 19 características + HistGradientBoosting | **0,603** | 0,595 | 3 min |
| 275 características + Random Forest | 0,600 | 0,589 | **2,5 h** |
| 275 características + HistGradientBoosting | 0,599 | 0,592 | 35 min |

Sin ajustar: 0,603. Ajustado: 0,603. **Diferencia −0,000.**

**Qué escribir:** el resultado es indiferente tanto a los hiperparámetros como al número de
características; las cuatro configuraciones caen entre 0,599 y 0,603. Señalar que 275
características con 2,5 horas de cómputo rinden igual que 19 en 3 minutos. Reemplazar el
párrafo de §4.3.1 que declara la falta de optimización como limitación.

### A2. Generalización a tipos de documento nunca vistos → nueva subsección en §4.5

Es la validación que confirma el resultado principal del Experimento 2c.

| Tipo excluido del entrenamiento | Exactitud | macro-F1 |
|---|---|---|
| doc | 0,889 | 0,881 |
| docx | 0,887 | 0,877 |
| xlsx | 0,884 | 0,872 |
| jpg | 0,874 | 0,811 |
| pptx | 0,867 | 0,864 |
| pdf | 0,863 | 0,836 |
| xls | 0,889 | 0,882 |
| **Promedio (7 tipos)** | **0,879** | **0,861** |

Referencia con el tipo presente en entrenamiento: 0,910. **Caída de solo −0,031.**

**Qué escribir:** en NapierOne todas las familias cifraron el mismo conjunto base de
documentos, por lo que cabía la sospecha de que el clasificador aprendiera rasgos del
documento original y no del ransomware. La validación *dejar-un-tipo-fuera* lo descarta: el
rendimiento se sostiene sobre tipos nunca vistos, incluido `jpg`, cuya estructura interna es
completamente distinta a la de un documento de Office. Metodología: entrenar con todos los
tipos salvo uno, evaluar sobre ese tipo; las familias que no conservan el nombre original
participan del entrenamiento pero no de la prueba.

### A3. Dónde reside la información dentro del archivo → nueva subsección en §4.5

- Importancia acumulada: **cabecera 0,204 · cola 0,796**.
- Desplazamientos más informativos: −5, −133, −1, −2, −3, −6, −4, −10, −100, −168, −257, −129.
- Figura disponible: `4_resultados/resultados_analisis_bytes/fig_importancia_por_posicion.png`
  (generada en el clúster; **falta bajarla**).

**Qué escribir:** el ransomware firma al final del archivo. Converge con el Experimento 2b,
donde once de las quince familias con firma binaria la tienen como sufijo y solo cuatro como
prefijo. Explicación mecánica: añadir un bloque de control al final es más simple que
insertarlo al principio, que obligaría a desplazar todo el contenido.

### A4. Ablación de ventana → nueva subsección en §4.5

| Ventana | Bytes leídos | Exactitud |
|---|---|---|
| 64 + 64 | 128 | 0,795 |
| 128 + 128 | 256 | 0,793 |
| 256 + 256 | 512 | 0,852 |
| 512 + 512 | 1.024 | 0,908 |
| **Solo cabecera** | 512 | **0,321** |
| **Solo cola** | 512 | **0,752** |

Figura ya generada: `images/fig_ablacion_ventana.png` (insertar en el capítulo).

**Qué escribir:** con 128 bytes se alcanza el 87 % del rendimiento máximo, dato relevante
para una implementación práctica. La cola sola supera al doble de la cabecera sola, pero
ambas regiones son necesarias para el máximo: la combinación suma 15 puntos sobre la cola.

### A5. Diagnóstico de las seis familias difíciles → nueva subsección en §4.5

Confusión dentro del grupo: **97,2 % a 99,4 %** en las seis. Forman un grupo de confusión mutua.

| Familia | Entropía cabecera | Entropía cola |
|---|---|---|
| CRYPTOLOCKER | 7,59 | 7,59 |
| WASTEDLOCKER | 7,59 | 7,58 |
| JIGSAW | 7,51 | 7,55 |
| DARKSIDE | 7,59 | 7,49 |
| **SUNCRYPT** | 7,59 | **4,78** |
| **NOTPETYA** | 7,34 | **6,58** |
| Resto de familias | 7,03 | 7,44 |

**Qué escribir, con el matiz:** no fallan todas por el mismo motivo. Las primeras cuatro
presentan entropía próxima al máximo en todo el archivo: el cifrado lo ocupa por completo y
no hay marca que aprender. Pero **SUNCRYPT y NOTPETYA sí añaden datos no aleatorios al
final** —SUNCRYPT tiene la entropía de cola más baja de todo el conjunto— y aun así no se
identifican bien. La explicación más plausible es que ese bloque **varía en cada archivo**
(una clave o un identificador por víctima): es estructura, pero no firma de familia.
⚠️ No escribir la conclusión simplificada de "no dejan estructura": es falsa para dos de las seis.

### A6. Familias que renombran el archivo por completo → §4.4 o §4.5

**BLACKMATTER y CERBER** sustituyen el nombre original por cadenas aleatorias
(`ontbgnqc`, `miwv13q3`, `ryidzs2b`…), de modo que pierden toda referencia al documento de
origen. Son 491 archivos del conjunto evaluado.

**Qué escribir:** con estas familias cualquier método basado en el nombre o la extensión es
inútil, y sin embargo el clasificador sobre bytes las identifica sin dificultad (CERBER
obtiene F1 = 1,00). Refuerza el argumento de emplear el contenido y no los metadatos.

### A7. Normalización de marcadores en las notas → nueva subsección junto a §4.7.5

Sustitución de correos, `.onion`, Bitcoin, URLs e identificadores por etiquetas de tipo.
130 de 144 notas modificadas, 1.281 sustituciones.

| Protocolo | Original | Normalizado | Diferencia |
|---|---|---|---|
| P2 · palabras | 0,414 | 0,414 | +0,000 |
| P2 · caracteres | 0,409 | 0,391 | −0,018 |
| P2 · combinado | 0,421 | 0,410 | −0,011 |
| P1 · promedio | 0,751 | 0,746 | −0,005 |

**Qué escribir:** no mejora; si acaso perjudica levemente la representación de caracteres, lo
que sugiere que los n-gramas extraían señal de la *forma* de los marcadores (longitud de una
dirección `.onion`, formato de una URL) y la etiqueta uniforme la destruye. Conclusión: la
variabilidad entre plantillas de una misma familia es **estructural** y no se reduce a los
datos de contacto. Fundamento bibliográfico de la técnica: Lemmou et al. (2021) y
Trujillo (2022) aplican sustituciones equivalentes.

### A8. La lectura conjunta de los tres resultados negativos → §4.11

Hiperparámetros de notas · normalización de marcadores · hiperparámetros de características
estadísticas. **Los tres sin mejora.** Escribir que convergen en la misma conclusión: el
techo no está en el método sino en los datos —en las notas, la cantidad de plantillas por
familia; en los archivos, la información contenida en las características estadísticas.

---

## B. Correcciones que necesita cada capítulo

### Capítulo 1 — Introducción
- [ ] Revisar si afirma que la identificación de familia por archivos cifrados es inviable.
      Ya no lo es: 0,910. Ajustar sin quitar el planteo del problema.
- [ ] Considerar incorporar las cifras de impacto económico de Patsakis et al. (2024) como
      motivación: NetWalker 27,5 M USD, Locky 14,0 M, REvil 12,1 M, MedusaLocker 5,3 M,
      HelloKitty 1,07 M — varias de ellas están entre las 30 familias del trabajo.

### Capítulo 2 — Marco teórico
- [ ] **§2.6, reformular la novedad.** Lemmou et al. (2021) **sí** identifican la familia a
      partir de la nota, mediante reglas y marcadores (correos, Bitcoin, `.onion`, palabras
      clave) más LSA como búsqueda de casi-duplicados: 181 de 182 notas en un escenario de
      mundo cerrado. Su componente de aprendizaje automático es solo binario (nombre de nota
      frente a nombre benigno, F 0,920). **Nunca afirmar que nadie clasificó familias por
      notas.** La formulación correcta: primer clasificador supervisado multiclase sobre el
      contenido completo, con medición explícita de la generalización a variantes no vistas y
      métricas macro.
- [ ] Añadir a los trabajos relacionados: Davies et al. (2023) sobre votación mayoritaria;
      Trujillo (2022), tesis UPC de detección binaria de notas; Bou-Harb et al. (2021), que
      atribuyen familia por actividades de red pre-ataque (complementario: artefactos
      pre-ataque frente a post-ataque).
- [ ] Incorporar la cita de Davies et al. (2023) que propone como mejora futura *aplicar
      procesamiento de lenguaje natural sobre los strings de las notas* — es el hueco que
      este trabajo llena, señalado por el propio grupo creador de NapierOne.

### Capítulo 3 — Metodología
- [ ] Describir la metodología de la validación *dejar-un-tipo-fuera* (A2).
- [ ] Describir la búsqueda de hiperparámetros sobre las características estadísticas (A1).
- [ ] Describir la normalización de marcadores como experimento planificado (A7).
- [ ] Revisar la sección de procedencia del corpus: **auditar las 37 notas etiquetadas
      "NapierOne/varios"**. Pista: el repositorio `kipziptie` que apareció en `Pruebas.xlsx`.
      NapierOne es un conjunto de archivos mixtos, **no** un corpus de notas.

### Capítulo 4 — Resultados
- [ ] Incorporar A1 a A8 (ver arriba).
- [ ] Insertar `images/fig_ablacion_ventana.png`.
- [ ] Bajar del clúster e insertar `fig_importancia_por_posicion.png`.
- [ ] Revisar que §4.1 y §4.2 (pruebas preliminares y Experimento 1, escritos antes) no
      contradigan la reformulación del Objetivo 2.

### Capítulo 5 — Conclusión
- [ ] **Se escribe al final, cuando todos los resultados estén cerrados.** Decisión tomada.
- [ ] Al escribirla, eliminar las cifras superadas que aún contiene: el «100 %» de la
      detección binaria (provenía de una prueba sin validación cruzada, con sobreajuste
      admitido en el propio texto) y el «15,4 %» de la multiclase, que era un valor de
      **error** leído como exactitud; el valor correcto es 9,9 % con dos características y
      60,3 % con diecinueve.

---

## C. Bibliografía

### Corregir
- [ ] **`lee2022`**: entrada corrupta, comparte título con `paper_3_enhancing_file_entropy`
      (Hsu et al. 2021) pero con otros autores. **Se cita cinco veces** para sostener que la
      entropía de Shannon no es la métrica óptima. Verificar cuál es el trabajo real.
- [ ] `cryptosheriff` está en el `.bib` pero nunca se cita, aunque la herramienta se discute
      en tres capítulos. Citarla donde corresponde.
- [ ] Cuatro entradas huérfanas: `SP-800`, `paper_1_dataset`, `codingo_ransomware`, `cryptosheriff`.
- [ ] ⚠️ **`latex_capitulos/referencias.bib` (en `_archivo/`) tiene al menos seis autorías
      incorrectas. No migrar nada de ahí.**

### Incorporar
- [ ] **Davies, Macfarlane y Buchanan (2022), «NapierOne»**, *FSI: Digital Investigation* 40,
      301330 — es la cita del propio conjunto de datos. **El PDF ya está** en
      `5_bibliografia/Tesis Carlos y Romina/Papers/NapierOne_Data_Set(1).pdf`.
- [ ] Davies et al. (2022), «Comparison of Entropy Calculation Methods for Ransomware
      Encrypted File Identification», *Entropy* 24(10), 1503 — compara once técnicas de
      entropía sobre más de 270.000 archivos.
- [ ] **Gómez Hernández et al. (2023)**, *Electronics* 12(21):4494 — **sugerido por el tutor
      el 13/03/2024** y citado en el paper del Simposio, pero ausente del `.bib`.
- [ ] Pont (2023), tesis doctoral, University of Kent — marcada como interesante por el tutor
      el 01/04/2024. Referencia canónica de la parte estadística.
- [ ] Trujillo (2022), tesis de máster UPC.
- [ ] Sokolova y Lapalme (2009) o equivalente — respaldo metodológico del macro-F1 y de la
      evaluación en multiclase desbalanceada. **Falta justamente donde más se necesita.**
- [ ] De Gaspari et al. (2020), «EnCoD» — respalda el resultado negativo del Objetivo 2.
- [ ] Yamany et al. (2022), *Electronics* — clasificación de familias por características
      estáticas del ejecutable.
- [ ] ✅ `davies2023majority` — ya incorporada el 2026-08-05.

---

## D. Limitaciones que hay que declarar explícitamente

1. **Una sola campaña por familia.** NapierOne representa cada familia con una única muestra
   ejecutada, por lo que el 0,910 mide identificación de *esa campaña*. La generalización a
   campañas futuras no es evaluable con los datos disponibles. **Es la limitación más
   importante del trabajo.** Ya declarada en §4.5.4; mantenerla visible.
2. **Baja variabilidad de plantillas en el corpus de notas**: 146 notas corresponden a 95
   contenidos distintos, y siete familias están representadas por una sola plantilla.
3. **Procedencia mixta de las notas**: 96 archivos brutos, 37 del corpus previo y 13
   transcripciones. Seis familias (BadRabbit, Chimera, Jigsaw, NotPetya, WannaCry,
   CryptoLocker) no tienen archivo bruto público. **Consultar con el tutor si son admisibles.**
4. **BLACKBASTA ausente** de la copia del clúster: los experimentos sobre archivos usan 29 de
   las 30 familias. Declararlo.
5. **Dos notas de DHARMA perdidas** por el antivirus (`Info__13.hta`, `Info__3.hta`). Las
   corridas de notas del clúster usan 144 y las de la PC 146. **Restaurarlas y unificar la
   base** antes de cerrar el documento.
6. La detección binaria (Experimento 1) se evaluó con 6 características sobre 1.600 archivos;
   no se repitió con el conjunto completo de 29.029.

---

## E. Consultas para la reunión con el tutor

1. ¿Se admiten como fuentes las notas obtenidas por transcripción documentada, para las seis
   familias sin archivo bruto público, declarándolo como limitación?
2. ¿Conviene reorganizar los capítulos de modo que cada objetivo específico corresponda a una
   sección, según lo sugerido el 08/05/2024?
3. El clúster dispone de GPU. ¿Incluir una comparación con modelos de tipo *transformer* como
   sección adicional, o dejarla como trabajo futuro dado el tamaño del corpus (146 documentos)?
4. Criterio de ampliación del corpus: se propone **al menos tres plantillas distintas por
   familia** en lugar de un número fijo de notas. ¿Se considera adecuado?
5. ¿Incorporar el aprendizaje *few-shot* (arXiv 1908.06750, sugerido el 06/06/2024) para las
   familias con una o dos plantillas, o dejarlo como trabajo futuro?

---

## F. Cierre del documento (Bloque E del plan)

- [ ] Conclusión completa (ver capítulo 5).
- [ ] Resumen en español y *abstract* en inglés — hoy dicen «Este es el resumen» y «This is
      the abstract».
- [ ] Carátulas: `caratula.tex` y `caratula2.tex` conservan «Titulo 1», «Nombre y apellido» y
      «Septiembre - 2019». `portada.tex` tiene «grado de XXXXXX».
- [ ] Dedicatoria: «Dedico a mi blah blah blah».
- [ ] Agradecimientos: vacío. **Incluir el agradecimiento al clúster del NIDTEC, que el
      reglamento de uso exige** (proyecto LABO16-167, PROCIENCIA/CONACYT).
- [ ] Lista de símbolos: vacía.
- [ ] Apéndice: revisar que el índice alfabético no quede vacío.
- [ ] Unificar el tamaño del corpus en todo el documento (146 notas).

---

## G. Nota práctica sobre el clúster

Los nodos **c1 y c3 tienen el acceso a `/scratch` degradado**: un trabajo puede quedar horas
consumiendo un 2 % de CPU sin leer datos. Lanzar siempre con `--nodelist=c2`. Conviene
informarlo al administrador del clúster.
