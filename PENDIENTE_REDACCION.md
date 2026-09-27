# Qué falta escribir en la tesis

_Actualizado el 2026-08-12. Inventario de todo lo que ya está medido pero todavía no
redactado, más las correcciones que cada capítulo necesita._

Los resultados están en `4_resultados/` y se regeneran con el código de `2_codigo/`.
**Ninguna cifra se transcribe a mano.**

---

## A. Resultados obtenidos que NO están todavía en el documento

> **✅ Estado al 2026-08-28: A1 a A6 están ESCRITOS** (frente de archivos cifrados), más tres
> subsecciones que no estaban en esta lista: §4.4.4 el criterio de detección del 2b, §4.4.5 la
> validación externa contra ID Ransomware y §4.5.4 la validación anidada con su desvío, y
> §4.5.9 la verificación de integridad (los 12 JPEG de CERBER). Detalle y cifras corregidas en
> el bloque «ESCRITO EN LA TESIS (2026-08-28)» de `ESTADO_TESIS.md`.
> **Quedan de esta sección: A7** (normalización de marcadores, frente de notas) **y A8** (la
> lectura conjunta de los negativos, que va en la discusión del capítulo).
>
> ⚠️ **Correcciones a esta lista, verificadas contra los CSV:** en **A3** la importancia de
> cabecera/cola es **0,2043 / 0,7957** (no 0,204/0,796 redondeado a mano: coincide) y los
> últimos 16 bytes suman **27,8 %**; en **A6** las familias que renombran son **CERBER con 981
> archivos y BLACKMATTER con 13**, no «491 archivos del conjunto evaluado».

### ✅ A1 — ESCRITO 2026-08-28 como §4.3.3. Hiperparámetros de las características estadísticas

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

### ✅ A2 — ESCRITO 2026-08-28 como §4.5.5. Generalización a tipos de documento nunca vistos

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

### ✅ A3 — ESCRITO 2026-08-28 dentro de §4.5.6 (con la figura). Dónde reside la información dentro del archivo

- Importancia acumulada: **cabecera 0,204 · cola 0,796**.
- Desplazamientos más informativos: −5, −133, −1, −2, −3, −6, −4, −10, −100, −168, −257, −129.
- Figura disponible: `4_resultados/resultados_analisis_bytes/fig_importancia_por_posicion.png`
  (generada en el clúster; **falta bajarla**).

**Qué escribir:** el ransomware firma al final del archivo. Converge con el Experimento 2b,
donde once de las quince familias con firma binaria la tienen como sufijo y solo cuatro como
prefijo. Explicación mecánica: añadir un bloque de control al final es más simple que
insertarlo al principio, que obligaría a desplazar todo el contenido.

### ✅ A4 — ESCRITO 2026-08-28 como §4.5.6, con la ablación EXTENDIDA. Ablación de ventana

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

### ✅ A5 — ESCRITO 2026-08-28 como §4.5.7, con el matiz. Diagnóstico de las seis familias difíciles

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

### ✅ A6 — ESCRITO 2026-08-28 como §4.5.8. Familias que renombran el archivo por completo

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
4. Alcance de la ampliación del corpus: las fuentes públicas ya fueron recolectadas de
   forma sistemática y para varias familias probablemente no existan plantillas
   adicionales publicadas. ¿Es aceptable **documentar el límite que impone la evidencia
   disponible** —reportando el desempeño según la cantidad de plantillas por familia— en
   lugar de comprometer una cantidad mínima que depende de terceros?
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

---

## H. Actualizar `GUIA_CODIGO.md` (pedido de Romina, 2026-08-18)

La guía cubre `clasificador_notas_v2.py`, `clasificador_bytes.py` y `deteccion_estructural.py`,
pero quedó atrás. Falta explicar, en el mismo estilo de orden de lectura:

- **`ablacion_ventana_extendida.py`** (nuevo) — la curva de ventana hasta 4096 bytes, el bloque
  del medio, y el control sin relleno de ceros que descarta el artefacto del tamaño.
- **`resumen_para_capitulo4.py`** (nuevo) — agrega los resultados multisemilla y emite las
  tablas del capítulo. Existe porque los experimentos escriben un archivo por semilla y el
  promedio se venía haciendo a mano.
- **`deteccion_estructural.py`** (cambiado) — criterio de mayoría en lugar de unanimidad byte a
  byte, exclusión del `.pdf` de documentación, muestreo aleatorio con semilla, y salida en una
  carpeta por semilla y job.
- **`clasificador_bytes.py`** (cambiado) — modo multisemilla, carpeta de salida propia por
  corrida, y las semillas que estaban clavadas.

Romina pidió que se lo explique cuando llegue a esa etapa, no ahora.

---

## I. Trabajo metodológico interno de la construcción del corpus (a redactar)

_Pedido de Romina (2026-08-20). Todo esto se hizo y sostiene las cifras del cap. 4, pero no
está escrito. Va en el capítulo de metodología (construcción del corpus de notas) + una nota
de limitaciones. Solo se AGREGA._

### I1. Criterio de «texto distinto» por similitud de coseno

- El corpus **no se cuenta por archivos sino por textos distintos (plantillas)**. Dos notas se
  consideran la misma plantilla si su **similitud de coseno ≥ 0,90** sobre **TF-IDF de
  caracteres `char_wb`, n-gramas 3-5**, agrupando por **componentes conexas**
  (`agrupar_neardups()` en `2_codigo/clasificador_notas_v2.py`, `UMBRAL_NEARDUP = 0.90`).
- Justificación medida (B.1): **lo que mueve el macro-F1 es el texto distinto, no la nota
  repetida**. Por eso una nota que repite un contenido ya presente no aporta.
- Consecuencia a mostrar con ejemplos: **archivos ≠ textos distintos**. WASTEDLOCKER = 4
  archivos → **1** plantilla; DHARMA 19 → 6; CERBER 18 → 8. El molde rígido de WastedLocker
  (~250 caracteres, solo cambia víctima/correos) explica su F1 por familia 0,000.

### I2. Verificación de cada nota candidata antes de incorporarla

- Toda nota recolectada se pasa por `2_codigo/verificar_nota_nueva.py` **antes** de sumarla:
  calcula el coseno contra todo el corpus y dictamina **«COPIA»** (≥ 0,90 con algo existente,
  no aporta) o **«TEXTO NUEVO»** (< 0,90, cuenta como plantilla nueva). Umbral 0,90, mismo
  criterio con el que se midió todo el frente.
- Corroboración independiente citable: **Windows Defender trae una firma propia para el TEXTO
  de la nota de Chimera** (`Ransom:HTML/Chicrypt.A`); una transcripción en texto plano la
  dispara. Es evidencia, de un proveedor de antivirus, de que **el texto de la nota por sí solo
  identifica a la familia** — la premisa del frente de notas.

### I3. Homónimos y trampa campaña-vs-familia (casos reales encontrados)

Declarar que la desambiguación de familias fue un trabajo explícito, con casos concretos:

- **Medusa ≠ MedusaLocker** (familias sin relación, FBI/CISA AA25-071A): 2 notas de Medusa
  estaban mal etiquetadas en MEDUZALOCKER → movidas a `3_datos/descartados_integridad/`.
- **Crypt0l0cker = TorrentLocker ≠ CryptoLocker (2013):** `lm_Crypt0l0cker_HOW_TO_RESTORE_FILES.html`
  estaba mal etiquetada en CRYPTOLOCKER → movida.
- **WastedLocker:** los textos «distintos» atribuibles son de **sucesores renombrados de Evil
  Corp**, no de WastedLocker → no se etiquetan como esa familia.
- No mezclar familias emparentadas o casi idénticas: **Ryuk vs Conti** (sucesor), **NotPetya vs
  BadRabbit** (texto casi idéntico, otra familia), **Nemty vs NemucodAES** (repo Lemmou).
- **`id-ransomware.blogspot.com` (Amigo-A / Andrew Ivanov) ≠ ID Ransomware** (servicio de
  MalwareHunterTeam con el que se compara en `Pruebas.xlsx`). SANS las lista como dos fuentes
  distintas. No confundirlas al citar.
- El homónimo también apareció en el frente de bytes (el «parpadeo» de familias con firma según
  la muestra).

### I4. Fuentes y criterio de admisión

- **Repos en disco (fuente más segura, sin red):** ThreatLabz `ransomware_notes`
  (`github.com/ThreatLabz`) y Lemmou `RansomNoteFiles` (`github.com/lemmou`, congelado en
  2019); **NapierOne** (archivos cifrados + parte de las notas); y pcrisk (Tomas Meskauskas).
- **Lista blanca de fuentes citables verificadas:** pcrisk, bleepingcomputer, malwarebytes,
  sophos, sonicwall, helpnetsecurity, cisecurity, cisa, ic3 (PDF primario del FBI),
  id-ransomware.blogspot (Amigo-A), `api.github`/`raw.githubusercontent` (solo texto/listado).
- **Uso real de `id-ransomware.blogspot.com` (Amigo-A / Andrew Ivanov):** se tomó **exactamente
  una nota** de esa fuente — MAZE, etapa **ChaCha** (`idr_maze_chacha_2019.txt`, transcripción
  de `DECRYPT-FILES.html` «0010 SYSTEM FAILURE 0010», 2019-05-13,
  `https://id-ransomware.blogspot.com/2019/05/chacha-ransomware.html`; base64 truncado con `***`
  en la fuente). Es la única nota del corpus proveniente de ese blog. Acreditar la fuente al
  citarla y **no confundir el blog con el servicio ID Ransomware** (ver I3).
- **Criterio de admisión de una fuente — 6 chequeos, en orden:** (1) ¿la recomienda un tercero
  confiable? (SANS lista id-ransomware.blogspot); (2) ¿autoría verificable?; (3) ¿el dominio
  puede caducar y ser recomprado?; (4) ¿redirige fuera de su dominio?; (5) ¿vendor original o
  agregador comercial?; (6) ¿homónimo?
- **Incidente que ilustra el chequeo 3 (contarlo):** `malwiki.org` respondía **301 y redirigía a
  un dominio sin relación** (dominio caducado y recomprado) → no se siguió el redirect ni se usó.
- **Reglas de integridad de la recolección:** solo se lee texto (no se bajan muestras, binarios
  ni `.zip`); no se siguen redirects fuera de dominio; no se usan agregadores comerciales de
  «recovery/decryptor».
- **Trazabilidad:** `3_datos/manifiesto_corpus_v2.csv` con columnas `familia, archivo,
  extension_original, tipo (bruto/transcripcion/corpus-existente), fuente` (autor, fecha, URL).
  En notas transcriptas la `extension_original` es un dato **documentado, no observado**, y así
  se declara.

### I5. Limitación honesta a declarar

**~24 % del corpus (37 filas `corpus-existente`) tiene fuente solo «NapierOne/varios» sin URL**,
y `chimera_note2.txt` sigue sin fuente rastreable. Pendiente de decisión del tutor; hay que
declararlo, no esconderlo.

---

## J. Extensión del frente de notas (a redactar como sección que se AGREGA al cap. 4)

_Trabajo del 2026-08-20. Es una **extensión con base declarada** (corpus extendido a **155 notas**);
**NO reemplaza** las cifras canónicas de 30 familias. Método detallado en la sección I; operativa en
`6_notas_trabajo/extension_notas_id-ransomware_2026-08-20.md`._

**Fuente y método:** `id-ransomware.blogspot.com` (Amigo-A, whitelisted por SANS), leída **sin
ejecutar JavaScript**; capturas OCR-eadas por **subagente-visión**; cada texto verificado con el
criterio de coseno ≥ 0,90; se redacta **solo lo que lleva al ejecutable/payload** (el `.rar` de
WannaCry), mientras correos/BTC/`.onion` de contacto quedan verbatim como el resto del corpus.

**Resultado sobre las 5 familias que estaban trabadas (corpus 151 → 155, +4 textos):**

| Familia | Antes | Ahora | Qué se agregó / concluyó |
|---|---:|---:|---|
| WANNACRY | 2 | **4** | +2 variantes del original marzo 2017 (Q&A y mensaje de pantalla). Cerrada. |
| RYUK | 2 | **4** | +2: nota corta «balance of shadow universe» y variante portal Tor 2021 (OCR de 18 capturas). Cerrada. |
| NOTPETYA | 2 | 2 | Agotado **confirmado** (texto + 2 capturas = copia). |
| CRYPTOLOCKER | 3 | 3 | Nota principal = original 2013 **con procedencia confirmada** (Amigo-A). La ventana de pago NO se cuenta como nota (decisión de alcance, análoga a Maze wallpaper-vs-sitio-Tor). |
| WASTEDLOCKER | 1 | 1 | **1 molde confirmado** (3 muestras 2020 incl. Garmin colapsan). Sucesores (SecCrypt, Phoenix, PAYLOADBIN, Macaw, Easy2lock, Hades) = otras familias. |

**Hallazgo transversal a escribir (importante):** para varias familias el techo es la **naturaleza
de la familia, no la recolección** — WastedLocker produce **un** molde; el CryptoLocker original 2013
**no dejaba archivo de nota** (solo ventana); NotPetya tuvo **una** nota. Esto **explica sus F1
por familia bajos o 0** y es un **resultado citable**, no una carencia de esfuerzo. Contrasta con
WannaCry y Ryuk, que sí tenían variantes distintas recuperables.

**✅ RE-MEDICIÓN HECHA (2026-08-20).** Corrida sobre **155 notas / 30 familias / 106 plantillas**,
a carpeta nueva `4_resultados/resultados_extension_155/` (canónicas intactas). Código: parámetro
`--salida` en `clasificador_notas_v2.py`, `curva_aprendizaje_notas.py` y `resumen_para_capitulo4.py`
(commit `a914e33`), **sin cambios de método**. Corpus verificado 1:1 contra el manifiesto (155 filas,
0 discrepancias). Control interno de la curva: k=todo reprodujo el evaluador canónico sobre 155.
Detalle y contraste de las predicciones preregistradas en `ESTADO_TESIS.md`, sección «RESULTADO DE
LA RE-MEDICIÓN SOBRE 155 NOTAS».

**Cifras globales a escribir (con base y métrica pegadas):**

| Métrica | 155 notas | Canónica |
|---|---|---|
| P2 macro-F1 (grupos, combinado+LinearSVC) | 0,527 ± 0,049 | 0,435 (146) · 0,421 (144) |
| P1 macro-F1 (estratificado, caracteres+LinearSVC) | 0,796 ± 0,019 | 0,760 (146) |
| P2ret macro-F1 (30 familias, k=todo, R=100) | 0,761 ± 0,060 | 0,616 (144) |

Reparto por familia **bajo P2 estricto** (2 pliegues): **0/30 ≥ 0,90 · 19/30 ≥ 0,50 · 11/30 < 0,50**.
**Bajo P2ret**: **14/30 ≥ 0,90**. ⚠️ El mismo corpus da 0/30 o 14/30 según el protocolo: **declarar
siempre cuál**.

**Contraste de las 3 predicciones preregistradas** (F1 por familia, P2ret 30fam k=todo, 144→155):

| Familia | 144 | 155 | Predicción | Veredicto |
|---|---:|---:|---|---|
| WANNACRY | 0,010 | 0,733 | (a) sube fuerte | cumple (fuerte) |
| RYUK | 0,032 | 0,271 | (a) sube fuerte | parcial (a 0,27, ±0,42) |
| NOTPETYA | 0,695 | 0,937 | (b) no se mueve | **falla** (+0,24) |
| CRYPTOLOCKER | 0,410 | 0,097 | (b) no se mueve | **falla** (−0,31) |
| WASTEDLOCKER | 0,000 | 0,987 | (b) no se mueve | **falla** (+0,99) |

(c) El P2 global subió **+0,092** (0,435→0,527), ~3× lo predicho como «modesto».

**Qué escribir:** WannaCry y Ryuk ya entran en las cifras de la extensión (variantes distintas
recuperadas, aunque Ryuk queda ruidosa). El resto de las trabadas confirma el hallazgo transversal
de arriba: su techo es la naturaleza de la familia, no la recolección.

### J.bis — Hallazgo metodológico: el F1 por familia NO es estable entre versiones del corpus

_Surgió de la re-medición 144→155 (2026-08-20). Va en la discusión + limitaciones. **Refina la
sección K**: sobre 155, WastedLocker ya no es de una sola plantilla._

El F1 por familia bajo P2/P2ret **cambia aunque a la familia no se le agregue nada**, por tres vías
que hay que declarar al comparar cifras entre versiones del corpus:

1. **Reagrupamiento global de casi-duplicados.** El IDF del TF-IDF se ajusta sobre todo el corpus; al
   crecer 144→155, familias al borde del umbral 0,90 se **descolapsan**. **WASTEDLOCKER**: sus 4 notas
   eran **1 plantilla** en 144 (F1=0 por construcción, ver K) y son **3 plantillas** en 155, así que
   pasa a evaluable y su F1 salta a **0,987** — **artefacto de partición, no aprendizaje** (las 3
   quedan a coseno ~0,89 entre sí, casi-duplicadas separadas por la deriva del umbral).
2. **Retiros de notas** cambian la composición. **CRYPTOLOCKER** perdió la nota `lm_Crypt0l0cker`
   (TorrentLocker, mal etiquetada; ver I3): 4→3 notas, 3→2 plantillas, F1 0,410→0,097.
3. **La frontera multiclase depende del corpus entero.** **NOTPETYA** no recibió ni perdió nada
   (sigue 2 notas / 2 plantillas) y su F1 subió 0,695→0,937: al afinarse la frontera con las demás
   familias, se volvió más separable. Es la refutación más limpia del supuesto «sin material nuevo ⇒
   el F1 no cambia».

**Consecuencia para la redacción:** toda comparación de F1 por familia entre corpus de distinto
tamaño debe declarar que el reagrupamiento se recalcula global. Es también la razón por la que las
cifras canónicas (146/144) y las de la extensión (155) **se reportan por separado, cada una con su
base declarada**, y no se «corrigen» unas con otras.

---

## K. Cómo interpretar los F1 por familia = 0 (familias de una sola plantilla) — para la discusión del cap. 4

_Material para redactar la discusión y para responder al tutor si pregunta por los F1 = 0
(WASTEDLOCKER, etc.). Aplica a las cifras canónicas, no solo a la extensión._

**El punto:** un F1 por familia de **0,000 bajo el protocolo P2ret NO significa «el modelo no sabe
clasificar esa familia»**. Es un **artefacto de la métrica** cuando la familia tiene **una sola
plantilla** de nota.

- **P2ret retiene una plantilla por familia para el test**, o sea que mide *«¿reconoce una variante
  NO vista de esta familia?»*. Para una familia con **una sola** plantilla, al retener su única nota
  no queda nada para entrenarla → nunca se predice esa etiqueta → **F1 = 0 por construcción**. La
  métrica se queda sin variante contra la cual medir generalización, porque **no existe**.
- **En uso real, esa familia se clasifica perfecto:** toda víctima recibe la misma nota; el modelo
  la reconoce por **coincidencia exacta / casi-exacta** (es lo que hace ID Ransomware). Es el caso
  **más fácil**, no imposible.
- **Tensión honesta (declararla):** entrenar *y* testear con la misma única nota daría F1 = 1,0 pero
  es **memorización** (fuga); retener la única nota (P2ret) da F1 = 0. Ninguna «mide» bien porque el
  concepto de *generalizar a variantes no vistas* está **vacío** para una familia sin variantes.

**Cómo se prueba que una familia «tiene una sola plantilla» (y no que «solo recolectamos una»):**
buscando más muestras y midiendo el coseno. Si varias muestras **colapsan** (> 0,90) → es la
familia. **Probado para WASTEDLOCKER** (3 muestras 2020 —BBA, Garmin, censurada— colapsan, molde
`.wasted_info` de ~250 car.). En la extensión de ThreatLabz, **14 familias** muestran lo mismo
(2-3 archivos que colapsan a 1): `akira`, `hive`, `medusa`, `monti`, `safepay`, `warlock`,
`ransomhouse`, `dataleak`, `embargo`, `fog`, `gunra`, `kawalocker`, `mallox`, `nitrogen`.
Distinto es tener **1 solo archivo sin haber buscado más** = incógnita (podría tener más).

**Redacción:** interpretar los F1 = 0 de familias de una plantilla como **«sin variantes distintas
para evaluar generalización; identificable por coincidencia exacta»**, es decir un **hallazgo sobre
la familia**, no una falla del clasificador. Reafirma por qué se cuenta *textos distintos* (para
medir generalización) y no *archivos* (para identificar, con uno alcanza).

---

## K. Estructura del frente de notas: Exp. 3, 3b, 3c, 3d (propuesta 2026-08-20, pedido de Romina)

El frente de archivos tiene escalera con nombre (2 → 2b → 2c → 2d); el de notas acumuló los
hallazgos bajo un solo «Experimento 3». Estructura espejo propuesta — **el Exp. 3 ya escrito no
se toca; 3b, 3c y 3d se AGREGAN como subsecciones nuevas**, cada una con su base declarada:

### Exp. 3 — El clasificador y los dos escenarios (YA ESCRITO, no tocar)
Base: 146/144 notas. TF-IDF + LinearSVC, protocolos P1/P2, macro-F1 0,760 ± 0,029 / 0,435 ±
0,057. El hallazgo de las plantillas (146 notas → 95 contenidos).

### Exp. 3b — ¿Cuánto dato hace falta? (la curva de aprendizaje — sale de B.1)
Base: 144 notas (re-medida sobre 155). Fuentes: `4_resultados/resumen_capitulo4/b1_*.csv` y
`resultados_extension_155/resultados_curva_notas/`.
- La moneda son los TEXTOS DISTINTOS, no las notas (curvas superpuestas en el eje de
  plantillas, diferencia media 0,0052 de macro-F1).
- El techo está en 4 plantillas por familia (paso 3→4: Δ macro-F1 +0,0297, IC 95 %
  [+0,0039; +0,0554]; paso 4→5: +0,0006, indistinguible de cero).
- La chatura de la curva de 30 familias era agotamiento del corpus, no saturación del
  aprendizaje (`n_fam_bajo_tope`).
- Contesta el pedido textual del tutor: cuántas notas hacen falta.

### Exp. 3c — Qué predice el rendimiento y qué NO lo infla (cohesión + grafo — sale de B.3)
Base 144: `resultados_grafo_marcadores/`. **Re-medición sobre 155 HECHA (2026-08-21):**
`resultados_grafo_marcadores_155/` (155 notas / 106 plantillas / 30 familias / 108 nodos),
mismo `grafo_marcadores.py` con `--salida` (commit `091cf7b`), sin cambios de método. Las dos
bases se reportan por separado (misma lógica que J.bis). Correlación recalculada contra el F1
por familia de `resultados_extension_155/` (P2ret, k=todo, 100 rep.).

- **Lo que predice el F1 de una familia es la COHESIÓN entre sus plantillas, y SE SOSTIENE
  sobre 155:** Spearman ρ **+0,69** (p ≈ 3·10⁻⁵, n=30), casi igual que sobre 144 (+0,704).
  Control de método: el mismo cálculo sobre 144 reproduce **+0,7041**. El **margen** pasó a ser
  el predictor más fuerte (ρ +0,75).
- ⚠️ **La cantidad de plantillas: sobre 144 era nula (ρ −0,108, n.s.); sobre 155 da ρ −0,49
  (p = 0,006), pero es un CONFUNDIDO de la recolección dirigida** (se sumaron plantillas justo a
  las familias difíciles), NO causal — no contradice B.1. Declararlo así.
- La **fracción de pares unidos por marcador** dejó de ser significativa (144: +0,425, sig.;
  155: +0,30, n.s.).
- **Cambios de cohesión 144→155, con causa (tabla en `ESTADO_TESIS.md`, «B.3 RE-MEDIDO»):**
  MEDUZALOCKER 0,43→0,65 (se limpió la contaminación Medusa; margen pasó a positivo),
  CHIMERA 0,15→0,34 (2 plantillas nuevas; deja de ser la más baja), JIGSAW 0,51→0,23 (sumó
  traducciones de/fr), RYUK 0,44→0,22 (ahora la más baja: fragmentos ultra-cortos solo-IOC),
  WASTEDLOCKER 1 plantilla→3 (deriva del IDF, ver J.bis; cohesión 0,90, la más alta),
  CRYPTOLOCKER 0,36→0,43 (se retiró la nota TorrentLocker mal etiquetada).
- **Ningún IOC operativo (email, onion, billetera) se comparte entre familias** (se sostiene en
  155): las aristas entre familias son solo URLs de torproject (bajaron de 7 a 5). Los
  marcadores son privados de cada familia.
- Los **parentescos por CONTENIDO** persisten y NO aparecieron nuevos: grupo 6 BLACKBASTA↔CONTI,
  grupo 56 DHARMA↔PHOBOS (salen de la deduplicación por contenido, NO del grafo de IOCs).
- **Familias con plantillas en IDIOMAS distintos (verificado nota por nota; es el insumo de la
  preregistración de 3e):** CHIMERA (de+en), JIGSAW (de+en+fr), GANDCRAB (en+fr). RYUK NO entra
  (falso positivo del detector automático). SUNCRYPT/TESLACRYPT tampoco (inglés con
  selector/enlace de traducción); WANNACRY es multi-idioma en la realidad pero el corpus solo
  tiene sus notas en inglés.
- **Control P3 (medido solo sobre 144, no re-corrido en 155):** el macro-F1 0,435 NO es búsqueda
  de datos repetidos — la caída completa la explica el agrupamiento grueso (azar
  0,2251 ± 0,0149 vs P3 real 0,2293). Es la respuesta anticipada a la objeción tipo Lemmou.

### Exp. 3d — La recolección dirigida como validación fuera de muestra (la corrida de 155)
Base: 155 notas / 106 plantillas / 30 familias. Fuentes: `resultados_extension_155/` y el
bloque «PREDICCIONES PREREGISTRADAS» de `ESTADO_TESIS.md`.
- Predicciones escritas ANTES de correr, guiadas por 3b: se recolectó hasta 4 plantillas en
  las familias señaladas.
- Resultado: P2 macro-F1 0,435 → 0,530 (+0,095, el salto más grande del frente — mayor que
  todo lo logrado por ajustes de método juntos) · P1 0,760 → 0,812.
- Recuperaciones por familia: CHIMERA F1 0 → 0,91; MAZE, MEDUZALOCKER, WANNACRY salen de ~0.
- Los dos fallos informativos de la predicción: NOTPETYA y WASTEDLOCKER subieron SIN material
  nuevo (las familias interactúan: mejorar a las vecinas reduce la confusión) y JIGSAW BAJÓ
  de 0,67 a 0,51 al sumarle traducciones (evidencia causal del choque idioma-vs-cohesión —
  cierra la pregunta de la señal 4 sin experimento aparte).
- El límite honesto: las familias que quedan bajas (RYUK, HELLOKITTY, BLACKBASTA) lo están
  por cohesión o confusión estructural, no por falta de notas.

### Exp. 3e — Representación semántica multilingüe — ✅ MEDIDO (2026-08-21): NO SE ADOPTA
Base: 155 notas / 106 plantillas / 30 familias, P2, LinearSVC, 10 semillas pareadas.
Modelo: paraphrase-multilingual-MiniLM-L12-v2, troceado en ventanas de 126 tokens con
mean-pooling (133 de 155 notas superaban la ventana del modelo). Fuente:
`4_resultados/resultados_embeddings_155/`.
- Embedding en LUGAR de TF-IDF: macro-F1 0,4532 ± 0,0436, Δ pareado −0,0733
  [−0,1073; −0,0393] — empeora. Solo CHIMERA sube limpio (Δ +0,269: traducciones literales);
  GANDCRAB baja (pierde su señal de superficie).
- Embedding CONCATENADO con TF-IDF (L2 por bloque): macro-F1 0,5358 ± 0,0406, Δ +0,0092
  [−0,0098; +0,0283] — plano. Y RYUK, el CONTROL NEGATIVO, sube más que las tres familias
  objetivo (Δ +0,128 [+0,055; +0,200]) ⇒ el caso preregistrado «mejora genérica, mecanismo
  NO probado».
- **Veredicto por criterio preregistrado: no adoptar.** Cuarto resultado negativo de método
  del frente de notas (hiperparámetros, abstracción de marcadores, control P3, embeddings):
  el techo lo pone el dato también contra representaciones semánticas. Se escribe como cierre
  de la escalera de método, igual que los negativos del frente de archivos.

**La narrativa espejo que esto arma:** en archivos, la escalera fue «lo estadístico no
discrimina → las firmas explican dónde vive la señal → el ML la extrae completa». En notas
queda «el clasificador funciona con techo → la curva dice cuánto dato hace falta → la cohesión
explica qué familia rinde y el control descarta la trampa → la intervención guiada por todo lo
anterior funciona como se predijo». Las dos terminan igual: el límite es del dato, está
acotado, y se demostró.
