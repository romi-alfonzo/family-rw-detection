# ESTADO DE LA TESIS — documento vivo

> **Cómo usar este archivo:** cuando abras un chat nuevo y se haya perdido el contexto,
> dile a Claude: *"lee ESTADO_TESIS.md en mi carpeta Tesis"*. Con eso retoma todo.
> Mantener actualizado al final de cada sesión.

_Última actualización: 2026-08-04_

> **⚠️ La carpeta fue reorganizada el 2026-08-04.** Ver `LEEME_ESTRUCTURA.md` para el mapa.
> Rutas nuevas (las de este documento que digan lo viejo, traducir así):
> `2_codigo/` scripts · `3_datos/corpus_v2/` corpus · `4_resultados/resultados_*/` salidas ·
> `1_documento/Plantilla_de_Tesis___Romina_Carlos/` tesis LaTeX ·
> `5_bibliografia/` papers (incluye `Leido/`) · `6_notas_trabajo/` los .md de fuentes ·
> `7_compartido_carlos/Tesis Carlos y Romina/` (ahí está `Pruebas.xlsx`) ·
> `_archivo/` material superado (v1 del clasificador, latex_capitulos, Plantilla inicial).
> Nada fue borrado.

---

## 1. Identificación
- **Título:** "Detección de familias de ransomware en base a archivos encriptados y notas de rescate"
- **Autores:** Romina Alfonzo, Carlos Urdapilleta
- **Tutor:** Cristian Cappo — Universidad Nacional de Asunción (FP-UNA)
- **Área:** Ciberseguridad + Aprendizaje automático
- **Pregunta central:** ¿Se puede clasificar automáticamente la familia de ransomware
  usando únicamente artefactos post-ataque (archivos cifrados + notas de rescate),
  como alternativa basada en ML a herramientas como ID Ransomware?

## 2. Objetivos
1. Evaluar métricas estadísticas (entropía, Chi², Monte Carlo) para **detección binaria**
   de archivos cifrados.
2. Demostrar las **limitaciones de la clasificación multiclase** por propiedades
   estadísticas de archivos.
3. Construir un **corpus de notas de rescate** (30 familias, dataset NapierOne).
4. Implementar un **clasificador NLP** con TF-IDF + LinearSVC / LogReg / RandomForest / KNN.

## 3. Estado de resultados
- **Detección binaria (cifrado vs no):** funciona bien. ✅
- **Multiclase por estadística de archivos:** NO discrimina entre familias. ✅ (esperado; es un hallazgo, no un fallo)
- **Clasificador de notas (NLP):** mejor configuración = LinearSVC + char n-grams (3-5).

### Métricas reales (reevaluación 2026-06-22, corpus de 156 notas / 30 familias, StratifiedKFold)
| Métrica | Palabras(1-2) | Caracteres(3-5) | Combinado |
|---|---|---|---|
| Accuracy global | 0.859 | **0.897** | 0.897 |
| Balanced accuracy | 0.826 | **0.861** | 0.846 |
| F1 weighted (lo que se reportaba ~87.58%) | 0.841 | 0.882 | 0.879 |
| **F1 macro** (promedia familias por igual) | 0.802 | **0.840** | 0.823 |

> **Hallazgo clave:** el "87.58%" era F1 *weighted*, inflado por las familias grandes
> (CERBER=21, GANDCRAB/DHARMA=10). Bajo **F1 macro / balanced accuracy** (lo que pide un
> revisor para multiclase desbalanceada) el rendimiento real ronda **0.84–0.86**, que sigue
> siendo bueno. **Reportar SIEMPRE macro-F1 y balanced accuracy además de accuracy.**

### Familias que el modelo NO acierta (F1 = 0.0) — todas con 2-3 notas
- **CHIMERA** (2), **CRYPTOLOCKER** (3), **WANNACRY** (2).
- Esto justifica empíricamente la necesidad de expandir el corpus (lo que pidió el tutor).

## 4. Pendientes (pedidos del tutor: más fuentes, mejor resultado, dataset más grande)
- [ ] **Expandir corpus** de familias con pocas notas. Ver §6 sobre fuentes.
- [ ] **Optimizar hiperparámetros** (GridSearchCV/RandomizedSearch) en el servidor de la facultad.
- [ ] **Más bibliografía** — ver `bibliografia_fuentes_nuevas.md` (ya iniciado).
- [ ] **Mejorar análisis de archivos encriptados** (reproducir entropía/Chi²/Monte Carlo y documentar por qué la multiclase falla).
- [ ] Re-correr el clasificador tras expandir corpus, idealmente con min 5 notas/familia para usar 5-fold real.

## 5. Archivos relevantes en la carpeta
- `clasificador_notas_ransomware.py` — pipeline NLP principal.
- `ransom_notes_corpus/` — corpus actual: 156 notas, 30 familias (la usada en los experimentos).
- `ransomware_notes/` — repo grande: **209 familias / 345 archivos** (fuente para expandir).
- `family-rw-detection/` — código de análisis de archivos / features.
- `latex_capitulos/` — capítulos LaTeX (intro, marco teórico, metodología, resultados, conclusión).
- `resultados_nlp.csv` — métricas previas.
- `eval_macro.py` (en outputs) — script de reevaluación con métricas macro y reporte por familia.

## 6. Corpus — DECISIÓN TOMADA: solo las 30 familias de NapierOne
**Se usan únicamente las 30 familias de `ransom_notes_corpus` (las que necesita la tesis).
NO se agregan familias nuevas.** La expansión, de hacerse, es solo en *profundidad*
(más notas para esas 30 familias), nunca en cantidad de familias.

### Fuentes de notas disponibles (procedencia, para citar correctamente)
- `ransom_notes_corpus/` (156 notas, 30 familias) → base actual, dataset **NapierOne**.
- `ransomware_notes/` → repositorio público **ThreatLabz (Zscaler)**:
  https://github.com/ThreatLabz/ransomware_notes — **209 familias / 317 archivos**.
  Solo aporta ~41 notas extra para las 21 familias que coinciden con las tuyas.
  Fuente reputable (citar como ThreatLabz/Zscaler si se usa).

## 6.bis Corpus reconstruido — `corpus_v2/` (2026-06-24)
Corpus limpio reconstruido desde **archivos brutos** (nombre y extensión original) de los repos,
deduplicado, sin tocar `ransom_notes_corpus/` (original intacto).
- **146 notas, 30 familias** (ninguna vacía). Manifiesto: `manifiesto_corpus_v2.csv`.
- Composición: 96 brutos + 37 del corpus existente (único) + 13 transcripciones pcrisk.
- Extensiones reales preservadas: .txt(117), .hta(17), .html(9), .htm(2), .readme_to_restore(1).
- Cada nota etiquetada `bruto` / `corpus-existente` / `transcripcion` con extensión y fuente.
- 6 familias sin bruto público (BadRabbit, Chimera, Jigsaw, NotPetya, WannaCry, CryptoLocker) → cubiertas con corpus existente/transcripción.
- Pendiente: adaptar el clasificador para usar `corpus_v2` con `extractor_notas.py` (multiformato) y añadir nombre/extensión como features; reentrenar midiendo macro-F1.

## 6.ter Sesión 2026-07-27 — Diagnóstico profundo + corrida canónica v2
> Detalle completo en `DIAGNOSTICO_2026-07-27.md` (leerlo junto con este archivo).

**Correcciones de registro (anulan lo dicho arriba donde contradigan):**
- El "87,58 %" histórico era **ACCURACY**, no F1 weighted (el F1 weighted era 85,61 %).
- **NapierOne es de Davies, Macfarlane & Buchanan (2022)**, no de Pont; y es un dataset
  de archivos mixtos, NO de notas. Solo 37/146 notas de corpus_v2 llevan la etiqueta
  ambigua "NapierOne/varios" → auditar procedencia antes de citarlo como fuente del corpus.
- La tesis de Pont NO clasifica notas. **Benchmark directo = Lemmou et al. 2021**
  (*Computers* 10(11):145, PDF en `Leido\` y en `Tesis Carlos y Romina\Papers\`).
- **CORRECCIÓN (2026-07-28, lectura completa de Lemmou):** Lemmou et al. SÍ identifican
  la familia a partir de la nota — NO afirmar que "nadie lo hizo". Su método: prototipo
  BASADO EN REGLAS/MARCADORES (extracción de emails, direcciones Bitcoin/Bitmessage,
  URLs onion, nombres de familia y keywords) + LSA como búsqueda de casi-duplicados
  (umbral 0,99995) contra su base de 176 notas / 62 familias → 181/182 identificadas
  (mundo cerrado, sin train/test). Su ML es SOLO binario (nombre de nota vs benigno,
  RF 98,32%). Lo que NO hacen: clasificador ML supervisado multiclase sobre contenido,
  ni medición de generalización a variantes no vistas, ni métricas macro. **La novedad
  de la tesis se reformula así:** primer clasificador supervisado multiclase por
  contenido con evaluación de generalización (P1/P2) y macro-F1 — complementario al
  identificador por marcadores de Lemmou (que exige base curada de IOCs actualizada).
  Dato extra utilizable: en el set de Lemmou, ID-Ransomware acierta 158/182 (86,8%).
  Sus 8 falsos positivos inter-familia (CryptoLocker↔TeslaCrypt↔AlphaCrypt,
  Rapid↔StorageCrypt) predicen las familias difíciles de esta tesis.
- Las cifras 16,67 / 66,67 / 71,93 % son **experimentos propios** (están en
  `Tesis Carlos y Romina\Pruebas.xlsx`), no bibliografía: redactarlas como experimento propio.
- El 0,897/0,840 del corpus original estaba inflado: 24 % de duplicados (156 notas → 118
  únicas), fuga de vocabulario TF-IDF, bug UTF-16 y CV real de 2 folds con 1 semilla.

**Arreglos hechos (código):**
- `extractor_notas.py`: ahora detecta UTF-16 (BOM/heurística de NULs) y cp1252.
  Antes, 13/146 notas de corpus_v2 (DHARMA, GANDCRAB, SUNCRYPT) entraban ilegibles.
- `beautifulsoup4` instalado (el clasificador v1 ni arrancaba sin él).
- **`clasificador_notas_v2.py` = script canónico** (v1 intacto como registro histórico):
  TF-IDF dentro de Pipeline (sin fuga), casi-duplicados agrupados (coseno char >0,90 +
  StratifiedGroupKFold), 10 semillas, 2 folds declarados, macro-F1 + balanced accuracy +
  reporte por familia + matriz de confusión. Correr con: `python clasificador_notas_v2.py`.

**Resultados canónicos (corpus_v2: 146 notas, 30 familias, 95 grupos de contenido):**
| Protocolo (pregunta) | Mejor config | Acc | Bal.acc | Macro-F1 |
|---|---|---|---|---|
| P1 "plantilla conocida" (estratificado) | char+LinearSVC | 0,818 | 0,777 | **0,760 ± 0,029** |
| P2 "variante nunca vista" (grupos) | comb+LinearSVC | 0,551 | 0,492 | **0,435 ± 0,057** |

- **Hallazgo clave: las notas son PLANTILLAS.** 146 notas = solo 95 contenidos distintos
  (DHARMA 19→6 plantillas; WASTEDLOCKER 4→1, inevaluable en P2). Esto ES el análisis de
  variabilidad que pidió el tutor (02/05/24) y justifica expandir el corpus en profundidad.
- P1 es el escenario comparable con ID Ransomware (71,93 % con notas): el nuestro da 82 %.
- Salidas y trazabilidad: `resultados_canonicos\` (resumen CSV, por-familia CSV,
  `fig_confusion_canonica.png`, `grupos_neardup.csv`, `manifiesto_corrida.json`).
  **Regla: ninguna cifra tipeada a mano — toda tabla se regenera de estos CSV.**

**HECHO 2026-07-28 (Fase 0 completa):** capítulo 4 real reconstruido en el documento vivo
(`Plantilla_de_Tesis___Romina_Carlos\resultados.tex`): pruebas preliminares + Exp. 1 (cifras
reales de exp1_binaria.csv, no las suavizadas) + Exp. 2 (9,9 %, corregido el falso "15,4 %")
+ Exp. 3 con protocolos P1/P2, tabla por familia, figura de confusión y tabla-escalera de
transparencia + evaluación de herramientas (Pruebas.xlsx redactada como experimento propio)
+ comparación + discusión. Metodología depurada (resultados extraídos; corpus_v2 146 notas;
métricas macro; protocolos P1/P2; párrafo del weighted corregido; metodología de herramientas).
Apéndice: binaria por familia + reproducibilidad. Figuras insertadas: entropía por familia y
confusión canónica. Compila limpio: 51 páginas, 0 referencias indefinidas.
Backups de los archivos previos en `backup_pre_cap4\`.

**HECHO 2026-07-28 (b):** preparado el trabajo para el servidor de la facultad:
`gridsearch_notas.py` (búsqueda de hiperparámetros ANIDADA: GridSearch dentro del fold de
entrenamiento, protocolos P1/P2, 10 semillas, scoring macro-F1; smoke test local OK, ya
muestra mejora: P2 0,459 vs 0,435 / P1 0,771 vs 0,760 con grilla mínima) +
`SERVIDOR_INSTRUCCIONES.md` (Trabajo A = hiperparámetros; Trabajo B = advanced_features
para blindar Exp. 2; nota: sklearn no usa GPU, aprovecha núcleos). Decisión de trabajo:
**en el documento solo AGREGAR contenido; el pulido fino queda para el final.**

**HECHO 2026-07-28 (c) — Plan del frente "archivos encriptados" (Objetivo 2):**
1. Blindar el negativo: `advanced_features.py` + `train_advanced.py` en servidor (Trabajo B).
2. **Experimento 2b NUEVO**: `deteccion_estructural.py` (Trabajo C, probado local) —
   descubre magic bytes/extensiones por familia automáticamente y clasifica por LOO.
   Reformula el Obj. 2: "la estadística no discrimina (9,9 %) pero los artefactos
   estructurales deliberados sí (subconjunto de familias)" = pedido del tutor 11/01/25
   + validación de las 9 firmas SI* de Pruebas.xlsx.
3. Al volver del servidor: AGREGAR al cap. 4 la sección de features avanzadas + la
   sección Experimento 2b + hiperparámetros (Trabajo A). Solo agregar; pulir al final.

**Fuente clave leída 2026-07-28 — "Majority Voting Approach to Ransomware Detection"
(carpeta reunion 02-05-2024):** es Davies, Macfarlane & Buchanan 2023 (arXiv 2305.18852,
el MISMO grupo de NapierOne, mismas 30/31 familias). Es la entrada corrupta `pont2023`
del bib viejo (autoría real = Davies). Detección BINARIA por votación de 23 tests
(0,9989 combinada) — tampoco clasifica familia. Usos para la tesis: (1) ancla del Exp. 1
(su test de entropía da 0,865 vs nuestro RF 0,886); (2) valida el Exp. 2b estructural
(su Magic Number Test, 0,961) y el uso de χ² sobre Shannon; (3) CITA DE ORO: propone
como mejora futura "aplicar NLP sobre los strings de notas" = el hueco que esta tesis
llena, dicho por el grupo de NapierOne en 2023; (4) patrón de votación citable para el
pipeline secuencial propuesto; (5) su ref [77] (Yamany 2022, Electronics) = familia por
features estáticas del ejecutable, related work pendiente de descargar. Incorporar al
.bib como davies2023majority en la pasada de bibliografía.

**Fuente leída 2026-07-28 — Thesis.pdf (reunion 02-05-2024) = Trujillo, Kim Kip (2022),
tesis de máster UPC "Ransomware note detection techniques using supervised ML"** (es el
"khammas2023" mal atribuido del bib viejo). BINARIA nota-vs-no-nota: 59 notas de Lemmou
(.txt) + 59 de 20_newsgroups, DT+SVM, bastan ~20 features; validación TEMPORAL (entrena
con notas ≤jul-2019, valida con 10 notas posteriores): DT ~95% acc / 100% prec / 90%
recall. Usos: (1) fila de related work (binaria por contenido); (2) su validación temporal
= idea de protocolo P3 citable como trabajo futuro; (3) su future work (extracción
HTML/RTF, multilenguaje, truncado) es lo que nuestro extractor YA hace → avance explícito
sobre el antecedente; (4) citas de motivación: atribución por notas es práctica manual
(Ryuk→Conti se estableció analizando notas; DarkSide/REvil comparten plantilla — ¡explica
confusiones inter-familia!). Citar como trujillo2022 (UPC, dir. René Serral).

## ⚠️ INCIDENTE 2026-08-04 — Windows Defender borró 2 notas del corpus

Al copiar el corpus para subirlo al cluster, **Windows Defender puso en cuarentena**
`3_datos/corpus_v2/DHARMA/Info__13.hta` y `DHARMA/Info__3.hta` (notas `.hta` reales de
ransomware = ejecutables HTML; Defender las trata como amenaza). También bloqueó sus
fuentes en `3_datos/fuentes_notas/RansomNoteFiles/Dharma/{abibo,cmb}/Info.hta`.

**Estado:** el corpus tiene **144 de 146 notas** (DHARMA pasó de 19 a 17). Las otras 144
están intactas. Quedan **15 `.hta` en riesgo** (9 DHARMA + 6 CERBER).

**Consecuencia metodológica a tener presente:** la corrida canónica de la PC fue sobre
**146** notas; lo que corra en el cluster será sobre **144**. NO mezclar los números.
Al restaurar las 2 notas, re-correr `clasificador_notas_v2.py` para tener todo sobre la
misma base y actualizar el capítulo 4.

**PENDIENTE (Romina, manual — son ajustes de seguridad del sistema):**
1. Seguridad de Windows → Protección antivirus → Historial de protección → **Restaurar**
   las detecciones de `Info__13.hta` / `Info__3.hta` del 2026-08-04.
2. Agregar **exclusión de carpeta** para `C:\Users\Romina\Tesis\3_datos` — sin esto,
   Defender va a seguir borrando notas cada vez que se copien.
3. Avisar para verificar integridad contra el manifiesto y re-correr la canónica.

Alternativa de recuperación si la cuarentena falla: el contenido está en el historial git
de `3_datos/fuentes_notas/RansomNoteFiles/.git` (se puede extraer el blob sin escribir un
`.hta` en disco, guardándolo con otra extensión y anotando la original en el manifiesto).

## RESULTADO NUEVO 2026-08-04 — Experimento 2b estructural (cluster, 29 familias, 1450 archivos)

Sobre los MISMOS archivos cifrados de NapierOne-small:
| Enfoque | Exactitud multiclase |
|---|---|
| Propiedades estadísticas (entropía, χ², Monte Carlo) — Exp. 2 | **9,9 %** |
| Artefactos estructurales (magic bytes + extensión) — Exp. 2b | **86,2 %** (1250/1450), cobertura 87,8 % |

- **25 de 29 familias** dejan marca estructural detectable automáticamente.
- **WANNACRY: prefijo `57414e4143525921...` = "WANACRY!"** → validación independiente de las
  9 familias marcadas «SI*» en `Pruebas.xlsx`.
- **4 familias sin marca**, entre ellas **NOTPETYA** → convergencia con Davies et al. (2023),
  que documenta explícitamente que NotPetya no modifica la extensión de los archivos. Citable.
- Firmas binarias largas encontradas: CERBER/LOCKBIT/RANSOMEXX/TESLACRYPT (64 B),
  MEDUZALOCKER (21 B), GANDCRAB (19 B), CUBA (prefijo 20 B "FIDEL.CA"), PHOBOS (7 B "LOCK96").

**Reformulación del Objetivo 2 (dos caras):** la estadística del cifrado no discrimina familias
(9,9 %), pero los artefactos que el ransomware inserta deliberadamente sí (86,2 %) — que es
exactamente el mecanismo por el que ID Ransomware identifica 20/30 familias por archivo.

### ABLACIÓN (corrida 2026-08-04, job 3540) — matiza el 86,2 %

| Modo | Cobertura | Exactitud global | Exactitud **entre los archivos cubiertos** |
|---|---|---|---|
| Solo extensión | 82,8 % (1200/1450) | 82,8 % | **100 %** (1200/1200) |
| Solo firmas binarias | 53,3 % (773/1450) | 51,7 % | **97,0 %** (750/773) |
| Combinado | 87,8 % (1273/1450) | 86,2 % | 98,2 % (1250/1273) |

**Interpretación honesta (así debe ir a la tesis, NO como "86 % de identificación"):**
- La **extensión explica casi todo** el resultado global (82,8 % de 86,2 %) y acierta el **100 %**
  cuando está presente. Eso es señal de artefacto del dataset: en NapierOne cada familia tiene
  UNA extensión constante (DHARMA `.iq20`, cuando en la práctica usa extensiones con ID de
  víctima). Mide identificación de **campaña**, es lo mismo que hace una tabla de reglas tipo
  ID Ransomware, y no sobreviviría a un cambio de extensión.
- Las **firmas binarias son el hallazgo defendible**: cubren la mitad del corpus (53,3 %) pero
  ahí identifican al **97 %**. Son marcadores que el propio ransomware escribe en el archivo
  para reconocer lo que ya cifró, por lo que son más estables entre campañas que la extensión.
- **4 familias no dejan nada:** BADRABBIT, JIGSAW, NOTPETYA, SUNCRYPT. NotPetya coincide con
  Davies et al. (2023), que documenta que no modifica la extensión. Citable.

### NARRATIVA UNIFICADORA DE LA TESIS (surge de comparar los dos frentes)

Los dos experimentos muestran **la misma lección** desde artefactos distintos: la señal fácil es
la identidad de la *campaña*; la señal robusta está en el *contenido*.

| Frente | Señal "fácil" (identidad de campaña/plantilla) | Señal robusta (contenido) |
|---|---|---|
| **Notas** | P1 plantilla conocida: macro-F1 0,760 | P2 variante nueva: macro-F1 0,435 |
| **Archivos** | Extensión: 82,8 % (100 % donde aplica) | Firmas binarias: 53,3 % cobertura, 97 % ahí |

Escribir esta simetría explícitamente en la discusión: es el aporte conceptual del trabajo y
explica por qué las herramientas basadas en reglas funcionan bien hasta que la campaña cambia.

## Lecciones del cluster NIDTEC (2026-08-04/05) — para metodología y futuras corridas

Tres trabajos fallaron por la misma causa raíz y quedaron corregidos. Vale documentarlo en
la sección de infraestructura de la tesis (y respalda el agradecimiento obligatorio al NIDTEC):

1. **`DefMemPerNode=2048`**: el cluster asigna 2 GB por defecto y hay que pedir memoria
   explícitamente con `#SBATCH --mem=` (máximo 64 GB). Los 5 scripts ya la piden.
2. **`n_jobs=-1` es peligroso en cluster compartido**: toma los 32 núcleos del nodo en vez de
   los asignados por SLURM. Todos los scripts leen ahora `SLURM_CPUS_PER_TASK`.
3. **Paralelismo anidado** en `train_advanced.py`: `cross_val_score(n_jobs=-1)` con modelos que
   también pedían `n_jobs=-1` → 32 procesos × 32 hilos, cada proceso con su copia de la matriz
   (29.029 × 275). Corregido: paralelismo solo en la CV, modelos con 1 hilo.
4. **`GradientBoosting` es inviable con 29 clases** (entrena n_clases × n_estimators = 2.900
   árboles por ajuste). Sustituido por **`HistGradientBoostingClassifier`**, que scikit-learn
   recomienda para n > 10.000. **Declararlo en la tesis** (cambio de modelo respecto del Exp. 2).
5. **Salida con buffer**: sin `python3.11 -u` los trabajos largos no muestran avance en el
   archivo de SLURM. Agregado en los 5 scripts, más avisos de progreso con tiempos.
6. Medición de costo (3.000 muestras, 275 features, 29 clases, 1 hilo): RF 68 s/ajuste,
   HistGB 126 s/ajuste, KNN despreciable. Sobre 29.029 muestras el entrenamiento completo
   estima **3-5 h**. La extracción de features ya tomó **5,5 h** y su CSV está guardado
   (`advanced_features.csv`) — hay un `job_train_advanced.sh` que solo entrena, para no repetirla.

## EXPERIMENTO 2c (nuevo, 2026-08-05) — ML sobre bytes de cabecera/cola

**Decisión de Romina:** no gastar cómputo en volver a demostrar que la estadística falla
(ya está demostrado); invertirlo en la vía que SÍ puede servir. Se descartó un gridsearch
sobre las features estadísticas y se creó en su lugar `2_codigo/clasificador_bytes.py`.

**Qué hace:** en vez de reglas de coincidencia exacta (Exp. 2b, que solo cubre el 53 % de
los archivos), entrena un clasificador sobre los **512 bytes de cabecera + 512 de cola**.
Cobertura 100 % y tolera variabilidad. **NO usa nombre ni extensión** — deliberado, porque
en el Exp. 2b la extensión aportaba 82,8 % pero es un identificador de campaña.

**Representaciones comparadas** (análogas a las del clasificador de notas):
- `posicional + RandomForest`: bytes en posiciones fijas; árboles parten por valor exacto.
- `n-gramas de bytes + LinearSVC` y `+ LogReg`: TF-IDF de n-gramas de bytes = el análogo
  directo de los n-gramas de caracteres de las notas. Captura marcas en posición variable.

**Error de diseño detectado y corregido antes de gastar cluster:** un modelo lineal sobre
bytes posicionales crudos es conceptualmente inválido (los valores de byte son categóricos,
no ordinales) y además 1.016 de 1.024 posiciones son ruido que diluye la señal. Verificado
con datos sintéticos: daba 0,24 donde debía dar ~1,0. Se eliminó esa configuración.

**Validación con datos sintéticos:** familias con firma (`WANACRY!`, `FIDEL.CA`, `LOCK96`)
→ F1 0,94-0,95; la familia sin firma se identifica por eliminación. El método funciona.

**Ejecución:** `job_bytes.sh` (8 núcleos, 32 GB, ~1-2 h). Lanzar con
`DATOS=/scratch/ralfonzo/Napierone-small sbatch --export=ALL,DATOS job_bytes.sh`.
Salidas: `4_resultados/resultados_bytes/` (resumen, por familia, manifiesto).

**Limitación a declarar (la misma de siempre):** NapierOne tiene una campaña por familia,
así que un acierto alto mide identificación de esa campaña; la generalización a campañas
nuevas no es evaluable con este dataset. Es el análogo de las plantillas en las notas.

## ⚠️ RESULTADO QUE OBLIGA A REVISAR EL OBJETIVO 2 (2026-08-05, job 3547)

**El "9,9 %" ya NO es el resultado del Experimento 2.** Con 29.029 archivos (1001/familia,
29 familias) y las 275 características avanzadas:

| Conjunto de características | RandomForest | KNN-5 | HistGB |
|---|---|---|---|
| Entropía + tamaño (2) — el baseline original | 0,166 | 0,126 | 0,163 |
| Estadísticas globales (9) | 0,391 | 0,300 | 0,399 |
| Frecuencia de bytes (256) | 0,184 | 0,082 | 0,199 |
| Todas (275) | 0,542 | 0,143 | 0,592 |
| **Estadísticas + derivadas (19)** | **0,603** | 0,299 | 0,598 |

Azar = 0,034. Selección Top-K con ANOVA: 0,363-0,387 (peor que las 19 elegidas a mano).

**LA AFIRMACIÓN "las propiedades estadísticas no discriminan familias" ES FALSA tal como
está escrita y hay que reformularla.** Lo correcto:
- La **entropía global del archivo completo** casi no discrimina (0,166 con 2 features).
- Las **estadísticas REGIONALES y de estructura** sí discriminan bastante (0,603 con 19).

**Por qué, y acá está lo bueno:** las características más importantes según el Random Forest
son estructurales, no criptográficas —
`longest_run` (0,072), `entropy_diff_hf` (0,041, diferencia de entropía cabecera vs cola),
`entropy_footer` (0,040), `entropy_header` (0,031), `zero_ratio`, `byte_freq_000`,
`block_entropy_std`. Todas miden **la presencia de datos NO aleatorios añadidos al archivo**,
es decir, exactamente las marcas del Experimento 2b, medidas de forma estadística.

**Convergencia con el Exp. 2b (validación cruzada entre experimentos):** el reporte por
familia es bimodal y coincide con quién deja marca. Con firma binaria en 2b → F1 alto acá:
TESLACRYPT 1,00 · CERBER 0,98 · CUBA 0,98 · PHOBOS 0,89 · RANSOMEXX 0,84 · GANDCRAB 0,80 ·
CONTI 0,78 · MAZE 0,78. Sin marca en 2b → F1 bajo acá: JIGSAW 0,06 · BADRABBIT 0,15 ·
NOTPETYA 0,24. **Excepción interesante: SUNCRYPT 0,68 sin tener firma exacta** → el
aprendizaje encuentra patrones parciales que la regla de coincidencia exacta se pierde
(justifica el Exp. 2c).

**Otro hallazgo:** las 275 features (0,592) rinden PEOR que 19 bien elegidas (0,603). Las 256
frecuencias de byte agregan ruido — maldición de la dimensionalidad. Reportarlo.

**Nueva formulación del Objetivo 2 para la tesis:** no "la estadística no sirve", sino *"la
información que permite distinguir familias en los archivos cifrados no está en las
propiedades criptográficas del contenido (entropía global, χ², Monte Carlo sobre el archivo
completo) sino en la estructura: en los artefactos no aleatorios que cada familia añade.
Medida globalmente, la aleatoriedad es indistinguible (0,166); medida por regiones y rachas,
alcanza 0,603; y localizada explícitamente como firmas, 0,97 donde aplica."*

## Gridsearch de notas (job 3548, 1 h 57 min, 144 notas)

| Protocolo | Mejor config individual | Combinado con mejores hiperparámetros |
|---|---|---|
| P2 (grupos, variante nueva) | caracteres+LinearSVC 0,416 ± 0,060 | **0,424 ± 0,064** |
| P1 (estratificado, plantilla conocida) | palabras+LinearSVC 0,754 ± 0,029 | **0,775 ± 0,032** |

**Hallazgo: el ajuste de hiperparámetros NO mejora de forma significativa** (referencia sin
ajustar sobre 144 notas: P1 ≈ 0,783 / P2 ≈ 0,427 en la corrida mínima). Las diferencias caen
dentro del desvío entre semillas. **Conclusión para la tesis:** la configuración por defecto
ya era adecuada y **el límite no está en los hiperparámetros sino en la diversidad de
plantillas del corpus** — refuerza cuantitativamente la necesidad de expandirlo en
profundidad. Es un resultado negativo útil: cierra la objeción "¿probaron ajustar el modelo?".

## EXPERIMENTO 2c — RESULTADO (job 3557, 2026-08-05)

| Configuración | n | Exactitud | macro-F1 |
|---|---|---|---|
| posicional + RandomForest (búsqueda anidada) | 5.800 | 0,897 | 0,897 |
| n-gramas de bytes + LogReg | 5.800 | 0,856 | 0,858 |
| n-gramas de bytes + LinearSVC | 5.800 | 0,849 | 0,846 |
| **FINAL: posicional + RandomForest** | **14.500** | **0,910** | **0,908** |

Hiperparámetros elegidos: `n_estimators=300, max_depth=20, min_samples_leaf=2,
max_features=0.3`. **Sin usar nombre ni extensión: solo 512 bytes de cabecera + 512 de cola.**

**Gana la representación posicional sobre los n-gramas** ⇒ las marcas están en **offsets
fijos**, no dispersas. Dato metodológico: contrasta con las notas, donde los n-gramas de
caracteres son los que mejor funcionan.

### El hallazgo central: 23 de 29 familias se identifican casi perfectamente

**F1 ≥ 0,98 (23 familias):** AVOSLOCKER, BADRABBIT, BLACKCAT, BLACKMATTER, CERBER, CHIMERA,
CLOP, CONTI, CUBA, DHARMA, GANDCRAB, HELLOKITTY, LOCKBIT, LORENZ, MAZE, MEDUZALOCKER,
NETWALKER, PHOBOS, RANSOMEXX, RYUK, SODINOKIBI, TESLACRYPT, WANNACRY.

**Difíciles (6):** SUNCRYPT 0,75 · WASTEDLOCKER 0,64 · CRYPTOLOCKER 0,61 · DARKSIDE 0,60 ·
JIGSAW 0,44 · NOTPETYA 0,38.

### Convergencia perfecta con el Exp. 2b (validación cruzada entre experimentos)

- Las **15 familias con firma binaria** en 2b → todas **F1 ≥ 0,99** acá. Sin excepción.
- De las **10 que solo tenían extensión** (sin marca en el contenido), **7 igual dan ≥0,99**
  (AVOSLOCKER, BLACKMATTER, CHIMERA, CLOP, DHARMA, RYUK, SODINOKIBI) ⇒ **el aprendizaje
  encuentra patrones de contenido que la regla de coincidencia exacta no detecta.**
- De las **4 sin marca alguna** en 2b: **BADRABBIT pasa de 0,15 a 0,98** (resuelta);
  SUNCRYPT 0,68→0,75; NOTPETYA 0,24→0,38; JIGSAW 0,06→0,44.

**DARKSIDE actúa de "imán"**: precisión 0,45 con recall 0,90, o sea absorbe los archivos
ambiguos de las otras familias difíciles. Las 6 difíciles forman un grupo de confusión mutua:
son las que **no marcan sus archivos** y cuyo cifrado sí es genuinamente indistinguible.

### Conclusión definitiva del frente de archivos (progresión para el cap. 4)

| Enfoque | Exactitud | Cobertura | Usa metadatos |
|---|---|---|---|
| Entropía global + tamaño | 0,166 | 100 % | no |
| 19 estadísticas regionales | 0,603 | 100 % | no |
| Firmas binarias exactas (2b) | 0,517 (0,97 donde aplica) | 53 % | no |
| Extensión del archivo (2b) | 0,828 | 83 % | **sí (identifica campaña)** |
| **ML sobre bytes (2c)** | **0,910** | **100 %** | **no** |

**El resultado negativo original queda acotado a 6 familias**, no a las 30: la
indistinguibilidad estadística es real solo para las familias que cifran sin dejar
estructura. Para las otras 23 la información está en el contenido y es extraíble.

⚠️ **Limitación que sigue vigente y hay que declarar:** NapierOne representa cada familia con
una sola campaña; el 0,910 mide identificación de esa campaña. La generalización a campañas
nuevas de la misma familia no es evaluable con este dataset (mismo fenómeno que las
plantillas en las notas). Es la limitación más importante a escribir en la tesis.

## CAPÍTULO 4 ESCRITO COMPLETO (2026-08-05)

La tesis pasó de 51 a **62 páginas**, 5 figuras, compila sin referencias indefinidas.
Estructura nueva del capítulo 4 (§4.1 a §4.11):
- §4.3.1 Ampliación del espacio de características (275 → 0,603) + §4.3.2 Dónde reside la
  información discriminante (tabla de importancias; reformulación del Objetivo 2)
- §4.4 Experimento 2b (motivación, marcas descubiertas, ablación)
- §4.5 Experimento 2c (diseño, resultados, análisis por familia, limitación)
- §4.6 Síntesis del frente de archivos (figura de progresión)
- §4.7.5 Optimización de hiperparámetros de notas (el negativo útil)
- §4.8 **Ajustes al protocolo experimental** — sección narrativa que documenta honestamente
  las 6 correcciones: codificación, duplicados, fuga en vectorización, semillas, métricas e
  infraestructura compartida (SLURM). Pedida expresamente por Romina.
- §4.10 Comparación actualizada · §4.11 Discusión con la simetría entre frentes

Figuras nuevas en `1_documento/.../images/` (generadas por `2_codigo/generar_figuras_cap4.py`,
reejecutable): `fig_progresion_archivos.png`, `fig_f1_por_familia_bytes.png`,
`fig_simetria_frentes.png`.

Bibliografía: agregada `davies2023majority` (Davies, Macfarlane & Buchanan 2023, arXiv
2305.18852) — se cita para la prueba de magic number y para la convergencia sobre NotPetya.

**Pendiente de la pasada final (Bloque E):** la conclusión sigue SIN tocar (cifras superadas
100 % y 15,4 %) por decisión de Romina; front matter; y verificar que el §4.1/§4.2 preliminar
no contradiga la reformulación del Objetivo 2.

## SPRINT 1.1 EJECUTADO (2026-08-05) — Normalización de marcadores: NO mejora

`2_codigo/normalizacion_marcadores.py`. Se sustituyeron los marcadores variables por
etiquetas de tipo (`[EMAIL]`, `[ONION]`, `[BTC]`, `[URL]`, `[ID]`, `[CLAVE]`): 130/144 notas
modificadas, 1.281 sustituciones.

| Protocolo | palabras | caracteres | combinado |
|---|---|---|---|
| P2 original → normalizado | 0,414 → 0,414 | 0,409 → **0,391** | 0,421 → 0,410 |
| P1 original → normalizado | 0,747 → 0,746 | 0,750 → 0,741 | 0,756 → 0,751 |

Diferencia media: **P2 −0,010 · P1 −0,005**. Todo dentro del desvío entre semillas
(0,03-0,06) ⇒ **sin efecto**; si acaso, leve perjuicio en la vista de caracteres.

**Interpretación:** bajo P2 el modelo ya no podía usar esos valores (la plantilla de prueba
tiene otros), así que quitarlos no le saca una muleta. Que la vista de CARACTERES sea la que
más pierde sugiere que los n-gramas extraían señal de la *forma* de los marcadores (longitud
de una .onion, formato de URL) y la etiqueta uniforme la destruye.

**Conclusión para la tesis:** la variabilidad entre plantillas de una misma familia es
ESTRUCTURAL, no se reduce a datos de contacto, y ningún preprocesamiento la resuelve. Junto
con el resultado de hiperparámetros, son **dos resultados negativos independientes** que
apuntan a lo mismo: el límite es la cantidad de plantillas del corpus. Escribirlo en el
cap. 4 (subsección junto a §4.7.5) — documenta que se intentó la corrección obvia.

## SPRINT 2 EN EJECUCIÓN (lanzado 2026-08-05)

Dos trabajos en el clúster, pendientes de resultado:
- `job_analisis_bytes.sh` → `analisis_bytes.py` (40-70 min). Cuatro análisis:
  **(a) generalización a tipos de archivo nunca vistos ← EL CRÍTICO**, puede confirmar o
  matizar el 0,910; (b) importancia por posición de byte + figura; (c) ablación de ventana
  (64/128/256/512, solo cabecera, solo cola); (d) diagnóstico de las 6 familias difíciles.
- `job_gridsearch_estadisticas.sh` → `gridsearch_estadisticas.py` (~30 min). Cierra el
  ÚNICO hueco de optimización declarado (§4.3.1): búsqueda anidada sobre las
  características estadísticas. Referencias: 0,603 sin ajustar · 0,910 del Exp. 2c.

**Qué hacer al volver:** si (a) mantiene el rendimiento (caída < 0,10), el resultado
principal queda confirmado y se agrega como subsección de validación en §4.5. Si cae más,
hay que matizar §4.5 y §4.6 antes de la reunión con Cappo.

Salidas esperadas en `4_resultados/resultados_analisis_bytes/` y
`4_resultados/resultados_gridsearch_estadisticas/`.

## PLAN DE MEJORAS (2026-08-05) → ver `PLAN_MEJORAS.md`

Cinco sprints, frentes separados. Resumen:
1. **Sin cluster, ya:** abstracción de marcadores en notas (Claude) + restaurar las 2 notas
   en cuarentena (Romina).
2. **Una tanda de cluster (~1 h):** hiperparámetros de las características estadísticas
   (único hueco de optimización que queda) + análisis de robustez del clasificador de bytes
   (generalización a tipos de archivo no vistos ← el crítico; importancia por offset;
   ablación de ventana; diagnóstico de las 6 difíciles).
3. **Manual de Romina, en paralelo:** ampliar corpus a ≥3 plantillas/familia (URLs listas) +
   auditar las 37 notas de procedencia ambigua.
4. Nombre de archivo genuino como feature + re-correr todo sobre la base final.
5. Cierre: actualizar cap. 4, reunión con Cappo, y Bloque E (conclusión, front matter).

**Estado de hiperparámetros (para no volver a dudar):** notas ✅ hecho (job 3548) ·
bytes/Exp. 2c ✅ hecho (anidado interno) · **características estadísticas ❌ pendiente**
(declarado como limitación en §4.3.1; ahora es barato: 28 s por configuración).

## HOJA DE RUTA (fijada 2026-08-04)

**Bloque A — Cluster NIDTEC (Romina, en paralelo a todo):** ✅ acceso concedido 2026-08-04
(usuario `ralfonzo`, master `arandu`, nodos c1-c4) y **NapierOne-small ya está en el cluster**.
Seguir **`SERVIDOR_PASOS_AHORA.md`** (guía específica del cluster; `SERVIDOR_INSTRUCCIONES.md`
queda como referencia conceptual de los 3 trabajos).
Datos del entorno: SLURM (`sbatch`, scripts listos en `2_codigo/slurm/`), usar `python3.11`
y `pip3.11`, trabajar en `/scratch/ralfonzo` (el HOME no tiene espacio), **sin acceso a
Internet** (el código ya funciona sin `beautifulsoup4`, con fallback regex verificado),
GPU disponible (no la usa sklearn), almacenamiento temporal (se borra 60 días después).
⚠️ Faltan en el dataset del cluster: `BLACKBASTA-small` y `Z-Safe` (benignos) → hay 29 de
30 familias; el Exp. 2 corre igual con 29, declarándolo. Preguntar si están en otro lado.
⚠️ **OBLIGACIÓN DEL REGLAMENTO:** mencionar el uso del cluster del NIDTEC en la tesis
(proyecto LABO16-167, CONACYT/PROCIENCIA, FPUNA) → va en agradecimientos, Bloque E.

**Bloque B — Corpus (antes de re-correr nada):**
1. Descargar las notas pcrisk ya listadas en `6_notas_trabajo/mas_notas_descarga.md`
   (14 familias con URL identificada) → objetivo: ≥5 notas Y ≥3 plantillas distintas por familia.
2. Auditar las 37 notas "NapierOne/varios" del manifiesto (pista: repo kipziptie).
3. Verificar BTC/claves de WannaCry/NotPetya/BadRabbit contra imágenes originales.
4. Re-correr `clasificador_notas_v2.py` → mejora esperada en P2 + habilita 5-fold.

**Bloque C — Documento, solo AGREGAR (con o sin servidor):**
1. Bibliografía: corregir lee2022 (verificar DOI real), agregar davies2022napierone,
   davies2022entropy, davies2023majority, trujillo2022, gomez2023 (pedido del tutor),
   pont2023 (tesis), sokolova2009 (métricas). Remapear novedad en §2.6 (formulación
   precisa vs Lemmou — ya redactada en la corrección del 2026-07-28).
2. Al volver el servidor: agregar secciones hiperparámetros + features avanzadas + Exp. 2b.

**REGLA (Romina, 2026-08-04): la conclusión NO se toca hasta el final de la tesis.**
Se redacta completa en el Bloque E, cuando todos los resultados estén cerrados.
(Ojo al llegar ahí: la versión actual cita cifras superadas — 100 % binaria y 15,4 %
multiclase — que NO deben sobrevivir a la reescritura final.)

**Bloque D — Reunión con Cappo (cuando A+B estén):** mostrar cap. 4 nuevo, hallazgo de
plantillas (su pedido de variabilidad del 02/05/24 respondido), P1/P2, comparación vs
ID Ransomware con Pruebas.xlsx como experimento propio (su pedido del 08/05/24).
Preguntarle: (a) ¿acepta notas por OCR/transcripción como fuente? (pendiente de
notas_familias_criticas.md); (b) ¿mapear objetivos específicos a capítulos? (su pedido
del 08/05/24); (c) ¿experimento transformer con GPU como sección extra o trabajo futuro?

**Bloque E — Cierre final (AL FINAL, una sola pasada):** **CONCLUSIÓN completa** (recién acá;
corregir las cifras superadas 100 %/15,4 %), resumen/abstract, carátulas, dedicatoria,
agradecimientos, lista de símbolos, estilo, huérfanas del .bib, duplicados PDF.

**Pendiente siguiente (Fases 1-2 del DIAGNOSTICO):** conclusión desincronizada (cita 100 % y
15,4 % viejos; lista como pendiente lo ya hecho), front matter (carátulas plantilla, resumen/
abstract sin escribir, dedicatoria "blah blah"), bibliografía (entrada lee2022 corrupta,
incorporar Davies 2022 ×2 / Gómez Hernández 2023 / Pont 2023 / Sokolova & Lapalme), auditar
las 37 notas "NapierOne/varios" del manifiesto, y Fase 3 (expandir corpus con URLs pcrisk ya
listadas, GridSearch en servidor, advanced_features para blindar Exp. 2).

## ★ REUNIÓN CON EL TUTOR 2026-08-12 — «Revisión de resultados»

Notas textuales del Prof. Cappo + el mapeo de cómo se aborda cada punto:
**`6_notas_trabajo/reunion_2026-08-12_revision_resultados.md`** (y el `.docx` original al lado).
Manda sobre la hoja de ruta previa. Resumen de lo accionable:

- **Lo más urgente:** la ablación de ventana (64/128/256/512 → 0,908) **sigue subiendo en el
  último punto**, así que el gráfico no muestra saturación. Correr 1024 y 2048 hasta que se
  aplane o baje, o corregir el dato. Es una crítica válida a una figura ya hecha.
- Subir **BLACKBASTA** al cluster → cierra el hueco de 29/30 familias (ver línea ~549).
- Probar **bytes del medio**, no solo cabecera y cola.
- **Curva de aprendizaje** en los dos frentes (cuántas muestras hacen falta) + desvío en el
  frente de archivos, que hoy va sin error.
- Ya contestable con datos existentes: validación separada (= P1/P2), justificación de ML
  frente a firmas (53,3 % de cobertura vs 0,910), y el **año de cada familia**, que está en
  `Pruebas.xlsx` (ver bloque siguiente).
- **Pendiente de decisión de Romina:** el tutor pide *majority voting* combinando los dos
  frentes, lo que choca con la decisión de mantenerlos independientes — y no hay muestras
  pareadas para evaluarlo. Detalle en el archivo de la reunión, sección D.

## FUENTE RECUPERADA 2026-08-13 — `Pruebas.xlsx`: comparación con herramientas públicas

**Ubicación:** `7_compartido_carlos/Tesis Carlos y Romina/Pruebas.xlsx`. Cuatro hojas. Es el
registro de las pruebas manuales contra ID Ransomware y Crypto Sheriff, y **el origen del
71,93 %** que se venía citando sin saber qué medía. Registrarlo acá para no volver a perderlo.

### Hoja «Deteccion de notas» — el 71,93 %
ID Ransomware acertó **41 de 57 notas de 22 familias** (0,7192982456). Las notas se bajaron de
tres repositorios públicos —threatlabz/ransomware_notes, kipziptie/ai_ransomware_note_detection
(GitLab) y RansomNoteFiles del propio Lemmou— eligiendo las familias de las que hay archivos
cifrados en NapierOne. Crypto Sheriff se descartó «por su baja deteccion con los archivos
encriptados».

> ⚠️ **No es el corpus de la tesis.** La tesis mide sobre 146 notas / 30 familias; esto son
> 57 notas / 22 familias, un subconjunto anterior. Al citarlo hay que escribir «57 notas de 22
> familias tomadas de los mismos repositorios públicos», **nunca** «sobre el mismo corpus».

Fallos: WASTEDLOCKER 0/1 · PHOBOS 0/1 · DARKSIDE 1/3 · RANSOMEXX 2/5 · CERBER 3/6 · CUBA 1/2 ·
TESLACRYPT 1/2 · BLACKBASTA 2/4 · BLACKMATTER 1/2 · BLACKCAT 3/4. Doce familias perfectas,
entre ellas GANDCRAB 8/8, CONTI 4/4, LOCKBIT 3/3, CLOP 3/3.

### Hoja «Deteccion de archivos encriptad» — la comparación fuerte del frente de archivos
30 familias de NapierOne subidas a las dos webs (Crypto Sheriff tiene tope de 1 MB, por eso
solo archivos chicos). Los «SI\*» son, textual, **«los que se detectan incluso al cambiar el
nombre al archivo»**:

| Herramienta | Detecta | Sobre 30 |
|---|---|---|
| Crypto Sheriff | 5 | 16,7 % |
| ID Ransomware, nombre original | 20 | 66,7 % |
| **ID Ransomware, con el nombre cambiado (SI\*)** | **9** | **30,0 %** |

Los 9 robustos al renombrado: GANDCRAB, LORENZ, MAZE, MEDUSALOCKER, PHOBOS, RYUK, SODINOKIBI,
TESLACRYPT, WANNACRY. Contra eso, el clasificador de bytes del Exp. 2c llega a exactitud 0,910
/ macro-F1 0,908 en 29 familias **sin usar nombre ni extensión**.
**Cuidado con la métrica:** lo de ID Ransomware es cobertura por familia (sí/no), no exactitud
por archivo. Enunciarlo como «cubre 9 de 30 familias» frente a «29 de 29», no como 30 % vs 91 %.

**Corroboración independiente del Exp. 2b:** la hoja guarda los `sample_bytes` que reporta la
herramienta y coinciden con las firmas halladas por cuenta propia — WANNACRY
`[0x00-0x08] 0x57414E4143525921` («WANACRY!»), RYUK `0x4845524D4553` («HERMES»), LORENZ
`[0x00-0x05] 0x2E737A3430`, TESLACRYPT `[0x00-0x30]`, MEDUSALOCKER `[0x5A20A-0x5A218]`,
GANDCRAB `[0x43614-0x4361C]`, MAZE `[0x58771-0x58779]`. Dos caminos distintos, mismas marcas.

### Hoja «Informacion sobre familias»
Las 30 familias de NapierOne con su **año** (2013 CRYPTOLOCKER → 2022 BLACKBASTA) y si hay nota
disponible: **22 sí, 8 no** (HELLOKITTY, SODINOKIBI, BADRABBIT, NOTPETYA, WANNACRY, JIGSAW,
CHIMERA, CRYPTOLOCKER). Sirve para la tabla descriptiva del corpus en el cap. 3.

### Hoja «Resultados» — el resultado negativo original del frente de archivos
Experimento inicial (etapa Carlos): 1 600 archivos, 50 encriptados por familia + 100 no
encriptados, `test_size` 0,4. Seis estadísticas —shannon, shannon de 100 bytes, chi cuadrado,
promedio, Monte Carlo, coeficiente de correlación serial de bytes— × seis modelos —logistic
regression, MLP, SVM, árbol de decisión, KNN, random forest—, en combinaciones de 1 a 6.
Individuales **0,036–0,086**; el máximo de toda la hoja es **0,228** (MLP, combinación de 3).
Es el punto de partida de la progresión del cap. 4 (0,228 → 0,603 estadísticas regionales →
0,910 bytes posicionales).

**Procedencia verificada 2026-08-13** leyendo los notebooks
`Notebooks/Pruebas multiclasificación/{Multiclass with multiple features, MulticlassDecisionTree}.ipynb`:

- **El dataset es NapierOne Tiny.** Las 30 carpetas de familia se llaman literalmente
  `AVOSLOCKER-tiny`, `BADRABBIT-tiny`, … `WASTEDLOCKER-tiny`, más una clase limpia `Z-Safe`.
  Son **31 clases**, así que el azar de este experimento es **0,032** y el 0,228 lo supera
  unas 7 veces — es un resultado pobre, no nulo. Decirlo así en el cap. 4.
- **La métrica es exactitud (accuracy).** En el código: `acc = accuracy_score(y_test, y_pred)`
  con `print(f'Precisión: {acc}')`. Queda confirmado, ya no hay que preguntarle a Carlos.
- `train_test_split(..., test_size=0.4, random_state=42)` — coincide con el encabezado de la
  hoja. El otro notebook usa 0,3; la fila 101 («80 test/ 20») sugiere que probaron más cortes.
- Las notas de esa etapa son **las del repositorio de Lemmou** (`RansomNoteFiles`), una de las
  tres fuentes listadas en la hoja de notas.

> ⚠️ **El dataset de 1 600 archivos NO está en la carpeta de la tesis.** Los notebooks lo leen
> de `Pruebas2.rar` en el Google Drive de Carlos (`/content/drive/MyDrive/…`), desde Colab. Lo
> que sí hay localmente es una **muestra chica**: `3_datos/archivos_cifrados/SVM/Pruebas/`
> (73 cifrados + 41 limpios) y una copia en `Notebooks/Datasets/`. Para reproducir el 0,228 hay
> que pedirle el `.rar` a Carlos, o rehacerlo bajando NapierOne Tiny.

## 7. Reglas / convenciones (IMPORTANTES — respetar en todo chat)
- **★ FUENTES FIDEDIGNAS:** todo dato, archivo, métrica o afirmación que vaya a la tesis
  debe provenir de una **fuente verificable y citable** (paper, dataset oficial, repo con
  procedencia conocida). Registrar SIEMPRE el origen para poder citarlo. No usar nada sin fuente.
- **Solo 30 familias** (NapierOne). No ampliar el número de familias.
- Norma de citación: _(confirmar — el .bib sugiere estilo del template UNA)_
- Idioma de la tesis: español.
- Reportar métricas: accuracy + **balanced accuracy + macro-F1** + classification_report por familia.
