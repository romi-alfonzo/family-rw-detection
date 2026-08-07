# DIAGNÓSTICO PROFUNDO DE LA TESIS — 2026-07-27

> Generado tras un análisis exhaustivo de: capítulos LaTeX (las 3 copias), código y
> experimentos (clasificador, family-rw-detection, CSVs de resultados), corpus (original
> y v2), bibliografía (.bib) y los 3 PDFs principales (Pont 2023, Lee 2019, Patsakis 2024),
> más las actas de reuniones con el tutor.
> Complementa a `ESTADO_TESIS.md` — no lo reemplaza.

---

## RESUMEN EJECUTIVO

La tesis tiene **material de sobra** (experimentos corridos, un capítulo de resultados
completo que está "perdido", 4 figuras generadas sin usar, y un posicionamiento
bibliográfico fuerte que nadie escribió aún). El estancamiento no es falta de contenido:
es que el contenido está **desorganizado, desincronizado y con números no trazables**.
El camino de salida es reconciliar y ensamblar, no producir desde cero.

**Hallazgo estratégico más importante:** no existe en la literatura revisada un
competidor directo de la clasificación supervisada **multiclase de 30 familias por
contenido de nota**. Lemmou et al. (2021) —el benchmark real— hace detección *binaria*
y similitud *no supervisada*, y concluye que la atribución por familia es problemática.
La tesis es una extensión legítima y defendible. Eso hay que escribirlo.

---

## A. HALLAZGOS CRÍTICOS DE INTEGRIDAD (resolver antes que nada)

### A1. El capítulo de Resultados no existe, pero SÍ existe uno completo "perdido"
- `Plantilla_de_Tesis___Romina_Carlos\resultados.tex` (documento vivo) = **1 línea de comentario**.
- Los resultados reales están mal ubicados dentro de Metodología (§3.5 y §3.6).
- **Existe un `resultados.tex` completo de 203 líneas** con los 3 experimentos redactados en:
  `Plantilla inicial\images\Plantilla_de_Tesis___Romina_Carlos\resultados.tex` (sepultado en `images/`).
- Hay 3 copias paralelas del documento: la viva (`Plantilla_de_Tesis___Romina_Carlos\`),
  el borrador `latex_capitulos\` (abril, nunca integrado, con la única estructura de
  resultados de 3 experimentos y 6 TODOs) y la copia perdida en `Plantilla inicial\images\`.

### A2. Las cifras NO son trazables a los CSV — algunas parecen inventadas/suavizadas
- **Cinco tamaños de corpus** circulan por los textos: 81 / 92 / 146 / 156 / 161.
- Las tablas de los borradores **no coinciden con sus propios CSV fuente**:
  - Binaria: rango real de Random Forest 69,3–97,3 → escrito 85,3–91,2 (mín/máx "suavizados").
  - Multiclase: CSV real GB 9,9 / RF 9,8 → escrito RF 12,1 / GB 11,5. Ninguna cifra coincide.
- El **"15,4 % de accuracy"** que sostiene una conclusión central es casi seguro un valor
  de **ERROR** leído como accuracy (la tabla se titula "Error promedio"). El CSV real dice
  que el mejor multiclase es **9,9 %**.
- El corpus "161 notas" (`metodologia.tex` §3.2.2) no cierra: el desglose por fuente suma
  156-157; el "161" parece copiado del dataset de Lemmou ("161 archivos, 62 familias").
- **Regla a adoptar: ninguna cifra tipeada a mano.** Todas las tablas deben regenerarse
  desde los CSV de una corrida canónica única.

### A3. Las métricas están infladas y el texto defiende la inflación
- `metodologia.tex:399` justifica el F1 *weighted* con un argumento **invertido**
  (dice que evita que las familias chicas dominen; en realidad hace que las GRANDES dominen).
- `metodologia.tex:109` **promete** métricas macro y reporte por familia que nunca se entregan.
- El macro-F1 (~0,84) **no aparece en ningún .tex**. El reporte por familia existe en CSV
  (`exp3_nlp_detallado.csv`) y muestra F1 = 0,000 en varias familias — está oculto, y es
  justamente la justificación empírica para expandir el corpus que pidió el tutor.
- **Corrección a `ESTADO_TESIS.md`:** el famoso "87,58 %" es la **accuracy** (weighted recall),
  no el F1 weighted (que es 85,61 %). La anotación actual también está mal etiquetada.

### A4. Atribución errónea de NapierOne y procedencia del corpus
- **NapierOne NO es de Pont.** Es de **Davies, Macfarlane & Buchanan (2022)**, Edinburgh
  Napier University ("NapierOne: A modern mixed file data set alternative to Govdocs1",
  *FSI: Digital Investigation* 40, 301330). Pont solo lo cita; su tesis usa Govdocs1.
- **NapierOne es un dataset de archivos mixtos, no un corpus de notas.** En
  `manifiesto_corpus_v2.csv` solo 37/146 notas (25 %) llevan la etiqueta ambigua
  "NapierOne/varios". Procedencia real: ThreatLabz 49, lemmou 47, "NapierOne/varios" 37,
  pcrisk 13. **Auditar el origen real de esas 37 notas** y reescribir la sección de
  procedencia: fuentes primarias = ThreatLabz + Lemmou + pcrisk; NapierOne aporta (a
  confirmar) la *lista de 30 familias*.
- Riesgo declarable: WannaCry, NotPetya y BadRabbit provienen de OCR de capturas
  (`notas_familias_criticas.md`) — la metodología debe decirlo.

### A5. Bibliografía: una entrada errónea muy citada + aportes del tutor ignorados
- **`lee2022` es una entrada corrupta**: mismo título que `paper_3_enhancing_file_entropy`
  (Hsu et al. 2021) con otros autores; se cita **5 veces** para sostener "la entropía de
  Shannon no es la métrica óptima". Verificar cuál es el paper real (probablemente
  Lee, Lee & Yim 2022, IEEE Access 10, DOI 10.1109/ACCESS.2022.3151354) y corregir.
- Faltan del .bib: el paper sugerido por el tutor (Gómez Hernández et al., *Electronics*
  12(21):4494, 2023 — ¡ya citado en el paper del Simposio!), la tesis de Pont, el paper
  de one-shot learning (arXiv 1908.06750, aportado por el tutor), los DOS papers de
  Davies et al. 2022 (NapierOne y comparación de 11 métodos de entropía), y las 9 fuentes
  de `bibliografia_fuentes_nuevas.md` (0 incorporadas).
- No hay ninguna cita metodológica sobre macro-F1 / evaluación multiclase desbalanceada
  (ej. Sokolova & Lapalme 2009) — justo donde más falta hace.
- `latex_capitulos\referencias.bib` tiene ≥6 entradas con autoría incorrecta
  (khammas2021notes, khammas2023, pont2023, herrerasilva2023, davies2022, davies2023napierone).
  **NO migrar**; si se recupera texto de esos borradores, remapear claves al .bib bueno.
- 4 entradas huérfanas en el .bib bueno; CryptoSheriff se discute pero nunca se cita a sí mismo.

---

## B. HALLAZGOS DE CÓDIGO Y METODOLOGÍA

### B1. Bug de decodificación (bloqueante para corpus_v2)
14 de 146 archivos no son UTF-8 (7 DHARMA UTF-16LE sin BOM, 6 GANDCRAB UTF-16,
1 SUNCRYPT cp1252). Con `errors="replace"` el texto queda intercalado con NULs:
- 13 notas (8,9 %) generan **vector de palabras vacío**.
- En char n-grams, los NUL son una **firma sintética** que regala separabilidad a DHARMA
  y GANDCRAB (fuga artificial).
- Con el loader corregido (BOM→UTF-16; >20 % NULs→UTF-16LE; fallback cp1252):
  corpus_v2 word accuracy 0,808 → **0,856**, macro-F1 0,767 → **0,783**.
- Arreglo: ~20 líneas en la lectura + conectar `extractor_notas.py` (hoy desconectado).
  Instalar `beautifulsoup4` (falta en el Python local y rompe el import del clasificador).

### B2. La validación real es 2-fold, no 5-fold
`n_folds = min(5, min_samples)` y hay familias con 2 notas → **2 folds** en todas las
corridas reportadas. Además: TF-IDF ajustado sobre TODO el corpus antes del CV (fuga de
vocabulario/IDF — `Pipeline` está importado y sin usar), una sola semilla sin
repeticiones, y el script no calcula macro-F1 ni balanced accuracy ni matriz de confusión
(todo importado, nada usado).

### B3. Duplicados: el 0,897/0,840 del corpus original está contaminado
- `ransom_notes_corpus`: 156 notas pero solo **118 textos únicos** (24 % de redundancia,
  38 grupos duplicados) → con shuffle, copias idénticas caen en train y test a la vez.
- `corpus_v2`: 0 duplicados exactos ✅ pero **28 pares con coseno > 0,90** (todos
  intra-familia: CUBA 0,994, GANDCRAB 0,989, MAZE 0,988, RYUK 0,986, CERBER .hta 0,95+).
- Solución: clusterizar por similitud > 0,90 y usar **StratifiedGroupKFold** +
  **RepeatedStratifiedKFold** (10 semillas).
- Resultado de referencia honesto en corpus_v2 (loader corregido, LinearSVC word, SKF=2):
  accuracy 0,86 / balanced 0,81 / **macro-F1 0,78** / weighted 0,84. Solo CHIMERA y
  WANNACRY quedan en F1=0 (antes eran 3). La caída vs 0,840 es la **corrección del sesgo**
  por duplicados — presentarla así, es un argumento de rigor, no una pérdida.

### B4. El feature de nombre de archivo (pendiente declarado) es una TRAMPA tal como está
- Nombre solo: macro-F1 0,655. Texto+nombre: 0,759 → **0,883** (+0,12, muy vendible)...
- ...pero enmascarando alias de familia en el nombre cae a **0,160**: el 51 % de los
  nombres (75/146) contiene el nombre de la familia **puesto por el curador, no por el
  atacante** (ThreatLabz 34/49, pcrisk 13/13, NapierOne 23/37). Los nombres genuinos son
  los de lemmou (42/47 sin alias).
- La extensión sola es inútil (macro-F1 0,038).
- Recomendación: recuperar `confianza_nombre_original` del manifiesto v1, marcar qué
  nombres son artefactos reales del atacante, construir el feature solo con esos, y
  reportar la ablación con/sin. Reportar 0,883 sin esta salvedad es indefendible.
- **No usar la columna `fuente` como feature** (pcrisk solo aparece en familias concretas
  → fuga de procedencia).

### B5. Ablación defensiva ya medida (oro para la defensa)
El 42 % de las notas contiene el nombre/alias de su familia en el cuerpo. Enmascarándolos,
el rendimiento casi no cambia (char macro-F1 0,759→0,756; word 0,783→0,787). **El
clasificador NO está simplemente leyendo la marca.** Incluir esta ablación en la tesis:
es la primera objeción obvia de cualquier revisor y ya está respondida.

### B6. Análisis de archivos cifrados (Objetivo 2) — flanco débil
- Bug en `helpers.calculate_chi_square`: divide por bytes observados, no por 256 → el
  estadístico no es χ² contra la uniforme (`advanced_features.py` ya lo corrige, pero los
  resultados guardados vienen del código viejo).
- El "9,8 %" multiclase se apoya en **solo 2 features** (entropía y tamaño). Un revisor
  dirá "no probaron suficientes features". La respuesta ya está escrita y sin ejecutar:
  `advanced_features.py` (275 features) + `train_advanced.py` (5 subconjuntos × RF/KNN/GB,
  StratifiedKFold(5) + SelectKBest). Correr en el servidor de la facultad → demostrar que
  la indistinguibilidad persiste incluso con features ricas = Objetivo 2 blindado.
- Los `*_iterations.py` eligen test_size por accuracy en test (sesgo optimista) — no reusar.
- Los `combined*.py` referencian CSVs que no existen — hoy no corren.

---

## C. POSICIONAMIENTO BIBLIOGRÁFICO (lo que hay que ESCRIBIR)

### C1. El mapa real del estado del arte
- **Pont (2023)** (tesis doctoral, U. Kent): NO clasifica notas. Es la referencia canónica
  para la mitad estadística: usa exactamente entropía, χ² (umbral 293,25), Monte Carlo π,
  media y correlación serial; demuestra FPR de hasta 92,8 % con estadísticos aislados
  (¡valida el Objetivo 2!); solo llega a 97,5 % agregando desv. estándar sobre lotes de
  50 archivos. Dato puente: observa que las notas de rescate son "pequeñas y de baja
  entropía". Propone como trabajo futuro "separar el contenido textual de una nota de su
  contenido gráfico" — literalmente el enfoque de esta tesis: citarlo como hueco
  identificado por él.
- **Lemmou et al. (2021)** (*Computers* 10(11):145 — ya está en `Leido\`): el benchmark
  directo. Binaria nombre-de-nota vs benigno: RF F 0,920 / acc 98,32 %; contenido: LSA no
  supervisado (372 TP / 8 FP). Sus 8 FP son confusiones ENTRE familias
  (CryptoLocker↔TeslaCrypt; Rapid↔StorageCrypt) → converge con los F1=0 de esta tesis en
  CRYPTOLOCKER: **no es defecto del modelo, es propiedad del dominio**. Además documenta
  que ID Ransomware falla en 29 familias y no usa el contenido de la nota.
- **Lee et al. (2019)** (IEEE Access 7): precedente del uso de LinearSVC en el dominio;
  F1=1,0 con 1.200 archivos = sobreajuste probable — usar críticamente. El contraste
  Lee (la entropía basta) ↔ Pont (FPR 92,8 %) es un párrafo excelente de estado del arte.
- **Patsakis et al. (2024)** (IJIS 23): motivación económica con cifras duras — NetWalker
  $27,5M, Locky $14M, REvil $12,1M, MedusaLocker $5,3M, HelloKitty $1,07M — varias de las
  30 familias. Para la introducción, NO para la tabla comparativa.

### C2. El argumento central de la defensa (escribirlo explícito)
Las cifras 97-98 % de Lemmou son de una tarea **binaria**; las de esta tesis son de
**30 clases** (baseline aleatorio ≈ 3,3 %). Un macro-F1 de ~0,78-0,84 en 30 clases con
familias de 2 notas NO es peor que 0,92 binario — es una tarea sustancialmente más
difícil y sin competidor directo en la literatura revisada. Si no se escribe este
contraste, el tribunal leerá "0,84 < 0,98".

### C3. Comparación empírica más valiosa disponible (bajo costo)
**Correr el corpus contra ID Ransomware y medir su accuracy.** Es la herramienta que la
tesis propone superar; Lemmou ya documentó sus fallas; el tutor pidió (08/05/24) describir
la metodología de esas pruebas. `Pruebas.xlsx` existe y no está en la tesis.

### C4. Pedidos del tutor aún sin atender (de las actas)
1. Paper MDPI Electronics 12(21):4494 (13/03/24) — no citado.
2. **Variabilidad de las notas intra-familia** (02/05/24) — no existe el análisis; es la
   justificación teórica de por qué funciona TF-IDF de caracteres; barato con el corpus actual.
3. Describir la metodología propia de pruebas con ID Ransomware/CryptoSheriff (08/05/24) —
   hoy esas cifras (16,67/66,67/72 %) se citan como bibliografía ajena, pero el paper del
   Simposio dice "obtuvimos" (primera persona). Aclarar autoría.
4. Metadatos/magic bytes de archivos cifrados según ID Ransomware (11/01/25) — única vía
   para matizar el resultado negativo del Exp. 2 (los magic bytes SÍ discriminan, la
   entropía no).
5. LSA (08/05/24) — investigar/comparar (Lemmou lo usa).
6. One-shot learning (06/06/24, arXiv 1908.06750) — respuesta metodológica a familias con
   2 notas; citar al menos como trabajo futuro.
7. Binaria por familia (27/02/25) — `exp1_por_familia.csv` existe y no está en el documento vivo.

### C5. Corpus: expansión pendiente ya planificada
14 familias con ≤4 notas tienen URL de pcrisk identificada y sin descargar
(`mas_notas_descarga.md`). Para 5-fold real hacen falta ≥5 notas en 22 familias.
Kaggle "Ransomware Note Dataset Collection" pendiente (ojo: puede duplicar ThreatLabz).
Verificar BTC/claves de WannaCry/NotPetya/BadRabbit contra imágenes originales.

---

## D. FORTALEZAS YA DISPONIBLES (nada que producir, solo ensamblar)

1. `resultados.tex` completo (203 líneas) recuperable de `Plantilla inicial\images\...`.
2. 4 figuras PNG generadas y nunca insertadas (`resultados_experimentos\fig_*.png`);
   `main.lof` está vacío — la tesis no tiene ni una figura.
3. Reporte por familia ya calculado (`exp3_nlp_detallado.csv`).
4. Ablación anti-"lee la marca" ya medida (B5).
5. Marco teórico (cap. 2) sólido y bien citado — el mejor capítulo.
6. Posicionamiento único en la literatura (C2).
7. `corpus_v2` bien construido (0 duplicados exactos, procedencia etiquetada, manifiesto
   consistente con el disco).

---

## E. PLAN DE ACCIÓN PRIORIZADO

### Fase 0 — Desbloqueo técnico y de estructura (esta semana)
- [ ] **E1. Arreglar decodificación** + conectar `extractor_notas.py` + `pip install beautifulsoup4`.
      Sin esto, cualquier número sobre corpus_v2 es inválido. (~20 líneas)
- [ ] **E2. Evaluación honesta**: Pipeline (TF-IDF dentro del CV), StratifiedGroupKFold
      (grupos = clusters coseno>0,90), RepeatedStratifiedKFold ×10, scoring =
      accuracy + balanced_accuracy + f1_macro, classification_report por familia,
      matriz de confusión. → **Corrida canónica única** que regenera TODAS las tablas.
- [ ] **E3. Recuperar el `resultados.tex` perdido**, crear el capítulo 4 real, mover ahí
      §3.5-3.6 de Metodología, corregir cifras contra los CSV de la corrida canónica.

### Fase 1 — Integridad del texto
- [ ] E4. Reescribir el párrafo del weighted (`metodologia.tex:399`) y reportar macro-F1 +
      balanced accuracy con el argumento de dificultad (C2). Documentar la brecha
      weighted↔macro como advertencia metodológica (aporte propio).
- [ ] E5. Unificar el tamaño del corpus (146, corpus_v2) en TODO el texto; eliminar
      81/92/156/161. Corregir el "15,4 %" (es un error, el real es 9,9 %).
- [ ] E6. Reescribir la procedencia del corpus (A4) y declarar el OCR de las 3 familias.
- [ ] E7. Insertar las 4 figuras; completar front matter (carátulas, dedicatoria,
      resumen/abstract — es lo primero que lee el tribunal).

### Fase 2 — Bibliografía (1-2 días)
- [ ] E8. Corregir `lee2022`; citar `cryptosheriff` donde corresponde; limpiar huérfanas.
- [ ] E9. Incorporar: Davies 2022 ×2 (NapierOne + entropía), Gómez Hernández 2023 (tutor),
      Pont 2023, Lemmou 2021 (¡el benchmark, ya leído!), one-shot learning, EnCoD,
      Sokolova & Lapalme (métricas). Remapear claves si se recupera texto de borradores.

### Fase 3 — Experimentos de valor agregado
- [ ] E10. **ID Ransomware vs corpus propio** (C3) — la comparación estrella, costo bajo.
- [ ] E11. Análisis de variabilidad intra-familia de las notas (pedido del tutor 02/05/24).
- [ ] E12. Expandir a ≥5 notas las 22 familias (URLs pcrisk ya listadas) → 5-fold real → re-correr.
- [ ] E13. Feature de nombre de archivo SOLO con nombres genuinos del atacante (B4), con ablación.
- [ ] E14. `advanced_features.py` + `train_advanced.py` en el servidor (275 features) → blindar Objetivo 2.
- [ ] E15. (Opcional, puente entre objetivos) Medir entropía de las 146 notas vs archivos
      cifrados — un gráfico que unifica la tesis; nadie en la bibliografía lo hizo.

---

## F. ADENDA (mismo día): hallazgos en `Tesis Carlos y Romina\`

### F1. `Pruebas.xlsx` — resuelve el misterio de las cifras 16,67 / 66,67 / 71,93 %
Las tres cifras que la tesis cita como bibliografía ajena (`paper_2_on_efectiveness`)
son en realidad **experimentos propios** registrados en este spreadsheet:
- Hoja "Deteccion de archivos encriptad": subieron archivos cifrados de NapierOne a
  CryptoSheriff e ID Ransomware, familia por familia. CryptoSheriff detecta **5/30 =
  16,67 %**; ID Ransomware **20/30 = 66,67 %** (verificado contando la hoja).
- Hoja "Deteccion de notas": subieron 57 notas de 22 familias a ID Ransomware:
  **41/57 = 71,93 %** (verificado).
- **Consecuencia:** el pedido del tutor (08/05/24) de "describir la metodología de sus
  pruebas" y el E10 del plan ya tienen los DATOS: solo falta redactarlos como experimento
  propio (sección de metodología + tabla). Esto además convierte un problema de integridad
  (mala atribución) en una FORTALEZA: es trabajo original ya hecho.
- Bonus: las filas "SI*" documentan los **magic bytes / custom rules** de ID Ransomware
  por familia (GANDCRAB, LORENZ, MAZE, MEDUSALOCKER, PHOBOS, RYUK, SODINOKIBI, TESLACRYPT,
  WANNACRY) → es el insumo del pedido del tutor del 11/01/25 (metadatos) y del argumento
  "los magic bytes discriminan, la entropía no".
- La hoja "Informacion sobre familias" tiene la tabla 30 familias × año × ¿hay nota? —
  útil para el capítulo del corpus.

### F2. `Papers\` — varios PDFs clave YA están descargados
- `NapierOne_Data_Set(1).pdf` → **el paper de NapierOne (Davies et al.) ya está** — tachar
  esa descarga de E9; solo falta citarlo.
- `computers-10-00145.pdf` → Lemmou 2021 (duplicado del de Leido\).
- `State of practice in ransom payments.pdf` → Patsakis (duplicado).
- `document.pdf` → **Bou-Harb et al. 2021, "On Ransomware Family Attribution Using
  Pre-Attack Paranoia Activities"** (IEEE TNSM). Trabajo relacionado de ATRIBUCIÓN DE
  FAMILIA (por actividades de red pre-ataque) — complementa el posicionamiento: ellos
  atribuyen familia con artefactos pre-ataque, esta tesis con artefactos post-ataque.
- `ClassificationofransomwarefamilieswithmachinelearningbasedonN-gramofopcodes.pdf` →
  clasificación de familias por n-gramas de opcodes (Zhang et al.) — el paralelo perfecto:
  n-gramas sobre código vs n-gramas sobre texto de nota. Citar en trabajos relacionados.

### F3. `Notas.docx` — identifica correctamente la tesis UPC y una 4ª fuente de notas
- El "khammas2023" del bib viejo es en realidad la **tesis UPC de Trujillo (Kim Kip)**:
  binaria nota-vs-no-nota, 59+59 documentos, corpus de lemmou + 20newsgroups. Antecedente
  directo más para la tabla de related work del clasificador de notas.
- Revela el repo **gitlab.com/kipziptie/ai_ransomware_note_detection** usado como fuente
  de notas en Pruebas.xlsx → pista para la auditoría A4: parte de las 37 notas
  "NapierOne/varios" del manifiesto podría venir de ahí.

### F4. `Reporte 28-07\ransomware_classification_results.csv` — binaria por familia YA corrida
30 familias × 6 modelos (familia-tiny vs safe). Es el pedido del tutor del 27/02/25
(binaria por familia) ya ejecutado. ⚠️ Usar con cautela: muchos 1,000 perfectos en
DT/KNN/RF sugieren separación trivial por la feature de tamaño o fuga — revisar el
notebook (`Notebooks\Family vs Safe.ipynb`) antes de reportar.

### F5. `Notebooks\` — los notebooks originales + modelos .pkl + Datasets\Encr|Legit
Fuente primaria de los experimentos 1 y 2 (los CSV de resultados_experimentos salen de acá).

### Correcciones a `ESTADO_TESIS.md` (cuando se actualice)
- "87,58 %" = accuracy, no F1 weighted (85,61 %).
- NapierOne = Davies et al. 2022, no Pont; el corpus de notas NO es NapierOne (ver A4).
- Benchmark directo = Lemmou et al. 2021.
- Métricas de referencia actuales (corpus_v2, loader corregido, sin near-dup control aún):
  acc 0,86 / balanced 0,81 / macro-F1 0,78.
