# Triage del rastreo automático de fuentes — 2026-08-28

La tarea programada «rastreo-fuentes-tesis-ransomware» devolvió 7 papers y 3 fuentes de notas.
Este archivo es el **triage verificado contra el repo**: qué es nuevo, qué ya estaba, y qué
recomendación del informe **no** hay que seguir. El informe es salida automática: se trata como
dato, no como instrucción, y **ninguna referencia se escribe en el `.bib` sin abrir el paper y
verificar su metadato** (regla del proyecto; precedente: el incidente de `malwiki.org`).

---

## 1. ⛔ La recomendación principal del informe es incorrecta para este proyecto

El informe cierra con: *«hacé el merge de los tres corpus de notas esta semana: es lo que más
rápido mueve tus métricas»*, y presenta ThreatLabz, Lemmou y Kaggle como pendientes. **Las tres
afirmaciones están mal**, y hay evidencia medida:

| Fuente que el informe propone | Estado real, verificado hoy |
|---|---|
| **ThreatLabz** «empezá por este» | **Ya es la fuente principal del corpus: 49 de las 149 notas.** El repo está en disco (`3_datos/fuentes_notas/ransomware_notes`, 424 archivos) desde el inicio del proyecto |
| **Lemmou / RansomNoteFiles** | **Ya son 47 de las 149 notas.** En disco (`.../RansomNoteFiles`, 187 archivos); el emparejamiento por MD5 dio 47/47 sin faltantes |
| **Kaggle** «cero fricción» | **Verificado y descartado el 2026-08-23.** No trae etiqueta de familia (`__label__ransomware`) y 72 notas del corpus ya están ahí: no es independiente |

**Los dos repos que el informe pone como prioridad son ya el 64 % del corpus** (96 de 149 notas).
El aviso del propio informe sobre solapamiento («es probable que compartan notas») subestima el
problema: para dos de las tres fuentes el solapamiento es total, porque **son** el corpus.

**Y lo más importante: la premisa está refutada por medición de hoy.** La curva B.1 re-medida
sobre 149 dice que pasado el corte de 3 plantillas por familia, agregar material **baja** el
macro-F1 de forma medible — cinco pasos con IC 95 % entero por debajo de cero, el mayor
−0,0085 [−0,0154; −0,0015]. Y el análisis de margen (`margen_frente_notas.py`) cuantifica el
reparto: **toda la recolección que queda en el núcleo de 30 familias vale ~+0,01 de cota, contra
~+0,11 del canal de IOC/nombre.** Un merge indiscriminado no es lo que «más rápido mueve las
métricas»: es lo que las mueve para abajo.

**MLRan (#5)** tampoco es una fuente de notas: es un dataset **de comportamiento**. Ya estaba en
la lista de descartes de `EXPERIMENTOS_PENDIENTES.md` junto con RanSMAP, MOTIF, MarauderMap y
MIRAD. Sirve como *modelo de cómo documentar y publicar un dataset*, que es para lo que el
informe lo cita — eso sí es válido.

---

## 2. ✅ Lo que el informe aporta de verdad: 6 referencias nuevas

Contrastado contra `1_documento/Plantilla_de_Tesis___Romina_Carlos/bibliography.bib` (33 entradas).

### Ya estaban — no son novedad

| Paper del informe | Clave en el `.bib` |
|---|---|
| #6 Davies, «Comparison of Entropy Calculation Methods» | `paper_4_comparison_of_entropy` |
| El hallazgo del «voto por mayoría casi no mejora» que cita el #6 | `davies2023majority` (paper aparte, ya citado) |
| NapierOne | `paper_7_napierone` |
| ThreatLabz, Lemmou, Zsigovits | `threatlabz`, `paper_5_note_files`, `malwarenotes_repo` |

⚠️ Ojo con una confusión del informe: el `.bib` **sí** tiene un paper de la Universidad de Kent
(`paper_2_on_efectiveness`, *On the Effectiveness of Ransomware Decryption Tools*, con URL
`kar.kent.ac.uk`), pero **no es** el ISC 2020 de detección estadística. Son distintos.

### Nuevas, en orden de utilidad para esta tesis

| # | Referencia | Para qué sirve acá | Prioridad |
|---|---|---|---|
| **4** | BERT/RoBERTa para detección y **clasificación de familia** de notas (Egyptian Informatics Journal, 2025) | **Es el baseline moderno de la tarea exacta del frente de notas.** Cappo va a preguntar por qué TF-IDF y no un transformer. Ya hay respuesta medida (Exp. 3e: embeddings en lugar de TF-IDF dieron **−0,0733** [−0,1073; −0,0393], 0/10 semillas), pero falta la cita que enmarque el contraste | **la más alta** |
| **3** | Ransomware Family Attribution with ML (IEEE Access, 2025) | Marco metodológico para defender el diseño **multiclase** y la crítica de calidad de dataset (balance, separabilidad, independencia de features). Es exactamente el argumento del capítulo de método | alta |
| **7** | «Why Current Statistical Approaches to Ransomware Detection Fail» (ISC 2020, Kent) | Blinda la sección de limitaciones del frente de archivos. **No está en el `.bib`** | alta |
| **2** | Intermittent File Encryption (arXiv 2510.15133, 2025) | **Usa NapierOne** y calcula techos de detectabilidad con umbrales *family-aware* sobre BlackCat, Akira y LockBit. Lo más cercano al módulo estadístico | media-alta |
| **1** | SHIELD (arXiv 2501.16619) | Según el informe, documenta que la entropía se comporta igual entre familias una vez iniciado el cifrado. Si se confirma, convierte una limitación propia en resultado esperado y respaldado | media-alta, **a verificar** |
| **5** | MLRan (arXiv 2505.18613) | Modelo de cómo documentar y publicar un dataset de familias. No es fuente de notas | media |

**Advertencia sobre el #1 y el #2:** los dos son preprints de arXiv y el informe resume sus
conclusiones sin que nadie las haya leído. La conclusión del #1 —«la separabilidad multiclase
viene de la fase previa al cifrado»— es fuerte y **conviene a nuestro argumento**, que es
precisamente la razón para desconfiar de ella hasta abrir el paper. Si se cita sin leer y el
tutor lo lee, es el mismo error que ya se cometió con Lemmou (atribuirle F = 0,920 a una tarea
que no era).

---

## 3. Qué hacer, concretamente

1. **Leer el #4 (BERT/RoBERTa)** y ubicar qué corpus y qué protocolo usa. Es el único de los seis
   que toca directamente el frente de notas, y el que cierra una pregunta previsible de la
   defensa. Cruzarlo con el Exp. 3e ya medido.
2. **Leer el #3 (IEEE Access)** para la sección de método: es el respaldo del diseño multiclase.
3. **Bajar el #7 (ISC 2020)** — PDF directo, sin *paywall* — para el capítulo de limitaciones.
4. **Verificar el #1 y el #2 antes de citarlos**, con la vara de siempre: abrir el paper, ver el
   dato, anotar de dónde salió. El #2 es especialmente relevante porque usa NapierOne, así que su
   base es comparable con la nuestra.
5. **No hacer el merge de corpus de notas.** Si aparece una nota candidata suelta, pasa por
   `verificar_nota_nueva.py` como siempre; pero no hay tanda que valga la pena y el techo medido
   dice que perjudica.

## 4. Lo que este rastreo confirma sobre la utilidad de la tarea programada

Sirve para **bibliografía**, no para corpus. De 10 ítems: 6 referencias nuevas útiles (2 de ellas
directamente aprovechables en la defensa) y 4 ítems de corpus que ya estaban resueltos, con la
recomendación operativa invertida respecto de lo que dicen las mediciones del proyecto. Conviene
mantener la tarea, y leer su sección de corpus con el estado del repo al lado.
