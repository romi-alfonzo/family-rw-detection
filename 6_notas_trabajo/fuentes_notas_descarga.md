# Fuentes de notas de rescate para descargar (vetadas) — 2026-06-22

Lista para que **vos descargues** las notas. Cada fuente indica procedencia, citabilidad y
qué familias aporta. Regla: registrar siempre el origen de cada nota que entre al corpus.
**Solo se usan las 30 familias de la tesis.** Antes de citar, verificar la licencia de cada repo.

---

## 1. Repositorios por familia (texto de notas, listos para procesar)

### ★★★ lemmou/RansomNoteFiles — LA MÁS CITABLE
- URL: https://github.com/lemmou/RansomNoteFiles
- Organizada **por familia y por versión**.
- Respalda académicamente a **Lemmou et al. (2021), "In-Depth Analysis of Ransom Note Files"**,
  paper que ya citás en tu metodología → fuente con autoría y publicación verificable.
- **Prioridad #1.** Ideal para reforzar familias y justificar la procedencia en metodología.

### ★★★ ThreatLabz/ransomware_notes (Zscaler) — ya la tenés
- URL: https://github.com/ThreatLabz/ransomware_notes
- Threat intel reputable (Zscaler). 209 familias / ~317 archivos.
- Aporta ~49 notas para tus 30 familias (algunas duplicadas). **0** para BADRABBIT, CHIMERA, JIGSAW, NOTPETYA, WANNACRY.

### ★★ Kaggle — Ransomware Note Dataset Collection (abiprasanth)
- URL: https://www.kaggle.com/datasets/abiprasanth/ransomware-note-dataset-collection
- Texto de notas para clasificación (BERT/FastText). Requiere cuenta Kaggle para descargar.
- ⚠️ Al bajarlo, **verificá autor/procedencia y licencia** (puede reempaquetar otros repos como ThreatLabz → posible duplicación).

### ★ Repos secundarios (verificar procedencia/licencia)
- eshlomo1/Ransomware-NOTE — https://github.com/eshlomo1/Ransomware-NOTE (notas + extensiones)
- kh4sh3i/Ransomware-Samples — https://github.com/kh4sh3i/Ransomware-Samples (por familia; son muestras, no notas curadas)
- albertzsigovits/malware-notes — https://github.com/albertzsigovits/malware-notes (notas técnicas por familia, p.ej. Clop)

---

## 2. Familias CRÍTICAS (las 5 con solo 2 notas) — fuentes primarias citables

ThreatLabz no cubre estas. Usar texto de notas publicado en reportes con autoría/fecha:

### WANNACRY
- Mandiant (Google Cloud) — Malware Profile: https://cloud.google.com/blog/topics/threat-intelligence/wannacry-malware-profile
- Antiy Labs — In-Depth Analysis Report: https://www.antiy.net/p/in-depth-analysis-report-on-wannacry-ransomware/
- Wikipedia (incluye texto de la nota, con referencias): https://en.wikipedia.org/wiki/WannaCry_ransomware_attack

### NOTPETYA
- BleepingComputer — The Week in Ransomware (NotPetya): https://www.bleepingcomputer.com/news/security/the-week-in-ransomware-june-30th-2017-notpetya/
- Hitachi HIRT — Security Alert NotPetya: https://www.hitachi.com/en/hirt/publications/hirt-pub/hirt-pub17010/

### BADRABBIT
- BleepingComputer — Bad Rabbit Outbreak: https://www.bleepingcomputer.com/news/security/bad-rabbit-ransomware-outbreak-hits-eastern-europe/
- Malwarebytes Labs — BadRabbit (variante de Petya/NotPetya): https://www.malwarebytes.com/blog/news/2017/10/badrabbit-closer-look-new-version-petyanotpetya
- Nota: el texto de BadRabbit es casi idéntico al de NotPetya (dato útil para la discusión).

### JIGSAW
- BleepingComputer — Jigsaw Decrypted: https://www.bleepingcomputer.com/news/security/jigsaw-ransomware-decrypted-will-delete-your-files-until-you-pay-the-ransom/
- Wikipedia: https://en.wikipedia.org/wiki/Jigsaw_(ransomware)

### CHIMERA — pendiente, fue la más difícil de encontrar
- No apareció texto de nota en fuentes reputables en la búsqueda inicial.
- Sugerencia: buscar en Trend Micro, Kaspersky Securelist y en lemmou/RansomNoteFiles (puede tenerla por versión).

---

## 3. Cómo proceder cuando descargues
1. Guardá cada fuente en una subcarpeta separada (no mezclar con `ransom_notes_corpus`).
2. Avisame y yo (con tu OK) deduplico por contenido contra lo que ya tenés y armo un manifiesto
   de procedencia (familia, archivo, fuente, fecha) para que todo sea citable.
3. Reentrenamos y comparamos métricas macro antes/después.
