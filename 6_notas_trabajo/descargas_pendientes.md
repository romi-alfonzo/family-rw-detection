# Descargas pendientes (lo que Claude no puede bajar) — 2026-06-22

Claude solo puede leer **texto de páginas web**, no archivos binarios (imágenes, PDFs, ZIPs).
Estas las descargás vos y Claude las procesa (OCR / extracción) con `extractor_notas.py`.
Guardá las imágenes en `imagenes_notas/` y los demás archivos donde se indique.
Meta: dataset **variado en formatos** (txt, html, pdf, imagen) y con las 30 familias reforzadas.

---

## A. Imágenes de notas — familias críticas

| Familia | Estado | Archivo / fuente |
|---|---|---|
| WannaCry | ✅ ya descargada | `Wana_Decrypt0r_screenshot.png` (OCR OK) |
| NotPetya | ✅ ya descargada | `mbr-ransom-note.jpg` (OCR OK con preprocesado) |
| BadRabbit | ✅ ya descargada | `oops_note.png` (OCR OK) |
| **Jigsaw** | ⬜ pendiente (tenés el texto, falta imagen para variedad) | Captura de la ventana "Billy puppet": artículo BleepingComputer https://www.bleepingcomputer.com/news/security/jigsaw-ransomware-decrypted-will-delete-your-files-until-you-pay-the-ransom/ o Wikipedia https://en.wikipedia.org/wiki/Jigsaw_(ransomware) |
| **Chimera** | ⬜ FALTA (sin texto aún) | Buscar captura/HTML de `YOUR_FILES_ARE_ENCRYPTED.HTML`. Fuentes: G DATA, Trend Micro, Kaspersky Securelist. Búsqueda sugerida: "Chimera ransomware ransom note YOUR_FILES_ARE_ENCRYPTED" |

## B. Para dataset variado — opcional, sumar formatos por familia

Para que el clasificador (y la defensa ante el tutor) demuestre que soporta todos los tipos,
conviene tener al menos un ejemplo de cada formato. Candidatos fáciles de conseguir:

- **PDF:** algunas familias dejan la nota en PDF (p. ej. notas de doble extorsión modernas).
  Revisá en `RansomNoteFiles/` y `ransomware_notes/` si hay `.pdf`; si no, buscar muestras en VirusTotal/repos.
- **Imagen/screenshot:** capturas de notas de CryptoLocker, TeslaCrypt, Cerber (hay muchas en
  BleepingComputer y en la galería de Hyphenet: https://hyphenet.com/ransomware-screenshots/).
- **HTML:** ya tenés varias (Cerber, etc.) en el corpus.

## C. Datasets/repos que requieren cuenta o descarga manual

| Recurso | Por qué vos | URL |
|---|---|---|
| Kaggle — Ransomware Note Dataset Collection | requiere login Kaggle | https://www.kaggle.com/datasets/abiprasanth/ransomware-note-dataset-collection |
| (ya hecho) lemmou/RansomNoteFiles | clonado ✅ | https://github.com/lemmou/RansomNoteFiles |

---

## Cómo seguimos
1. Guardás lo de la sección A (al menos Chimera, que es la única familia sin nota).
2. Me decís "listo" y corro OCR/extracción, con manifiesto de procedencia (familia, archivo, fuente, fecha).
3. Con tu OK, integro todo al corpus (sin pisar el original) y reentrenamos comparando macro-F1.
