# Dónde descargar más notas (todos los formatos) — 2026-06-22

Objetivo: sumar **profundidad** (más notas por familia) y **variedad de formato** (txt, html, imagen, pdf)
a las 30 familias. Reglas: solo las 30 familias; registrar la fuente de cada nota.
Claude no puede bajar binarios/imágenes ni datasets con login → esos los descargás vos.

---

## FUENTES MAESTRAS (varias familias, varios formatos)

| Fuente | Qué trae | Formato | Citable |
|---|---|---|---|
| **pcrisk.com** | guía por familia con **texto de la nota + captura** | texto + imagen | sí (autor + fecha) |
| **BleepingComputer** | artículos por familia con nota + screenshot | texto + imagen | sí |
| **Hyphenet (galería)** https://hyphenet.com/ransomware-screenshots/ | ~47 capturas de notas | imagen | medio (verificar) |
| **ThreatLabz** (ya clonado) | notas históricas | txt | sí (Zscaler) |
| **lemmou/RansomNoteFiles** (ya clonado) | notas por familia/versión | txt/html/hta/rtf/pdf | sí (paper Lemmou) |
| **Kaggle** (requiere login) https://www.kaggle.com/datasets/abiprasanth/ransomware-note-dataset-collection | notas para clasificación | txt | verificar licencia |
| **NapierOne** (tu dataset base) https://www.napierone.com/ | dataset de archivos + notas | varios | sí |
| **VX-Underground** https://vx-underground.org/ | archivo de malware y notas | varios | sí (verificar uso) |

> Cómo usar pcrisk: en cada guía hay el texto de la nota y/o una captura. Guardá **una sola
> versión por nota** — la que sea (texto O imagen), NO las dos. No duplicar el mismo contenido
> en dos formatos. El formato da igual; lo que importa es que cada nota sea distinta.

---

## URLs DIRECTAS por familia que necesita más notas (≤4 actualmente)

| Familia | Notas hoy | Enlace pcrisk (texto + captura) |
|---|---|---|
| AvosLocker | 4 | https://www.pcrisk.com/removal-guides/21388-avoslocker-ransomware |
| LockBit | 4 | https://www.pcrisk.com/removal-guides/16476-lockbit-ransomware (+ 2.0: /21605, 3.0: /24242) |
| Phobos | 4 | https://www.pcrisk.com/removal-guides/14258-phobos-ransomware |
| Ryuk | 4 | https://www.pcrisk.com/removal-guides/13394-ryuk-ransomware |
| Lorenz | 4 | https://www.pcrisk.com/removal-guides/21237-lorenz-ransomware |
| NetWalker | 3 | https://www.pcrisk.com/removal-guides/17729-netwalker-ransomware |
| SunCrypt | 3 | https://www.pcrisk.com/removal-guides/18646-suncrypt-ransomware |
| MeduzaLocker | 4 | familia MedusaLocker — variantes en pcrisk (Chip: /34945, Rapid: /28682, Stolen: /34139). Buscar "MedusaLocker" |
| BlackMatter | 3 | buscar en https://www.pcrisk.com/search → "BlackMatter" |
| CryptoLocker | 3 | buscar "CryptoLocker" (ojo: el original, no Crypt0l0cker) |
| Cuba | 3 | buscar "Cuba ransomware" |
| DarkSide | 3 | buscar "DarkSide" |
| HelloKitty | 4 | buscar "HelloKitty" |
| Maze | 3 | buscar "Maze ransomware" |

(Para las que dicen "buscar": pcrisk.com → ícono de lupa → nombre de la familia. Casi todas tienen guía.)

---

## Flujo de trabajo
1. Para cada familia, abrí su guía pcrisk: copiá el texto de la nota a un `.txt` y guardá la captura en `imagenes_notas/`.
   Nombrá con prefijo de fuente, p. ej. `pcrisk_avoslocker_1.txt` / `pcrisk_avoslocker_1.png`.
2. Para imágenes sueltas, usá Hyphenet o las capturas de BleepingComputer.
3. Cuando tengas un lote, decime "listo" → corro OCR/extracción de todo y armo el manifiesto de procedencia.
4. Con tu OK, integro al corpus (sin pisar el original) y reentrenamos midiendo macro-F1 antes/después.
