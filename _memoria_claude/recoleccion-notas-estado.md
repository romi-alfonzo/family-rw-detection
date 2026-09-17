---
name: recoleccion-notas-estado
description: Estado del frente de recolección de notas de rescate; dónde está el traspaso y qué NO tocar
metadata: 
  node_type: memory
  type: project
  originSessionId: 44976c18-bca6-4076-8fec-d9b45f54272c
  modified: 2026-08-20T14:52:16.514Z
---

Frente de recolección de notas para B.1 (conseguir textos de nota **distintos** en 9 familias).
Al retomar, leer **`6_notas_trabajo/HANDOFF_recoleccion_2026-08-20.md`** (entrada operativa vigente;
el `_2026-08-19.md` sigue útil para las URLs por familia y las reglas de fuentes) + el tablero
`6_notas_trabajo/lista_recoleccion_por_familia.md`.

Al 2026-08-20: **corpus 151 notas (manifiesto 151, coinciden 1:1), 8 textos nuevos**. Las 4 familias
accionables por texto están CERRADAS: MEDUZALOCKER y JIGSAW (4 c/u); CHIMERA (3, techo, nota alemana
`hns_chimera_aleman.txt`); MAZE (4, nota ChaCha `idr_maze_chacha_2019.txt` de id-ransomware/Amigo-A).
Resto agotado en texto (RYUK, NOTPETYA, WANNACRY → solo OCR/muestra viva) o bloqueado (CRYPTOLOCKER,
WASTEDLOCKER). Verificador canónico:
`2_codigo/verificar_nota_nueva.py` (umbral 0,90). Este frente es **solo recolección**: las
mediciones (`curva_aprendizaje_notas.py`) corren en OTRO chat.

**Bloqueos YA RESUELTOS:** (1) exclusión de Windows Defender de `C:\Users\Romina\Tesis\3_datos`
**puesta el 2026-08-20** (vía correcta: EXCLUIR, nunca evadir). (2) OCR: tesseract funciona
(autodetectado) y además un subagente puede leer la imagen en su contexto y devolver solo texto
(así se recuperaron los valores en rojo de la nota alemana de Chimera sin mostrarla en el chat).

**Problemas de integridad del corpus** (detalle en ESTADO_TESIS.md). **RESUELTO el 2026-08-19
(decisión de Romina):** las 3 notas mal etiquetadas por «familia equivocada» se **movieron** (no
borraron) a `3_datos/descartados_integridad/`: 2 de Medusa que estaban en MEDUZALOCKER (FBI/CISA
AA25-071A) y `lm_Crypt0l0cker_HOW_TO_RESTORE_FILES.html` que es TorrentLocker. **Corpus 150→147,
manifiesto 152→149; MEDUZALOCKER 6→4, CRYPTOLOCKER 4→3.** Al re-medir cambia la base. Siguen
pendientes (dependen del tutor): `chimera_note2.txt` sin fuente y las ~34 notas sin URL
rastreable. El 144-vs-146 quedó **RESUELTO el 2026-08-20**: las 2 `.hta` de DHARMA eran las variantes
**abibo** y **cmb** del repo Lemmou; estaban en el **git local** (commit `5c4455e`) y se restauraron sin
descargar (bytes originales, MD5 verificado) → `Info__3.hta`/`Info__13.hta`. Corpus 150, manifiesto y
disco coinciden 1:1. Repo Lemmou = `https://github.com/lemmou/RansomNoteFiles` (rama master).

Relacionado: [[tesis-ransomware-contexto]], [[comparacion-id-ransomware]] (ojo: el blog
id-ransomware.blogspot.com de Amigo-A NO es el servicio ID Ransomware de MalwareHunterTeam).
