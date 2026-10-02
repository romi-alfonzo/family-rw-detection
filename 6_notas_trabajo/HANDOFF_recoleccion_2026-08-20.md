# TRASPASO — Recolección de notas (2026-08-20)

> **Para el próximo chat:** leé esto + `ESTADO_TESIS.md` (bloques ★ y las ACTUALIZACIONES
> 2026-08-20) + el tablero `6_notas_trabajo/lista_recoleccion_por_familia.md`.
> Reemplaza como entrada operativa al `HANDOFF_recoleccion_2026-08-19.md` (que sigue válido
> para las URLs de descarga por familia y las reglas de fuentes en detalle).

## Estado actual (verificado contra disco)

- **Corpus: 151 notas.** Manifiesto: `3_datos/manifiesto_corpus_v2.csv`, **151 filas**.
  **Coinciden 1:1** (cross-check hecho: 0 filas sin archivo, 0 archivos sin fila).
- **Defender: `C:\Users\Romina\Tesis\3_datos` YA EXCLUIDA** (2026-08-20). Escribir `.txt`/`.hta`
  de notas conocidas ya no dispara cuarentena.
- **8 textos nuevos** recolectados en total en esta tanda (arrancó en 6 el 19-08).

## Las 9 familias de B.1 (objetivo 4 textos citables c/u)

| Familia | Citables | Estado |
|---|---|---|
| MEDUZALOCKER | 4 | ✅ cerrada |
| JIGSAW | 4 | ✅ cerrada |
| CHIMERA | 3 | ✅ cerrada por techo (texto agotado) |
| MAZE | 4 | ✅ cerrada (+ nota ChaCha, 2026-08-20) |
| RYUK | 2 | 🔻 texto agotado; única vía sin usar: id-ransomware.blogspot.com (opcional) |
| NOTPETYA | 2 | 🔻 agotado (OCR de la imagen dio copia) |
| WANNACRY | 2 | 🔻 agotado en fuentes citables (28 traducciones solo en la muestra) |
| CRYPTOLOCKER | ~2 | ⛔ bloqueada (el original no deja archivo de nota) |
| WASTEDLOCKER | 1 | ⛔ bloqueada (molde rígido) |

**Las 4 familias que podían dar texto citable están las 4 cerradas.** El resto no se destraba
descargando: es OCR de figuras / muestra viva (decisión del tutor) o familias bloqueadas.

## Qué se hizo hoy (no rehacer)

1. **Defender:** exclusión de `3_datos` (la puso Romina).
2. **Integridad — 3 notas mal etiquetadas movidas** (no borradas) a `3_datos/descartados_integridad/`
   (con README, reversible): 2 de Medusa que estaban en MEDUZALOCKER (FBI/CISA AA25-071A) y
   `lm_Crypt0l0cker_HOW_TO_RESTORE_FILES.html` (TorrentLocker). MEDUZALOCKER 6→4, CRYPTOLOCKER 4→3.
3. **CHIMERA +1:** nota alemana por OCR (`corpus_v2/CHIMERA/hns_chimera_aleman.txt`), Help Net
   Security. La inglesa de Malwarebytes salió COPIA.
4. **DHARMA:** las 2 `.hta` perdidas (`Info__3`/`Info__13`) eran las variantes abibo/cmb del repo
   Lemmou; se **restauraron del git local** (bytes originales, MD5 verificado). DHARMA 17→19.
5. **MAZE +1 → cerrada:** nota de la etapa **ChaCha** («0010 SYSTEM FAILURE 0010») desde
   `id-ransomware.blogspot.com` (Amigo-A, 2019-05-13). `corpus_v2/MAZE/idr_maze_chacha_2019.txt`.

## Pendientes

- **CHIMERA — vistazo humano (de Romina):** en `hns_chimera_aleman.txt`, la dirección BTC
  `1GaVKrVT17DN4dnWbTqGB9qG3rQrk1JBe9` y la URL `https://mega.nz/ChimeraDecrypter` vienen de OCR de
  visión. Confirmarlas contra la imagen (`recoleccion_2026-08/CHIMERA/imagenes/hns_chimera_03112015.jpg`)
  **antes de citarlas textualmente**. El monto `2,45267544 BTC` ya está corroborado. El manifiesto
  tiene la marca «revisar contra imagen antes de citarlos».
- **RYUK (opcional):** única fuente sin usar = `id-ransomware.blogspot.com` (buscar Ryuk). Publica
  notas como texto. Si se retoma, mismo flujo que MAZE.
- **Integridad (decisión del tutor, NO tocar sin ella):** `chimera_note2.txt` sin fuente; ~34 notas
  con fuente solo «NapierOne/varios» sin URL; y avalar el retiro de las 3 notas mal etiquetadas.
- **Re-medir en OTRO chat** (el corpus cambió): `python curva_aprendizaje_notas.py` y
  `python resumen_para_capitulo4.py --solo b1`.

## Reglas de trabajo de este frente

- **Flujo acordado hoy: Romina descarga a mano** (texto/imágenes), **Claude procesa** (verifica con
  `2_codigo/verificar_nota_nueva.py`, umbral 0,90, y si da «TEXTO NUEVO» copia a `corpus_v2/<FAM>/`
  y agrega fila al manifiesto con `extension_original` DOCUMENTADA).
- **Nunca evadir Defender**, solo excluir. **No recrear HTML** (guardar `.txt`, anotar la extensión).
- **Fuentes:** whitelist (pcrisk, bleepingcomputer, malwarebytes, sophos, helpnetsecurity, cis, cisa,
  ic3, id-ransomware.blogspot.com, github solo texto). No seguir redirects fuera de dominio; no bajar
  muestras/binarios/zip. Detalle en el handoff del 19-08 y en el tablero.
- **Este frente es solo recolección.** Las mediciones corren en otro chat.
- OCR sin mostrar imágenes en el chat: tesseract, o un subagente que lea la imagen en su contexto y
  devuelva solo texto (así se recuperó el verbatim de la nota alemana de Chimera).
