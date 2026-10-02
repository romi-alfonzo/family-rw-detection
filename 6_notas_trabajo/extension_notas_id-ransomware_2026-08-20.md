# Extensión del frente de notas vía id-ransomware.blogspot (2026-08-20)

Tanda de recolección sobre las 5 familias que estaban trabadas (RYUK, NOTPETYA, WANNACRY,
CRYPTOLOCKER, WASTEDLOCKER). **Todo va como EXTENSIÓN al final del cap. 4; NO reemplaza cifras
canónicas** (decisión 2026-08-20). Corpus **151 → 155** notas (integridad 1:1 verificada).

## Método (seguro, reproducible)
- Fuente: `id-ransomware.blogspot.com` (Amigo-A / Andrew Ivanov), whitelisted (SANS).
- Lectura **sin ejecutar JavaScript** (`WebFetch` baja HTML y lo lee; no corre JS) y **sin bajar
  ejecutables/muestras**. Solo se descargaron **imágenes** (capturas `.jpg/.png` de CDN de
  Google/Twitter) a `3_datos/` (excluida de Defender). Nunca se abrieron enlaces internos de las notas.
- OCR de capturas: **subagente que lee la imagen en su contexto y devuelve solo texto** (no se
  muestran imágenes en el chat).
- Cada texto pasó por `2_codigo/verificar_nota_nueva.py` (coseno ≥ 0,90, char 3-5). Solo se
  incorporó lo que dio **TEXTO NUEVO**.
- **Redacción:** se redacta SOLO lo que lleva al ejecutable/payload (p. ej. el link Dropbox `.rar`
  de WannaCry → `[URL de descarga removida]`). Correos, BTC y `.onion` **de contacto** quedan
  verbatim (no llevan al payload; son marcadores como en el resto del corpus).

## Resultados
| Familia | Antes | Ahora | Detalle |
|---|---:|---:|---|
| **WANNACRY** | 2 | **4** ✅ | +2: `idr_wannacry_qa_2017.txt` (Q&A) y `idr_wannacry_screenlock_2017.txt`. Cerrada. |
| **RYUK** | 2 | **4** ✅ | +2 vía OCR de 18 capturas (abr 2019–ene 2021): `idr_ryuk_balance_2019.txt` (nota corta «balance of shadow universe») y `idr_ryuk_portal_2021.txt` (portal Tor 2021, `.onion` verbatim). Cerrada. |
| **NOTPETYA** | 2 | 2 | OCR de sus 2 capturas de nota → COPIA (0,916/0,920). Cuerpo idéntico, solo cambia la installation key. **Agotado confirmado (texto+imagen).** |

Descartes correctos (falsos positivos / homónimos):
- RYUK molde negociación oct-2019: coseno 0,880 pero era `pcrisk_ryuk_1` **recortada** en la captura → no se incorporó.
- RYUK: 12 de 18 capturas eran la misma nota corta con distinto email → colapsan en una.
- NOTPETYA: se excluyeron Petya-2016, GoldenEye, Mischa, CHKDSK falso y diagramas (otras familias / no-notas).

## CRYPTOLOCKER y WASTEDLOCKER — el techo es la FAMILIA, no la búsqueda
Verificado leyendo el corpus (2026-08-20):
- **WASTEDLOCKER = 1 plantilla real.** Sus 4 archivos (`note_pcrisk`, `_bba` = BBA Aviation,
  `_rlh` = RL Hudson, `wastedlocker.txt` = 88828@protonmail.ch) son **el mismo molde** de ~230-280
  car.: «YOUR NETWORK IS ENCRYPTED NOW / USE <emails> TO GET THE PRICE / DO NOT RENAME OR MOVE THE
  FILE / THE FILE IS ENCRYPTED WITH THE FOLLOWING KEY…». Solo cambia víctima y correos. Cualquier
  nota nueva colapsa. Los textos «distintos» atribuibles son **sucesores de Evil Corp**
  (Hades, Phoenix, PAYLOADBIN, Macaw, SecCrypt, Easy2lock) = **otras familias** (campaña-vs-familia).
  → No es falta de fuentes: la familia produce **una** nota. F1 0,000 es un **resultado**, no un hueco.
- **CRYPTOLOCKER = pocas plantillas por naturaleza.** El original 2013 **no dejaba archivo de nota**:
  mostraba una **ventana** («Your personal files are encrypted!») y guardaba la lista en el registro.
  El corpus ya tiene ~2-3 textos de esa ventana (`cryptolocker_note1`, `note2`, `pcrisk_cryptolocker_1`;
  note1≈pcrisk). Ojo homónimo: Crypt0L0cker = TorrentLocker (otra familia, esa sí deja DECRYPT_INSTRUCTIONS).
  - **Único lugar legítimo sin tocar:** `https://id-ransomware.blogspot.com/2020/12/cryptolocker.html`
    (Amigo-A) — la única fuente que **separa el original de los homónimos**. Sirve para verificar cuáles
    de nuestros 3 son auténticos y quizá sumar el mensaje de ventana original canónico. No llega a 4.

**Actualización (OCR hecho, 2026-08-20):** se leyeron ambas páginas (sin JS, solo imágenes, OCR por
subagente, verificación coseno).
- **CRYPTOLOCKER:** 8 capturas, todas el original 2013. La **nota principal (pantalla de bloqueo)
  coincide VERBATIM con la nuestra** (COPIA 0,986 vs `pcrisk_cryptolocker_1`) → **procedencia
  confirmada** (sale del 24% sin fuente; ahora citable = Amigo-A). Aparece la **ventana de PAGO**
  («Payment for private key / MoneyPak, Ukash, paysafecard, cashU, Bitcoin»): da TEXTO NUEVO (0,344),
  pero es **interfaz de pago, no una nota** → **decisión de alcance PENDIENTE** (análoga a Maze:
  wallpaper entra, sitio de pago no). +0 salvo que se decida contarla.
- **WASTEDLOCKER:** 3 capturas 2020 (BBA, GARMIN, una censurada) = **el mismo molde** verbatim; Garmin
  (correos 88828/47266) = COPIA 0,934 de `wastedlocker.txt`. Confirmado **1 sola plantilla**. Los
  sucesores con cuerpo distinto (SecCrypt, Phoenix CryptoLocker, PAYLOADBIN, Macaw, Easy2lock, Hades)
  son **otras familias** — no se incorporan (campaña-vs-familia). +0. F1 0,000 es **resultado**, no hueco.
- Imágenes fuente: `recoleccion_2026-08/{CRYPTOLOCKER,WASTEDLOCKER}/imagenes/`.

## Pendiente
- **Re-medir en OTRO chat, a carpeta NUEVA** (extensión): `curva_aprendizaje_notas.py` y
  `resumen_para_capitulo4.py --solo b1`. No sobrescribir los CSV canónicos.
- Capturas fuente guardadas en `3_datos/recoleccion_2026-08/{RYUK,NOTPETYA}/imagenes/`.
