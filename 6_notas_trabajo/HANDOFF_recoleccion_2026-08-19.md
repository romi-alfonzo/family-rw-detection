# TRASPASO — Recolección de notas de rescate (2026-08-19)

> **Para el próximo chat:** leé esto + `ESTADO_TESIS.md` (sección «LOTE 1 RECOLECTADO») +
> `6_notas_trabajo/lista_recoleccion_por_familia.md`. Este archivo es el resumen operativo
> para que **Romina descargue a mano** y un chat nuevo continúe sin repetir trabajo.
> Tarea única de este frente: conseguir textos de nota **distintos** para 9 familias de B.1.

---

## 1. LO QUE YA ESTÁ HECHO (no rehacer)

**6 textos nuevos verificados. Corpus 144 → 150 notas.** Manifiesto: 152 filas.

| Familia | Antes | Ahora | Estado |
|---|---|---|---|
| MEDUZALOCKER | 3 | 4 legítimos | ✅ cerrada (+2) |
| JIGSAW | 2 | 4 | ✅ cerrada (+2) |
| MAZE | 2 | 3 | +1, falta 1 |
| CHIMERA | 2 | 3 (2 citables) | +1, falta 1 |
| RYUK | 2 | 2 | texto agotado |
| NOTPETYA / WANNACRY / CRYPTOLOCKER / WASTEDLOCKER | — | — | bloqueadas, ver §5 |

Archivos nuevos ya incorporados a `3_datos/corpus_v2/<FAMILIA>/` y anotados en
`3_datos/manifiesto_corpus_v2.csv` con fuente (vendor · autor · fecha · URL):
`pcrisk_medusalocker_chip.txt`, `pcrisk_medusalocker_rapid.txt`, `pcrisk_jigsaw_aleman.txt`,
`pcrisk_jigsaw_frances.txt`, `pcrisk_chimera_ingles_autentico.txt`, `pcrisk_maze_wallpaper.txt`.

Copias de todo lo recolectado (incluye descartes e imágenes) en
`3_datos/recoleccion_2026-08/` (ignorada por git).

**NO se corrió ninguna medición** (decisión de Romina: este frente es solo recolección).
Respaldo de las salidas de B.1 sobre 144 notas en
`4_resultados/_respaldo_b1_144notas_2026-08-19/`. Al cerrar cada lote, en OTRO chat:
`python curva_aprendizaje_notas.py` y `python resumen_para_capitulo4.py --solo b1`.

**Código commiteado a `develop`:** `verificar_nota_nueva.py`, `rigidez_plantillas.py`.
Sin commitear (documentos, como corresponde): `ESTADO_TESIS.md`, `CLAUDE.md`,
`lista_recoleccion_por_familia.md`, este archivo.

---

## 2. CÓMO VERIFICAR CADA NOTA (obligatorio antes de contarla)

```
cd C:\Users\Romina\Tesis\2_codigo
python verificar_nota_nueva.py <ruta_a_la_nota> [otra ...]
```

Dice «TEXTO NUEVO» o «COPIA de plantilla existente». Umbral 0,90 (coseno char 3-5), el mismo
criterio con que se midió todo el frente. Base: 150 notas → 100 textos distintos.

**Dos trampas ya medidas, no repetir:**
- **Nunca verificar sobre un texto truncado.** Un fragmento cortado da falso «texto nuevo»
  (pasó con JIGSAW: fragmento 0,595 = «nuevo», texto completo 0,965 = copia). Conseguir el
  verbatim COMPLETO primero.
- El conteo de textos distintos tiene ruido de ±1 al reajustar el TF-IDF; recalcularlo sobre
  el corpus tras cada lote, no sumar aritméticamente.

---

## 3. ⚠️ EL TEMA DEFENDER — LA VÍA CORRECTA ES EXCLUIR, NO EVADIR

**Qué pasa:** Windows Defender tiene firmas para el **texto** de varias notas de rescate
(detectó mis transcripciones de Chimera como `Ransom:HTML/Chicrypt.A`, sev. 5) y para los
`.hta` reales. Cuando el texto/archivo coincide, lo pone en cuarentena y lo borra.

**La solución soportada y correcta es una EXCLUSIÓN DE CARPETA. NO se debe evadir la
detección** (renombrar, cambiar codificación, comprimir con contraseña, partir el archivo):
eso es evasión de antivirus, no se hace, y además **alteraría los artefactos**, que es lo peor
para un corpus de tesis. La exclusión deja el archivo intacto y es lo que recomienda Microsoft
para carpetas de análisis de malware.

**Lo hace Romina** (config de Windows, Claude no la toca). Pasos:
1. Seguridad de Windows → Protección antivirus y contra amenazas → Administrar la configuración
2. Exclusiones → Agregar una exclusión → Carpeta
3. Elegir: `C:\Users\Romina\Tesis\3_datos`
4. (Opcional, para el material bajo código) también:
   `C:\Users\Romina\Tesis\2_codigo\family-rw-detection\ransom_notes_corpus`

Con eso, descargar `.hta` y transcribir notas a `.txt` deja de disparar cuarentena.
**Mientras no esté la exclusión, transcribir texto de notas conocidas va a fallar.**

**Verificado que la cuarentena NO afectó el repositorio ni el corpus:** 150/150 archivos
legibles; los `.hta` y notas afectadas están todos bajo `.gitignore` (`git ls-files` = 0).

### 3.1 Flujo automático imagen → texto (script nuevo)

`2_codigo/preprocesar_notas_imagen.py` — Romina descarga las capturas a mano, las deja en una
carpeta, corre el script y obtiene el texto por OCR + la metadata, listo para verificar. **No
toca el corpus ni el manifiesto.** Flujo:

1. Descargar imágenes a `3_datos/recoleccion_2026-08/<FAMILIA>/imagenes/`.
2. `python preprocesar_notas_imagen.py <esa_carpeta>` → crea `procedencia.csv` (plantilla).
3. Completar `procedencia.csv`: `familia, extension_original, fuente, url, idioma`
   (`idioma` = eng/deu/fra/spa o «eng+deu»). La **extension_original se pone desde la doc del
   vendor**, no se adivina.
4. Correr de nuevo → genera `<img>.ocr.txt` (el texto) + `candidatos_ocr.csv`.
5. `python verificar_nota_nueva.py <img>.ocr.txt`. Si es «TEXTO NUEVO», copiar a
   `corpus_v2/<FAMILIA>/` y agregar la fila al manifiesto con la `extension_original` documentada.

**Sobre «de qué tipo de archivo es»:** la extensión NO sale de los píxeles (una captura no
distingue .txt de .html). El script la toma del `procedencia.csv` (dato documentado) y aparte
calcula una `pista_visual` débil (ventana-app / documento-texto / html) solo como control de
consistencia. **No se «recrea» el .html**: guardar texto envuelto en marcado inventado sería
fabricar el artefacto; se guarda .txt y se registra la extensión aparte.

**ESTADO al cerrar el chat (2026-08-19) — EL OCR YA FUNCIONA:**
- ✅ **tesseract v5.4.0** (`C:\Program Files\Tesseract-OCR\`) + pytesseract. El script
  **autodetecta** el binario aunque no esté en el PATH (winget lo instaló silencioso, sin PATH).
- ✅ **Packs de idioma deu/fra/spa RESUELTOS.** winget instaló solo `eng`+`osd`, así que se
  bajaron `deu/fra/spa.traineddata` del repo oficial `tesseract-ocr/tessdata_fast` a
  `2_codigo/tessdata/` (junto con copias de eng/osd). El script apunta ahí con `TESSDATA_PREFIX`.
  Esa carpeta está en `.gitignore` (datos de herramienta, no van al repo).
- ✅ **Preprocesamiento afinado para notas de rescate:** invierte las imágenes de fondo oscuro
  (texto claro sobre negro), sin lo cual el alemán salía «Opfer»→«Opier». Validado: la nota
  alemana de Chimera sale legible y con diacríticos correctos (ö ü ä ß).
- ⚠️ **FALTA confirmar la exclusión de Defender** de `3_datos` (§3). Necesita admin, no se pudo
  verificar. Sin ella, cuando el script **escriba** el `.ocr.txt` de una nota conocida (Chimera
  dispara `Ransom:HTML/Chicrypt.A`), Defender lo borra. Por eso el OCR se validó **en memoria**,
  sin escribir a disco todavía.
- ⚠️ **El OCR da un BORRADOR ~95%, no verbatim.** Pierde valores en rojo de bajo contraste (p.
  ej. la dirección BTC y «2,45267544 Bitcoins» de la nota alemana no salieron) y comete algún
  error («Videos»→«Vidcos»). **Hay que corregirlo contra la imagen antes de que entre al
  corpus.** Es el uso normal de OCR para material difícil.

**Primera tarea concreta del próximo chat** (una vez confirmada la exclusión de Defender):
```
cd C:\Users\Romina\Tesis\2_codigo
python preprocesar_notas_imagen.py ..\3_datos\recoleccion_2026-08\CHIMERA\imagenes
# completar procedencia.csv (familia=CHIMERA, extension_original=.html, idioma=deu/eng, fuente/url)
# correr de nuevo -> genera los .ocr.txt
# corregir el .ocr.txt aleman contra la imagen (recuperar la dir BTC y el monto)
python verificar_nota_nueva.py ..\3_datos\recoleccion_2026-08\CHIMERA\imagenes\hns_chimera_03112015.ocr.txt
# si es TEXTO NUEVO: copiar a corpus_v2\CHIMERA\ y agregar fila al manifiesto (ext .html documentada)
```
La nota alemana trae dirección BTC y enlace mega.nz, distinta de `chimera_note1.txt` (onion) →
probable texto nuevo, cerraría CHIMERA.

Comandos de instalación de referencia (ya ejecutados, para reproducir en otra máquina):
```
winget install --id UB-Mannheim.TesseractOCR
pip install pytesseract
# packs de idioma (winget no los incluye): bajar de github.com/tesseract-ocr/tessdata_fast
#   deu/fra/spa.traineddata a 2_codigo/tessdata/  (+ copiar eng/osd de Program Files\...\tessdata)
```

### 3.2 Sobre «las imágenes se ven raras»

Las 3 imágenes que descargué son **capturas de pantalla de notas de rescate reales** (fondo
negro, texto rojo estilo consola) — se ven así porque ESO es una nota de rescate, no porque
sean peligrosas. Verifiqué sus bytes: son JPG/PNG legítimos (`FFD8FF` / `89504E47`), no
ejecutables. **Una imagen no puede infectar por descargarla.** Defender NO marca imágenes,
solo el texto de las notas y los `.hta`. Para descargar capturas a mano no hace falta ninguna
exclusión: abrir la URL en el navegador y «Guardar imagen como…».

---

## 4. PARA DESCARGAR A MANO — URLs EXACTAS Y QUÉ HACER CON CADA UNA

Regla de oro: **guardar el archivo en `C:\Users\Romina\Tesis\3_datos\recoleccion_2026-08\<FAMILIA>\`**
(ya excluida), y después pasarle `verificar_nota_nueva.py`. Solo si da «TEXTO NUEVO» se copia a
`corpus_v2/<FAMILIA>/` y se agrega la fila al manifiesto.

### CHIMERA — falta 1 (lo más cerca de cerrar)
Ya tenés en disco 3 imágenes en `recoleccion_2026-08/CHIMERA/imagenes/`. Dos son la nota, y son
**versiones distintas** de las que ya están en el corpus (traen dirección BTC y enlace mega.nz):
- `mwb_chimera_nota_ingles.png` — nota inglesa con «Amount: 0,93945085 Bitcoins»
- `hns_chimera_03112015.jpg` — nota alemana con «Forderung: 2,45267544 Bitcoins»

**Acción:** una vez puesta la exclusión, transcribir esas 2 imágenes a `.txt` (por OCR con
tesseract, o pidiéndole a un chat que las lea). La alemana con BTC parece texto nuevo (usa
mega.nz en vez de la dirección onion de `chimera_note1.txt`). El tipo de archivo original es
**`.html`** — confirmado por 3 fuentes: nombre `YOUR_FILES_ARE_ENCRYPTED.HTML` (pcrisk),
«HTML file dropped» (Malwarebytes), y la propia firma `Ransom:HTML/Chicrypt.A` de Defender.
Fuente citable de cada imagen:
- inglesa: Malwarebytes, hasherezade, 2015-12-09 — `malwarebytes.com/blog/news/2015/12/inside-chimera-ransomware-the-first-doxingware-in-wild`
- alemana: Help Net Security, Zeljka Zorz, 2015-11-03 — `helpnetsecurity.com/2015/11/03/chimera-crypto-ransomware-is-hitting-german-companies/`

### RYUK — falta la nota corta de 2019 (está identificada, es imagen)
Es la variante corta con un correo + «balance of shadow universe». Fuentes con la captura:
- CIS Security Primer – Ryuk (2020-01-10) — `cisecurity.org/insights/white-papers/security-primer-ryuk`
- BleepingComputer, «Ryuk Ransomware Is Making Victims Left and Right»
- **Fuente de texto pendiente y prometedora:** `id-ransomware.blogspot.com` (buscar Ryuk) y
  el paper de Virus Bulletin «Shinigami's revenge» (VB2019) — ver §6.

### MAZE — falta 1
pcrisk agotado (sus bloques ya están o son copias). Pendiente:
- `id-ransomware.blogspot.com` — buscar **Maze** y **ChaCha** (nombre anterior, la nota de esa
  etapa puede ser texto distinto)
- McAfee/Trellix «Ransomware Maze» y Sophos — verificar si dan texto o solo captura

### NOTPETYA — solo por OCR (imagen ya en disco)
`3_datos/fuentes_notas/imagenes_notas/mbr-ransom-note.jpg`. **Ya la transcribí: es COPIA de
`notpetya_note1.txt` (0,925), NO aporta.** No hay más texto público. Cerrada salvo muestra viva.

### WANNACRY — agotada en fuentes citables
Las 28 notas traducidas (`msg/m_*.wnry`) existen solo dentro de la muestra; ningún vendor las
publica como texto. Único zip que apareció estaba en `transfer.sh` (host caído) — **no bajar
zips de muestras.** Solo se consigue extrayéndolas de una muestra viva → decisión del tutor.

### CRYPTOLOCKER — bloqueada (no existe archivo de nota)
El original de 2013 mostraba una ventana, no dejaba archivo (verificado: Keith Jarvis,
SecureWorks, dic-2013). Cualquier `.txt` es de un homónimo. Si se decide transcribir la
ventana: `id-ransomware.blogspot.com/2020/12/cryptolocker.html` (Amigo-A, la única que
distingue el original). ⚠️ La nota `lm_Crypt0l0cker_HOW_TO_RESTORE_FILES.html` que YA está en
el corpus es **Crypt0l0cker = TorrentLocker**, otra familia.

### WASTEDLOCKER — bloqueada (molde rígido)
Sus 4 notas colapsan en 1 plantilla; cualquier nota nueva colapsa igual. No es falta de
esfuerzo. Los textos «distintos» atribuibles son de sucesores de Evil Corp (otra familia).

---

## 5. PROBLEMAS DE INTEGRIDAD DEL CORPUS (para el tutor, NO tocar sin decisión)

Detalle completo en `ESTADO_TESIS.md`. Resumen:

1. **MEDUZALOCKER:** 2 de sus notas (`note_threatlabz_!!!READ_ME_MEDUSA!!!*.txt`) son de
   **Medusa**, familia sin relación con MedusaLocker (FBI/CISA **AA25-071A**, 2025-03-12,
   `ic3.gov/CSA/2025/250312.pdf`). Sacarlas la deja en 4 legítimas.
2. **CHIMERA:** `chimera_note2.txt` no coincide con ninguna fuente (inglés que calca el alemán
   + contenido agregado; coseno 0,502 con el inglés de pcrisk). Contenido no falso (el «2.45
   BTC» está documentado), pero sin fuente rastreable.
3. **CRYPTOLOCKER:** una de sus 3 plantillas es TorrentLocker (ver §4).
4. **37 de 152 notas (24 %) tipo `corpus-existente`** tienen fuente solo «NapierOne/varios»,
   sin URL — concentradas en las familias problemáticas. Choca con la regla de «toda cifra con
   fuente citable». Método de validación probado: OCR de imagen independiente + umbral 0,90
   (así se confirmó `notpetya_note1.txt`). Prioridad: las 11 de WASTEDLOCKER, CHIMERA,
   MEDUZALOCKER, CRYPTOLOCKER.
5. **144 vs 146:** las 2 notas que faltan son `corpus_v2/DHARMA/Info__13.hta` e `Info__3.hta`,
   **puestas en cuarentena por Defender el 2026-08-04 y NO recuperables** (no hay copia en el
   proyecto; el repo de Lemmou tiene otras 9 `.hta` de Dharma, ya en el corpus). Para volver a
   146 habría que **rebajarlas de la cuarentena de Defender** (lo hace Romina) o re-descargar
   esas 2 variantes. La cifra oficial de la tesis es 146; las corridas del clúster corren sobre
   144 y lo declaran.

### Hallazgo aprovechable para la tesis
Defender trae firma para el **texto** de la nota de Chimera (`Ransom:HTML/Chicrypt.A`): una
transcripción en texto plano, sin nada ejecutable, alcanza para dispararla. Es corroboración
independiente, de un vendor AV, de que **el texto de la nota por sí solo identifica la
familia** — la premisa del frente de notas. Citable.

---

## 6. REGLAS DE FUENTES (por qué se confía o no en una URL)

Detalle en `lista_recoleccion_por_familia.md`. Lo esencial:
- **Solo se lee texto y se descargan imágenes. NO** muestras, binarios, `.zip` ni ejecutables.
- **No seguir redirects fuera del dominio.** Caso real: `malwiki.org` redirige (301) a
  `mufasatotoamanah.com`, dominio recomprado — **NO usar malwiki.org.**
- **Chequeo más fuerte: ¿la recomienda un tercero confiable?** `id-ransomware.blogspot.com`
  está listada por **SANS** («Recommended Sources for Ransomware Information», Nickels/Chapman,
  2021-07-02) → habilitada, pendiente de usar. Es subdominio de blogspot (Google), no caduca
  como un dominio propio.
- ⚠️ **`id-ransomware.blogspot.com` (blog de Amigo-A) ≠ ID Ransomware (MalwareHunterTeam)**, el
  servicio del 71,93 % de `Pruebas.xlsx`. Son cosas distintas, no confundir al citar.
- Lista blanca verificada: pcrisk, BleepingComputer, Malwarebytes, Sophos, SonicWall,
  HelpNetSecurity, CIS, CISA, IC3, GitHub (solo listados/texto). Trend Micro da 403.
- Repos en disco (fuente más segura, sin red): `3_datos/fuentes_notas/ransomware_notes`
  (ThreatLabz) y `.../RansomNoteFiles` (Lemmou).

### Truco para extraer el texto de una guía pcrisk sin traer basura del sitio
El bloque de la nota está en un `<blockquote>`. Con el navegador: `fetch(url)` → parsear →
tomar el `<blockquote>` (reemplazando `<br>` por saltos). Así se evita el texto comercial del
sitio. pcrisk defanguea URLs (`hxxp`), tapa marcadores con `-` y a veces mete etiquetas de
botones: se transcribe TAL CUAL, no se corrige (y se declara en la tesis).

---

## 7. QUÉ NO HACER (acordado en este chat)
- ❌ No evadir Defender ofuscando/renombrando/comprimiendo. Solo exclusión de carpeta (Romina).
- ❌ No mostrar imágenes en el chat (pedido de Romina). Para OCR: tesseract, o un subagente que
  lea la imagen en su propio contexto y devuelva solo texto.
- ❌ No commitear datos (`3_datos/` en `.gitignore`). No commitear documentos salvo que Romina
  lo pida. Solo código va a `develop` en el momento.
- ❌ No correr mediciones en este frente.
- ❌ No completar texto «a ojo». Verbatim de fuente citable o no entra.
- ❌ No tocar los 4 problemas de integridad del §5 sin decisión de Romina/tutor.
