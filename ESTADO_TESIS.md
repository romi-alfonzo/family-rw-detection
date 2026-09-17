# ESTADO DE LA TESIS — documento vivo

> **Cómo usar este archivo:** cuando abras un chat nuevo y se haya perdido el contexto,
> dile a Claude: *"lee ESTADO_TESIS.md en mi carpeta Tesis"*. Con eso retoma todo.
> Mantener actualizado al final de cada sesión.

_Última actualización: 2026-08-17_

> **⚠️ La carpeta fue reorganizada el 2026-08-04.** Ver `LEEME_ESTRUCTURA.md` para el mapa.
> Rutas nuevas (las de este documento que digan lo viejo, traducir así):
> `2_codigo/` scripts · `3_datos/corpus_v2/` corpus · `4_resultados/resultados_*/` salidas ·
> `1_documento/Plantilla_de_Tesis___Romina_Carlos/` tesis LaTeX ·
> `5_bibliografia/` papers (incluye `Leido/`) · `6_notas_trabajo/` los .md de fuentes ·
> `7_compartido_carlos/Tesis Carlos y Romina/` (ahí está `Pruebas.xlsx`) ·
> `_archivo/` material superado (v1 del clasificador, latex_capitulos, Plantilla inicial).
> Nada fue borrado.

---

## ★ SPRINT C / RECOLECCIÓN — HERRAMIENTA DE VERIFICACIÓN Y PRIMER DICTAMEN (2026-08-19)

Plan operativo: `6_notas_trabajo/plan_recoleccion_notas_2026-08-19.md`. Objetivo del lote:
**17 textos distintos en 9 familias** (prioridad 1 de B.1).

### ★★★ LOTE 1 RECOLECTADO — 6 TEXTOS NUEVOS VERIFICADOS (2026-08-19)

> **▶ PARA CONTINUAR EN OTRO CHAT: `6_notas_trabajo/HANDOFF_recoleccion_2026-08-19.md`** —
> resumen operativo con las URLs para descargar a mano, el tema Defender (excluir, no evadir),
> y los 4 problemas de integridad del corpus.
> **Tablero de seguimiento por familia, con lo agotado y lo pendiente:**
> `6_notas_trabajo/lista_recoleccion_por_familia.md`. Se trabaja una familia hasta cerrarla.
> Ese archivo también fija las **reglas de fuentes** y la lista blanca de dominios.

> 🔧 **ACTUALIZACIÓN 2026-08-19 (tarde): 3 notas mal etiquetadas retiradas del corpus.**
> Decisión de Romina (se resolvió el problema de integridad §5-1 y §5-3 en su parte de
> «familia equivocada»). Se **movieron** —no se borraron— a `3_datos/descartados_integridad/`
> (con README que documenta motivo y fuente; reversible):
> `note_threatlabz_!!!READ_ME_MEDUSA!!!.txt` y `_2.txt` (son de **Medusa**, FBI/CISA AA25-071A,
> no MedusaLocker) y `lm_Crypt0l0cker_HOW_TO_RESTORE_FILES.html` (es **Crypt0l0cker =
> TorrentLocker**). **MEDUZALOCKER 6→4** (sigue cerrada, 4 legítimas), **CRYPTOLOCKER 4→3**.
> **Corpus 150→147; manifiesto 152→149 filas.** `chimera_note2.txt` (sin fuente) se dejó en su
> familia: es problema de «falta URL», no de familia equivocada. Al re-medir en otro chat
> cambia la base: re-correr `curva_aprendizaje_notas.py` y `resumen_para_capitulo4.py --solo b1`.

> 🧩 **ACTUALIZACIÓN 2026-08-19 (tarde): CHIMERA +1 texto nuevo (la nota alemana).** Se completó
> el OCR de `hns_chimera_03112015.jpg` (Help Net Security, Zeljka Zorz, 2015-11-03). tesseract dio
> el cuerpo pero perdió los valores en rojo; un subagente leyó la imagen en su contexto y recuperó
> el verbatim completo: dir BTC `1GaVKrVT17DN4dnWbTqGB9qG3rQrk1JBe9`, monto `2,45267544 Bitcoins`
> (coincide con lo ya documentado en el traspaso → corroborado), URL `https://mega.nz/ChimeraDecrypter`.
> `verificar_nota_nueva.py`: **TEXTO NUEVO** (vecina `chimera_note1.txt`, coseno 0,785; la distingue
> mega.nz frente al `.onion`). Copiada como `corpus_v2/CHIMERA/hns_chimera_aleman.txt`, `.txt` con
> `extension_original=.html` documentada (NO se recreó el HTML: sería fabricar el artefacto). La nota
> inglesa de Malwarebytes salió **COPIA** de la de pcrisk (coseno 0,943), no aporta. **Corpus 147→148;
> manifiesto 149→150** (150 = 148 en disco + 2 filas fantasma de DHARMA Info__3/__13). ⚠️ La dir BTC y
> la URL exacta vienen de OCR de visión: conviene un vistazo humano final contra la imagen antes de
> citarlas textualmente (el monto ya está corroborado). CHIMERA queda en 3 textos citables (techo
> realista: el texto está agotado).

> 🧩 **ACTUALIZACIÓN 2026-08-20: MAZE +1 texto nuevo → CERRADA (4).** Nota de la etapa **ChaCha**
> (mayo 2019, precursora de Maze) desde `id-ransomware.blogspot.com` (Amigo-A / Andrew Ivanov,
> 2019-05-13, fuente whitelist SANS). Es la nota `DECRYPT-FILES.html` con título «0010 SYSTEM FAILURE
> 0010» y contacto `getmyfilesback@airmail.cc` — distinta de las notas Maze del corpus.
> `verificar_nota_nueva.py`: **TEXTO NUEVO** (vecina `pcrisk_maze_1.txt`, coseno 0,698). Guardada como
> `corpus_v2/MAZE/idr_maze_chacha_2019.txt`, `.txt` con `extension_original=.html` documentada; el blob
> base64 aparece **truncado con `***`** en la fuente y se transcribió tal cual (declararlo). **Corpus
> 150→151; manifiesto 151; coinciden 1:1.** Con esto las **4 familias accionables por texto están
> cerradas** (MEDUZALOCKER, JIGSAW, CHIMERA por techo, MAZE); el resto está agotado en texto (RYUK,
> NOTPETYA, WANNACRY → solo OCR/muestra viva) o bloqueado (CRYPTOLOCKER, WASTEDLOCKER).

> 🔍 **DHARMA — origen de las 2 `.hta` perdidas, identificado por hash (2026-08-19):** `Info__3.hta`
> e `Info__13.hta` son los `Info.hta` de las variantes **abibo** y **cmb** del repo Lemmou
> (`3_datos/fuentes_notas/RansomNoteFiles/Dharma/`). Prueba: las 9 `.hta` que sobreviven en el corpus
> calzan MD5 con 9 de las 11 variantes del repo (4k, Arrow, bip, bkp, brrr, manpecame, monro, skynet,
> stopencrypt); las 2 que no mapean son abibo (carpeta vacía) y cmb (le queda solo `FILES ENCRYPTED.txt`).
> Defender las puso en cuarentena **en el corpus Y en el repo de origen**, por eso no hay copia local.
> **✅ RESUELTO el 2026-08-20 (opción recuperar):** los 2 `.hta` estaban en el **git local** del repo
> (commit `5c4455e`), así que se restauraron **sin descargar**, con los bytes originales (MD5 verificado:
> abibo `687c8592…` → `Info__3.hta`; cmb `01d6de95…` → `Info__13.hta`; ambos distintos de los 9 previos).
> Se repusieron también en el repo de origen (`RansomNoteFiles/Dharma/abibo|cmb/Info.hta`). Cierra el gap
> 144-vs-146. **DHARMA 17→19; corpus 148→150; manifiesto y disco ahora coinciden 1:1 (150=150), sin filas
> fantasma** (verificado con cross-check). Nota: qué variante era originalmente `Info__3` vs `Info__13` se
> había perdido, así que esa asignación de números es arbitraria (abibo→3, cmb→13). Dharma es molde rígido
> → no cambia resultados.

> ⛔ **INCIDENTE DE FUENTE, anotarlo para no repetirlo:** `malwiki.org`, que aparece en
> búsquedas como fuente de notas de rescate, **responde 301 y redirige a
> `mufasatotoamanah.com`**, dominio sin relación con seguridad informática (parece dominio
> expirado y recomprado). **No se siguió el redirect y no se consultó.** En todo el chat se
> leyó **solo texto**: no se descargó ninguna muestra, binario, `.zip` ni imagen. Se
> descartaron por esta regla el `.zip` de las 28 notas de WannaCry alojado en `transfer.sh` y
> dos repos de muestras vivas de Jigsaw.

**Corpus: 144 → 150 notas.** Manifiesto actualizado (`3_datos/manifiesto_corpus_v2.csv`,
152 filas). Carpeta de aterrizaje con todo lo recolectado, incluidas las candidatas
descartadas: `3_datos/recoleccion_2026-08/<FAMILIA>/` (dentro de `3_datos/`, o sea ignorada
por git — confirmado con `git check-ignore`).

| Familia | Textos antes | Nuevos | Ahora | Objetivo B.1 |
|---|---|---|---|---|
| **MEDUZALOCKER** | 3 (1 de ellos de otra familia, ver abajo) | **+2** | 5 nominales / **4 legítimos** | ✅ 4 |
| **JIGSAW** | 2 | **+2** | **4** | ✅ 4 |
| **CHIMERA** | 2 (1 sin fuente rastreable, ver abajo) | **+1** | 3 nominales / **2 citables** | ✗ falta 1 |
| **MAZE** | 2 | **+1** | **3** | ✗ falta 1 |

**Lo incorporado, con procedencia citable:**

| Archivo | Familia | Fuente | Coseno con su vecina más cercana |
|---|---|---|---|
| `pcrisk_medusalocker_chip.txt` | MEDUZALOCKER | pcrisk · Tomas Meskauskas · 12-05-2026 · guía 34945 · nota `Recovery_README.html` | 0,781 |
| `pcrisk_medusalocker_rapid.txt` | MEDUZALOCKER | pcrisk · Tomas Meskauskas · 28-03-2024 · guía 28682 · nota `How_to_back_files.html` | 0,856 |
| `pcrisk_jigsaw_aleman.txt` | JIGSAW | pcrisk · Meskauskas · guía 9942 · variante alemana `.AFD` (actualización 06-06-2016) | 0,611 |
| `pcrisk_jigsaw_frances.txt` | JIGSAW | pcrisk · Meskauskas · guía 9942 · variante «Anti-Capitalist Jigsaw» `.fun` | 0,484 |
| `pcrisk_chimera_ingles_autentico.txt` | CHIMERA | pcrisk · Meskauskas · guía 9542 · nota `YOUR_FILES_ARE_ENCRYPTED.HTML` | 0,502 |
| `pcrisk_maze_wallpaper.txt` | MAZE | pcrisk · Meskauskas · guía 16145 «Maze 2019» · **fondo de escritorio** | 0,430 |

⚠️ **Criterio de alcance que se fijó al aceptar el wallpaper de MAZE, y hay que declararlo:**
entra lo que el malware **muestra o deja en la máquina de la víctima** (archivo, ventana
emergente, fondo de escritorio); **no entra el sitio web del atacante.** Por eso se aceptó el
wallpaper —el corpus ya tiene mensajes en pantalla: `wannacry_note1.txt` es la ventana de Wana
Decrypt0r y las 2 notas nuevas de JIGSAW son ventanas emergentes— y **se descartó** el tercer
bloque de la guía 16145, que es la página del sitio Tor de pago.

### ★★★ EL OCR NO DIO TEXTOS NUEVOS PERO SÍ ALGO MEJOR: VALIDÓ EL CORPUS

**Corrección a lo que se dijo antes en este chat:** se había escrito que el OCR estaba bloqueado
por falta de `tesseract`. **Es falso: las imágenes se pueden leer y transcribir directamente,
con mejor precisión que tesseract**, y sin instalar nada. Se hizo con las 2 imágenes relevantes
que **ya estaban en disco** (`3_datos/fuentes_notas/imagenes_notas/`), sin descargar nada.

**1) `mbr-ransom-note.jpg` (NotPetya, pantalla de arranque MBR) → NO aporta texto, pero valida.**
La transcripción completa resultó **COPIA de `notpetya_note1.txt`, coseno 0,925.** O sea:

- ✅ **`notpetya_note1.txt` queda CONFIRMADA como auténtica** contra una imagen independiente.
  Es la primera nota `corpus-existente` del proyecto con respaldo visual.
- ⚠️ **`notpetya_note2.txt` tiene un párrafo que NO está en la imagen:** «IMPORTANT: Do not
  attempt to remove the encryption software or modify any encrypted files. This will permanently
  destroy your data.» Y el resto es la misma nota reescrita en prosa más suelta.

**2) `Wana_Decrypt0r_screenshot.png` → corrobora `wannacry_note1.txt`** (el texto de la ventana
coincide) **y aporta evidencia visual directa de la localización**: en la esquina superior
derecha de la ventana **se ve el desplegable de idioma con «English» seleccionado**, que es
justamente el mecanismo de los 28 `msg/m_*.wnry`. **No se transcribió como nota nueva** porque
el panel tiene scroll y el texto está cortado abajo — y transcribir un texto cortado es
exactamente la trampa medida más arriba (el truncado da falso «texto nuevo»).

### ⛔⛔ Y ASÍ APARECIÓ EL PATRÓN: EL «SEGUNDO TEXTO» SIN FUENTE SE REPITE

**Dos familias, el mismo patrón, las dos con `tipo = corpus-existente` y fuente
«NapierOne/varios»:**

| Familia | note1 | note2 |
|---|---|---|
| NOTPETYA | ✅ auténtica (coseno 0,925 con la imagen) | ⚠️ reescritura + párrafo «IMPORTANT» que no está en la imagen |
| CHIMERA | ✅ auténtica (alemán, la nota era bilingüe) | ⚠️ reescritura del alemán + contenido agregado; coseno 0,502 con el inglés de pcrisk |

**Cuantificado sobre el manifiesto completo (152 filas):**

| `tipo` | Notas | Fuente citable |
|---|---|---|
| `bruto` | 96 | ✅ archivo de repositorio |
| `transcripcion` | 19 | ✅ URL + autor + fecha |
| **`corpus-existente`** | **37** | ⛔ **solo «NapierOne/varios», sin URL** |

**37 de 152 notas —el 24 % del corpus— no tienen fuente rastreable**, y se concentran justo en
las familias problemáticas: CRYPTOLOCKER 3 · WASTEDLOCKER 3 · MEDUZALOCKER 3 · LOCKBIT 4 ·
NOTPETYA, WANNACRY, CHIMERA, RYUK, JIGSAW, BADRABBIT, PHOBOS, AVOSLOCKER, HELLOKITTY 2 c/u.

> **Esto choca de frente con la regla del proyecto de que toda cifra que va a la tesis necesita
> fuente verificable y citable.** No es que las 37 sean falsas —`notpetya_note1.txt` acaba de
> quedar confirmada— pero **hoy no se puede citar su procedencia una por una.** Es un tema de
> metodología que hay que llevarle al tutor, y el camino de validación ya está probado: **OCR de
> una imagen independiente y verificación con el umbral de 0,90.**
>
> **Prioridad sugerida para validar:** las de las familias que se reportan con F1 por familia
> 0,000 (WASTEDLOCKER, CHIMERA, MEDUZALOCKER) y CRYPTOLOCKER, que además tiene el problema de
> homonimia. Son 11 notas.

⚠️ **OCR con tesseract (para el pipeline automático, no para esto):** `extractor_notas.py` hace OCR de
imágenes y PDF escaneado con **pytesseract**, pero en esta máquina **no está instalado
`tesseract`** ni están `pytesseract`/`cv2`/`easyocr` (solo `PIL`). Sin eso, **una imagen puesta
en `corpus_v2` no rompe: se omite con ADVERTENCIA y la nota no cuenta.** Lo instala Romina
(`winget install --id UB-Mannheim.TesseractOCR` y `pip install pytesseract opencv-python`).
Desbloquea CHIMERA (3 capturas identificadas) y NOTPETYA (imagen ya en disco).

Las cuatro son **transcripciones** (`tipo = transcripcion`), autorizadas por el tutor
(reunión 2026-08-12 punto 4, confirmado el 2026-08-16). **Se transcribió tal como lo publica
la fuente, sin corregir nada:** eso incluye el defanging de pcrisk (`hxxps://`), los
marcadores tapados (`-`) y, en la francesa, las etiquetas de los botones de la ventana
(`[View encrypted files]`), porque Jigsaw muestra su mensaje en una ventana y pcrisk rotula
el bloque como «Text presented in a pop-up window». **Declararlo así en la tesis.**

**ESTADO DE LAS 9 FAMILIAS DE LA LISTA, AL CERRAR ESTE LOTE:**

| Familia | Textos | Falta | Situación |
|---|---|---|---|
| MEDUZALOCKER | 4 legítimos | ✅ 0 | **cerrada** (y 2 notas de Medusa a sacar) |
| JIGSAW | 4 | ✅ 0 | **cerrada** (efecto incierto, ver el choque con B.3) |
| CHIMERA | 2 citables | 2 | 1 recolectada; `chimera_note2.txt` a resolver |
| MAZE | 2 | 2 | búsqueda en curso, sin resultado todavía |
| RYUK | 2 | 2 | búsqueda en curso, sin resultado todavía |
| WANNACRY | 2 | 2 | 🔻 **agotada en fuentes citables** (ver abajo) |
| NOTPETYA | 2 | 2 | sin explorar en este chat; solo vía OCR |
| CRYPTOLOCKER | 3 (1 dudosa) | 1 | ⛔ **no existe archivo de nota** (ver abajo) |
| WASTEDLOCKER | 1 | 3 | ⛔ molde rígido, 4 notas → 1 plantilla |

**Lo que queda para el próximo chat de recolección:** MAZE y RYUK (las dos con F1 bajo y con
bruto público, son las de mejor pronóstico), NOTPETYA por OCR de la imagen que ya está en
`3_datos/fuentes_notas/imagenes_notas/`, y el segundo texto de CHIMERA.

**⛔ NO SE CORRIÓ NINGUNA MEDICIÓN.** Decisión de Romina en este chat: el trabajo es solo
recolección. `curva_aprendizaje_notas.py` y `resumen_para_capitulo4.py --solo b1` **quedan
pendientes para otro chat**, así que **todavía no se sabe cuánto movió el macro-F1**. Las
salidas de B.1 sobre la base de 144 notas quedaron respaldadas en
`4_resultados/_respaldo_b1_144notas_2026-08-19/` (8 archivos) para no perder la base citable
al regenerarlas. Esa carpeta figura como no rastreada en git: **no commitearla**.

### ⚠️ EL CONTEO DE TEXTOS DISTINTOS NO ES PERFECTAMENTE ESTABLE

Al incorporar las 2 notas de MEDUZALOCKER el conteo pasó de **95 a 98** textos distintos,
no a 97. La diferencia de 1 **no es una nota**: es que el IDF del TF-IDF se ajusta sobre el
corpus, y al crecer el corpus un par que estaba al borde de 0,90 cambió de lado (el grupo de
5 notas de CERBER se parte en 2 + 3). Corre para los dos lados: agregar las candidatas de
JIGSAW llevaba los grupos del corpus de 98 a 97.

**Consecuencia práctica:** el número de textos distintos hay que **recalcularlo sobre el
corpus** después de cada lote, y tiene un ruido de ±1 por pares al borde del umbral. No
sumar aritméticamente «textos anteriores + nuevos». El verificador ya cuenta el aporte como
«grupos formados solo por candidatas», que es inmune a esta deriva, y avisa aparte cuando
detecta el corrimiento.

### ✅ CORRECCIÓN A LO QUE SE PREVIÓ EN ESTE MISMO CHAT: MEDUZALOCKER SÍ RINDE

Antes de medir se anticipó que MedusaLocker sería como WASTEDLOCKER, porque **las 10
variantes revisadas en pcrisk** (Rapid, Stolen, Chip, LockLock, Karma, Protect, Infected,
Crypto, Luck, End) **usan el mismo molde** «/!\ YOUR COMPANY NETWORK HAS BEEN PENETRATED /!\».
**La previsión era equivocada:** el molde varía lo suficiente para cruzar el umbral. De 5
candidatas salieron **2 textos distintos**, a coseno 0,773-0,856 contra la nota que ya
estaba — todas por debajo de 0,90. Los bloques que las separan son de contenido real, no de
formato: la existente trae las 4 instrucciones de Tor; un grupo trae «* Tor-chat to always be
in touch»; el otro agrega el párrafo «IMPORTANT! / middlemen / scams» y qTox.

> **La lección es del método, no de MedusaLocker: la rigidez del molde se ve a ojo, pero el
> umbral de 0,90 no. Hay que medir cada candidata, no descartar una familia por inspección.**
> Con WASTEDLOCKER la inspección y la medición coinciden (4 notas de 3 víctimas y 2 fuentes
> → 1 plantilla); con MEDUZALOCKER no coincidieron.

### ▶ HALLAZGO PARA EL RESTO DE LA RECOLECCIÓN: EL EJE PRODUCTIVO ES EL IDIOMA

Los 2 textos nuevos de JIGSAW no son variantes de contacto: son **la nota traducida a otro
idioma** (alemán y francés), y por eso dan los cosenos más bajos de todo el lote (0,611 y
0,484). Es el eje con más rendimiento por hora encontrado hasta ahora. pcrisk documenta para
Jigsaw variantes adicionales en **turco** (`.ram`), **portugués** y **polaco**, y reskins con
marca propia (Koolova, IT.Books, Ransomnix, «Different Jigsaw»).

### ⛔⛔ PERO EL EJE DEL IDIOMA CHOCA CONTRA B.3, Y CHIMERA ES LA PRUEBA

**No hay que salir a recolectar traducciones sin entender esto primero.** El eje que más
rinde para *contar* textos distintos es exactamente el que más **baja la cohesión**, y B.3
midió que la cohesión es el mejor predictor del F1 por familia (Spearman ρ **+0,704**,
p = 2,0·10⁻⁵). Los dos efectos tiran para lados opuestos.

**La prueba está en el corpus y no se había leído así.** Se abrieron las 2 notas de CHIMERA:

- `chimera_note1.txt` está **en alemán** («Sie wurden Opfer der Chimera Malware…»)
- `chimera_note2.txt` está **en inglés** («You became a victim of the Chimera malware…»)
- **Es el MISMO mensaje en dos idiomas.**

> **CHIMERA tiene la cohesión más baja de las 30 familias (0,1538) y F1 por familia
> 0,000 ± 0,000, y ahora se sabe por qué: sus dos «plantillas» son una sola nota traducida.**
> No es que Chimera varíe mucho su mensaje — es que el corpus guarda dos idiomas del mismo
> texto. El caso límite que B.3 citaba como «la familia con menos cohesión del corpus» tiene
> una explicación concreta, y es lingüística, no de comportamiento del malware.

**Consecuencia inmediata y honesta sobre las 2 notas de JIGSAW que se acaban de incorporar:**
son textos legítimos de la familia, verificados como distintos y con fuente citable, así que
se dejan. Pero **su efecto sobre el F1 por familia de JIGSAW es genuinamente incierto y podría
ser negativo**, porque replican la estructura que hundió a CHIMERA. Ya estaba anticipado en
este documento: «sumar un texto ayuda un poco a cualquier familia, pero **no convierte a una
familia de cohesión baja en una de cohesión alta**». **Lo resuelve la medición, que queda para
el otro chat: hay que mirar el F1 por familia de JIGSAW y su cohesión, antes y después.**

⚠️ **Y por eso NO se recolectaron más traducciones**, aunque pcrisk ofrece turco, portugués y
polaco de Jigsaw y sería lo más rápido de juntar. Sumar tres idiomas más a una familia que ya
llegó al objetivo de 4 arriesga empujarla hacia el patrón de CHIMERA sin ganancia medible
(B.1: el paso 4→5 vale +0,0006 de macro-F1, IC 95 % [−0,0207; +0,0219], indistinguible de
cero). **Decisión: frenar el eje idioma hasta que se mida el efecto de estas dos notas.**

### ⛔⛔⛔ HALLAZGO DE INTEGRIDAD: `chimera_note2.txt` NO COINCIDE CON NINGUNA FUENTE

Al abrir las notas de CHIMERA para entender su cohesión apareció algo peor que un problema de
etiqueta. **El inglés auténtico de la nota de Chimera, tal como lo publica pcrisk (guía 9542,
Tomas Meskauskas), está mal traducido del alemán:**

> «**Your are** victim of the Chimera malware. Your private files are encrypted and can not be
> restored without **a special edgy file**. Maybe some programs no longer function properly…
> If you don't pay your private data, which include pictures and videos will be published on
> the Internet **in relation on your name**.»

«Your are», «a special edgy file», «in relation on your name»: es traducción automática del
alemán, y por eso es creíble como artefacto real. **`chimera_note2.txt` del corpus dice otra
cosa, en inglés perfectamente correcto**, y calca la nota alemana frase por frase:

| `chimera_note1.txt` (alemán, en el corpus) | `chimera_note2.txt` (inglés, en el corpus) | pcrisk (inglés auténtico) |
|---|---|---|
| «Sie wurden Opfer der Chimera Malware.» | «You became a victim of the Chimera malware.» | «**Your are** victim of the Chimera malware.» |
| «…ohne eine spezielle Schluessel-Datei nicht wiederherstellbar.» | «…are not recoverable without a special **key** file.» | «…can not be restored without a special **edgy** file.» |
| «Moeglicherweise funktionieren einige Programme nicht mehr ordnungsgemaess!» | «Some programs may no longer function properly.» | «Maybe some programs no longer function properly:» |
| «Ihr Transaktions-Schluessel:» | «Your transaction key:» | (no aparece) |

Y además `chimera_note2.txt` **agrega contenido que no está en la nota alemana ni en pcrisk**:
«All your personal data, business documents, photos, and credentials will be made publicly
available. This includes data from browsers, email clients, and FTP applications», más
«Payment required: 2.45 BTC» y «Payment deadline: 7 days».

**Lo que se puede afirmar, sin sobreactuar:** `chimera_note2.txt` **no coincide con el texto
que publica la fuente citable**, su procedencia en el manifiesto es solo `corpus-existente /
NapierOne varios` (no rastreable a una URL), y su estructura es la de una **traducción del
alemán con contenido agregado**. Coseno con el inglés auténtico: **0,502**. No se puede
demostrar desde acá que sea fabricada, pero **no está en condiciones de respaldar una cifra de
la tesis hasta que se le encuentre fuente.**

**Y hay una pista documental fuerte a favor de la sospecha:** `6_notas_trabajo/descargas_pendientes.md`
registra a **Chimera como «⬜ FALTA (sin texto aún)»** y la llama «la única familia sin nota»,
mientras `notas_familias_criticas.md` guarda el texto auténtico de pcrisk como «✅ texto
verbatim conseguido (alta confianza)» — **que nunca se incorporó**. O sea: el corpus terminó
con dos notas de Chimera de origen no rastreable, y el texto citable que sí se había
conseguido quedó afuera. **Eso último ya está corregido en este lote.**

**Acción tomada:** se incorporó `pcrisk_chimera_ingles_autentico.txt` (texto nuevo verificado,
coseno 0,502 con `chimera_note2.txt`). CHIMERA pasa de 2 a **3 textos**, de los cuales **2 con
fuente citable**. **NO se tocó `chimera_note2.txt`** — sacarla es decisión de Romina y el
tutor, y cambia las cifras del capítulo 4.

#### ✅ CORRECCIÓN AL PÁRRAFO DE ARRIBA — ERA DEMASIADO DURO, Y LA FUENTE LO ACLARA

Al seguir buscando aparecieron dos datos que **matizan la sospecha**, y corresponde dejarlos
escritos con el mismo énfasis:

1. **La nota de Chimera era bilingüe por diseño.** hasherezade, Malwarebytes Labs, 09-12-2015:
   «there is an HTML file dropped… The HTML can be displayed in two languages – English and
   German». O sea que tener una nota en alemán y otra en inglés **no es un artefacto del
   corpus: es cómo venía el archivo.** El par alemán/inglés es legítimo.
2. **El dato de «2.45 BTC» de `chimera_note2.txt` está documentado por vendors**, no inventado:
   Help Net Security (Zeljka Zorz, 03-11-2015) y Trend Micro reportan el pedido de 2,45
   bitcoin. Lo mismo el robo de credenciales y la amenaza de publicación.

**Lo que sigue en pie:** `chimera_note2.txt` **no coincide con el inglés que publica pcrisk**
(coseno 0,502) y su procedencia no es rastreable a una URL. Lo más probable es que sea una
**traducción del alemán armada con datos de reportes de vendors**, no un artefacto capturado.
**Sigue necesitando fuente antes de respaldar una cifra**, pero **ya no hay razón para
sospechar que el contenido sea falso.**

> ⚠️ **Y sobre B.3, la lectura correcta es esta:** la cohesión de CHIMERA de **0,1538, la más
> baja de las 30**, y su F1 por familia **0,000 ± 0,000**, se explican porque sus dos plantillas
> son **el mismo mensaje en dos idiomas** — y eso es una **propiedad real de la familia**, no un
> error de corpus, porque el malware mandaba las dos. **CHIMERA sigue siendo citable como caso
> límite de cohesión baja, pero hay que decir POR QUÉ es baja: es bilingüe.** Eso es un hallazgo
> mejor y más defendible que «es la familia que menos se parece a sí misma», y conecta directo
> con el choque idioma-vs-cohesión de más arriba.

#### 🔻 CHIMERA: AGOTADA EN FUENTES CON TEXTO

Revisadas todas las fuentes técnicas de la familia: **la única con texto seleccionable es
pcrisk (guía 9542), y ya está incorporada.** Las demás publican la nota **solo como captura**:

| Fuente | Autor · fecha | Formato |
|---|---|---|
| Malwarebytes Labs, «Inside Chimera Ransomware — the first doxingware in wild» | hasherezade · 09-12-2015 | 🖼️ captura (versión inglesa del HTML) |
| SonicWall, «Chimera Ransomware uses Bitmessage over TOR» | 23-10-2015 | 🖼️ captura (Figura 6) |
| Help Net Security | Zeljka Zorz · 03-11-2015 | 🖼️ captura |
| Trend Micro | — | responde 403 a descarga automática |

**Única vía restante para el 2º texto: OCR de esas capturas**, que el tutor autorizó (reunión
2026-08-12 punto 4). **Requiere que Romina habilite la descarga de las imágenes** — en este
chat solo se leyó texto, no se bajó ningún archivo.

### 🔻 RESULTADO NEGATIVO DOCUMENTADO: WANNACRY

WannaCry **sí** localiza su nota: la muestra trae **28 archivos** `msg/m_*.wnry` (m_bulgarian,
m_chinese simplificado y tradicional, m_croatian, m_czech, m_danish, m_dutch, m_english,
m_filipino, m_finnish, m_french, m_german, m_greek, m_indonesian, m_italian, m_japanese,
m_korean, m_latvian, m_norwegian, m_polish, m_portuguese, m_romanian, m_russian, m_slovak,
m_spanish, m_swedish, m_turkish, m_vietnamese). En principio sería la familia más rica del
corpus, y con **archivos brutos**, no transcripciones.

**No se pudo obtener ninguno, y la razón hay que declararla:**
1. Los dos textos en inglés **ya están en el corpus**: `wannacry_note1.txt` es la ventana de
   Wana Decrypt0r («What Happened to My Computer?») y `wannacry_note2.txt` es el
   `@Please_Read_Me@.txt` («Ooops, your important files are encrypted»).
2. **Ningún vendor publica las versiones traducidas como texto.** Se revisaron fuentes en
   español y en alemán: todas describen la nota o la muestran en captura, ninguna transcribe
   la versión localizada. Los sitios hermanos de pcrisk en otros idiomas (p. ej. `dieviren.de`)
   traducen **el artículo**, no la nota: el bloque que publican sigue siendo el inglés.
3. El repositorio `Ruddernation-Designs/WannaCry-Decompiled` **no** trae la carpeta `msg`
   (solo `README.md`, `decryptor.c` y `worm.c`).
4. El único enlace a las 28 notas que apareció es un **.zip en `transfer.sh`**, host
   discontinuado, citado en un *factsheet* de terceros. **No se descargó**, y no por el enlace
   roto: no se bajan comprimidos ni muestras de repositorios de malware. Se leyó **solo texto**
   en todo el chat.

> **Conclusión para la tesis: las 28 notas localizadas de WannaCry existen dentro de la
> muestra pero no están disponibles como texto citable.** Para conseguirlas habría que
> extraerlas de una muestra viva, que es una decisión de Romina y el tutor, no de este chat.
> Mientras eso no pase, **WANNACRY queda en 2 textos y su F1 por familia de 0,010 ± 0,100 no
> es un problema de esfuerzo de búsqueda.**

### ⏸️ CANDIDATA EN ESPERA, POR ATRIBUCIÓN DÉBIL

`3_datos/recoleccion_2026-08/JIGSAW/pcrisk_jigsaw_hacked.txt` (237 caracteres, «YOUR COMPUTER
HAS BEEN ENCRYPTED YOU MUST PAY .25 BITCOINS…») **verifica como texto nuevo** (coseno 0,601
con `jigsaw_note1.txt`) pero **NO se incorporó**: pcrisk la describe como «another ransomware
infection **based on the source code of** jigsaw ransomware», que es atribución más débil que
las otras dos, a las que llama «variant of Jigsaw ransomware». Es el mismo criterio que
descarta las notas de Medusa: **hace falta que la fuente diga que es la familia, no que
derive de su código.** Queda a decisión de Romina y el tutor.

### `2_codigo/verificar_nota_nueva.py` — el filtro de entrada, ya funcionando

Recibe una o varias notas candidatas y dictamina **TEXTO NUEVO** o **COPIA de plantilla
existente** (y de cuál). Reusa `agrupar_neardups()` de `clasificador_notas_v2.py` con
`UMBRAL_NEARDUP = 0,90` — el mismo criterio con el que se midió todo el frente de notas,
sin criterio propio nuevo. Línea base verificada al correrlo: **144 notas → 95 textos
distintos**, que es la cifra ya registrada. No toca el corpus ni el manifiesto: solo lee.

**Detalle del criterio que hay que declarar** (lo detectó el propio script y se dejó
avisado en la salida): el TF-IDF se reajusta con las candidatas adentro, así que el IDF se
mueve y **pares del corpus al borde del umbral pueden cambiar de lado**. Con las dos
candidatas de JIGSAW, el grupo de 5 notas de CERBER se partió en 2 + 3 y los grupos del
corpus pasaron de 95 a 96 sin que ninguna candidata sea de CERBER. El script cuenta el
aporte como «grupos formados solo por candidatas» —inmune a esa deriva— y reporta el
corrimiento aparte. **La cifra oficial de textos distintos se recalcula sobre el corpus una
vez incorporadas las notas, nunca se lee de esta salida.**

### ▶ PRIMER DICTAMEN: de las 2 variantes de JIGSAW ya transcriptas, solo 1 aporta

`6_notas_trabajo/notas_familias_criticas.md` traía texto verbatim para JIGSAW (fuente:
BleepingComputer, «Jigsaw Ransomware Decrypted», 2016-04-11, notas de MalwareHunterTeam).
Pasadas por el verificador contra el corpus de 144 notas:

| Candidata | Veredicto | Coseno con la vecina |
|---|---|---|
| Variante 1 («Your computer files have been encrypted…») | **COPIA** de `JIGSAW/jigsaw_note1.txt` | **0,912** (> 0,90) |
| Variante 2 («I want to play a game with you…») | **TEXTO NUEVO** | 0,595 con `JIGSAW/jigsaw_note2.txt` |

**Aporte real: 1 texto nuevo, no 2.** La variante 1 ya está en el corpus: el archivo de
trabajo la anotaba como conseguida sin haberla cruzado contra lo que ya había. Es
exactamente el error que la regla de B.1 previene, y apareció en la primera candidata que
se revisó.

⚠️ **La variante 2 tampoco se incorpora, y por una razón distinta a la que se creyó primero.**
El texto de `notas_familias_criticas.md` estaba cortado con «…» (191 caracteres, tres líneas),
así que se buscó el verbatim completo en pcrisk antes de incorporarlo. **Con el texto completo
resulta ser COPIA de `JIGSAW/jigsaw_note2.txt`, a coseno 0,965.** Ya estaba en el corpus.

> ### ⚠️⚠️ TRAMPA METODOLÓGICA MEDIDA, Y VALE PARA TODA LA RECOLECCIÓN
> **Un fragmento truncado puede dar un falso «texto nuevo».** El mismo contenido dio:
>
> | Qué se le pasó al verificador | Veredicto | Coseno con `jigsaw_note2.txt` |
> |---|---|---|
> | Fragmento de 191 caracteres | «TEXTO NUEVO» | 0,595 |
> | **Texto completo de 920 caracteres** | **COPIA** | **0,965** |
>
> El verificador compara lo que se le da, no lo que la nota es. Truncar baja la similitud y
> disfraza una copia de texto nuevo. **Regla: nunca verificar sobre un extracto. Conseguir
> el verbatim completo primero, verificar después.** Las dos entradas de JIGSAW del archivo
> de trabajo resultaron ser copias de lo que ya había — ninguna de las dos aportaba.

### ★★★ RIGIDEZ DE PLANTILLA: EL CRITERIO QUE FALTABA PARA ORDENAR LA RECOLECCIÓN

Sonda local sobre el corpus (mismo criterio canónico, char_wb 3-5, umbral 0,90): cómo
agrupan hoy las notas de cada familia prioritaria y **cuánto se parecen las notas que la
familia publica con distinto contenido variable**. Mide algo que ni B.1 ni B.3 miraban:
**la probabilidad de que una nota nueva de esa familia colapse contra las que ya están.**

| Familia | Notas | Plantillas | Coseno máx. entre plantillas | Largo (caracteres) | Colapso observado |
|---|---|---|---|---|---|
| **WASTEDLOCKER** | 4 | **1** | — | 230-277 | **4 → 1** (coseno mínimo del componente 0,894) |
| CHIMERA | 2 | 2 | 0,154 | 610-730 | no |
| MAZE | 3 | 2 | 0,413 | 1449-2023 | 2 → 1 a coseno 0,985 |
| **MEDUZALOCKER** | 4 | **3** | 0,462 | 1391-3911 | 2 → 1 a coseno 0,943 |
| WANNACRY | 2 | 2 | 0,445 | 516-1680 | no |
| RYUK | 4 | 2 | 0,439 | 710-1981 | **3 → 1** a coseno 0,986 |
| CRYPTOLOCKER | 4 | 3 | 0,440 | 920-943 | 2 → 1 a coseno 0,988 |
| JIGSAW | 2 | 2 | 0,507 | 804-1000 | no |
| NOTPETYA | 2 | 2 | **0,817** | 749-846 | no |

> **WASTEDLOCKER usa UNA plantilla rígida y eso probablemente la vuelve irrecolectable.**
> Sus 4 notas vienen de **3 víctimas distintas** (`BBA Aviation`, `RL Hudson`, y una con
> correos `88828@PROTONMAIL.CH | 47266@AIRMAIL.CC`) y de **2 fuentes distintas** (pcrisk y
> ThreatLabz), y **las 4 colapsan en una sola plantilla**. El molde es de ~250 caracteres
> («YOUR NETWORK IS ENCRYPTED NOW / USE … TO GET THE PRICE FOR YOUR DATA / … THE FILE IS
> ENCRYPTED WITH THE FOLLOWING KEY») y lo único que varía es el nombre de la víctima y los
> correos. **Cualquier nota de WastedLocker que aparezca va a ser ese mismo molde y va a
> colapsar.** Su F1 por familia 0,000 no es un problema de cantidad de material: es que la
> familia no produce textos distintos.

**Consecuencia, y da vuelta el orden del plan operativo.** El plan del 2026-08-19 pone a
WASTEDLOCKER primera porque con 1 plantilla es inevaluable bajo P2ret (F1 0 por
construcción, límite declarado 4 de B.1) y porque el tramo 1→2 es el más empinado de la
curva (+0,1626 de macro-F1, IC 95 % [+0,1240; +0,2012], sobre el subconjunto de 5 familias;
+0,1254 [+0,1009; +0,1499] sobre el de 11 — **macro-F1 de subconjunto, no F1 por familia**).
Ese razonamiento sigue siendo correcto en valor, pero **el valor por texto no sirve si el
rendimiento por hora de búsqueda es cero.** WASTEDLOCKER pasa de primera a caso a documentar.

⚠️ **Y hay una trampa a evitar en WASTEDLOCKER:** los textos realmente distintos que se le
podrían atribuir pertenecen a los sucesores renombrados del mismo grupo (Evil Corp), no a
WastedLocker. Meterlos sería etiquetar otra familia como WASTEDLOCKER — la misma trampa
campaña-vs-familia del Exp. 2d. **Antes de escribir esto en la tesis hay que confirmar los
nombres y las fechas con fuente citable; acá queda como hipótesis, no como dato.**

### ▶ POR DÓNDE EMPEZAR: MEDUZALOCKER

1. **Cierra con un solo texto** (tiene 3, el objetivo es 4).
2. **Tiene margen para mejorar:** F1 por familia 0,000 ± 0,000 (P2ret, k=todo, 144 notas,
   100 repeticiones). No es como DARKSIDE/NETWALKER/SUNCRYPT, que ya están en 1,000.
3. **Verificado que la familia SÍ varía su texto:** 4 notas → 3 plantillas, coseno máximo
   entre plantillas 0,462, y notas de 1391 a 3911 caracteres de prosa. Es el opuesto exacto
   de WASTEDLOCKER: acá una nota nueva tiene chance real de no colapsar.
4. **Sin OCR:** hay material bruto público (ThreatLabz ya aportó 2 de sus notas) y no
   depende de transcribir capturas.

**Orden propuesto detrás:** MAZE (2 textos, F1 0,000, notas largas, bruto público) → RYUK
(2 textos, F1 0,032, pero su molde corto ya colapsó 3 notas: buscar versiones largas) →
JIGSAW (ya hay 1 texto verificado como nuevo, solo falta completarlo contra la fuente) →
CRYPTOLOCKER (1 texto, pero ver la advertencia de abajo) → WANNACRY y NOTPETYA (solo OCR;
NOTPETYA además ya está en 0,695 y sus dos plantillas tienen coseno 0,817 entre sí) →
CHIMERA última (cohesión 0,1538, la peor de las 30).

### ⛔ HALLAZGO GRAVE: 1 DE LAS 3 PLANTILLAS DE MEDUZALOCKER ES DE OTRA FAMILIA

Al ir a buscar material para MEDUZALOCKER apareció esto, y hay que resolverlo antes de
sumarle un texto. **Verificado por hash MD5**, no por parecido de nombre:

| Nota del corpus | MD5 (12) | De dónde salió realmente |
|---|---|---|
| `HOW_TO_RECOVER_DATA.html` | 47B66D8AC466 | ThreatLabz **`medusalocker/`** ✅ |
| `note_pcrisk.txt` | 7072AD12C571 | pcrisk, texto «All your data are encrypted!» con correos `Folieloi@protonmail.com` / `Ctorsenoria@tutanota.com` ✅ |
| `note_threatlabz_!!!READ_ME_MEDUSA!!!.txt` | 16CBE088F88F | ThreatLabz **`medusa/`** ⛔ |
| `note_threatlabz_!!!READ_ME_MEDUSA!!!_2.txt` | F2248CE174E9 | ThreatLabz **`medusa/`** ⛔ |

**El repo de ThreatLabz mantiene `medusa/` y `medusalocker/` como carpetas separadas**, y las
dos notas `!!!READ_ME_MEDUSA!!!` salieron de `medusa/`. **Medusa y MedusaLocker son familias
distintas**, y la fuente es de máxima autoridad:

> «The Medusa ransomware variant is unrelated to the MedusaLocker variant and the Medusa
> mobile malware variant per the FBI's investigation.»
> — FBI / CISA / MS-ISAC, *#StopRansomware: Medusa Ransomware*, **AA25-071A**, 12-03-2025,
> pág. 2. PDF primario accesible en `https://www.ic3.gov/CSA/2025/250312.pdf`
> (la página de CISA responde 403 a descarga automática).

Y el alias que usa la tesis ya está fijado en `PLAN_MEJORAS.md:234`: **MEDUZALOCKER =
MedusaLocker**. O sea que las dos notas de Medusa **no corresponden a la familia**.

**Efecto medido:** esas 2 notas colapsan entre sí en 1 plantilla (coseno 0,943), así que de
las **3 plantillas de MEDUZALOCKER, 1 es de otra familia** — un tercio de la clase. Su F1 por
familia 0,000 ± 0,000 (P2ret, k=todo, 144 notas, 100 repeticiones) tiene ahora una
explicación candidata que **no es** falta de material: se le pide al modelo aprender una
clase que contiene dos familias sin relación.

**NO se toca en este chat.** Sacar esas 2 notas baja el corpus a 142 notas y cambia el conteo
de plantillas, o sea todas las cifras del frente de notas y del capítulo 4. Es decisión de
Romina y del tutor. Pero **cambia la recolección ahora mismo**: MEDUZALOCKER pasa de «le
falta 1 texto para llegar a 4» a «tiene 2 plantillas legítimas y le faltan 2», y el material
que se busque tiene que ser **MedusaLocker verificado por nombre en la fuente**, nunca Medusa.

⚠️ **Y hay una trampa gemela para el resto de la búsqueda:** varios nombres de variante que
figuran en `mas_notas_descarga.md` como «variantes de MedusaLocker» (Chip, Rapid) coinciden
con nombres de familias **independientes y anteriores** que están en el repo de Lemmou como
carpetas propias (`Chip/CHIP_FILES.txt`, `Rapid/DECRYPT.[].txt`). No se incorpora ninguna
nota por coincidencia de nombre de variante: hace falta que la fuente diga **MedusaLocker**.

### ⚠️ HALLAZGO COLATERAL A RESOLVER ANTES DE TOCAR CRYPTOLOCKER

Una de las 3 plantillas de CRYPTOLOCKER es
`lm_Crypt0l0cker_HOW_TO_RESTORE_FILES.html`, o sea **Crypt0l0cker, que es TorrentLocker y
no el CryptoLocker original de 2013**. El propio plan operativo advierte «ojo: el original
de 2013, no Crypt0l0cker», pero la nota ya está adentro del corpus contada como plantilla
de CRYPTOLOCKER. Sumarle un 4º texto a esa familia es construir sobre una etiqueta dudosa.
**No se toca en este chat** (mover una nota del corpus cambia las cifras del capítulo 4):
queda anotado para decidirlo con el tutor, y explica parte de su F1 por familia de
0,410 ± 0,456 — el desvío es enorme justamente porque las plantillas no son de la misma cosa.

#### ⛔ Y HAY UNA RAZÓN ESTRUCTURAL: EL CRYPTOLOCKER ORIGINAL NO DEJABA ARCHIVO DE NOTA

Verificado por mí en la fuente técnica primaria — **Keith Jarvis, Dell SecureWorks CTU,
diciembre de 2013**, hoy alojado en `https://www.sophos.com/en-us/research/cryptolocker-ransomware`
(el URL viejo de secureworks.com redirige ahí):

- **El mensaje se mostraba en una ventana de la aplicación, no en un archivo**: «The victim is
  presented with a splash screen containing instructions and an ominous countdown timer».
- **La lista de archivos cifrados iba al registro de Windows**, no a un archivo de texto: «the
  malware stores the location of every encrypted file in the Files subkey of the
  HKCU\SOFTWARE\CryptoLocker (or CryptoLocker_0388) registry key».
- El contenido cifrado **reemplaza el archivo original** en disco.

> **Consecuencia dura para la recolección: para el CryptoLocker original de 2013 NO EXISTE
> archivo bruto de nota, y no puede existir.** Cualquier «nota de CryptoLocker» que aparezca
> como `.txt` o `.html` en un repositorio es, por construcción, de un homónimo. La única vía
> es la transcripción de la ventana, y hay que **declarar en la tesis que la fuente es una
> transcripción de GUI y no un archivo de nota.** Esto va más allá de «no tiene bruto
> público» (ESTADO_TESIS.md, lista de 5 familias): es que el artefacto no existe.

Las dos fuentes con texto seleccionable de esa ventana, para cuando se decida transcribirla:
`https://id-ransomware.blogspot.com/2020/12/cryptolocker.html` (Amigo-A / Andrew Ivanov,
distingue explícitamente el original de los homónimos — la más confiable) y
`https://www.pcrisk.com/removal-guides/7327-cryptolocker` (⚠️ esta le atribuye al original la
extensión `.encrypted`, lo que sugiere contaminación con un homónimo: **no usarla sola**).
Las demás fuentes revisadas publican la ventana **solo como captura**: BleepingComputer
(Lawrence Abrams, 14-10-2013), Sophos/SecureWorks, Softpanorama.

⚠️ **PREGUNTA ABIERTA QUE NO ME CORRESPONDE RESOLVER EN ESTE CHAT, pero que no puedo dejar
sin anotar.** El frente de archivos cifrados (cerrado) lista a **CRYPTOLOCKER entre las 26
familias con «extensión fija»** (ver la tabla más abajo en este documento). Si el original de
2013 reemplazaba el archivo sin agregar extensión, entonces o los archivos de CRYPTOLOCKER de
NapierOne son de un homónimo, o el comportamiento de extensión del original es distinto de lo
que se supone. **Ojo: la fuente de SecureWorks NO afirma explícitamente que no agregara
extensión** —eso apareció en la búsqueda secundaria y NO lo pude confirmar en fuente
primaria—, así que esto queda como **pregunta a verificar**, no como hallazgo. No toqué nada
del frente de archivos. Es para Romina y el tutor.

## ★★★ B.3 — EL GRAFO YA DIO EL DIAGNÓSTICO CLAVE (2026-08-19, local, `--solo-grafo`)

`2_codigo/grafo_marcadores.py` · salidas en `4_resultados/resultados_grafo_marcadores/`.
**Falta correr P1/P2/P3 completo** (va al clúster, `slurm/job_grafo_marcadores.sh`); lo
que sigue es la parte del grafo y la cohesión, que ya corrió local en segundos.

Base: 144 notas · 30 familias · **97 nodos** (el nodo es el par familia#plantilla, no la
plantilla sola, porque dos componentes de casi-duplicados contienen notas de dos familias
distintas: grupo 6 = BLACKBASTA + CONTI, grupo 55 = DHARMA + PHOBOS).

Marcadores hallados: **URL 159 · ONION 106 · EMAIL 74 · CLAVE 37 · ID 9 · BTC 5.**
**11 de 97 plantillas no tienen ningún marcador** — dato relevante para C.bis: para esas
plantillas una vista de marcadores no tiene nada que mirar.

### ★★★ P3 CORRIDO, Y EL CONTROL DE AZAR DA VUELTA LA LECTURA (2026-08-19)

`b3_protocolos.csv`. Combinado + LinearSVC, 2 pliegues, 10 semillas, mismo corpus de 144
notas. **Todos los números son macro-F1 sobre 30 familias (azar 0,033).**

| Protocolo | Grupos | macro-F1 | Exactitud | Exactitud balanceada |
|---|---|---|---|---|
| P1 «plantilla conocida» | 95 | 0,7501 ± 0,0267 | 0,8139 | 0,7738 |
| **P2 «variante nunca vista»** | **95** | **0,4210 ± 0,0508** | 0,5569 | 0,4754 |
| P3 sin exclusión, sin filtro | 43 | 0,1129 ± 0,0353 | 0,1465 | 0,1483 |
| P3 sin exclusión, filtro ≤2 fam. | 58 | 0,1741 ± 0,0665 | 0,2333 | 0,2080 |
| P3 con exclusión, sin filtro | 48 | 0,1583 ± 0,0382 | 0,2035 | 0,1944 |
| **P3 con exclusión, filtro ≤2 fam.** | **65** | **0,2293 ± 0,0535** | 0,2806 | 0,2731 |

### El control de azar, que es lo que hace reportable el número

La lectura ingenua sería: «P3 cae 0,1917 respecto de P2 ⇒ casi la mitad del 0,4210 venía de
reconocer IOCs repetidos y no de entender el texto». **Esa lectura es FALSA**, y hacía falta
un control para saberlo, porque bajo P3 hay una segunda causa mezclada: los grupos bajan de
95 a 65, con lo cual **7 familias quedan enteras en un solo pliegue** (contra 4,2 en P2 —su
F1 es 0 por construcción) y cada pliegue entrena con menos unidades independientes.

**El control:** agrupamientos al azar con el **mismo perfil de tamaños por familia** que las
componentes reales —misma cantidad de bloques y del mismo porte en cada familia— pero
eligiendo al azar qué plantillas caen juntas. Aísla «cuántas plantillas se fusionaron» de
«se fusionaron JUSTO las que comparten IOCs».

| | macro-F1 |
|---|---|
| P2 | 0,4210 |
| **Azar, mismo perfil de tamaños (n = 20)** | **0,2251 ± 0,0149** · IC 95 % de la media [0,2186; 0,2316] · rango 0,2039–0,2543 |
| P3 real | 0,2293 |

**Descomposición de la caída total de 0,1917** (n = 20 agrupamientos al azar):
- **agrupamiento más grueso: 0,1959**
- **continuidad de IOCs: −0,0042** — y el signo es **negativo**: agrupar por IOCs lastima
  **menos** que un agrupamiento al azar del mismo porte, que es lo contrario de lo que
  predice la objeción.

**El P3 real cae en el percentil 65 de la nube de azar** (13 de 20 valores por debajo);
prueba t de una muestra t = −1,26, **p = 0,223**: indistinguible. Con n = 5 la primera
corrida había dado azar 0,2290 ± 0,0209 y una diferencia de −0,0003; con n = 20 la
conclusión no cambia y el intervalo se aprieta a la mitad.

> **CONCLUSIÓN, y es un resultado FUERTE a favor de la tesis: el macro-F1 de 0,4210 en P2 NO
> viene de reconocer IOCs repetidos.** Agrupar por componente conexa no lastima porque le
> quite al modelo una muleta de IOCs; lastima **solo** porque el agrupamiento es más grueso.
> El azar con el mismo perfil de tamaños da 0,2290 y el P3 real 0,2293: indistinguibles.

**Por qué importa para la defensa.** Es exactamente la objeción que el tutor —que conoce a
Lemmou— puede plantear: «tu 0,435 es búsqueda de IOCs disfrazada de clasificación». La
respuesta ahora es un número y un control, no un argumento. Y refuerza la diferencia con
Lemmou et al. (2021), cuyo método de identificación de familia **sí** es búsqueda de
casi-duplicados por reglas y marcadores.

**Consecuencia metodológica: P3 no reemplaza a P2 ni lo mejora.** No es un protocolo más
informativo, es el mismo protocolo con un agrupamiento más grueso y por lo tanto más ruidoso.
**El protocolo que se reporta en la tesis sigue siendo P2**; P3 entra como el control que
descarta la contaminación por IOCs. Escribirlo así y no como «tenemos un tercer protocolo».

**Límites de este control:**
1. ✅ **Resuelto: n = 20 agrupamientos al azar** (la primera corrida fue con n = 5 y dio lo
   mismo). Desvío ± 0,0149, error estándar de la media 0,0033. La conclusión aguanta y la
   cifra es citable.
2. **Simplificación declarada:** las pocas componentes que cruzan familias se reparten dentro
   de cada familia por separado en el control. Son 7 aristas entre familias de 105, así que
   el efecto es menor, pero está declarado.
3. El control usa la variante **con exclusión de nombre de familia y filtro ≤2 familias**, que
   es la más estricta y la más defendible. Las otras tres variantes no tienen control de azar
   corrido: no citarlas sin él.

### ▶ EL HALLAZGO QUE REORDENA EL PLAN DE RECOLECCIÓN

Correlación de Spearman contra el **F1 por familia** (de B.1: 30 familias · P2ret ·
k=todo · 144 notas · 100 repeticiones), n = 29 familias (WASTEDLOCKER queda afuera: con
una sola plantilla no tiene cohesión definible):

| Predictor | Spearman ρ | p |
|---|---|---|
| **Cohesión interna** (coseno medio entre plantillas de la familia) | **+0,704** | 2,0·10⁻⁵ |
| **Margen** (cohesión interna − parecido máximo a plantilla ajena) | **+0,674** | 6,1·10⁻⁵ |
| Fracción de pares de la familia unidos por un marcador compartido | +0,425 | 0,022 |
| Parecido máximo a una plantilla de otra familia | −0,140 | 0,47 (no signif.) |
| **Cantidad de plantillas de la familia** | **−0,108** | **0,58 (no signif.)** |

> **Cuánto se parecen entre sí las plantillas de una familia explica su desempeño
> (ρ +0,704). Cuántas plantillas tiene NO explica nada (ρ −0,108, p = 0,58).**

**Esto NO contradice a B.1, y la distinción hay que escribirla con cuidado:**
- B.1 es **causal**: se quitaron plantillas y se midió la caída. Agregar plantillas a una
  familia **sí** mejora (paso 3→4: +0,0297 de macro-F1, IC 95 % [+0,0039; +0,0554]).
- B.3 es **correlacional entre familias**: lo que explica que una familia ande bien o mal
  **no** es su cantidad de plantillas, es cuánto se parecen entre sí.
- Las dos cosas conviven: sumar un texto ayuda un poco a cualquier familia, pero **no
  convierte a una familia de cohesión baja en una de cohesión alta**, porque el texto
  nuevo probablemente tampoco se parezca a los que ya están.

### Cohesión por familia — la tabla que ordena la recolección

`b3_cohesion_por_familia.csv`. Coseno char 3-5 entre centroides de plantilla; por
construcción todos **por debajo de 0,90**, que es el umbral de casi-duplicado.

**Las 9 familias de margen positivo tienen F1 por familia medio 0,919** (mínimo 0,670):
JIGSAW, NOTPETYA, AVOSLOCKER, CUBA, TESLACRYPT, BADRABBIT, BLACKMATTER, DARKSIDE,
NETWALKER. **Las 20 de margen negativo, 0,511** (mínimo 0,000).

Cohesión interna de las familias de la lista de recolección de B.1, ordenadas de peor a
mejor: **CHIMERA 0,1538** (la más baja de las 30) · CRYPTOLOCKER 0,3564 · MAZE 0,4114 ·
RYUK 0,4388 · WANNACRY 0,4449 · MEDUZALOCKER 0,4317 · JIGSAW 0,5075 · NOTPETYA 0,8169.
Para comparar, las que aciertan con solo 2 plantillas: DARKSIDE 0,8723 · BLACKMATTER
0,8715 · NETWALKER 0,8053 · CUBA 0,7641.

**CHIMERA es el caso límite y conviene citarlo:** sus dos plantillas tienen coseno 0,1538
entre sí y **un solo marcador en total**. Es la familia con menos cohesión del corpus y su
F1 por familia es 0,000. Recolectarle un tercer texto es lo que menos probabilidad tiene
de servir de toda la lista.

**Contraejemplo que hay que declarar, para no vender el predictor como una regla:**
**SUNCRYPT tiene cohesión interna 0,4529 y margen −0,1275, pero F1 por familia 1,000**, y
cero marcadores. Se acierta por vocabulario propio, no por parecerse a sí misma. La
correlación es una tendencia fuerte, **no** una ley: dentro del grupo de margen negativo
el F1 va de 0,000 a 1,000.

### El grafo: cuatro variantes, y el filtro de valores genéricos no es opcional

`b3_variantes.csv`. Nodos 97 en todas.

| Variante | Aristas | dentro de familia | entre familias | Componentes | Mayor | Familias inevaluables bajo P3 |
|---|---|---|---|---|---|---|
| sin exclusión, sin filtro | 279 | 146 | 133 | 43 | **39** | **13** |
| sin exclusión, filtro ≤2 familias | 126 | 119 | 7 | 58 | 8 | 10 |
| con exclusión, sin filtro | 258 | 125 | 133 | 48 | 24 | 9 |
| **con exclusión, filtro ≤2 familias** | **105** | **98** | **7** | **65** | **8** | **7** |

- **Sin el filtro de valores genéricos el grafo colapsa:** una componente de **39 nodos de
  97** y 13 familias que quedan inevaluables bajo P3. Es exactamente el riesgo que se
  anticipó: valores de infraestructura común (tipo torproject.org) unen todo. `b3_valores_
  compartidos.csv` está ordenado por número de familias para auditar quién pesa.
- **El criterio de circularidad elimina 21 aristas dentro de familia** (146 → 125 sin
  filtro; 119 → 98 con filtro), o sea que **cerca del 15-18 % de la continuidad de IOCs
  dentro de una familia viene de valores que llevan el nombre de la familia adentro.** Ese
  es el número que habilita la afirmación cuantitativa sobre el 181/182 de Lemmou.
  ⚠️ Antes de citarlo hay que **auditar a mano `b3_valores_excluidos.csv`**: el criterio es
  subcadena en cualquier posición y «conti» está dentro de «continue», «cerber» dentro de
  «cerberus». Puede haber exclusiones espurias.
- **Las aristas entre familias caen de 133 a 7 con el filtro.** Las 133 eran casi todas
  infraestructura compartida. ⚠️ **Las 7 que sobreviven NO son parentescos**: son
  `https://torproject.org` con una grafía peculiar y un enlace viejo de descarga de Tor. Ver
  la «AUDITORÍA DE B.3 CERRADA», que lo verificó valor por valor.

### Lo que falta de B.3 — ✅ **NADA: los tres puntos están cerrados**

1. ✅ **P1/P2/P3 completo, corrido local** el 2026-08-19, más el control de azar con n = 20.
   Ver la sección de P3 más abajo. **Y la premisa de este punto era errónea:** la diferencia
   P2 − P3 **no** dice por sí sola cuánto venía de la continuidad de IOCs; hace falta el
   control de azar, y con él el aporte de los IOCs es −0,0042 de macro-F1, o sea nulo.
2. ✅ **`b3_valores_excluidos.csv` auditado** (2026-08-19, commit c654e72): **cero exclusiones
   espurias**, el caso «conti dentro de continue» no existe. Los 21 de 146 se confirman, con
   una precisión que hay que decir al citar el 15-18 %: **se concentran en LOCKBIT**, no están
   repartidos entre familias.
3. ✅ **Las 7 aristas entre familias revisadas, y refutaron la expectativa que había escrito
   acá.** No reencuentran los parentescos. **NUNCA escribir «el grafo reencuentra los
   parentescos».** El resultado citable es el inverso: tras filtrar la infraestructura común,
   **ningún IOC operativo se comparte entre familias distintas** — los marcadores reales son
   privados de cada familia.

## ★★★ B.1 CERRADO — CURVA DE APRENDIZAJE DE NOTAS (2026-08-19, local, 100 repeticiones)

`2_codigo/curva_aprendizaje_notas.py` · agregados con `resumen_para_capitulo4.py --solo b1`
· salidas en `4_resultados/resultados_curva_notas/` y `4_resultados/resumen_capitulo4/`
(`b1_curva.csv`, `b1_deltas_pareados.csv`, `b1_extrapolacion.csv`,
`b1_costo_recoleccion.csv`, `fig_b1_curva_notas.png`).
Base: **144 notas · 95 plantillas · 30 familias**. Control de corrección: diferencia
**0,00e+00** contra el evaluador canónico en los dos protocolos.

### ▶ LA RESPUESTA AL PEDIDO DEL TUTOR, EN UN NÚMERO

> **Hacen falta 4 textos distintos (plantillas) por familia. Eso cuesta 33 plantillas
> nuevas repartidas en 19 familias — es decir, al menos 33 notas nuevas, cada una de
> contenido distinto. De 4 a 5 la ganancia es +0,0006 de macro-F1, con IC 95 %
> [−0,0207; +0,0219]: indistinguible de cero.**

El corte en 4 no es una impresión, es el último paso que mueve la aguja de forma
medible. Sobre las **5 familias que tienen ≥ 5 plantillas** (conjunto de clases
constante, azar macro-F1 0,200), diferencias **pareadas** de macro-F1 con IC 95 %:

| Paso | Δ macro-F1 | IC 95 % | ¿Aporta? |
|---|---|---|---|
| 1 → 2 plantillas | **+0,1626** | [+0,1240; +0,2012] | sí |
| 2 → 3 | **+0,0454** | [+0,0123; +0,0786] | sí |
| 3 → 4 | **+0,0297** | [+0,0039; +0,0554] | sí |
| **4 → 5** | **+0,0006** | **[−0,0207; +0,0219]** | **NO** |

Y sobre las **11 familias con ≥ 4 plantillas** (azar macro-F1 0,091) la curva **sigue
subiendo** hasta el final: 1→2 **+0,1254** [+0,1009; +0,1499] · 2→3 **+0,0426**
[+0,0242; +0,0611] · 3→todo (3,87 de media) **+0,0385** [+0,0206; +0,0565], las tres
significativas. Las dos curvas concuerdan: **el techo está en 4, no antes.**

### ⚠️ CORRECCIÓN A LO QUE SE SUPUSO ANTES DE MEDIR

Antes de correr se anticipó que la curva saldría plana y que la conclusión sería
«documentar el límite y pasar a few-shot». **Es al revés, y el error tiene un mecanismo
identificado.** La curva de las 30 familias efectivamente se aplana (de k=3 en adelante
quedan +0,012 de macro-F1 en cuatro pasos), pero esa chatura es **agotamiento del corpus,
no saturación del aprendizaje**: la columna `n_fam_bajo_tope` de `b1_curva.csv` cae de
16,8 familias en k=1 a 5,0 en k=3 y a 0,0 en k=7. A partir de k=3 casi ninguna familia
puede aportar una plantilla más, así que el tope deja de morder y la curva se aplana
**por falta de material, no porque el modelo dejara de aprender**. En los subconjuntos
donde sí hay material (11 y 5 familias) la curva sigue subiendo.

**Consecuencia para el Sprint C: la rama que se activa es RECOLECTAR, con un objetivo
acotado y finito (33 notas de contenido distinto en 19 familias), no «recolectar
indefinidamente». Few-shot deja de ser la única salida y pasa a ser el complemento para
las familias donde la recolección no alcance.**

### El hallazgo metodológico: la moneda son los textos distintos, no las notas

Se corrió cada curva con **dos formas de recolectar**: tope por *plantillas* (las notas
que se suman son textos distintos) y tope por *notas* (notas al azar, que muchas veces
repiten un texto ya presente). Mirando el **mismo dato sobre dos ejes** (paneles (a) y (b)
de la figura), sobre 30 familias y P2ret:

- **Sobre el eje de plantillas las dos curvas se superponen**: diferencia absoluta media
  **0,0052** de macro-F1, máxima **0,0109** — un orden de magnitud por debajo del desvío
  típico de la métrica (± 0,0486).
- **Sobre el eje de notas se separan**: hasta **0,0351** de macro-F1 en el punto más bajo.
- El mismo macro-F1 se alcanza por dos caminos con muy distinto gasto de notas:
  **1,53 plantillas/familia con 2,11 notas/familia → macro-F1 0,5973**, contra
  **1,46 plantillas/familia con 1,61 notas/familia → macro-F1 0,6005**. Mismo resultado,
  **24 % menos notas**, porque el primer camino arrastra copias dentro de cada plantilla.

**Regla práctica para la recolección, y es citable:** al presupuestar, contar **textos
distintos**. Una nota que repite un contenido ya presente en el corpus no mueve el
macro-F1 de forma medible. Esto conecta con el Sprint 1.1 y con el hallazgo de que 146
notas son solo 95 plantillas: la profundidad del corpus está en la diversidad.

### Las curvas, completas

**30 familias · P2ret (una plantilla retenida por familia, 100 repeticiones) · tope por
plantillas** — azar macro-F1 0,033:

| Plantillas/fam (tope k) | macro-F1 | Exactitud | Notas train/fam | Familias que aún pueden dar más |
|---|---|---|---|---|
| 1 | 0,5405 ± 0,0619 | 0,5721 ± 0,0804 | 1,31 | 16,8 |
| 2 | 0,5973 ± 0,0571 | 0,6132 ± 0,0754 | 2,11 | 10,6 |
| 3 | 0,6043 ± 0,0504 | 0,6166 ± 0,0769 | 2,65 | 5,0 |
| 4 | 0,6121 ± 0,0474 | 0,6240 ± 0,0750 | 2,95 | 2,9 |
| 5 | 0,6141 ± 0,0458 | 0,6269 ± 0,0754 | 3,15 | 1,0 |
| 6 | 0,6140 ± 0,0439 | 0,6279 ± 0,0736 | 3,21 | 1,0 |
| 7 = todo | 0,6164 ± 0,0450 | 0,6285 ± 0,0734 | 3,28 | 0,0 |

**11 familias con ≥ 4 plantillas** (BLACKBASTA, BLACKCAT, CERBER, CLOP, DHARMA, GANDCRAB,
HELLOKITTY, LOCKBIT, PHOBOS, RANSOMEXX, TESLACRYPT) · P2ret · azar macro-F1 0,091:
k=1 **0,5438 ± 0,1368** · k=2 **0,6692 ± 0,1404** · k=3 **0,7118 ± 0,1303** ·
todo (3,87) **0,7504 ± 0,1111**.

**5 familias con ≥ 5 plantillas** (CERBER, DHARMA, GANDCRAB, HELLOKITTY, LOCKBIT) · P2ret
· azar macro-F1 0,200: k=1 **0,5713 ± 0,1978** · k=2 **0,7339 ± 0,1990** ·
k=3 **0,7793 ± 0,1881** · k=4 **0,8089 ± 0,1789** · todo (5) **0,8095 ± 0,1727**.

> **Los tres conjuntos NO son comparables entre sí** (30, 11 y 5 clases; azar macro-F1
> 0,033 / 0,091 / 0,200) y **P2ret no es el P2 canónico**: la retención deja hasta n−1
> plantillas del lado de entrenamiento y evalúa contra una plantilla por familia, así
> que su macro-F1 de 0,6164 sobre 30 familias **no reemplaza** al 0,4210 ± 0,0508 del P2
> de 2 pliegues sobre las mismas 144 notas, ni al **0,4353 ± 0,0565 oficial** sobre 146.

**Bajo el P2 canónico de 2 pliegues** la curva se aplana igual y antes: k=1 0,3933 ±
0,0533 · k=2 0,4167 ± 0,0572 · k=3 **0,4301 ± 0,0570** · k=4 0,4232 ± 0,0568 · todo
0,4210 ± 0,0508. **El máximo de la curva está en k=3 y es 0,4301**, apenas 0,009 por
encima del corpus completo. Bajo **P1**: k=1 0,6412 ± 0,0238 · k=2 0,7335 ± 0,0279 ·
k=3 0,7447 ± 0,0324 · k=4 0,7543 ± 0,0379 · todo 0,7501 ± 0,0267; de k=3 en adelante
nada es significativo.

### Costo de recolección (aritmética sobre el corpus, sin modelo)

De `b1_costo_recoleccion.csv`. Cada plantilla nueva exige **al menos** una nota nueva, y
tiene que ser de contenido distinto:

| Objetivo (plantillas/familia) | Plantillas nuevas | Familias a completar |
|---|---|---|
| 3 | **14** | 13 |
| **4 (el objetivo medido)** | **33** | **19** |
| 5 | 58 | 25 |
| 6 | 85 | 27 |
| 8 | 143 | 29 |

### ★ LISTA DE RECOLECCIÓN, Y EL REFINAMIENTO QUE LA CORTA A LA MITAD (2026-08-19)

`4_resultados/resumen_capitulo4/b1_familias_a_recolectar.csv`, regenerable con
`resumen_para_capitulo4.py --solo b1`. **La cifra de la columna F1 es F1 POR FAMILIA**
(30 familias · P2ret · k=todo · 144 notas · 100 repeticiones), **no** el macro-F1, que es
el promedio de las 30 y vale 0,6164 ± 0,0450 en ese mismo protocolo.

**El objetivo de 33 textos nuevos en 19 familias es correcto pero grueso: al cruzarlo con
el F1 por familia, la mitad de ese esfuerzo iría a familias que ya aciertan perfecto.**

**PRIORIDAD REAL — 9 familias, 17 textos nuevos.** Son las que necesitan textos *y* hoy
andan mal (F1 por familia < 0,70):

| Familia | Plantillas hoy | Faltan para 4 | F1 por familia hoy |
|---|---|---|---|
| WASTEDLOCKER | 1 | **3** | 0,000 ± 0,000 |
| CHIMERA | 2 | 2 | 0,000 ± 0,000 |
| MAZE | 2 | 2 | 0,000 ± 0,000 |
| MEDUZALOCKER | 3 | 1 | 0,000 ± 0,000 |
| WANNACRY | 2 | 2 | 0,010 ± 0,100 |
| RYUK | 2 | 2 | 0,032 ± 0,160 |
| CRYPTOLOCKER | 3 | 1 | 0,410 ± 0,456 |
| JIGSAW | 2 | 2 | 0,670 ± 0,451 |
| NOTPETYA | 2 | 2 | 0,695 ± 0,114 |
| **TOTAL** | | **17** | |

**LO QUE NO HAY QUE PRIORIZAR — 10 familias, 16 textos.** Necesitan textos para llegar a 4,
pero **ya aciertan casi o totalmente** con 2 o 3 plantillas: DARKSIDE, NETWALKER y SUNCRYPT
dan **F1 por familia 1,000 ± 0,000**; BLACKMATTER 0,992 ± 0,060 · BADRABBIT 0,983 ± 0,073 ·
CUBA 0,975 ± 0,086 · AVOSLOCKER 0,973 ± 0,091 · SODINOKIBI 0,863 ± 0,179 · CONTI 0,846 ±
0,157 · LORENZ 0,807 ± 0,367. Sumarles plantillas no puede mejorar lo que ya está en 1,000.

**Y las 11 que ya tienen 4 o más no entran en la lista:** BLACKBASTA, BLACKCAT, CERBER, CLOP,
DHARMA, GANDCRAB, HELLOKITTY, LOCKBIT, PHOBOS, RANSOMEXX, TESLACRYPT.

**Consecuencia para el Sprint C: el lote objetivo baja de 33 a 17 textos distintos en 9
familias.** Es el plan de recolección concreto, y explica de paso por qué el macro-F1 global
está en 0,4210 ± 0,0508 bajo el P2 canónico: **cuatro familias tienen F1 por familia
exactamente 0,000** (WASTEDLOCKER, CHIMERA, MAZE, MEDUZALOCKER) y dos más están por debajo
de 0,05 (WANNACRY, RYUK). Seis familias en cero o casi arrastran el promedio de treinta.

> ⚠️ **Contraste que hay que escribir en el capítulo, porque es contraintuitivo:** tener
> pocas plantillas **no** condena a una familia. DARKSIDE, NETWALKER y SUNCRYPT tienen 2
> plantillas cada una y dan F1 por familia 1,000 ± 0,000; CHIMERA y MAZE tienen también 2 y
> dan 0,000 ± 0,000. **La cantidad de plantillas no explica sola el desempeño por familia:
> lo que importa es cuánto se parecen entre sí las plantillas de la familia.** Eso es
> justamente lo que va a medir B.3, y es el elemento de acción 1 del tutor.

### Lo que NO se puede afirmar con esto — límites declarados

1. **El techo extrapolado no es reportable.** El ajuste de ley de potencia inversa sobre
   las 11 familias da techo macro-F1 0,803 pero con **IC 95 % [0,724; 1,698]**, y sobre
   las 5 familias 0,873 con **IC 95 % [0,794; 1,214]**: los límites superiores pasan de
   1, que es imposible para un F1. Con 3 y 4 puntos el ajuste tiene tantos parámetros
   como datos. **Se reportan las diferencias pareadas observadas, no el techo ajustado.**
   Los objetivos altos lo confirman: para macro-F1 0,80 el ajuste de las 11 familias pide
   97,6 plantillas/familia con IC [5,0; 103,8] y solo alcanzable en el 54 % del bootstrap
   — o sea, sin información.
2. **Las líneas «NO ALCANZABLE» de las 30 familias en `b1_extrapolacion.csv` NO dicen que
   recolectar no sirva.** Dicen que *ningún recorte del corpus actual* pasa de macro-F1
   0,430 en P2 (IC 95 % [0,399; 0,469]) ni de 0,622 en P2ret. Es agotamiento del material,
   no una cota sobre lo que daría un corpus más diverso. **No citarlas como argumento
   contra la recolección.**
3. **Supuesto que no se puede verificar:** que las plantillas nuevas se comporten como las
   existentes. Puede que una familia tenga pocas plantillas *porque* varía poco, y en ese
   caso sumarle textos rendiría menos que lo que predice la curva de las 11.
4. **WASTEDLOCKER tiene una sola plantilla**, así que en P2ret queda siempre sin
   entrenamiento (`n_fam_sin_train = 1,0` en toda la curva de 30 familias) y su F1 es 0
   por construcción. Bajo el P2 canónico son 4,2 familias las que quedan así.
5. **Base 144 notas**, no 146. Las dos notas en cuarentena cuestan 0,0096 de macro-F1 en
   P1 y 0,0143 en P2 (medido, ver la sección siguiente), las dos por debajo del desvío
   entre semillas.

## ★ B.1/B.3 — VERIFICACIONES PREVIAS AL DISEÑO DE LA CURVA (2026-08-18)

Todo lo de esta sección se verificó **abriendo los archivos y recorriendo el corpus**, no de
memoria. Es la base sobre la que se diseña la curva de aprendizaje de notas (B.1).

### 1. La estructura de plantillas NO cambió con el incidente de Defender
El corpus en disco hoy tiene **144 notas** (DHARMA 17, faltan `Info__13.hta` e `Info__3.hta`).
Recorriendo `3_datos/corpus_v2` con `agrupar_neardups()` (coseno char 3-5 > 0,90):

| | Corrida canónica (146 notas) | Corpus de hoy (144 notas) |
|---|---|---|
| Grupos de contenido (plantillas) | **95** | **95** |
| Plantillas por familia | 1×1, 2×12, 3×6, 4×6, 5×2, 6×2, 8×1 | **idéntico** |
| Grupos que cruzan familias | 2 | **los mismos 2** |

**Las dos notas en cuarentena caen las dos dentro del grupo 55** (la plantilla grande de
DHARMA, 11 notas), así que su ausencia no elimina ninguna plantilla. **Consecuencia práctica:
el eje x de B.1 —plantillas por familia— es el mismo sobre 144 y sobre 146 notas**, y B.1 se
puede correr hoy sin esperar a que Romina restaure los `.hta`. Lo que sí cambia con las dos
notas es el eje de P1 (que cuenta notas), y ahí hay que declarar la base.

### 2. Dos plantillas son compartidas ENTRE familias (hallazgo nuevo, no registrado antes)
La suma de plantillas por familia da **97**, pero hay **95** componentes: dos grupos contienen
notas de dos familias distintas.

| Grupo | Familias | Notas |
|---|---|---|
| 6 | **BLACKBASTA + CONTI** | `BLACKBASTA/blackbasta2.txt`, `CONTI/conti4.txt` |
| 55 | **DHARMA + PHOBOS** | 11 notas de DHARMA + `PHOBOS/pcrisk_phobos_1.txt` |

Los dos pares coinciden con parentescos documentados en la bibliografía (Black Basta y Conti;
Phobos como derivado de Dharma/CrySiS), así que el corpus **reproduce por contenido** una
relación conocida entre familias — es material para el capítulo, no un defecto del corpus.

**Tres consecuencias operativas:**
- Bajo P2, `StratifiedGroupKFold` manda las dos familias del grupo al **mismo pliegue**: nunca
  quedan una en train y la otra en test. No es un error, pero hay que declararlo.
- Hay parentesco entre familias medible a nivel de **contenido**. ⚠️ **Predicción que escribí
  acá y que B.3 REFUTÓ:** decía «el grafo de B.3 debería reencontrarlo por IOCs». **No lo
  reencuentra.** DHARMA↔PHOBOS no aparece en ninguna variante del grafo, y BLACKBASTA↔CONTI
  aparece solo por una grafía peculiar de una URL de torproject. Los parentescos los encontró
  la **deduplicación por contenido** (grupos 6 y 55), no los marcadores. Ver la «AUDITORÍA DE
  B.3 CERRADA».
- Encaja con lo que ya dice `corrida_canonica_por_familia.csv`. **Ojo con la métrica: son F1
  POR FAMILIA, no macro-F1** (el macro-F1 es el promedio de las 30 y vale 0,435 ± 0,057).
  Base: protocolo P2, vista combinada + LinearSVC, 146 notas, 2 pliegues, promedio de 10
  semillas, según `manifiesto_corrida.json`. **F1 por familia: BLACKBASTA 0,215 · DHARMA 0,483**
  (entre las más bajas de las 30) frente a **PHOBOS 0,637 · CONTI 0,786**. La confusión que se
  sospechaba tiene ahora un mecanismo verificado.

### 2.bis Cuánto cuestan las dos notas en cuarentena — MEDIDO (2026-08-18)

Se resolvió la duda «¿hay que rehacer la corrida canónica sobre 144?». **No hace falta.**
Corriendo `evaluar()` de `clasificador_notas_v2.py` sobre el corpus de hoy, con la misma
configuración ganadora de cada protocolo (P1 = caracteres + LinearSVC · P2 = combinado +
LinearSVC, 2 pliegues, 10 semillas):

| Protocolo | 146 notas (cifra oficial de la tesis) | 144 notas (corpus de hoy) | Costo de las 2 notas |
|---|---|---|---|
| P1 «plantilla conocida», macro-F1 | 0,7597 ± 0,0289 | **0,7501 ± 0,0267** | **0,0096** |
| P2 «variante nunca vista», macro-F1 | 0,4353 ± 0,0565 | **0,4210 ± 0,0508** | **0,0143** |

**Las dos pérdidas son bastante menores que el desvío entre semillas** (0,0096 contra ± 0,0267
en P1; 0,0143 contra ± 0,0508 en P2). Conclusión operativa: **el capítulo 4 no se toca y la
cifra oficial sigue siendo la de 146 notas.** B.1 corre sobre 144 y lo declara. Restaurar los
dos `.hta` sigue siendo deseable por integridad del corpus, pero **ya no bloquea ni cambia
ninguna conclusión**.

### 2.ter El script de B.1 y su control de corrección

`2_codigo/curva_aprendizaje_notas.py` (nuevo, local, sin cluster). Corre con `--validar`,
`--rapido` o completo. Salidas en `4_resultados/resultados_curva_notas/`: una fila **por
repetición y por punto**; los promedios los calcula `resumen_para_capitulo4.py --solo b1`
(función `resumen_b1()`), nunca a mano.

**Control de corrección, y pasó exacto:** el punto k = «todo el corpus» de la curva tiene que
dar lo mismo que `evaluar()` del script canónico sobre el mismo corpus. **Diferencia 0,00e+00
en los dos protocolos.** El script aborta si no coincide, así que ninguna cifra de la curva
puede salir de un camino de código distinto al canónico.

**Tres decisiones de diseño que hay que declarar en la tesis:**
1. **Dos unidades en el eje x, y la diferencia entre ellas es el resultado.** Tope por
   *plantillas* = las notas que se agregan son textos distintos (**diversidad**, caro); tope
   por *notas* = notas al azar, que muchas veces repiten un texto ya presente (**volumen**,
   barato). Las dos curvas se miran sobre el mismo eje de notas de entrenamiento por familia:
   la separación entre ambas es cuánto vale la diversidad, medido.
2. **Tres conjuntos de familias, NO comparables entre sí** — 30 familias (azar macro-F1 0,033),
   las 11 con ≥ 4 plantillas (azar 0,091) y las 5 con ≥ 5 plantillas (azar 0,200). Con las 30
   la curva satura por agotamiento; con las 11 el conjunto de clases es constante de punta a
   punta y es la única extrapolable. **Toda tabla lleva el azar de su conjunto pegado.**
3. **Protocolo P2ret (retención de una plantilla por familia, repetido).** Bajo el P2 canónico
   de 2 pliegues una familia de 2 plantillas aporta 1 sola al entrenamiento, así que el tope k
   casi no muerde y la curva no tendría alcance. La retención deja hasta n−1 plantillas del
   lado de entrenamiento y es exactamente la pregunta de despliegue. Se reporta aparte de P2.

**Anidamiento y estadística:** el orden de plantillas/notas de cada familia se sortea una vez
por repetición y k toma el prefijo, así que el entrenamiento de k está contenido en el de k+1.
Los puntos quedan **pareados** y el aporte de cada paso se calcula como diferencia por
repetición con IC 95 %, no restando dos medias independientes: con ± 0,05 de desvío en el
macro-F1 de P2, restar medias sueltas no distingue nada.

### 3. Cuántos puntos admite realmente el eje x
Familias que pueden aportar al menos k plantillas: **k≥1: 30 · k≥2: 29 · k≥3: 17 · k≥4: 11 ·
k≥5: 5 · k≥6: 3 · k≥7: 1 · k≥8: 1** (CERBER es la única con 8). Esto acota el diseño de B.1:
una curva con **el conjunto de clases fijo** solo llega a 3 puntos con 11 familias, o a 4
puntos con 5 familias. **Hay que declarar el alcance del eje x como parte del resultado.**

## ✅ A.2 CERRADO — DESVÍO DEL EXP. 2c SOBRE 10 SEMILLAS (job 3648, 2026-08-17)

Diez semillas (0-9), hiperparámetros fijos, 500 archivos/familia, 30 familias, 5 pliegues.
~560 s por semilla, 93 min en total. **Cifras para el capítulo:**

> **Exp. 2c — exactitud 0,912 ± 0,002 · macro-F1 0,911 ± 0,001** (media ± desvío, 10 semillas)

| | Media | Desvío | Mín | Máx | Rango |
|---|---|---|---|---|---|
| Exactitud | 0,9120 | 0,0016 | 0,9093 (s7) | 0,9147 (s2) | 0,0054 |
| macro-F1 | 0,9111 | 0,0014 | 0,9084 (s7) | 0,9131 (s2) | 0,0047 |

**Con esto los dos frentes tienen error reportado y el pedido del tutor queda cubierto.**

### El 0,9089 del job 3639 queda explicado
El 3639 usó `semilla_final=7`. En el multisemilla, la **semilla 7 da 0,9093 / 0,9084** — y es
la más baja de las diez. La diferencia con el 3639 es **+0,0004 de exactitud**, o sea unos
**6 archivos de 15.000**, del mismo orden que los 12 JPEG en claro que el 3639 excluía.

Queda cerrada la sospecha que se había anotado al ver las cuatro primeras semillas por encima
de 0,9089: **era ruido de semilla, no un problema de configuración.** El 0,9089 no es un
número raro, es simplemente el peor de diez. **La cifra honesta a reportar es la media,
0,912 ± 0,002**, no el valor puntual.

### La comparación de desvíos entre frentes es, en sí misma, un resultado

| Medición | Desvío | Base |
|---|---|---|
| Archivos, Exp. 2c (macro-F1) | **± 0,001** | 15.000 archivos |
| Archivos, Exp. 2b (combinado) | ± 0,001 | 1.500 archivos |
| Notas, P1 (macro-F1) | ± 0,029 | 146 notas |
| Notas, P2 (macro-F1) | **± 0,057** | 95 plantillas |

El frente de notas tiene **entre 20 y 40 veces más dispersión** que el de archivos. No es que
un método sea peor que el otro: es que uno se mide sobre 15.000 muestras y el otro sobre 95
plantillas. **Las barras de error son evidencia directa de la tesis de que el techo lo pone el
dato y no el método** — y refuerzan los tres resultados negativos convergentes. Escribirlo en
el cap. 4 junto a la discusión del límite del corpus.

## ✅ RESUELTO — LA FIRMA DE BADRABBIT ES LA PALABRA «encrypted» (2026-08-17)

Volcado de los últimos 18 bytes de **todos** los archivos de `BADRABBIT-small`:

```
    965 65006e006300720079007000740065006400
      1 fe8564b129add8c65100e7e5ef64e5544e80
      1 fe21dbb554d2d817a1753c0d512f8bfa07ee
      ... (el resto, todos distintos entre sí)
```

`65 00 6e 00 63 00 72 00 79 00 70 00 74 00 65 00 64 00` es **`encrypted` en UTF-16
little-endian** — el marcador documentado de BadRabbit. **La firma es real y tiene nombre.**

**La aritmética del parpadeo queda cerrada.** 965 de ~1.000 archivos (96,5 %) llevan el
marcador; el resto no. La probabilidad de que los 50 archivos sorteados lo lleven todos es
0,965⁵⁰ ≈ **0,17**. O sea que **la semilla 42 fue la excepción afortunada**: en ~83 % de las
semillas el detector NO encuentra la firma de BADRABBIT. Lo raro era el acierto, no el fallo.

**Corrección a mi hipótesis previa:** no son archivos «sin cifrar» como los 12 JPEG de CERBER.
Los últimos 18 bytes de los ~35 restantes son todos distintos entre sí y de aspecto aleatorio,
o sea que están cifrados; simplemente no llevan el marcador al final, o lo llevan a otra
distancia. **No hay indicio de contaminación del corpus en BADRABBIT**, así que esto NO afecta
al Exp. 2c.

**Arreglo que se desprende:** aplicar al prefijo/sufijo el **mismo criterio de mayoría** que ya
se aplicó a la extensión, en vez de unanimidad byte a byte. Con umbral 0,90, BADRABBIT
mostraría su marcador de 18 bytes en todas las semillas y se reportaría con su cobertura real
(96,5 %). Un mismo cambio corrige la inestabilidad y mejora la cifra.

**Y es un resultado mejor, no peor:** «BADRABBIT marca sus archivos con la cadena `encrypted`
en UTF-16, en el 96,5 % de los casos» es más fuerte que «BADRABBIT no deja marca».

## 🚨 TERCER DEFECTO, MÁS GRAVE — LA DETECCIÓN DE FIRMAS BINARIAS ES INESTABLE (2026-08-17)

Comparando la corrida local (semilla 42) contra la del cluster (semilla 1), **la misma familia
da resultados distintos**:

| Familia | Semilla 42 | Semilla 1 |
|---|---|---|
| **BADRABBIT** | sufijo común de **18 bytes**, `marca_detectable=True` | prefijo 0, sufijo 0, **`False`** |

Y en consecuencia cambia la lista que va al capítulo 4: con semilla 42 las familias sin marca
son **NOTPETYA y SUNCRYPT (2)**; con semilla 1 son **BADRABBIT, NOTPETYA y SUNCRYPT (3)**.

**Esto es más serio que la discusión del umbral de extensión.** El umbral afecta a la extensión,
que es metadato; esto afecta a las **firmas binarias**, que son el hallazgo propio del Exp. 2b y
el argumento de que la marca vive en el contenido. Las cifras de cobertura y exactitud del
modo «solo firmas binarias» dependen de la semilla, y hoy se reportan como si fueran fijas.

> ✅ **RESUELTO el 2026-08-17.** El volcado descartó la hipótesis de contaminación (los ~35 sin
> marcador están cifrados) y el criterio de mayoría ya está aplicado al prefijo/sufijo y probado.
> Ver la sección «EL ARREGLO DEL CASO BADRABBIT, YA IMPLEMENTADO». La hipótesis de abajo queda
> como registro de lo que se pensó, **no** como pendiente.

**Hipótesis a verificar (no afirmar sin comprobar):** que la carpeta de BADRABBIT contenga
archivos **sin cifrar**, igual que los 12 JPEG en claro hallados en CERBER. Un solo archivo sin
la marca dentro de la muestra de 50 destruye el sufijo común de toda la familia — que es el
mismo mecanismo que ya se vio con la extensión de JIGSAW, pero sobre las firmas. BadRabbit
tiene documentado que agrega un marcador al final de los archivos que cifra, así que un sufijo
de 18 bytes es plausible y probablemente real; lo que falla es la muestra, no la familia.

**Prueba decisiva:** contar cuántos archivos de BADRABBIT comparten los últimos 18 bytes. Si la
mayoría los comparte y unos pocos no, está confirmado.

**Consecuencia para el plan:** la corrida multisemilla del detector deja de ser opcional y deja
de ser sobre el umbral. Hay que reportar, sobre 10 semillas, la varianza de: familias con marca,
cobertura y exactitud de los tres modos. Y hay que auditar la integridad de las carpetas de
todas las familias, no solo de CERBER.

## ⚠️ DOS DEFECTOS HALLADOS AL AUDITAR LOS CSV (2026-08-17)

Verificado abriendo los archivos bajados, no leyendo logs.

### 1. El umbral del detector estructural NO estaba demostrado — ✅ **DEMOSTRADO 2026-08-17**
`clasificacion_loo_unanimidad.csv` y `clasificacion_loo_umbral_90.csv` son **idénticos byte a
byte**, igual que los `marcas_por_familia_*`. Con la semilla usada (42), tras excluir el `.pdf`,
la unanimidad ya pasaba sola — JIGSAW incluido. **El trabajo lo hizo la exclusión del PDF, no el
umbral.**

JIGSAW tiene 990 `.fun` sobre 997 archivos sin contar PDF ⇒ la probabilidad de que 50 al azar
sean todos `.fun` es **0,6968** (hipergeométrica exacta, `comb(990,50)/comb(997,50)`) ⇒ la
unanimidad falla con probabilidad **0,3032**. Escribir «unanimidad y umbral 0,90 dan lo mismo»
es cierto para la semilla 42 y falso en general.

#### ★ RESULTADO — 10 semillas, el criterio de unanimidad PARPADEA (registrado de lo pegado)

Corrida en el cluster (`srun --nodelist=c2`, 10 corridas de `deteccion_estructural.py --semilla
1..10`, extensión y `marca_detectable` de JIGSAW bajo cada criterio):

| Semilla | Unanimidad | Umbral 0,90 |
|---|---|---|
| **1** | **(vacía), False** | `.fun`, True |
| 2–10 | `.fun`, True | `.fun`, True |

- **La unanimidad falla en 1 de 10 semillas; el umbral acierta en las 10.** Es exactamente el
  parpadeo que se predijo: JIGSAW aparece o desaparece de la lista de familias con marca según
  qué 50 archivos toque el sorteo. **El umbral queda JUSTIFICADO y se declara en el capítulo.**
  Queda descartada la alternativa «unanimidad + exclusión del `.pdf`», que era más simple pero
  no es estable.
- **Honestidad sobre la tasa:** lo esperado eran ~3,0 fallos en 10 y se observó 1. No hay
  contradicción —P(≤1 fallo | p=0,3032, n=10) = 0,144— pero 10 ensayos son pocos: la tasa
  observada 1/10 tiene IC 95 % de Wilson **[0,018; 0,404]**, que contiene a 0,30. **En la tesis
  se reporta el hecho (falla en 1 de 10) y la probabilidad teórica (0,30), sin presentar 1/10
  como estimación de la tasa.**
- **La corrida canónica (job 3638) usó la semilla 42, una de las que coinciden.** Por eso sus dos
  CSV salieron idénticos: es una de las ~7 de cada 10 en que la unanimidad sobrevive de casualidad.
  El fraseo del punto 5 de la auditoría se mantiene y ahora tiene respaldo empírico.

⚠️ **Efecto colateral del bucle: `deteccion_estructural.py` tenía el MISMO defecto de carpeta
única que `clasificador_bytes.py`**, así que las 10 corridas se pisaron entre sí y
`resultados_estructural/` **en el cluster** quedó con los CSV de la **semilla 10**, no con los
del job 3638. Los locales en `4_resultados/resultados_estructural/` son los del 3638 y están
intactos (siguen siendo la fuente canónica), pero **no volver a bajar esa carpeta del cluster sin
re-correr con semilla 42**. Del bucle sobrevive solo la línea de JIGSAW, no la ablación por
semilla. → ✅ Arreglado el 2026-08-17, igual que el clasificador de bytes.

#### ★★ CUÁNTO CUESTA EL PARPADEO — semilla 1 completa (2026-08-17, `srun`, 30 fam. / 1.500 arch.)

Registrado de lo pegado. La semilla 1 es la única de las diez en que la unanimidad falla, así que
es el caso peor y da la cota del daño:

| Modo | Unanimidad | Umbral 0,90 | Diferencia |
|---|---|---|---|
| Solo extensión | 0,833 (1250/1500) | **0,866** (1299/1500) | **+0,033** |
| Solo firmas binarias | 0,533 (800/1500) | 0,533 (800/1500) | 0,000 |
| **Combinado** | **0,867** (1300/1500) | **0,899** (1349/1500) | **+0,032** |
| Cobertura combinada | 0,880 | 0,913 | +0,033 |
| Familias con marca | 26/30 | 27/30 | +1 |

- **El criterio de unanimidad no solo cambia la lista de familias: cuesta 0,032 de exactitud
  combinada.** Ya no hay que argumentar el cambio de criterio en abstracto, hay un número.
- Las firmas binarias dan idéntico en los dos criterios (0,533), como corresponde: **el umbral
  solo afecta la extensión.** Sirve de control de que el parche no toca lo que no debía.
- Bajo unanimidad, la semilla 1 reproduce exactamente el cuadro viejo del job 3632: las cuatro
  sin marca son BADRABBIT, JIGSAW, NOTPETYA, SUNCRYPT.

#### ★★ HALLAZGO NO BUSCADO: la firma binaria de BADRABBIT también depende de la semilla

**Bajo umbral 0,90, la semilla 1 deja a BADRABBIT SIN MARCA** (`['BADRABBIT', 'NOTPETYA',
'SUNCRYPT']`, 27/30). Con la semilla 42 (job 3638) BADRABBIT **sí** tiene marca: sufijo de 18 B
`65006e006300720079007000740065006400` = «encrypted» en UTF-16LE — verificado en
`4_resultados/resultados_estructural/marcas_por_familia_umbral_90.csv`, fila 3, con extensión
vacía. Las dos fuentes están confirmadas abriendo los archivos, y se contradicen entre semillas.

**Qué implica, y es lo importante:** BADRABBIT no cambia la extensión, así que su única marca
posible es la firma. Si desaparece con otra muestra, el sufijo «encrypted» **no está en todos sus
archivos** — está en los 50 que sorteó la semilla 42. Y eso expone una **asimetría del parche del
16-08**: se le puso umbral de mayoría a la extensión, pero el criterio de firma binaria sigue
siendo *unanimidad sobre los bytes* (prefijo/sufijo común a TODOS los archivos de la muestra, sin
umbral). **Es el mismo defecto que acabamos de corregir, en el otro eje del detector.**

**Consecuencia sobre lo ya escrito:** el hallazgo «BADRABBIT SÍ deja firma — y resuelve una
anomalía registrada» (sección del job 3638) **queda matizado, no anulado**: la firma existe y es
reproducible con semilla 42, pero no es universal. Al escribirlo hay que decir «detectada en la
muestra de 50 archivos de la semilla 42», no «BADRABBIT deja firma».

#### ★★★ MECANISMO COMPLETO, VERIFICADO CSV CONTRA CSV (2026-08-17)

Comparando `marcas_por_familia_umbral_90.csv` de la semilla 1 (cluster) contra el del job 3638
(semilla 42, local) fila por fila: **de las 30 familias, 28 tienen marcas idénticas.** Solo dos
cambian, y cada una por un motivo distinto:

| Familia | Semilla 42 (job 3638) | Semilla 1 | Naturaleza del cambio |
|---|---|---|---|
| **BADRABBIT** | sufijo **18 B** «encrypted» UTF-16LE | sufijo **0 B**, sin marca | **colapso total del LCS** |
| **BLACKBASTA** | sufijo **2 B** (`0000`, < MIN_MARCA) | sufijo **4 B** | **cruce del umbral MIN_MARCA=4** |

**Son dos fragilidades distintas del detector, no una:**
1. **El sufijo común es un LCS, o sea unanimidad byte a byte: un solo archivo discrepante lo lleva
   a CERO.** BADRABBIT no pasó de 18 a 3 bytes: pasó a 0. Alcanza que un archivo de los 50 difiera
   en el último byte.
2. **MIN_MARCA = 4 es un borde duro.** BLACKBASTA ronda los 2-4 bytes de sufijo común y entra o
   sale de la lista de familias con firma según la muestra.

**Y esto explica los dos números que parecían raros, con la aritmética cerrada:**

- **Por qué `solo_firmas_binarias` da 800/1500 en las DOS semillas.** En el LOO del 3638
  (`clasificacion_loo_umbral_90.csv`, local) hay exactamente **16 familias con
  `recall_solo_firmas_binarias = 1,0`**, y 16 × 50 = 800. En la semilla 1, BADRABBIT sale de esa
  lista y BLACKBASTA entra: **siguen siendo 16 familias, pero no las mismas.** El 0,533 idéntico
  **es compensación, no estabilidad** — y escribirlo como «las firmas binarias son estables entre
  semillas» sería un error de lectura.
- **Por qué el combinado cae 0,933 → 0,899.** Son **1400 → 1349 aciertos, es decir −51**, y
  BADRABBIT aporta 50 archivos. En el LOO del 3638 BADRABBIT tiene
  `recall_solo_extension = 0,0` y `recall_solo_firmas = 1,0`: **la firma era su ÚNICA vía de
  identificación** (no cambia la extensión). Al perderla, pierde los 50. BLACKBASTA no compensa
  acá porque ya acertaba por extensión, así que ganar firma no le suma nada en modo combinado.
  **La caída del combinado entre semillas es, casi exactamente, BADRABBIT.**

Las otras familias que dependen de una sola vía, según el mismo LOO: **MAZE** vive solo de su
firma (extensión 0,0 — extensión aleatoria por lote) y **NOTPETYA / SUNCRYPT** no tienen ninguna
(0,0 en los tres modos). MAZE es entonces la próxima candidata a parpadear si su sufijo de 8 B se
rompe en otra muestra.

#### ✅ TODO CONFIRMADO CON LOS DOS CSV DE LA SEMILLA 1 (2026-08-17)

`clasificacion_loo_umbral_90.csv` de la semilla 1, contra el del job 3638:

| Familia | LOO job 3638 (ext / firma / comb.) | LOO semilla 1 | Lectura |
|---|---|---|---|
| **BADRABBIT** | 0,0 / **1,0** / **1,0** | 0,0 / **0,0** / **0,0** | pierde su única vía: −50 archivos |
| **BLACKBASTA** | 1,0 / **0,0** / 1,0 | 1,0 / **1,0** / 1,0 | gana firma; en combinado no suma |
| **JIGSAW** | 1,0 / 0,0 / 1,0 | **0,98** / 0,0 / **0,98** | ver abajo |

**La aritmética cierra exacta en las dos corridas** (no queda nada sin explicar):
- Semilla 1: 26 familias × 1,0 + JIGSAW 0,98 (=49) + 3 familias en 0,0 → 1300 + 49 = **1349/1500
  = 0,899** ✓ lo reportado.
- Job 3638: 28 × 1,0 + 2 en 0,0 → **1400/1500 = 0,933** ✓ lo reportado.
- Firmas: **16 familias con recall 1,0 en ambas semillas** (BADRABBIT sale, BLACKBASTA entra) →
  16 × 50 = **800/1500 = 0,533** en las dos. **Compensación confirmada, no estabilidad.**

**El JIGSAW 0,98 es la prueba interna de por qué la semilla 1 es la que rompe la unanimidad.** Su
muestra de 50 contiene exactamente **un** archivo que no es `.fun`: con umbral 0,90 la familia
conserva la extensión, pero ese archivo suelto no matchea y falla ⇒ 49/50 = 0,98. En el job 3638
los 50 eran `.fun` y daba 1,0. **La huella del archivo discrepante quedó registrada en el
recall.** Es el detalle que convierte «la unanimidad falló en la semilla 1» en un mecanismo
completamente trazado.

#### ⚠ ¿Cuántas «firmas» son relleno? — verificado, y mi hipótesis inicial era medio falsa

Hex de los sufijos, semilla 1:

| Familia | Sufijo | Cuenta como firma (≥4 B) | Naturaleza |
|---|---|---|---|
| CONTI | `0000000000` (5 B) | **sí** | **ceros puros = relleno** |
| BLACKBASTA | `00020000` (4 B) | **sí** | un único byte no nulo: trailer estructurado débil |
| HELLOKITTY | `dadcccab` (4 B) | sí | datos reales |
| AVOSLOCKER | `3d3d` (2 B) = «==» | no | probable relleno base64 |
| DHARMA | `00` (1 B) | no | relleno |

- **BLACKBASTA NO es relleno de ceros puros** como había supuesto: es `00020000`, con un `02`.
  Probablemente un campo de un trailer (longitud o versión). Que en la semilla 42 diera `0000` y
  acá `00020000` sugiere que **el sufijo real es más largo y el LCS lo trunca** según qué archivos
  entren en la muestra. Firma débil, no espuria.
- **CONTI sí es relleno puro y cuenta como firma en las dos semillas** (`recall_solo_firmas` 1,0).
  Cinco ceros consecutivos no son un marcador deliberado del atacante: lo más probable es el
  relleno del cifrado por bloques.
- De las 16 firmas del 3638, **15 tienen contenido no trivial** (cadenas legibles «WANACRY!»,
  «FIDEL.CA», «LOCK96», «.sz40», «encrypted»; blobs de 64 B en CERBER, LOCKBIT, RANSOMEXX,
  TESLACRYPT; y los valores que coinciden con los `sample_bytes` de ID Ransomware en MAZE,
  GANDCRAB, MEDUZALOCKER). **La única excepción es CONTI.**

**A declarar en el capítulo:** no todas las firmas descubiertas son marcadores deliberados —
1 de 16 es relleno y 1 más es débil. Encontrarlo nosotros vale más que que lo pregunte el tutor.
**Mejora posible del detector (no hacer ahora, anotarla):** exigir que la firma tenga entropía o
al menos un byte no nulo distinto de relleno, para no contar padding como marca.

## ★★★ A.4 CERRADO — EXP. 2b SOBRE 10 SEMILLAS (2026-08-17, `srun`, 30 fam. / 1.500 arch.)

**Este es el resultado que cierra el frente del detector.** Diez corridas, semillas 1-10, los dos
criterios sobre la misma muestra. Registrado de lo pegado; medias y desvíos calculados de esas
cifras.

| Modo | Unanimidad (criterio viejo) | **Umbral 0,90 (criterio nuevo)** |
|---|---|---|
| Solo firmas binarias | 0,5099 ± 0,0159 [0,500; 0,533] | **0,5634 ± 0,0020** [0,560; 0,566] |
| **Combinado** | **0,9000 ± 0,0156** [0,867; 0,933] | **0,9321 ± 0,0011** [0,930; 0,933] |
| Familias con marca | **26, 27 o 28** según el sorteo | **28/30 en las diez, sin excepción** |

**Las tres conclusiones, en orden de importancia para la tesis:**

1. **El criterio de mayoría reduce el desvío 14 veces en el combinado** (0,0156 → 0,0011) **y 8
   veces en las firmas** (0,0159 → 0,0020). El resultado deja de depender del sorteo: eso es lo
   que hace publicable la cifra.
2. **Y además mejora la media**: combinado **+0,0321**, firmas **+0,0535**. No hay que elegir
   entre estabilidad y rendimiento; el mismo cambio da las dos cosas.
3. **El conteo de familias con marca pasa a ser 28/30 en las diez semillas.** Bajo unanimidad
   oscilaba entre 26 y 28 — o sea que *la lista de familias que va al capítulo* dependía de la
   semilla. Con el criterio nuevo, **NOTPETYA y SUNCRYPT son las únicas dos sin marca, siempre.**

**⚠️ Dato que hay que escribir con honestidad: el 0,933 que YA está en el capítulo (job 3638,
semilla 42) es el MÁXIMO del rango bajo unanimidad, no su media (0,9000).** Salió esa cifra por
suerte del sorteo. Bajo el criterio nuevo, **0,933 pasa a ser el valor esperado** (media 0,9321).
La cifra escrita queda en pie, pero **por un motivo distinto del que se creía**: antes era el techo
afortunado, ahora es la media legítima. Escribirlo así.

**✅ La predicción teórica se cumplió, con datos independientes.** Bajo unanimidad, BADRABBIT fue
detectado en **1 de 10 semillas** (solo la 3). La hipergeométrica sobre la base real (822/857)
predecía p = 0,1167, es decir **1,2 de 10**. El modelo probabilístico queda validado: no era una
racionalización a posteriori.

**JIGSAW,** por su parte, perdió la extensión bajo unanimidad solo en la semilla 1 — **el mismo
resultado que el bucle anterior**, lo que confirma que las corridas son reproducibles por semilla.

**Corrección a lo que anoté con la sola semilla 42:** ahí escribí que el cambio de criterio «casi
no mueve las cifras, su valor es la estabilidad». Era cierto *para esa semilla* —la afortunada— e
incompleto en general: sobre las diez, el criterio nuevo **mejora la media y reduce el desvío**.

**Cifras del Exp. 2b para el capítulo 4** (criterio declarado = umbral 0,90, media ± desvío sobre
10 semillas): **combinado 0,932 ± 0,001 · solo firmas 0,563 ± 0,002 · 28 de 30 familias con
marca.** La extensión sola no cambia entre criterios (0,867). ⚠️ Falta calcular la media del modo
«solo extensión» y de las coberturas: están en los `log_estr_s*.txt` del cluster, **bajarlos antes
de que se limpie el `/scratch`**.

## ★★★ EL ARREGLO DEL CASO BADRABBIT, YA IMPLEMENTADO (2026-08-17)

> El volcado, la aritmética del 0,17 y el descarte de la contaminación están en la sección
> **«RESUELTO — LA FIRMA DE BADRABBIT ES LA PALABRA «encrypted»»** al principio de este
> documento. Acá va solo lo que se hizo con eso.

### ✅ La cifra definitiva, sobre la base que el detector realmente usa (2026-08-17)

Recuento en el cluster excluyendo los `.pdf`, que es lo que el detector muestrea:
**822 de 857 archivos llevan el marcador = 95,92 %.**

| Base | Cobertura | P(los 50 lo lleven todos) | El detector NO la encuentra en |
|---|---|---|---|
| **Sin `.pdf` — la del detector** | **822/857 = 95,92 %** | **0,1167** (hipergeométrica) | **88,3 % de las semillas** |
| Con `.pdf` (primer conteo) | 965/1.000 = 96,50 % | 0,1608 | 83,9 % |

**Las cifras a citar en la tesis son 95,92 % de cobertura y 0,117 de probabilidad de detección**
(hipergeométrica exacta, `comb(822,50)/comb(857,50)`; el 0,965⁵⁰ ≈ 0,17 era binomial y sobre la
base con `.pdf`). La conclusión se refuerza: **el detector fallaba en ~88 % de las semillas**, no
en 83 %.

**Hallazgo lateral que sale del mismo conteo:** 1.000 − 857 = **143 `.pdf`**, y 965 − 822 =
**143 con marcador**. Los números coinciden ⇒ **los 143 `.pdf` de BADRABBIT son archivos cifrados
CON el marcador**, no documentación. Consecuencia incómoda y cuantificada: el filtro `.pdf`
**empeora** la detección de BADRABBIT (descarta 143 archivos marcados y baja la cobertura de
96,50 % a 95,92 %). El caveat del filtro ya estaba anotado; ahora tiene número. **En BADRABBIT y
NOTPETYA el `.pdf` no es documentación** — son familias que conservan la extensión original.

### ✅ CAMBIO DE CRITERIO IMPLEMENTADO Y PROBADO (`deteccion_estructural.py`)

**El umbral de mayoría se aplica ahora también al prefijo y al sufijo**, no solo a la extensión.
La firma binaria exigía unanimidad byte a byte, que es *más* frágil que la extensión: un único
archivo discrepante lleva el trozo común a **cero** (por eso BADRABBIT pasó de 18 B a 0, no a 3).

1. `prefijo_mayoritario` / `sufijo_mayoritario` reemplazan al LCS: el trozo más largo compartido
   por al menos el umbral de los archivos.
2. **Cada marca sale con su COBERTURA** (`prefijo_cobertura`, `sufijo_cobertura`,
   `extension_cobertura` en el CSV) — el pedido 3: «tiene marca» y «todos sus archivos la tienen»
   son cosas distintas, y para el detector manda la segunda.
3. El LOO ya no recalcula las marcas de las demás familias por cada archivo (no dependen del
   archivo evaluado): mismo resultado exacto, 30 veces menos trabajo.
4. Los **dos criterios se siguen reportando** en la misma corrida, así que `_unanimidad` conserva
   el comportamiento viejo completo y `_umbral_90` trae el nuevo. Se mantiene la decisión ya
   tomada de reportar ambos.

**Verificaciones hechas antes de subirlo:**
- **Equivalencia:** con umbral 1,0 el algoritmo nuevo da **exactamente** el mismo resultado que el
  LCS anterior — 6.000 comparaciones sobre datos aleatorios, **0 diferencias**. Las cifras del job
  3638 siguen reproducibles.
- **Caso BADRABBIT sintético** (familia con el marcador en 96/100 archivos y sin extensión propia):
  bajo unanimidad queda **sin marca**; bajo umbral 0,90 aparece `sufijo 18B …encrypted (96%)` y el
  combinado sube de **0,800 a 0,992**. Los 4 archivos que genuinamente no lo llevan fallan, que es
  la cobertura real y no un error.
- Costo: 6 s con 250 archivos / 5 familias ⇒ estimo **3-6 min** con 1.500 / 30, contra los 53 s del
  3638 (el criterio de mayoría es más caro que el LCS). Pedir `--time=00:30:00`.

### ★ CORRIDA CON EL CRITERIO NUEVO — job 3650, semilla 42 (2026-08-17)

**1) Retrocompatibilidad confirmada en datos reales, no solo en el test sintético.** El criterio
`unanimidad` reproduce el job 3638 **cifra por cifra**: extensión 0,867 (1300/1500) · firmas 0,533
(800/1500, cobertura 0,541) · combinado 0,933 (1400/1500, cobertura 0,941) · 28/30 con marca ·
sin marca NOTPETYA y SUNCRYPT. Nada se movió donde no debía moverse.

**2) Qué cambia con el criterio de mayoría (`umbral_90`):**

| Modo | Unanimidad | Umbral 0,90 |
|---|---|---|
| Solo extensión | 0,867 | 0,867 (igual) |
| **Solo firmas binarias** | 0,533 (800), cob. 0,541 | **0,563 (845), cob. 0,571** |
| Combinado | 0,933 | 0,933 (igual) |

Dos familias ganan firma: **BLACKBASTA** sufijo 4 B `00020000` al **98 %** y **CUBA**, ver abajo.
**El combinado NO se mueve porque las dos ya acertaban por extensión.**

⚠️ **Lo importante de leer bien:** con la semilla 42 el cambio de criterio casi no mueve las
cifras (+0,030 en firmas, 0 en el resto). **El valor del cambio NO es la cifra de esta semilla,
es la estabilidad entre semillas** — con la semilla 42 BADRABBIT aparecía igual bajo unanimidad
(es la afortunada del 12 %). La ganancia se ve en el otro ~88 % de las semillas, y eso lo mide
A.4. **No escribir «el criterio de mayoría mejora el resultado» apoyándose en esta corrida.**

**3) El riesgo de relleno quedó AUDITADO y limpio.** Temía que el umbral en las firmas inflara
marcas de ceros: no pasó. Las dos firmas nuevas son `00020000` (tiene un byte no nulo) y la de
CUBA (empieza con «FIDEL.CA»). CONTI sigue con sus `0000000000` al 100 %, igual que antes — el
umbral no agregó ninguna falsa marca de relleno.

**4) HALLAZGO NUEVO: la ventana de 64 bytes está TRUNCANDO las firmas.** CUBA pasa de **prefijo
20 B al 100 %** a **prefijo 64 B al 92 %**, y 64 es exactamente `N_BYTES`, el máximo que el
detector mira. O sea que **la firma de CUBA tiene al menos 64 bytes**, no 20. Están topeadas en 64
también TESLACRYPT (prefijo, 100 %) y CERBER, LOCKBIT, RANSOMEXX (sufijos, 100 %): **cinco firmas
contra el techo de la ventana.** Es un dato utilizable —«al menos 64 bytes», más fuerte que
20— y es barato de medir: subir `N_BYTES` y volver a correr. **Anotarlo como pendiente, no
cambiarlo antes de A.4**, porque mover la ventana también mueve todas las cifras.

## ★ PEDIDOS DE CAPPO YA VERIFICADOS EN EL CÓDIGO (2026-08-17) — no reabrir

Sobre «error» y «train/validation/test», verificado leyendo `clasificador_bytes.py` y
`4_resultados/resultados_bytes/bytes_resumen.csv` (job 3639):

| Etapa | Qué es | Exactitud | macro-F1 | n |
|---|---|---|---|---|
| **Búsqueda anidada** | `RandomizedSearchCV` **dentro** del pliegue de entrenamiento, evaluación en el pliegue externo que no participó de la selección | **0,8913** | **0,8920** | 6.000 |
| Final | re-evaluación con hiperparámetros fijos sobre otra muestra | **0,9089** | **0,9074** | 15.000 |

- **El 0,8913 es la cifra metodológicamente inatacable** y contesta el pedido de validación
  separada: ya existe validación cruzada anidada, no hay que agregar nada.
- **El matiz a declarar:** en la etapa final los hiperparámetros se eligieron con archivos que en
  parte reaparecen en la muestra de evaluación. **Dejar el 0,8913 disponible como cifra
  conservadora** y decir explícitamente de dónde sale cada una.
- Las tres representaciones de la etapa de búsqueda, del mismo CSV: posicional + RF **0,8913** ·
  n-gramas + LogReg **0,8623** · n-gramas + LinearSVC **0,8515**. La posicional gana también acá.
- **Lo único que falta del pedido del tutor sobre archivos es el desvío: es el job 3648.**

#### ★★ CONSECUENCIA: el Exp. 2b también necesita desvío, no solo el 2c

Comparando el **mismo criterio** (umbral 0,90) entre dos semillas: combinado **0,899** (semilla 1)
contra **0,933** (semilla 42, job 3638). **Diferencia 0,034 entre muestras.** Las cifras del
Exp. 2b están hoy reportadas como valores exactos (0,867 / 0,533 / 0,933) y **tienen una
dispersión del mismo orden que la que A.2 está midiendo para el 2c**.

**Tarea nueva (barata: 27 s por semilla ⇒ ~5 min las diez):** correr el detector sobre 10 semillas
y reportar el 2b como media ± desvío, igual que el resto. Ahora es posible sin perder datos porque
la carpeta de salida ya lleva la semilla. **No escribir el capítulo del 2b con cifras puntuales
antes de tener esto.**

### 2. `clasificador_bytes.py` sobrescribe su carpeta de salida — ✅ **ARREGLADO 2026-08-17**
Escribía siempre en `resultados_bytes/`: por eso el job 3639 pisó los CSV del 3633 y se perdió
su reporte por familia completo. **El próximo paso del plan es A.2, que corre varias semillas.**
Si cada una escribiera en la misma carpeta, quedaría solo la última y se perdería justo la
dispersión que pidió el tutor.

**Qué se cambió** (probado local con dataset sintético de 4 familias, antes de gastar cluster):
1. **Carpeta única por corrida:** `resultados_bytes_s<semilla>[_job<SLURM_JOB_ID>]`, armada en
   `carpeta_salida()`. `resultados_bytes/` (el 3639) queda intacta y ya no la pisa nadie.
2. **Salvaguarda:** si la carpeta destino ya tiene archivos `bytes_*`, el script **aborta** con
   un mensaje que dice qué hacer. Para sobrescribir hay que pedirlo con `--forzar`.
3. **Semillas parametrizables** — sin esto, «correr diez semillas» daba diez veces el mismo
   número: el muestreo estaba clavado en 42 (búsqueda) y 7 (etapa final), y todos los
   `random_state` en 42. Ahora `--semilla` / `--semilla-final`, **con esos mismos valores por
   defecto para que el job 3639 siga siendo reproducible**.
4. **Modo `--multisemilla 1,2,3,…`** (esto ES el A.2): repite solo la evaluación final con los
   hiperparámetros ya elegidos por la búsqueda anidada (`HIPER_2C`, constante nombrada en el
   script) y escribe `bytes_multisemilla.csv` (una fila por semilla),
   `bytes_multisemilla_por_familia.csv` (**F1 por familia y por semilla ⇒ desvío por familia,
   que es lo que hace falta para las seis difíciles**) y `bytes_multisemilla_resumen.csv`
   (media, desvío, mínimo, máximo). Los CSV se reescriben en cada iteración: si el trabajo se
   corta por tiempo, lo ya corrido no se pierde.
5. La semilla gobierna **las tres fuentes de azar**: qué archivos se muestrean, cómo se parten
   los pliegues y la aleatoriedad interna del bosque.
6. **El manifiesto guarda ahora `SLURM_JOB_ID`**, las semillas y los pliegues (hallazgo 7 de la
   auditoría, que pedía exactamente esto para scripts futuros).

**A declarar en la tesis:** en el modo multisemilla los hiperparámetros quedan **fijos**. Lo que
se mide es la dispersión de la estimación, no una nueva selección de modelo — es el mismo
criterio que usa `clasificador_notas_v2.py`, que reporta 10 semillas con la configuración ya
elegida.

**Listo para subir (2026-08-17):** los tres archivos están copiados en
`PARA_SUBIR_AL_CLUSTER/` (verificados por hash contra `2_codigo/`): `clasificador_bytes.py`,
`deteccion_estructural.py` y el nuevo `job_bytes_multisemilla.sh`. `LEEME_PRIMERO.txt` de esa
carpeta tiene arriba un bloque con fecha que dice qué subir y los dos comandos a lanzar (A.2 y la
semilla 1 del detector). Recordar que esa carpeta está en `.gitignore`: no entra en los commits,
hay que mantenerla a mano.

### Nota menor pero contagiosa — ✅ **ARREGLADA 2026-08-17**
`bytes_manifiesto.json` guardaba comparaciones hardcodeadas (`extension_sola: 0.828`,
`estadisticas_2_features: 0.099`) que quedaron viejas tras el parche del detector (los valores
correctos del job 3638 son 0,867 y 0,533). Un manifiesto que propaga cifras congeladas es peor
que no tenerlas. **Se eliminó el bloque `comparacion` del script**, con un comentario en el
lugar explicando por qué: un manifiesto describe SU corrida; las comparaciones entre
experimentos se arman leyendo los CSV de cada uno.

### Estado verificado de la descarga
Todo bajado en `4_resultados/` el 2026-08-17 (22:04 y 22:34): `resultados_ablacion_extendida/`
5 archivos · `resultados_analisis_bytes/` 6 · `resultados_bytes/` 3 · `resultados_estructural/`
7. `bytes_resumen.csv` confirma que es el **3639**: 0,9089 / 0,9089 / 0,9074 sobre 15.000
archivos y 30 familias. **Solo faltan los `slurm-*.out`.**

## ✅ AUDITORÍA DE B.3 CERRADA (2026-08-19) — el filtro es limpio; los parentescos NO salen del grafo

Los dos pendientes que dejó B.3, auditados sobre `b3_valores_excluidos.csv` y `b3_aristas.csv`:

### 1. El filtro de circularidad pasa la auditoría — el 15-18 % es citable
Se revisaron **todos** los valores excluidos. Cada uno contiene de verdad el nombre o alias de
su familia, porque son infraestructura con marca: `contirecovery.xyz`, `bastad5…onion` (lleva
«basta»), `alphvmmm…onion`, decenas de `lockbit*.onion`, `gandcrabmfe6mnef.onion`,
`phobos_helper@xmpp.jp`, `mazedecrypt.top`, `medusa*`, `cuba*`, `darksidedxcftmqa.onion`.
**El temido falso positivo tipo «conti dentro de continue» no existe: cero exclusiones
espurias.** La aritmética cierra: 146 aristas dentro de familia sin exclusión → 125 con
exclusión = **21 eliminadas**, como estaba anotado. Nota para el capítulo: las exclusiones se
concentran en LOCKBIT (la mayoría de los 21), así que el «15-18 % de la continuidad viene de
valores con el nombre adentro» es en buena parte un fenómeno de LOCKBIT, no repartido.

### 2. ⚠️ Las 7 aristas entre familias NO validan los parentescos — corregir la expectativa
Se esperaba que fueran BLACKBASTA↔CONTI y DHARMA↔PHOBOS por marcadores compartidos. La
realidad, valores completos:

- BLACKBASTA↔CONTI (4 aristas): comparten la URL `https://torproject.org` — **la grafía exacta
  sin `www` y sin barra final**, que escapó al filtro de infraestructura común porque solo esas
  dos familias la escriben así.
- CERBER↔CRYPTOLOCKER (3 aristas): comparten un enlace viejo de descarga de Tor
  (`download-easy.html.en`). Dos familias antiguas usando la misma página de época. **No es
  parentesco.**
- **DHARMA↔PHOBOS no aparece en ninguna variante**: su vínculo es por contenido casi duplicado
  (grupo 55), no por IOCs; el email `phobos_helper@…` queda excluido por llevar el nombre.

**Conclusión honesta para el capítulo:** los parentescos BLACKBASTA↔CONTI y DHARMA↔PHOBOS los
encontró la **deduplicación por contenido** (grupos 6 y 55), no el grafo de marcadores. Tras
filtrar la infraestructura común, **en este corpus no queda ningún IOC operativo (email,
billetera, onion) compartido entre familias distintas** — los IOCs reales son privados de cada
familia. Eso es un resultado en sí (los marcadores no confunden familias entre sí), pero la
frase «el grafo reencuentra los parentescos» **no debe escribirse**: la única arista
BLACKBASTA↔CONTI que sobrevive es una grafía peculiar de una URL pública, consistente con la
plantilla copiada pero no un IOC compartido.

## ★ DECISIÓN 2026-08-20 — EXTENSIÓN DE FAMILIAS EN EL FRENTE DE NOTAS

El tutor pidió agregar más familias. Resolución de Romina: **el núcleo de 30 familias queda
intacto y registrado tal como se midió; la extensión entra como experimento nuevo que se
AGREGA al capítulo, con su base declarada.** Nada de lo escrito sobre 30 se corrige, porque
no está mal: está medido sobre otra base. Reglas de la extensión:

- Solo frente de **notas** (en archivos no existe dato público fuera de NapierOne — límite
  externo, citable).
- Las familias nuevas se etiquetan como extensión en el manifiesto y sus cifras se reportan
  **por separado** del núcleo, cada una con su azar y su conteo de familias.
- Umbral de admisión que sale de B.1: una familia nueva necesita **≥ 2 plantillas** para ser
  evaluable en P2 y **~4 para rendir**; sumar familias de 1 nota solo arrastra el macro-F1
  hacia abajo sin aportar información.
- Fuente principal ya identificada: ThreatLabz (209 familias / 345 archivos registrados en §6).

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

## SPRINT 2 — ✅ CERRADO (lanzado 2026-08-05, resultados incorporados)

**Los dos trabajos volvieron y sus resultados ya están arriba en este documento.** El crítico,
(a) generalización a tipos de archivo nunca vistos, dio **0,879 frente a 0,910: una caída de
0,031**, muy por debajo del umbral de 0,10 que se había fijado ⇒ **el resultado principal
queda confirmado**, no hay que matizar §4.5 ni §4.6. El gridsearch de estadísticas dio
0,599–0,603, diferencia −0,000, y cerró el último hueco de optimización declarado.
_(Se deja el texto original abajo como registro de lo que se había planificado.)_

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
GPU disponible (no la usa sklearn), almacenamiento declarado como temporal.
> ⚠️ **CORREGIDO 2026-08-17:** el reglamento dice que `/scratch` se borra a los 60 días
> del fin de uso, pero **en la práctica no se limpia** — Romina tiene ahí archivos de
> más de un año. Bajar los resultados igual, por respaldo, pero **no usar el borrado
> como argumento de urgencia**.
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
Preguntarle: (a) ✅ **RESUELTA — el tutor autorizó OCR/transcripción e incluso la ampliación
sintética** (punto 4 textual de la reunión 2026-08-12: «se puede generar datos sintéticos si es
necesario»; confirmado por Romina 2026-08-16). Sigue vigente el caveat propio: si se generan
sintéticas, evaluar SOLO contra notas reales; (b) ¿mapear objetivos específicos a capítulos?
(su pedido del 08/05/24); (c) ¿experimento transformer con GPU como sección extra o trabajo
futuro?

**Bloque E — Cierre final (AL FINAL, una sola pasada):** **CONCLUSIÓN completa** (recién acá;
corregir las cifras superadas 100 %/15,4 %), resumen/abstract, carátulas, dedicatoria,
agradecimientos, lista de símbolos, estilo, huérfanas del .bib, duplicados PDF.

**Pendiente siguiente (Fases 1-2 del DIAGNOSTICO):** conclusión desincronizada (cita 100 % y
15,4 % viejos; lista como pendiente lo ya hecho), front matter (carátulas plantilla, resumen/
abstract sin escribir, dedicatoria "blah blah"), bibliografía (entrada lee2022 corrupta,
incorporar Davies 2022 ×2 / Gómez Hernández 2023 / Pont 2023 / Sokolova & Lapalme), auditar
las 37 notas "NapierOne/varios" del manifiesto, y Fase 3 (expandir corpus con URLs pcrisk ya
listadas, GridSearch en servidor, advanced_features para blindar Exp. 2).

## ▶ RETOMAR ACÁ — 2026-08-17: job 3639 verificado; falta bajar TODO del cluster

| Job | Qué es | Estado | Salida a bajar |
|---|---|---|---|
| 3630 | Ablación de ventana extendida + bloque del medio | ✅ 192,4 min | `resultados_ablacion_extendida/` |
| 3633 | Exp. 2c sobre 30 familias (con los 12 JPEG de CERBER) | ✅ · ⚠ CSV pisados por el 3639 | solo queda `slurm-bytes-3633.out` |
| 3632 | Exp. 2b estructural sobre 30 | ⚠ superado por 3638 | — |
| 3638 | Exp. 2b con detector corregido (sin `.pdf`, muestreo aleatorio, 2 criterios) | ✅ 53 s | `resultados_estructural/` |
| **3639** | **Exp. 2c sin los 12 JPEG en claro de CERBER** | ✅ 45 min (17-08 00:22) | `resultados_bytes/` |
| **3648** | **A.2: Exp. 2c sobre 10 semillas (0-9), desvío del frente de archivos** | ⏳ lanzado 17-08 | `resultados_bytes_multisemilla_job3648/` |
| — | Detector, semilla 1 (costo del parpadeo) — `srun`, 30 fam. | ✅ 17-08 | `resultados_estructural_s1_job*/` |

### ✅ Job 3639 (2026-08-17) — la contaminación de CERBER no sostenía el resultado

`exactitud 0,909 | balanced 0,909 | macro-F1 0,907` — 30 familias × 500 = 15.000 archivos,
mismos hiperparámetros elegidos por la búsqueda anidada (300 / prof. 20 / hoja 2 / 0,3).
Contra el job 3633 (0,9105 / 0,9105 / 0,9097): **diferencia −0,002 a −0,003, muy por debajo
del umbral de 0,005 fijado** ⇒ queda verificado y escribible que los 12 archivos sin cifrar
no sostenían el resultado. **CERBER da precisión 1,00 / recall 1,00 / F1 1,00** (antes
1,00/0,99): la firma de 64 bytes alcanza sola; la cabecera JFIF no era la muleta.
Del extracto visible (A–C): AVOSLOCKER 1,00 · BADRABBIT 0,98 · BLACKBASTA 1,00 ·
BLACKCAT 1,00 · BLACKMATTER 0,99 · CERBER 1,00 — consistente con el 3633.

**⚠ Confirmado (ls del 17-08): el 3639 PISÓ los CSV del 3633.** `resultados_bytes/` contiene
solo los tres archivos del 17-08 01:22 (`bytes_resumen.csv`, `bytes_por_familia.txt`,
`bytes_manifiesto.json`), todos de la corrida sin JPEG. Las cifras del 3633 sobreviven en
`slurm-bytes-3633.out` (bajarlo) y en las secciones de este documento, pero su reporte por
familia completo se perdió (el log solo imprime el extracto A–C). Causa: `clasificador_bytes.py`
escribe siempre en la misma carpeta; para futuras verificaciones, renombrar la carpeta de
salida antes de relanzar.

**PROPUESTA (decidir Romina):** tratar el **3639 como corrida canónica del Exp. 2c** — es la
del corpus verificado sin archivos en claro y la única con CSV conservados — y citar el 3633
como control de robustez («incluir los 12 archivos mueve las métricas menos de 0,003»).
El 3639 corrió con los 12 `.jpg` movidos a `CERBER-small/_sin_cifrar/` (exclusión reversible).

**✅ DESCARGA HECHA (17-08 22:04) Y AUDITADA — ver la sección de auditoría más abajo.**
Faltan SOLO los logs `slurm-*.out` (en particular `slurm-bytes-3633.out`, única traza del
job 3633 desde que el 3639 pisó sus CSV). Comando en el cluster y bajar con WinSCP:

```bash
cd /scratch/ralfonzo/tesis && tar czf logs_slurm_2026-08-17.tgz slurm-*.out
```

## ★ AUDITORÍA DE LA DESCARGA (2026-08-17, subagente de verificación)

**43 de 43 cifras de referencia CONFIRMADAS contra los CSV/JSON bajados; 15 CSV + 5 JSON +
2 PNG íntegros, ninguno vacío ni corrupto.** Los archivos locales son desde ahora la fuente
canónica de todas las tablas del cap. 4.

**Las seis difíciles del 3639 (de `bytes_por_familia.txt`, corpus limpio):** NOTPETYA 0,33 ·
JIGSAW 0,44 · CRYPTOLOCKER 0,59 · DARKSIDE 0,59 · WASTEDLOCKER 0,63 · SUNCRYPT 0,72.
**Mismas seis**, con la séptima peor (BADRABBIT 0,98) a 0,26 de distancia — el grupo está
nítidamente separado. ⚠ Dirección: cinco de las seis BAJARON hasta −0,03 respecto del 3633 ⇒
escribir «excluir los JPEG no altera los resultados», nunca «mejora».

**Hallazgo científico nuevo (corrige la hipótesis registrada en la sección de la ablación):**
la importancia por posición NO se concentra «entre los bytes 128 y 512»: se concentra en los
**últimos ~16 bytes del archivo** (27,2 % de toda la importancia; cola total 79,6 % contra
cabecera 20,4 %; los 12 offsets más importantes son todos de cola: −5, −133, −1, −2, −3, −6,
−4, −10, −100, −168…). Existe un **pico secundario real en la cola entre −130 y −170**
(la banda 128–255 de la cola acumula 21,5 %), y ese pico es lo que explica el salto de la
curva de ventana entre 128 y 512. Formulación para la tesis: *«la señal dominante está pegada
al final del archivo; ampliar la ventana de 128 a 512 incorpora un pico secundario en
−130/−170 que aporta el resto»*. Converge con: 15 de las 16 firmas del 2b son sufijos, y
`solo cola` 0,756 contra `solo cabecera` 0,338. Fuente: `b_importancia_por_posicion.csv`
(1.024 filas, importancias suman 1,0) + `fig_importancia_por_posicion.png`.

**Diagnóstico de las seis difíciles (`d_familias_dificiles.csv`), listo para §4.5.3:**
el **97–99 % de sus errores son confusiones entre ellas mismas** (cluster cerrado, no ruido
difuso). Entropía de cabecera 7,51–7,59 contra 7,03 del resto: sus cabeceras son MÁS
aleatorias que el promedio — no hay marca que aprender. Excepciones informativas: SUNCRYPT
tiene cola de baja entropía (4,78) y NOTPETYA 6,58 — algo estructurado al final, coherente
con que alcancen F1 0,72 y 0,33 sin tener firma detectable.

**Hallazgos que piden acción (en orden):**
1. **`generar_figuras_cap4.py` tiene cifras hardcodeadas VIEJAS**: su dict de F1 es de 29
   familias (falta BLACKBASTA) y de una corrida anterior al 3639; su Exp. 2b es el
   pre-corrección (15 firmas, 4 sin marca — BADRABBIT y JIGSAW pintados mal); su figura de
   progresión usa 0,517/0,533/0,828 superados por 0,5333/0,5413/0,8667. **Si se regeneran las
   figuras hoy, salen con números superados.** → Reescribirlo para que LEA los CSV bajados
   (regla: ninguna cifra tipeada a mano) antes de tocar el cap. 4.
2. **La exclusión de los 12 JPEG de CERBER no quedó registrada en ningún artefacto**: el
   manifiesto del 3639 no la menciona y el script solo filtra `.pdf`. Se hizo moviendo los 12
   a `CERBER-small/_sin_cifrar/` en el cluster (los scripts solo toman archivos del nivel de
   la carpeta de familia). → Declararla explícitamente en el cap. 4 y en el apéndice de
   reproducibilidad; el manifiesto por sí solo no la prueba.
3. **Dos CSV del detector VIEJO venían mezclados en `resultados_estructural/`**
   (`marcas_por_familia.csv` y `clasificacion_loo.csv`, sin sufijo de criterio = job 3632
   pre-corrección, con BADRABBIT/JIGSAW «sin marca»). → **Movidos el 2026-08-17 a
   `4_resultados/_historico/resultados_estructural_job3632_precorreccion/`** para que nadie
   los cite por error. Los válidos son los `*_umbral_90.csv` / `*_unanimidad.csv`.
4. **El bloque `comparacion` de `bytes_manifiesto.json` está hardcodeado en el código fuente**
   (`clasificador_bytes.py:279-281`) y trae los valores del 2b pre-corrección (0,533/0,828).
   NO citarlo como medición del job 3639.
5. **Unanimidad y umbral 0,90 dieron CSV byte-idénticos (verificado por hash), pero por suerte
   del muestreo**: los archivos anómalos de JIGSAW no cayeron en la muestra de 50
   (probabilidad 0,6968). Fraseo para la tesis: el umbral sigue siendo necesario en general;
   que acá coincida es evidencia de que no se eligió por conveniencia, no de que dé igual.
   ✅ **RESUELTO 2026-08-17 con 10 semillas: la unanimidad falla en 1 de 10, el umbral en 0 de
   10.** El fraseo se mantiene y ahora está respaldado por medición, no solo por el cálculo.
6. **Al citar la ablación extendida: el CSV de referencia es `accuracy`, la figura grafica
   macro-F1.** Ambas columnas están en `a_curva_ablacion.csv`; aclarar cuál se usa en cada
   tabla/figura. El control «sin relleno» va +0,0037…+0,0053 por encima (media +0,0047):
   decir «~0,005», no «0,005 uniforme». El subconjunto sin relleno es constante en los 7
   puntos (n=14.783 = archivos ≥8.192 B, según `0_tamanos.csv`).
7. **Ningún manifiesto guarda el job ID de SLURM**, y el de la ablación tampoco registra
   pliegues/semilla de la CV (están solo en el código: `ablacion_ventana_extendida.py:117-121`,
   StratifiedKFold(3), semilla 42). Para la tesis se declara desde el código; para scripts
   futuros, agregar `SLURM_JOB_ID` y la CV al manifiesto.

Nota: también bajó `resultados_gridsearch/` (el de notas NLP del 2026-08-05, 144 notas) —
íntegro, sin novedades. En `c_ablacion_ventana.csv` (29 familias) el tramo 64→128 BAJA
(0,7946→0,7932) mientras que sobre 30 familias sube (0,7959→0,7980): ese tramo es ruido,
describirlo como «sin ganancia», sin asignarle dirección.

**Pendiente de escribir, ya medido:** el Exp. 2b sobre 30 familias (incluida la corroboración
cruzada con `Pruebas.xlsx`), la ablación extendida y el Exp. 2c sobre 30.

**Siguiente experimento a preparar:** Sprint B.1 (curva de aprendizaje de notas) **junto con
B.3 (grafo de marcadores compartidos → protocolo P3)** — mismos datos, sin cluster; diseño
completo fijado el 2026-08-16 en `PLAN_MEJORAS.md` (B.3, C.bis y protocolo de sintéticas).
En paralelo, ya autorizado por el tutor: recolección pcrisk/OCR con lote chico (1-2 familias)
midiendo rendimiento por hora.

## ✅ ABLACIÓN DE VENTANA EXTENDIDA + BLOQUE DEL MEDIO (job 3630, 2026-08-16)

`ablacion_ventana_extendida.py` con `--por-familia 500` sobre `Napierone-small`:
**15.000 archivos, 30 familias**, 8 núcleos, 192,4 min. Validación cruzada de 3 pliegues,
**una sola semilla** (por eso no trae desvío: el A.2 del plan sigue pendiente). Hiperparámetros
del Exp. 2c, ajustados para 512+512 y dejados sin tocar para que las cifras sean comparables —
es una limitación a declarar, no un error.

### (0) El riesgo de relleno era chico, y el control salió limpio igual

Tamaños: **mínimo 1.040 · mediana 80.929 · máximo 32.155.703 bytes**.

| Ventana | Archivos más cortos que la ventana |
|---|---|
| 64+64 … **512+512** | **0 (0,0 %)** |
| 1024+1024 | 84 (0,6 %) |
| 2048+2048 | 143 (1,0 %) |
| 4096+4096 | 217 (1,4 %) |

**Hasta 512+512 —donde está el máximo— ningún archivo se rellena con ceros**, así que la
sospecha que motivó el control no aplica al punto que importa. El subconjunto limpio
(≥ 8.192 bytes) son **14.783 archivos y las 30 familias conservan ≥ 30 ejemplares**, o sea que
el control se hace sobre el 98,6 % del corpus y sin perder familias. Escribir el control
igual: el argumento es más fuerte cuando se muestra que se buscó el artefacto y no estaba.

### (a) La curva SATURA en 512+512 — contestado el reclamo del tutor

| Ventana | Bytes | Corpus completo | Solo archivos sin relleno |
|---|---|---|---|
| 64+64 | 128 | 0,796 | 0,801 |
| 128+128 | 256 | 0,798 | 0,803 |
| 256+256 | 512 | 0,851 | 0,856 |
| **512+512** | **1024** | **0,904** | **0,909** |
| 1024+1024 | 2048 | 0,903 | 0,907 |
| 2048+2048 | 4096 | 0,902 | 0,907 |
| 4096+4096 | 8192 | 0,901 | 0,905 |

- **El máximo está en 512+512 y a partir de ahí la curva baja levemente** (−0,003 al octuplicar
  los bytes). La figura ya muestra saturación: era exactamente lo que el tutor marcó como
  crítica válida el 12-08.
- **El control sin relleno se comporta igual** y va sistemáticamente ~0,005 por encima. Como
  hasta 512 no hay relleno posible, esa diferencia constante **no es el relleno**: son los
  archivos chicos, que son intrínsecamente más difíciles. La forma de la curva es la misma en
  las dos vistas ⇒ el salto de 256 a 512 es señal real.
- **La curva tiene un escalón, no una rampa:** plana entre 64 y 128 (0,796 → 0,798), salta en
  256 (0,851) y otra vez en 512 (0,904). **VERIFICADO 2026-08-17 contra
  `b_importancia_por_posicion.csv`: la hipótesis «la información decisiva está entre los bytes
  128 y 512» era INCORRECTA tal como estaba enunciada.** La señal dominante está en los
  **últimos ~16 bytes** (27,2 % de la importancia; cola 79,6 % vs cabecera 20,4 %); lo que
  explica el salto 128→512 es un **pico secundario en la cola entre −130 y −170**. La pista de
  RYUK («HERMES» a offset lejano) sigue abierta pero ya no como explicación principal.
  Detalle en la sección «AUDITORÍA DE LA DESCARGA».
- **Reproducibilidad:** la ablación previa sobre 29 familias daba 64→0,795 · 128→0,793 ·
  256→0,852 · 512→0,908. Sobre 30 familias y otro muestreo: 0,796 · 0,798 · 0,851 · 0,904.
  Coinciden dentro de ±0,005 — es una comprobación de estabilidad citable.
- **Consecuencia práctica:** 1.024 bytes por archivo bastan. Leer 8 veces más no aporta y cuesta
  8 veces más — argumento de costo computacional utilizable en la tesis.
- La leve caída con ventanas grandes es coherente con la maldición de la dimensionalidad ya
  observada en las 275 características estadísticas (0,592 contra 0,603 con 19).

### (b) Los bytes del medio no llevan casi información — contestada la otra pregunta

| Configuración | Bytes | Exactitud | macro-F1 |
|---|---|---|---|
| Solo medio | 1024 | **0,056** | 0,058 |
| Solo cabecera | 512 | 0,338 | 0,348 |
| Solo cola | 512 | **0,756** | 0,746 |
| Cabecera + cola | 1024 | **0,904** | 0,905 |
| Cabecera + cola + medio | 2048 | 0,902 | 0,903 |

- **El medio da 0,056 con azar en 0,033**: apenas por encima del azar, y agregarlo a los
  extremos no mejora nada (0,902 contra 0,904). Es la respuesta empírica a «¿por qué no se
  revisan los bytes del medio?»: porque ahí el cifrado sí es indistinguible. Refuerza la
  reformulación del Objetivo 2 en vez de contradecirla.
- **La cola vale más del doble que la cabecera** (0,756 contra 0,338) y **converge con el
  Exp. 2b**, que encontró 11 sufijos y solo 4 prefijos. Dos métodos independientes vuelven a
  decir lo mismo: la marca la escribe el ransomware al final, después de cifrar.
- Ninguno de los dos extremos por separado se acerca a la combinación (0,904): son
  complementarios, no redundantes.

## ✅ EXP. 2c RE-CORRIDO SOBRE 30 FAMILIAS (job 3633, 2026-08-16)

`clasificador_bytes.py` sobre **30 familias × 500 archivos = 15.000** (antes 29 × 500 = 14.500).
**El resultado principal de la tesis no se mueve al sumar BLACKBASTA:**

| | 29 familias (job 3557) | **30 familias (job 3633)** |
|---|---|---|
| Exactitud | 0,910 | **0,9105** |
| Balanced accuracy | — | **0,9105** |
| macro-F1 | 0,908 | **0,9097** |

Mismos hiperparámetros elegidos por la búsqueda anidada: `n_estimators=300, max_depth=20,
min_samples_leaf=2, max_features=0.3`. 568 s de ajuste final. Sigue **sin usar nombre ni
extensión**.

### Comparación de representaciones (etapa de búsqueda, 6.000 archivos)

| Configuración | Exactitud | macro-F1 |
|---|---|---|
| **Posicional + RandomForest** | **0,8990** | **0,8986** |
| N-gramas de bytes + LogReg | 0,8602 | 0,8639 |
| N-gramas de bytes + LinearSVC | 0,8498 | 0,8472 |
| **FINAL: posicional + RF, 15.000 archivos** | **0,9105** | **0,9097** |

Fuente: `resultados_bytes/bytes_resumen.csv`. **La representación posicional vuelve a ganar**
sobre los n-gramas por ~0,04, igual que con 29 familias ⇒ las marcas están en offsets fijos.
Contraste metodológico con las notas, donde ganan los n-gramas de caracteres.

### Reporte por familia — las seis difíciles son EXACTAMENTE las mismas

Fuente: `resultados_bytes/bytes_por_familia.txt`. Promedios macro: precisión 0,93 · recall 0,91.

**24 familias con F1 ≥ 0,98** (antes eran 23 sobre 29): AVOSLOCKER, BADRABBIT 0,98, BLACKBASTA,
BLACKCAT, BLACKMATTER 0,99, CERBER, CHIMERA, CLOP, CONTI, CUBA, DHARMA, GANDCRAB,
HELLOKITTY 0,99, LOCKBIT, LORENZ 0,99, MAZE, MEDUZALOCKER, NETWALKER, PHOBOS, RANSOMEXX, RYUK,
SODINOKIBI, TESLACRYPT, WANNACRY (las no anotadas dan 1,00).
**BLACKBASTA entra con F1 = 1,00** (precisión 1,00 / recall 0,99) pese a no tener firma binaria
en el Exp. 2b: solo aportaba la extensión `.basta`, que este clasificador no usa.

| Familia difícil | Precisión | Recall | F1 (30 fam.) | F1 antes (29 fam.) |
|---|---|---|---|---|
| SUNCRYPT | 0,77 | 0,69 | 0,73 | 0,75 |
| WASTEDLOCKER | 0,74 | 0,56 | 0,64 | 0,64 |
| CRYPTOLOCKER | 0,84 | 0,47 | 0,60 | 0,61 |
| DARKSIDE | 0,45 | 0,84 | 0,59 | 0,60 |
| JIGSAW | 0,37 | 0,58 | 0,45 | 0,44 |
| NOTPETYA | 0,77 | 0,24 | 0,36 | 0,38 |

**Las seis son las mismas y las cifras se mueven ±0,02.** Sumar una familia entera no cambia el
cuadro: es la mejor evidencia de que el grupo de confusión es una propiedad de esas familias y
no del muestreo. Escribirlo así.

**El mecanismo del grupo de confusión se ve en precisión/recall, y hay dos roles:**
- **Imanes** (precisión baja, recall alto): DARKSIDE 0,45/0,84 y JIGSAW 0,37/0,58 absorben los
  archivos ambiguos de las otras.
- **Absorbidas** (precisión alta, recall bajo): NOTPETYA 0,77/0,24, CRYPTOLOCKER 0,84/0,47 y
  WASTEDLOCKER 0,74/0,56 — cuando el modelo dice «NOTPETYA» casi siempre acierta, pero
  reconoce apenas una cuarta parte de sus archivos.

Esto es más informativo que el F1 solo y **hay que reportarlo con las tres cifras**: el error no
es ruido difuso, es un intercambio dentro de un grupo cerrado de familias que cifran sin dejar
estructura. Es el mismo grupo que el Exp. 2b marca sin firma (BADRABBIT es la excepción:
sin marca en 2b pero F1 0,98 acá).

**Los dos frentes ya están sobre 30 familias.** Desaparece la asimetría 30/29 que había que
explicar en cada tabla del capítulo 4; el azar del frente de archivos pasa a 0,033.

### Hallazgo lateral verificado 2026-08-16 — dos familias renombran el archivo entero

Los 505 nombres no canónicos del log de la ablación (499 `desconocido` + 6 con basura tipo
`081baaun`) **no son de BLACKBASTA**, que sí sigue la convención (`0001-doc.doc.basta`). Conteo
de nombres que no matchean `^\d+-[a-z0-9]` por carpeta:

| Familia | Archivos con nombre no canónico | Ejemplos |
|---|---|---|
| **CERBER** | **981** | `002PWX5w8Z.bed4`, `-00CAjTujp.bed4` |
| **BLACKMATTER** | **13** | `00n0P97.HpWl7Oyll`, `01EPRnN.HpWl7Oyll` |
| WASTEDLOCKER / WANNACRY / TESLACRYPT | 1 cada una | — |

**CERBER y BLACKMATTER reemplazan el nombre base por una cadena aleatoria**, no solo agregan
extensión. Es comportamiento del ransomware, no un defecto del dataset, y es **dato utilizable
en la tesis**: refuerza por qué la comparación honesta con ID Ransomware es la del nombre
cambiado (9 de 30 familias) y toca la decisión D.2 del plan (nombre de archivo como
característica) — para estas familias el nombre original directamente no existe.

### ⚠️ CONSTATACIÓN: 12 archivos sin cifrar dentro de CERBER-small

Verificado con volcado hexadecimal el 2026-08-16. `CERBER-small` tiene **988 `.bed4` + 12 `.jpg`
+ 1 `.pdf`**. Los 12 `.jpg` conservan nombre y extensión originales (`0045-jpg-fromweb.jpg`) y
los tres inspeccionados **empiezan con `ffd8ffe0 0010 4a46 4946` = cabecera JPEG/JFIF**: son
**imágenes en claro, sin cifrar** (falta pasar el volcado por los 12 antes de moverlos).
(Los 26 nombres «canónicos» contados antes son 12 `.jpg` sin cifrar + 14 `.bed4` a los que
CERBER cifró sin renombrar la base.)

- **No atribuir la causa.** «Error de etiquetado» es una lectura; la otra es que CERBER no los
  cifró (varias familias saltan archivos por tamaño o ubicación). Con estos datos no se puede
  decidir. En la tesis va como constatación: 12 archivos sin cifrar, verificado por magic bytes.
- **Magnitud:** 12 sobre 1.001 archivos de CERBER (1,2 %); en la muestra de 500 por familia caen
  ~6, o sea 0,04 % del corpus de 15.000.
- **Pero no es solo cosmético:** son archivos en claro etiquetados como CERBER, así que el
  clasificador de bytes puede aprender «cabecera JPEG válida ⇒ CERBER». CERBER da precisión
  1,00 / recall 0,99, compatible con eso.
- **Conecta con el pliegue de jpg** del análisis de generalización: CERBER es una de las 28
  familias presentes ahí **precisamente por estos archivos**, y ese pliegue tiene el macro-F1
  más bajo de los siete (0,811).
- **DECISIÓN (chat padre, 2026-08-16): excluirlos y re-correr el Exp. 2c**, para poder escribir
  que se verificó que el 0,9105 no se mueve. Exclusión: mover los 12 a un subdirectorio
  `_sin_cifrar/` dentro de `CERBER-small` (los tres scripts filtran con `is_file()`, así que un
  subdirectorio queda fuera automáticamente; reversible y visible).

### INVENTARIO DE EXTENSIONES POR FAMILIA (2026-08-16) — explica el frente entero

Conteo de extensiones finales en cada carpeta de `Napierone-small`. **Es el material que
faltaba para explicar *por qué* unas familias dejan marca y otras no**, en vez de solo
constatarlo. Cuatro comportamientos distintos:

| Comportamiento | Familias | Extensiones observadas |
|---|---|---|
| **Extensión fija** (26) | AVOSLOCKER, BLACKBASTA, BLACKCAT, BLACKMATTER, CERBER, CHIMERA, CLOP, CONTI, CRYPTOLOCKER, CUBA, DARKSIDE, DHARMA, GANDCRAB, HELLOKITTY, LOCKBIT, LORENZ, MEDUZALOCKER, NETWALKER, PHOBOS, RANSOMEXX, RYUK, SODINOKIBI, TESLACRYPT, WANNACRY, WASTEDLOCKER, **JIGSAW** | 1.000 archivos con la misma |
| **No cambia la extensión** | **BADRABBIT** (144 pdf, 143 xls, 143 pptx…), **NOTPETYA** (168 pdf, 167 pptx, 167 docx…) | las originales |
| **Extensión aleatoria por lote** | **MAZE** | `.TPjsq`, `.jaUH`, `.bJ3jUCt`… 5 archivos cada una |
| **Extensión aleatoria por archivo** | **SUNCRYPT** | cadenas hexadecimales de 64 caracteres, únicas |

**Esto cierra el círculo con el Exp. 2b y con el Exp. 2c:** las familias sin marca de extensión
son exactamente las que no la cambian (BADRABBIT, NOTPETYA) o la aleatorizan (MAZE, SUNCRYPT), y
tres de ellas están entre las seis difíciles del Exp. 2c. MAZE se salva porque sí deja sufijo
binario. **Es explicación mecánica, no correlación** — va al capítulo 4.

**Además: cada familia tiene exactamente 1 archivo `.pdf`**, casi con seguridad documentación de
NapierOne y no una muestra cifrada. `clasificador_bytes.py:101` y la ablación **ya lo excluyen**
(`p.suffix.lower() != ".pdf"`); `deteccion_estructural.py` **no**.
⚠️ Efecto colateral de esa exclusión: a BADRABBIT y NOTPETYA, que conservan las extensiones
originales, se les descartan también sus ~144 y ~168 archivos **realmente cifrados** con
extensión `.pdf`. No invalida nada (se muestrean 500 de ~850), pero hay que saberlo.

### ⚠ Discrepancia a resolver: JIGSAW SÍ tiene extensión fija (`.fun`)

El Exp. 2b lo reporta entre las **cuatro sin marca estructural**, pero el inventario muestra
**990 archivos `.fun`** (+ 7 `.pptx` + 3 `.pdf`). La causa probable está en el código, no en los
datos: `deteccion_estructural.py:87` exige que la extensión sea común a **todos** los archivos
de la muestra (`c[1] == len(exts)`), y la muestra son los **primeros 50 en orden alfabético**
sin barajar ni excluir `.pdf` (`deteccion_estructural.py:98`). Un solo archivo distinto entre
esos 50 anula la extensión de toda la familia.

**CONFIRMADO 2026-08-16.** Los primeros 50 archivos de JIGSAW son **47 `.fun` + 1 `.doc` +
1 `.pdf` + 1 `.pptx`**: tres archivos rompen la unanimidad y el detector descarta `.fun` para
toda la familia. **No es que JIGSAW no deje marca: es que la regla es demasiado estricta.**

Con la extensión detectada, la cifra correcta pasa a **27 de 30 familias con marca**, y las que
realmente no dejan ninguna quedan en **BADRABBIT, NOTPETYA y SUNCRYPT** — exactamente las tres
que el inventario predice (dos no renombran, una aleatoriza por archivo). El cuadro se vuelve
coherente de punta a punta.

**Dato que además muestra lo frágil que es el muestreo:** los primeros 50 de CERBER son
38 `.bed4` + 12 `.jpg`, o sea que ahí la unanimidad **también** debería fallar — y sin embargo
el Exp. 2b sí le asignó `.bed4`. La explicación es que `ls` y el `sorted()` de Python no ordenan
igual: Python compara por código de carácter y los nombres que empiezan con `-`
(`-00CAjTujp.bed4`) le caen primero, de modo que su muestra de 50 no es la misma que la de `ls`.
**Confirmar con `LC_ALL=C sort`, que sí reproduce el orden de Python.**

**DECISIÓN TOMADA (chat padre, 2026-08-16): opción 1, con parche ya hecho y probado.**
`deteccion_estructural.py` corregido:
1. **Excluye el `.pdf`** de cada carpeta — el argumento principal no es JIGSAW sino la
   consistencia: `clasificador_bytes.py` y `ablacion_ventana_extendida.py` ya lo filtraban y el
   estructural no. Inconsistencia entre scripts propios, no cambio de criterio.
2. **Umbral de mayoría** en la extensión, expuesto como `--umbral` (default 0,90) y guardado en
   el manifiesto. No opcional: con 990/1000 `.fun`, la probabilidad de que 50 archivos al azar
   sean todos `.fun` es 0,99⁵⁰ ≈ 0,6 — ni muestreando al azar la unanimidad es estable,
   dependería de la semilla.
3. **Muestreo aleatorio con semilla fija** (`--semilla`, default 42) en vez de los primeros 50
   alfabéticos. Corrige el sesgo de CERBER: Python ordena por código de carácter y `-` (0x2D)
   va antes que los dígitos, así que la muestra real eran archivos que empiezan con `-`.
4. **Reporta LAS DOS cifras en una sola corrida** — unanimidad y umbral 0,90 sobre la misma
   muestra, con CSV separados por criterio (`marcas_por_familia_<criterio>.csv`,
   `clasificacion_loo_<criterio>.csv`). Si solo apareciera el número nuevo, se leería como que
   se eligió el criterio que daba mejor. El umbral se declara en el capítulo.

**Smoke test local (2026-08-16) con dataset sintético de 3 familias** (una con prefijo+extensión,
una con el caso JIGSAW 18 `.fun`+1 `.pptx`+1 `.pdf`, una sin marca): la unanimidad pierde
`.fun`, el umbral 0,90 lo recupera, el `.pdf` queda excluido, las dos cifras salen juntas.
Funciona. `job_estructural.sh` actualizado con `--nodelist=c2`.
**✅ RESUELTO — corrido como job 3638 el 2026-08-16; resultados en la sección
«EXP. 2b CORREGIDO» más arriba.** Cifras nuevas: 0,867 / 0,533 / **0,933** (cobertura 0,941),
28/30 con marca, y firma nueva de BADRABBIT (sufijo UTF-16LE «encrypted»).

### ⚠️ Límite del análisis de «generalización a tipos de archivo nunca vistos»

Consecuencia directa de lo anterior, sobre un resultado **ya escrito** en el plan como
«0,879 frente a 0,910, caída de 0,031». El CSV `resultados_analisis_bytes/a_generalizacion_tipos.csv`:

| Tipo excluido | n prueba | **n familias** | Exactitud | macro-F1 |
|---|---|---|---|---|
| doc | 1.991 | 27 | 0,8890 | 0,8812 |
| docx | 1.978 | 27 | 0,8868 | 0,8774 |
| jpg | 2.397 | **28** | 0,8736 | 0,8110 |
| pdf | 1.798 | **25** | 0,8626 | 0,8361 |
| pptx | 1.926 | 27 | 0,8666 | 0,8644 |
| xls | 1.948 | 27 | 0,8891 | 0,8825 |
| xlsx | 1.962 | 27 | 0,8838 | 0,8724 |

**Ningún pliegue cubre las 29 familias: cubren entre 25 y 28.** La razón es la misma — las
familias que renombran el archivo no tienen tipo de documento reconocible, así que **CERBER
queda fuera de casi todos los pliegues** (aparece solo en el de jpg, el único con 28). El 0,879
es el promedio de esa columna y **la conclusión se sostiene** (la caída sigue siendo chica),
pero hay que enunciarlo con la cobertura real: *«entre 25 y 28 de las 29 familias según el
tipo excluido»*, no «las 29». Es exactamente el tipo de detalle que el tutor puede preguntar.
Del extracto por familia visible en el log: AVOSLOCKER 1,00 · BADRABBIT 0,98 · **BLACKBASTA
1,00** · BLACKCAT 1,00 · BLACKMATTER 0,99 · CERBER 1,00. Falta el reporte completo (el CSV) para
confirmar si las **seis difíciles** siguen siendo las mismas — dato necesario antes de reescribir
§4.5.3.

## ★ EXP. 2b CORREGIDO (job 3638, 2026-08-16) — REEMPLAZA al job 3632

Corrida con el detector parcheado (sin `.pdf`, muestreo aleatorio semilla 42, dos criterios de
extensión sobre la misma muestra de 1.500 archivos / 30 familias). **Las cifras del job 3632,
ya escritas en el cap. 4, quedan superadas.** 53 segundos.

| Modo | Job 3632 (viejo) | **Job 3638 (corregido)** | Cobertura nueva |
|---|---|---|---|
| Solo extensión | 0,833 | **0,867** (1300/1500) | 0,867 |
| Solo firmas binarias | 0,500 | **0,533** (800/1500) | 0,541 |
| Combinado | 0,866 | **0,933** (1400/1500) | **0,941** |

**28 de 30 familias con marca** (antes 26). Sin marca quedan solo **NOTPETYA y SUNCRYPT**.
Firmas binarias: **16** (4 prefijos: CUBA, LORENZ, TESLACRYPT, WANNACRY; **12 sufijos**, antes 11).

### Hallazgo nuevo: BADRABBIT SÍ deja firma — y resuelve una anomalía registrada
> ⚠️ **MATIZADO el 2026-08-17: la firma NO es universal.** Con la semilla 1 el detector deja a
> BADRABBIT **sin marca**. La firma es reproducible con la semilla 42 pero no está en todos sus
> archivos, así que hay que escribirla como «detectada en la muestra de la semilla 42», nunca
> como «BADRABBIT deja firma». Detalle y consecuencias en el bloque de los dos defectos, al
> principio de este documento.

El detector corregido encuentra en BADRABBIT un **sufijo de 18 bytes**:
`65006e006300720079007000740065006400` = **«encrypted» en UTF-16LE — CONFIRMADO byte por byte**
el 2026-08-16 contra `marcas_por_familia_umbral_90.csv` (fila: prefijo 0, sufijo 18 B, sin
extensión común, marca detectable). Es del mismo tipo que el «WANACRY!» de WannaCry: una
cadena legible que el ransomware escribe deliberadamente.
**Pendiente solo la fuente citable** sobre el marcador de BadRabbit para el related work
(el hallazgo propio ya es reportable por sí mismo; una cita externa lo convertiría en la
séptima corroboración independiente, junto a las seis de `Pruebas.xlsx`).

Antes no aparecía porque el `.pdf` de documentación integraba la muestra y rompía el sufijo
común. **Esto explica la excepción anotada en el Exp. 2c** («BADRABBIT: sin marca en 2b pero
F1 0,98») — no era que el ML viera algo invisible: la marca existía y el detector viejo no la
encontraba. La convergencia entre 2b y 2c queda ahora limpia:

| Familia | Marca en 2b corregido | F1 en 2c |
|---|---|---|
| NOTPETYA | ninguna | 0,36 |
| SUNCRYPT | ninguna | 0,73 |
| JIGSAW | solo extensión (2c no la usa) | 0,44 |
| CRYPTOLOCKER / DARKSIDE / WASTEDLOCKER | solo extensión (2c no la usa) | 0,60 / 0,59 / 0,64 |
| BADRABBIT | **sufijo «encrypted»** | **0,98** |

Las seis difíciles del 2c son exactamente las familias sin firma *en el contenido*; las que solo
tienen extensión siguen difíciles para el 2c porque el 2c no usa la extensión. Sin excepciones.

### Los dos criterios dieron IDÉNTICO — pero el umbral sigue siendo necesario

Unanimidad y umbral 0,90 produjeron el mismo resultado en esta muestra: con semilla 42, los 50
archivos muestreados de JIGSAW salieron todos `.fun`. **Es suerte del sorteo**: la probabilidad
es **0,6968** (hipergeométrica exacta sobre 990 `.fun` de 997 archivos sin `.pdf`), o sea que con
otra semilla la unanimidad falla ~3 de cada 10 veces.
El criterio que se declara en el capítulo es **umbral 0,90**; que la unanimidad coincida acá se
reporta como evidencia de que el umbral no se eligió por conveniencia (los CSV de ambos
criterios quedan guardados).

> ✅ **COMPROBADO EMPÍRICAMENTE el 2026-08-17 (10 semillas):** bajo unanimidad, JIGSAW pierde la
> extensión `.fun` en la semilla 1 y la conserva en las otras nueve; bajo umbral 0,90 la conserva
> en las diez. **El criterio de unanimidad parpadea y el umbral no.** Detalle, cifras y caveat
> sobre la tasa observada (1/10 frente a 0,30 esperado) en el bloque «DOS DEFECTOS HALLADOS AL
> AUDITAR LOS CSV» al principio de este documento.

**Caveat del filtro `.pdf` a declarar:** también excluye los `.pdf` *genuinamente cifrados* de
BADRABBIT y NOTPETYA (conservan la extensión original). Sus muestras salen de los demás tipos;
no afecta las conclusiones, pero decirlo.

## ✅ EXP. 2b RE-CORRIDO SOBRE 30 FAMILIAS (job 3632, 2026-08-15) — ⚠ SUPERADO por el job 3638

BLACKBASTA ya está en el cluster. `deteccion_estructural.py` sobre **1.500 archivos, 30
familias, 50 por familia**. **Ninguna conclusión cambia** — es el resultado que se buscaba.

| Modo | Antes (29 fam.) | Ahora (30 fam.) | Cobertura | Acierto donde hay marca |
|---|---|---|---|---|
| Solo extensión | 0,828 | **0,833** | 0,833 | **100,0 %** (1250/1250) |
| Solo firmas binarias | 0,517 | **0,500** | 0,516 | 96,9 % (750/774) |
| Combinado | 0,862 | **0,866** | 0,882 | 98,2 % (1299/1323) |

- **26 de 30** familias dejan marca detectable (antes 25 de 29).
- **BLACKBASTA aporta extensión `.basta` pero ninguna firma binaria.** Por eso el conteo de
  firmas se mantiene en **15** (4 prefijos: CUBA, LORENZ, TESLACRYPT, WANNACRY; 11 sufijos).
- **Las cuatro sin marca son las mismas de siempre:** BADRABBIT, JIGSAW, NOTPETYA, SUNCRYPT.

### Corroboración independiente con `Pruebas.xlsx` (fuerte, escribirlo en el cap. 4)
Cuatro firmas halladas por el script coinciden **exactamente** con los `sample_bytes` que
ID Ransomware reporta por su cuenta, sin que ninguno de los dos supiera del otro:

| Familia | Nuestro detector | ID Ransomware (`Pruebas.xlsx`) |
|---|---|---|
| WANNACRY | prefijo `57414e4143525921` | `[0x00-0x08] 0x57414E4143525921` («WANACRY!») |
| LORENZ | prefijo `2e737a3430` | `[0x00-0x05] 0x2E737A3430` («.sz40») |
| MAZE | sufijo `0000000066116166` | `[0x58771-0x58779] 0x0000000066116166` |
| GANDCRAB | sufijo `1829899381820300` | `[0x43614-0x4361C] 0x1829899381820300` |

Más CUBA `464944454c2e4341` = «FIDEL.CA» y PHOBOS `4c4f434b3936` = «LOCK96». Dos métodos
independientes llegan a las mismas marcas: es validación externa del Experimento 2b.

### ⚠ Pista abierta: RYUK
`Pruebas.xlsx` registra para RYUK `[0x584D0-0x58792] 0x4845524D4553` = **«HERMES»** (Ryuk
deriva de Hermes). Nuestro detector **no la encuentra**: RYUK aparece solo con extensión
`.ryk`. Hipótesis a verificar —no afirmar sin comprobar— que el marcador no está a distancia
fija del final porque después va un blob de clave de longitud variable, y el detector solo
mira 128 bytes de cada extremo. Se comprueba con un volcado hexadecimal, y conecta con la
pregunta del tutor sobre los bytes del medio.

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

## FUENTE RECUPERADA 2026-08-15 — `Pruebas.xlsx`: comparación con herramientas públicas

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

**Procedencia verificada 2026-08-15** leyendo los notebooks
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
