# Lista de recolección por familia — seguimiento

_Última actualización: 2026-08-19._ Objetivo de B.1: **4 textos distintos por familia.**
Se trabaja **una familia hasta cerrarla**, no salteando. Cada nota nueva se verifica con
`2_codigo/verificar_nota_nueva.py` antes de contarla (umbral 0,90).

Corpus: **151 notas** (manifiesto 151 filas; coinciden 1:1). DHARMA `Info__3/__13` restaurados del git local el 2026-08-20.
Aterrizaje: `3_datos/recoleccion_2026-08/<FAMILIA>/`.
**Total recolectado y verificado: 8 textos nuevos.**

> **Cambios 2026-08-19 (tarde), detalle en `ESTADO_TESIS.md`:** (a) CHIMERA +1 (nota alemana por OCR,
> ver abajo). (b) Se **movieron** 3 notas mal etiquetadas a `3_datos/descartados_integridad/`:
> 2 de Medusa que estaban en MEDUZALOCKER y la de TorrentLocker en CRYPTOLOCKER (MEDUZALOCKER 6→4,
> CRYPTOLOCKER 4→3). (c) DHARMA: las 2 `.hta` perdidas se identificaron como las variantes abibo/cmb
> del repo Lemmou (cuarentena en corpus y origen); **restauradas del git local el 2026-08-20** (bytes
> originales, MD5 verificado) → corpus a 150, cierra el gap 144-vs-146.

## 🔒 REGLAS DE FUENTES — leer antes de consultar cualquier URL

**Incidente que motiva esta sección (2026-08-19):** `malwiki.org`, que apareció en una búsqueda
como fuente de notas de Ryuk, **responde 301 y redirige a `mufasatotoamanah.com`**, un dominio
sin relación con seguridad informática. Parece dominio expirado y recomprado. **No se siguió el
redirect. NO consultar malwiki.org.**

**Reglas que se aplican en la recolección:**

1. **Solo se lee texto.** No se descargan archivos: ni muestras, ni binarios, ni `.zip`, ni
   imágenes. (Por eso el OCR está pendiente de que Romina habilite el paso de descarga.)
2. **Los redirects a otro dominio no se siguen automáticamente.** Si una URL redirige fuera del
   dominio consultado, se corta y se anota.
3. **Nada de repositorios de muestras vivas.** Se puede leer el listado de archivos de un repo,
   nunca bajar el contenido ejecutable. Ya se descartaron por esto: `LeechxSys/Jigsawsource`
   (un `.rar` de 4,7 MB) y `kh4sh3i/Ransomware-Samples` (zips de binarios).
4. **Nada de hosts de archivos anónimos.** Se descartó
   `transfer.sh/…/WANNACRYDECRYPTOR-Ransomware-Messages-all-langs.zip` (las 28 notas de
   WannaCry) aunque era justo lo que se buscaba.
5. **Preferir el dominio del vendor sobre agregadores.** Los sitios que reempaquetan contenido
   de seguridad («decryptor», «recovery», «ransomwarehelp» y similares) no se usan como fuente:
   no son citables y varios son comerciales con contenido copiado.

### Cómo se decide si una fuente sirve — 6 chequeos, en este orden

Se hacen **antes** de abrir la fuente, y el 1 y el 3 son los que atrapan los casos peligrosos:

1. **¿La recomienda un tercero confiable?** Es el chequeo más fuerte y no requiere tocar la
   fuente. Ejemplo real: `id-ransomware.blogspot.com` **está listada por SANS** en «Recommended
   Sources for Ransomware Information» (Katie Nickels y Ryan Chapman, 02-07-2021), en la sección
   *General Sources*. Eso la habilita.
2. **¿La autoría es verificable?** Nombre real con historial, no anónima. (Amigo-A = Andrew
   Ivanov; pcrisk = Tomas Meskauskas; el análisis de Chimera = hasherezade.)
3. **¿El dominio puede caducar y ser recomprado?** Este fue el modo de falla de `malwiki.org`:
   dominio propio, caducó, lo compró otro. **Un subdominio de una plataforma grande
   (`*.blogspot.com`, de Google) no se puede comprar por separado**, así que no tiene ese
   riesgo — el riesgo ahí es distinto y menor: que el dueño del blog cambie el contenido.
4. **¿Redirige fuera de su dominio?** Si redirige, se corta y se anota.
5. **¿Es el vendor original o un agregador?** Los sitios de «recovery», «decryptor» o
   «ransomware help» copian contenido, son comerciales y no son citables.
6. **¿El nombre coincide con otra cosa?** La trampa del homónimo, que en este proyecto ya pegó
   dos veces. Ojo con esta en particular: **`id-ransomware.blogspot.com` (el blog de Amigo-A)
   NO es lo mismo que ID Ransomware, el servicio de identificación de MalwareHunterTeam** con
   el que se compara en `Pruebas.xlsx` (el 71,93 % = 41/57 notas de 22 familias). SANS las lista
   como dos fuentes separadas. **No confundirlas al citar.**

**Lista blanca usada hasta ahora (todas verificadas, todas dieron texto o captura legítima):**
`pcrisk.com` · `bleepingcomputer.com` · `malwarebytes.com` · `sophos.com` · `sonicwall.com` ·
`helpnetsecurity.com` · `cisecurity.org` · `cisa.gov` · `ic3.gov` (PDF primario del FBI) ·
`api.github.com` y `raw.githubusercontent.com` (solo listados y texto) · `trendmicro.com`
(responde 403) · `id-ransomware.blogspot.com` (**pendiente de usar**, señalada como confiable
por su autoría —Amigo-A / Andrew Ivanov— pero **todavía no consultada**).

Y los dos repos que ya están **en disco**, que son la fuente más segura porque no requieren red:
`3_datos/fuentes_notas/ransomware_notes` (ThreatLabz) y `.../RansomNoteFiles` (Lemmou).

## Tablero

| # | Familia | Textos | Faltan | Estado |
|---|---|---|---|---|
| 1 | MEDUZALOCKER | 4 citables (5 nominales) | 0 | ✅ **CERRADA** (+2) |
| 2 | JIGSAW | 4 | 0 | ✅ **CERRADA** (+2) |
| 3 | MAZE | 4 | 0 | ✅ **CERRADA** (+1 nota ChaCha «0010 SYSTEM FAILURE», id-ransomware/Amigo-A) |
| 4 | CHIMERA | 3 citables | 0 | ✅ **cerrada por OCR** (nota alemana, +1) — texto agotado, techo 3 |
| 5 | RYUK | 2 | 2 | 🔻 **texto agotado**, queda OCR |
| 6 | NOTPETYA | 2 | 2 | ⬜ sin empezar, solo OCR |
| 7 | WANNACRY | 2 | 2 | 🔻 **agotada** en fuentes citables |
| 8 | CRYPTOLOCKER | 3 (1 dudosa) | 1 | ⛔ **bloqueada**: no existe archivo de nota |
| 9 | WASTEDLOCKER | 1 | 3 | ⛔ **bloqueada**: molde rígido |

## ✅ OCR: se hace leyendo la imagen directamente, sin tesseract

**Corrección:** antes se anotó que el OCR estaba bloqueado. No lo está — las imágenes se leen y
se transcriben directamente, con mejor precisión que tesseract y sin instalar nada. Hecho con
las 2 imágenes relevantes que ya estaban en `3_datos/fuentes_notas/imagenes_notas/`.

**Resultado: 0 textos nuevos, pero validó el corpus.**

| Imagen | Familia | Resultado |
|---|---|---|
| `mbr-ransom-note.jpg` | NOTPETYA | COPIA de `notpetya_note1.txt` (coseno **0,925**) → **la confirma como auténtica**; y revela que `notpetya_note2.txt` tiene un párrafo que no está en la imagen |
| `Wana_Decrypt0r_screenshot.png` | WANNACRY | corrobora `wannacry_note1.txt`; **se ve el desplegable de idioma**, evidencia visual de los 28 `m_*.wnry`. No transcripta: el panel tiene scroll y el texto está cortado |
| `oops_note.png` | BADRABBIT | no se usó: BADRABBIT no está en la lista (F1 por familia 0,983) |

**Tesseract sigue haciendo falta solo para el pipeline automático** (`extractor_notas.py` hace
OCR de imágenes y PDF con pytesseract, líneas 91-99 y 118-123). Sin él, una imagen puesta en
`corpus_v2` **no rompe: se omite con ADVERTENCIA y la nota no cuenta.** Si se quiere que el
clasificador procese imágenes solo:

```bash
winget install --id UB-Mannheim.TesseractOCR
```

## 🛑 INCIDENTE: DEFENDER PUSO EN CUARENTENA LAS TRANSCRIPCIONES DE CHIMERA (2026-08-19)

**Qué pasó.** Se transcribieron las 2 capturas de Chimera (alemana de Help Net Security,
inglesa de Malwarebytes) a `.txt`. **Windows Defender las detectó y las eliminó** antes de que
el verificador pudiera leerlas:

- Detección: **`Ransom:HTML/Chicrypt.A`, severidad 5** («Chicrypt» = Chimera).
- Archivos eliminados: `ocr_chimera_aleman_btc.txt` (1081 B) y `ocr_chimera_ingles_btc.txt`
  (683 B), a las 17:31 del 19-08-2026.
- Python devolvía `OSError [Errno 22]` al abrirlos, que es cómo se ve el bloqueo de Defender.

**✅ El corpus quedó INTACTO: 150 de 150 archivos legibles, 0 bloqueados.** Las 3 imágenes
descargadas también sobrevivieron (Defender no marca las imágenes, solo el texto).

**No se intentó ningún rodeo.** Nada de renombrar, partir el archivo ni cambiar la
codificación para que la firma no coincida: eso es evasión de detección y no se hace. **La
exclusión la configura Romina** (ya figuraba como pendiente en el plan de recolección):

```
Exclusión de carpeta a agregar en Seguridad de Windows: C:\Users\Romina\Tesis\3_datos
```

**Precedente, para que quede claro que no es nuevo:** el 06-08-2026 Defender ya se había llevado
notas de TESLACRYPT (`Ransom:HTML/Tescrypt.D`) desde
`2_codigo/family-rw-detection/ransom_notes_corpus/`. ✅ **Verificado que esos 181 archivos NO
están en el repositorio** (`git ls-files` da 0; `.gitignore` cubre `ransom_notes_corpus/`).

### ▶ Y de paso, un hallazgo que sirve para la tesis

**Defender trae una firma propia para el TEXTO de la nota de Chimera**, y una transcripción en
texto plano —sin nada ejecutable— alcanza para dispararla. Es **corroboración independiente, de
un vendor de antivirus, de que el texto de la nota por sí solo identifica a la familia**, que es
justamente la premisa del frente de notas. Vale citarlo.

### ⚠️ El OCR no puede pasar por el chat

Decisión de Romina (19-08-2026): **no mostrar las imágenes en el chat.** Leer una imagen la
renderiza en la conversación, así que la transcripción por visión directa queda descartada como
método. Las dos vías que quedan:

1. **tesseract** (`winget install --id UB-Mannheim.TesseractOCR` + `pip install pytesseract`),
   que además habilita el OCR automático de `extractor_notas.py`.
2. **Un subagente** que lea la imagen en su propio contexto y devuelva solo el texto.

### ⚠️ Sobre el tipo de archivo original de una nota transcripta

Pregunta válida: si la fuente es una captura, ¿cómo se sabe la extensión original? **No sale del
artefacto, sale de la documentación**, y hay que declararlo así. Para **CHIMERA se sabe, y por
tres vías independientes que coinciden:**

1. `notas_familias_criticas.md` y pcrisk registran el nombre de archivo
   **`YOUR_FILES_ARE_ENCRYPTED.HTML`**
2. hasherezade (Malwarebytes, 09-12-2015): «there is an **HTML file** dropped»
3. **El propio nombre de la firma de Defender: `Ransom:HTML/Chicrypt.A`**

Por eso el manifiesto registra `extension_original = .html` para esa familia. **Regla general:
en notas transcriptas la columna `extension_original` es un dato DOCUMENTADO, no observado, y
así hay que decirlo en la tesis.** El archivo en disco se guarda como `.txt` porque es texto:
ponerle `.html` sin marcado sería fabricar el formato.

## ⛔ 24 % DEL CORPUS NO TIENE FUENTE RASTREABLE

Del manifiesto (152 filas): `bruto` 96 · `transcripcion` 19 · **`corpus-existente` 37**, y estas
últimas dicen solo «NapierOne/varios», sin URL. Se concentran en las familias problemáticas:
CRYPTOLOCKER 3 · WASTEDLOCKER 3 · MEDUZALOCKER 3 · LOCKBIT 4 · NOTPETYA, WANNACRY, CHIMERA,
RYUK, JIGSAW, BADRABBIT, PHOBOS, AVOSLOCKER, HELLOKITTY 2 c/u.

**El método de validación ya está probado** (OCR de imagen independiente + umbral 0,90) y
`notpetya_note1.txt` ya quedó confirmada así. **Prioridad: las 11 notas de WASTEDLOCKER,
CHIMERA, MEDUZALOCKER y CRYPTOLOCKER**, que son las familias que se reportan con F1 por familia
0,000 o con problema de homonimia.

### Sobre replicar la extensión original

Se registra en la columna `extension_original` del manifiesto y **no** se falsifica el formato:
una transcripción del texto renderizado se guarda como `.txt` aunque la nota original fuera
`.html`, porque inventar el marcado sería fabricar el artefacto. Cuando el original es `.txt`
(caso `DECRYPT-FILES.txt` de Maze) la extensión coincide exacto. Extensiones que Windows
Defender pone en cuarentena (`.hta`) no se crean nunca.

---

## 1. ✅ MEDUZALOCKER — CERRADA

- [x] `pcrisk_medusalocker_chip.txt` — pcrisk guía 34945 · Meskauskas · 12-05-2026 · coseno 0,781
- [x] `pcrisk_medusalocker_rapid.txt` — pcrisk guía 28682 · Meskauskas · 28-03-2024 · coseno 0,856

**Fuentes ya revisadas (no repetir):** ThreatLabz `medusalocker/` (agotado, 1 archivo, ya en
corpus) · **10 variantes de pcrisk** (Rapid, Stolen, Chip, LockLock, Karma, Protect, Infected,
Crypto, Luck, End) — todas el mismo molde «YOUR COMPANY NETWORK HAS BEEN PENETRATED», de las
5 probadas salieron 2 textos distintos.

⚠️ **Pendiente de decisión (Romina + tutor):** sacar `note_threatlabz_!!!READ_ME_MEDUSA!!!.txt`
y `_2.txt`, que son de **Medusa**, familia sin relación con MedusaLocker (FBI/CISA AA25-071A).
Si se sacan, la familia queda en 4 y sigue cerrada.

## 2. ✅ JIGSAW — CERRADA

- [x] `pcrisk_jigsaw_aleman.txt` — variante `.AFD` alemana · coseno 0,611
- [x] `pcrisk_jigsaw_frances.txt` — variante «Anti-Capitalist Jigsaw» `.fun` · coseno 0,484

**Fuentes ya revisadas:** pcrisk guía 9942 (6 bloques, los 2 primeros ya estaban en el corpus)
· BleepingComputer «Jigsaw Decrypted» (las 2 variantes que traía eran copias de lo que ya
había) · ThreatLabz (no tiene jigsaw) · repos de código fuente de Jigsaw (solo etiquetas de
UI, no la nota).

- [ ] ⏸️ `pcrisk_jigsaw_hacked.txt` — **en espera**: verifica como texto nuevo (coseno 0,601)
      pero pcrisk la llama «based on the source code of jigsaw», atribución débil. Decidir.
- [ ] ⏸️ Variantes en turco (`.ram`), portugués y polaco — **frenadas a propósito**: la familia
      ya está en 4 y el paso 4→5 vale +0,0006 de macro-F1 (IC 95 % [−0,0207; +0,0219]).
      Además sumar idiomas baja la cohesión. Reabrir solo si la medición dice que conviene.

## 4. 🔻 CHIMERA — falta 1, y el texto está AGOTADO

- [x] `pcrisk_chimera_ingles_autentico.txt` — pcrisk guía 9542 · coseno 0,502

**Fuentes revisadas — pcrisk es la ÚNICA con texto seleccionable, y ya está incorporada:**

| Fuente | Autor · fecha | Formato |
|---|---|---|
| pcrisk guía 9542 | Meskauskas | ✅ texto — **ya incorporado** |
| Malwarebytes, «Inside Chimera Ransomware» | hasherezade · 09-12-2015 | 🖼️ solo captura |
| SonicWall | 23-10-2015 | 🖼️ solo captura (Fig. 6) |
| Help Net Security | Zeljka Zorz · 03-11-2015 | 🖼️ solo captura |
| Trend Micro | — | responde 403 |
| ThreatLabz · RansomNoteFiles (Lemmou) | — | no tienen chimera |

**Dato clave de hasherezade:** «The HTML can be displayed in two languages – English and
German». **La nota era bilingüe por diseño**, así que el par alemán/inglés del corpus es
legítimo — y explica que CHIMERA tenga la cohesión más baja de las 30 (0,1538).

⚠️ **`chimera_note2.txt` sigue sin fuente**: no coincide con el inglés de pcrisk (coseno 0,502)
y su procedencia es solo «NapierOne/varios». Probablemente sea una traducción del alemán armada
con datos de reportes (el «2.45 BTC» sí está documentado por Help Net Security y Trend Micro).
No es contenido falso, pero **necesita fuente antes de respaldar una cifra.**

**Única vía para el texto que falta:**
- [ ] **OCR de las 3 capturas** de arriba (autorizado por el tutor). Bloqueado: falta tesseract.

## 3. 🔶 MAZE — falta 1

- [x] `pcrisk_maze_wallpaper.txt` — pcrisk guía 16145 «Maze 2019» · **fondo de escritorio** ·
      coseno 0,430 (el más bajo de todo el lote junto con el francés de Jigsaw)

**Fuentes ya revisadas (pcrisk AGOTADO para esta familia):** ThreatLabz `maze/` (1 archivo, ya
en corpus) · **pcrisk guía 15133** (su bloque es COPIA de `pcrisk_maze_1.txt`, coseno **0,904**
— apenas por encima del umbral) · **pcrisk guía 16145 «Maze 2019»**, 3 bloques: el
`DECRYPT-FILES.txt` es COPIA de las notas de ThreatLabz (coseno **0,980** y 0,966), el
wallpaper es el que sirvió, y el tercero se **descartó por criterio** (ver abajo).

⚠️ **Decisión de alcance que hay que declarar:** el texto nuevo de MAZE es el **mensaje del
fondo de escritorio**, no un archivo dejado en disco. Se aceptó porque el corpus **ya incluye
mensajes en pantalla**: `wannacry_note1.txt` es la ventana de Wana Decrypt0r y las dos notas
nuevas de JIGSAW son ventanas emergentes. **Pero se descartó el tercer bloque de pcrisk, que es
la página del sitio Tor de pago**: esa no la deja el malware en la máquina, es el sitio web del
atacante. **Criterio adoptado: entra lo que el malware muestra o deja en la máquina de la
víctima; no entra la web del atacante.**

**Próxima acción concreta para el texto que falta:**
- [ ] McAfee/Trellix «Ransomware Maze» y Sophos «Maze ransomware: extorting victims for 1 year»
      — análisis extensos, verificar si traen texto seleccionable o solo capturas
- [ ] id-ransomware.blogspot.com — Maze y **ChaCha** (nombre anterior de Maze, mayo 2019):
      la nota de la etapa ChaCha puede ser un texto distinto
- [ ] `MAZE-README.txt` — nombre de nota alternativo documentado, buscar su texto

## 5. 🔻 RYUK — faltan 2, y el texto está AGOTADO

**En el corpus, ya identificados uno por uno:**
- `pcrisk_ryuk_1.txt` = la nota **larga** «Gentlemen! Your business is at serious risk…»
- `note_pcrisk.txt`, `note_variant_email.txt`, `ryuk.txt` = el molde **«Your network has been
  penetrated»**, que colapsa en 1 plantilla (coseno 0,986). Los cuatro terminan en «No system
  is safe».

**Fuentes revisadas:**

| Fuente | Resultado |
|---|---|
| ThreatLabz `ryuk/` (local) | agotado, 1 archivo, ya en corpus |
| **pcrisk guía 13394** | 1 solo bloque = la nota «Gentlemen!» que **ya está** en el corpus |
| BleepingComputer, «Ryuk Ransomware Is Making Victims Left and Right» | 🖼️ solo capturas |
| CIS Security Primer – Ryuk (10-01-2020) | 🖼️ las notas están en figuras, no en texto |
| ~~malwiki.org~~ | ⛔ **dominio secuestrado**, ver reglas de fuentes |

**La tercera nota que falta existe y está identificada:** es la variante **corta de 2019**, que
según CIS trae solo un correo, el nombre del ransomware y la frase «balance of shadow universe».
**Pero las dos fuentes que la muestran la publican como imagen.**

**Próxima acción concreta:**
- [ ] OCR de las figuras de CIS Security Primer o de BleepingComputer (bloqueado: falta tesseract)
- [ ] `id-ransomware.blogspot.com` — Ryuk. **Es la fuente pendiente más prometedora** de toda la
      lista: publica notas como texto y todavía no se consultó para ninguna familia
- [ ] Virus Bulletin, «VB2019 paper: Shinigami's revenge: the long tail of the Ryuk malware» —
      paper académico, puede traer la nota transcripta (y sería la cita más fuerte)
- [ ] ⚠️ **NO** usar notas de Conti: es el sucesor, familia distinta

## 6. ⬜ NOTPETYA — faltan 2

**En el corpus:** 2 textos, y son los más parecidos entre sí de toda la lista (coseno 0,817,
consistente con su cohesión de 0,8169). F1 por familia ya en 0,695.

**Próxima acción concreta:**
- [ ] OCR de `3_datos/fuentes_notas/imagenes_notas/mbr-ransom-note.jpg` (ya está en disco; hubo
      un OCR previo con upscale ×3 + binarizado registrado en `notas_familias_criticas.md`)
- [ ] Hitachi HIRT y BleepingComputer — ya listadas en `fuentes_notas_descarga.md`
- [ ] ⚠️ **NO** usar notas de BadRabbit aunque el texto sea casi idéntico: es otra familia
      (y ya está en F1 0,983, no necesita nada)

## 7. 🔻 WANNACRY — AGOTADA en fuentes citables

**En el corpus:** los 2 textos en inglés — la ventana de Wana Decrypt0r y el
`@Please_Read_Me@.txt`.

**Por qué se cierra como resultado, no como tarea incumplida:** la muestra trae **28 archivos
`msg/m_*.wnry`** con la nota traducida, pero:
- [x] Ningún vendor publica las versiones traducidas como texto (revisado en español y alemán)
- [x] Los sitios hermanos de pcrisk en otros idiomas traducen el artículo, no la nota
- [x] `Ruddernation-Designs/WannaCry-Decompiled` no trae la carpeta `msg`
- [x] El único enlace al conjunto es un `.zip` en `transfer.sh`, host discontinuado — y no se
      bajan comprimidos ni muestras de repos de malware

**Única vía restante:** extraer de una muestra viva. **Decisión de Romina y el tutor**, no de
un chat de recolección.

## 8. ⛔ CRYPTOLOCKER — BLOQUEADA

**El original de 2013 no dejaba archivo de nota.** Verificado en Keith Jarvis, Dell SecureWorks,
diciembre 2013: el mensaje iba en una ventana y la lista de archivos cifrados al registro
(`HKCU\SOFTWARE\CryptoLocker\Files`). **No puede existir bruto.**

- [ ] Si se decide transcribir la ventana: `https://id-ransomware.blogspot.com/2020/12/cryptolocker.html`
      (Amigo-A, la única que distingue el original de los homónimos)
- [ ] ⚠️ `pcrisk/7327-cryptolocker` le atribuye extensión `.encrypted` al original, lo que
      sugiere contaminación: **no usarla sola**
- [ ] ⚠️ Resolver antes: `lm_Crypt0l0cker_HOW_TO_RESTORE_FILES.html` del corpus es
      **Crypt0l0cker = TorrentLocker**, otra familia

## 9. ⛔ WASTEDLOCKER — BLOQUEADA

**Molde rígido, medido:** 4 notas de **3 víctimas** (BBA Aviation, RL Hudson, y una con correos
`88828@PROTONMAIL.CH`) y **2 fuentes** (pcrisk, ThreatLabz) **colapsan en 1 sola plantilla**.
El molde tiene ~250 caracteres y solo varía el nombre de la víctima y los correos.

**Cualquier nota nueva va a colapsar.** Su F1 por familia 0,000 no es falta de esfuerzo de
búsqueda: la familia no produce textos distintos.

- [ ] ⚠️ Los textos distintos atribuibles pertenecen a **sucesores renombrados de Evil Corp**,
      no a WastedLocker. Usarlos sería etiquetar otra familia (trampa campaña-vs-familia).
      Verificar nombres y fechas con fuente citable antes de escribirlo en la tesis.
