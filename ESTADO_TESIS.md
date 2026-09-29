# ESTADO DE LA TESIS — documento vivo

> **Cómo usar este archivo:** cuando abras un chat nuevo y se haya perdido el contexto,
> dile a Claude: *"lee ESTADO_TESIS.md en mi carpeta Tesis"*. Con eso retoma todo.
> Mantener actualizado al final de cada sesión.

_Última actualización: 2026-08-25_

> ### ▶▶ PARA CONTINUAR EN OTRO CHAT (2026-08-25)
> **`6_notas_trabajo/HANDOFF_2026-08-25_limpieza_y_proximos_pasos.md`** — traspaso operativo:
> qué se limpió (corpus **155 → 149**), las cifras vigentes, los próximos pasos en orden, y la
> lista de lo que **NO** hay que volver a intentar.
> **⚠️ Toda cifra anterior al 2026-08-25 está sobre 155 o menos notas.** Al citar, decir la base.
> **⚠️ Las carpetas `_150` están SUPERADAS** (previas a retirar `LOCKBIT/lb20.txt`): no citarlas.

> ### ⏰ AVISO DE FECHAS (2026-08-28)
> **El reloj de la máquina estaba 3 días atrasado** durante la sesión del 28 de agosto: marcaba
> el 25 y se corrigió (por NTP) a mitad del trabajo. Consecuencia: **varias cifras y notas
> escritas ese día quedaron etiquetadas «2026-08-25» cuando se midieron el 2026-08-28.**
> - **Los encabezados de los 5 bloques afectados ya están corregidos a 2026-08-28**: M.1
>   re-medida, la corrección de las dos familias de 1 plantilla, B.1 re-medida, el análisis de
>   margen, y la decisión de alcance de la ventana de pago de CryptoLocker.
> - **Puede quedar «2026-08-25» dentro del cuerpo de esos bloques**, en el manifiesto
>   `manifiesto_corpus_v2_150ventana.csv`, en el campo `fecha` de los JSON escritos ese día
>   (`margen_notas_149/manifiesto_margen.json` dice 2026-08-25) y en la fecha de modificación
>   de los archivos generados. **Si una fecha del 25 aparece junto a una cifra sobre 149 o 150,
>   es del 28.** Lo del 25 es la limpieza del corpus, M.3 y el error de HelloKitty.
> - Los resultados **no** están afectados: solo la etiqueta. Nada se re-midió.

> ### ⚠️ EL TECHO DE LA ENTROPIA NO ES 8,0 (2026-08-29) — AFECTA COMO SE LEE §4.5.7
> Saltó midiendo el perfil de entropía de CERBER. **La entropía de Shannon estimada sobre una
> muestra chica no puede llegar a 8,0**, porque la muestra no alcanza a visitar los 256 valores.
> Techo medido por simulación (4.000 repeticiones de bytes uniformes):
>
> | Bloque | Entropía media de datos ALEATORIOS | Desvío |
> |---|---|---|
> | 256 bytes | **7,175** | 0,052 |
> | **512 bytes** | **7,590** | **0,034** |
> | 1.024 bytes | 7,809 | 0,017 |
> | 4.096 bytes | 7,955 | 0,004 |
> | 65.536 bytes | 7,997 | 0,000 |
>
> **La tabla de familias difíciles se mide sobre ventanas de 512 bytes**, o sea que su techo es
> **7,590**, no 8,0. Leídas así, las cifras dicen algo más fuerte:
> - **CRYPTOLOCKER (7,594 / 7,589) y WASTEDLOCKER (7,591 / 7,585) están EXACTAMENTE en el
>   techo** — indistinguibles de datos uniformes en los dos extremos.
> - DARKSIDE: cabecera en el techo (7,593), **cola 3σ por debajo** (7,491).
> - JIGSAW: algo por debajo en ambos (7,514 / 7,551).
> - **Resto de familias 7,03 en cabecera: más de 15σ bajo el techo.**
>
> El «próximos a 8,0 bits/byte» de la Figura 4.1 **sigue bien**: esa figura se calcula sobre
> archivos completos, de decenas de miles de bytes, donde el techo sí es ≈ 8,0. Lo que no se
> podía hacer era leer 7,59 (ventana de 512) contra 8,0 (archivo completo) como si fueran la
> misma escala. **Ya se agregó un párrafo a §4.5.7 que lo declara.** Solo se agregó.

> ### 🚨🚨 HALLAZGO DEL JOB 3772 (2026-08-29) — CERBER DEJA LA CABECERA EN CLARO
> Registrado de lo pegado por Romina en el chat, del log `slurm-exp2d-3772.out`. El control de
> integridad por magia de tipo que se agregó al Exp. 2d el 29-08 disparó en **tres familias**, y
> son **tres cosas distintas**, con severidad distinta. **No mezclarlas.**
>
> ```
> ⚠ ARCHIVOS QUE PARECEN ESTAR EN CLARO (magia de tipo conocido):
>    CERBER    500 archivo(s)  ej.: 6aeFyFo2Es.bed4 [JPEG], 87VdlVNpOH.bed4 [ZIP/OOXML], 42Z8Mxseue.bed4 [PDF]
>    JIGSAW      3 archivo(s)  ej.: 0089-pptx.pptx [ZIP/OOXML], 0024-pptx.pptx, 0025-pptx.pptx
>    NOTPETYA   16 archivo(s)  ej.: 0030-doc.doc [OLE/Office], 0155-xls.xls, 0043-xls.xls
> ```
>
> #### 1. CERBER: 500 de 500 — NO es contaminación, es CIFRADO PARCIAL (a verificar)
> Los archivos llevan **nombre renombrado por CERBER** (`6aeFyFo2Es.bed4`, base aleatoria +
> `.bed4`), o sea que CERBER los procesó, y sin embargo **empiezan con la magia del documento
> original** — y con magias **distintas entre sí** (JPEG, ZIP/OOXML, PDF). La lectura que se
> sostiene: **CERBER cifra parcialmente y deja la cabecera del documento original en claro.**
> No son los 12 JPEG del 2026-08-16: esos conservaban su nombre original y ya están apartados
> en `_sin_cifrar/`. Estos son 500 de 500.
>
> **No es falso positivo del detector:** las magias son de 3 y 4 bytes, así que sobre bytes
> aleatorios la probabilidad es $2^{-24}$ a $2^{-32}$; en 15.000 archivos se esperarían **0,001
> coincidencias**, no 500.
>
> **Y CERBER sí modifica los archivos:** el Exp. 2b le encontró un **sufijo constante de 64
> bytes con cobertura 1,00**. Un `.docx` intacto no comparte 64 bytes finales con un `.jpg`
> intacto. El cuadro coherente es **cabecera original preservada + cola propia de CERBER**.
>
> **Converge con algo ya medido y no explicado:** en `d_familias_dificiles.csv` el «resto de
> familias» tiene entropía de **cabecera 7,03**, bastante más baja que las seis difíciles (7,59).
> Esa cabecera menos aleatoria del resto es, al menos en parte, esto.
>
> **Qué toca y qué no:**
> - **NO invalida el F1 = 1,000 de CERBER en el Exp. 2c**: lo explica su firma de cola, que
>   tiene cobertura 1,00 y es propia de la familia. La cabecera preservada es del documento de
>   origen y la comparten todas las familias, así que no discrimina.
> - **SÍ obliga a un matiz en §4.5.5** (dejar-un-tipo-fuera): para CERBER el tipo de documento
>   **sí está a la vista en los bytes**. La conclusión agregada se sostiene — son 25 a 28
>   familias por pliegue y CERBER aparece en uno solo — pero hay que declararlo.
> - Es, en sí mismo, **un resultado sobre la familia** y probablemente citable (el cifrado
>   parcial es técnica documentada de ransomware, para ganar velocidad).
>
> **VERIFICAR ANTES DE ESCRIBIR NADA** (volcado de los primeros 16 bytes; comparar contra un
> original de NapierOne del mismo tipo):
>
> ```bash
> cd /scratch/ralfonzo/tesis && for f in $(ls /scratch/ralfonzo/Napierone-small/CERBER-small/ | head -6); do echo "== $f"; od -A d -t x1 -N 16 "/scratch/ralfonzo/Napierone-small/CERBER-small/$f" | head -1; done
> ```
>
> #### 2. JIGSAW (3) y NOTPETYA (16): esto sí es del tipo de los 12 JPEG
> Conservan **nombre Y extensión originales** (`0089-pptx.pptx`, `0030-doc.doc`) y magia de
> documento: son archivos **sin cifrar**, no cifrado parcial. Son **19 sobre 15.000 = 0,13 %**.
> - Cuadra con lo ya medido: en el Exp. 2b la **cobertura de extensión de JIGSAW era 0,98**, no
>   1,00. Ese 2 % que faltaba es esto.
> - NOTPETYA no cambia extensiones nunca, así que sus 16 no se distinguen por nombre de los
>   otros 484: no envenenan la columna del nombre más que el resto.
> - **Decisión tomada: dejar correr el 3772.** Excluir los 12 JPEG de CERBER movió las métricas
>   0,001 a 0,003; 19 archivos van a mover menos, y el log deja documentado exactamente qué
>   había en los datos. Re-correr 4-5 h por 0,13 % no se justifica **antes** de decidir qué se
>   hace con CERBER, que es la pregunta grande.
>
> #### 3. El diagnóstico de circularidad de la extensión: confirmado y con número
> ```
> familias con UNA sola extensión: 25 de 30
> extensiones compartidas por más de una familia: 5 de 905
>    .doc / .docx / .xls / .xlsx -> BADRABBIT, NOTPETYA
>    .pptx -> BADRABBIT, JIGSAW, NOTPETYA
> exactitud de una TABLA DE CONSULTA que solo mira la extensión: 0,9724
> ```
> **La columna (3) del 2d queda declarada como memorización de un diccionario**, tal como se
> diseñó: una tabla de consulta que solo mira la extensión ya acierta **0,9724 de exactitud**.
> Las únicas cinco extensiones ambiguas son las de ofimática, y las comparten las dos familias
> que **no renombran** (BADRABBIT y NOTPETYA) más los tres archivos sin cifrar de JIGSAW.
> **905 extensiones distintas** sobre 30 familias: SUNCRYPT asigna una por archivo.
>
> #### Jobs
> **3771 cancelado** (llevaba los hiperparámetros viejos). **3772 corriendo** desde el
> 2026-08-28 23:29 con la versión corregida.

> ### 🚨 REVISION DEL JOB DE EXP. 2d (2026-08-29) — TENIA LOS HIPERPARAMETROS MAL
> `exp2d_nombre_extension.py` declaraba usar los hiperparámetros del Exp. 2c y tenía puestos
> **los de sklearn por defecto**: `max_depth=None, min_samples_leaf=1, max_features="sqrt"`.
> Los buenos, verificados contra **tres** fuentes — `clasificador_bytes.py` línea 88
> (`HIPER_2C`), `resultados_bytes/bytes_manifiesto.json` del job 3639 y el log
> `slurm-bytesms-3648.out` — son **300 árboles, profundidad 20, hoja 2, `max_features=0.3`**.
>
> **Por qué importaba:** la columna (1) del 2d es «solo bytes» y tiene que **reproducir la
> referencia publicada** (0,912 ± 0,002 de exactitud). Con los otros hiperparámetros no la
> reproduce, y entonces las tres columnas dejan de ser comparables contra el Exp. 2c — que es
> justo lo que el experimento quiere medir. Corregido y **commiteado** (`d537ebd`).
>
> **Otras tres mejoras al mismo job:**
> 1. **Se agregó A.3, la curva de aprendizaje**, que seguún `EXPERIMENTOS_PENDIENTES.md` iba
>    en este mismo job porque comparte la carga de datos. Tamaños 10 a 500 archivos/familia,
>    **subconjuntos anidados**, solo bytes, con delta pareado entre pasos consecutivos e
>    IC 95 % — el mismo criterio de lectura que B.1 en notas.
> 2. **Control de integridad por magia de tipo.** Este experimento usa el NOMBRE, así que un
>    archivo en claro con su nombre original lo arruina. El script cuenta y lista los archivos
>    que empiezan con magia conocida (JPEG, PDF, ZIP/OOXML, OLE, PNG, GIF, RTF, GZIP).
> 3. El job pide `--time=12:00:00` y `--nodelist=c2`, que no tenía.
>
> **⚠️ ANTES DE LANZAR, verificar en el clúster que los 12 JPEG siguen apartados:**
> `ls /scratch/ralfonzo/Napierone-small/CERBER-small/_sin_cifrar/`. Si no están ahí, el 2d da
> un resultado falso. El script lo detecta igual y lo avisa en el log.
>
> Probado de punta a punta con un corpus sintético (5 familias con firma y extensión
> controladas): las tres columnas, los delta con IC, la curva y el diagnóstico de circularidad
> funcionan. **Estimado real: 4-5 h**, no 3-4 — con `max_features=0.3` cada división mira 307
> columnas y no 32.


> ### ✓✓ ESCRITO EN LA TESIS (2026-08-28) — EL FRENTE DE ARCHIVOS QUEDA REDACTADO
> Se agregaron **nueve subsecciones** a `resultados.tex`, todas con base y métrica declaradas.
> **Solo se agregó: no se tocó una línea de lo ya escrito.** Compila: **74 páginas, 0 errores,
> 0 referencias sin resolver.** Respaldo previo en `resultados.tex.antes_archivos`.
>
> | § | Subsección | Qué cierra | Fuente verificada |
> |---|---|---|---|
> | 4.3.3 | Optimización de hiperparámetros sobre las características estadísticas | A1 — cierra la única limitación de optimización declarada | `resultados_gridsearch_estadisticas/` |
> | 4.4.4 | El criterio de detección: de la unanimidad a la mayoría | BADRABBIT + estabilidad del 2b | `resultados_estructural_10semillas/` (10 manifiestos) |
> | 4.4.5 | Validación externa de las firmas descubiertas | 6 coincidencias con ID Ransomware | `Pruebas.xlsx`, hoja «Deteccion de archivos encriptad» |
> | 4.5.4 | Validación anidada y estabilidad de la estimación | pedido de Cappo sobre train/test y desvío | `resultados_bytes/bytes_resumen.csv` + `..._multisemilla_job3648/` |
> | 4.5.5 | Generalización a tipos de documento nunca vistos | A2 | `resultados_analisis_bytes/a_generalizacion_tipos.csv` |
> | 4.5.6 | Cuántos bytes hacen falta y dónde reside la señal | A3 + A4 + bloque del medio + control de relleno | `resultados_ablacion_extendida/` + `b_importancia_por_posicion.csv` |
> | 4.5.7 | Por qué fallan las seis familias difíciles | A5, con el matiz de SUNCRYPT/NOTPETYA | `d_familias_dificiles.csv` + por-familia de 10 semillas |
> | 4.5.8 | Familias que destruyen el nombre original | A6 | recuento de nombres del 2026-08-16 |
> | 4.5.9 | Verificación de integridad del conjunto de datos | los 12 JPEG de CERBER | job 3639 vs 3633 |
>
> **Figuras insertadas** (ya estaban en `images/`, sin usar): `fig_ablacion_extendida.png`
> (Fig. 4.3) y `fig_importancia_por_posicion.png` (Fig. 4.4).
>
> #### Cifras que se corrigieron al verificar contra los CSV (no citar las viejas)
> - **El desvío del criterio de mayoría se reduce 16 veces, no 14.** Unanimidad
>   0,900 ± 0,016 frente a mayoría 0,932 ± 0,001 (desvío muestral sobre las 10 semillas:
>   0,0157 y 0,00098; el cociente es 16,1). **El informe semanal del 18-08 que se le pasó a
>   Cappo dice «catorce»** — salió de redondear antes de dividir. La cifra buena es 16.
> - **Firmas binarias: 17 familias bajo el criterio de mayoría** (4 prefijos — CUBA, LORENZ,
>   TESLACRYPT, WANNACRY — y **13 sufijos**), idénticas en las 10 semillas. El «15» que ya
>   estaba escrito es el valor **bajo unanimidad**, y ahí oscila: 15 en 7 semillas y 16 en 3.
>   Las dos que entran y salen son BADRABBIT y BLACKBASTA. Las dos bases quedan declaradas.
> - **BADRABBIT bajo unanimidad se detecta en 1 de 10 semillas**, contra las 1,7 que predice
>   $0{,}965^{50} \approx 0{,}17$. La predicción teórica y la medición coinciden.
> - **La cobertura de BADRABBIT en las muestras de 50 va de 0,92 a 1,00, media ≈ 0,96** —
>   reproduce por vía independiente el 96,5 % del volcado sobre la familia completa.
> - **Últimos 16 bytes = 27,8 % de la importancia** (calculado sobre `b_importancia_por_posicion.csv`,
>   offsets −1 a −16). En este documento figuraba «27,2 %», de una banda distinta.
> - **CERBER renombra 981 archivos, BLACKMATTER 13.** El «491 archivos» de
>   `PENDIENTE_REDACCION.md` (A6) estaba mal; el recuento verificado del 16-08 es el bueno.
>
> #### Las 6 coincidencias con ID Ransomware, verificadas una por una en `Pruebas.xlsx`
> Idénticas: **LORENZ** y **MAZE**. La firma hallada **contiene** a la declarada:
> **WANNACRY, TESLACRYPT, GANDCRAB, MEDUZALOCKER**.
> **CUBA y PHOBOS NO cuentan como corroboración por bytes**: la celda de CUBA está vacía y la
> de PHOBOS declara una regla («0x00 metadata; divider»), no una secuencia. SODINOKIBI igual
> (CRC32 de la clave pública). CONTI corrobora la **extensión** `.MRBNY`, no los bytes.
> **RYUK sigue siendo la única discrepancia** y se escribió como tal: la herramienta reporta
> `HERMES` en `[0x584D0-0x58792]` — un rango de 706 bytes — y nuestro detector no la halla.
>
> #### ✅ RESUELTO EL MISMO DÍA — los logs de SLURM ya están en disco
> Romina bajó `logs_slurm_2026-08-17.tgz` (23 logs) y quedaron en
> **`4_resultados/_logs_slurm_2026-08-17/`**. Con eso el frente de archivos **no tiene
> ninguna cifra sin respaldo local**. Lo que el log del 3633 obliga a corregir:
>
> - **El «0,9105 / 0,9097» que este documento atribuía al job 3633 NO está en su log.**
>   `slurm-bytes-3633.out` imprime **tres decimales**: `exactitud 0.910 | balanced 0.910 |
>   macro-F1 0.910`. Los cuatro decimales venían de los CSV que el 3639 pisó; son
>   compatibles con el log pero **no verificables**. En la tesis quedó escrita la
>   comparación a la precisión que el artefacto sostiene: **0,909 / 0,907 (corpus
>   verificado, job 3639) frente a 0,910 / 0,910 (con los 12 JPEG, job 3633)**, diferencia
>   de 0,001 y 0,003. Misma conclusión, ahora citable. **No volver a escribir 0,9105.**
> - **Confirmado en el log:** CERBER pasa de `1.00 / 0.99` (3633) a `1.00 / 1.00` (3639).
>
> #### ✅ Queda fijada la procedencia de cada fila de la Tabla 4.11 (Exp. 2c)
> Los tres logs lo dejan cerrado, y es lo que hace falta para la pasada de pulido:
>
> | Job | Base | Búsqueda anidada (exactitud / macro-F1) | Etapa final |
> |---|---|---|---|
> | **3557** | 29 fam. · 5.800 arch. | posicional **0,897 / 0,897** · LogReg 0,856 / 0,858 · LinearSVC 0,849 / 0,846 | 0,910 / 0,908 (14.500) |
> | 3633 | 30 fam. · 6.000 arch. · **con** los 12 JPEG | posicional 0,899 / 0,899 · LogReg 0,860 / 0,864 · LinearSVC 0,850 / 0,847 | 0,910 / 0,910 (15.000) |
> | **3639** | 30 fam. · 6.000 arch. · **sin** los 12 JPEG (**canónico**) | posicional **0,891 / 0,892** · LogReg 0,862 / 0,865 · LinearSVC 0,852 / 0,849 | 0,909 / 0,907 (15.000) |
> | 3648 | 30 fam. · 10 semillas | — (hiperparámetros fijos) | **0,9120 ± 0,0016 / 0,9111 ± 0,0014** |
>
> **Las tres filas de búsqueda de la Tabla 4.11 son del job 3557, o sea 29 familias**, y la
> fila final es del 3648, o sea 30. La tabla no lo declara. Los valores equivalentes sobre 30
> familias (0,891 / 0,862 / 0,852) están escritos en la nueva §4.5.4 con su base.
>
> #### ⚠️ Dos incoherencias PREVIAS que quedaron a la vista (no se tocaron)
> 1. El pie de la **Figura 4.2** dice «siete de las **diez** que solo poseían extensión propia»
>    y el cuerpo de §4.5.3 dice «**once** familias». Son 11 (verificado). Es una palabra.
> 2. La **Tabla 4.11** (Exp. 2c) mezcla bases: las filas de búsqueda son de **5.800 archivos /
>    29 familias** (0,897) y la fila final es de **15.000 / 30**. Los valores de la etapa de
>    búsqueda sobre 30 familias son **0,8913 / 0,8623 / 0,8515**, y quedan escritos en la nueva
>    §4.5.4. En la pasada de pulido hay que decidir si la tabla se unifica.


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

## ★★★ B.3 RE-MEDIDO SOBRE 155 NOTAS (2026-08-20, local, `--solo-grafo`)

> **Base declarada: 155 notas, 106 plantillas, 30 familias.** Es una RE-MEDICIÓN, mismo
> código (`2_codigo/grafo_marcadores.py`), sin tocar el método. La corrida vieja (144 notas /
> 97 nodos) queda intacta más abajo; esto se AGREGA con su propia base.

**Comando** (carpeta de salida NUEVA, no se pisó la del 19-08):
```
python grafo_marcadores.py --solo-grafo --salida C:/Users/Romina/Tesis/4_resultados/resultados_grafo_marcadores_155
```
Se agregó `--salida` al script (cambio mínimo, mismo patrón que `clasificador_notas_v2.py`;
commit `091cf7b` en develop, sin coautoría). Salidas en
`4_resultados/resultados_grafo_marcadores_155/`. El F1 por familia para la correlación sale de
`4_resultados/resultados_extension_155/` (B.1, P2ret, k=todo, 100 repeticiones), no de una
corrida nueva. **P1/P2/P3 + control de azar NO se re-corrieron** (no estaban entre lo pedido y
el item de correlación usa el F1 de B.1 ya existente; es un paso aparte, un solo comando).

**Recuento del grafo, 155 vs 144:** notas 155 (era 144) · plantillas 106 (era 95) · **nodos
familia#plantilla 108** (eran 97; siguen siendo 2 grupos que cruzan familias, ver item 2).
Marcadores por tipo: **URL 154 · ONION 99 · EMAIL 80 · CLAVE 36 · BTC 7 · ID 6** (eran URL 159
· ONION 106 · EMAIL 74 · CLAVE 37 · ID 9 · BTC 5). **Plantillas sin ningún marcador: 16 de 108**
(eran 11 de 97).

### 1) Cohesión por familia sobre 155 — coseno char 3-5 medio entre plantillas (por construcción < 0,90)

`b3_cohesion_por_familia.csv`. Todas las cifras son **coseno medio entre plantillas de la
familia, base 155 notas**. Comparadas con la tabla vieja (base 144). Las tres causas previstas
se confirman, y aparecen tres cambios no previstos:

| Familia | Coh. 144 | Coh. 155 | n_plant. 144→155 | Qué pasó |
|---|---|---|---|---|
| **MEDUZALOCKER** | 0,4317 | **0,6548** ▲ | 3→4 | se limpió la contaminación Medusa + 2 notas reales (chip, rapid); **margen pasó a POSITIVO** (−0,1685 → +0,0607) |
| **CHIMERA** | 0,1538 | **0,3402** ▲ | 2→4 | 2 plantillas nuevas (inglés auténtico pcrisk + alemán HNS); ahora hay pares de-de y en-en que suben la media. **Ya no es la más baja de las 30** |
| **JIGSAW** | 0,5075 | **0,2300** ▼ | 2→4 | sumó traducciones (alemán g91, francés g92): baja la cohesión como se anticipó |
| **RYUK** | 0,4388 | **0,2175** ▼ | 2→4 | **es ahora la cohesión MÁS BAJA de las 30**; no es por idioma (ver item 4) sino por mezclar fragmentos ultra-cortos solo-IOC (idr_balance_2019, idr_portal_2021) con la prosa pcrisk |
| **WASTEDLOCKER** | — (1 plant.) | **0,8967** ▲ | 1→3 | ya no es 1 molde rígido: 4 notas → 3 plantillas por **deriva del IDF** al crecer el corpus (el par estaba en coseno mín. 0,894, al borde de 0,90). Ahora es la cohesión MÁS ALTA y margen MÁS ALTO (+0,5075). **Reevaluar el relato de «familia irrecolectable»** |
| **CRYPTOLOCKER** | 0,3564 | 0,4337 ▲ | 3→2 | perdió una plantilla (salió el Crypt0l0cker/TorrentLocker en la limpieza de integridad) |
| GANDCRAB | 0,6033 | 0,5588 ▼ | 5→4 | perdió una plantilla por deriva del IDF |

El resto se mueve poco (±0,01): AVOSLOCKER 0,8103→0,8159 · BLACKCAT 0,6297→0,6326 · CONTI
0,7776→0,7794 · DARKSIDE 0,8723→0,8714 · HELLOKITTY 0,3108→0,3111 · LOCKBIT 0,4296→0,4291 ·
NETWALKER 0,8053→0,8121 · NOTPETYA 0,8169→0,8128 · SUNCRYPT 0,4529→0,4536 · TESLACRYPT
0,7998→0,7996 · WANNACRY 0,4449→0,5262. **La deriva del IDF (documentada en la sección de
recolección) mueve el conteo de plantillas de familias al borde del umbral; hay que citar la
cohesión SIEMPRE con la base 155.**

### 2) Plantillas compartidas entre familias — SIGUEN SIENDO LAS MISMAS

**Los 2 parentescos por CONTENIDO (grupos de casi-duplicados que cruzan familias) persisten, y
no aparecieron nuevos** (`grupos_neardup.csv` de `resultados_extension_155`):

| Grupo (155) | Familias | Notas |
|---|---|---|
| **grupo 6** | BLACKBASTA + CONTI | `blackbasta2.txt` + `conti4.txt` |
| **grupo 56** (era grupo 55 en 144) | DHARMA + PHOBOS | 11 notas DHARMA (.hta/.txt) + `pcrisk_phobos_1.txt` |

El renumerado 55→56 es solo por el crecimiento del corpus; **es el mismo parentesco**. Por eso
108 nodos = 106 plantillas + 2 (los 2 grupos que aportan un nodo por familia). Las notas nuevas
**no crearon ninguna componente compartida nueva**.

**Aristas del grafo de marcadores entre familias** (variante canónica con_exclusion_filtro2):
**5 aristas, todas URLs de torproject** — bajaron de 7 (144) a 5 (155). Detalle: 4 son
BLACKBASTA#5/#6 ↔ CONTI#42/#44 por `https://torproject.org`, y **1 NUEVA** AVOSLOCKER#0 ↔
CLOP#41 por `https://www.torproject.org/download/`. **Ningún IOC operativo se comparte entre
familias distintas** — igual que en 144: las aristas cruzadas son infraestructura Tor común, no
parentescos. (El parentesco BLACKBASTA↔CONTI y DHARMA↔PHOBOS sale de la deduplicación por
contenido —grupos 6 y 56—, NO de los marcadores. No escribir nunca «el grafo reencuentra los
parentescos».)

### 3) Correlación cohesión → F1 por familia — SE SOSTIENE sobre 155

Spearman contra el **F1 por familia de B.1** (`resultados_extension_155`, curva 30fam · P2ret ·
k=todo · **155 notas** · 100 repeticiones). **Control:** con el mismo cálculo sobre la base
vieja (cohesión 144 + F1 144) se **reproduce exactamente** la cifra registrada, ρ = **+0,7041**
(n=29), lo que valida el método antes de aplicarlo a 155.

| Predictor | ρ 144 (n=29) | ρ 155 (n=30) | ρ 155 (n=29, sin WASTED, 1:1) |
|---|---|---|---|
| **Cohesión interna** (coseno medio entre plantillas) | **+0,7041** (p=2·10⁻⁵) | **+0,6879** (p=2,7·10⁻⁵) | **+0,6925** (p=3,1·10⁻⁵) |
| **Margen** (cohesión − máx. a plantilla ajena) | +0,6741 (p=6·10⁻⁵) | **+0,7462** (p=2,2·10⁻⁶) | **+0,7500** (p=2,8·10⁻⁶) |
| Fracción de pares unidos por marcador | +0,4245 (p=0,022) | +0,3016 (p=0,11, **n.s.**) | +0,3431 (p=0,068, **n.s.**) |
| Cantidad de plantillas de la familia | −0,1084 (p=0,58, n.s.) | **−0,4885** (p=0,006) | −0,4730 (p=0,010) |

- **El resultado central aguanta:** la cohesión interna sigue siendo un predictor fuerte del F1
  por familia (ρ +0,69 sobre 155, contra +0,70 sobre 144). **Es n=30 ahora**, no 29: a 155
  todas las familias tienen ≥2 plantillas (WASTEDLOCKER pasó de 1 a 3), así que ninguna queda
  fuera por falta de cohesión definible.
- **El margen pasó a ser el predictor MÁS fuerte** (ρ +0,75). Coherente con que la recolección
  llenó familias de baja cohesión: el margen las separa mejor.
- ⚠️ **La fracción de pares unidos por marcador DEJÓ de ser significativa** (era p=0,022, ahora
  p=0,11). Con más plantillas por familia, la continuidad de IOCs pesa menos en el F1.
- ⚠️⚠️ **La cantidad de plantillas volteó a NEGATIVA y significativa** (ρ −0,49, p=0,006), cuando
  en 144 era nula (ρ −0,11, n.s.). **Esto es un CONFUNDIDO de la recolección dirigida, NO un
  hallazgo causal:** las notas nuevas se sumaron justo a las familias difíciles y de baja
  cohesión (CHIMERA, JIGSAW, RYUK, MAZE, MEDUZALOCKER, WANNACRY → todas a 4 plantillas), así que
  «muchas plantillas» quedó correlacionado con «era una familia difícil». **No contradice a
  B.1** (que es causal, quitando plantillas y midiendo la caída). Hay que escribirlo con este
  cuidado: la correlación entre familias está contaminada por a cuáles se les recolectó.

**Contexto (F1 por familia P2ret k=todo, 100 rep, base 144 → base 155):** la recolección movió
fuerte a las difíciles: CHIMERA 0,000→0,912 · WANNACRY 0,010→0,733 · MEDUZALOCKER 0,000→0,608 ·
MAZE 0,000→0,561 · RYUK 0,032→0,271 · WASTEDLOCKER (indef.)→0,987. Bajaron JIGSAW 0,670→0,505
(las traducciones, como se anticipó) y CRYPTOLOCKER 0,410→0,097 (perdió la plantilla
TorrentLocker). *(Son cifras de B.1/P2ret, contexto de la correlación, no de B.3.)*

### 4) Familias con plantillas en IDIOMAS distintos — insumo para preregistrar embeddings

Detección por heurístico de palabras-función sobre el texto extraído, **verificada a mano nota
por nota** (el detector marcó 4, una era falso positivo):

| Familia | Idiomas | Plantillas |
|---|---|---|
| **JIGSAW** | inglés + alemán + francés | g89/g90 (en), g91 (de: «Guten Tag, bedauerlicherweise…»), g92 (fr: «BONJOUR VOUS VENEZ DE VOUS FAIRE HACKER…») |
| **CHIMERA** | inglés + alemán | g35/g37 (en), g34/g36 (de: «Sie wurden Opfer der Chimera Malware…») |
| **GANDCRAB** | inglés + francés | g75 (fr: «Attention! Tous vos fichiers… sont cryptés», nota completa; hallazgo nuevo) + 7 en |

- ⛔ **RYUK NO va en la lista** — el heurístico la marcó `pt` por FALSO POSITIVO sobre
  `idr_ryuk_balance_2019.txt`, que son solo 2 emails + «Ryuk / balance of shadow universe»
  (inglés, 6 tokens). Verificado a mano. Su cohesión baja es por longitud/estructura, no idioma.
- **Casos que NO son multi-idioma pero conviene declarar** para no confundir al preregistrar:
  SUNCRYPT (`suncrypt.html` trae las etiquetas «EN DE FR ES JP» de un selector, pero el cuerpo
  es inglés) y TESLACRYPT («NOT YOUR LANGUAGE? USE Google Translate» + link, cuerpo inglés): son
  inglés con afordancia de traducción, no plantillas localizadas.
- **WANNACRY es multi-idioma en la realidad pero NO en el corpus:** las 4 plantillas del corpus
  son inglés; las 28 localizadas `m_*.wnry` no se pudieron obtener (ver resultado negativo
  WANNACRY). Es una familia multi-idioma latente sin material en disco.

> **El experimento de embeddings (paso siguiente, que NO se arranca acá) se preregistra sobre
> ESTA salida: la predicción es que un embedding multilingüe sube la cohesión / el F1 de
> JIGSAW, CHIMERA y GANDCRAB (las 3 con plantillas realmente en otro idioma), y NO debería mover
> a RYUK (baja por longitud) ni a las de cohesión ya alta.** Predicción sobre
> `4_resultados/resultados_grafo_marcadores_155/`.

---

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

## ⚠️ AUDITORÍA DE NOMBRES CONTRA MISP (2026-08-22) — SOLO 10 DE 155 SON GENUINOS

Cruce automático de los nombres de archivo del corpus (155 notas) contra los
`ransomnotes-filenames` del catálogo MISP (537 nombres reales en 298 entradas). Resultado:

**Solo 10 de 155 notas (6,5 %) tienen un nombre que coincide con un nombre documentado.**
Coinciden: CERBER 5 (`# DECRYPT MY FILES #.html/.txt`), DHARMA 2 (`Info.hta`,
`FILES ENCRYPTED.txt`), BLACKBASTA 1 (`instructions_read_me.txt`), MEDUZALOCKER 1
(`HOW_TO_RECOVER_DATA.html`), TESLACRYPT 1 (`Howto_RESTORE_FILES.html`). El resto son nombres
puestos por el curador del repositorio (`pcrisk_ryuk_1.txt`, `idr_ryuk_balance_2019.txt`,
`note_variant_email.txt`…).

**Consecuencia para M.2 / D.2 (nombre de la nota como característica): SE CAE.** Con 10 notas
genuinas repartidas en 5 familias no hay material para una vista de nombre; usar los nombres
del curador sería circular (el curador escribió el nombre sabiendo la familia). El
desbloqueo que se anotó el 2026-08-20 era prematuro: MISP aporta los nombres de referencia,
pero el CORPUS no los conserva.

**Lo que SÍ habilita este cruce, y hay que escribirlo:**
1. Es una **verificación de autenticidad del corpus**: las notas que conservan nombre original
   coinciden con lo documentado por una fuente independiente ⇒ el corpus es genuino donde
   dice serlo. Va al cap. 3 como control de procedencia.
2. Explica por qué **ID Ransomware rinde mejor en producción que en nuestro corpus**: su señal
   principal es el nombre del archivo, que nuestro corpus mayormente perdió al recolectarse
   de repositorios. Es un límite del corpus público, no del método — y es citable.
3. Refuerza la limitación ya declarada: los repositorios de notas **pierden metadatos** que en
   un incidente real sí están disponibles.

**Registrar en el manifiesto** la columna procedencia del nombre (`genuino_misp` para las 10,
`curador` para el resto) — es barato y deja la auditoría reproducible.

## ★ RESPUESTA A CAPPO (2026-08-22) — ¿otros trabajos analizan el catálogo MISP? NO como corpus

Pregunta del tutor (WhatsApp 21-08): «¿Otros trabajos analizan este corpus base [el JSON de
MISP]? ¿Qué resultados obtienen?». Barrido web con verificación página por página (4
rastreadores: el galaxy mismo, RansomLook, la planilla «Ransomware Overview», el blog de
Amigo-A) + búsqueda de cierre. **Conclusión: ningún trabajo publicado usa el galaxy
«Ransomware» de MISP (ni sus fuentes) como corpus analizado con métricas. Es un catálogo de
inteligencia de amenazas — se usa como referencia y enriquecimiento, no como dataset
evaluado.** En detalle:

- **Galaxy MISP:** tras ~12 combinaciones de búsqueda, cero papers que lo usen como datos o
  ground truth. Lo más cercano usa OTROS clusters del proyecto (threat-actor.json: dAPTaset,
  Springer 2020; arXiv 2103.02301) o solo el formato MISP, sin métricas sobre el catálogo.
- **RansomLook (ransomlook.io):** sí aparece como fuente de datos, pero de METADATOS de
  sitios de filtración, nunca de notas para clasificar familia. El único análisis académico:
  Müller & Yannikos (arXiv 2605.24559, 2026), 25.761 posts de víctimas — estadística
  descriptiva (top 1 % de grupos concentra 21,29 % de los ataques), sin clasificación.
- **Blog de Amigo-A (id-ransomware.blogspot):** usado como catálogo de referencia (446 de las
  2.135 entradas del galaxy lo citan; G DATA lo recomienda para identificación manual). Ningún
  trabajo lo analiza como corpus. Precisión útil: el corpus de Lemmou NO salió de ahí (salió
  de Malware Traffic Analysis, Hybrid Analysis y otros).
- **Planilla «Ransomware Overview» (2016-17):** solo referencia comunitaria; nada la analiza.

### Ampliación (2026-08-22): ¿es FUENTE FIABLE citable? SÍ — el matiz era otro

Aclaración de Romina: la pregunta del tutor no es si alguien lo evalúa con clasificadores,
sino si otros trabajos lo usan como fuente confiable. Respuesta: **sí, en tres niveles.**

1. **Papers académicos lo usan como fuente de conocimiento:** dAPTaset (Laurenza &
   Lazzeretti, workshop de ESORICS, Springer 2020) integra clusters de misp-galaxy para
   construir su base de APTs; el paper de inferencia de threat actors (arXiv 2103.02301,
   2021) usa el cluster Threat Actor del galaxy como base de conocimiento. Y RansomLook
   (fuente del galaxy) es fuente de datos en un paper revisado por pares (RDBAlert, MDPI
   Electronics 2025) y en Müller & Yannikos (arXiv, 2026).
2. **Respaldo institucional del proyecto MISP:** paper fundacional revisado por pares
   (Wagner et al., «MISP — The Design and Implementation of a Collaborative Threat
   Intelligence Sharing Platform», ACM WISCS 2016); mantenido por CIRCL (el CERT de
   Luxemburgo); usado por CERTs nacionales, CERT-EU, sector financiero y militar — las
   instancias de CIRCL superan las 1.200 organizaciones y 4.000 usuarios. El galaxy es la
   base de conocimiento oficial del proyecto (misp-galaxy.org), versionada en GitHub.
3. **Cómo citarlo bien en la tesis:** «MISP Project, galaxy Ransomware», con versión y fecha
   de la copia (v172, copia 2026-08-20), citando además la fuente primaria de cada dato
   cuando exista (p. ej. Amigo-A). Matiz a declarar: es comunitario y contiene entradas de
   imitadores — por eso el mapeo familia→entrada se hace a mano.

**Consecuencia para el pipeline:** no existen números publicados sobre esta base contra los
cuales compararse — el catálogo aporta METADATOS citables (nombres reales de nota, extensiones
por familia, alias), que es el rol que ya se le dio. Las comparaciones válidas del pipeline
siguen siendo: Lemmou et al. 2021 (notas; su F 0,920 es la binaria de nombres), ID Ransomware
(evaluación propia en Pruebas.xlsx) y NapierOne (frente de archivos). Que nadie haya explotado
el catálogo como fuente de features con procedencia es, además, espacio de novedad para D.2.

## 📌 PREDICCIONES PREREGISTRADAS — experimento de embeddings multilingües (Exp. 3e)

Escritas el 2026-08-20, ANTES de correr, sobre la lista de familias multi-idioma verificada
nota por nota en B.3-155. Mecanismo que se pone a prueba: la representación actual (n-gramas
TF-IDF) es ciega al significado entre idiomas y parafraseo; un embedding multilingüe no.

**Diseño:** misma partición y protocolo que la base (P2 sobre 155 notas / 106 plantillas /
30 familias, 10 semillas, carpeta de salida nueva). Única variable: la representación
(embedding multilingüe en lugar de / además de TF-IDF). Base a batir: macro-F1 0,530 ± 0,062.
> **ENMIENDA 2026-08-20, previa a la ejecución (sin resultados a la vista):** el 0,530 ±
> 0,062 citado arriba corresponde a grupos+combinado+**Regresión Logística** — error de
> quien preregistró al tomar «la mejor config» de la corrida 155. El modelo canónico del
> frente es **LinearSVC**, y «única variable: la representación» exige compararlo contra
> sí mismo. **Base a batir corregida: macro-F1 0,5265 ± 0,0490 (LinearSVC, combinado,
> P2 sobre 155)**, y el control TF-IDF del experimento debe reproducirla dentro del mismo
> proceso antes de evaluar los embeddings.

**Predicciones:**
1. **Suben las tres familias multi-idioma verificadas: JIGSAW (en+de+fr), CHIMERA (en+de) y
   GANDCRAB (en+fr)** — su F1 por familia en P2 mejora respecto de la base 155.
2. **RYUK NO se mueve** (control negativo): su cohesión baja viene de fragmentos ultra-cortos
   solo-IOC, no de idiomas — un embedding no puede inventar contenido donde no lo hay.
3. El macro-F1 global: sin predicción direccional fuerte — la mejora de 3 familias sobre 30
   puede quedar dentro del ruido global aunque el mecanismo funcione. Por eso el criterio
   principal es por familia, no global.

**Criterio de adopción (fijado a priori):** se adopta la representación si (a) las tres
familias nombradas suben su F1 por familia con Δ pareado positivo sobre las mismas semillas,
(b) RYUK no sube de forma comparable, y (c) el macro-F1 global no EMPEORA (Δ pareado, IC 95 %
no enteramente negativo). Si suben las tres Y TAMBIÉN RYUK, la mejora es genérica y el
mecanismo NO queda probado — se reporta igual, con esa lectura.

**Lectura si falla:** si las multi-idioma no suben, la variación entre sus plantillas es de
contenido y no de superficie ⇒ el techo del corpus queda confirmado también contra
representaciones semánticas, y se documenta como cierre del frente de método.

## 📌 RESULTADO Exp. 3e — EMBEDDINGS MULTILINGÜES (2026-08-21, local, CPU)

> **Base declarada: 155 notas · 106 plantillas · 30 familias.** Prueba fuera de muestra de las
> predicciones preregistradas de arriba (escritas antes de correr). **Veredicto: NO ADOPTAR
> ninguna de las dos variantes** — (a) embedding solo empeora; (b) concatenado da una **mejora
> genérica con el mecanismo NO probado** (sube también el control negativo). Código:
> `2_codigo/experimento_embeddings.py` + `2_codigo/construir_embeddings_notas.py` (commits
> `d5ad3d5` y `1ebb5ff` en develop, sin coautoría). Salidas en
> `4_resultados/resultados_embeddings_155/` (**no se tocó `resultados_canonicos/` ni
> `resultados_extension_155/`**).

> ⚠️ **DOS CORRECCIONES METODOLÓGICAS aplicadas antes de la corrida definitiva** (pedido del
> tutor/Romina; ambas al manifiesto). Una corrida preliminar SIN estas correcciones quedó
> **superada** — no citar sus cifras:
> 1. **Truncado por `max_seq_length`.** El modelo corta a 128 tokens y **133 de 155 notas
>    (86 %) superan la ventana** (mediana 346 tokens, máx. 12023). Sin corregir, cada nota
>    larga se representaba solo por su comienzo. **Solución: trocear cada nota en ventanas de
>    126 tokens no solapadas, embeber cada trozo y promediar (mean pooling) + re-normalizar L2.**
>    El efecto fue real: JIGSAW en la variante (a) pasó de **−0,169 (truncado) a +0,021 (con
>    troceado)** — el «baja» preliminar era en parte «no vio el texto».
> 2. **Escala en la concatenación.** TF-IDF combinado son ~10⁴ dims ralas y el embedding 384
>    densas; concatenados crudos el embedding queda aplastado. **Solución: L2 por bloque antes
>    de concatenar** (cada bloque a norma 1).

**Modelo (dependencia a declarar en metodología):**
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, dim 384, `max_seq_length=128`.
Embeddings por **troceado + mean pooling** (ventana 126 tokens, no solapada; 133/155 notas
troceadas; máx. 96 trozos) y re-normalizados L2. Versiones: `sentence-transformers 6.0.0`,
`torch 2.13.0+cpu`, `transformers 5.15.1`, `scikit-learn 1.6.1`, Python 3.11. El modelo es
preentrenado y FIJO (no se ajusta con los datos ⇒ sin fuga aunque se calcule sobre todo el
corpus). Los embeddings se computan en un proceso aparte (torch importado primero) para evitar
un conflicto de DLL torch/MKL en Windows; no afecta las cifras.

**Diseño ejecutado tal cual:** mismo protocolo P2 (`grupos`, `StratifiedGroupKFold` 2 pliegues),
**mismas 10 semillas**, **mismo agrupamiento de casi-duplicados** (coseno char 3-5 > 0,90 →
106 plantillas) y **mismo clasificador LinearSVC(C=1, class_weight=balanced)**. Única variable:
la representación (variante (b): L2 por bloque antes de concatenar). **Control de partición: la
base TF-IDF recomputada en el mismo proceso da macro-F1 0,5265 ± 0,0490 = idéntica a la
almacenada (dif 0,0000)** ⇒ las particiones son las mismas y el Δ pareado por semilla es válido.

> ⚠️ **Reconciliación del número base.** La preregistración citó «0,530 ± 0,062»; esa cifra es
> la fila **Regresión Logística** (`grupos+combinado+LogReg` = 0,5300 ± 0,0617). El diseño exige
> «única variable = la representación», así que el modelo se mantiene fijo en **LinearSVC** (el
> canónico del frente de notas) y la base a batir es **macro-F1 0,5265 ± 0,0490**
> (`grupos+combinado+LinearSVC`, `resultados_extension_155/corrida_canonica_resumen.csv`). No es
> improvisación: es la única lectura que respeta «mismo protocolo, única variable».

### (1) F1 por familia — las 4 preregistradas (macro-métrica: F1 por familia, media ± desvío sobre 10 semillas; base 155)

Δ = variante − base, **pareado por semilla**. IC 95 % del Δ (t de Student, df=9). «sem+» = semillas con Δ>0.
Cifras corregidas (troceado + L2 por bloque).

**Variante (a) — embedding EN LUGAR de TF-IDF:**
| Familia | Predicción | F1 base | F1 variante | Δ pareado | IC 95 % | sem+ | Veredicto |
|---|---|---|---|---|---|---|---|
| CHIMERA (en+de) | sube | 0,553 ± 0,249 | **0,822 ± 0,306** | **+0,269** | [+0,152; +0,385] | 9/10 | ✅ SUBE fuerte |
| JIGSAW (en+de+fr) | sube | 0,282 ± 0,321 | 0,303 ± 0,319 | +0,021 | [−0,222; +0,263] | 2/10 | ⚠️ plano (ruido) |
| GANDCRAB (en+fr) | sube | 0,670 ± 0,320 | 0,456 ± 0,236 | **−0,215** | [−0,429; −0,000] | 2/10 | ❌ BAJA |
| RYUK (control −) | no se mueve | 0,260 ± 0,266 | 0,318 ± 0,297 | +0,058 | [−0,030; +0,146] | 4/10 | ✓ no sube signif. |

**Variante (b) — embedding CONCATENADO con TF-IDF (L2 por bloque):**
| Familia | Predicción | F1 base | F1 variante | Δ pareado | IC 95 % | sem+ | Veredicto |
|---|---|---|---|---|---|---|---|
| CHIMERA (en+de) | sube | 0,553 ± 0,249 | 0,612 ± 0,261 | +0,059 | [−0,050; +0,168] | 3/10 | ⚠️ sube (n.s.) |
| JIGSAW (en+de+fr) | sube | 0,282 ± 0,321 | 0,332 ± 0,331 | +0,050 | [−0,103; +0,203] | 3/10 | ⚠️ sube (n.s.) |
| GANDCRAB (en+fr) | sube | 0,670 ± 0,320 | 0,712 ± 0,274 | +0,042 | [−0,061; +0,144] | 4/10 | ⚠️ sube (n.s.) |
| RYUK (control −) | no se mueve | 0,260 ± 0,266 | **0,388 ± 0,240** | **+0,128** | [+0,055; +0,200] | 8/10 | ⛔ SUBE MÁS que las 3 |

### (2) macro-F1 global (30 familias, azar 0,033) — Δ pareado e IC 95 %, base 155

| Variante | macro-F1 base | macro-F1 variante | Δ pareado | IC 95 % | sem+ |
|---|---|---|---|---|---|
| (a) embedding en lugar de TF-IDF | 0,5265 ± 0,0490 | **0,4532 ± 0,0436** | **−0,0733** | [−0,1073; −0,0393] | 0/10 |
| (b) embedding concatenado (L2/bloque) | 0,5265 ± 0,0490 | **0,5358 ± 0,0406** | **+0,0092** | [−0,0098; +0,0283] | 7/10 |

- **(a) empeora de forma significativa** (0/10 semillas mejoran, IC 95 % enteramente negativo):
  el embedding solo pierde ~0,07 de macro-F1 contra TF-IDF ⇒ la señal de superficie (n-gramas)
  es la que sostiene el desempeño global; el multilingüe chico no la reemplaza. (Con el troceado
  la caída bajó de −0,1145 a −0,0733: ver el texto completo ayudó, pero no alcanza.)
- **(b) es plano:** Δ +0,0092, IC 95 % **incluye 0** ⇒ no mejora de forma significativa, aunque
  7/10 semillas queden del lado positivo.

### (3) Veredicto contra el criterio de adopción preregistrado, por variante

- **(a) embedding en lugar de TF-IDF → NO ADOPTAR.** Falla (a): de las tres multi-idioma sube
  fuerte solo CHIMERA (+0,269, 9/10); JIGSAW queda plano (+0,021, IC incluye 0) y **GANDCRAB
  baja** (−0,215). Falla (c): el macro-F1 global empeora con IC 95 % enteramente negativo.
- **(b) embedding concatenado → NO ADOPTAR, por «mejora genérica / mecanismo NO probado».**
  Se cumple (a) en signo —las tres suben algo (CHIMERA +0,059, JIGSAW +0,050, GANDCRAB +0,042),
  aunque **ninguna es significativa** (los tres IC 95 % incluyen 0)— y se cumple (c) —el global
  no empeora—. **Pero RYUK (control negativo) sube +0,128 (8/10, IC 95 % enteramente positivo),
  MÁS que cualquiera de las tres multi-idioma.** Es exactamente el caso que la preregistración
  fijó a priori: «si suben las tres Y TAMBIÉN RYUK, la mejora es genérica y el mecanismo NO queda
  probado». El efecto de (b) es una ganancia difusa de agregar features densas —que ayuda a
  cualquier familia, y a RYUK más que a nadie—, **no** el cierre de brechas entre idiomas.

### Lectura honesta

**El único efecto atribuible al mecanismo lingüístico es CHIMERA en la variante (a)** (+0,269,
9/10 semillas): sus plantillas son **el mismo mensaje traducido literalmente** (en+de) y el
embedding multilingüe las acerca — justo lo predicho. **No generaliza:**
- **GANDCRAB baja en (a)** pese a tener una plantilla francesa: su bloque francés es una nota
  larga y distinta, y al reemplazar TF-IDF por el embedding se pierde la señal de superficie que
  la venía identificando bien (base 0,670).
- **JIGSAW no se mueve** de forma clara: su baja cohesión es de contenido, no solo de idioma —
  sus dos plantillas inglesas ya son mensajes distintos («Your computer files have been
  encrypted» vs «I want to play a game with you»), verificado en B.3-155. Un embedding no
  inventa el contenido que falta.
- **En (b) suben las tres pero RYUK sube más**, y RYUK no tiene componente de idioma (control):
  la suba es genérica.

> **Conclusión para la tesis:** cambiar la representación por un embedding multilingüe chico
> **no supera la base TF-IDF** en el frente de notas —variante (a) peor de forma significativa,
> variante (b) estadísticamente indistinguible y con el mecanismo no probado (sube el control)—.
> El único efecto lingüístico limpio es CHIMERA, insuficiente para adoptar. **Refuerza el
> argumento central: el techo lo pone el dato (la diversidad de contenido del corpus), no la
> representación** — ahora también contra representaciones semánticas, no solo contra ajuste de
> hiperparámetros. Cierre del frente de método en notas; ampliar corpus (contenido nuevo) sigue
> siendo la única palanca con evidencia. **Y valida el control del truncado:** sin trocear, el
> «baja» de JIGSAW era artefacto de no ver el texto (−0,169 → +0,021).
>
> *(Todas las cifras: F1 por familia y macro-F1 sobre 30 familias, P2 = protocolo `grupos`,
> LinearSVC, 10 semillas, base 155 notas / 106 plantillas. Δ pareado por semilla. Azar
> macro-F1 0,033. Embedding: troceado ventana 126 + mean pooling; variante (b): L2 por bloque.)*


## ★ FUENTE NUEVA DEL TUTOR (2026-08-20) — catálogo MISP «Ransomware»

El tutor envió el JSON del galaxy «Ransomware» del proyecto MISP (2.135 entradas; fuentes:
MISP Project, id-ransomware.blogspot/Amigo-A, ransomlook.io). Guardado en
`3_datos/misp_ransomware_galaxy/misp_galaxy_ransomware_2026-08-20.json` (fuera del repo).
**Las 30 familias del núcleo tienen entrada.** Qué aporta, en orden de valor:

1. **Extensiones documentadas por familia → evidencia citable de campaña-vs-familia.** Hoy la
   afirmación «la extensión identifica la campaña, no la familia» se sostiene con razonamiento
   y el ejemplo de DHARMA. MISP la convierte en tabla: **DHARMA 21 extensiones documentadas,
   MEDUZALOCKER 36, JIGSAW 19, CLOP 8** — contra la ÚNICA extensión por familia que muestra
   NapierOne (una campaña). Refuerza el Exp. 2b y el diseño del 2d con fuente externa.
2. **`ransomnotes-filenames` (298 entradas) → nombres REALES de nota de rescate.** Era el
   bloqueo de la señal 1: los nombres de ThreatLabz son etiquetas del curador. Acá hay nombres
   documentados (DHARMA: `README.txt`, `Info.hta`, `FILES ENCRYPTED.txt`…). Habilita la
   feature de nombre con procedencia y alimenta el Exp. 2d.
3. **`ransomnotes` (115 entradas) → textos de notas** con procedencia citable — posible fuente
   de plantillas para las familias que siguen bajas (pasar SIEMPRE por el verificador de
   casi-duplicados antes de contarlas).
4. **`synonyms` (207 entradas) → alias con fuente** para el filtro de circularidad (hoy la
   lista de alias es artesanal) y, crítico para la extensión a familias nuevas: **deduplicar
   por alias antes de sumar una familia** (no agregar «REvil» y «Sodinokibi» como dos).
5. **`date`** → año por familia con fuente adicional a `Pruebas.xlsx`.

⚠️ Advertencias: es un catálogo comunitario, con entradas de imitadores y calidad despareja
(«CerberTear» no es CERBER; hay entradas «(Fake)»). El mapeo familia→entrada hay que hacerlo
**a mano, una por una** — el emparejamiento automático rápido que se usó para el censo inicial
agarró imitadores en al menos CERBER y CRYPTOLOCKER. Tarea corta, pendiente. Citar como
«MISP Project, galaxy Ransomware» con la fecha de la copia.

## 📌 PREDICCIONES PREREGISTRADAS — antes de re-medir sobre 155 notas (2026-08-20)

Escritas ANTES de correr la medición sobre el corpus de 155 notas, para que la corrida sea una
prueba de B.1 fuera de muestra y no una lectura a posteriori. Base de la predicción: el Δ del
paso 3→4 plantillas medido por B.1 (macro-F1 +0,0297, IC 95 % [+0,0039; +0,0554], pareado).

1. **F1 por familia de WANNACRY y RYUK sube fuerte** (hoy 0,01 ± 0,10 y 0,03 ± 0,16 en P2ret;
   pasaron de 2 a 4 textos distintos).
2. **NOTPETYA, CRYPTOLOCKER y WASTEDLOCKER no se mueven** (no se les agregó nada; techo
   confirmado en la sesión de recolección).
3. **El macro-F1 global de P2 sube de forma modesta**, del orden del Δ de B.1 diluido por las
   familias sin cambio.

Reglas de la corrida: mismo código y configuración que la canónica, 10 semillas, **carpeta de
salida NUEVA** (no pisar `resultados_canonicos/`), base declarada «155 notas, 30 familias».
Después de esta corrida y NO antes: mejoras de features de a una (primera: vista de marcadores
por forma, C.bis), cada una contra esta base con criterio de adopción fijado a priori
(IC 95 % del Δ pareado excluye 0). La ampliación a familias nuevas va al final, con el método
congelado.

## 📌 RESULTADO DE LA RE-MEDICIÓN SOBRE 155 NOTAS (2026-08-20) — EXTENSIÓN QUE SE AGREGA

Prueba fuera de muestra de las predicciones preregistradas de arriba. **Base declarada:
155 notas · 30 familias · 106 plantillas** (casi-dup coseno char 3-5 > 0,90). Corpus
verificado 1:1 contra `manifiesto_corpus_v2.csv` (155 filas; 0 discrepancias en ambos
sentidos; extracción 128 texto + 27 html; ninguna nota omitida). Salidas en
`4_resultados/resultados_extension_155/` (**NO se tocó `resultados_canonicos/`**). Mismo
código y configuración que la canónica (10 semillas P1/P2; R=100 P2ret). El control interno
de la curva pasó: k=todo reprodujo el evaluador canónico sobre 155 (aborta si no coincide).

> **Las cifras canónicas de 146/144 NO se corrigen: tienen otra base. Esto es una medición
> aparte, sobre 155.** Código en commit `a914e33` (parámetro `--salida`, sin cambios de método).

### Cifras globales (155 vs canónica)
| Métrica | 155 notas | Canónica | Δ |
|---|---|---|---|
| P2 macro-F1 (grupos, combinado+LinearSVC) | **0,5265 ± 0,0490** | 0,435 ± 0,057 (146) · 0,4210 (144) | +0,092 · +0,106 |
| P1 macro-F1 (estratificado, caracteres+LinearSVC) | **0,7955 ± 0,0194** | 0,760 ± 0,029 (146) · 0,7501 (144) | +0,036 |
| P2ret macro-F1 (30fam, k=todo, R=100) | **0,7612 ± 0,0597** | 0,6164 ± 0,0450 (144) | +0,1448 |

Curva P2 (30fam, plantillas) sobre 155 **satura en k≈2**: deltas 1→2 +0,0476 (sig), 2→3
+0,0109 (no sig), 3→4 −0,0040 (no sig). El paso 3→4 que en 144 era +0,0297 (sig) ahora es
nulo. **El aporte vino del NIVEL (familias que dejaron de dar 0), no de seguir trepando la
curva.**

### Contraste de las TRES predicciones (per-familia: 30fam · P2ret · plantillas · k=todo, R=100)

| Familia | F1 144 | F1 155 | Δ | Predicción | Veredicto |
|---|---|---|---|---|---|
| WANNACRY | 0,0100 ± 0,100 | 0,7333 ± 0,395 | +0,723 | (a) sube fuerte | ✅ CUMPLE (fuerte) |
| RYUK | 0,0324 ± 0,160 | 0,2706 ± 0,419 | +0,238 | (a) sube fuerte | ⚠️ PARCIAL: sube, pero solo a 0,27 ± 0,42 |
| NOTPETYA | 0,6950 ± 0,114 | 0,9367 ± 0,131 | +0,242 | (b) no se mueve | ❌ FALLA: subió +0,24 |
| CRYPTOLOCKER | 0,4100 ± 0,456 | 0,0967 ± 0,269 | −0,313 | (b) no se mueve | ❌ FALLA: bajó −0,31 |
| WASTEDLOCKER | 0,0000 ± 0,000 | 0,9874 ± 0,073 | +0,987 | (b) no se mueve | ❌ FALLA: subió +0,99 |

(Confirmado que la base per-familia es la correcta: los valores 144 de WANNACRY 0,0100 y RYUK
0,0324 coinciden con los citados en la preregistración.)

**(a) — CUMPLE para WANNACRY (fuerte), PARCIAL para RYUK.** WANNACRY 0,01→0,73 confirma el
efecto del paso 2→4 plantillas. RYUK subió +0,24 pero solo a 0,27 ± 0,42: dirección correcta,
magnitud débil y varianza enorme (sigue fallando la mayoría de las repeticiones). No suavizar
a «cumple»: es media victoria.

**(b) — FALLA en las TRES, cada una por un mecanismo distinto. Este es el hallazgo:**
- **WASTEDLOCKER 0,000 → 0,987 sin agregar una sola nota.** Descolapso del reagrupamiento: las
  4 notas que en 144 formaban 1 plantilla (unidas por transitividad, coseno mínimo 0,894) se
  parten en **3 plantillas** en 155 porque el IDF se reajusta sobre el corpus más grande. Con
  ≥2 plantillas ya es evaluable en P2ret, y como las 3 quedan a ~0,89 entre sí, la plantilla
  retenida se parece mucho a las de entrenamiento → F1 0,987. **Artefacto de partición, NO
  aprendizaje. Anticipado antes de correr.**
- **CRYPTOLOCKER 0,410 → 0,097.** Se le **quitó** la nota mal etiquetada (`lm_Crypt0l0cker` =
  TorrentLocker, retirada el 2026-08-19). 4→3 notas, 3→2 plantillas: cambió su composición. No
  es «no se le agregó nada», es «se le sacó algo». Con 2 plantillas queda frágil en P2ret.
- **NOTPETYA 0,695 → 0,937 sin agregar ni quitar nada (sigue 2 notas / 2 plantillas).** El F1
  por familia en multiclase depende del corpus entero (frontera, IDF, clases negativas): al
  crecer las demás familias y afinarse la frontera, NOTPETYA se volvió más separable. **Es la
  refutación más limpia del supuesto de (b): «sin material nuevo ⇒ el F1 no cambia» es falso.**

**(c) — FALLA por lo alto.** Predijo subida «modesta, del orden del Δ 3→4 de B.1 (+0,0297)
diluido». El P2 canónico subió **+0,092** (0,435→0,5265), ~3× lo predicho; el P2ret macro
+0,1448. Mecanismo: la predicción supuso que solo WANNACRY/RYUK moverían la aguja, pero
**cinco** familias que en 144 daban ~0,000 saltaron en 155 —WANNACRY, CHIMERA (0→0,912),
MAZE (0→0,561), MEDUZALOCKER (0→0,608), WASTEDLOCKER (0→0,987)—, cuatro porque la recolección
les agregó textos distintos (dejaron de ser monoplantilla) y WASTEDLOCKER por el artefacto de
reagrupamiento; más corrimientos indirectos de frontera (NOTPETYA +0,24, CERBER 0,869→0,999).
El macro-F1 promedia 30 familias por igual, así que cinco familias pasando de 0 a 0,56–0,99
domina. CRYPTOLOCKER (−0,31) compensa en parte.

### Lección metodológica (para el capítulo, con base declarada)
El **F1 por familia bajo P2/P2ret no es estable frente a cambios del corpus**, ni siquiera para
familias sin material nuevo, por tres vías: (i) el reagrupamiento de casi-duplicados se
recalcula global (IDF sobre todo el corpus) y familias al borde del umbral 0,90 se descolapsan
(WASTEDLOCKER); (ii) retiros de notas cambian la composición (CRYPTOLOCKER); (iii) la frontera
multiclase depende del corpus entero (NOTPETYA). Comparar F1 por familia entre versiones del
corpus exige declararlo. **No se tocó el método** (decisión del pedido): se reporta el efecto
con su mecanismo. Nota lateral que confirma una anticipación previa: **JIGSAW bajó 0,670→0,505**
al sumarle traducciones — coherente con el choque idioma-vs-cohesión ya documentado.

### Pendiente (Paso 2, en otro chat)
Sobre ESTA base de 155: mejoras de features de a una (primera C.bis, vista de marcadores por
forma), cada una con criterio de adopción a priori (IC 95 % del Δ pareado excluye 0).
La ampliación a familias nuevas de ThreatLabz sigue pendiente de decisión con el tutor.

## ★ DISEÑO DE LA EXTENSIÓN COMO VALIDACIÓN (2026-08-21) — la cohesión predice, se preregistra

Aclaración de propósito, tras la pregunta de Romina («¿cómo agregar familias si las que
tenemos fallan?»): **la extensión no persigue subir el número — mide si el método escala, y
el macro-F1 sobre más familias VA A DAR MÁS BAJO que sobre 30 (más clases = problema más
difícil). Decírselo a Cappo de entrada.** El entregable es la curva macro-F1 vs cantidad de
familias y el análisis por familia, no una cifra mayor.

**Diseño acordado — la extensión como validación fuera de muestra de B.3:**
1. Para cada familia candidata (ThreatLabz, deduplicada por alias con MISP), medir su
   **cohesión** antes de clasificar.
2. **Predecir por escrito** cuáles rendirán (cohesión alta) y cuáles no, usando la
   correlación de B.3-155 (Spearman ρ +0,69 entre cohesión y F1 por familia).
3. Correr y contrastar predicción contra resultado.

Si acierta sobre familias que no participaron en derivar la correlación, ρ +0,69 pasa de
observación a regla validada — el hallazgo más fuerte del frente. Si falla, se reporta que
la correlación era específica del corpus. Umbral de admisión vigente: ≥ 2 plantillas para
ser evaluable, ~4 para rendir; familias de 1 nota no entran.

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

## 📌 PREDICCIONES PREREGISTRADAS — M.1 CASCADA IOC→TEXTO (2026-08-22)

Escritas **ANTES de implementar y correr** el experimento, calculadas únicamente desde el
grafo B.3-155 ya publicado (`4_resultados/resultados_grafo_marcadores_155/`, CSV
`b3_aristas.csv` + `b3_marcadores_por_plantilla.csv`). Es el espejo del Exp. 2b en el frente
de notas: **regla exacta donde aplica + clasificador de texto como respaldo**.

**Mecanismo medido que la sostiene (B.3-155):** tras filtrar infraestructura común, **ningún
IOC operativo se comparte entre familias distintas** — las 5 aristas entre familias del grafo
son URLs de torproject, no marcadores de campaña. Por lo tanto un IOC ya visto identifica la
familia casi sin error, pero solo cubre las notas que reutilizan infraestructura conocida.

**Base declarada:** 155 notas · 106 plantillas (108 nodos familia#plantilla) · 30 familias.
Protocolo P2 (`grupos`, StratifiedGroupKFold 2 pliegues), 10 semillas, LinearSVC combinado
como capa de respaldo. Base a batir: **macro-F1 0,5265 ± 0,0490** (LinearSVC, combinado, P2
sobre 155; `resultados_extension_155/resultados_canonicos/corrida_canonica_resumen.csv`).

### (a) Cobertura esperada de la regla

Derivada del grafo, en dos niveles. **Cota del grafo** = fracción que comparte al menos un IOC
con otra plantilla de su familia (supone que el vecino siempre está en entrenamiento).
**Estimación bajo 2 pliegues** = la anterior corregida por la probabilidad hipergeométrica de
que al menos uno de los k vecinos caiga en el pliegue de entrenamiento,
`1 − C(g−1,k)/C(n−1,k)` promediada sobre los dos tamaños de pliegue de la familia.

| Variante | Cota del grafo (plantillas) | Cota del grafo (notas) | **Cobertura esperada, 2 pliegues** |
|---|---|---|---|
| **Sin** filtro de circularidad | 61/108 = 0,565 | 99/155 = 0,639 | **0,584** |
| **Con** filtro de circularidad | 53/108 = 0,491 | 89/155 = 0,574 | **0,530** |

El filtro de circularidad cuesta ~0,05 de cobertura: saca 74 valores que contienen el nombre
o alias de la familia, y deja **LOCKBIT y CHIMERA en cobertura 0,000** (LOCKBIT pasa de 4 notas
cubiertas a 0 — es la concentración ya documentada en B.3).

**Dos razones por las que la cobertura medida puede quedar por DEBAJO de la estimación, ya
declaradas:** (1) el grafo agrega los IOCs de *todas* las notas de una plantilla, mientras que
la cascada decide **nota por nota** — una nota concreta puede no llevar el IOC que une a su
plantilla con la vecina; (2) el diccionario se construye solo con el pliegue de entrenamiento.
Y una razón por la que puede quedar por ENCIMA: los IOCs compartidos con OTRA familia
(2 plantillas / 3 notas sin filtro, 4 plantillas / 6 notas con filtro) también disparan la
regla, pero cuando lo hacen **acierta poco o nada**.

### (b) Acierto donde aplica

**Cercano a 1: se predice ≥ 0,95.** Es consecuencia directa del mecanismo de B.3 (IOCs
privados de cada familia). No se predice exactamente 1,000: la vía de error identificada de
antemano son las **URLs de infraestructura común** (torproject y similares) que en un pliegue
de entrenamiento aparecen bajo una sola familia y en prueba aparecen en otra. La regla de
unanimidad (todas las coincidencias apuntan a UNA familia) filtra el conflicto visible, no este
caso. Si el acierto medido cae por debajo de 0,90, el filtro de genéricos de B.3
(`MAX_FAMILIAS_VALOR`) pasa a ser necesario y se declarará como hallazgo.

### (c) Familias que más deberían beneficiarse

Las de `pares_unidos_por_marcador` alto en `b3_cohesion_por_familia.csv`, cruzadas con su
cobertura y su F1 de partida (F1 base = `corrida_canonica_por_familia.csv` de
`resultados_extension_155`, P2, 10 semillas):

| Familia | frac. pares unidos | Cobertura sin filtro | F1 base | Margen |
|---|---|---|---|---|
| **BLACKBASTA** | 0,667 | 1,000 | **0,122** | el mayor margen de las 30 |
| **CLOP** | 0,500 | 0,750 | 0,316 | alto |
| **BLACKMATTER** | 1,000 | 1,000 | 0,413 | alto |
| **SODINOKIBI** | 0,333 | 1,000 | 0,474 | medio |
| **LORENZ** | 1,000 | 1,000 | 0,550 | medio |
| **NOTPETYA** | 1,000 | 1,000 | 0,581 | medio |
| **PHOBOS** | 0,500 | 0,750 | 0,612 | medio |
| **GANDCRAB** | 1,000 | 1,000 | 0,681 | bajo (ya alto) |
| **NETWALKER** | 1,000 | 1,000 | 0,700 | bajo (ya alto) |
| **TESLACRYPT** | 1,000 | 1,000 | 0,789 | bajo (ya alto) |

**Controles negativos preregistrados: RYUK y HELLOKITTY.** Las dos tienen
`pares_unidos_por_marcador = 0` y **cobertura 0,000** en las dos variantes, con F1 base bajo
(RYUK 0,225 · HELLOKITTY 0,137). **No deben moverse.** Si suben, la ganancia no viene del
mecanismo de IOCs y el experimento no prueba lo que dice probar. *(HELLOKITTY tiene 5
plantillas y solo 2 marcadores en total; RYUK tiene 9 marcadores pero ninguno repetido entre
sus plantillas.)*

**BLACKBASTA es el caso interesante y hay que leerlo con cuidado:** comparte plantilla de texto
con CONTI (grupo 6 de casi-duplicados) pero **no comparte IOCs** con ella, así que es
exactamente la familia que la cascada debería rescatar — y es la predicción que conecta M.1
con M.4.

### (d) Criterio de adopción (fijado a priori)

**Se adopta la cascada si el macro-F1 del combinado supera la base con Δ pareado por semilla
cuyo IC 95 % (t de Student, df=9) excluye el cero.** Las dos variantes (con y sin filtro de
circularidad) se juzgan por separado con el mismo criterio.

Condiciones de lectura, también a priori:
- Si el macro-F1 sube **y** suben las familias de (c) **sin** que se muevan RYUK/HELLOKITTY →
  el mecanismo queda probado.
- Si sube el macro-F1 pero **también** suben los controles negativos → mejora genérica,
  mecanismo NO probado (misma regla que cerró el Exp. 3e).
- Si la cobertura medida queda muy por debajo de la estimación de (a) → se reporta la cascada
  como **regla de alta precisión y baja cobertura**, igual que las firmas binarias del Exp. 2b,
  y se declara que no mueve el agregado.
- **Puerta de entrada:** la capa de texto sola debe reproducir macro-F1 0,5265 ± 0,0490. Si no
  reproduce, la corrida se detiene y no se reporta ninguna cifra.

Salidas en carpeta NUEVA: `4_resultados/resultados_cascada_155/`. No se toca
`resultados_canonicos/`, `resultados_extension_155/` ni el capítulo 4.

## 📌 RESULTADO M.1 — CASCADA IOC→TEXTO (2026-08-22, local)

> **Base declarada: 155 notas · 106 plantillas · 30 familias.** Prueba fuera de muestra de las
> predicciones preregistradas del bloque anterior (escritas antes de implementar y correr).
> **Veredicto: la variante SIN filtro de circularidad SE ADOPTA por el criterio preregistrado;
> la variante CON filtro NO.** Código: `2_codigo/cascada_ioc_notas.py` (commit `3473c40` en
> develop, sin coautoría). Salidas en `4_resultados/resultados_cascada_155/` — **no se tocó
> `resultados_canonicos/`, `resultados_extension_155/` ni el capítulo 4**.

**Puerta de entrada superada exactamente:** la capa de texto sola dio **macro-F1 0,5265 ±
0,0490**, diferencia **0,0000** contra la base almacenada (LinearSVC, combinado, P2 sobre 155).
El script aborta si no coincide dentro de 0,003, así que la partición es la misma y el Δ
pareado por semilla es válido, no una comparación entre corridas distintas.

**Qué se corrió:** IOCs extraídos **solo del pliegue de entrenamiento** con los patrones
canónicos de `normalizacion_marcadores.PATRONES` (los mismos del Sprint 1.1 y de B.3, no un
criterio nuevo); diccionario valor→familia; en prueba, si la nota contiene un IOC del
diccionario y **todas** sus coincidencias apuntan a UNA familia se le asigna, y el conflicto o
la ausencia caen al LinearSVC combinado entrenado en ese mismo pliegue. P2 (`grupos`,
StratifiedGroupKFold 2 pliegues), mismas 10 semillas. En el corpus hay **265 valores de IOC
distintos**; 18 de las 155 notas no tienen ningún IOC.

### (1) Las tres columnas del Exp. 2b (media ± desvío, 10 semillas, base 155)

| Variante | Cobertura de la regla | Acierto donde aplica | macro-F1 combinado | Exactitud |
|---|---|---|---|---|
| texto solo (base) | — | — | 0,5265 ± 0,0490 | 0,6026 |
| **sin filtro de circularidad** | **0,2613 ± 0,0264** (40,5 de 155 notas) | **0,9814 ± 0,0241** | **0,5453 ± 0,0485** | **0,6348** |
| **con filtro de circularidad** | 0,2419 ± 0,0287 (37,5 notas) | 0,9719 ± 0,0370 | 0,5323 ± 0,0486 | 0,6239 |

**Δ pareado por semilla contra la base, IC 95 % (t de Student, df=9):**

| Variante | Δ macro-F1 | IC 95 % | semillas con Δ>0 | Δ exactitud | IC 95 % |
|---|---|---|---|---|---|
| **sin circularidad** | **+0,0188** | **[+0,0128; +0,0248]** | **10/10** | **+0,0323** | [+0,0236; +0,0410] |
| con circularidad | +0,0058 | [−0,0013; +0,0130] | 6/10 | +0,0213 | [+0,0120; +0,0305] |

Exactitud balanceada: base 0,5818 → 0,6012 (sin circularidad) · 0,5864 (con circularidad).

**Donde la regla aplica, gana claramente al texto:** acierto de la regla 0,9814 contra 0,8569
del clasificador de texto **sobre esas mismas notas** (+0,1244). No es que la regla cubra lo
fácil: cubre notas que el texto erraba en un 14 % de los casos.

### (2) Veredicto contra el criterio de adopción preregistrado

El criterio fijado a priori era: *se adopta si el macro-F1 combinado supera la base con Δ
pareado cuyo IC 95 % excluye el cero.*

- **SIN filtro de circularidad → SE ADOPTA.** Δ +0,0188 con IC 95 % [+0,0128; +0,0248], que
  excluye el cero, y **10/10 semillas** del lado positivo.
- **CON filtro de circularidad → NO se adopta.** Δ +0,0058 con IC 95 % [−0,0013; +0,0130], que
  **incluye el cero** (6/10 semillas). Queda como **regla de alta precisión y baja cobertura
  que no mueve el agregado** — exactamente la lectura que el preregistro dejó prevista.

**El mecanismo queda probado, y esto es lo importante:** los dos controles negativos preregistrados
**no fueron tocados por la regla en ninguna semilla** — cero notas de RYUK y cero de HELLOKITTY
recibieron asignación por IOC, tal como predecía su cobertura 0,000 en el grafo.

> ⚠️ **Matiz metodológico que hubo que agregar al leer los resultados.** HELLOKITTY mueve su F1
> +0,0097 pese a no ser tocada. No es una violación del control: el Δ de F1 de una familia mezcla
> **recall propio** (notas suyas que la regla asignó) con **precisión ajena** (notas de otras
> familias que el texto le atribuía por error y la regla reasignó). HELLOKITTY tiene **0 notas
> propias asignadas y 16 falsos positivos ajenos corregidos** (12 de DHARMA, 2 de BLACKBASTA, 2 de
> PHOBOS). RYUK tiene 0 y 0, y su Δ es exactamente 0,0000. El script ahora reporta las dos
> columnas por familia (`notas_propias_asignadas`, `falsos_positivos_ajenos_corregidos`) y juzga
> el control por «¿la regla la tocó?», no por «¿su F1 se movió?». **Sin esta descomposición el
> veredicto automático decía «mejora genérica» y habría sido falso.**

### (3) F1 por familia — preregistradas y controles (Δ pareado, IC 95 %, base 155)

Variante **sin filtro de circularidad** (la adoptada). «propias» = notas de la familia asignadas
por la regla en las 10 semillas; «ajenas» = falsos positivos del texto corregidos.

| Familia | Predicción | F1 base | F1 cascada | Δ pareado | IC 95 % | propias / ajenas | Veredicto |
|---|---|---|---|---|---|---|---|
| **BLACKBASTA** | el mayor margen | 0,179 ± 0,243 | **0,307 ± 0,222** | **+0,128** | [+0,038; +0,219] | 8 / 1 | ✅ sube, IC excluye 0 |
| **LORENZ** | sube | 0,536 ± 0,324 | 0,625 ± 0,361 | **+0,089** | [+0,021; +0,157] | 8 / 0 | ✅ sube |
| **CLOP** | sube | 0,346 ± 0,228 | 0,419 ± 0,234 | **+0,074** | [+0,006; +0,141] | 23 / 2 | ✅ sube |
| **TESLACRYPT** | sube (ya alta) | 0,789 ± 0,286 | 0,848 ± 0,306 | **+0,059** | [+0,022; +0,096] | 81 / 13 | ✅ sube |
| **PHOBOS** | sube | 0,567 ± 0,302 | 0,621 ± 0,320 | +0,054 | [−0,022; +0,130] | 21 / 8 | ⚠️ sube, n.s. |
| **SODINOKIBI** | sube | 0,427 ± 0,245 | 0,451 ± 0,270 | +0,024 | [−0,006; +0,054] | 0 / 5 | ⚠️ solo precisión ajena |
| **GANDCRAB** | sube | 0,670 ± 0,320 | 0,670 ± 0,320 | +0,000 | [0,000; 0,000] | 0 / 0 | ❌ la regla nunca aplicó |
| **NOTPETYA** | sube | 0,581 ± 0,227 | 0,581 ± 0,227 | +0,000 | [0,000; 0,000] | 18 / 0 | ➖ aplicó, no cambió nada |
| **NETWALKER** | sube | 0,686 ± 0,475 | 0,686 ± 0,475 | +0,000 | [0,000; 0,000] | 14 / 0 | ➖ aplicó, no cambió nada |
| **BLACKMATTER** | sube | 0,377 ± 0,416 | 0,377 ± 0,416 | +0,000 | [0,000; 0,000] | 5 / 0 | ➖ aplicó, no cambió nada |
| **RYUK** *(control −)* | no se mueve | 0,260 ± 0,266 | 0,260 ± 0,266 | **+0,000** | [0,000; 0,000] | **0 / 0** | ✅ control intacto |
| **HELLOKITTY** *(control −)* | no se mueve | 0,133 ± 0,127 | 0,143 ± 0,134 | +0,010 | [+0,000; +0,019] | **0 / 16** | ✅ no tocada (ver matiz) |

**Las que suben y no estaban preregistradas** (se reportan por transparencia, no como
predicción acertada): **DHARMA +0,1026** [+0,0368; +0,1684] (60 notas propias asignadas),
**CHIMERA +0,0681** [+0,0048; +0,1314], **CERBER +0,0373** [+0,0143; +0,0602].

**Ninguna familia baja de forma significativa.** Las dos que bajan tienen IC 95 % que incluye el
cero y son precisamente las víctimas de los errores de Tor (ver punto 5): **CONTI −0,0935**
[−0,2012; +0,0142] y **AVOSLOCKER −0,0333** [−0,1087; +0,0421].

**BLACKBASTA confirma la predicción y además delata su origen:** sube +0,128 sin filtro de
circularidad pero solo +0,012 (IC incluye 0) con el filtro. Su señal de IOC **es circular** —
contiene el alias `basta`. Hay que decirlo así al escribirlo; es el mismo tipo de advertencia que
B.3 dejó sobre LOCKBIT.

### (4) La cobertura quedó MUY por debajo de lo preregistrado, y la causa está identificada

Predicho 0,584 (sin circularidad) y 0,530 (con); medido **0,2613** y **0,2419**. La causa
principal no era ninguna de las dos que el preregistro anticipó:

**De las 1.550 decisiones (155 notas × 10 semillas), 362 (23,4 %) terminaron en CONFLICTO** —
la nota tenía coincidencias que apuntaban a más de una familia y por diseño cayó al texto. Y el
conflicto lo producen **4 valores de los 265 del corpus, los cuatro URLs de torproject**:

| Valor | Familias que lo usan |
|---|---|
| `https://www.torproject.org/` | BLACKBASTA, BLACKCAT, BLACKMATTER, CERBER, GANDCRAB, LOCKBIT, MAZE |
| `https://torproject.org/` | BLACKCAT, DARKSIDE, LORENZ, NETWALKER, SODINOKIBI |
| `https://www.torproject.org/download/` | AVOSLOCKER, CLOP |
| `https://torproject.org` | BLACKBASTA, CONTI |

**Los 261 valores restantes (98,5 %) son privados de una sola familia.** Es la confirmación
independiente y cuantificada del mecanismo de B.3: *ningún IOC operativo se comparte entre
familias; lo único compartido es el sitio de descarga de Tor.*

El efecto es brutal en familias concretas: **GANDCRAB tuvo 90 conflictos y 0 asignaciones** —
cobertura medida 0,000 contra 1,000 predicha por el grafo—, y lo mismo BLACKCAT (40 conflictos)
y SODINOKIBI (30). Una nota con un IOC privado que la identifica sin ambigüedad **queda bloqueada
entera** porque además menciona el enlace de descarga de Tor.

### (5) Los 8 errores de la regla son, uno por uno, las aristas de Tor que B.3 ya había documentado

Sobre 405 asignaciones en la variante sin circularidad hubo **8 errores** (acierto 0,9814):

| Nota | Familia real → asignada | Causa |
|---|---|---|
| `CONTI\conti2.txt`, `CONTI\conti3.txt` (3 semillas c/u) | CONTI → BLACKBASTA | `https://torproject.org` |
| `AVOSLOCKER\avoslocker.txt` | AVOSLOCKER → CLOP | `https://www.torproject.org/download/` |
| `CLOP\Details_Cleo.txt` | CLOP → AVOSLOCKER | la misma URL, al revés |

Son **exactamente** las aristas entre familias del grafo B.3-155 (las 4 BLACKBASTA↔CONTI y la
nueva AVOSLOCKER↔CLOP). **Validación cruzada entre B.3 y M.1:** el grafo predijo dónde iba a
fallar la regla, y falló ahí y solo ahí.

### (6) Diagnóstico POST-HOC — no preregistrado, NO adoptable

Decidido **después** de ver los resultados, y por eso etiquetado así en el código, en los CSV y
en el manifiesto: **no se le aplica el criterio de adopción y no es candidato a adoptarse.**
Contesta una sola pregunta: cuánta cobertura bloquean las URLs de Tor. Se descartan del
diccionario los valores que en el entrenamiento aparecen en más de una familia (es el filtro de
genéricos de B.3, `MAX_FAMILIAS_VALOR`) y después se aplica la misma regla.

| Variante post-hoc | Cobertura | Acierto donde aplica | macro-F1 | Δ pareado | IC 95 % |
|---|---|---|---|---|---|
| sin circularidad + solo IOCs privados | **0,3974** | **0,9875** | 0,5658 ± 0,0477 | +0,0393 | [+0,0300; +0,0486] |
| con circularidad + solo IOCs privados | 0,3368 | 0,9796 | 0,5472 ± 0,0476 | +0,0207 | [+0,0137; +0,0277] |

Quitar cuatro URLs sube la cobertura de 0,2613 a **0,3974** y el macro-F1 de 0,5453 a **0,5658**.
Aun así **no llega a los 0,584 preregistrados**: el resto de la brecha son las dos razones que el
preregistro sí anticipó —el grafo agrega los IOCs de todas las notas de una plantilla mientras la
cascada decide nota por nota, y el diccionario se arma con la mitad del corpus—. **Si esta
variante se quisiera usar, hay que preregistrarla y volver a correrla como experimento propio;
citarla como resultado sería elegir la configuración después de ver los números.**

### (7) Qué se puede escribir en la tesis con esto

**La simetría de los dos frentes queda completa, y con la misma estructura de tres columnas:**

| Frente | Regla exacta: cobertura | Acierto donde aplica | ML que cubre el resto |
|---|---|---|---|
| **Archivos** (Exp. 2b / 2c) | firmas binarias 0,563 | 0,970 | bytes + RF: exactitud 0,912 ± 0,002, 100 % de cobertura |
| **Notas** (M.1 / Exp. 3) | IOCs 0,261 ± 0,026 | **0,981 ± 0,024** | TF-IDF + LinearSVC: macro-F1 0,5265 ± 0,0490 |

Las dos reglas exactas se comportan igual: **precisión altísima donde aplican, cobertura
parcial**, y el aprendizaje se queda con el resto. En notas la cascada además **sí mueve el
agregado** (+0,0188 de macro-F1, IC 95 % [+0,0128; +0,0248], 10/10 semillas), cosa que ningún
otro cambio de método logró en este frente: hiperparámetros, abstracción de marcadores y
embeddings dieron los tres resultado nulo o negativo. **M.1 es la primera mejora de método con
IC 95 % que excluye el cero en el frente de notas.**

Es también la respuesta cuantificada a Lemmou et al. (2021): su identificación de familia es
búsqueda por reglas y marcadores en mundo cerrado; medida aquí bajo train/test con plantillas
separadas, esa vía **cubre el 26 % de las notas** —el 40 % si se le saca la infraestructura Tor
compartida— y acierta el 98 % donde cubre. No reemplaza al clasificador: lo complementa.

**Límites a declarar con la cifra:**
1. La ganancia depende del filtro de circularidad: **+0,0188 sin filtro y +0,0058 (n.s.) con
   filtro.** Parte de la señal es el nombre de la familia dentro de sus propios IOCs
   (BLACKBASTA es el caso claro). Reportar siempre las dos.
2. La cobertura del 26 % es **sobre este corpus**, donde varias familias reutilizan
   infraestructura entre plantillas. No es una tasa de despliegue.
3. Cuatro URLs de Tor bloquean casi un cuarto de las decisiones. Es una limitación de la regla
   de unanimidad tal como se preregistró, no del mecanismo.

## 📌 VERIFICACIÓN «datoss.json» = CATÁLOGO MISP + CRUCE RÁPIDO CON LAS 30 (2026-08-22)

- `C:\Users\Romina\Downloads\datoss\datoss.json` es **byte a byte la misma copia** del catálogo
  MISP que envió el tutor: MD5 `0c6632bd0b4cc219e0528f77b4bb925c`, idéntico a
  `3_datos/misp_ransomware_galaxy/misp_galaxy_ransomware_2026-08-20.json`. **No es un dato
  nuevo** — no hay nada que importar; la carpeta de Downloads no trae readme ni corpus, solo
  ese JSON.
- Cruce automático contra las 30 familias (por `value`+`synonyms`, subcadena, **SIN auditar**):
  **15/30 con `ransomnotes-filenames`** (TESLACRYPT 18 sumando sus 4 entradas versionadas,
  CERBER 12, MEDUZALOCKER 11, CLOP 4, CHIMERA 3, DHARMA 6…) · **13/30 con `extensions`** ·
  **solo 5/30 con texto de nota utilizable** (>80 caracteres, no-URL: GANDCRAB 4, DHARMA 3,
  AVOSLOCKER, BLACKBASTA y RANSOMEXX 1 c/u). NOTPETYA no tiene entrada propia (solo «Petya»).
- ⚠️ Esos conteos son **cotas superiores**: el match por subcadena arrastra homónimos e
  imitadores — «cyclops» matchea CLOP, «Fake Cerber»/«CerberTear» caen en CERBER, CRYPTOLOCKER
  arrastra 14 entradas homónimas (CryptoLocker3, MSN CryptoLocker…), «Jokeroo» cae en GANDCRAB.
  **Confirma que el mapeo manual familia→entrada (transversal de `EXPERIMENTOS_PENDIENTES.md`)
  es imprescindible antes de correr M.2** — el cruce automático solo no alcanza.
- Para la **extensión a familias nuevas**: **92 entradas con texto de nota utilizable fuera de
  las 30** (53 de ellas además con nombre de archivo de nota). La procedencia se audita por
  nota (muchas `ransomnotes-refs` apuntan a id-ransomware.blogspot) y todo pasa por
  `verificar_nota_nueva.py`, como fija `PLAN_MEJORAS.md` §MISP.
- **Lectura sobria:** MISP casi no aporta textos nuevos a las 30 canónicas. Su valor es el
  **metadato** (nombres reales de nota, extensiones, synonyms para deduplicar alias), es decir
  M.2 y la tabla citable de extensiones — no es un corpus de notas.

### ★ MAPEO MISP BORRADOR GENERADO Y VERIFICADO ADVERSARIALMENTE (2026-08-22)

**El prerequisito de M.2 quedó preparado para la auditoría de Romina.** Script nuevo
`2_codigo/mapeo_misp_familias.py` (commits `84268e5` y el de corrección en develop, sin
coautoría): busca candidatas por alias exacto/subcadena + 6 controles manuales, y emite
**82 candidatas, todas con dictamen borrador (INCLUIR/EXCLUIR/REVISAR) y motivo**. Salidas
(fuera de git, derivadas del catálogo): `3_datos/misp_ransomware_galaxy/mapeo_borrador/`
— `mapeo_misp_borrador.csv` (columna `dictamen_romina` vacía para la auditoría; abre en
Excel) y `RESUMEN_mapeo.md`. **El borrador NO es citable; el mapeo auditado sí.**

**Verificación adversarial por subagente contra el JSON completo: 74/76 dictámenes
resistieron.** Correcciones aplicadas: `mailto` INCLUIR→REVISAR (el catálogo no dice en
ningún lado que mailto = NetWalker; hace falta fuente citable, mismo rasero que Ako) y
`cerberimposter` REVISAR→EXCLUIR (la propia entrada dice que no reutiliza el código de
Cerber). Hallazgos verificados que valen para la tesis:

1. **Confirmación independiente de la limpieza del 2026-08-19:** la entrada TorrentLocker
   de MISP (syn Crypt0L0cker/CryptoFortress/Teerac) documenta como nombres de nota
   `HOW_TO_RESTORE_FILES.html` y `DECRYPT_INSTRUCTIONS.html` + 8 versiones multiidioma —
   **exactamente el nombre de la nota `lm_Crypt0l0cker_HOW_TO_RESTORE_FILES.html` retirada
   del corpus**. El barrido por subcadena no la capturaba (Crypt0L0cker se escribe con
   ceros): se agregó como control manual EXCLUIR.
2. **NotPetya NO tiene entrada en el catálogo bajo ningún alias** (verificado exhaustivo:
   ExPetr/Nyetya/Petna/EternalPetya/PetrWrap/DiskCoder, cero resultados en value+synonyms
   de las 2135). Para la tabla citable de NOTPETYA usar CCN-CERT (README.TXT), no MISP.
3. **BlackMatter existe SOLO como sinónimo de Darkside** en todo el catálogo. La tesis las
   trata como 2 clases: BLACKMATTER queda **sin entrada MISP propia** (declararlo en M.2).
4. **Colisión real de nombre de nota a declarar en M.2:** `YOUR_FILES_ARE_ENCRYPTED.*`
   aparece en CUATRO entradas — Chimera (.HTML/.TXT), SunCrypt (.HTML), Mischa (.HTML) y
   Petya (.TXT). El nombre de nota no es unívoco entre familias.
5. Las ext `.encrypted/.ENC` de la entrada CryptoLocker son casi idénticas a las de
   TorrentLocker (`.Encrypted/.enc`): **sospecha de contaminación entre entradas del
   catálogo** — refuerza la pregunta abierta sobre CRYPTOLOCKER en NapierOne; no citar esas
   extensiones sin verificación.

**Cobertura tras el borrador (cifras borrador, no citables hasta auditar):** 14 familias
quedan con ≥1 nombre de nota citable vía MISP (AVOSLOCKER, BLACKBASTA, CERBER 12, CHIMERA 3,
CLOP 4, DHARMA 6, GANDCRAB 2, LOCKBIT, MEDUZALOCKER 11, RANSOMEXX 2, RYUK, SUNCRYPT,
TESLACRYPT 18, WASTEDLOCKER) · 2 sin entrada propia (BLACKMATTER, NOTPETYA) · 5 filas
REVISAR para llevar al tutor: Hunt (¿variantes de afiliado de Dharma cuentan?), Wcry
(¿pre-WannaCry cuenta?), Ako/MedusaReborn, mailto, SZ40. Los conteos de la tabla del cruce
rápido de arriba quedan superados por este mapeo (el cruce rápido era sin dictaminar).

> ✅ **Actualización 2026-08-23: `mailto` pasó de REVISAR a INCLUIR.** No lo resolvió el
> catálogo sino una fuente externa: **INCIBE-CERT** (CERT nacional de España) dice
> «NetWalker ransomware, also known as Mailto or Koko», y PCrisk lo equipara en el título de
> su entrada. **Quedan 4 REVISAR** (Hunt, Ako, Wcry, SZ40). Recuento final del borrador:
> **35 INCLUIR · 43 EXCLUIR · 4 REVISAR** sobre 82 candidatas.

## ★★★ TABLA CITABLE DE NOMBRES DE NOTA (30/30) Y AUDITORÍA DE PROCEDENCIA (2026-08-23)

> **Los dos prerequisitos de M.2 quedaron hechos y MEDIDOS.** Y la medición cambia el diseño
> del experimento: hay que leerla antes de correr M.2. Código: `2_codigo/tabla_nombres_notas.py`
> y `2_codigo/auditoria_nombres_corpus.py` (commits `9389437` y `38ac9e7` en develop, sin
> coautoría). Salidas en `3_datos/nombres_notas/` (fuera de git).

### (1) Tabla de nombres: **30/30 familias con al menos un nombre citable**

Recolección con 6 agentes en paralelo + **un verificador adversarial por lote que abrió cada
URL**: 76 hallazgos, **75 confirmados** (98,7 %). Cobertura: **137 filas, 30/30 familias**,
**8/30 con advisory oficial o CERT** (el resto id-ransomware / pcrisk / vendor / MISP).
Advisories confirmados leyendo el documento: **AA23-353A** (BLACKCAT, `RECOVER-(7 chars)
FILES.txt`), **AA24-060A** (PHOBOS, 3 nombres con 3 fuentes), **AA21-291A** (BLACKMATTER),
HHS HC3 (MAZE), CCN-CERT (NOTPETYA `README.TXT`), CERT-In (CONTI, WANNACRY), **INCIBE-CERT**
(NETWALKER). ⚠️ **CISA responde 403**: los advisories se verificaron sobre el **espejo de
ic3.gov** — anotarlo al citar.

**El único rechazo del verificador fue correcto y corrige un dato:** se había propuesto
BADRABBIT como `SIN_ARCHIVO`, pero Securelist menciona un «Readme file» y otras fuentes dan
`Readme.txt` — **BadRabbit sí deja archivo**. La verificación adversarial evitó escribir un
negativo falso.

**`SIN_ARCHIVO` confirmado (es RESULTADO, no hueco), con 2-3 fuentes cada uno:**
**CRYPTOLOCKER** (pantalla de bloqueo; Amigo-A + SecureWorks/Sophos — corrobora lo ya
registrado) y **JIGSAW** (ventana con temporizador; Amigo-A + pcrisk + BleepingComputer).

**Colisiones de nombre entre familias — el límite declarado de M.2:**

| Nombre | Familias |
|---|---|
| `README.txt` | BADRABBIT, BLACKBASTA, CONTI, DHARMA, NOTPETYA |
| `Info.hta` | **DHARMA, PHOBOS** |
| `{ID}-readme.txt` | NETWALKER, SODINOKIBI |
| `YOUR_FILES_ARE_ENCRYPTED.HTML` | CHIMERA, SUNCRYPT |

> **`Info.hta` en DHARMA y PHOBOS es el hallazgo más fuerte para el capítulo:** son
> exactamente el par que M.4 iba a atacar por «mismo texto, IOCs distintos». Ahora se sabe que
> **también comparten el nombre de la nota**, así que la vista del nombre **no puede
> separarlos** — es una limitación estructural del linaje, no del método.

### (2) Auditoría de procedencia: **solo 47 de 155 notas (30 %) tienen nombre usable**

155 notas emparejadas por MD5 contra el repo de Lemmou (187 archivos; **47/47 de las notas
`fuente=lemmou` emparejaron, 0 sin match**):

| Procedencia del nombre | Notas | ¿Sirve para M.2? |
|---|---|---|
| `genuino` (el corpus tiene el nombre real) | 21 | ✅ tal cual |
| `genuino_renombrado` (contenido verificado, nombre tocado por el curador) | 26 | ✅ con el nombre ORIGINAL del repo |
| `curador` (nombre inventado: `blackbasta1.txt`, `pcrisk_cuba_1.txt`) | 76 | ⛔ circular |
| `sin_verificar` | 32 | ⚠️ no hasta traerlo de la fuente |

**⛔ Circularidad concreta encontrada: 5 notas de CERBER con prefijo `lm_Cerber_`**
(`lm_Cerber__HELP_DECRYPT_[]_.hta` → genuino `_HELP_DECRYPT_[]_.hta`). Si M.2 usara el nombre
del corpus tal cual **estaría leyendo la etiqueta**. El nombre genuino recuperado NO la
contiene. Más las 76 `curador`, que codifican la familia por construcción.

> **La distinción que hay que escribir bien:** un nombre genuino que contiene el nombre de la
> familia (`RyukReadMe.txt`) **no es circular** — lo bautizó el malware y es justo la señal que
> explota ID Ransomware. Circular es que **el curador** haya puesto la familia en el nombre.
> La auditoría separa las dos cosas; sin esa separación, M.2 sería o circular o vacío.

**Los 47 nombres usables se concentran en 4 familias:** DHARMA 18, CERBER 14, GANDCRAB 7,
TESLACRYPT 8. **Las otras 26 familias tienen CERO nombre genuino en disco.**

### (3) ⚠️ CONSECUENCIA DE DISEÑO: M.2 CAMBIA DE FORMA, Y HAY QUE DECIDIRLO

La auditoría **descarta una de las dos versiones posibles de M.2**:

- **(a) nombre de la nota como feature por nota, bajo P2** → **inviable, y ahora está medido**:
  el 70 % de los nombres del corpus los puso el recolector. Entrenar con ellos es circular;
  excluirlos deja 4 familias con nombre y 26 sin nada. No es un problema de esfuerzo.
- **(b) regla nombre→familia, con la estructura de tres columnas de M.1/Exp. 2b** → viable
  como **regla** (el diccionario sale de la tabla citable, externa al corpus, igual que el
  diccionario de IOCs salía solo del pliegue de entrenamiento). **Pero su evaluación solo se
  puede hacer sobre las 47 notas con nombre genuino, o sea 4 familias** — hay que declarar esa
  base, y no es comparable con el macro-F1 sobre 30.

**Predicción a preregistrar antes de correr (b), si se corre:** cobertura alta y acierto alto
en DHARMA/CERBER/GANDCRAB/TESLACRYPT, salvo la confusión DHARMA↔PHOBOS por `Info.hta`; y
**cero aporte fuera de esas 4 familias**, por falta de nombre, no por falta de señal.

**Lo que sí se puede escribir sin correr nada más:** la **tabla de nombres por familia
(30/30, 137 filas con fuente)** como aporte descriptivo citable, la **medición de circularidad
del corpus** (47/155 usables, 5 prefijos `lm_`), las **4 colisiones** y los **2 `SIN_ARCHIVO`
confirmados**. Es material de capítulo 4 y de limitaciones, ya verificado.

## 📌 PREREGISTRO — M.6 CASCADA COMBINADA: IOCs PRIVADOS + NOMBRE GENUINO → TEXTO (2026-08-23)

> **Escrito ANTES de implementar y correr.** Base: 155 notas · 106 plantillas · 30 familias ·
> P2 (`grupos`, StratifiedGroupKFold 2 pliegues) · **10 semillas NUEVAS (100-109)**, distintas
> de las 10 de M.1, para que la estimación no reuse las semillas con las que se vio el
> resultado post-hoc. Salida en carpeta NUEVA `4_resultados/resultados_cascada_combinada_155/`.

### Qué se corre y por qué es legítimo correrlo

Dos cambios sobre M.1, en la MISMA capa de reglas y con la MISMA regla de unanimidad:

1. **Filtro de genéricos (`solo_privados`):** se descartan del diccionario los valores que en
   el ENTRENAMIENTO aparecen en más de una familia. **No es un criterio nuevo:** es
   `MAX_FAMILIAS_VALOR` de B.3, declarado **antes** de que M.1 se corriera. En M.1 apareció
   como diagnóstico post-hoc (cobertura 0,3974 · macro-F1 0,5658 · Δ +0,0393
   [+0,0300; +0,0486]) y por eso **no era citable**. Acá se preregistra y se corre con semillas
   nuevas.
   > ⚠️ **Declararlo así en la tesis, sin adornar:** re-correr sobre el MISMO corpus **no
   > elimina** que la variante se eligió después de ver resultados. Lo que la hace defendible
   > es que el criterio es **anterior e independiente** (B.3) y que la estimación se rehace con
   > semillas nuevas. Es una atenuación, no una prueba fuera de muestra.
2. **Nivel nuevo de NOMBRE DE ARCHIVO de la nota**, la señal de ID Ransomware. Diccionario
   nombre→familia construido **solo con el pliegue de entrenamiento**, y **solo con nombres
   auditados**: los 47 verificados por MD5 contra el repo de Lemmou (con el nombre ORIGINAL,
   no el del corpus) + los 17 recuperados de la fuente de cada nota. **Los 76 nombres
   `curador` NUNCA entran** — sería leer la etiqueta.

### Hechos del corpus medidos ANTES de correr (definen el techo)

| Hecho | Valor |
|---|---|
| Notas con nombre genuino | **64 / 155**, en **14 familias** |
| Notas con al menos 1 IOC | 137 / 155 |
| Notas con nombre **y** IOC | **63** |
| Notas con nombre y **sin** IOC (la ganancia posible del nivel nuevo) | **1** |
| Notas sin nombre ni IOC (piso irrecuperable por regla) | 17 |
| Nombres distintos | 40, con **1 sola colisión**: `info.hta` → DHARMA, PHOBOS |

### Predicciones (con criterio de falsación explícito)

1. **El nivel de nombre aporta ≈ 0.** Como 63 de 64 notas con nombre ya tienen IOC, el nivel
   nuevo puede sumar **como máximo 1 nota de cobertura (+0,006)**. Predicción: Δ macro-F1 del
   nombre **sobre** la variante de IOCs privados **< 0,005, con IC 95 % que incluye el cero**.
   **Las dos señales son redundantes en este corpus, no complementarias.**
2. **La ganancia real viene del filtro de genéricos:** cobertura **≈ 0,40** (0,3974 medido
   post-hoc) y macro-F1 **≈ 0,565** (esperado 0,55-0,58), Δ contra la base 0,5265 **positivo
   con IC 95 % que excluye el cero**.
3. **`info.hta` NO separa DHARMA de PHOBOS:** las asignaciones por ese nombre deben dar
   **conflicto** (dos familias) y caer al texto. Si alguna se asigna, hay un error de
   implementación.
4. **Controles negativos** (sin nombre genuino y sin IOC privado): **BADRABBIT, BLACKCAT,
   CHIMERA, CLOP-sin-IOC, JIGSAW, NOTPETYA** no deben recibir asignación por nombre. RYUK y
   HELLOKITTY, controles de M.1 por IOC, **ahora SÍ pueden moverse** porque tienen nombre
   genuino recuperado (`RyukReadMe.txt`, `read_me_ldk.txt`) — **deja de ser control**, hay que
   decirlo al reportar.
5. **La cobertura medida va a quedar por debajo de la estimada**, como en M.1 (predicho 0,584 /
   medido 0,2613): el diccionario se arma con la mitad del corpus y decide nota por nota.

### Criterio de adopción, fijado a priori

- **Se adopta la variante combinada** si su macro-F1 supera la base con **Δ pareado por semilla
  cuyo IC 95 % excluye el cero**.
- **El nivel de nombre se adopta por separado** solo si aporta Δ > 0 con IC 95 % que excluye el
  cero **sobre la variante de IOCs privados** (no contra la base pelada: eso confundiría los
  dos efectos).
- Se reportan **las dos variantes de circularidad** (con y sin filtro), como en M.1.
- Si la predicción 1 se confirma, el resultado que se escribe es: **el nombre de la nota es
  redundante con los IOCs en este corpus** — y es el sexto negativo de método convergente
  (hiperparámetros, abstracción de marcadores, embeddings, recolectar >4 plantillas, M.2 por
  nota, y este). **Refuerza que el techo lo pone el dato.**

## 📌 RESULTADO M.6 — CASCADA COMBINADA (2026-08-23, local)

> **Base declarada: 155 notas · 106 plantillas · 30 familias · P2 · semillas NUEVAS 100-109.**
> Código: `2_codigo/cascada_combinada_notas.py`. Salidas en
> `4_resultados/resultados_cascada_combinada_155/` — **no se tocó `resultados_canonicos/`,
> `resultados_extension_155/`, `resultados_cascada_155/` ni el capítulo 4.**
> **Veredicto: SE ADOPTA la variante combinada, Y el nivel de nombre se adopta por separado.
> Dos de las cinco predicciones preregistradas FALLARON** (se reportan como tales).

### (1) Las tres columnas del Exp. 2b (media ± desvío, 10 semillas nuevas)

| Variante | Cobertura | (de la cual por nombre) | Acierto donde aplica | macro-F1 | Δ pareado | IC 95 % | semillas |
|---|---|---|---|---|---|---|---|
| texto solo (base, semillas 100-109) | — | — | — | **0,4603 ± 0,0648** | — | — | — |
| privados sin circ. | 0,3871 | 0,0000 | 0,9512 | 0,4940 | +0,0338 | [+0,0187; +0,0488] | 10/10 |
| **privados sin circ. + NOMBRE** | **0,4484** | **0,1065** | **0,9557** | **0,5037** | **+0,0435** | **[+0,0299; +0,0570]** | **10/10** |
| privados con circ. | 0,3348 | 0,0000 | 0,9356 | 0,4813 | +0,0211 | [+0,0065; +0,0356] | 8/10 |
| privados con circ. + NOMBRE | 0,3897 | 0,0987 | 0,9416 | 0,4885 | +0,0283 | [+0,0158; +0,0407] | 9/10 |

**Aporte AISLADO del nivel de nombre** (contra la variante de IOCs privados, como exigía el
preregistro — no contra la base pelada):

| Comparación | Δ macro-F1 | IC 95 % | semillas | ¿Se adopta? |
|---|---|---|---|---|
| sin circ. + nombre − sin circ. | **+0,0097** | **[+0,0013; +0,0181]** | 6/10 | **SÍ** |
| con circ. + nombre − con circ. | +0,0072 | [+0,0001; +0,0143] | 5/10 | SÍ (al límite) |

### (2) ⛔ PREDICCIÓN 1 FALSADA — y el mecanismo es un hallazgo mejor que la predicción

Se predijo que el nombre aportaría **< 0,005 con IC que incluye el cero**, porque solo **1** de
las 64 notas con nombre carecía de IOC. **Falso: aporta +0,0097 con IC que excluye el cero, y
la cobertura sube de 0,3871 a 0,4484** (+0,061 ≈ 9,5 notas por semilla, no 1).

**El error del cálculo a priori, identificado:** se contaron las notas *sin ningún IOC*, pero lo
que importa es las notas **sin ningún IOC visto en ENTRENAMIENTO**. Y ahí está la diferencia
estructural:

> **El nombre de la nota se REPITE entre notas de la misma familia; el IOC casi nunca.**
> `Info.hta` aparece en 11 notas de DHARMA, así que un pliegue de entrenamiento casi siempre lo
> contiene. Los IOCs operativos (correo, `.onion`, BTC) son en su mayoría **únicos de una nota**,
> así que una nota de prueba con IOC nuevo no tiene con qué emparejar. **El nombre generaliza
> entre campañas; el IOC no.** Es exactamente la hipótesis original de D.2 («muchas familias
> conservan el nombre de la nota entre campañas»), que resulta **confirmada** — la estimación a
> priori era la equivocada, no la hipótesis.

### (3) ⚠️ PREDICCIÓN 3 FALSADA EN PARTE — la colisión `info.hta` casi no hace daño

Se predijo que `info.hta` daría **conflicto** y caería al texto. **No es lo que pasa:** de **78
asignaciones por `info.hta`, 75 aciertan DHARMA, 2 aciertan PHOBOS y solo 1 falla**
(PHOBOS→DHARMA).

**Por qué:** el filtro de genéricos estima la privacidad **solo en el pliegue de entrenamiento**.
Si la nota de PHOBOS con `info.hta` cae en prueba, en entrenamiento el nombre parece privado de
DHARMA y se usa. **PHOBOS igual sube +0,1451** de F1.

### (4) 🔴 HALLAZGO NO PREREGISTRADO E IMPORTANTE: LA BASE ES INESTABLE ANTE LA SEMILLA

**El texto solo da macro-F1 0,4603 ± 0,0648 con las semillas 100-109, contra 0,5265 ± 0,0490
con las semillas 0-9: una diferencia de 0,0662.** El control de sanidad del script abortó la
primera corrida por esto (la tolerancia inicial de 0,035 estaba mal fundada).

**Causa:** con **2 pliegues y 30 clases**, hay semillas donde familias enteras caen en un solo
pliegue y su F1 se va a 0. El rango por semilla va de **0,3665 a 0,5421**.

> **Consecuencia para la tesis, y hay que decirla:** el valor absoluto **0,5265 es específico de
> las semillas 0-9** y es más frágil de lo que sugiere su «± 0,0490». **Los Δ pareados NO están
> afectados** (base y variante salen de la misma partición en cada semilla), y por eso M.1 y M.6
> siguen en pie. Pero si el 0,5265 se cita como el resultado del frente de notas, corresponde
> **declarar que es sobre 10 semillas fijas** y idealmente re-medirlo con más semillas.

### (5) F1 por familia — la variante adoptada (Δ pareado, IC 95 %)

**Suben con IC que excluye el cero (8 familias):**

| Familia | F1 base | F1 M.6 | Δ | IC 95 % | asignadas por nombre |
|---|---|---|---|---|---|
| **DHARMA** | 0,4759 | **0,7479** | **+0,2720** | [+0,1342; +0,4098] | **96** |
| **BLACKBASTA** | 0,2143 | 0,4566 | +0,2423 | [+0,0373; +0,4474] | 0 |
| LORENZ | 0,3499 | 0,4790 | +0,1291 | [+0,0571; +0,2010] | 0 |
| SODINOKIBI | 0,3011 | 0,3959 | +0,0947 | [+0,0034; +0,1861] | 0 |
| RYUK | 0,2357 | 0,2956 | +0,0600 | [+0,0012; +0,1188] | 12 |
| TESLACRYPT | 0,8144 | 0,8705 | +0,0560 | [+0,0128; +0,0992] | 0 |
| CERBER | 0,8458 | 0,8939 | +0,0481 | [+0,0108; +0,0855] | 0 |
| JIGSAW | 0,4456 | 0,4726 | +0,0270 | [+0,0018; +0,0521] | 0 |

**DHARMA +0,2720 es la mayor mejora de una familia en todo el frente de notas**, y viene del
nombre (96 de sus 133 asignaciones). **GANDCRAB +0,1425** (36 por nombre) y **PHOBOS +0,1451**
también suben pero con IC que incluye el cero. **Ninguna familia baja de forma significativa**:
AVOSLOCKER −0,0583 y CONTI −0,0279, las dos con IC que incluye el cero.

**RYUK deja de ser control negativo**, como se preregistró: ahora recibe 12 asignaciones por
nombre (`RyukReadMe.txt`). Los **5 controles de nombre preregistrados** (BADRABBIT, BLACKCAT,
CHIMERA, JIGSAW, NOTPETYA) tuvieron **0 asignaciones por nombre**: control intacto.

### (6) Los 30 errores tienen UNA sola causa, la misma de M.1

Sobre **695 asignaciones hubo 30 errores** (acierto 0,9557). **29 de los 30 son las 4 URLs de
torproject** ya documentadas en M.1 y B.3 — GANDCRAB→CERBER (9, verificado: comparten
`https://www.torproject.org/`), CONTI→BLACKBASTA (9), CLOP↔AVOSLOCKER (4), etc. **1 solo es del
nombre** (`info.hta`).

> **Limitación central a declarar:** el filtro de genéricos **reduce pero no elimina** el
> problema de la infraestructura compartida, porque la privacidad se estima **en el pliegue de
> entrenamiento**. Un valor realmente compartido que en entrenamiento aparece en una sola
> familia **parece privado** y produce errores confiados en prueba. Es el mismo mecanismo que
> explica `info.hta`.

### (7) Qué se puede escribir con esto

**El frente de notas pasa de 0,4603 a 0,5037 de macro-F1 (Δ +0,0435, IC 95 % [+0,0299;
+0,0570], 10/10 semillas), con cobertura 0,4484 y acierto 0,9557 donde la regla aplica.** Es la
mejora de método más grande del frente, y **replica la estructura del Exp. 2b de archivos**:
regla exacta de altísima precisión y cobertura parcial, más aprendizaje para el resto.

**Y responde a Lemmou et al. (2021) con más precisión que antes:** su vía de reglas y marcadores,
medida aquí bajo train/test con plantillas separadas, cubre el **45 %** de las notas y acierta el
**96 %** donde cubre — con nombre de archivo incluido, que es lo que hace ID Ransomware.

**Límites a declarar junto con la cifra:** (a) la base es inestable ante la semilla (punto 4);
(b) solo 64/155 notas tienen nombre auditado, en 14 familias, así que el aporte del nombre está
acotado por el corpus, no por el método; (c) 29 de 30 errores son 4 URLs de Tor; (d) la variante
con filtro de circularidad da menos (+0,0283): parte de la señal es el nombre de la familia
dentro de sus propios marcadores. Reportar siempre las dos.

## 📌 RE-MEDICIÓN CON 50 SEMILLAS + RESULTADO M.7 (METADATO) — 2026-08-23

### (A) La base re-medida con 50 semillas: **0,4958 ± 0,0733**

Pedido de Romina tras el hallazgo de inestabilidad. `cascada_combinada_notas.py` acepta ahora
`--n-semillas` y `--semilla-inicial`. Corrido con **50 semillas (0-49)**; salidas en
`4_resultados/resultados_cascada_combinada_155_50semillas/`.

| | 10 semillas (100-109) | **50 semillas (0-49)** |
|---|---|---|
| Base, texto solo | 0,4603 ± 0,0648 | **0,4958 ± 0,0733** |
| M.6 (privados + nombre) | 0,5037 | **0,5471** |
| Δ pareado | +0,0435 [+0,0299; +0,0570], 10/10 | **+0,0513 [+0,0453; +0,0573], 50/50** |
| Cobertura | 0,4484 | 0,4547 (0,1155 por nombre) |
| Acierto donde aplica | 0,9557 | **0,9805** |
| Aporte aislado del nombre | +0,0097 [+0,0013; +0,0181] | **+0,0088 [+0,0062; +0,0114]**, 35/50 |

> **Conclusión sobre la base:** el **0,5265 publicado salió de las semillas 0-9 y está en el
> extremo alto de la distribución**; la estimación sobre 50 semillas es **0,4958 ± 0,0733**.
> **M.6 se refuerza:** Δ +0,0513 con **50/50 semillas** a favor e IC más angosto. Al escribir el
> capítulo, citar la base con su número de semillas; lo prolijo es re-medir P1/P2 con 50.

### (B) M.7 — EL METADATO DE LA NOTA (extensión + tamaño) **NO AYUDA**

Idea de Romina: el análisis forense usa metadatos del archivo, no solo el contenido. Y es
información que el modelo **no tenía**, porque el TF-IDF normaliza L2 y descarta el largo.
Código: `2_codigo/metadatos_notas.py`. 50 semillas, P2, mismo protocolo.
Salidas: `4_resultados/resultados_metadatos_155/`.

| Variante | macro-F1 | Δ pareado | IC 95 % | semillas | ¿Significativo? |
|---|---|---|---|---|---|
| texto (referencia) | 0,4958 ± 0,0733 | — | — | — | — |
| texto + extensión | 0,4982 | +0,0023 | [−0,0031; +0,0078] | 28/50 | **no** |
| texto + largo | 0,4914 | **−0,0044** | **[−0,0086; −0,0001]** | 23/50 | **sí, NEGATIVO** |
| texto + extensión + largo | 0,4955 | −0,0003 | [−0,0065; +0,0059] | 26/50 | no |
| **solo metadato (sin texto)** | **0,0446** | −0,4513 | [−0,4711; −0,4314] | 0/50 | sí |

**Predicción preregistrada (aporte < 0,01 con IC que incluye el cero): se cumple** para
extensión y para extensión+largo. **El largo es levemente peor que no usarlo.**

**Lo que cierra el caso: el metadato solo da macro-F1 0,0446, contra un azar de 0,033.** Casi
nada. Y eso desmonta la información mutua que parecía prometedora (43 % de la entiqueta):
**con las etiquetas PERMUTADAS al azar la IM ya daba 1,394 bits de 2,058 observados** — con 30
clases y 155 notas, contar celdas sobreajusta. **La IM cruda no es evidencia; el macro-F1 bajo
P2 sí.**

**Dos confundidos que hay que declarar si esto se escribe:**

1. **El largo es en parte el MÉTODO DE RECOLECCIÓN, no la familia.** Notas `bruto` (archivos
   del repo Lemmou): largo medio **4.016** caracteres · `corpus-existente` **1.100** ·
   `transcripcion` **1.032**. Y IM(tipo ; familia) = 0,751 de 4,597 bits: **el método ya
   predice el 16 % de la familia.** Los 37.602 caracteres de la nota más larga de CERBER son
   el markup del `.hta`, no el mensaje.
2. **La extensión es casi constante:** 117 de 155 notas son `.txt`, presente en las 30
   familias. `.hta` aparece en solo 2 (DHARMA, CERBER).

**Y el control del confundido lo confirma:** las familias que más suben con extensión+largo son
**DHARMA +0,1436** (sus 19 notas son **todas `bruto`**, 11 de ellas `.hta`) y **TESLACRYPT
+0,0311** (9 notas, todas `bruto`) — o sea, justo donde el largo está leyendo el método. Suben
también CRYPTOLOCKER +0,1210 y JIGSAW +0,0933, que son transcripciones cortas y uniformes.
**Pero el macro-F1 neto es cero o negativo: es REDISTRIBUCIÓN entre familias, no mejora.**

> **Resultado a escribir: el metadato de archivo de la nota (extensión y tamaño) no aporta al
> frente de notas.** Es el **séptimo negativo de método convergente** —hiperparámetros,
> abstracción de marcadores, embeddings, recolectar más de 4 plantillas, M.2 por nota, y ahora
> metadato— y todos apuntan a lo mismo: **el techo lo pone el dato, no el método.** Con una
> excepción, que es la que sí funcionó: **las reglas exactas sobre marcadores y nombre de
> archivo (M.1/M.6)**, porque no compiten con el texto: cubren lo que el texto no puede.

## 📌 ¿SIRVE UN DATASET DE NOMBRES POR FAMILIA? SÍ — MEDIDO (2026-08-23)

Pregunta de Romina. Código: `2_codigo/evaluar_tabla_nombres.py`. Salidas:
`3_datos/nombres_notas/eval_tabla_{resumen,detalle}.csv`.

### El uso que NO sirve, y hay que dejarlo escrito

**Como feature por nota, no se puede.** Asignarle a cada nota el nombre documentado de SU
familia convierte el nombre en **una función de la etiqueta**: la regla acertaría casi todo y
no significaría nada. Es la misma circularidad del prefijo `lm_Cerber_`, pero sistemática.

### El uso que SÍ sirve: diccionario EXTERNO de consulta (lo que hace ID Ransomware)

Y su evaluación es **más limpia que la de M.6**, porque no hace falta partición: el diccionario
sale de **literatura externa** (advisories, CERT, id-ransomware, pcrisk, MISP) y los nombres de
prueba salen de **artefactos auditados** (64 notas con nombre verificado por MD5 o declarado por
la fuente de esa nota). **Ninguna de las dos puntas usa la etiqueta de la nota evaluada, así que
no hay nada que se pueda filtrar.** Mide el techo real de la señal.

Diccionario: 132 filas útiles → **58 nombres literales + 41 plantillas**. Prueba: **64 nombres
en 14 familias**.

| Modo de emparejar | Cobertura | Acierto donde decide | Errores | Ambiguos (con la correcta) | Sin cobertura |
|---|---|---|---|---|---|
| **exacto** | **0,5000** (32/64) | **1,0000** (20/20) | **0** | 12 | 32 |
| exacto + patrón | 0,6094 (39/64) | 0,9630 (26/27) | 1 | 12 | 25 |

> **Donde la tabla da una respuesta única, acierta el 100 % en modo exacto (20 de 20), sin un
> solo error.** Es la cifra más limpia de todo el frente de notas, y es la comparación directa
> y cuantificada contra ID Ransomware.

**Los 12 ambiguos son honestos:** todos son `info.hta` → {DHARMA, PHOBOS}, y **los 12 contienen
la familia correcta**. La tabla no se equivoca: dice con razón que ese nombre pertenece a dos
familias. Es el límite estructural del linaje, ya documentado.

**El único error del modo patrón es un artefacto de mi conversión a regex**, no una colisión
real: `Recovery_README.html` (MEDUZALOCKER) queda capturado por `RECOVER<5_chars>.html` de
TESLACRYPT porque el `.+` se come «y_readme». **Por eso la cifra citable es la del modo exacto**
y el modo patrón se reporta como cota optimista.

### Por qué la cobertura es 0,50 y no más — y no es culpa de la tabla

| Familia | Aciertos | Por qué falla el resto |
|---|---|---|
| CUBA, DARKSIDE, HELLOKITTY, LOCKBIT, LORENZ, MAZE, WANNACRY | **todos** (11/11) | — |
| GANDCRAB | 0/7 | el corpus guarda `[]-DECRYPT.txt`: **Lemmou reemplazó la parte variable por `[]`**, y la tabla tiene `GDCB-DECRYPT.txt` / `CRAB-Decrypt.txt` |
| CERBER | 4/14 | ídem, `_READ_THIS_FILE_[]_.txt` contra `_HELP_DECRYPT_[A-Z0-9]{4-8}_.hta` |
| DHARMA | 6/18 | 11 de sus 18 son los `info.hta` ambiguos (con la correcta dentro) |
| TESLACRYPT | 4/8 | mismo `[]` |

**La mitad de la no-cobertura es un artefacto del corpus, no de la señal:** el repo de Lemmou
normalizó el ID de víctima a `[]`, así que esos nombres no pueden emparejar con la tabla ni con
nada. Es un límite del dato, medible y declarable.

### Qué se escribe con esto

1. **La tabla de nombres por familia es un aporte de la tesis por sí misma**, y ahora tiene una
   evaluación: **cobertura 0,50 · acierto 1,000 donde decide** sobre nombres auditados.
2. **Es la respuesta cuantificada a ID Ransomware con su propio mecanismo.** Su 71,93 % son
   41/57 notas de 22 familias (`Pruebas.xlsx`), medido sobre otro conjunto; acá, con el
   diccionario de nombres, la precisión donde aplica es total y el límite es la cobertura.
3. **Sirve en despliegue, no en entrenamiento.** Ese es el enunciado correcto: como tabla de
   consulta identifica sin error; como feature sería circular. Los dos usos NO son lo mismo y
   conviene decirlo explícitamente en el capítulo de método.

## ★★★ POR QUÉ EL FRENTE DE NOTAS ESTÁ EN 0,55 Y NO MÁS: EL TECHO ES EL PROTOCOLO (2026-08-23)

Pregunta de Romina («qué más exprimimos»). La respuesta quedó medida y es estructural.

### (1) P2 usa 2 pliegues porque **está forzado**, no por elección conservadora

Recuento de plantillas por familia (155 notas → 108 nodos familia#plantilla):

| Plantillas | Familias | Cuáles |
|---|---|---|
| **2** | **8** | BADRABBIT, BLACKMATTER, CRYPTOLOCKER, CUBA, DARKSIDE, NETWALKER, NOTPETYA, SUNCRYPT |
| 3 | 5 | AVOSLOCKER, CONTI, LORENZ, SODINOKIBI, WASTEDLOCKER |
| ≥4 | 17 | el resto (CERBER 8, DHARMA 6, LOCKBIT 6, HELLOKITTY 5, …) |

**Con 8 familias de 2 plantillas, el máximo de pliegues que deja cada familia representada en
todos es 2.** Y con 2 pliegues **el modelo entrena con ~54 de las 108 plantillas: menos de 2
por clase para 30 clases.** No es un defecto del método: es lo que el corpus permite. Pasar a
3 pliegues subiría el entrenamiento a ~72 plantillas (+18), y **no se puede** sin dejar 8
familias fuera de algún pliegue.

### (2) El 0,5471 es un promedio arrastrado por esas 8 familias

Descomposición de M.6 (50 semillas, variante adoptada), `m6_por_familia.csv`:

| Grupo | Familias | F1 base | F1 M.6 | Δ |
|---|---|---|---|---|
| **2 plantillas** | 8 | 0,3935 | **0,3958** | **+0,0023** (M.6 no las puede ayudar) |
| 3 plantillas | 5 | 0,6153 | 0,6381 | +0,0228 |
| **≥4 plantillas** | **17** | 0,5088 | **0,5916** | **+0,0828** |
| TODAS (30) | 30 | 0,4958 | 0,5471 | +0,0513 |

> **Cómo hay que reportarlo, y vale más que arañar otro 0,01:** el macro-F1 sobre las **17
> familias con ≥4 plantillas es 0,5916**; las **8 con 2 plantillas quedan en 0,3958** y M.6 no
> las mueve (+0,0023) porque no tienen nombre genuino y sus IOCs no se repiten. **El 0,5471
> sobre 30 clases subestima lo que el método hace**, porque promedia 8 clases que entrenan con
> UNA plantilla. Única familia por debajo de 0,20: **CRYPTOLOCKER** (2 plantillas, y su nota
> ni existe como archivo — ver `SIN_ARCHIVO`).

### (3) Y recolectarles plantillas NO es la salida: ya está medido

B.1 sobre 155 notas: la curva **satura en k≈2**. Deltas **1→2 +0,0476 (sig.) · 2→3 +0,0109
(NO sig.) · 3→4 −0,0040 (NO sig.)**. Conseguirle una 3ª plantilla a esas 8 familias vale
**+0,011 no significativo**. Lo que había servido al pasar de 144 a 155 notas fue **el NIVEL**
(familias que dejaron de dar 0), no seguir trepando la curva.

### (4) Balance del frente: 7 negativos, 2 positivos, y 2 palancas chicas sin usar

**No funcionó (medido):** hiperparámetros (840 configs) · abstracción de marcadores ·
embeddings multilingües · recolectar más allá de 2-4 plantillas · M.2 como feature por nota ·
M.7 metadato (extensión y tamaño) · más pliegues (imposible).

**Sí funcionó:** **M.1** cascada IOC (+0,0188) y **M.6** IOCs privados + nombre genuino
(**+0,0513**, 50/50 semillas) → **0,5471**. Las dos son reglas exactas: no compiten con el
texto, **cubren lo que el texto no puede**.

**Lo único que queda sin probar, y las dos son chicas:**
1. **La tabla externa de nombres como nivel de la cascada.** Medido aparte: acierto **1,000**
   donde decide (20/20) y cubre **32 de 64** nombres auditados, contra los ~18 que cubre el
   diccionario armado con el pliegue de entrenamiento. **Casi el doble de cobertura de nombre,
   sin necesitar entrenamiento** (el diccionario es externo, no hay fuga posible).
2. **Lista declarada de infraestructura compartida** (`torproject.org`). **29 de los 30 errores
   de M.6 son esas 4 URLs.** Es a priori, por conocimiento de dominio, no post-hoc.

Estimación honesta de las dos juntas: **+0,02 a +0,03**, o sea llegar a ~0,57. Después de eso,
**el frente de notas está en su techo con este corpus**, y lo que queda es escribirlo bien.

## ★★★★ LA RESPUESTA A «¿CÓMO HIZO LEMMOU?» — ES EL PROTOCOLO, MEDIDO (2026-08-23)

> **Hallazgo más importante de la sesión.** Código: `2_codigo/protocolo_lemmou.py`. Salida:
> `4_resultados/resultados_protocolo_lemmou/comparacion_protocolos.csv`.
> **El mismo corpus, el mismo vectorizador, el mismo clasificador. Lo único que cambia es el
> protocolo de evaluación.**

| Protocolo | Exactitud | Exact. balanceada | macro-F1 |
|---|---|---|---|
| **L (Lemmou): 1-NN coseno, mundo cerrado, leave-one-out, SIN train/test** | **0,8387** | 0,8188 | **0,8055** |
| L + **LSA** (TruncatedSVD 100) — literalmente su método | 0,8387 | 0,8188 | 0,8055 |
| P1 (estratificado: separa notas, **no** plantillas), 50 semillas | 0,8365 | 0,8143 | 0,8020 ± 0,0308 |
| **P2 (grupos: separa PLANTILLAS) — el de la tesis**, 50 semillas | 0,5813 | 0,5529 | **0,4958 ± 0,0733** |

### El mecanismo, cuantificado

- **71 de las 155 notas (45,8 %) tienen como vecina más cercana a una nota de su MISMA
  plantilla.**
- Acierto cuando la vecina es su casi-duplicada: **0,9577**.
- Acierto cuando la vecina es de **otra** plantilla: **0,7381** ← esto es lo que P2 mide.

> **Traducción:** el 181/182 de Lemmou et al. (2021) sale de buscar cada nota contra una base
> **que contiene esa misma nota o su gemela**. No es clasificación de una plantilla no vista:
> es **recuperación de casi-duplicados en mundo cerrado**. Sobre NUESTRO corpus, ese mismo
> protocolo da **macro-F1 0,8055**. La caída a 0,4958 **no es del método: es del protocolo**,
> que separa plantillas a propósito para que la nota de prueba nunca tenga a su gemela en
> entrenamiento.

**Dato extra:** el LSA (SVD 100) da **exactamente el mismo resultado** que el coseno crudo
(0,8055). Coherente con el negativo de embeddings: en este corpus la reducción dimensional no
agrega nada.

### Qué se escribe con esto — y cambia el relato del capítulo

1. **La respuesta al «no se puede clasificar»: SÍ se puede, a macro-F1 0,8055, si se acepta el
   mundo cerrado.** El 0,4958 es **el precio de medir generalización a plantillas no vistas**.
   Son dos preguntas distintas y hay que reportar las dos.
2. **Es la comparación honesta y cuantificada con el trabajo previo**, que hasta ahora solo se
   podía argumentar: «bajo el protocolo de Lemmou et al. nuestro método alcanza macro-F1
   0,8055 sobre 30 familias; bajo train/test con separación de plantillas cae a 0,4958±0,0733.
   La diferencia es enteramente el protocolo».
3. ⚠️ **El 0,8055 NO es un resultado de generalización y no se puede citar como el resultado de
   la tesis.** Es leave-one-out sobre corpus cerrado: reportarlo como tal sería exactamente el
   error que la tesis evita. Va como **medición de comparabilidad**, con esta advertencia
   pegada.
4. Recordar el otro punto ya fijado: su **F = 0,920 es la tarea BINARIA** nombre-de-nota frente
   a nombre-benigno, **no** familia por nombre. Son tres cifras distintas de ese paper y se
   confunden fácil: 181/182 (familia, mundo cerrado), F=0,920 (binaria de nombre), y nada
   comparable a un macro-F1 bajo train/test.

### Y explica por qué P1 ≈ L

P1 (0,8020) queda **casi igual que L (0,8055)** porque el estratificado deja casi-duplicadas
repartidas entre entrenamiento y prueba: en la práctica es también mundo cerrado. **La única
partición que mide algo distinto es P2.** Eso valida a posteriori la decisión de usar P2 como
protocolo principal de la tesis.

## ★★★ EL TECHO POR FAMILIA: LO QUE MANDA ES QUE ALGO SE REPITA ENTRE PLANTILLAS (2026-08-23)

> Pregunta de Romina: «si una familia tiene solo 2 plantillas, ¿no es casi imposible estudiar
> una para identificar la otra, salvo que haya similitud en el nombre de archivo o metadatos?»
> **La conclusión es correcta, pero el mecanismo resultó ser OTRO — y es más accionable.**
> Código: `2_codigo/techo_por_familia.py`. Salida:
> `4_resultados/resultados_techo_por_familia/techo_por_familia.csv`.

### El hallazgo, contraintuitivo y medido

| Grupo | Coseno medio entre plantillas | Nombres que CRUZAN plantillas | IOCs que CRUZAN plantillas |
|---|---|---|---|
| **8 familias de 2 plantillas** | **0,7105** | **0** | **6** |
| 17 familias de ≥4 plantillas | 0,4423 | 6 | **67** |

**Las familias de 2 plantillas tienen las plantillas MÁS parecidas entre sí** (0,71 contra
0,44). O sea que **la hipótesis «son imposibles porque sus dos plantillas son muy distintas» es
FALSA.** Lo que les falta es lo otro: **un marcador o un nombre de archivo que se repita de una
plantilla a la otra.** Y eso es exactamente lo que una regla necesita para cruzar la partición
de P2: un IOC que aparece en UNA sola plantilla no sirve, porque si esa plantilla está en
prueba, su valor no está en el diccionario de entrenamiento.

**La correspondencia con el Δ de M.6 es directa:**

| Familia | Plantillas | nombres que cruzan | IOCs que cruzan | Δ M.6 |
|---|---|---|---|---|
| BLACKBASTA | 4 | 0 | 3 | **+0,4473** |
| DHARMA | 6 | 2 | 7 | **+0,2240** |
| RYUK | 4 | 1 | 0 | +0,1021 |
| LORENZ | 3 | 0 | 2 | +0,1005 |
| CERBER | 8 | 0 | 21 | +0,0518 (ya estaba en 0,85) |
| **las 8 de 2 plantillas** | 2 | **0** | 0-2 | **+0,000 a +0,007** |

### Casos límite que conviene citar

- **CRYPTOLOCKER**: 2 plantillas, coseno 0,4129 entre ellas, **0 nombres y 0 IOCs que cruzan**
  → F1 **0,0773** y M.6 no la mueve. Es la familia estructuralmente más difícil del corpus, y
  además su nota **no existe como archivo** (`SIN_ARCHIVO` confirmado). Caso de manual.
- **NOTPETYA**: coseno 0,8149 entre sus 2 plantillas —o sea, se parecen bastante— y aun así F1
  0,3798. Alta cohesión **no** alcanza: con 1 plantilla en entrenamiento frente a 29 clases
  más, el clasificador no tiene con qué.
- **WASTEDLOCKER**: coseno máximo 0,9065 entre plantillas, al borde del umbral de
  casi-duplicado. Su F1 0,6370 es más alto de lo que sugería el relato de «familia
  irrecolectable».

### ⚠️ DISCREPANCIA CON B.3 QUE HAY QUE RESOLVER ANTES DE ESCRIBIR

Acá el Spearman(coseno entre plantillas ; F1) da **+0,2872 (p = 0,12), NO significativo** —
y con M.6 baja a +0,1488 (p = 0,43). **B.3 había reportado ρ +0,704 (p = 2,0·10⁻⁵).**

**No es una contradicción: son F1 distintos.** B.3 correlacionó su cohesión con el **F1 por
familia de B.1 (P2ret, k=todo, 100 repeticiones)**, y acá se usa el **F1 de P2 con LinearSVC,
50 semillas**. Además la cohesión de B.3 promedia pares de notas y esta promedia
**representantes de plantilla**. **Hay que decidir cuál de las dos se cita en el capítulo y
declarar la definición exacta**; citar «ρ +0,704» al lado de esta tabla sin aclarar la
diferencia sería un error.

### Consecuencia accionable: el camino para esas 8 familias NO es más plantillas

B.1 ya midió que la curva satura en k≈2 (2→3 vale +0,011 no significativo). **Lo que les
falta es un nombre de archivo, y no lo tienen:** las 8 suman **0 nombres que cruzan
plantillas**. Y varias de ellas —BADRABBIT, NOTPETYA, CRYPTOLOCKER— tienen sus notas como
`corpus-existente` con fuente «NapierOne/varios», o sea **sin procedencia rastreable**.

> **Esto le da un valor nuevo y concreto a la auditoría B.2 (las 34 notas sin fuente, ya
> pendiente): establecer su procedencia es la única vía para recuperarles el nombre de archivo,
> y son justamente las familias que más lo necesitan.** No es una tarea de prolijidad
> bibliográfica: es la única palanca identificada para el grupo de familias que hoy está en
> 0,3958.

## ★★★ AUDITORÍA B.2 EJECUTADA: 13 de 18 CON PROCEDENCIA, Y 4 PROBLEMAS DE INTEGRIDAD (2026-08-23)

> Búsqueda **por el TEXTO** de cada nota huérfana (18 notas de las 8 familias de 2 plantillas),
> con un verificador adversarial por lote que abrió cada URL. Datos crudos:
> `3_datos/nombres_notas/procedencia_2026-08-23.json`. **13 confirmadas · 3 rechazadas · 2 sin
> fuente.** Es la primera ejecución real del item **B.2**, pendiente desde
> `DIAGNOSTICO_2026-07-27.md` punto A4.

### (1) Lo que se recuperó

**6 nombres de archivo nuevos, cada uno declarado por la fuente de ESA nota:**

| Familia | Nota | Nombre declarado |
|---|---|---|
| CUBA | `pcrisk_cuba_variant2.txt` | `!! READ ME !!.txt` |
| BLACKMATTER | `note_pcrisk.txt` | `[random_string].README.txt` |
| SUNCRYPT | `note_pcrisk.txt` y `suncrypt.html` | `YOUR_FILES_ARE_ENCRYPTED.HTML` |
| DARKSIDE | `note_pcrisk_variant.txt` | `README.[victim's_ID].TXT` |
| NETWALKER | `note_pcrisk.txt` | `[random-string]-Readme.txt` |

**Y una CORRECCIÓN importante que atrapó el verificador:** se había propuesto NOTPETYA como
`SIN_ARCHIVO`, y **es falso**. La propia página que la fila citaba (Amigo-A, Petya NSA-EE,
27-06-2017) tiene dos encabezados literales «Содержание текстовой записки **README.TXT**
(один вариант / другой вариант)», cada uno seguido de este mismo texto. **NotPetya SÍ deja
archivo, y se llama `README.TXT`** — coincide con lo que ya decía CCN-CERT. Sin el verificador
se escribía un negativo falso.

**Fuente oficial nueva para CRYPTOLOCKER:** FBI PSA del 28-10-2013,
`https://www.ic3.gov/PSA/2013/PSA131028.pdf`.

### (2) ⛔ CUATRO PROBLEMAS DE INTEGRIDAD, uno por uno

**(a) `BADRABBIT/badrabbit_note2.txt` es una PARÁFRASIS de `note1`, no una nota independiente.**
Verificado leyendo los dos archivos completos: comparten los mismos datos (0,05 BTC ≈ 280 USD,
40 horas, el mismo `.onion caforssztxqzf2nm`) reescritos con otras palabras, y note2 pierde la
línea de apertura «Oops! Your files have been encrypted.» que **las dos** versiones documentadas
sí traen. Las frases de su segunda mitad («Your personal installation key is required for
authentication», «a limited time window of 40 hours») **no aparecen en ninguna publicación**
(se buscaron entre comillas en pcrisk, Amigo-A, Securelist, Talos, Varonis).
> **Consecuencia: BADRABBIT tiene 1 plantilla real, no 2.** Pasa al grupo de WASTEDLOCKER
> (inevaluable bajo P2 por construcción). Hay que decidirlo con el tutor porque cambia el
> conteo de plantillas y la tabla del techo por familia.

**(b) `CRYPTOLOCKER/cryptolocker_note2.txt` NO tiene fuente, y su cuerpo está reescrito.**
Se buscaron 6 frases distintivas entre comillas: cero resultados. Dice «safely encrypted on
this PC» y «RSA-2048 **and AES-256** ciphers», cuando el CryptoLocker real declara **solo
RSA-2048** y su lista es «photos, videos, documents, etc.». **Mismo patrón que
`chimera_note2.txt`.** De las 2 notas de CRYPTOLOCKER solo `note1` es verificable — y explica
por qué esa familia tiene el F1 más bajo del corpus (0,0773).

**(c) DOS placeholders sintéticos sin documentar.** Barrido sobre las 155 notas: solo 2 archivos
traen placeholders en estilo `[MAYUSCULAS_CON_GUION]`:
`BADRABBIT/badrabbit_note1.txt` → `[RANDOM_KEY_PLACEHOLDER_A1B2C3D4E5F6]` y
`NOTPETYA/notpetya_note1.txt` → `[UNIQUE_KEY_PLACEHOLDER]`. **No los produce el malware ni son
la convención de un curador público.** Lo más probable es que sean **redacción de la clave de
instalación** hecha por quien armó el corpus fundacional — defendible, pero **hoy no está
declarado en ningún lado**, así que un lector no puede distinguir redacción de fabricación.
⚠️ Ojo: `notpetya_note1.txt` **ya está confirmada como auténtica** por OCR contra una imagen
independiente (coseno 0,925), así que acá el problema es de documentación, no del texto.

**(d) `CUBA/cuba.txt`: cuerpo documentado, correo primario que no aparece en ninguna fuente.**
Su `roselondon@cock.li` no figura en pcrisk ni en Amigo-A (que sí listan `cloudkey@cock.li`,
`waterstatus@cock.li`, `filebase@cock.li`, `belingmor@cock.li` y otros). Probablemente sea una
variante auténtica no publicada, pero **la procedencia queda NO establecida.**

**Y un matiz sobre `BLACKMATTER/blackmatter.txt`:** su fuente real es **ThreatLabz** (como ya
decía el manifiesto), no la página de Amigo-A que un agente propuso. La versión del corpus es
**distinta** de la que publican pcrisk y Amigo-A: le falta la línea «We have downloaded 1TB
from your fileserver.» y todo el bloque «>> Data leak includes», y trae `[snip]` en la ruta
`.onion`. No es un error del corpus: es que el repo público redacta.

### (3) 📌 EL PATRÓN SISTEMÁTICO, y qué falta revisar

Siete familias tienen el par `<familia>_note1.txt` / `_note2.txt`, todas del lote fundacional.
**De los 7 `note2`, ya hay CUATRO con problema confirmado:**

| `note2` | Estado |
|---|---|
| CHIMERA | ⛔ reescritura del alemán + contenido agregado (documentado 2026-08-19) |
| NOTPETYA | ⛔ párrafo «IMPORTANT» que no está en la imagen (documentado 2026-08-19) |
| **BADRABBIT** | ⛔ **paráfrasis de note1, sin fuente** (hoy) |
| **CRYPTOLOCKER** | ⛔ **sin fuente, cuerpo reescrito** (hoy) |
| HELLOKITTY | ❓ **sin revisar** |
| JIGSAW | ❓ **sin revisar** |
| WANNACRY | ❓ **sin revisar** |

> **Ya no es una sospecha caso por caso: es un patrón del lote fundacional.** Los archivos
> `*_note2.txt` son sistemáticamente sospechosos. **Quedan 3 por revisar (HELLOKITTY, JIGSAW,
> WANNACRY) y es lo próximo de B.2.** También hay 87 apariciones de `[snip]` que son la
> convención de redacción de ThreatLabz: legítimas, pero **hay que declararlas en la tesis**
> porque significan que varios artefactos vienen parcialmente redactados de origen.

### (4) El control ya impide que esto se repita

`2_codigo/validar_procedencia.py` (commit `94c45c4`) falla si entra una nota sin fuente
citable, si la deuda crece, si una transcripción no trae URL, o si manifiesto y disco no
coinciden 1:1. La deuda heredada quedó congelada en `3_datos/deuda_procedencia.json`: **de 47
detectadas bajó a 34** al completar las 13 URLs de pcrisk que faltaban, y con las 13
procedencias de hoy puede bajar más. **Solo puede bajar.**

## ★★★★ B.2 CERRADA: DE 47 SIN FUENTE A 8, Y 9 NOTAS SON REESCRITURAS DE OTRAS DEL CORPUS (2026-08-23)

> Segunda tanda: las 26 notas restantes, búsqueda **por el TEXTO** con verificador adversarial
> por lote. **18 de 26 confirmadas.** Datos crudos:
> `3_datos/nombres_notas/procedencia_lote2_2026-08-23.json`.

### (1) El saldo de procedencia

| Momento | Sin fuente |
|---|---|
| Al correr el control por primera vez | **47** |
| Tras completar 13 URLs de pcrisk que faltaban en el manifiesto | 34 |
| Tras la 1ª tanda de auditoría (8 resueltas) | 26 |
| **Tras la 2ª tanda (18 resueltas) — hoy** | **8** |

**De 47 a 8: se resolvió el 83 % de la deuda de procedencia en un día.** El 24 % del corpus sin
fuente rastreable que arrastraba el proyecto desde `DIAGNOSTICO_2026-07-27.md` queda en **5 %**.

### (2) Nombres de archivo rescatados: **18**

| Familia | Nota | Nombre declarado por su fuente |
|---|---|---|
| AVOSLOCKER | `note_pcrisk.txt` | `GET_YOUR_FILES_BACK.txt` |
| BLACKMATTER | `note_pcrisk.txt` | `[random_string].README.txt` |
| CUBA | `pcrisk_cuba_variant2.txt` | `!! READ ME !!.txt` |
| DARKSIDE | `note_pcrisk_variant.txt` | `README.[victim's_ID].TXT` |
| **LOCKBIT** | `lb20` / `lb30b` / `lb40` / `lb50` | `Restore-My-Files.txt` · `[ID].README.txt` · `[rand].README.txt` · **`ReadMeForDecrypt.txt`** |
| MAZE | `malware_notes_maze.txt` | `DECRYPT-FILES.txt` |
| MEDUZALOCKER | `note_pcrisk.txt` | `HOW_TO_RECOVER_DATA.html` |
| NETWALKER | `note_pcrisk.txt` | `[random-string]-Readme.txt` |
| PHOBOS | `note_pcrisk_hta.txt` / `_txt.txt` | `info.hta` · `info.txt` |
| RYUK | `note_pcrisk.txt` | `RyukReadMe.txt` |
| SUNCRYPT | `note_pcrisk.txt` / `suncrypt.html` | `YOUR_FILES_ARE_ENCRYPTED.HTML` |
| WASTEDLOCKER | `note_pcrisk_bba` / `_rlh` | `[original].bbawasted_info` · `[original].rlhwasted_info` |

**Nombres usables para M.6 pasan de 64 a ~82 de 155.** LOCKBIT gana 4 (tenía 0 que cruzaran
plantillas) y PHOBOS gana `info.hta` **con procedencia propia**, lo que refuerza la colisión ya
documentada con DHARMA.

**8 `SIN_ARCHIVO` confirmados** (es resultado, no hueco): BADRABBIT note1 · CRYPTOLOCKER note1 y
`pcrisk_cryptolocker_1` · **JIGSAW note1 y note2** · NOTPETYA note1 · **WANNACRY note1 y note2**.

### (3) ⛔⛔ EL HALLAZGO GRAVE: 9 NOTAS SON REESCRITURAS O DUPLICADOS DE OTRAS DEL CORPUS

Ya no es el patrón `_note2`: es más amplio y **cruza el nombre de archivo**.

| Nota del corpus | Es reescritura / duplicado de |
|---|---|
| `BADRABBIT/badrabbit_note2.txt` | `badrabbit_note1.txt` |
| `CHIMERA/chimera_note2.txt` | `chimera_note1.txt` |
| `CRYPTOLOCKER/cryptolocker_note2.txt` | `cryptolocker_note1.txt` |
| `NOTPETYA/notpetya_note2.txt` | `notpetya_note1.txt` |
| **`MAZE/malware_notes_maze.txt`** | **`MAZE/DECRYPT-FILES.txt`** (mismo directorio) |
| **`RYUK/note_variant_email.txt`** | **`RYUK/note_pcrisk.txt`** y `RYUK/ryuk.txt` |
| **`WASTEDLOCKER/note_pcrisk.txt`** | **`note_pcrisk_bba.txt`** |
| **`LOCKBIT/lb20.txt`** | **`lockbit2.txt`** — duplicado EXACTO, no paráfrasis |
| **`LOCKBIT/lb30b.txt`** | **`[id].README.txt`** — mismo texto defangueado y TRUNCADO |

> **Por qué el agrupador de casi-duplicados no los detectó:** el umbral de 0,90 sobre TF-IDF
> char 3-5 no captura una **reescritura** (parafrasear baja el coseno por debajo de 0,90) ni un
> texto **truncado** (el truncado baja la similitud — trampa ya medida en este proyecto el
> 2026-08-19 con JIGSAW). Así que estas 9 cuentan hoy como plantillas distintas **y no lo son.**

**Consecuencia directa sobre las cifras:** de las 106 plantillas, **hasta 9 podrían ser
espurias**. Y golpea justo donde más duele:
- **BADRABBIT, CRYPTOLOCKER y NOTPETYA pasarían de 2 plantillas a 1** → se suman a WASTEDLOCKER
  como **inevaluables bajo P2 por construcción** (F1 0 por diseño, límite declarado de B.1).
- Cambia la tabla del techo por familia, el conteo de 8 familias de 2 plantillas, y todas las
  cifras de P2 / M.1 / M.6.

⚠️ **NO se tocó nada.** Sacar o fusionar notas cambia el capítulo 4 completo: es decisión de
Romina y del tutor. Pero **hay que decidirlo antes de escribir las cifras finales.**

### (4) Las 8 que quedan sin fuente — y 5 de ellas son de las reescrituras

`BADRABBIT/badrabbit_note2` · `CHIMERA/chimera_note1` y `note2` · `CRYPTOLOCKER/cryptolocker_note2`
· `HELLOKITTY/hellokitty_note1` y `note2` · `NOTPETYA/notpetya_note2` · `RYUK/note_variant_email`.

**HELLOKITTY es el caso más serio que queda:** sus DOS notas fundacionales no tienen fuente
(`note1` dio *no encontrado*, `note2` *parecido pero distinto*), **y las dos tienen sus
marcadores reemplazados por placeholders sintéticos** (`[EMAIL_ADDRESS]`, `[VICTIM_ID]`,
`[BACKUP_EMAIL]`, `[KEY_ID]`), con lo cual quedan con **cero IOCs**. Era uno de los dos
controles negativos de M.1 con cobertura 0,000: **ahora se sabe que su cobertura nula es un
artefacto del corpus, no una propiedad de la familia.**

**Y la respuesta al pedido puntual sobre los tres `_note2` que faltaban:**
- **JIGSAW: LIMPIA.** note1 y note2 son textos independientes, los dos con procedencia y los dos
  `SIN_ARCHIVO` (ventana con temporizador).
- **WANNACRY: LIMPIA.** Ídem, los dos con procedencia y `SIN_ARCHIVO`.
- **HELLOKITTY: PROBLEMA.** Ninguna de las dos tiene fuente.

### (5) Placeholders sintéticos: barrido completo

**8 notas** del lote fundacional traen marcadores reemplazados por placeholders en estilo
`[MAYUSCULAS]`: `[EMAIL_ADDRESS]`, `[BITCOIN_ADDRESS]`, `[ONION_ADDRESS]`, `[VICTIM_ID]`,
`[UNIQUE_KEY]`, `[KEY_ID]`, `[ADDRESS]`, `[BACKUP_EMAIL]`,
`[RANDOM_KEY_PLACEHOLDER_A1B2C3D4E5F6]`. **5 de ellas quedan con CERO IOCs extraíbles.**
Familias afectadas: BADRABBIT, CHIMERA (×2), CRYPTOLOCKER, HELLOKITTY (×2), JIGSAW, NOTPETYA.

⚠️ Matiz honesto: los placeholders explican **5 de las 30** notas del corpus sin ningún IOC. Las
otras 25 son casos legítimos — los `[]` con que el repo de Lemmou normalizó el ID de víctima
(DHARMA `FILES ENCRYPTED.txt`, GANDCRAB `[]-DECRYPT`) y las redacciones con `-` de pcrisk.

**Aparte, `[snip]` × 87:** es la convención de redacción del repo público de ThreatLabz.
Legítima, pero **declararla en la tesis**: varios artefactos vienen parcialmente redactados de
origen, y eso limita cuántos IOCs se pueden extraer.

## 📌 ORIGEN DE LAS NOTAS SINTÉTICAS: DOS PISTAS VERIFICADAS Y DESCARTADAS (2026-08-23)

> Pregunta de Romina: «¿en algún momento se crearon notas sintéticas y se creó así?». Se
> investigó con evidencia. **La respuesta corta: el patrón es real, pero el origen sigue sin
> identificarse, y las dos pistas disponibles quedaron descartadas.** Anotado para que nadie
> las vuelva a seguir.

### ⛔ PISTA 1 DESCARTADA — `gitlab.com/kipziptie/ai_ransomware_note_detection`

Era la pista que `DIAGNOSTICO_2026-07-27.md` (punto F3) dejó anotada: «parte de las 37 notas
NapierOne/varios podría venir de ahí». **Verificado leyendo el repo por la API de GitLab
(solo texto, sin descargas): NO es el origen.**

- El repo existe (Kim Trujillo, marzo 2022, 46 commits) y su tesis asociada está en UPC Commons
  (`https://upcommons.upc.edu/handle/2117/386579`, abril 2023): tarea **binaria** nota-vs-no-nota
  con 59 + 59 muestras, árboles de decisión y SVM.
- **`RESEARCHER_GENERATED_SAMPLES/` contiene UN solo archivo**, `Kims_first_ransomnote.txt`, y
  es una **nota en broma**: firma «jobhopper gang», pide **10 galletitas** de rescate y trae el
  correo real de la estudiante. Es una prueba de juguete, no un corpus sintético.
- **`python_preprocessor/rawNotes/` tiene 126 notas y NINGUNA** es de las 6 familias con
  problema (BADRABBIT, CHIMERA, CRYPTOLOCKER, HELLOKITTY, NOTPETYA, RYUK).
- `NEW_RANSOMNOTE_SAMPLES/` trae 10 notas (mountlocker, darkside, babuk, cuba, conti, revil,
  BlackMatter…), tampoco coinciden.
- **No usa placeholders** del estilo `[EMAIL_ADDRESS]`.
- Y usa **la misma colección de Lemmou** que este corpus ya tiene verificada por MD5.

### ⛔ PISTA 2 DESCARTADA — que la convención de placeholders venga de un dataset publicado

Búsqueda de las cadenas literales `"RANDOM_KEY_PLACEHOLDER"` y `"UNIQUE_KEY_PLACEHOLDER"`:
**cero resultados** en la web indexada. Ninguna colección publicada usa esa convención.

> **Esto INVIERTE la hipótesis que se había planteado.** Si los placeholders vinieran heredados
> de otro dataset, la convención aparecería publicada en alguna parte. No aparece. **Lo más
> probable es que se hayan producido ad hoc**, no que se hayan heredado. Sigue sin poder
> establecerse **quién** ni **cuándo**: el manifiesto no tiene historia porque `3_datos/` está
> fuera de git por regla del proyecto (y esa regla es correcta: es malware real).

### Lo que queda comprobado, y cómo hay que escribirlo

**Probado:** 8 notas del lote fundacional tienen sus marcadores reemplazados por placeholders
que no produce el malware ni son la convención de ThreatLabz (`[snip]`); 9 notas son
reescrituras o duplicados de otras del propio corpus; y en varias el texto auténtico está
publicado y **difiere**.

**No probado:** quién las produjo y con qué intención. **No corresponde afirmarlo en la tesis.**
Lo que sí corresponde es **declarar el hallazgo con su evidencia** y decir que la procedencia de
esas 8 no pudo establecerse. La auditoría en sí es un aporte metodológico.

### Pistas que quedan sin agotar

1. **Kaggle «Ransomware Note Dataset Collection»** (`abiprasanth`): **no se pudo inspeccionar**
   —la página es JS y pide sesión—. Romina sí puede abrirla con su navegador.
2. **Preguntarle a Carlos.** `Pruebas.xlsx` es material compartido y el lote fundacional es
   anterior al seguimiento de procedencia. Es la vía más directa.
3. **Group-IB, «Notes From the Most Active Ransomware Groups»**
   (`https://www.group-ib.com/resources/ransomware-notes/`): colección de notas de vendor que
   apareció en la búsqueda y **no está en la lista blanca todavía**. Sirve como fuente de
   procedencia y como material de recolección futura.

⚠️ Apareció también **`ransomizer.com`**, un generador de notas de rescate. Se anota por
completitud: **NO hay ninguna evidencia que lo vincule con este corpus**, y no debe citarse como
hipótesis sin evidencia.

## 📌 PISTA 3 y 4 VERIFICADAS: KAGGLE Y GROUP-IB TAMPOCO SON EL ORIGEN (2026-08-23)

### ⛔ PISTA 3 DESCARTADA — dataset de Kaggle «Ransomware Note Dataset Collection»

Romina lo descargó (`Downloads/archive/_label_ransomwareOnly.txt`, 769 notas en formato
fastText, 818 KB, **un solo `.txt`, sin ejecutables**). Comparado contra el corpus con criterio
estricto: **coincidencia de ≥200 caracteres normalizados** (minúsculas, sin puntuación),
descartando las líneas de menos de 150 caracteres.

- **72 de las 155 notas del corpus (46 %) están en ese dataset.** Era esperable: los dos beben
  de las mismas fuentes públicas (ThreatLabz, Lemmou, pcrisk), así que **el solapamiento NO
  prueba que el corpus haya copiado de ahí**.
- **De las 8 sin fuente, solo 1 aparece:** `RYUK/note_variant_email.txt` (línea 211), y ya se
  sabía que es `RYUK/note_pcrisk.txt` **sin la primera línea**. **Las otras 7 NO están.**

> ⚠️ **Corrección de un error propio, para que quede el número bueno:** una primera pasada
> reportó «3 de 8» y «41 de 155». **Estaba mal**: el criterio de subcadena matcheaba contra una
> línea del dataset que tiene **un solo carácter**, así que daba positivo con cualquier texto.
> Las cifras válidas son las de arriba (72/155 y 1/8), con umbral de 200 caracteres.

**Dos subproductos que sí valen:**

1. **`hellokitty_note2.txt` tiene camino de procedencia.** En Kaggle (línea 244) hay una nota
   **casi idéntica** pero con el **correo real `moremo123123@cock.li`** y con `[snip]` —la
   convención de ThreatLabz—. O sea: **ese texto existe con sus marcadores reales**, y la
   versión del corpus es esa misma con los marcadores sustituidos por `[EMAIL_ADDRESS]` y
   `[VICTIM_ID]`. Refuerza que los placeholders son **sustitución posterior**, no fabricación
   del texto.
2. **Chequeo cruzado independiente del conteo de plantillas.** Varias notas del corpus colapsan
   a la **misma línea** de Kaggle: 6 archivos de CERBER → línea 70 · 3 de DARKSIDE → línea 92 ·
   3 de NETWALKER → línea 85 · 3 de RYUK → línea 211 · 5 de GANDCRAB → línea 22. Es evidencia
   **externa** de que esos grupos son la misma plantilla, y coincide con las reescrituras
   detectadas por la auditoría. **Sirve como validación de la limpieza que se decida hacer.**

### ⛔ PISTA 4 DESCARTADA (para esto) — Group-IB «Notes From the Most Active Ransomware Groups»

Publica **texto completo** de notas de **43 grupos ACTIVOS** (LockBit, Black Basta, Cl0p, Akira,
BianLian, Medusa, Everest, y de 2025 Dire Wolf, Devman, FunkSec). **No incluye ninguna de las 6
familias con problema** —BadRabbit, Chimera, CryptoLocker, HelloKitty, NotPetya, Ryuk—, y es
coherente: cubre grupos vigentes, no históricos.

> ✅ **PERO es material de primera para el experimento de EXTENSIÓN a familias nuevas** (item
> «Ext.» de `EXPERIMENTOS_PENDIENTES.md`, pendiente de decisión con Cappo): 43 grupos con texto
> completo y vendor reconocido, mejor fuente que scrapear. **Evaluar sumarlo a la lista blanca.**

### 📄 Fuente identificada en `Downloads`: CCN-CERT ID-20/17

`ccn-certid-20-17ransomnotpetya1499243530.pdf` es el **informe oficial del CCN-CERT (España),
ID-20/17, «Código dañino Ransom.Petya/NotPetya», julio de 2017**. Es el respaldo citable de
`README.TXT` como nombre de nota de NotPetya. **Mover a `5_bibliografia/` y citarlo.**

### Estado del rastreo de origen: CERRADO SIN RESULTADO

**Cuatro pistas verificadas y descartadas** (repo GitLab de Trujillo · convención de
placeholders publicada · dataset de Kaggle · Group-IB). **7 de las 8 notas no aparecen en
ninguna fuente revisable.** Lo que queda es **preguntarle a Carlos**: `Pruebas.xlsx` es material
compartido y el lote fundacional es anterior al seguimiento de procedencia.

⚠️ Se mantiene la nota de honestidad: apareció `ransomizer.com` (generador de notas) en una
búsqueda. **No hay evidencia que lo vincule con este corpus y NO debe citarse como hipótesis.**

## ⛔⛔⛔ ERROR DE ETIQUETA CONFIRMADO: `hellokitty_note2.txt` ES UNA NOTA DE DHARMA (2026-08-25)

> **Lo encontró Romina buscando a mano.** Es el hallazgo más importante de la auditoría: no es
> un problema de procedencia, es una **etiqueta de familia equivocada**, y afecta el
> entrenamiento del clasificador.

### La evidencia

**`ransomlook.io/notes/dharma`** publica una nota **atribuida a Dharma** que empieza textual:

> «All your files have been encrypted! All your files have been encrypted due to a security
> problem with your PC. If you want to restore them, write us to the e-mail
> **moremo123123@cock.li**»

`3_datos/corpus_v2/HELLOKITTY/hellokitty_note2.txt` es **ese mismo texto**, con los marcadores
sustituidos: el correo por `[EMAIL_ADDRESS]`, el ID por `[VICTIM_ID]` y el correo alternativo
por `[BACKUP_EMAIL]`.

**Confirmación cruzada independiente:** el dataset de Kaggle (línea 244) trae la MISMA nota con
el **correo real `moremo123123@cock.li`** y con `[snip]` (convención ThreatLabz). Y la auditoría
por búsqueda ya había avisado que su frase de apertura devuelve una plantilla que las fuentes
atribuyen **siempre a Dharma/CrySiS o Phobos, nunca a HelloKitty**.

⚠️ **Un detalle a verificar contra la página real antes de citar:** el resumen automático de
ransomlook menciona «hasta 5 archivos» de descifrado gratuito y la nota del corpus dice «up to
3 files». Puede ser variante distinta del mismo molde Dharma (Dharma tiene muchas) o un error
del resumen. **La identificación NO depende de ese detalle** —el correo y la apertura verbatim
son concluyentes— pero conviene mirarlo.

### Consecuencias, y por qué esto sí hay que resolverlo antes de escribir cifras

1. **HELLOKITTY tiene 5 notas y una es de otra familia.** Su F1 es de los más bajos del corpus
   (0,1622 base · 0,2023 con M.6) y ahora hay una causa concreta: se le pide al modelo aprender
   una clase contaminada. **Es el mismo patrón que se documentó el 2026-08-19 con MEDUZALOCKER**
   (2 notas de Medusa dentro de MedusaLocker), y aquel caso se resolvió retirándolas.
2. **Contamina la confusión DHARMA↔HELLOKITTY** y probablemente también PHOBOS, que comparte el
   molde. Conecta con la colisión `info.hta` DHARMA/PHOBOS ya documentada.
3. **Es un error de ETIQUETA, no de procedencia:** aunque se le encuentre fuente, la nota sigue
   estando en la carpeta equivocada.

### Acción propuesta (decisión de Romina + Cappo, NO se tocó nada)

Mover `hellokitty_note2.txt` a `3_datos/descartados_integridad/` —el mismo procedimiento
reversible que se usó con las 2 notas de Medusa el 2026-08-19, con README que documente motivo
y fuente— **o** reetiquetarla como DHARMA. Lo primero es más conservador: sumarla a DHARMA
requeriría confirmar que la variante es de Dharma y no de Phobos, que comparten molde.
**Efecto:** HELLOKITTY 5→4 notas, corpus 155→154, y cambian las cifras de esa familia.

### Fuente nueva a evaluar para la lista blanca: `ransomlook.io`

Publica notas con atribución de familia y **ya está en la cadena del proyecto**: figura entre
los `authors` del propio catálogo MISP que envió el tutor. Sirvió para resolver este caso y
puede servir para las 6 notas sin fuente que quedan.

### 🔴 AMPLIACIÓN: `hellokitty_note1.txt` TAMBIÉN es del linaje Dharma/Phobos (2026-08-25)

Al abrir `hellokitty_note1.txt` completa apareció que **su cierre es el boilerplate del linaje
Dharma/Phobos**. Barrido sobre las 155 notas buscando las dos frases
(«Do not try to decrypt your data using third party software, it may cause permanent data
loss» / «Decryption of your files with the help of third parties may cause increased price»):

| Familia | Notas con ese cierre |
|---|---|
| **DHARMA** | **12** (los 11 `.hta` + `dharma.txt`) |
| **PHOBOS** | 2 |
| **HELLOKITTY** | **2 — exactamente las dos fundacionales sin fuente** |
| LOCKBIT | 2 (solo la segunda frase; más débil, puede ser fraseo común) |

**Ninguna de las otras 3 notas de HELLOKITTY lo trae**: `pcrisk_hellokitty_1.txt`
(`read_me_unlock.txt`, ataque a CD Projekt), `pcrisk_hellokitty_2.txt` (`read_me_ldk.txt`) y la
de ThreatLabz son del estilo de **caza mayor** («more than 200 GB of critical data»), que es lo
que ransomlook publica como nota de HelloKitty. **Son dos cosas distintas dentro de la misma
clase.**

**Niveles de certeza, que hay que respetar al escribirlo:**
- `hellokitty_note2.txt` → **CONFIRMADO Dharma**: ransomlook lo atribuye a Dharma y el correo
  `moremo123123@cock.li` coincide.
- `hellokitty_note1.txt` → **fuerte sospecha, no confirmado**: comparte el boilerplate del
  linaje, no se le encontró fuente, usa los mismos placeholders y está en la misma carpeta que
  la confirmada. **No hay una fuente que la atribuya explícitamente a Dharma.**

> **Explica el F1 de HELLOKITTY (0,1622 base · 0,2023 con M.6), uno de los más bajos del
> corpus: 2 de sus 5 notas son de otro linaje.** Es el tercer caso del mismo patrón, después de
> MEDUZALOCKER (2 notas de Medusa, resuelto el 2026-08-19) y CRYPTOLOCKER (Crypt0l0cker =
> TorrentLocker). **Conviene revisar el resto del corpus con este método** —buscar boilerplate
> compartido entre familias— porque encontró en un paso lo que las búsquedas por texto no
> habían resuelto.

### ✋ `ransomlook.io`: CARACTERIZACIÓN CORREGIDA (2026-08-25)

> **Corrección de una afirmación mía que era demasiado dura.** Yo había escrito que ransomlook
> «no alcanza el estándar de fuente citable». **Romina lo cuestionó y tenía razón.**

**Lo que sí se verificó a favor:**
- **MISP lo usa como fuente:** aparece **589 veces** en el catálogo que envió el tutor y figura
  entre sus `authors`.
- Tiene **atribución de familia por grupo** — es lo que permitió identificar que
  `hellokitty_note2.txt` es de Dharma.
- Es un proyecto abierto con procedencia conocida, que es literalmente el criterio de
  `CLAUDE.md` («paper, dataset oficial, **repo con procedencia conocida**»).

**Sus limitaciones reales, que son otras:**
- **No trae autor ni fecha por nota**, a diferencia de pcrisk (Tomas Meskauskas, con fecha) o de
  un advisory oficial.
- **A veces republica material de terceros sin declararlo.** Verificado en el acto: su nota de
  HelloKitty es **exactamente** `HELLOKITTY/[File_Name].README_TO_RESTORE` del corpus, incluidos
  los `[snip]` y el `[binary_data]` — o sea, **es la nota de ThreatLabz reproducida**, no
  material propio.

> **Caracterización correcta:** fuente **secundaria de agregación**, usable **con la limitación
> declarada**, y **hay que verificar caso por caso si el material es suyo o espejo de otro**. Es
> el mismo trato que el proyecto le da a `id-ransomware.blogspot` (en lista blanca porque SANS
> lo recomienda). **No es «no citable».**

**Decisión de alcance de Romina, que se mantiene:** no se incorporan notas de ransomlook al
corpus. Pero el fundamento es de alcance, no de calidad de la fuente.

**Y el hallazgo del error de etiqueta no depende de ella**, lo cual sigue siendo lo importante:
se sostiene con el dataset de Kaggle (línea 244, correo real `moremo123123@cock.li`) y sobre
todo con **el propio corpus** (boilerplate compartido con 12 notas de DHARMA y 2 de PHOBOS, y
con ninguna otra de HELLOKITTY). **Al escribirlo, apoyarse en esa última.**

## ★★★★ RE-MEDICIÓN COMPLETA SOBRE EL CORPUS LIMPIO: 149 NOTAS (2026-08-25)

> **Se AGREGA, no reemplaza.** Todas las corridas sobre 155 quedan intactas con su propia base;
> esto es una re-medición sobre el corpus **después de la limpieza de integridad** del 24-25 de
> agosto. Salidas en carpetas NUEVAS con sufijo `_149`.
> ⚠️ Hay además una corrida intermedia `resultados_notas_150` y
> `resultados_grafo_marcadores_150` que quedó **SUPERADA** (se hizo antes de retirar
> `LOCKBIT/lb20.txt`). No citarla.

### (1) Qué se limpió, y por qué el corpus pasó de 155 a 149

**Seis notas retiradas** a `3_datos/descartados_integridad/` (movidas, no borradas; reversible)
y **tres incorporadas** transcritas verbatim. Detalle completo en el README de esa carpeta.

| # | Nota | Motivo |
|---|---|---|
| 1 | `HELLOKITTY/hellokitty_note2.txt` | **CONFIRMADO: es de Dharma.** ransomlook lo atribuye a Dharma con el correo real `moremo123123@cock.li`; Kaggle lo trae con ese correo; comparte el boilerplate con 12 notas de DHARMA |
| 2 | `HELLOKITTY/hellokitty_note1.txt` | linaje Dharma/Phobos (mismo boilerplate, sin fuente) — **sospecha fuerte, no confirmada** |
| 3 | `CHIMERA/chimera_note1.txt` | alemán sintético; la auténtica (`hns_chimera_aleman.txt`) **ya estaba en el corpus** |
| 4 | `CHIMERA/chimera_note2.txt` | inglés sintético; la auténtica (`pcrisk_chimera_ingles_autentico.txt`) ya estaba |
| 5 | `CRYPTOLOCKER/cryptolocker_note2.txt` | sintética; dice «RSA-2048 **and AES-256**» y CryptoLocker no dejaba archivo de nota |
| 6 | `LOCKBIT/lb20.txt` | **duplicado por defanging**: mismo texto que `lockbit2.txt` (477 car. los dos), solo cambia `hxxp`→`http` y `.oni0n`→`.onion` |

**Tres incorporadas, verbatim desde Amigo-A** (`tipo=transcripcion`), reemplazando material
sintético de BADRABBIT y NOTPETYA: `idr_badrabbit_pantalla_2017.txt` (SIN_ARCHIVO, `key#1`),
`idr_badrabbit_readme_2017.txt` (`README.txt`, `key#2`) e
`idr_notpetya_readme_var2_2017.txt` (`README.TXT`, «otra variante»).

**Erratas de la fuente conservadas a propósito** (corregirlas sería repetir el error que se
está limpiando): `key#l` con **ele minúscula** (byte `0x6c`) en la pantalla de BadRabbit, y
`waste your **tine**` en la variante 2 de NotPetya. Y las **truncaduras son de la fuente**
(`ZORqoZdoI+vr6*****`, `*****`, `***`), no del corpus — eso es lo que las distingue de los
placeholders sintéticos retirados.

**NO incorporada, y hay que decir por qué:** la «una variante» de NotPetya está **corrupta en
la propia fuente** (líneas desordenadas, falta la dirección de correo). Incorporarla habría
reintroducido el problema.

**Deuda de procedencia: 47 → 0.** Las 149 notas tienen fuente citable.

### (2) Las cifras nuevas, contra las de 155

| | 155 | **149 (limpio)** |
|---|---|---|
| Notas · plantillas | 155 · 106 | **149 · 99** |
| P2 base (texto solo, M.6, 50 semillas) | 0,4958 ± 0,0733 | **0,4593 ± 0,0752** |
| **M.6** (IOCs privados + nombre) | 0,5471 | **0,5191** |
| **Δ pareado** | +0,0513 [+0,0453; +0,0573] | **+0,0599 [+0,0530; +0,0668]**, 50/50 |
| Cobertura (de la cual por nombre) | 0,4547 (0,1155) | **0,4546 (0,1180)** |
| Acierto donde aplica | 0,9805 | 0,9755 |
| Aporte aislado del nombre | +0,0088 [+0,0062; +0,0114] | **+0,0108 [+0,0077; +0,0138]** |
| L (protocolo Lemmou) | 0,8055 | **0,7767** |
| P1 | 0,8020 | **0,7889** |

> **M.6 salió MÁS FUERTE en el corpus limpio** (+0,0599 contra +0,0513), y el aporte del nombre
> también subió. Los valores absolutos bajan porque se retiraron notas; **lo que mide el método
> mejoró.**

**Clasificador canónico sobre 149** (`resultados_notas_149`): P2 grupos+combinado+LinearSVC
**macro-F1 0,468 ± 0,100** · bal.acc 0,528 · acc 0,580. P1 estratificado+combinado **0,799 ±
0,026** (caracteres 0,805). **Grafo B.3 sobre 149:** 99 plantillas → 101 nodos, marcadores URL
153 · ONION 98 · EMAIL 79 · CLAVE 33 · BTC 6 · ID 6, **13 plantillas sin ningún marcador**.

### (3) ★ EL CERO ESTRUCTURAL, AHORA MEDIDO — y cómo hay que reportarlo

**BADRABBIT y CRYPTOLOCKER quedaron con 1 plantilla y su F1 es exactamente 0,0000.** El propio
`clasificador_notas_v2.py` lo advierte solo: «familias con un único grupo de contenido (con el
protocolo grupos nunca aparecen en train y test a la vez, su F1 tenderá a 0)».

**Y en las tres es una propiedad REAL del malware, no un hueco de recolección:**

| Familia | Plantillas | Por qué 1 |
|---|---|---|
| WASTEDLOCKER | 1 molde | 4 notas de **3 víctimas** y **2 fuentes**; solo cambian víctima y correos |
| BADRABBIT | 1 | las dos variantes auténticas (pantalla y `README.txt`) son el mismo texto: difieren solo en `key#1` vs `key#2` |
| CRYPTOLOCKER | 1 | era una **ventana** con un solo mensaje |

| Conjunto | Familias | F1 base | F1 M.6 |
|---|---|---|---|
| Todas | 30 | 0,4593 | 0,5191 |
| **Evaluables (sin las de 1 plantilla)** | **28** | **0,4921** | **0,5562** |

> **+0,037 de macro-F1 solo por dejar de promediar dos ceros estructurales.** No es un truco: es
> no contar como fracaso del método algo que el método **no puede evaluar por construcción**.
> **Reportar las dos cifras**, con la lista de familias inevaluables y la prueba de que su única
> plantilla es propiedad del malware (varias víctimas / varias fuentes / mismo molde).

### (4) ⚠️ UNA PREDICCIÓN MÍA QUE FALLÓ: HELLOKITTY no subió

Se predijo que HELLOKITTY subiría al retirarle las 2 notas de linaje Dharma. **No pasó:**
F1 base 0,1622 → **0,1714**, con M.6 0,2023 → **0,1741**. Prácticamente igual.

**Por qué:** su cohesión entre plantillas es **0,2643** — las 3 notas que quedan (el ataque a CD
Projekt, `read_me_ldk`, y la de ThreatLabz) son de campañas y víctimas distintas y se parecen
poco entre sí. **HELLOKITTY es genuinamente difícil, no solo estaba contaminada.** La limpieza
se justifica por corrección, **no por rendimiento**, y así hay que escribirlo.

### (5) Error de código encontrado y corregido

`techo_por_familia.py` leía las columnas de F1 desde la corrida de M.6 de **155** con una ruta
fija, así que la primera pasada sobre 149 salió con columnas viejas. Se le agregó `--m6`. **Al
re-medir cualquier cosa, verificar de dónde lee cada script sus insumos.**

### (6) Lo que queda pendiente de re-correr

`curva_aprendizaje_notas.py` (B.1) y `cascada_ioc_notas.py` (M.1) sobre 149, y
`resumen_para_capitulo4.py` para regenerar las tablas del capítulo.

### ✅ LA LIMPIEZA SÍ FUNCIONÓ: 166 CONFUSIONES ELIMINADAS (2026-08-25, matriz de confusión)

> **Corrige la lectura anterior de que «la limpieza se justifica por corrección, no por
> rendimiento».** Es demasiado pesimista: el efecto es grande y medible, solo que no se ve en el
> F1 de HELLOKITTY. Fuente: `m6_diagnostico_nota_por_nota.csv` de las corridas 155 y 149.

**Invasiones a HELLOKITTY** (notas de OTRAS familias predichas como HELLOKITTY, variante
adoptada, 50 semillas):

| | 155 (contaminado) | 149 (limpio) |
|---|---|---|
| **Total** | **314** | **47** |
| desde DHARMA | **146** | **0** |
| desde PHOBOS | 20 | **0** |
| desde CUBA | 49 | 0 |

**Las 146 confusiones DHARMA→HELLOKITTY y las 20 de PHOBOS desaparecieron por completo.** Era
exactamente el efecto que causaban las 2 notas de linaje Dharma metidas en HELLOKITTY, y la
limpieza lo eliminó.

### Por qué el F1 de HELLOKITTY no subió igual: precisión sí, recall no

El F1 mezcla dos cosas y solo una mejoró.
- **Precisión: mejoró mucho** (314 → 47 invasiones).
- **Recall: sigue malo.** De sus propias notas acierta solo el **15 %**. Sus 3 notas se dispersan
  a MAZE (19 %), RANSOMEXX (17 %), BLACKBASTA (13 %) y CLOP (10 %).

> **El clasificador ya no confunde otras familias CON HelloKitty, pero sigue sin reconocer A
> HelloKitty.** Y eso es el problema estructural de siempre: 3 plantillas con cohesión 0,2643,
> de campañas y víctimas distintas (CD Projekt, `read_me_ldk`, ThreatLabz). Bajo P2 con 2
> pliegues entrena con una o dos y evalúa en la otra.

**Cómo escribirlo:** la limpieza de integridad **mejoró la precisión de forma masiva** y eso hay
que reportarlo con estas cifras; el recall de HELLOKITTY es un límite de dato (pocas plantillas,
poco cohesionadas) que ninguna limpieza puede arreglar. **Un F1 plano puede esconder una mejora
grande** — reportar precisión y recall por separado en las familias afectadas.

## 📌 RESULTADO M.3 — ABSTENCIÓN POR UMBRAL DE CONFIANZA (2026-08-25)

> **Base: 149 notas · 99 plantillas · 30 familias · P2 · 50 semillas.** Código:
> `2_codigo/abstencion_notas.py`. Salidas en `4_resultados/resultados_abstencion_149/`.
> **Predicción preregistrada (PLAN_MEJORAS.md §M.3, escrita antes de implementar): «no sube el
> macro-F1, cambia el reporte». SE CUMPLE.**

**Cómo se mide la confianza:** LinearSVC no da probabilidades, así que se usa el **margen entre
la primera y la segunda clase** de `decision_function`. Arquitectura: si la regla exacta de M.6
aplica se contesta siempre (ya mide 0,9755 de acierto); si no aplica, decide el texto y ahí se
aplica el umbral.

### La curva precisión-vs-cobertura — M.6 (reglas + texto)

| Umbral | Cobertura | **Acierto donde contesta** | macro-F1 de las respondidas | Abstiene |
|---|---|---|---|---|
| 0,00 (sin abstención) | 1,0000 | 0,6601 | 0,5191 | 0 |
| 0,10 | 0,8244 | 0,7796 | 0,6292 | 26 |
| 0,30 | 0,7019 | 0,8649 | 0,7500 | 44 |
| **0,50** | **0,6459** | **0,9000** | 0,7938 | 53 |
| 1,00 | 0,5495 | **0,9687** | 0,9129 | 67 |
| 1,50 | 0,4681 | 0,9763 | 0,9365 | 79 |

**La frase para la defensa:** «el sistema **contesta el 65 % de las veces y acierta el 90 %**»
(umbral 0,50). O, si se prefiere más conservador: «contesta el 55 % y acierta el 97 %».

### Y el mismo ejercicio sin la capa de reglas, que prueba que M.6 aporta

| A cobertura ≈ 65 % | Acierto |
|---|---|
| **M.6 (reglas + texto)**, umbral 0,50 | **0,9000** |
| solo texto, umbral 0,20 (cobertura 0,6536) | 0,7620 |

**A igual cobertura, M.6 acierta +0,14 más que el texto solo.** Es una segunda demostración
independiente del valor de la cascada, distinta del Δ de macro-F1.

**Piso de cobertura:** la capa de reglas resuelve **67,7 notas de 149 (45 %)** en toda semilla,
así que la cobertura de M.6 nunca baja de ~0,47 aunque el umbral suba. El texto solo, en cambio,
degenera: a umbral 1,50 contesta 5,5 notas de 149 (cobertura 0,037).

### Lectura honesta, para no sobrevender

1. **El macro-F1 global NO mejora.** A umbral 0 es exactamente 0,5191, el valor de M.6. El
   macro-F1 «de las respondidas» sube hasta 0,9365 pero **se calcula solo sobre las notas que
   contesta**: no es comparable con el macro-F1 del sistema completo y **no debe citarse como
   si lo fuera**. Lo que cambia es **qué afirma el sistema**, no cuánto sabe.
2. **La abstención no resuelve el mundo cerrado.** Una nota de una familia fuera de las 30 puede
   caer con margen alto y ser respondida con confianza y error. Esto **reduce** el problema, no
   lo elimina; el mundo cerrado sigue siendo una limitación declarada.
3. El umbral es un **parámetro de despliegue**, no un hiperparámetro ajustado a los datos: se
   reporta la curva completa y el punto de operación se elige según el costo del error.

**Estado: M.3 CERRADO.** Sale de `EXPERIMENTOS_PENDIENTES.md`.

## 📌 RESULTADO M.1 — CASCADA IOC→TEXTO RE-MEDIDA SOBRE 149 (2026-08-28)

> **Base: 149 notas · 99 plantillas · 30 familias · P2 · 10 semillas.** Salidas en
> `4_resultados/resultados_cascada_149/`, log en `4_resultados/_log_cascada_149.txt`. La
> corrida sobre 155 (`resultados_cascada_155/`) queda **intacta con su propia base**: esto se
> AGREGA. **Veredictos idénticos a los de 155: sin circularidad SE ADOPTA, con circularidad NO.**

### ⚠️ Antes hubo que arreglar el script: la puerta de entrada estaba clavada en 155

`cascada_ioc_notas.py` tenía `BASE_MACRO_F1 = 0.5265` (base 155) con `TOL_BASE = 0.003` y un
`sys.exit`. Sobre 149 la capa de texto da 0,4680, o sea 0,059 de diferencia: **el script abortaba
sin escribir nada**. Es el mismo patrón que se encontró en `techo_por_familia.py`. Arreglado con
`--base-desde <corrida_canonica_resumen.csv>`, que lee la fila `grupos/combinado/LinearSVC` del
**mismo** corpus que se está midiendo, más `--base` / `--base-std` / `--tol` para declararla a
mano; los valores por defecto siguen siendo los de 155, así que reproducir la corrida vieja no
cambia. El manifiesto ahora registra la base **realmente usada** y su fuente. Commits `f09151e`
y `4b0febc` en develop (el segundo saca el «base 155» fijo del encabezado impreso).

**Puerta de entrada superada exactamente:** capa de texto sola **0,4680 ± 0,1003**, diferencia
**0,0000** contra la base leída de `resultados_notas_149/corrida_canonica_resumen.csv`. La
partición es la misma, así que el Δ pareado por semilla es válido.

**IOCs en el corpus de 149** (patrones canónicos de `normalizacion_marcadores.PATRONES`):
URL 219 · ONION 121 · EMAIL 87 · CLAVE 35 · ID 8 · BTC 6. **15 de 149 notas no tienen ningún
IOC** (sobre 155 eran 18).

### (1) Las tres columnas del Exp. 2b — 149 contra 155

| Variante | Base | Cobertura | Acierto donde aplica | macro-F1 combinado | Exactitud |
|---|---|---|---|---|---|
| texto solo | 155 | — | — | 0,5265 ± 0,0490 | 0,6026 |
| texto solo | **149** | — | — | **0,4680 ± 0,1003** | **0,5799** |
| sin circularidad | 155 | 0,2613 ± 0,0264 | 0,9814 ± 0,0241 | 0,5453 ± 0,0485 | 0,6348 |
| **sin circularidad** | **149** | **0,2523 ± 0,0392** (37,6 notas) | **0,9561 ± 0,0635** | **0,4959 ± 0,0994** | **0,6208** |
| con circularidad | 155 | 0,2419 ± 0,0287 | 0,9719 ± 0,0370 | 0,5323 ± 0,0486 | 0,6239 |
| con circularidad | **149** | 0,2302 ± 0,0376 (34,3 notas) | 0,9450 ± 0,0883 | 0,4742 ± 0,1058 | 0,6067 |

**Δ pareado por semilla contra la base (t de Student, df=9):**

| Variante | Base | Δ macro-F1 | IC 95 % | sem. Δ>0 | Δ exactitud | IC 95 % |
|---|---|---|---|---|---|---|
| sin circularidad | 155 | +0,0188 | [+0,0128; +0,0248] | 10/10 | +0,0323 | [+0,0236; +0,0410] |
| **sin circularidad** | **149** | **+0,0279** | **[+0,0124; +0,0434]** | **10/10** | **+0,0409** | [+0,0255; +0,0564] |
| con circularidad | 155 | +0,0058 | [−0,0013; +0,0130] | 6/10 | +0,0213 | [+0,0120; +0,0305] |
| con circularidad | **149** | +0,0062 | [−0,0024; +0,0148] | 8/10 | +0,0268 | [+0,0162; +0,0375] |

Exactitud balanceada sobre 149: base 0,5281 → **0,5575** (sin circularidad) · 0,5328 (con).

**El Δ CRECIÓ con el corpus limpio: +0,0188 → +0,0279.** Es el mismo comportamiento que M.6
(+0,0513 → +0,0599): los valores absolutos bajan porque se retiraron notas, y **lo que mide el
método mejora**. Tercera corrida independiente que muestra el mismo signo.

**Donde la regla aplica sigue ganándole al texto, y por más:** acierto de la regla **0,9561**
contra **0,7932** del clasificador de texto sobre esas mismas notas → **+0,1629** (sobre 155 era
+0,1244). La regla no cubre lo fácil: cubre notas que el texto erraba en 1 de cada 5.

### (2) Veredicto contra el criterio preregistrado — sin cambios respecto de 155

- **SIN filtro de circularidad → SE ADOPTA.** Δ +0,0279, IC 95 % [+0,0124; +0,0434] excluye el
  cero, **10/10 semillas** positivas.
- **CON filtro → NO se adopta.** Δ +0,0062, IC 95 % [−0,0024; +0,0148] **incluye** el cero
  (8/10 semillas). Sigue siendo regla de alta precisión y baja cobertura que no mueve el agregado.
- **Los dos controles negativos siguen sin ser tocados por la regla en ninguna semilla:** RYUK y
  HELLOKITTY con Δ exactamente 0,0000 en las dos variantes. El mecanismo queda probado otra vez,
  ahora sobre el corpus donde HELLOKITTY ya no tiene las dos notas de Dharma.

**F1 por familia con IC 95 % que excluye el cero (sin circularidad, 149):** LORENZ +0,129
[+0,034; +0,224] · BLACKBASTA +0,101 [+0,019; +0,183] · PHOBOS +0,078 [+0,024; +0,133]. Suben
pero con IC que incluye el cero: TESLACRYPT +0,041 · SODINOKIBI +0,034 · CLOP +0,029. Sin
movimiento: BLACKMATTER, NOTPETYA, GANDCRAB, NETWALKER (Δ exactamente 0,000).

### (3) Diagnóstico post-hoc (NO preregistrado, NO adoptable) — solo IOCs privados

Descartando del diccionario los valores que en el entrenamiento aparecen en más de una familia
(el filtro de genéricos de B.3), que es lo que mide cuánta cobertura bloquean las URLs de Tor:

| Variante post-hoc | Cobertura | Acierto donde aplica | macro-F1 | Δ | IC 95 % |
|---|---|---|---|---|---|
| sin circ. + solo privados | **0,3872** (57,7 notas) | **0,9686** | 0,5237 | +0,0557 | [+0,0381; +0,0732] |
| con circ. + solo privados | 0,3235 (48,2 notas) | 0,9581 | 0,4961 | +0,0281 | [+0,0139; +0,0423] |

Sigue valiendo la lectura de 155: **el filtro de genéricos casi duplica la cobertura sin perder
precisión**, y es el camino que M.6 ya adoptó. No es adoptable por sí mismo porque se decidió
después de ver los resultados.

### (4) ⚠️ Hallazgo colateral que hay que declarar: la dispersión creció

> ⛔ **CORREGIDO el mismo día con 50 semillas.** El 0,1003 de abajo es una estimación de 10
> semillas y **sobreestima**: sobre el mismo corpus de 149, con 50 semillas el desvío es
> **0,0752** y la media **0,4593** (no 0,4680). La dispersión es **1,53× la de 155**, no el
> doble. Lo que sigue valiendo: es más alta, los IC son más anchos, y **para cifras finales van
> 50 semillas**. Detalle en el bloque «RESULTADO CON 50 SEMILLAS» de la decisión de alcance.

La base P2 pasó de **0,5265 ± 0,0490** (155, 10 semillas) a **0,4680 ± 0,1003** (149, 10
semillas): el desvío **casi se duplicó con esa estimación** (ver la corrección de arriba). El F1 por semilla de la capa de texto va de **0,2954**
(semilla 3) a **0,5988** (semilla 0). Consecuencia práctica: **todos los IC 95 % de 149 son más
anchos que los de 155** — el de M.1 pasó de ±0,006 a ±0,016 de semiancho. Los veredictos no
cambian, pero cualquier Δ chico medido sobre 149 con 10 semillas va a tener IC que incluya el
cero. **Para cifras finales sobre 149, usar 50 semillas** (es lo que ya hacen M.6 y M.3).

**Dato relacionado, medido, pero que NO alcanza para explicarlo.** El control de corrección de
`curva_aprendizaje_notas.py --validar` reporta cuántas familias quedan **sin ninguna nota de
entrenamiento** en promedio bajo P2: **2,8 sobre 155** (`resultados_extension_155/log_curva.txt`)
contra **3,9 sobre 149** (`_log_curva_149_validar.txt`). Subió, y 2 de esas 3,9 son BADRABBIT y
CRYPTOLOCKER, que con 1 sola plantilla caen enteras en un pliegue **en toda semilla**. Pero eso
es un aporte *constante*: baja la media, no agrega varianza. **La causa del salto de dispersión
queda sin establecer** — anotado como pregunta abierta, no como explicación. El control de
corrección sí pasó exacto en las dos bases (P1 y P2 reproducen el evaluador canónico con
diferencia 0,00e+00), así que no es un error de partición.

**Estado: M.1 re-medido y cerrado sobre 149.** Falta la curva B.1 y el resumen del capítulo 4.

## ⛔ CORRECCIÓN AL TRASPASO: SON **DOS** FAMILIAS DE 1 PLANTILLA, NO TRES (2026-08-28)

> El traspaso `HANDOFF_2026-08-25_limpieza_y_proximos_pasos.md` §3(b) dice «las tres familias de
> 1 plantilla: WASTEDLOCKER · BADRABBIT · CRYPTOLOCKER». **Medido contra el agrupador de
> casi-duplicados (coseno char 3-5 > 0,90, el mismo de todas las corridas): son DOS.** El
> traspaso ya quedó corregido.

| Familia | Notas | Plantillas | Cosenos internos medidos |
|---|---|---|---|
| **BADRABBIT** | 2 | **1** | 0,9490 → colapsan (las 2 variantes auténticas difieren en `key#1` vs `key#2`) |
| **CRYPTOLOCKER** | 2 | **1** | 0,9883 → colapsan |
| WASTEDLOCKER | 4 | **3** | 0,8861 · 0,8879 · 0,8957 · 0,8977 · 0,8995 · **0,9742** (solo este último par colapsa) |

**Lo que NO cambia:** el número que va a la decisión de reporte. Verificado sobre
`resultados_cascada_combinada_149/m6_por_familia.csv`: las únicas familias con F1 = 0,0000 en
base **y** en variante son **BADRABBIT y CRYPTOLOCKER**, en las cuatro variantes de M.6. Las
**28 evaluables** dan base **0,4921** y M.6 **0,5562** (contra 0,4593 y 0,5191 sobre 30):
**+0,0328 en la base y +0,0371 en M.6**. La propuesta para Cappo se sostiene igual.

**Lo que sí cambia, y es lo interesante:** WASTEDLOCKER **es evaluable**, y su caso es el opuesto
al que decía el traspaso. Sus 4 notas están **pegadas al umbral por debajo** (0,886 a 0,900; una
a **0,8995**, tres milésimas del corte). Cualitativamente son el mismo molde —eso era correcto—,
pero el agrupador las cuenta como 3 plantillas. Es una familia cuyo conteo de plantillas
**depende del umbral**, no una familia de plantilla única. Como argumento para Cappo es más
fuerte planteado así: BADRABBIT y CRYPTOLOCKER son inevaluables por propiedad del malware, y
WASTEDLOCKER muestra que el umbral de 0,90 tiene casos de frontera.

**Contexto que además cambió con la limpieza:** sobre 155 **ninguna** familia tenía 1 sola
plantilla (8 tenían 2); sobre 149 hay **2 con una** y 7 con dos. La distribución completa de
plantillas por familia: 155 → `{2:8, 3:5, 4:13, 5:1, 6:2, 8:1}`; 149 → `{1:2, 2:7, 3:6, 4:12,
5:1, 6:1, 8:1}`. Eso es lo que hace estructuralmente inevaluables a esas dos familias bajo P2.

### Y de paso, una corrección al §5-3 del traspaso: una de las 4 reescrituras SÍ infla el conteo

El traspaso afirma que las 4 reescrituras conservadas «no inflan el conteo, porque su coseno con
el original es ≥ 0,90 y el agrupador ya las colapsa». **Medido: vale para 3 de las 4.**

| Reescritura | Coseno máx. con una hermana | ¿La colapsa el agrupador? |
|---|---|---|
| `RYUK/note_variant_email.txt` | 0,9861 | sí (grupo de 3) |
| `MAZE/malware_notes_maze.txt` | 0,9837 | sí (grupo de 2) |
| `LOCKBIT/lb30b.txt` | 0,9132 | sí (grupo de 2) |
| **`WASTEDLOCKER/note_pcrisk.txt`** | **0,8977** | **NO — es plantilla propia (grupo 145)** |

Esa nota es, ella sola, la razón por la que WASTEDLOCKER cuenta 3 plantillas y no 2. Sigue
declarada como reescritura, pero **hay que declarar también que aporta una plantilla al conteo**,
porque la moneda de B.1 son las plantillas.

## ★★★ B.1 RE-MEDIDA SOBRE 149: EL CORTE SE MUEVE DE 4 A 3 PLANTILLAS (2026-08-28)

> **Base: 149 notas · 99 plantillas · 30 familias.** R = 100 repeticiones de retención para
> P2ret y los subconjuntos; R = 10 para P1 y P2 sin retención. Salidas en
> `4_resultados/resultados_curva_149/` y `4_resultados/resumen_cap4_149/`; logs
> `_log_curva_149_validar.txt`, `_log_curva_149.txt`, `_log_resumen_cap4_149.txt`.
> **Con esto queda cerrada la re-medición del frente de notas sobre el corpus limpio.**

**Control de corrección exacto en las dos ramas:** P1 canónico 0,804779 vs curva 0,804779 ·
P2 canónico 0,468038 vs curva 0,468038, **diferencia 0,00e+00** en las dos. La partición y el
evaluador son los mismos que los del clasificador canónico, así que la curva es comparable.

### (1) La respuesta al pedido del tutor cambió de número: son 3 textos distintos, no 4

⚠️ **Y no es efecto de la limpieza: es OTRA BASE.** La B.1 cerrada el 2026-08-19 era sobre
**144 notas / 95 plantillas**. Entre esa corrida y esta hubo recolección (hasta 155 notas / 106
plantillas) **y después** la limpieza de integridad (149 / 99). Lo que cambió el corte es tener
más plantillas repartidas en más familias, no haber retirado 6 notas. Prueba: las familias con
≥ 4 plantillas pasaron de **11 a 15**, y las con ≥ 5 de **5 a 3**.

**Los tres conjuntos concuerdan: el último paso que mueve la aguja es 2 → 3.**

| Conjunto (149) | Δ 1→2 | Δ 2→3 | Δ 3→4 (o 3→todo) |
|---|---|---|---|
| 30 familias · P2ret · plantillas | **+0,0786** [+0,0643; +0,0928] sí | **+0,0146** [+0,0054; +0,0237] sí | **−0,0057** [−0,0130; +0,0015] **NO** |
| «11fam» = **15 familias** con ≥4 · P2ret | **+0,1069** [+0,0820; +0,1317] sí | **+0,0604** [+0,0393; +0,0815] sí | **−0,0014** [−0,0177; +0,0149] **NO** |
| «5fam» = **3 familias** con ≥5 · P2ret | +0,0581 [+0,0128; +0,1034] sí | **+0,1074** [+0,0586; +0,1561] sí | +0,0254 [−0,0095; +0,0604] **NO** |
| 30 familias · P1 · plantillas | +0,1252 [+0,1054; +0,1449] sí | +0,0253 [+0,0125; +0,0381] sí | +0,0065 [−0,0076; +0,0205] **NO** |
| 30 familias · P2 · plantillas | +0,0354 [+0,0200; +0,0507] sí | −0,0034 [−0,0237; +0,0170] **NO** | +0,0005 **NO** |

Sobre 144 el último paso significativo era **3 → 4** (+0,0297 [+0,0039; +0,0554] en el conjunto
de 5 familias). Sobre 149 ese paso ya no lo es en ningún conjunto. **P2 sin retención satura
todavía antes: en 2 plantillas.**

**Lo que eso cambia en el costo de recolección — a la mitad y algo más:**

| Objetivo | Plantillas nuevas | Familias a completar |
|---|---|---|
| todas las familias con **3** plantillas | **11** | **9** |
| todas con 4 plantillas | 26 | 15 |
| (sobre 144, para llegar a 4) | 33 | 19 |

### (2) El techo estimado, que es la cifra más fuerte del capítulo

Ajuste de ley de potencia con bootstrap, solo donde el conjunto de clases es constante:

| Curva (149) | Techo estimado de macro-F1 | ¿Llega a 0,70? | ¿Llega a 0,50? |
|---|---|---|---|
| 30 fam · **P1** · plantillas | **0,822** [0,803; 0,847] | sí, con 1,2 plantillas/fam | sí, con 0,7 |
| 30 fam · **P2ret** · plantillas | **0,651** [0,639; 0,663] | **NO ALCANZABLE** (0 % del bootstrap) | sí, con 0,9 |
| 30 fam · **P2** · plantillas | **0,470** [0,411; 0,527] | **NO ALCANZABLE** | **NO ALCANZABLE** (16 % del bootstrap) |

**La frase citable, con su métrica y su base:** bajo P2 sobre 149 notas, el techo que estima el
ajuste es **macro-F1 0,470** y **ni siquiera 0,50 es alcanzable agregando notas**. P1 llega a
0,80 con 2,9 plantillas por familia (99 % del bootstrap). Es la forma cuantitativa de la
conclusión que ya tenían B.3 y el protocolo Lemmou: **lo que limita no es el modelo, es el
protocolo de evaluación** — y ahora está con techo, IC y probabilidad de bootstrap.

### (3) Hallazgo NUEVO que estaba mal etiquetado: pasado el corte, sumar notas EMPEORA

El script marcaba como «indistinguible de cero» cualquier paso que no fuera significativamente
positivo, incluidos los que tienen el IC 95 % **entero por debajo del cero**. Sobre 149 hay
cinco, y no son ruido:

| Paso (30 fam · P2ret) | Δ macro-F1 | IC 95 % |
|---|---|---|
| 4 → 6 notas | −0,0069 | [−0,0136; −0,0002] |
| 6 → 8 notas | −0,0082 | [−0,0140; −0,0024] |
| 8 → 12 notas | −0,0085 | [−0,0154; −0,0015] |
| 12 → todo | −0,0067 | [−0,0118; −0,0016] |
| **4 → 5 plantillas** | **−0,0084** | **[−0,0149; −0,0020]** |

**Pasado el punto de saturación, agregar material no es neutro: mide peor, de forma
medible.** Refuerza la regla práctica de B.1 (contar textos distintos, no notas) con un
argumento más fuerte que «no aporta»: aporta negativo. Arreglado en el código (etiqueta de tres
casos, commit `ad05071`), y el CSV `b1_deltas_pareados.csv` ahora trae la columna
`significativo_negativo`.

### (4) ⚠️ TRAMPA DE NOMENCLATURA que hay que declarar al citar

Las etiquetas `11fam` y `5fam` son **nombres históricos, no conteos**. Sobre 149 esos
subconjuntos tienen **15 y 3 familias**, así que su azar es **0,067 y 0,333**, no 0,091 y 0,200.
El resumen **imprimía los valores viejos** porque estaban escritos en el código: corregido para
que los lea de `manifiesto_b1.json` y avise cuando el nombre ya no coincide (commit `db9aab4`).
Las tres curvas no son comparables entre sí; toda tabla tiene que decir cuántas familias y qué
azar.

**Dos ajustes de extrapolación salieron degenerados y NO se pueden citar:** «11fam · P2ret ·
plantillas» da techo 3,669 y «5fam · P2ret · plantillas» 3,661 — un macro-F1 mayor que 1 es
imposible; el ajuste no converge con 3-4 puntos sobre una curva que todavía sube. El script
ahora los marca «⚠ AJUSTE NO VÁLIDO: techo > 1, NO CITAR» en pantalla y con la columna
`ajuste_valido` en el CSV (commit `2c04e07`).

### (5) La lista de recolección sobre 149, y dos confirmaciones que salen de ella

F1 por familia de la misma corrida (30fam · P2ret · k=todo · 149 notas) — **es F1 de la familia,
NO el macro-F1**:

| Familia | Plantillas | Faltan → 3 | F1 hoy |
|---|---|---|---|
| BADRABBIT | 1 | 2 | **0,000 ± 0,000** |
| CRYPTOLOCKER | 1 | 2 | **0,000 ± 0,000** |
| CHIMERA | 2 | 1 | **0,010 ± 0,100** |
| NOTPETYA | 2 | 1 | 0,480 ± 0,040 |
| CUBA · BLACKMATTER · NETWALKER · DARKSIDE · SUNCRYPT | 2 | 1 | 0,994 a 1,000 |
| HELLOKITTY | 3 | 0 | 0,157 ± 0,346 |
| LORENZ | 3 | 0 | 0,735 ± 0,401 |
| CONTI | 3 | 0 | 0,851 ± 0,156 |
| SODINOKIBI | 3 | 0 | 0,860 ± 0,185 |
| AVOSLOCKER | 3 | 0 | 0,983 ± 0,073 |
| **WASTEDLOCKER** | **3** | 0 | **0,984 ± 0,092** |

Ya en 4 o más, 15 familias: BLACKBASTA, RYUK, WANNACRY, JIGSAW, MAZE, MEDUZALOCKER, CLOP,
LOCKBIT, DHARMA, GANDCRAB, PHOBOS, RANSOMEXX, TESLACRYPT, BLACKCAT, CERBER.

1. **Confirma la corrección del traspaso por una vía independiente:** WASTEDLOCKER aparece con
   **3 plantillas y F1 0,984 ± 0,092**. No es una familia inevaluable — es una de las que mejor
   rinden. Las únicas con F1 0,000 son BADRABBIT y CRYPTOLOCKER.
2. **CHIMERA es ahora la peor familia con material: F1 0,010 ± 0,100 con 2 plantillas.** Tiene
   explicación medida: sus dos notas son la alemana y la inglesa, con **coseno 0,2535** entre
   sí. Dos plantillas que no se parecen en nada no le enseñan la familia al modelo; es el caso
   extremo de la correlación cohesión→F1 de B.3.

**Estado: paso (a) del traspaso COMPLETO.** M.1, curva B.1 y resumen del capítulo 4 re-medidos
sobre 149. Lo que sigue es la decisión de reporte para Cappo (28 evaluables) y el frente de
archivos (Exp. 2d + A.3).

## ★★★★ DONDE QUEDA MARGEN EN NOTAS: NO ES MÁS NOTAS, ES OTRO CANAL (2026-08-28)

> **Base: 149 notas · 99 plantillas · 30 familias.** Código: `2_codigo/margen_frente_notas.py`
> (commit `a3cd306` en develop). Salidas en `4_resultados/margen_notas_149/`, log
> `_log_margen_149.txt`. Cruza tres fuentes ya medidas —`resultados_curva_149`,
> `resultados_cascada_combinada_149`, `resultados_techo_por_familia_149`— sin correr nada nuevo.
> **Pregunta que contesta:** con la curva re-medida y M.6 adoptado, ¿en qué conviene gastar el
> esfuerzo que queda?

### (1) La trampa que había que cerrar antes de concluir cualquier cosa: efecto propio vs ajeno

El tope `k` de la curva B.1 se aplica a **todas** las familias a la vez. Para una familia con
`n` plantillas y el paso `k → k+1`:

- si **n ≥ k+1**, el paso le agrega una plantilla **propia**: el Δ mide lo que gana esa familia
  con más dato **suyo**. Es lo único que sirve para decidir si conviene recolectar para ella.
- si **n ≤ k**, el tope **no le muerde** ni antes ni después: su entrenamiento es idéntico y el Δ
  mide solo el efecto **ajeno** (que las OTRAS familias tengan más dato).

Leer el segundo como si fuera el primero lleva a la conclusión contraria. Ejemplo real: en el
paso 2→3, BLACKMATTER (+0,030), CUBA (+0,027) y DARKSIDE (+0,029) «suben» — pero tienen 2
plantillas, así que **no recibieron ninguna plantilla nueva**: lo que mejora es que las demás
familias dejan de robarles notas. Recolectar para ellas no está probado por ese número.

### (2) Cuánto paga de verdad una plantilla más (solo efecto PROPIO, 100 repeticiones)

| Paso | Familias | SUBE | BAJA | sin efecto | Δ mediano |
|---|---|---|---|---|---|
| **1 → 2** | 28 | **22** | 1 | 5 | **+0,0673** |
| **2 → 3** | 21 | 7 | 2 | 12 | +0,0211 |
| **3 → 4** | 15 | **1** | **3** | 11 | +0,0014 |

- **La segunda plantilla es la que vale:** 22 de 28 familias suben, hasta +0,227 (MAZE), +0,210
  (CERBER), +0,207 (MEDUZALOCKER), +0,194 (SODINOKIBI), +0,185 (RANSOMEXX), +0,183 (JIGSAW).
- **La tercera paga en una minoría:** LOCKBIT +0,184 · CLOP +0,086 · BLACKBASTA +0,080 ·
  DHARMA +0,061 · TESLACRYPT +0,040 · CONTI +0,021 · WASTEDLOCKER +0,019. Y **baja** en dos:
  HELLOKITTY −0,108 y WANNACRY −0,059.
- **La cuarta es neutral o mala:** sube en una sola (CERBER +0,046) y **baja en tres**
  (MEDUZALOCKER −0,039 · BLACKCAT −0,023 · TESLACRYPT −0,023).

**El caso más elocuente es CHIMERA:** su segunda plantilla le **bajó** el F1 **−0,202**
[−0,301; −0,104]. Es la única familia cuyo paso 1→2 es negativo, y sus dos plantillas tienen
coseno **0,2535** entre sí (alemana vs inglesa). Un texto que no se parece a la familia no
enseña la familia: la confunde.

**Y CHIMERA es además la familia más atropellada por las demás:** en efecto ajeno pierde
−0,093 (2→3), −0,143 (3→4) y −0,123 (4→5). Cada vez que las otras familias tienen más dato,
CHIMERA pierde notas. Es el perfil exacto de una clase con poco material y sin cohesión.

### (3) La cohesión predice el NIVEL de F1, no el valor de la próxima nota

Esto corrige una inferencia que parece razonable y es falsa:

| Relación (Spearman) | ρ | p | n |
|---|---|---|---|
| cohesión vs **F1 base** | **+0,481** | **0,0096** | 28 |
| n_plantillas vs F1 base (30 familias) | +0,381 | 0,0376 | 30 |
| n_plantillas vs F1 base (**sin las 2 de 1 plantilla**) | +0,231 | 0,2365 | 28 |
| **cohesión vs Δ PROPIO** (valor marginal de una plantilla más) | **+0,010** | **0,9319** | 70 |

Dos lecturas, las dos importantes:

1. **El efecto aparente de «más plantillas» se lo llevan las dos familias de 1 plantilla.**
   Sacando BADRABBIT y CRYPTOLOCKER, la cantidad de plantillas **deja de predecir** el F1 base
   (p = 0,24). La cohesión sí lo predice (ρ +0,481, p = 0,0096), confirmando B.3 sobre 149.
2. **Pero la cohesión NO sirve para priorizar recolección** (ρ +0,010, p = 0,93). Predice dónde
   está el techo, no cuánto sube con la próxima nota. **No usar la cohesión como criterio de
   recolección** — era la conclusión tentadora y los datos la rechazan.

### (4) Donde SÍ hay margen: 11 de 30 familias tienen cobertura de regla EXACTAMENTE 0

Cobertura de la regla de M.6 (`privados_sin_circ_MAS_NOMBRE`, 50 semillas) por familia:

**Cobertura 0 — dependen 100 % del texto (11 familias):** BADRABBIT, CRYPTOLOCKER, CUBA,
DARKSIDE, HELLOKITTY, JIGSAW, MEDUZALOCKER, NOTPETYA, SUNCRYPT, WANNACRY, WASTEDLOCKER.

Y el acierto del texto solo en esas notas dice cuánto duele: HELLOKITTY **0,153** · JIGSAW
0,300 · SUNCRYPT 0,420 · CUBA 0,480 · NOTPETYA 0,520 · MEDUZALOCKER 0,600 · DARKSIDE 0,640 ·
WANNACRY 0,710 · WASTEDLOCKER 0,800 · BADRABBIT y CRYPTOLOCKER 0,000.

**El peor caso de todos es RYUK, y no está en esa lista:** cobertura 0,153, y donde la regla no
cubre el texto acierta **0,087**. Es la peor combinación del corpus: el 85 % de sus notas caen
al texto y el texto se equivoca en 9 de cada 10.

**Cota superior DECLARADA del canal** — supuesto explícito y falsable: que las familias no
cubiertas alcancen el F1 medio de las 12 que hoy tienen cobertura ≥ 0,50, que es **0,6751**.
**No es una predicción**, es una cota, y se reporta como tal (igual que la tercera columna del
Exp. 2d):

| Escenario | macro-F1 | Δ sobre 0,5191 |
|---|---|---|
| si el canal llegara a las 11 de cobertura 0 | 0,6319 | **+0,1128** |
| si llegara a las 7 de cobertura < 0,50 | 0,5669 | +0,0478 |
| si llegara a las dos cosas | 0,6797 | **+0,1605** |

Las que más aportarían: BADRABBIT +0,0225 · CRYPTOLOCKER +0,0225 · HELLOKITTY +0,0167 ·
RYUK +0,0132 · NOTPETYA +0,0127 · JIGSAW +0,0108 · SUNCRYPT +0,0090 · MAZE +0,0088.

### (5) La comparación que ordena el trabajo que queda

| Palanca | Valor | Costo | Estado |
|---|---|---|---|
| **Extender el canal (IOC/nombre) a las 11 de cobertura 0** | cota **+0,113** | M.2 rediseñado | 🟡 |
| Extender el canal a las 7 de cobertura baja (RYUK primero) | cota **+0,048** | ídem | 🟡 |
| Recuperar el nombre original de las **32 notas `sin_verificar`** | no medido, alimenta lo de arriba | 32 búsquedas | 🟢 acotado |
| **2.ª plantilla para CRYPTOLOCKER** (candidata en 8 imágenes en disco) | cota +0,0225 | 1 decisión de alcance + OCR | 🔴 decisión |
| 2.ª plantilla para BADRABBIT | cota +0,0225 | sin candidata hallada | ⛔ |
| 3.ª plantilla para las 7 familias de 2 plantillas | medido: paga en 1 de 3 casos, ~+0,02 de F1 propio ⇒ ~+0,001 de macro cada una. **CHIMERA empeoraría** | 7 notas | 🟡 poco |
| 4.ª plantilla para cualquiera | medido: **neutral o negativo** (3 familias bajaron) | — | ⛔ **no hacer** |

**Un orden de magnitud de diferencia, medido:** el canal vale ~+0,11 de cota; toda la
recolección de notas que queda en el núcleo de 30 vale ~+0,01. **El frente de notas dejó de
estar limitado por cuántas notas hay y pasó a estar limitado por cuántos canales se usan.**

### (6) El estado del canal de nombre, que es el que se puede mover

Auditoría sobre las **149** (`3_datos/nombres_notas/auditoria_nombres_corpus.csv`):

| Procedencia del nombre | Notas | ¿Sirve? |
|---|---|---|
| `curador` (inventado por quien recolectó) | 53 | ⛔ circular |
| **`sin_verificar`** | **32** | ⚠️ **recuperable de la fuente** |
| `genuino_renombrado` | 26 | ✅ con el nombre original del repo |
| `genuino` | 21 | ✅ |
| `genuino_de_la_fuente` | 17 | ✅ |

**64 notas de 149 tienen nombre usable, en 14 familias** — pero el canal de nombre solo dispara
en **6**: DHARMA 0,518 · GANDCRAB 0,520 · MAZE 0,200 · RYUK 0,153 · LOCKBIT 0,149 · PHOBOS
0,025. Tener nombre no alcanza: el nombre además tiene que ser **discriminante** (visto en
entrenamiento y apuntando a una sola familia), y ahí pegan las colisiones ya documentadas
(`README.txt` en 5 familias, `Info.hta` en DHARMA y PHOBOS).

**Las 32 `sin_verificar` son el lote acotado que queda**, repartidas en 22 familias: CERBER 4 ·
RANSOMEXX 3 · CLOP, GANDCRAB, LOCKBIT, LORENZ, MAZE 2 cada una · y 1 en cada una de las otras
15. Recuperar su nombre original de la fuente es trabajo finito y no requiere recolectar notas
nuevas.

### (7) Qué NO aportan las fuentes nuevas al núcleo de 30 (y para qué sí sirven)

- **Group-IB** (43 grupos activos con nota completa y atribución): verificado, **no cubre
  ninguna de las 30**. Sirve **solo** para el experimento de extensión.
- **Kaggle**: ya descartado y no reabrir — sin etiquetas de familia (`__label__ransomware`), y
  72 de las notas del corpus ya están ahí, así que tampoco es independiente.
- **MISP** (1.375 referencias sin usar): su valor **no** es la vista de nombre —eso ya se midió
  y se cae— sino la **bibliografía** y el análisis año-vs-F1 que pidió Cappo.
- **id-ransomware / Amigo-A**: agotado para las 5 familias trabadas, con una excepción abierta:
  la **ventana de pago de CryptoLocker**, que dio TEXTO NUEVO (coseno 0,344 contra la nota que
  ya está) y quedó en decisión de alcance —¿una interfaz de pago es una nota?—. Es la única
  plantilla nueva identificada que le cambiaría el resultado a una familia estructuralmente
  inevaluable. **Las 8 imágenes fuente están en disco**
  (`3_datos/recoleccion_2026-08/CRYPTOLOCKER/imagenes/`) pero **el texto transcripto no se
  guardó**: vive solo en un chat viejo. Hay que volver a hacer el OCR.

## ★★★ DECISIÓN DE ALCANCE: LA VENTANA DE PAGO DE CRYPTOLOCKER CUENTA COMO NOTA (2026-08-28)

> **Decisión de Romina**, tomada el 2026-08-25: *«que cuente como nota, es lo único que hay»*.
> Cierra la decisión de alcance que quedó abierta el 2026-08-20 en
> `6_notas_trabajo/extension_notas_id-ransomware_2026-08-20.md`.
> **Se implementa como BASE PARALELA de 150 notas, no reemplazando el corpus canónico de 149.**

### (1) Qué se incorporó, y por qué esa pantalla y no otra

CryptoLocker no deja archivo de nota: muestra una interfaz. De las **8 capturas** del original
2013 en `3_datos/recoleccion_2026-08/CRYPTOLOCKER/imagenes/`, seis son pantallas de pago (una por
método: MoneyPak, paysafecard, Ukash, cashU, Bitcoin, más la del desplegable vacío). **Se
incorporó una sola: la de Bitcoin** (`cryptolocker_original_8.png`), porque es la **única con
texto escrito por el malware y con un IOC**. Las otras cinco tienen como cuerpo la descripción
comercial del proveedor de pago (Green Dot, Ukash, paysafecard, cashU), no texto del atacante.

Archivo: `CRYPTOLOCKER/idr_cryptolocker_ventana_pago_bitcoin_2013.txt` (673 caracteres),
`tipo=transcripcion`, fuente id-ransomware.blogspot.com (Amigo-A / Andrew Ivanov),
`extension_original=.txt` con `nombre_genuino=SIN_ARCHIVO`.

### (2) La transcripción está verificada por checksum, no por lectura

La nota contiene la dirección **`1KP72fBmh3XBRfuJDMn53APaqM6iMRspCh`**. Una dirección Bitcoin
lleva 4 bytes de checksum (doble SHA-256 sobre el cuerpo), así que **un solo carácter mal leído
rompe la validación**. Verificado: **Base58Check VÁLIDA**, checksum leído `fd747ed2` = calculado
`fd747ed2`, 25 bytes, versión P2PKH mainnet. **La transcripción del dato crítico no depende de
mi lectura: depende de la aritmética.** Es el control que conviene repetir con cualquier IOC que
entre por OCR.

### (3) El verificador de casi-duplicados: TEXTO NUEVO, pero con una señal de alarma

`python verificar_nota_nueva.py` sobre el corpus de 149:

| | |
|---|---|
| Veredicto | **TEXTO NUEVO** — plantillas **99 → 100** |
| Coseno con `cryptolocker_note1.txt` | **0,3801** |
| Coseno con `pcrisk_cryptolocker_1.txt` | **0,3848** |
| **Vecino más cercano en TODO el corpus** | **JIGSAW/`jigsaw_note1.txt` 0,4729** |
| Siguientes vecinos | WANNACRY 0,4107 · CHIMERA 0,4024 · DHARMA ×5 (0,387-0,394) |
| **Cohesión que le queda a CRYPTOLOCKER** | **0,3824** con 2 plantillas |

⚠️ **La alarma hay que declararla:** la plantilla nueva **se parece más a JIGSAW que a su propia
familia**. Es el patrón exacto de CHIMERA —cuya segunda plantilla, con coseno 0,2535 contra la
primera, le **bajó** el F1 **−0,202**— aunque menos extremo (0,382 contra 0,254). Lo que se
espera, entonces, no es que CRYPTOLOCKER pase a rendir bien: es que **deje de ser
estructuralmente inevaluable**. Que eso mejore o empeore el macro-F1 es una pregunta empírica y
se mide, no se argumenta.

### (4) Dos consecuencias de método que hay que declarar en la tesis

1. **El primer párrafo de la nota no lo escribió el atacante.** Es la descripción genérica de
   Bitcoin («Bitcoin is a cryptocurrency where the creation and transfer of bitcoins is based on
   an open-source cryptographic protocol…»). Al ser CRYPTOLOCKER la única familia del corpus con
   texto así, el clasificador podría aprenderlo como firma de CryptoLocker — un discriminante
   **perfecto y vacío**, de la misma clase que el duplicado por *defanging* que se limpió el
   24-25 de agosto. **Hay que declararlo**, y conviene medir la variante sin ese párrafo antes de
   citar cualquier mejora como mérito del método.
2. **La decisión fija un precedente que hay que aplicar de forma uniforme.** El documento del
   2026-08-20 había fijado el criterio contrario para MAZE («el wallpaper entra, el sitio de pago
   no»). Si una interfaz de pago cuenta como nota, ese criterio queda **inconsistente** y hay que
   revisar dos cosas: el sitio de pago de Maze, y **M.5** (los fragmentos de 6 tokens
   solo-contacto de RYUK, que es la misma pregunta con otra forma). O se aplica a los tres, o el
   criterio es *ad hoc*. **Queda como punto para Cappo.**

### (5) Por qué base PARALELA y no cambio del corpus canónico

`CORPUS_DIR` es una variable de entorno (`clasificador_notas_v2.py:77`), así que se puede medir
otro corpus sin tocar el de 149:

- Corpus: `3_datos/corpus_v2_150ventana/` (copia de `corpus_v2` + la nota nueva).
- Manifiesto: `3_datos/manifiesto_corpus_v2_150ventana.csv` (150 filas; el de 149 sigue intacto
  y sigue pasando `validar_procedencia.py`).
- Salidas: sufijo **`_150ventana`**, no `_150` — **las carpetas `_150` que ya existen son otro
  corpus** (el previo a retirar `lb20.txt`) y están superadas. El sufijo dice qué las distingue.

Ventaja concreta: **todas las cifras de 149 medidas hoy siguen siendo reproducibles**. Si el
número sobre 150 convence, promover la nota a canónica es copiar un archivo y agregar una fila.

### (6) RESULTADO CON 50 SEMILLAS: la ventana de pago NO mueve el macro-F1, y eso está bien

> Corridas: `resultados_notas_149_50sem/` y `resultados_notas_150ventana_50sem/`
> (`clasificador_notas_v2.py --solo-canonica --semillas 50`, commit `90017f5`; para la base de
> 150, con `CORPUS_DIR` apuntando a `corpus_v2_150ventana`). Logs `_log_clas_149_50sem.txt` y
> `_log_clas_150ventana_50sem.txt`.

| P2 · grupos·combinado·LinearSVC | 149 | 150 con la ventana |
|---|---|---|
| 10 semillas | 0,4680 ± 0,1003 | 0,4862 ± 0,0624 |
| **50 semillas** | **0,4593 ± 0,0752** | **0,4694 ± 0,0621** |

**Contraste con 50 semillas:** diferencia de medias **+0,0102**, Welch **t = +0,74, p = 0,4623**
→ **NO detectable**. Varianzas: F = 1,464, **p = 0,1856** → la reducción de dispersión **tampoco**
es detectable. Con 10 semillas la diferencia parecía +0,0182; con 50 se reduce a +0,0102 y sigue
dentro del ruido.

**Control de consistencia que vale registrar:** esta corrida (semillas 0-49) da **0,4593 ±
0,0752**, y la capa de texto de M.6 (semillas 100-149, otro script) dio **0,4593 ± 0,0752**. Son
conjuntos de semillas **disjuntos** y coinciden en las cuatro cifras; el error estándar de la
media es 0,011, así que el acuerdo en el último decimal es suerte, pero la base queda confirmada
por dos caminos independientes.

**Lo que sí cambió, por familia (50 semillas):**

| Familia | 149 | 150 | Δ |
|---|---|---|---|
| **CRYPTOLOCKER** | **0,0000** | **0,0610** | **+0,0610** |
| BADRABBIT | 0,0000 | 0,0000 | 0,0000 |
| HELLOKITTY | 0,1710 | 0,1920 | +0,0210 |
| JIGSAW | 0,3280 | 0,3150 | −0,0130 |
| CHIMERA | 0,1380 | 0,1130 | −0,0250 |
| WANNACRY | 0,6430 | 0,6160 | −0,0270 |

El riesgo que se había anticipado —que la plantilla nueva, más parecida a JIGSAW (0,473) que a su
propia familia, le robara notas— **se materializó pero es chico: JIGSAW −0,0130.**

**Familias con F1 exactamente 0,0000: de 2 a 1.** Queda solo BADRABBIT.

### (7) El efecto real no es de rendimiento, es de honestidad — y hay que reportarlo así

| Conjunto | 149 | 150 con la ventana |
|---|---|---|
| Todas (30 familias) | 0,4593 | 0,4694 |
| **Evaluables** | **28 → 0,4921** | **29 → 0,4856** |

Leer las dos filas juntas es lo que da la conclusión correcta: al incorporar la ventana, el
macro-F1 de **todas** sube un poco (no detectable) y el de las **evaluables baja** (0,4921 →
0,4856), porque CRYPTOLOCKER entra al conjunto evaluable con 0,061 y le baja el promedio.

**O sea: la ventana de pago no mejora el método. Lo que hace es convertir un cero estructural en
un fracaso medido.** Y eso, para la defensa, es una posición más fuerte que la anterior:

- Antes: «CRYPTOLOCKER no se puede evaluar bajo P2, hay que excluirla» — un argumento que hay que
  defender.
- Ahora: «CRYPTOLOCKER se evalúa y da macro-F1 0,061: el método falla en esta familia, y la razón
  está medida — sus dos plantillas tienen coseno 0,38 entre sí y la nueva se parece más a JIGSAW
  que a ella misma».

Y la lista de inevaluables que hay que explicarle a Cappo **se reduce de 2 familias a 1**.

**Recomendación de reporte:** promover la nota a canónica **es defendible pero no urgente**, y la
decisión se puede tomar con este número a la vista. Si se promueve, hay que re-medir todo sobre
150 (M.1, M.6, M.3, curva, protocolo Lemmou, grafo) y renumerar la base en el capítulo 4. Si no se
promueve, queda como **experimento de alcance declarado**: «con la ventana de pago incorporada,
CRYPTOLOCKER deja de ser inevaluable y el macro-F1 no cambia de forma detectable (+0,0102, p =
0,46)» — que es una frase citable y responde la pregunta sin tocar el capítulo.

### (8) ⚠️ CORRECCIÓN al bloque de M.1 de este mismo día: el desvío de P2 no se duplicó

El bloque «RESULTADO M.1 — CASCADA IOC→TEXTO RE-MEDIDA SOBRE 149», sección (4), dice que la
dispersión **se duplicó** (0,0490 sobre 155 → 0,1003 sobre 149). **Ese 0,1003 es una estimación
de 10 semillas y sobreestima.** Medido con 50 semillas sobre el mismo corpus de 149, el desvío es
**0,0752**. Contra el 0,0490 de la base de 155 (que también es de 10 semillas, así que también es
ruidoso) la razón es **1,53×**, no 2×.

Lo que se sostiene: **la dispersión sobre 149 es mayor que la de 155 y los IC son más anchos**, y
la recomendación práctica no cambia — **para cifras finales, 50 semillas**. Lo que hay que corregir
es la magnitud: no digas «se duplicó», decí «es 1,5 veces mayor», y citá siempre con cuántas
semillas se midió. La media también se movió con las semillas: 0,4680 (10) frente a **0,4593**
(50).

## ★★★★ PRUEBA CON UNA NOTA REAL FUERA DEL CORPUS: EL MODELO NO LA RECONOCE, Y AVISA (2026-08-28)

> Nota de un incidente real, traída por Romina el 2026-08-28: captura de un `cat` sobre
> `/opt/zimbra/data/ldap/mdb/db/!README_RECOVER.txt` en un servidor de correo Zimbra
> comprometido, más su transcripción (`readme_recover.txt`, 1.033 caracteres). Es el primer
> caso de **inferencia sobre una nota que no está en el corpus**, y salió el mejor ejemplo
> disponible de la limitación de mundo cerrado. Código nuevo: `2_codigo/clasificar_nota_suelta.py`
> y `2_codigo/validar_direccion_cripto.py` (commit `f0ae154`).

### (1) Qué contestó el modelo

Entrenado sobre las 149 notas con la configuración canónica (TF-IDF combinado + LinearSVC):

| | |
|---|---|
| Predicción | **BLACKBASTA** |
| **Margen (1.ª vs 2.ª clase)** | **0,0501** |
| En el punto de operación de M.3 (umbral 0,50) | **SE ABSTIENE** |
| Vecino más cercano (coseno char 3-5) | 0,5004 — `RYUK/pcrisk_ryuk_1.txt` (el umbral de casi-duplicado es 0,90) |
| ¿Aplica la regla exacta de M.6? | **No.** Ninguno de sus 2 IOCs se vio en el corpus |

**Y los dos datos que convierten esto en un «no sé» y no en un error silencioso:**

1. **Ninguna de las 30 clases da `decision_function` positiva** (máximo −0,6965, mínimo −1,1003).
   En las 149 notas del corpus, **siempre** hay al menos una clase positiva: **0 de 149** quedan
   sin ninguna. Es decir: ningún clasificador uno-contra-resto la reclama.
2. **Su margen, 0,0501, es más bajo que el de cualquier nota del corpus**: mínimo 0,129, mediana
   1,646, percentil 10 en 1,234. **0 de 149** notas del corpus tienen un margen tan chico.
   ⚠️ Esos márgenes del corpus están medidos **en muestra** (el modelo se entrenó con ellas), así
   que son una referencia optimista; sirven como contraste cualitativo, no como distribución nula.

**Lectura:** el modelo está obligado a nombrar una de las 30 y nombra BLACKBASTA, pero lo hace con
un margen 25 veces menor que su caso más dudoso y sin que ninguna clase la reclame. **En el punto
de operación reportado se abstiene**, que es la respuesta correcta. La nota no es de ninguna de las
30: es de doble extorsión (sitio de filtración, rescate en **50 XMR**, contacto en `proton.me`),
un perfil que no existe en el corpus.

**Para la tesis esto vale como demostración, no como anécdota:** M.3 se cerró midiendo abstención
**dentro** del corpus, con notas cuya familia sí estaba entre las 30. Este es el caso que M.3 no
podía probar —una familia fuera del conjunto cerrado— y el mecanismo se comporta como se
esperaba. Va al capítulo de limitaciones junto con el mundo cerrado.

**No es candidata al corpus:** no hay atribución de familia. Sin etiqueta verificable no entra
(regla de fuentes). Sirve como caso de prueba, no como dato de entrenamiento.

### (2) Hallazgo de integridad: la transcripción tiene 2 caracteres mal en la dirección Monero

La nota pide **50 XMR** a una dirección de 95 caracteres. La transcripción y la captura **no
coinciden**, y el checksum resuelve cuál es la correcta sin depender de la lectura de nadie.

Las direcciones Monero llevan 4 bytes de checksum = primeros 4 bytes de **Keccak-256** del cuerpo.
No hay librería de Keccak en el entorno (`hashlib.sha3_256` **no** sirve: usa el relleno de NIST
`0x06`, y Monero usa el Keccak original `0x01`), así que se implementó en
`validar_direccion_cripto.py`, **con autotest contra dos vectores públicos que corre antes de
dictaminar**.

| | Dirección | Checksum |
|---|---|---|
| **Correcta** | `…iegPesnddiVw47XdwZs1MZ68LFVjayY2` | **VÁLIDO** (`7b217faf` = calculado) |
| La del `.txt` | `…iegPe3nddiVw47XdwZs1HZ68LFVjayY2` | **NO válido**, y ninguna corrección de un solo carácter lo arregla |

Diferencias exactas: **posición 68**, el `.txt` dice `3` y va `s`; **posición 83**, dice `H` y va
`M`. Dirección completa correcta:

```
45EQACa2DVwHEMP2TcvhrgQgCeDUiwLnbcWrTmbUrHqZ1pFu8KJodc93PwXwEKeiegPesnddiVw47XdwZs1MZ68LFVjayY2
```

Byte de red `0x12` = mainnet estándar, 69 bytes: **es una dirección real y bien formada**, no un
placeholder. Y de paso quedó confirmado que mi propia lectura de la captura también tenía un error
—leí `l` (ele) donde va `1`—, imposible en Base58, que excluye `l`, `I`, `0` y `O`. **Moraleja
operativa: cualquier dirección que entre al corpus por OCR se valida por checksum, no por
lectura.** El mismo control ya se aplicó a la dirección BTC de la ventana de pago de CryptoLocker.

### (3) Hueco en el extractor de IOCs: no hay patrón de Monero

`normalizacion_marcadores.PATRONES` tiene `[BTC]` = `\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b`, que
**solo cubre Bitcoin legacy**. Consecuencia medida en esta nota: la dirección Monero cayó en
`[CLAVE]` (el cajón genérico de cadenas base64 de 40+ caracteres), no en un marcador de
criptomoneda. **Faltan dos patrones:**

- **Monero:** `\b[48][1-9A-HJ-NP-Za-km-z]{94}\b` (95 caracteres; 106 si es *integrated address*).
- **Bitcoin bech32:** `\bbc1[a-z0-9]{25,62}\b` — tampoco lo cubre el patrón actual.

Impacto real sobre lo ya medido: **ninguno**, porque en el corpus de 149 hay 6 marcadores `[BTC]`
y ninguna nota trae Monero (son familias de 2013-2022, cuando el rescate se pedía en BTC). Pero
**sí importa para la extensión a familias nuevas y para cualquier caso de 2023 en adelante**, donde
XMR es la moneda dominante: hoy esas direcciones no se contarían como IOC de criptomoneda. Anotado
como arreglo pendiente, con la advertencia de que **tocar `PATRONES` cambia B.3, M.1 y M.6**: si se
agregan, hay que re-medir los tres y declarar la base.

### (4) Intento de atribución AGOTADO por las dos vías locales, y un criterio de mundo abierto que aparece de regalo

**(a) Entrenando con más familias: no la reconoce, y el aviso es el mismo.** Código:
`2_codigo/reconocer_con_expansion.py` (commit `0241caa`). Se entrenó la misma configuración
canónica con el espacio de etiquetas ampliado a los repos que ya están en disco:

| Entrenamiento | Notas | Familias | Plantillas | Clases con `decision_function` > 0 | Máx. coseno |
|---|---|---|---|---|---|
| corpus canónico | 149 | 30 | 99 | **0 de 30** | 0,5004 (RYUK) |
| **Lemmou** (`RansomNoteFiles`) | 183 | **68** | 127 | **0 de 68** | 0,4696 (CRYPTFILE2) |
| Lemmou + ThreatLabz + pcrisk + corpus | **773** | **291** | 495 | **0 de 291** | 0,5253 (AILOCK) |

**Ninguna clase la reclama en ninguno de los tres espacios de etiquetas**, y el vecino más
parecido de todo el disco queda en 0,5253, muy por debajo del umbral de casi-duplicado de 0,90.
Ampliar el catálogo de 30 a 291 familias **no la hace reconocible**: el problema no es el tamaño
del catálogo.

**(b) En el catálogo MISP tampoco está.** Sobre
`3_datos/misp_ransomware_galaxy/misp_galaxy_ransomware_2026-08-20.json`: **2.135 entradas**, de
las cuales **298 traen `ransomnotes-filenames`**. Buscados el nombre del archivo, el correo, la
dirección Monero y cinco frases de la nota: **ningún acierto**. Los 80 nombres con «recover» del
catálogo (34 familias) son de otro formato — `!Recovery_[random_chars].txt` (CryLocker, CryptXXX,
Sage 2.0), `readme_to_recover_files` (MedusaLocker), `read_me_for_recover_your_files.txt`
(StorageCrypter) —, ninguno es `!README_RECOVER.txt`. `proton.me` aparece en la entrada `crynox`,
pero es coincidencia de proveedor de correo, no atribución.

**Conclusión: la nota no se puede atribuir con nada local.** Y por eso mismo **no entra al
corpus**: sin etiqueta de familia verificable no hay dato de entrenamiento, solo un caso de
prueba. Lo que sí queda es su caracterización: doble extorsión, rescate en 50 XMR, contacto
únicamente por correo (`proton.me`) y **sin URL ni `.onion` del sitio de filtración que la propia
nota menciona**.

### (5) ▶ CANDIDATO A EXPERIMENTO NUEVO: «ninguna clase positiva» como criterio de mundo abierto

Sale de esta prueba y merece medirse. M.3 detecta la duda con un **umbral sobre el margen**, que
obliga a elegir un número y depende de cuántas clases haya. Los datos de esta noche sugieren un
criterio más simple:

| | Clases con `decision_function` > 0 |
|---|---|
| la nota fuera de catálogo, con 30 / 68 / 291 familias | **0**, en los tres casos |
| las 149 notas del corpus (en muestra) | **0 de 149** quedan sin ninguna clase positiva |

O sea: «**si ninguna clase da puntaje positivo, contestar “no sé”**» separó el caso de fuera del
catálogo de todos los de dentro, **sin umbral que elegir** y de forma estable frente a un cambio
de 30 a 291 clases. Además el margen de la nota (0,0501) quedó por debajo del mínimo de las 149
(0,129), así que los dos criterios apuntan igual.

**Lo que falta para que sea reportable** (no está medido, es una propuesta):
1. Medir la regla **fuera de muestra**, con validación cruzada sobre las 149: cuántas veces el
   máximo queda ≤ 0 en notas cuya familia SÍ está. Si es ~0 %, la regla cuesta casi nada de
   cobertura. La comparación de esta noche usa márgenes **en muestra**, que son optimistas.
2. Conseguir **más notas fuera de catálogo** para estimar la otra cara (cuántas veces la regla NO
   avisa cuando debería). Con un solo caso no se puede.
3. Preregistrar antes de correr, con criterio de adopción, como el resto.

Encaja con el pedido explícito de Romina (2026-08-28): *«siempre debe poder decir no sé, por si
acaso»*. Hoy el sistema puede abstenerse; esto lo haría sin depender de un umbral elegido a mano.

## ★★★★ BÚSQUEDA EXHAUSTIVA DE LA NOTA DEL INCIDENTE, Y DOS HALLAZGOS QUE VALEN MÁS QUE ELLA (2026-08-28)

> Continuación del bloque anterior. Romina pidió buscar la nota en todas las fuentes
> disponibles, incluidas las que estaban citadas en el `.bib` pero **no** en disco. Se agotó la
> búsqueda. **La nota no se puede atribuir con ninguna fuente pública disponible** — y tampoco
> la reconoce ID Ransomware, que es la herramienta de producción hecha para esto (probado por
> Romina).

### (1) Todas las fuentes consultadas, y lo que dio cada una

| Fuente | Tamaño | Cómo se buscó | Resultado |
|---|---|---|---|
| Corpus canónico | 149 notas · 30 fam. | texto + nombre | 0 de 30 clases positivas · coseno máx. 0,5004 |
| **Lemmou** `RansomNoteFiles` | 183 notas · 68 fam. | texto + nombre | 0 de 68 · coseno máx. 0,4696 |
| **ThreatLabz** `ransomware_notes` | 422 notas · 223 fam. | texto + nombre | sin coincidencia · coseno máx. 0,5253 (AILOCK) |
| `notas_pcrisk` | 19 notas · 13 fam. | texto + nombre | sin coincidencia |
| **MISP galaxy** (JSON local) | **2.135 entradas** · 298 con nombre de nota | nombre, correo, dirección XMR, 5 frases | **ningún acierto** |
| **`albertzsigovits/malware-notes`** (bajado hoy) | `_ransom_notes.md`, 28 familias | texto completo | **ningún acierto**; no contiene ni la palabra «XMR» (son notas de 2019-2020) |
| **`f6-dfir/Ransomware`** (bajado hoy) | **182 notas · 32 familias de 2023-2025** | texto + nombre | 0 de 31 clases positivas · coseno máx. **0,4401** (PROTON) |
| Tabla de nombres citables propia | 137 nombres · 30/30 fam. | nombre | sin coincidencia |
| **ID Ransomware** (servicio) | producción | Romina subió la nota | **NO la detecta** |
| **`codingo/Ransomware-Json-Dataset`** (bajado hoy) | **411 familias**, 217 con `ransomNoteFilenames` | nombre de nota + frases | **ningún acierto** (sus 38 nombres con «recover» son de familias de 2016-2017) |
| **Group-IB** «Ransomware Notes» (leído hoy) | **47 grupos activos**, con el TEXTO COMPLETO de cada nota | texto, nombre, moneda, contacto | **ningún acierto**: ni `README_RECOVER.txt`, ni XMR, ni contacto por correo |
| **TODAS JUNTAS** | **955 notas · 320 familias · 630 plantillas** | un solo modelo | **0 de 320 clases positivas** · coseno máx. **0,5151** (AILOCK) |

**La cifra definitiva:** entrenando un solo modelo con **las cinco fuentes juntas**
—955 notas, **320 familias**, 630 plantillas distintas— **ninguna clase da
`decision_function` positiva** (máximo −0,7235) y el vecino más parecido de todo el
material disponible queda en **0,5151**. Log: `4_resultados/_log_reconocer_todo.txt`.
Pasar de 30 a 320 familias no cambia el veredicto: **el problema no es el tamaño del
catálogo, es que esta familia no está en ninguno.**

**La única pista fue el NOMBRE, y no aguantó el texto.** `!README_RECOVER.txt` es casi idéntico a
`README-RECOVER.txt` de **KRYBIT** y a `README-RECOVER-[rand].txt` de **QILIN** (ThreatLabz). Pero
el coseno del texto es **0,1260 con KRYBIT y 0,2252 con QILIN**: no es ninguna de las dos. La nota
de Qilin viene firmada («`-- Qilin`»), trae dos `.onion` de su blog y lista los datos robados; la
del incidente no tiene nada de eso. **Es un buen ejemplo de la limitación ya documentada del canal
de nombre: `README-RECOVER` es genérico y lo usan familias sin relación.**

**Y una caracterización que sale de comparar con Group-IB:** en sus 47 grupos activos las
notas mandan a un **chat en Tor** y piden **Bitcoin**. La del incidente hace lo contrario:
pide **Monero**, da **solo un correo** (`proton.me`) y **no incluye ninguna `.onion`** — aunque
amenaza con un sitio de filtración. Eso la vuelve **atípica para un grupo profesional de 2026**
y apunta más a un actor chico, nuevo, o a una plantilla genérica reutilizada. Es
caracterización, no atribución.

### (1.bis) Cómo se comparó en cada fuente — la distinción importa al declararlo

No todas las fuentes se buscaron igual, porque no todas contienen lo mismo. Al escribir
esto en la tesis hay que decir **qué se comparó**, no solo cuántas familias se revisaron:

| Tipo de comparación | Alcance |
|---|---|
| **Por TEXTO completo** (TF-IDF char_wb 3-5 + coseno, y modelo entrenado) | **1.112 notas**: corpus 149 · Lemmou 183 · ThreatLabz 422 · f6-dfir 182 · pcrisk 19 · **MISP 136** (campo `ransomnotes`, 117 entradas lo tienen) · **Zsigovits 21** |
| Por nombre de archivo y metadatos | codingo (411 familias, 217 con `ransomNoteFilenames`) y el resto de las 2.135 entradas de MISP — **esos datasets NO traen el texto de la nota**, solo nombre, extensiones y referencias: no hay texto que comparar |
| Lectura de página | Group-IB, 47 grupos activos con el texto completo publicado |

**Coseno máximo en todo el material comparado por texto: 0,5151** (AILOCK, ThreatLabz), contra un umbral de casi-duplicado de 0,90. En la extracción de MISP + Zsigovits el máximo fue **0,4107** (Lockergoga).

⚠️ **Corrección de proceso:** en la primera pasada MISP y Zsigovits se buscaron **solo** con grep de nombre y frases, aunque los dos tenían texto disponible. Se cerró después extrayendo los 157 textos y comparándolos con el criterio del proyecto. La conclusión no cambió, pero el hueco era real: **una fuente con texto se compara por texto, no por frases.**

### (2) ★ El dato que importa para la tesis: ID Ransomware TAMPOCO la detecta

Nuestro modelo se abstiene; la herramienta de referencia —la del 71,93 % con la que se compara el
trabajo (`Pruebas.xlsx`, 41/57 notas de 22 familias)— **tampoco la identifica**. Eso reencuadra la
limitación de mundo cerrado: **no es una debilidad de este método, es el estado del campo.** Sobre
un caso real de 2026, ni el clasificador propio ni el servicio de producción pueden nombrar la
familia, porque la familia no está en ningún catálogo público. Es una frase para el capítulo de
limitaciones que antes no se podía escribir con evidencia.

### (3) ★★★ FUENTE NUEVA INCORPORADA: 182 notas de 32 familias de 2023-2025

`3_datos/fuentes_notas/f6dfir_ransom_notes/` (683 KB, **fuera de git**). Del repo público
`f6-dfir/Ransomware`, de una empresa de DFIR, organizado por familia con subcarpeta
`ransom_notes/`. Familias: **3119, BadRep, Bearlyfy, C77L, CyberSex, Enmity, Fonix, HardBit,
HeadMare, HsHarada, LokiLocker, Masque, MorLock, Muliaka, PE32, Procrustes, Proton, Proxima,
RCRU64, Sauron, Sezar7, Shadow, Sojusz, Surtr, THOR, TeslaRVNG, ToxUnit, VoidCrypt, Werewolves,
Zgut, hacking_cat, kann**.

**Es exactamente el hueco del corpus:** las 30 familias canónicas son de 2013-2022 y piden BTC;
estas son de 2023-2025. Sirve para el experimento **Ext.** (que está 🔴 esperando decisión de
Cappo), y como control externo del canal de nombre. **Descarga acotada por regla:** solo rutas con
`/ransom_notes/`, solo `.txt/.html/.hta`, solo menores a 20 KB → 182 de 587 archivos. **Los otros
405 no se bajaron** (descifradores `.py/.cs/.bin`, herramientas, y archivos cifrados). 25 fallaron
por caracteres especiales en el nombre (`#`), recuperables si hacen falta.

### (4) ⚠️⚠️ HAY QUE RE-VERIFICAR UNA LIMITACIÓN DECLARADA: archivos cifrados fuera de NapierOne

`EXPERIMENTOS_PENDIENTES.md` afirma, como límite externo citable: *«No hay extensión posible a más
familias: no existen archivos cifrados públicos fuera de NapierOne»*. **Ese repo parece ser un
contraejemplo.** Su listado incluye archivos con extensión de familia y ~3 MB cada uno:

`.enc` (6) · `.ryk` (2) · `.blackbit` (2) · `.hardbit4` (2) · `.black` (2) · `.nigra` (2) ·
`.surtr` (2) · `.secles` (2) · `.clear` (2) · `.demo` (2) · `.3ae00608` · `.qgvsrdbp` ·
`.0000000000000000` · `.b4pahv` · `.tb56ez` · `.l8rt4whox` · `.hmc` · `.pe32c`…

⚠️ **NO se descargó ninguno** (regla del proyecto: nada de muestras ni binarios) y por lo tanto
**esto NO está verificado**: se vieron nombres y tamaños en el listado de la API, no el contenido.
Podrían ser archivos de prueba cifrados por el equipo de DFIR y no muestras de víctimas, y en ese
caso no serían equivalentes a NapierOne. Pero **la afirmación “no existen archivos cifrados
públicos fuera de NapierOne” es, como mínimo, demasiado fuerte y hay que revisarla antes de
escribirla en la tesis.** Si resulta que sirven, cambia una limitación declarada del frente de
archivos y habilita el Exp. 2d sobre más familias. **Decisión de Romina + Cappo**, porque implica
bajar archivos cifrados y eso toca la regla de fuentes y el antivirus.

### (6) ✅ RESUELTO POR OTRA VÍA: la nota SÍ está documentada, y su familia NO TIENE NOMBRE

> Fuente que faltaba, sugerida por Romina: el foro de soporte de BleepingComputer
> (`bleepingcomputer.com`, **ya estaba en la lista blanca**). Hilo:
> `bleepingcomputer.com/forums/t/818309/unknown-ransomware-with-elock-extension-readme-recovertxt/`

**El título del hilo es la respuesta: «unknown ransomware».** La nota está documentada
públicamente —misma frase de apertura, mismo `!README_RECOVER.txt`, mismos 50 XMR, mismo
correo— y **ningún investigador del hilo le pone nombre de familia**. Por eso no apareció en
ninguno de los catálogos: **no es que falte en nuestras fuentes, es que la familia todavía no
tiene nombre.**

| Dato | Valor |
|---|---|
| Nota | `!README_RECOVER.txt` |
| **Extensión de los archivos cifrados** | **`.elock`** ← dato nuevo, no estaba en la nota |
| Rescate | 50 XMR |
| Contacto | `cccsitadm@proton.me` |
| **Vector de entrada** | **CVE-2024-45519** (Zimbra postjournal RCE) — hipótesis de un respondedor, con víctimas en Zimbra 8.8.11 y 8.8.15 |
| Estado de la campaña | **activa**: posts de «ayer» y «hoy», ataques referidos al 27 de agosto |
| Familia | **sin identificar** |

**★ Confirmación externa de la dirección Monero.** El foro publica la dirección así:

```
45EQACa2DVwHEMP2TcvhrgQgCeDUiwLnbcWrTmbUrHqZ1pFu8KJodc93PwXwEKeiegPesnddiVw47XdwZs1MZ68LFVjayY2
```

**Es exactamente la que el checksum Keccak-256 había reconstruido**, no la de la transcripción.
O sea: el control por checksum acertó, y ahora está corroborado por una fuente independiente. Es
el argumento más fuerte posible para la regla «toda dirección que entre por OCR se valida por
checksum, no por lectura».

⚠️ **Cuidado con el hilo, que es contenido de usuarios:** la atribución del CVE es la hipótesis
de un respondedor, no un análisis publicado, y en el hilo circula al menos un identificador de
CVE aparentemente inventado. Sirve como pista y como corroboración de los IOCs; **no como fuente
citable de atribución**. Y **la nota sigue sin ser candidata al corpus**: sin nombre de familia no
hay etiqueta.

### (7) Lo que esto significa para el capítulo de limitaciones

Tres sistemas independientes dieron el mismo resultado sobre el mismo caso real:

| Sistema | Respuesta |
|---|---|
| **Este clasificador** (30, 68, 291 y 320 familias) | **se abstiene** — 0 clases positivas, margen 0,05 |
| **ID Ransomware** (producción, el 71,93 % de referencia) | no la detecta |
| **La comunidad** (foro de BleepingComputer) | «unknown ransomware» |

**La abstención del modelo no fue un fracaso: fue la respuesta correcta, y la única correcta
disponible.** Es la validación externa que M.3 no podía tener midiendo solo dentro del corpus, y
convierte la limitación de mundo cerrado en un resultado documentado con un caso real, activo y
multi-víctima de agosto de 2026 — no en una advertencia teórica.

**Y refuerza el orden de prioridades medido**: contra una familia nueva, ni más notas ni más
familias en el catálogo ayudan (se probó de 30 a 320). Lo que ayuda es **poder decir «no sé»**, que
es lo que Romina pidió y para lo que ya hay un criterio candidato en la sección (5).

### (8) El hilo completo: 6-7 víctimas, una sola dirección, y CIFRADO PARCIAL

Segunda lectura del hilo (15 posts, 2 páginas). Datos que no estaban en la nota:

| Indicador | Valor |
|---|---|
| Víctimas distintas | **6-7**, todas servidores **Zimbra sobre Linux** |
| Fecha de los ataques | **27 de agosto de 2026**, a horas parecidas |
| **Dirección Monero** | **la MISMA para todas las víctimas** |
| **Correo** | **el MISMO para todas** (`cccsitadm@proton.me`) |
| Extensión | `.elock` |
| SHA1 de muestra | `92eb29d03bc6038dfe8e591576eae67f4d95e392` (⚠️ **no se descargó**: regla de fuentes) |
| Clave pública RSA | publicada por un respondedor en el hilo |
| Versiones afectadas | Zimbra 8.8.11 y 8.8.15 |
| Estado | sin descifrador |

**★ Hallazgo 1: una sola dirección y un solo correo para 6-7 víctimas.** Una operación
profesional emite billetera e identificador **por víctima** —es lo que permite saber quién pagó—.
Compartir una sola dirección entre todas las víctimas de la campaña indica un actor **poco
sofisticado**, y convierte la dirección en un IOC **de campaña**, no de víctima. Encaja con la
caracterización que ya salió al comparar con Group-IB (XMR + solo correo + sin `.onion`, al revés
de los 47 grupos activos).

**★★★ Hallazgo 2, y es el que le sirve a la tesis: el cifrado es PARCIAL.** Un respondedor, tras
analizar el binario, reporta que **«only beginning of each file is somehow encrypted»** — solo el
principio de cada archivo queda cifrado. Dos consecuencias:

1. **Para el incidente:** los archivos grandes podrían ser **recuperables en su mayor parte**. Es
   lo más accionable de todo el hilo.
2. **Para el frente de archivos de esta tesis:** el clasificador de bytes lee **512 bytes del
   principio + 512 del final** (`clasificador_bytes.py`). Sobre un archivo cifrado así, la
   ventana del principio vería datos cifrados y **la del final vería texto claro**. O sea que el
   diseño de características de esta tesis **se comporta de una forma predecible y medible frente
   al cifrado parcial**, y este es un caso real para ilustrarlo. Conecta directamente con A.1
   (ablación de ventana) y con el Exp. 2d.
3. **Y con la bibliografía del mismo día:** el rastreo automático del 2026-08-28 trajo
   *«Intermittent File Encryption in Ransomware: Measurement, Modeling, and Detection»*
   (arXiv 2510.15133), que modela exactamente este fenómeno **y usa NapierOne**. El caso real
   exhibe el fenómeno que ese paper mide. Es la razón para leerlo, y sube su prioridad respecto
   de la que le puse en el triage.

⚠️ Las dos son afirmaciones **de un foro**: el análisis del binario es de un respondedor, no un
informe publicado. Para la tesis hay que verificarlo con el paper y con NapierOne, no citar el
foro como evidencia técnica. Para el incidente, en cambio, alcanza para intentar la recuperación
parcial.

## ★★★★★ IDENTIFICAR NO ES CLASIFICAR: LOS CEROS ESTRUCTURALES SE IDENTIFICAN PERFECTO (2026-08-28)

> **La idea es de Romina:** *«una nota no nos sirve para clasificar, pero sí para descubrir
> quién es, si el texto es similar»*. Es exacta, y medirla cambia cómo hay que reportar los
> ceros. Código: `2_codigo/identificar_vs_clasificar.py`.

**Son dos tareas distintas con requisitos de dato distintos:**

| | Clasificar (lo que mide P2) | Identificar por similitud |
|---|---|---|
| Cómo funciona | se entrena un modelo, se evalúa con StratifiedGroupKFold sobre grupos | se busca el vecino más cercano en un catálogo y se devuelve su familia |
| Plantillas mínimas por familia | **2** (si no, nunca está en train y test a la vez) | **1** |
| Qué hace ID Ransomware, y el paso de LSA de Lemmou | — | esto |

**Medido sobre las 149 (1 vecino, dejando una NOTA afuera cada vez):**

| | Valor |
|---|---|
| Acierto global identificando | **0,8188** (122 de 149) |
| macro-F1 clasificando (P2) | 0,4680 |
| Coseno del vecino cuando ACIERTA | mediana **0,932** · mínimo 0,437 |
| Coseno del vecino cuando FALLA | mediana **0,543** · máximo 0,956 |

**★ Y el resultado que reordena el argumento de los ceros:**

| Familia | F1 clasificando | Identifica |
|---|---|---|
| **BADRABBIT** | **0,000** | **2 de 2 = 1,000** |
| **CRYPTOLOCKER** | **0,000** | **2 de 2 = 1,000** |

**Su F1 = 0 es una propiedad DEL PROTOCOLO, no del dato.** Con una sola plantilla no puede
haber entrenamiento y prueba a la vez, pero sus dos notas **se reconocen entre sí sin
problema** (cosenos 0,9490 y 0,9883). Esto es más fuerte que la propuesta de reportar «28
evaluables»: no hace falta excluirlas ni pedir disculpas por ellas — hay que decir **qué tarea
no pueden hacer y cuál sí**.

Y no es un caso aislado: **7 familias más ganan más de 0,30 identificando respecto de
clasificar** (BLACKMATTER, CUBA, DARKSIDE, NETWALKER, NOTPETYA 1,000 contra 0,34-0,63;
SODINOKIBI 1,000 contra 0,290; MAZE 1,000 contra 0,409; TESLACRYPT 1,000 contra 0,548;
DHARMA 1,000 contra 0,578).

**Dónde identificar es PEOR, y hay que declararlo:** CHIMERA **0,000** (sus 2 plantillas tienen
coseno 0,2535 entre sí: no se reconocen) y HELLOKITTY **0,000**. En esas dos, el modelo
entrenado le gana a la búsqueda por vecino.

⚠️ **Límites de esta medición:** (a) es *leave-one-out* sobre **notas**, no sobre plantillas, así
que aprovecha que una familia tenga dos notas casi iguales — que es exactamente el punto, pero
hay que decirlo; (b) el mundo cerrado **sigue valiendo**: para una nota de fuera del catálogo el
vecino más cercano va a ser incorrecto, y la separación de cosenos (0,932 al acertar contra
0,543 al fallar) es la que da el criterio de corte; (c) no reemplaza P2 ni es la métrica del
método propuesto: es otra tarea, y se reporta como tal.

**Para la tesis:** habilita escribir una sección corta y fuerte —«clasificación frente a
identificación»— que convierte dos limitaciones en una distinción de tareas, y conecta con el
protocolo Lemmou (L = 0,7767) que ya está medido.

## ★★★ EL PAPER DE CIFRADO INTERMITENTE MIDE, SOBRE NAPIERONE, EL PUNTO DÉBIL DE NUESTRO DISEÑO (2026-08-28)

*Intermittent File Encryption in Ransomware: Measurement, Modeling, and Detection* — Ineza,
Jackson, Niyonkuru, Kevil, Serwadda. arXiv **2510.15133** (oct-2025, revisado ago-2026).
Leído hoy a raíz del caso real, que cifra **solo el principio de cada archivo**.

**Por qué es el paper más relevante del frente de archivos:**

1. **Usa NapierOne**, el mismo corpus: ~26.500 archivos, 11 tipos, 3 clases de compresión.
2. Caracteriza **9 familias con cifrado intermitente**, y **4 son de nuestras 30**:
   **BLACKCAT, DARKSIDE, BLACKMATTER y LOCKBIT** (más Akira, Royal, Play, Qyick, Qilin).
   O sea: **puede haber archivos parcialmente cifrados dentro de nuestro propio conjunto.**
3. **★ El hallazgo que nos toca directo:** un modelo entrenado **solo con los extremos del
   archivo** («whole-file baseline trained only on endpoints») da **61,48 %** de acierto sobre
   cifrado parcial, mientras que trocear el archivo («chunk-level») da **97,1 %**.
   **Nuestro `clasificador_bytes.py` lee exactamente los extremos: 512 bytes del principio +
   512 del final.** El paper cuantifica el costo de esa decisión de diseño frente al cifrado
   intermitente.
4. **NO hace clasificación multiclase de familia**: solo detección binaria cifrado/no cifrado.
   **Nuestra contribución no se solapa con la suya** — la complementa, y por eso se puede citar
   sin debilitar el aporte propio.

**Cifras citables:** 97,1 % por troceo · **61,48 % con extremos** · 84,36 % entrenando con
cifrado intermitente pero sobre archivo completo · pendiente de entropía por tipo: **+0,35
bits** por cada 0,1 de fracción cifrada en XLS contra **+0,0036** en MP4 (97× de diferencia) ·
constante por formato c²_F de **0,00008 (PNG) a 0,179 (XLS)**, que fija la fracción de cifrado a
partir de la cual un detector por KL queda ciego.

**Qué habilita, en orden:**
- **Declarar la limitación con número ajeno**, no con una conjetura: la ventana 512+512 es
  vulnerable al cifrado intermitente, y hay un 61,48 % publicado que lo dimensiona.
- **Un experimento nuevo y chico:** medir nuestro clasificador de bytes sobre archivos de
  NapierOne cifrados parcialmente (head-only al 10 %, 25 %, 50 %) y ver cuánto cae. Es la
  réplica de su Figura principal con nuestro modelo, sobre el mismo corpus. Conecta con A.1
  (ablación de ventana) y da la contra-medida (trocear) ya validada por ellos.
- **Revisar si BLACKCAT, DARKSIDE, BLACKMATTER y LOCKBIT** tienen archivos parcialmente
  cifrados en nuestro subconjunto de NapierOne: si los tienen, parte del error del Exp. 2c en
  esas familias podría explicarse por esto y no por el método.

## 📌 PREREGISTRO — P2-LOGO: leave-one-template-out como protocolo ADICIONAL (2026-09-09)

> **Contexto:** el tutor fijó que sin superar macro-F1 0,50 el experimento no sirve. Estado
> verificado hoy: texto solo bajo P2 **0,4593** [IC de la media 0,4379; 0,4807]; M.6 **0,5191**
> [0,4967; 0,5415] — la media supera 0,50 pero el IC lo roza. Se pidió un análisis desde cero.

### El hallazgo del análisis desde cero

`clasificador_notas_v2.py` fija `N_FOLDS = 2` con el comentario «limitado por las familias con 2
notas». **Ese argumento excluye 3+ pliegues de k-fold, pero NO excluye leave-one-group-out
(LOGO)**, que funciona con cualquier número de grupos ≥ 2. **LOGO nunca se probó en el frente
de notas** (verificado: solo `analisis_bytes.py`, del frente de archivos, lo importa).

**Qué cambia:** con 2 pliegues cada familia entrena con ~k/2 plantillas; con LOGO entrena con
k−1. Para las 12 familias de 4 plantillas es 2 → 3; para CERBER (8) es 4 → 7. **Con 2 pliegues
se descarta la mitad del entrenamiento en cada corte.** La garantía central de P2 se conserva
intacta: **la plantilla de prueba nunca está en entrenamiento.**

### Por qué es legítimo y no «buscar el número»

1. LOGO es el protocolo estándar de máximo aprovechamiento para datos agrupados pequeños
   (n = 99 grupos). Es, si acaso, **más** estándar que 2-fold para este tamaño.
2. La justificación del 2-fold era operativa (familias con 2 notas), no metodológica, y no
   aplica a LOGO.
3. **Se reporta JUNTO a P2, no en su lugar.** Se agrega; no se reescribe nada.
4. La predicción se escribe **antes** de correr, con criterio de falsación.

### Predicciones (falsables)

1. **Texto solo bajo LOGO sube de forma sustancial respecto de P2** (0,4593), porque el
   entrenamiento por familia casi se duplica. Rango esperado: **0,55–0,65**. Debe quedar por
   **debajo de P1** (0,7889), porque P1 filtra casi-duplicados en ambos lados del corte.
   **Falsación:** si LOGO ≤ P2 + 0,03, la restricción de 2 pliegues NO era la que ataba y el
   techo es puramente de datos.
2. **M.6 bajo LOGO también sube**, y su Δ sobre texto-LOGO será **menor** que bajo P2 (la regla
   de IOCs ganaba parte de su cobertura porque el texto tenía poco entrenamiento).
3. **Las familias que más suben son las de 3-4 plantillas con F1 bajo bajo P2** (JIGSAW, MAZE,
   SUNCRYPT, CLOP, SODINOKIBI, LOCKBIT): pasan de entrenar con 1,5-2 a entrenar con 2-3.
4. **BADRABBIT y CRYPTOLOCKER siguen en 0,0000** (1 plantilla: bajo LOGO no hay con qué
   entrenar). **HELLOKITTY sube poco** (cohesión 0,26: más datos no arregla heterogeneidad).

### Criterio de adopción

LOGO se **reporta como protocolo adicional** si la predicción 1 se cumple. **No reemplaza P2.**
Si LOGO supera 0,50 con IC que excluye 0,50, se declara que **el umbral del tutor se supera bajo
el protocolo de máximo aprovechamiento**, y se explica la diferencia con P2 como efecto del
tamaño de entrenamiento, no del método.

## ★★★ EL TUTOR PIDE «MÉTRICAS» PARA LA INSUFICIENCIA DE DATOS — YA ESTÁN HECHAS (2026-09-09)

> **Contexto.** WhatsApp del 9/9: Romina pregunta qué pasa con las familias de una sola
> plantilla; Cappo responde «**No se procesa**» y «**Por eso necesitamos métricas**». Además
> circula un PDF generado por IA (`Downloads/Ransomware - Notas de rescate insuficientes.pdf`,
> 9 pág.) con un checklist de cómo demostrar que las notas por familia son insuficientes.
> **El checklist describe, casi punto por punto, lo que la tesis ya tiene medido — con un
> protocolo más estricto del que el PDF propone.** Y el «50 %» que el tutor exige lo está
> juzgando sobre el **último informe que vio (18-08): P2 = 0,435 sobre 146 notas, texto solo**.
> No ha visto M.6, M.3, la limpieza ni nada posterior.

### Cruce del checklist del PDF con lo que existe

| El PDF recomienda | Estado en la tesis |
|---|---|
| Curvas de aprendizaje con fracciones crecientes; estimar la pendiente final | ✅ **B.1**, y más fuerte: ajuste de ley de potencia F(k)=F∞−a·k^(−b) con **techo F∞ y IC bootstrap (2.000)**. Techo P2 **0,470** [0,411; 0,527]; 0,50 «no alcanzable con más datos» |
| IC 95 % y dispersión entre particiones, no un número suelto | ✅ En todo el frente: **50 semillas**, Δ pareado por semilla, IC 95 % t-Student |
| F1 / precisión / recall por familia, balanced accuracy, matriz de confusión | ✅ `corrida_canonica_por_familia.csv`, `fig_confusion_canonica.png`, bal.acc en todas las tablas |
| **MCC** | ❌ **faltaba → agregado hoy**: texto solo **0,5610 ± 0,0789**, M.6 **0,6446 ± 0,0736** (P2, 50 semillas, 149) |
| Dispersión N vs F1 y **CV(F1) por familia** («la evidencia más fuerte») | ❌ **la tabla faltaba → hecha hoy**: `m6_cv_por_familia.csv`. **Spearman(n_plantillas ; F1) = +0,496 (p=0,005)** y **Spearman(n_plantillas ; CV) = −0,647 (p<0,001)**. Es exactamente el patrón N↓ ⇒ Var(F1)↑ que el PDF pide ver |
| Subsampling de familias grandes para estimar N mínimo | ✅ **B.1 «tope por plantillas»** es exactamente ese experimento; corte en **3 plantillas**, paso 2→3 último significativo |
| Baseline char n-gram TF-IDF + Linear SVM | ✅ **es el clasificador canónico** |
| Después un Transformer / Sentence-BERT | ✅ **Exp. 3e**, `paraphrase-multilingual-MiniLM-L12-v2`: solo empeora (−0,0733), concatenado plano |
| **Independencia de muestras**: agrupar casi-duplicados, N efectivo ≠ N crudo, split aleatorio infla | ✅✅ **Es P2 entero.** N crudo **149** → N efectivo **99 plantillas** (coseno char 3-5 ≥ 0,90). Y el «98 % falso» del PDF está **cuantificado**: P1 (split de notas) 0,7889 vs P2 (split de plantillas) 0,4593; protocolo de Lemmou (mundo cerrado) 0,7767 |
| Imbalance ratio | ❌ **agregado hoy**: 19/2 = **9,5** |
| Número efectivo E_n (Cui et al.) | ❌ no; marginal, se puede agregar en una línea |
| RQ: «¿cuántas notas independientes por familia hacen falta?» | ✅ **B.1 la contesta**: corte 3 plantillas; y **más allá del corte sumar material BAJA el F1** (5 pasos con IC entero < 0) |

**El único hueco real del checklist eran MCC y la tabla CV(F1). Los dos están hechos.**

### Lo que hay que decirle a Cappo, en este orden

1. **«No se procesa» es correcto y está medido:** BADRABBIT y CRYPTOLOCKER (1 plantilla) dan
   **F1 exactamente 0,0000** bajo P2 — el propio script lo advierte. Y en las dos es
   **propiedad real del malware** (WASTEDLOCKER no: tiene 3 plantillas, es el caso frontera).
2. **Las métricas que pide existen y van más allá del checklist.** Tabla de arriba.
3. **El 0,435 que vio es de hace un mes.** Hoy: texto solo 0,4593; **M.6 0,5191**
   [0,4967; 0,5416]; y sobre las **28 evaluables M.6 = 0,5562 [0,5322; 0,5802] — supera 0,50
   con el intervalo entero**.
4. **La insuficiencia de datos no es una excusa: es el resultado central**, exactamente como el
   PDF recomienda formularlo («convertirlo en una de las preguntas centrales del TFG»). Y el
   techo de 0,470 con IC es la versión cuantificada de esa conclusión.

⚠️ Lo que el PDF **no** contempla y la tesis sí: que **más plantillas empeora** pasado el corte
(la curva no es monótona), que **la cohesión** entre plantillas predice el F1 mejor que N, y
que **reglas exactas** (M.6) atraviesan el techo del texto. Eso es aporte propio.

## ★★★★ RESULTADO P2-LOGO: EL PROTOCOLO QUE NADIE PROBÓ CAMBIA LA ESCALA (2026-09-09)

> **Base: 149 notas · 99 plantillas · 30 familias · 10 semillas.** Código:
> `2_codigo/protocolo_logo.py` (commits `059cf5f`, `7dd6db9`). Salidas en
> `4_resultados/resultados_protocolo_logo_149/`. **Se AGREGA como protocolo adicional; P2 no se
> reemplaza.** Predicciones preregistradas en el bloque anterior; **dos de cuatro fallaron** y
> se reportan.

### El resultado

| Protocolo | Capa | macro-F1 | IC 95 % | exactitud | bal.acc | cobertura regla |
|---|---|---|---|---|---|---|
| P2 (2 pliegues, 10 sem.) | texto solo | 0,4680 ± 0,100 | entre semillas | 0,580 | 0,528 | — |
| P2 (2 pliegues, 10 sem.) | M.6 | 0,5318 ± 0,104 | entre semillas | 0,665 | 0,589 | 0,443 |
| **LOGO** (99 pliegues) | **texto solo** | **0,6747** | boot. notas **[0,567; 0,712]** | 0,738 | 0,718 | — |
| **LOGO** (99 pliegues) | **M.6** | **0,7742** | boot. notas **[0,663; 0,810]** | **0,859** | **0,810** | **0,604** |

**Δ pareado por semilla, LOGO − P2:** texto **+0,2067** [+0,135; +0,279], M.6 **+0,2424**
[+0,168; +0,317], **10/10 semillas** en los dos. **M.6 sobre texto, bajo LOGO: +0,0994.**

**Sobre las 28 evaluables:** LOGO texto **0,7229**, LOGO M.6 **0,8295**. Bajo LOGO-M.6, **26 de
30 familias superan 0,50 y 22 superan 0,70**.

> **Lo que esto dice, sin adornar:** mismo corpus, misma vista, mismo clasificador, misma
> garantía (la plantilla de prueba nunca está en entrenamiento). Lo único que cambia es que el
> modelo entrena con k−1 plantillas por familia en vez de ~k/2. **Y el macro-F1 pasa de 0,47 a
> 0,67 (texto) y de 0,53 a 0,77 (M.6).** El «techo» de 0,470 que B.1 extrapoló era el techo
> **de P2 con 2 pliegues**, no del problema. Con más entrenamiento el mismo método llega a 0,77.

### ⚠️ La objeción que va a venir primero —«¿es fuga?»— y su respuesta

1. **Los grupos son idénticos a los de P2** (casi-duplicados a coseno char 3-5 ≥ 0,90).
2. **Comprobación directa, medida:** en los 99 pliegues, el coseno máximo entre la nota de
   prueba y cualquier nota de entrenamiento es **0,8995** (umbral 0,90); **0 pliegues** con una
   casi-copia en entrenamiento. En P1, en cambio, **61 de 149 notas de prueba (41 %)** tienen su
   casi-copia en entrenamiento (semilla 0) — eso es lo que P1 mide y LOGO no. **Es la
   cuantificación exacta del «98 % falso» que advierte el PDF del tutor.**
3. **LOGO queda por debajo de P1** (0,6747 < 0,7889), como se predijo: P1 es el protocolo que
   SÍ deja casi-duplicados a los dos lados del corte.
4. **Tres familias caen a 0,000 bajo LOGO-texto** (HELLOKITTY, CHIMERA, más las dos de 1
   plantilla). **Una fuga inflaría todo; esto corta para los dos lados.**

### Balance honesto de las 4 predicciones preregistradas

| # | Predicción | Resultado | Veredicto |
|---|---|---|---|
| 1 | texto-LOGO sube sustancialmente, rango 0,55-0,65, debajo de P1 | **0,6747**, debajo de P1 | ✅ dirección y mecanismo; **magnitud subestimada** (quedó arriba del rango) |
| 2 | el Δ de M.6 sobre texto es **menor** bajo LOGO que bajo P2 | P2 +0,064 → LOGO **+0,099** | ❌ **FALLÓ: es mayor.** La regla aporta más cuando el texto está mejor entrenado, no menos |
| 3 | las que más suben: 3-4 plantillas con F1 bajo (JIGSAW, MAZE, SUNCRYPT, CLOP, SODINOKIBI, LOCKBIT) | las que más suben son **las de 2 plantillas** (SUNCRYPT, NETWALKER, DARKSIDE, CUBA, BLACKMATTER → **1,000**), y SODINOKIBI, LOCKBIT | ⚠️ parcial: acertó familias, **erró el mecanismo** (ver abajo) |
| 4 | BADRABBIT y CRYPTOLOCKER siguen en 0; HELLOKITTY **sube poco** | las dos en 0 ✓; **HELLOKITTY cae a 0,000** (era 0,214) | ⚠️ mitad: ✅ los ceros estructurales; ❌ HELLOKITTY **bajó**, no subió |

### El mecanismo que las predicciones no vieron — y es el hallazgo

Se predijo que LOGO ayudaría a cada familia por **su propio** entrenamiento (k/2 → k−1). Pero
una familia de **2 plantillas** entrena con **1** plantilla propia tanto en P2 como en LOGO — y
aun así salta de ~0,45 a **1,000**. **Lo que cambia no es su entrenamiento: es el de las otras
29.** Con 2 pliegues, las 29 competidoras están tan mal entrenadas como ella y sus fronteras
sangran hacia todos lados; con LOGO todas las fronteras son nítidas y **la precisión se
dispara**. Es un efecto **global**, no por familia.

Y por eso HELLOKITTY y CHIMERA-texto **bajan a 0**: con fronteras nítidas en las 29 vecinas, una
familia cuyas 3 plantillas no se parecen entre sí **no tiene dónde esconderse** — su plantilla
de prueba es absorbida con confianza por una vecina bien definida. Bajo P2 la confusión general
le daba aciertos casuales. **LOGO es más honesto en los dos sentidos:** premia a las familias
coherentes y castiga sin piedad a las heterogéneas. (CHIMERA se recupera a **0,800** con M.6:
la regla la rescata donde el texto no puede.)

### Consecuencia para la tesis, y hay que decidirla con Cappo

1. **El «50 %» se supera con holgura bajo LOGO**: texto solo 0,6747 [0,567; 0,712], M.6 0,7742
   [0,663; 0,810], **IC bootstrap entero por encima de 0,50** en los dos.
2. **Hay que re-leer el techo de B.1.** El 0,470 era el techo *de la curva bajo P2 con 2
   pliegues*. No estaba mal calculado; estaba midiendo otra cosa. **Con LOGO la curva de
   aprendizaje tendría que re-correrse** (B.1 bajo LOGO) para saber si el techo real del
   problema es ~0,77 o más.
3. **Reportar los tres protocolos juntos**, con lo que mide cada uno: P1 (mundo casi cerrado)
   0,79 · **LOGO (máximo entrenamiento, plantilla no vista) 0,67 / 0,77** · P2 (mitad del
   entrenamiento, plantilla no vista) 0,46 / 0,52. La diferencia P2→LOGO es **tamaño de
   entrenamiento**, no método ni fuga.
4. ⚠️ **Autocrítica que hay que escribir:** P2 con 2 pliegues fue la elección canónica de todo
   el frente de notas durante meses, justificada por «hay familias con 2 notas». Esa
   justificación excluye k-fold con k≥3 pero **nunca excluyó LOGO**, que es el protocolo
   estándar para exactamente este caso. **Nadie lo preguntó hasta hoy.** Todas las cifras de
   notas del capítulo 4 están bajo P2 y son correctas *para P2*; lo que cambia es que P2 no era
   el protocolo más informativo disponible.

**Pendiente inmediato:** B.1 (curva y techo) bajo LOGO, M.3 (abstención) bajo LOGO, y 50
semillas de P2 en el mismo proceso para que la comparación tenga la misma base que M.6-149.

### RESULTADO P2-LOGO a 50 semillas (2026-09-10) — CIFRAS DEFINITIVAS del protocolo

`protocolo_logo.py --n-semillas 50` (commit `84187b7`), log `4_resultados/_log_logo_149_50sem.txt`,
salidas en `resultados_protocolo_logo_149_50sem/`. **Control de equivalencia cumplido:** la semilla 0
reproduce P2 0,5988 / 0,6320 y LOGO 0,6747 / 0,7742 de la corrida de 10. **Control externo cumplido:**
P2 texto da **0,4593**, exactamente la cifra canónica de `resultados_notas_149` a 50 semillas — el
script reproduce el evaluador canónico.

| Protocolo | Capa | macro-F1 | ± sd | IC 95 % | exact. | bal. | cob. regla | ¿>0,50? |
|---|---|---|---|---|---|---|---|---|
| P2 | texto solo | 0,4593 | 0,075 | semillas [0,438; 0,481] | 0,579 | 0,520 | — | NO |
| P2 | M.6 | 0,5191 | 0,079 | semillas [0,497; 0,542] | 0,660 | 0,578 | 0,455 | media sí, IC toca |
| **LOGO** | texto solo | **0,6747** | 0 | boot. notas [0,567; 0,712] | 0,738 | 0,718 | — | **SÍ** |
| **LOGO** | M.6 | **0,7742** | 0 | boot. notas [0,663; 0,810] | 0,859 | 0,810 | 0,604 | **SÍ** |

**Δ pareados por semilla (50 semillas):** LOGO − P2 texto **+0,2155** [+0,194; +0,237], 50/50;
LOGO − P2 M.6 **+0,2550** [+0,233; +0,278], 50/50; M.6 − texto bajo LOGO +0,0994.
⚠️ El IC de este último sale degenerado [+0,0994; +0,0994] porque LOGO es determinista: la
incertidumbre de ese Δ hay que darla con bootstrap por notas, no por semillas; el script no
guarda predicciones por nota, así que queda como pendiente menor (los IC de cada capa por
separado sí están).

**Por familia (texto solo), las que más ganan:** SUNCRYPT 0,40→**1,00**, CUBA 0,43→**1,00**,
NETWALKER 0,45→**1,00**, BLACKMATTER 0,52→**1,00**, DARKSIDE 0,56→**1,00**, LOCKBIT 0,44→0,86,
SODINOKIBI 0,36→0,75, RANSOMEXX 0,50→0,80. **Las cinco que llegan a 1,00 son familias de 2
plantillas.** Bajo P2 de 2 pliegues cada una queda con 1 plantilla en train igual que bajo LOGO;
lo que cambia es el resto del corpus (mitad vs. todo), o sea la nitidez global de la frontera. Es
la evidencia por familia del mecanismo.
**Las que no se mueven o caen:** BADRABBIT 0→0 y CRYPTOLOCKER 0→0 (plantilla única, estructural);
HELLOKITTY 0,17→0,00 y CHIMERA 0,14→0,00 (2 plantillas disímiles entre sí: entrenar con una no
sirve para la otra); PHOBOS 0,44→0,42 (heterogénea).

**Control externo también para M.6 — y una errata mía, corregida dos veces:** escribí que el M.6
canónico a 50 semillas daba 0,5318 y que había una «discrepancia». No la hay. El **0,5318 es real,
pero es otra cosa**: es el P2-M.6 de `protocolo_logo.py` a **10 semillas** (media de las semillas
0–9 de `_log_logo_149.txt`, verificada: 0.5318; texto 0.4680), la fila que
está en la tabla del bloque anterior. Lo confundí con el canónico. El canónico de la cascada sobre
149 a 50 semillas (`_log_m6_149.txt`) da base texto **0,4593** y variante adoptada
`privados_sin_circ_MAS_NOMBRE` **0,5191**, Δ +0,0599 [+0,0530; +0,0668], 50/50 — **exactamente** lo
que reproduce la columna P2 de `protocolo_logo.py` a 50 semillas (0,4593 / 0,5191). Por inspección,
`dicc_privados` + `regla` implementan esa variante (filtro de genéricos + nivel de nombre, sin filtro
de circularidad, unanimidad). Primera versión de esta nota decía «el 0,5318 no figura en ningún
archivo»: también falso; figura como cifra a 10 semillas. Lección repetida dos veces en una hora:
verificar, no recordar, y **mirar la base (n de semillas) antes de comparar dos cifras**.

→ Revisión independiente: ver `6_notas_trabajo/REVISION_LOGO_2026-09-17_informe.md` (veredicto: **no adoptar LOGO como cifra principal** — el salto P2→LOGO es en un 95 % artefacto de la partición de P2 (P2bal, misma mitad de entrenamiento, da 0,6651 ± 0,032; LOGO 0,6747); el IC por plantilla [0,539; 0,722] sí excluye 0,50, pero las 5 familias en 1,00 son pares de notas contenidas una en otra y con fusión por contención ≥ 0,8 LOGO-texto cae a 0,4186 sobre 30 familias).

### Nota operativa: la corrida de 50 semillas (2026-09-10)

La primera corrida `protocolo_logo.py --n-semillas 50` (lanzada 17:04 del 9/9) **murió a las
23:38 del 9/9 por apagado de la PC** (registro de eventos de Windows: 6006 a las 23:37:58 y 1074
«Apagar» a las 23:39:36), sin llegar a escribir: log vacío y exit 4. No fue error del script.
Antes de relanzar se verificó en `_log_logo_149.txt` que **LOGO da idéntico en las 10 semillas**
(0,6747 / 0,7742, cuarto decimal): partición fija + LinearSVC convexo. Con eso, el script ahora
evalúa LOGO una sola vez y lo reutiliza (commit `84187b7`), exactamente equivalente y ~4.850
ajustes menos; la corrida de 50 pasa de horas a minutos. Relanzada el 10/9 con `python -u`
(log línea a línea) en `resultados_protocolo_logo_149_50sem/`. Control: la semilla 0 debe
reproducir P2 0,5988 / 0,6320 y LOGO 0,6747 / 0,7742 de la corrida de 10.

### PRECISIÓN a lo anterior (2026-09-09, misma sesión): LOGO **confirma** B.1, no lo derriba

Al preparar la extensión de `curva_aprendizaje_notas.py` para LOGO encontré que **B.1 ya tiene un
protocolo de máximo entrenamiento: P2ret** (retiene UNA plantilla por familia; entrena con las
demás, 70 plantillas; 100 repeticiones). Verificado en
`4_resultados/resultados_curva_149/b1_curva_por_repeticion.csv` y `_log_resumen_cap4_149.txt`:

| Protocolo (k=todo, texto solo) | macro-F1 | plantillas en train | **familias sin train / pliegue** |
|---|---|---|---|
| P1 (2 pliegues, por nota) | 0,8048 ± 0,028 | 58,8 | 0 |
| P2 (2 pliegues, por plantilla) | 0,4680 ± 0,100 | 50,5 | **3,9** |
| P2ret (1 plantilla/familia fuera) | **0,6433 ± 0,061** | 70,2 | 2,0 |
| LOGO (1 plantilla fuera) | **0,6747** [0,567; 0,712] | 98 | 2,0 (solo en sus 2 pliegues) |

**Corrección de lo que escribí arriba:** el «techo 0,470» de B.1 que cité es el de la **fila P2**;
la fila P2ret ya daba **0,643 en k=todo** y un techo extrapolado de ~0,66. LOGO (0,675) está
+0,03 por encima, lo esperable por 28 plantillas más de entrenamiento. **No hace falta re-correr
B.1 bajo LOGO**: la pregunta «¿el techo real es ~0,77 o más?» ya está contestada por P2ret para
texto solo (~0,66) y por LOGO para M.6 (0,774; el M.6 no está en B.1).

**Dos cosas nuevas que salen del CSV y van a la defensa:**
1. **Bajo P2, 3,9 familias por pliegue quedan sin ninguna plantilla de entrenamiento** (F1
   estructural 0). P2 no solo entrena con la mitad: borra ~4 familias del entrenamiento en cada
   corte. Es una parte cuantificable del hueco P2→LOGO (0,468→0,675).
2. **Bajo P2ret y LOGO quedan 2: las familias de plantilla única (BADRABBIT, CRYPTOLOCKER).** Es la
   pregunta del tutor («¿y las de una sola plantilla?») en números: en texto puro **no se procesan,
   F1 = 0 por construcción**; M.6 las rescata solo si comparten IOC/nombre genuino (ver tabla por
   familia de LOGO). Familias con 2 plantillas: 7 (BLACKMATTER, CHIMERA, CUBA, DARKSIDE, NETWALKER, NOTPETYA, SUNCRYPT).
3. **La curva P2ret ya mostraba la meseta:** 0,67 en k=3–4 notas/familia y **baja** hasta 0,643 en
   k=todo, con IC 95 % que excluye el cero por debajo en 4→6, 6→8, 8→12 y 12→todo. Es la respuesta
   a «¿por qué no mejora agregar más plantillas?»: a partir de 3–4 el aporte marginal es ≤0 en
   este corpus.

**La autocrítica se acota:** no es que «nadie probó máximo entrenamiento» — P2ret lo hizo y dio
0,643. Lo que faltó fue **usarlo como cifra de cabecera y leer la diferencia P2↔P2ret como efecto
del protocolo**, no del método. LOGO agrega el ajuste fino (una plantilla fuera, 99 pliegues
deterministas, sin la varianza de elegir cuál retener) y la extensión a M.6.

### M.3 bajo LOGO: contesta el 85 % y acierta el 94 % (2026-09-09)

`abstencion_notas.py --protocolo LOGO --n-semillas 1` (commit `ecf550c`). Salidas en
`4_resultados/resultados_abstencion_149_LOGO/`. Misma arquitectura: la regla de M.6 contesta
siempre; el umbral de margen se aplica solo cuando decide el texto.

| Umbral | Cobertura | **Acierto donde contesta** | Abstiene | resueltas por regla / por texto |
|---|---|---|---|---|
| 0,00 | 1,000 | 0,8591 | 0 | 90 / 59 |
| 0,20 | 0,899 | 0,9179 | 15 | 90 / 44 |
| **0,50** | **0,846** | **0,9444** | 23 | 90 / 36 |
| **1,00** | **0,752** | **0,9911** | 37 | 90 / 22 |
| 1,50 | 0,638 | 1,0000 | 54 | 90 / 5 |

**Comparado con P2 (mismo umbral 0,50): cobertura 0,646 → 0,846 y acierto 0,900 → 0,944.**
La capa de reglas resuelve **90 de 149 notas (60 %)** en todo pliegue — coincide con la
cobertura 0,604 de `protocolo_logo.py` — así que la cobertura de M.6 nunca baja de 0,60 aunque
el umbral suba; el texto solo degenera a 0,114 con umbral 1,50.

**Frases para la defensa, bajo el protocolo de máximo entrenamiento:** «contesta el 85 % de las
veces y acierta el 94 %» (umbral 0,50), o «contesta el 75 % y acierta el 99 %» (umbral 1,00).
Misma advertencia que en P2: el macro-F1 *de las respondidas* no es comparable con el del
sistema completo y no se cita como tal; la abstención reduce el problema de mundo cerrado, no lo
elimina.

---

# Exp. 2d — job 3771: corrió un script VIEJO. Qué se salva y qué hay que repetir (2026-09-10)

**Resumen en una línea:** el job 3771 se lanzó con la versión de
`exp2d_nombre_extension.py` **anterior al commit d537ebd**, o sea con los hiperparámetros por
defecto de sklearn y no con los del Exp. 2c. Sus tres columnas **no son el experimento** y hay
que repetirlas. El **diagnóstico de la extensión sí se salva**, porque es una propiedad del
conjunto de datos y no del modelo, y es el hallazgo más fuerte de la corrida.

## Cómo se sabe que corrió el script viejo (cuatro indicios, ninguno ambiguo)

1. **Fechas.** El job arrancó `Fri Aug 28 11:12:32 PM -04 2026`. El commit d537ebd
   —«corrige los hiperparámetros, que decían ser los del Exp. 2c y eran los de sklearn por
   defecto»— es del **2026-08-29**, el día siguiente. La corrección llegó después del job.
2. **Los hiperparámetros viejos eran** `max_depth=None, min_samples_leaf=1,
   max_features="sqrt"` (verificado con `git show d537ebd`), no `300/20/2/0,3`.
3. **Falta el control de integridad en el log.** La línea «Control de integridad: ningún
   archivo con magia de tipo conocido. OK.» se agregó en el **mismo** commit d537ebd, y en el
   log del 3771 no aparece: después de `WASTEDLOCKER 500 archivos` viene directo el
   diagnóstico. Confirmación independiente de la versión.
4. **Los tiempos cierran.** 250 s por semilla para tres columnas (~83 s por evaluación) contra
   **~560 s por semilla** del job 3648. `max_features="sqrt"` mira 32 características por corte
   y `0.3` mira 307: diez veces más barato por corte. El script viejo tenía que ser más rápido,
   y lo fue.

**Consecuencia sobre la cancelación:** el `scancel` de las 23:28 fue, con toda probabilidad,
**deliberado** —se detectó el error de hiperparámetros y se mató el job— y d537ebd es la
respuesta. No se perdió nada válido. Esa versión tampoco tenía A.3, así que la curva de
aprendizaje nunca iba a salir de ahí.

## ⭐ LO QUE SÍ SE SALVA: la extensión ES la etiqueta (job 3771, semilla 0)

Esto no depende de los hiperparámetros ni del modelo. Es un recuento sobre los nombres de
15 000 archivos de las 30 familias, y por eso vale tal como está:

| Medida | Valor |
|---|---|
| Familias con **una sola extensión** | **25 de 30** |
| Extensiones compartidas por más de una familia | **5 de 905** |
| **Exactitud de una tabla de consulta que solo mira la extensión** | **0,9724** |

Las cinco extensiones compartidas son `.doc`, `.docx`, `.pptx`, `.xls`, `.xlsx`, y las comparten
**BADRABBIT, NOTPETYA y JIGSAW**: son las familias que **no renombran** el archivo y dejan la
extensión original. Las otras 25 renombran, cada una con su extensión propia y única.

**Una tabla de consulta sin aprendizaje —«mirá la extensión, decí la familia más frecuente»—
acierta 0,9724, contra 0,9019 de macro-F1 del clasificador de bytes.** Le gana por +0,07 al
método de la tesis. Con eso, la columna (3) queda declarada como lo que el script ya decía que
era: **memorización de un diccionario, no capacidad de identificar familias.**

*Matiz que hay que escribir junto al número:* el 0,9724 se mide **dentro de la muestra** (la
familia mayoritaria de cada extensión, evaluada sobre las mismas filas), así que para una
extensión que aparece una sola vez acierta por construcción — es una **cota superior**. Lo que
no es artefacto de muestreo es la parte estructural: **25 de 30 familias tienen una extensión
única**, y para esas 25 la extensión *es* la etiqueta, sin ambigüedad posible.

## Las tres columnas del job 3771 — con hiperparámetros equivocados, NO CITAR

Cuatro semillas completas (0 a 3); la 4 se cortó. Media ± desvío, y delta pareado por semilla
con IC 95 % (n = 4, t de Student):

| Columna | Exactitud | macro-F1 |
|---|---|---|
| (1) solo bytes | 0,9018 ± 0,0025 | 0,9019 ± 0,0024 |
| (2) bytes + forma del nombre | 0,9986 ± 0,0001 | 0,9986 ± 0,0001 |
| (3) bytes + extensión literal | 0,9842 ± 0,0006 | 0,9840 ± 0,0007 |

| Delta pareado (macro-F1) | Δ | IC 95 % | Semillas a favor |
|---|---|---|---|
| (2) − (1) | **+0,0967** | [+0,0928; +0,1005] | 4/4 |
| (3) − (1) | **+0,0821** | [+0,0777; +0,0864] | 4/4 |
| **(2) − (3)** | **+0,0146** | [+0,0134; +0,0158] | 4/4 |

**Puerta de entrada: FALLA.** La columna (1) da 0,9018 ± 0,0025 de exactitud contra los
**0,9120 ± 0,0016** publicados del Exp. 2c (job 3648). La diferencia de −0,0102 es
exactamente lo que anticipaba el mensaje de d537ebd, y ya está explicada por el script viejo:
no es un problema de datos ni un anómalo sin causa. **Ninguna de las tres cifras se cita.**

## Lo que estas cifras sí anticipan (dirección, no magnitud)

**La forma del nombre le gana a la extensión literal** por +0,0146 de macro-F1, con las 4/4
semillas a favor y un IC que excluye el cero por lejos. Eso invierte la predicción que estaba
escrita —«el nombre no aporta, lo que aportaba en el Exp. 2b era la extensión»— y apunta a que
la columna (2) esté midiendo lo mismo que la (3), por otra puerta: **la forma del nombre es un
identificador de campaña mejor que la extensión.**

Tiene sentido leyendo `forma_del_nombre()`: incluye `len_ext`, `len_base`, `prop_hex`,
`base_toda_hex`, `ext_solo_letras`, `ext_con_digitos`, `n_puntos`. Eso *es* el esquema de
renombrado. Con 25 de 30 familias renombrando cada una a su manera, el esquema identifica la
familia **sin mirar un solo byte del contenido**. Y el diagnóstico previo del script medía la
circularidad de la extensión **literal**, no la de la **forma**: la columna (2) entró por la
puerta que quedó sin vigilar.

## Corrección del diseño (commiteada y pusheada)

1. **Dos columnas de control SIN bytes**, misma partición, delta pareado:
   `0a_solo_forma_del_nombre` y `0b_solo_extension_literal`. Son las que hacen interpretable la
   columna (2):
   - si **(0a) ≈ 0,98–0,99** → el nombre solo ya resuelve la tarea, y el +0,097 de la columna
     (2) no es «los bytes ayudados por el nombre» sino **el nombre con los bytes de
     acompañantes**. La columna (2) pasa a ser una segunda cota superior declarada, hermana de
     la (3), y no un método defendible sobre este conjunto;
   - si **(0a) queda claramente por debajo de (2)** → la combinación aporta algo que ninguna
     parte tiene sola, y ahí la columna (2) es reportable.
   *Predicción registrada antes de correr: (0a) entre 0,97 y 0,99.* **→ FALSADA el 17-09: dio 0,5771. Ver «EXP. 2d CERRADO — job 3937».**
2. **Puerta de entrada** `REF_2C = 0,9120` / `TOL_2C = 0,0050`: compara la columna (1) contra el
   Exp. 2c y avisa en el log. Si hubiera existido, el 3771 se detectaba en la primera semilla en
   vez de a ojo.
3. **Guardado incremental** de `exp2d_por_semilla.csv` al cerrar cada semilla. En el 3771 los
   `to_csv` estaban todos después del bucle, así que la cancelación en la semilla 4 borró las
   cuatro ya medidas: **el log quedó como único registro**.

## El límite de fondo, que ninguna corrida va a resolver

**En NapierOne cada familia corresponde a una sola campaña.** Por lo tanto ninguna medición
sobre este conjunto puede separar «forma del nombre» de «identidad de la campaña», por más
control que se agregue. Los controles (0a) y (0b) sirven para *acotar* cuánto de la columna (2)
es nombre, no para rescatarla. El conjunto que separaría las dos cosas —varias campañas por
familia— es justamente el que no se tuvo para este trabajo, que es la limitación ya escrita para
el Exp. 2c y el objeto de `nota_limitacion_napierone.tex`.

**Cómo reportar el Exp. 2d, entonces:** la columna (1) es el método; las columnas (2) y (3) son
**dos cotas superiores declaradas** que miden hasta dónde llega un identificador de campaña, y
el número que las vuelve honestas es el **0,9724 de la tabla de consulta**, que muestra que ese
techo se alcanza sin aprendizaje alguno.

## Pendiente

- Relanzar el 2d con el script corregido (ya está en `PARA_SUBIR_AL_CLUSTER/`): cinco columnas,
  cinco semillas, más A.3.
- Leer, en la corrida nueva, **(0a) antes que nada**: decide cómo se reporta la columna (2).
- Verificar que el control de integridad diga «ningún archivo con magia de tipo conocido» — si
  aparecen los 12 JPEG de `CERBER-small`, la columna del nombre no es interpretable.

---

---

# ✅ CERBER: CIFRADO PARCIAL VERIFICADO — cierra el «(a verificar)» del job 3772 (2026-09-10)

Este bloque **no repite** el hallazgo del job 3772 (arriba, «CERBER DEJA LA CABECERA EN CLARO»).
Cierra la verificación que ese bloque dejaba pendiente y **corrige un razonamiento** de ahí.

## La medición

Volcado de entropía y de los primeros 32 bytes sobre `CERBER-small`, pegado por Romina:

```
CERBER-small: 989 archivos sueltos
  subcarpeta _sin_cifrar: 12 archivos
con magia de tipo en claro: 989/989

--QJB2Q7nF.bed4     12295B  JPEG  H(cab)=6.33  H(med)=7.59  H(cola)=7.63
   ff d8 ff e0 00 10 4a 46 49 46 00 01 01 00 00 01 00 01 00 00 ff fe 00 3b 43 52 45 41 54 4f 52 3a
-00CAjTujp.bed4     95142B  OLE   H(cab)=0.89  H(med)=7.58  H(cola)=7.64
   d0 cf 11 e0 a1 b1 1a e1 00 00 00 00 ... 3e 00 03 00 fe ff 09 00
-07I7Ci0DW.bed4    112672B  ZIP   H(cab)=0.88  H(med)=7.58  H(cola)=7.58
   50 4b 03 04 14 00 06 00 08 00 00 00 21 00 15 f2 d8 a4 c6 01 00 00 97 09 00 00 13 00 08 02 5b 43
```
(los otros tres, dos ZIP y un JPEG, dan lo mismo: H(cab) 0,88-6,46 · H(med) 7,58-7,64 ·
H(cola) 7,58-7,65)

## Qué queda establecido

1. **H1 CONFIRMADA — cifrado parcial, no contaminación.** Cabecera de **entropía baja**
   (0,88-6,46) y **cuerpo y cola de entropía alta** (7,58-7,65, el techo de una ventana de 512
   bytes). Los archivos **están cifrados**; lo que no está cifrado es el principio. La lectura
   del job 3772 era la correcta.
2. **Son 989 de 989, no 500 de 500.** El 500/500 del log era el tamaño de la muestra
   (`--por-familia 500`). Sobre la carpeta completa, **todos** los archivos tienen magia en
   claro. Sin excepciones.
3. **Los 12 JPEG siguen apartados** en `_sin_cifrar/`, con sus 12 archivos. Era el chequeo
   previo al lanzamiento que pedía el bloque del 3772: **cumplido**. Los 989 son otra cosa.
4. **Cuánto se preserva: del orden de la ventana entera, no unos pocos bytes.** En OLE y ZIP la
   entropía de los 512 bytes de cabecera es **0,88**, o sea que casi toda la ventana es
   estructura original: cabecera OLE completa (`d0cf11e0 a1b11ae1` + relleno de ceros), o
   cabecera local de ZIP más el nombre `[Content_Types].xml` (`13 00` = 19 caracteres, y
   `5b 43` = `[C`). En JPEG llega a 6,33-6,46 porque después del JFIF y del marcador COM
   (`ff fe`, con el texto `CREATOR:`) ya entra ciframiento.

## ⚠ Corrección al bloque del job 3772

Ese bloque dice, sobre el F1 = 1,000 de CERBER en el Exp. 2c: «*La cabecera preservada es del
documento de origen y la comparten todas las familias, así que no discrimina*».

**Eso no se sostiene contra la medición.** Las otras familias **cifran su cabecera**: en
`d_familias_dificiles.csv` la entropía de cabecera es **7,03** para el resto de familias y
**7,59** para las seis difíciles. La de CERBER es **0,88-6,46**. Todas las familias parten de
documentos con cabecera, sí, pero **solo CERBER la deja legible**, así que la cabecera **sí
discrimina**, y con margen enorme: es la diferencia entre una ventana de 512 bytes casi
constante y una de entropía casi máxima.

**Lo que no cambia:** el F1 = 1,000 sigue explicado, y ahora está **sobredeterminado** — hay
**dos** pistas triviales independientes, el sufijo constante de 64 bytes con cobertura 1,00 del
Exp. 2b **y** la cabecera en claro. No hace falta corregir el 0,912 por esto; hace falta que la
tesis no diga que la cabecera no discrimina.

## Lo que esto habilita

CERBER es **un caso real de cifrado parcial dentro del corpus canónico**, medido y no supuesto.
Eso conecta directo con el experimento pendiente de cifrado intermitente y con arXiv 2510.15133
(detección solo en los extremos 61,48 % contra 97,1 % a nivel de bloque): la pregunta «¿cuánto
degrada la ventana 512+512 con cifrado parcial?» tiene, en CERBER, una respuesta **medible sobre
datos propios** — y del lado bueno, porque la ventana cae justo donde el texto quedó en claro.

**Decisión que sigue abierta para Romina y Cappo:** si se declara como resultado sobre la
familia (cifrado parcial documentado, con el volcado como evidencia) o si además se cambia algo
del diseño de la ventana. Nada de esto se escribe en la tesis sin esa decisión.

---

---

# ✅ A.3 CERRADA — curva de aprendizaje del frente de archivos (job 3772, 2026-08-29)

Registrado de lo pegado por Romina el 2026-09-17, del log `slurm-exp2d-3772.out`. **El job 3772
SÍ terminó** (Fin: Sat Aug 29 02:58:47), y sus resultados nunca se habían anotado: el registro
quedó en «3772 corriendo». Salidas en `/scratch/ralfonzo/tesis/resultados_exp2d_job3772`.

Solo bytes (la representación canónica), subconjuntos **anidados**, 5 pliegues, 3 semillas de
submuestreo. 4434 s (74 min).

| Archivos/familia | Total | Exactitud | macro-F1 |
|---|---|---|---|
| 10 | 300 | 0,8111 ± 0,0158 | 0,7809 ± 0,0184 |
| 25 | 750 | 0,8627 ± 0,0074 | 0,8464 ± 0,0086 |
| 50 | 1500 | 0,8809 ± 0,0105 | 0,8723 ± 0,0114 |
| 100 | 3000 | 0,8918 ± 0,0070 | 0,8880 ± 0,0054 |
| 200 | 6000 | 0,9055 ± 0,0029 | 0,9045 ± 0,0024 |
| 350 | 10500 | 0,9102 ± 0,0008 | 0,9096 ± 0,0009 |
| **500** | **15000** | **0,9124 ± 0,0005** | **0,9117 ± 0,0005** |

## ⭐ Puerta de entrada: PASA

**Con 500 archivos/familia la curva da exactitud 0,9124 ± 0,0005 contra los 0,9120 ± 0,0016
publicados del Exp. 2c** (job 3648). Reproduce la referencia. Eso **confirma el diagnóstico**
sobre el job 3771: su 0,9018 venía de los hiperparámetros por defecto de sklearn, no de los
datos ni del diseño. Con `300/20/2/0,3` el número vuelve a su lugar.

## Delta pareado entre tamaños consecutivos (macro-F1, IC 95 %, n = 3)

| Paso | Δ | IC 95 % | |
|---|---|---|---|
| 10 → 25 | +0,0654 | [+0,0015; +0,1294] | aporta |
| 25 → 50 | +0,0259 | [+0,0146; +0,0372] | aporta |
| 50 → 100 | +0,0158 | [−0,0023; +0,0338] | indistinguible |
| 100 → 200 | +0,0164 | [+0,0082; +0,0246] | aporta |
| 200 → 350 | +0,0051 | [−0,0007; +0,0109] | indistinguible |
| 350 → 500 | +0,0021 | [−0,0006; +0,0048] | indistinguible |

## Cómo leerlo, con el matiz que corresponde

El script concluye «el último paso que aporta de forma medible es 100 → 200». **Es correcto
pero no hay que leerlo como un punto de saturación limpio**, y conviene decir por qué: son
**3 semillas**, así que el t de Student vale 4,303 y los IC son anchos. Se nota en que
50 → 100 (+0,0158) queda indistinguible mientras 100 → 200 (+0,0164) sí aporta, con deltas
casi iguales: lo que cambia es la dispersión, no el efecto. «Indistinguible de cero» acá
significa **no medible con tres semillas**, no «cero».

La lectura que sí se sostiene, y que es la que contesta la pregunta de Cappo:

> **Con 200 archivos por familia el frente de archivos ya está a 0,9045 de macro-F1. Los 300
> archivos adicionales hasta 500 aportan +0,0072 en total, y ningún paso individual de ahí en
> adelante se distingue de cero.**

Y un dato que vale por sí solo: **con 10 archivos por familia —300 en total— ya da 0,7809 de
macro-F1**. El frente de archivos necesita muy poca muestra, que es el contraste exacto con el
frente de notas, donde la moneda son plantillas y hay familias con una sola.

## Alcance acordado

`EXPERIMENTOS_PENDIENTES.md` fija para A.3: **techo de esfuerzo UN párrafo, no se re-corre por
ningún motivo, y si el resultado no aporta no se escribe.** Queda medida y registrada; **si va
a la tesis y en qué forma lo decide Romina.** Si va, el párrafo es el bloque citado arriba más
la tabla, y la comparación con B.1 (la curva del frente de notas) es lo que le da sentido:
misma pregunta, monedas distintas.

---

---

# ✅ EXP. 2d CERRADO — job 3937 (2026-09-10/11): la forma del nombre COMPLEMENTA a los bytes, no los reemplaza

Registrado de lo pegado por Romina el 2026-09-17, del log `slurm-exp2d-3937.out`. Cinco columnas
sobre la **misma partición**, 5 semillas, 5 pliegues, 500 archivos/familia, hiperparámetros del
Exp. 2c (`300/20/2/0,3`). Salidas en `/scratch/ralfonzo/tesis/resultados_exp2d_job3937`.

## Las cinco columnas

| Columna | Exactitud | macro-F1 |
|---|---|---|
| (0a) **solo** forma del nombre | 0,5830 ± 0,0024 | **0,5771 ± 0,0033** |
| (0b) **solo** extensión literal | 0,9380 ± 0,0011 | **0,9244 ± 0,0036** |
| (1) solo bytes | **0,9128 ± 0,0012** | **0,9117 ± 0,0011** |
| (2) bytes + forma del nombre | **0,9998 ± 0,0002** | **0,9998 ± 0,0002** |
| (3) bytes + extensión literal | 0,9718 ± 0,0020 | 0,9699 ± 0,0024 |

**Puerta de entrada: PASA.** «la columna (1) reproduce el Exp. 2c (0,9128 vs 0,9120). OK.»
Es la tercera confirmación independiente del diagnóstico del 3771 (con A.3 del 3772 y con
esta): el 0,9018 era por los hiperparámetros.

## Delta pareado por semilla contra «solo bytes» (macro-F1, IC 95 %, n = 5)

| Columna | Δ | IC 95 % | Semillas |
|---|---|---|---|
| (2) bytes + forma del nombre | **+0,0880** | [+0,0866; +0,0894] | 5/5 |
| (3) bytes + extensión literal | **+0,0581** | [+0,0555; +0,0608] | 5/5 |
| (0a) solo forma del nombre | **−0,3347** | [−0,3389; −0,3304] | 0/5 |
| (0b) solo extensión literal | +0,0127 | [+0,0083; +0,0171] | 5/5 |

(en exactitud: +0,0869 · +0,0590 · −0,3298 · +0,0251, todos con IC del mismo signo)

## ❌ La predicción registrada FALLÓ — y eso decide la lectura

Estaba escrito antes de correr: *«(0a) entre 0,97 y 0,99»*, y con ella la lectura de que la
columna (2) sería «el nombre con los bytes de acompañantes», una segunda cota superior.

**Dio 0,5771.** No 0,97: cuarenta puntos por debajo. **La forma del nombre sola NO resuelve la
tarea.** Y sin embargo, sumada a los bytes, llega a 0,9998 — tres errores en quince mil.

También estaba escrita la regla de decisión, con las dos ramas:

> «si **(0a) queda claramente por debajo de (2)** → la combinación aporta algo que ninguna
> parte tiene sola, y ahí la columna (2) es reportable.»

Estamos en esa rama, y por lejos: 0,58 sola, 0,91 los bytes solos, **0,9998 juntos**. Es
**complementariedad genuina**: la forma del nombre define grupos gruesos de familias (largo de
la extensión, composición de la base) con muchas colisiones —por eso sola da 0,58—, y los bytes
separan dentro de cada grupo. Ninguno de los dos hace el trabajo del otro.

**Consecuencia: la columna (2) pasa de «cota superior declarada» a RESULTADO REPORTABLE, con
una limitación declarada** (abajo). Es la respuesta al elemento de acción 2 del tutor, y es
positiva: cruzar los bytes con la forma del nombre casi elimina el error del frente de archivos
sobre NapierOne.

## Lo que el control (0b) confirma sobre la extensión literal

- **La extensión literal sola da 0,9244 de macro-F1, más que los bytes solos** (0,9117):
  +0,0127 [+0,0083; +0,0171], 5/5. Un diccionario le gana al método. Se declara tal cual.
- **0,9380 de exactitud validada contra 0,9724 de la tabla de consulta dentro de la muestra.**
  La diferencia son las extensiones que aparecen una sola vez (SUNCRYPT asigna una por archivo):
  dentro de la muestra aciertan por construcción, en validación cruzada no se pueden aprender.
  El matiz que se anotó al registrar el 0,9724 era exactamente este.
- **Sumada a los bytes rinde MENOS que la forma**: (3) 0,9699 contra (2) 0,9998. Sola, la
  extensión literal le gana a la forma por 0,35; combinada con los bytes, pierde por 0,03. Con
  900 columnas dispersas de one-hot y `max_features=0,3`, el bosque diluye la señal de los
  bytes; con 18 columnas densas de forma, no. Y para las cinco familias que no renombran, la
  extensión literal trae el **tipo del documento original**, que es ruido para identificar la
  familia.

## La limitación que va pegada al 0,9998, sin excepción

**En NapierOne cada familia es una sola campaña.** Los controles descartan que «el nombre solo
lo haga» —eso se midió y es falso—, pero **no pueden descartar que la forma del nombre sea una
huella de la campaña** que casualmente complementa a los bytes. Si la forma generaliza a otra
campaña de la misma familia depende de si el esquema de renombrado lo fija el código del
malware (extensión aleatoria de largo fijo, base hexadecimal) o lo elige el operador (extensión
con nombre). **Eso no se puede medir con este conjunto**, y es la misma limitación ya escrita
para el Exp. 2c: el conjunto con varias campañas por familia es el que no se tuvo.

**Cómo se escribe, entonces:** «sobre NapierOne, agregar la forma del nombre a los bytes lleva
el macro-F1 de 0,912 a 0,9998 (Δ = +0,088, IC 95 % [+0,087; +0,089], 5/5 semillas). La forma
del nombre por sí sola alcanza 0,577, de modo que la mejora es de la combinación y no del nombre.
Con una campaña por familia, este resultado no permite afirmar que la forma del nombre
generalice a campañas no vistas.» Las tres frases van juntas o no va ninguna.

## Contexto que hay que tener presente al leer las cinco columnas

- El control de integridad del 3937 disparó igual que en el 3772: **CERBER 500/500** con
  cabecera en claro (cifrado parcial verificado el 10-09), **JIGSAW 3** y **NOTPETYA 16**
  archivos sin cifrar con nombre original. Los 19 sin cifrar son 0,13 % y no explican un salto
  de +0,088; CERBER ya estaba en 1,000 con bytes solos, así que no mueve los deltas.
- **La pregunta natural que sigue, y que NO se corre sin decidirlo:** qué confusiones de los
  bytes resuelve la forma. La hipótesis coherente con el Exp. 2c es que las seis familias
  difíciles (NOTPETYA 0,39 · JIGSAW 0,44 · DARKSIDE 0,60 · CRYPTOLOCKER 0,60 · WASTEDLOCKER
  0,63 · SUNCRYPT 0,74) se confunden entre sí como «ciframiento genérico» y tienen esquemas de
  renombrado distintos. **Es hipótesis, no medición**: el 2d no guarda el reporte por familia.
  Si Romina y Cappo quieren el dato, es agregar un `classification_report` por columna y
  re-correr; son ~2 h de clúster. No se hace por iniciativa propia.

## Estado del Exp. 2d

**Cerrado y completo.** Tres corridas: 3771 (script viejo, no citable), 3772 (tres columnas +
A.3, terminó el 29-08), 3937 (cinco columnas + A.3, terminó el 11-09). Las cifras a usar son
**las del 3937** para las columnas y las de **cualquiera de los dos** para A.3 (misma partición,
mismas semillas: idénticas).

---

---

# ✓ Exp. 2d — CSV descargados y verificados contra lo registrado (2026-09-17)

Romina bajó `exp2d_3772_3937.tar.gz` (6,6 KB). Extraído en
`4_resultados/resultados_exp2d/resultados_exp2d_job3772/` y `.../job3937/`, más los dos logs
SLURM en `4_resultados/resultados_exp2d/`. Respaldo del tar en `4_resultados/_respaldo_exp2d/`.
Todo bajo patrones que `.gitignore` excluye.

**Verificación:** `exp2d_resumen.csv` y `exp2d_deltas.csv` del 3937 coinciden **exactamente**
con las cifras registradas desde el log pegado (las cinco columnas y los ocho deltas con IC).
`a3_curva_resumen.csv` de 3772 y 3937 son **idénticos** al redondeo de 4 decimales, como se
había anticipado. El manifiesto del 3937 confirma `300/20/2/0,3`, 500/familia, semillas 0-4,
5 pliegues, y el diagnóstico de extensión (25/30 · 5/905 · 0,9724).

**Un dato nuevo del manifiesto, con su base:** `archivos_con_magia_en_claro` dice
`CERBER: 500, JIGSAW: 4, NOTPETYA: 24`. El log de la semilla 0 decía JIGSAW 3 y NOTPETYA 16.
No es contradicción: el manifiesto guarda la **última semilla**, y cada semilla muestrea 500
archivos distintos de cada carpeta. Entonces el «19 sobre 15.000 = 0,13 %» anotado en el
bloque del 3772 **es de la semilla 0**; en la semilla 4 son 28. **El total de archivos sin
cifrar en las carpetas completas de JIGSAW y NOTPETYA no está medido** — para CERBER sí
(989/989). Queda pedido el conteo sobre carpeta completa.

---

---

# ✅ INTEGRIDAD DEL CORPUS DE ARCHIVOS, SOBRE LAS 30 CARPETAS COMPLETAS (2026-09-17)

Registrado de lo pegado por Romina. Conteo de archivos cuyo **offset 0** coincide con una de
ocho magias de tipo en claro (JPEG, PDF, ZIP/OOXML, OLE, PNG, GIF, RTF, GZIP), sobre **todos**
los archivos de nivel superior de cada carpeta de `Napierone-small`, excluyendo el `.pdf`
descriptivo. Es el conteo **definitivo**: los anteriores (500/500, 3, 16, 4, 24) eran sobre la
muestra de 500 por familia de cada semilla.

| Familia | En claro | Total | % | Qué es |
|---|---|---|---|---|
| **CERBER** | **988** | 988 | **100 %** | cifrado parcial: cabecera original preservada, cuerpo y cola cifrados (verificado 10-09 por entropía) |
| **NOTPETYA** | **32** | 833 | **3,8 %** | archivos **sin cifrar**, con nombre y extensión originales |
| **JIGSAW** | **8** | 998 | **0,8 %** | archivos **sin cifrar**, con nombre y extensión originales |
| las otras 27 | **0** | 26 857 | 0 % | — |
| **TOTAL** | **1028** | **29 676** | 3,5 % | 988 de cifrado parcial + **40 sin cifrar** |

(CERBER: 988 y no 989 porque este conteo excluye el `.pdf` descriptivo que el anterior contaba.
Los 12 JPEG apartados en `_sin_cifrar/` siguen ahí y no entran en el 988.)

## Lo que queda establecido

1. **Veintisiete familias sin una sola anomalía.** El control es exacto en el offset 0 y la
   probabilidad de falso positivo sobre ciframiento es ~2⁻²⁴ por archivo: sobre 26 857 archivos
   se esperaban 0,002 coincidencias, y hubo cero. El detector no inventa.
2. **Dos defectos de naturaleza distinta, que no se mezclan:**
   - CERBER: **el 100 % de la familia**. No es un defecto del corpus, es comportamiento del
     malware (cifrado parcial). Se declara como resultado sobre la familia.
   - NOTPETYA y JIGSAW: **40 archivos sin cifrar** (0,13 % del corpus). Es el mismo defecto de
     los 12 JPEG de CERBER del 2026-08-16: originales que el ransomware no procesó.
3. **La base del «19 sobre 15.000» queda corregida.** Ese número era la muestra de la semilla 0.
   La población es **40 sobre 29 676 = 0,13 %** (misma proporción, por casualidad). En una
   muestra de 500/familia se esperan ~4 de JIGSAW y ~19 de NOTPETYA; las semillas dieron 3-4 y
   16-24. Consistente.
4. **Tamaños de carpeta desparejos**, que conviene tener anotados: BADRABBIT **857**, NOTPETYA
   **833**, CERBER 988 (+12 apartados +1 pdf), JIGSAW 998; el resto 1000. Con `--por-familia
   500` no afecta a ninguna corrida.

## Una observación que NO es medición, y por qué importa

NOTPETYA es **la peor familia del Exp. 2c (F1 0,3942)** y es también la que más archivos sin
cifrar tiene (3,8 %). No se puede afirmar que lo uno cause lo otro: 3,8 % de la familia no
explica un F1 de 0,39. Pero hay algo que sí conviene tener presente al leer las confusiones, si
alguna vez se piden por familia: **la cabecera de un archivo sin cifrar (OLE, ZIP) es
indistinguible de la cabecera de un archivo de CERBER**, porque CERBER la deja en claro. La
mitad del vector de esos 40 archivos se parece a CERBER. Queda como hipótesis para el reporte
por familia, si se corre.

## Recomendación (la decisión es de Romina y Cappo)

Tratar los 40 igual que se trató a los 12 JPEG de CERBER, **por consistencia**: moverlos a
`NOTPETYA-small/_sin_cifrar/` y `JIGSAW-small/_sin_cifrar/` (exclusión reversible, los scripts
ya filtran con `is_file()`), declararlo en la subsección de integridad, y **no re-correr nada**:
apartar los 12 JPEG movió las métricas 0,001-0,003, y 40 está en el mismo orden. Lo publicado
sigue siendo válido tal como está, con la base declarada («incluye 40 archivos sin cifrar,
0,13 %»). Si en cambio se decide dejarlos, también es defendible: hay que declararlo igual.

**Lo que NO se recomienda:** tocar CERBER. Está cifrada; la ventana lee su cabecera en claro, y
eso se declara, no se «arregla» excluyendo la familia.

**Cifra citable para la tesis**, con su base: «verificación de integridad por magia de tipo
sobre los 29 676 archivos de las 30 familias: 27 familias sin anomalías; CERBER con cabecera
original preservada en el 100 % de sus 988 archivos (cifrado parcial); 40 archivos sin cifrar
(32 de NOTPETYA, 8 de JIGSAW; 0,13 % del corpus)».

---

---

# ✅ 40 archivos sin cifrar apartados + ⚠ SOSPECHA NUEVA: el filtro `.pdf` borra muestras válidas (2026-09-22)

## Cuarentena hecha

```
NOTPETYA-small apartados: 32 | quedan: 969
JIGSAW-small   apartados:  8 | quedan: 993
```

**32 y 8, exactamente los previstos.** Quedan en `NOTPETYA-small/_sin_cifrar/` y
`JIGSAW-small/_sin_cifrar/`, igual que los 12 JPEG de CERBER: exclusión **reversible**, los
cinco scripts del frente filtran con `is_file()` y no bajan a subdirectorios. **No se re-corre
nada**: apartar los 12 de CERBER movió 0,001-0,003 y 40 es del mismo orden. Lo publicado sigue
válido; la subsección de integridad declara la base.

## ⚠ Lo que delatan los «quedan», y que NO estaba visto

`quedan` cuenta **todos** los archivos, incluidos los `.pdf`; el censo del 17-09 contaba
**sin** `.pdf`. Restando:

| Familia | Total | Sin `.pdf` (censo 17-09) | ⇒ con extensión `.pdf` |
|---|---|---|---|
| NOTPETYA | 1001 | 833 | **168** |
| JIGSAW | 1001 | 998 | 3 |
| BADRABBIT | ~1001 | 857 | **~144** |

Los cinco scripts del frente de archivos filtran igual, con el mismo comentario:

```python
# excluir archivos que no son muestras cifradas (p. ej. el PDF descriptivo)
archivos = [p for p in archivos if p.suffix.lower() != ".pdf"]
```
(`clasificador_bytes.py:131` · `exp2d_nombre_extension.py:135` · `analisis_bytes.py:98` ·
`ablacion_ventana_extendida.py:98` · `deteccion_estructural.py:181`)

**La hipótesis, coherente con lo ya escrito en el capítulo:** el filtro se diseñó para sacar
**un** PDF descriptivo de NapierOne. Pero **NOTPETYA y BADRABBIT no cambian la extensión** —está
escrito en §Exp. 2b, citando a Davies et al.—, así que sus documentos PDF cifrados **siguen
llamándose `.pdf`** y el filtro **los borra a todos**. JIGSAW sí renombra, y por eso tiene 3 y
no 168.

**Si se confirma, el alcance es este, y hay que decirlo con cuidado:**
- No es que las cifras publicadas estén mal: están medidas sobre el conjunto que el script
  arma, y ese conjunto es reproducible. **Lo que está mal es la descripción implícita** de que
  se usan todos los archivos de cada familia.
- **NOTPETYA pierde ~17 % de su familia, y toda de un mismo tipo de documento.** Es la peor
  familia del Exp. 2c (F1 0,394 ± 0,023). Que le falte sistemáticamente un tipo entero es al
  menos un factor a considerar en su diagnóstico.
- Toca la validación **dejar-un-tipo-fuera** (§subsec:exp2c_tipos): el pliegue `pdf` es el de
  menos familias (**25**) y el de menor macro-F1 (**0,836**). La explicación escrita hoy es que
  las familias que renombran no tienen tipo asignable; **si además falta el `.pdf` de las que
  no renombran, la explicación está incompleta.**
- El muestreo `--por-familia 500` toma de 833 y no de 1001 en NOTPETYA: el sorteo sigue siendo
  válido, la población de la que sortea es otra.

**PENDIENTE DE VERIFICAR antes de escribir una sola línea de esto:** que esos `.pdf` sean
muestras **cifradas** (entropía de cabecera ~7,6) y no PDF descriptivos en claro. Es un comando.
Si resultan estar en claro, no hay problema y el filtro estaba bien.

---

---

# 🚨 CONFIRMADO: el filtro `.pdf` borraba 310 muestras cifradas (2026-09-22)

Verificado sobre las 30 carpetas, pegado por Romina. **La sospecha era correcta.**

| | `.pdf` | Qué son |
|---|---|---|
| 27 familias | **1 cada una** | `<FAMILIA>.pdf`, ~3,5 MB, cabecera `25 50 44 46 2d 31 2e 34` (`%PDF-1.4`), H(cab) **5,41**. Es la documentación de NapierOne. El filtro acertaba. |
| **BADRABBIT** | **144** | 1 documentación + **143 muestras CIFRADAS**: `0143-pdf.pdf`, H(cab) **7,57**, cabecera `12 73 c3 f8 c9 f6 28 cc` — sin magia PDF |
| **NOTPETYA** | **168** | 1 documentación + **167 muestras CIFRADAS**: `0167-pdf.pdf`, H(cab) **7,58**, cabecera `2e 53 43 83 df 9f 83 73` |
| JIGSAW | 3 | 1 documentación + 2 con cabecera `%PDF-1.3`: `0090-pdf.pdf` (H 3,80, **en claro**) y `0004-pdf.pdf` (H 7,50, cabecera PDF con cuerpo de alta entropía) |

**310 archivos cifrados descartados en silencio** (143 + 167), todos de las **dos únicas
familias que no cambian la extensión**. A las 28 que renombran no les quitaba nada: sus PDF
cifrados se llaman `.bed4`, `.encrypted`, etc. **Era un sesgo sistemático contra BADRABBIT y
NOTPETYA**, y NOTPETYA es la peor familia del Exp. 2c (F1 0,394 ± 0,023) a la que le faltaba
el **17 % de su familia, toda de un mismo tipo de documento**.

**Corregido en los cinco scripts** (commit pusheado): `es_documentacion(p, familia)` descarta
solo el archivo cuyo nombre base coincide con el de la familia. Probado con siete casos.

## ⚠ Dos cosas que quedaron colgando

1. **Los 2 `.pdf` de JIGSAW no se apartaron en la cuarentena del 22-09**, porque el comando de
   cuarentena filtraba `.pdf` igual que los scripts. `0090-pdf.pdf` está en claro (H 3,80) y
   debería ir a `_sin_cifrar/`. `0004-pdf.pdf` es dudoso: cabecera `%PDF-1.3` legible pero
   H 7,50 en los primeros 512 bytes; **no clasificar sin mirarlo**. Con el filtro corregido,
   el control de integridad los va a levantar solo en la próxima corrida.
2. **Hipótesis sobre NOTPETYA, NO medida:** los dos `.pdf` cifrados muestreados comparten los
   **mismos 8 primeros bytes** (`2e 53 43 83 df 9f 83 73`). Si eso es general, NOTPETYA cifra
   de forma **determinista sin vector de inicialización por archivo**, de modo que dos archivos
   con el mismo prefijo en claro dan el mismo prefijo cifrado. Eso explicaría por qué el
   Exp. 2b la reporta **sin marca alguna**: el Exp. 2b exige un prefijo común a **todos** los
   archivos de la familia, y aquí el prefijo cifrado depende del **tipo de documento**. Sería
   una corrección a una afirmación publicada («dos familias no presentan marca alguna»).
   **Verificar antes de escribir nada.**

## Qué cambia y qué no

- **No invalida ninguna cifra publicada.** Están medidas sobre el conjunto que armaba el
  script anterior, y ese conjunto es reproducible. Lo que era inexacto es la **descripción
  implícita** de usar todos los archivos de cada familia.
- **Habilita volver a medir sobre el corpus completo**, y ahí sí el 0,912 puede moverse: es la
  primera corrección que podría **subir** el número en vez de bajarlo, porque devuelve 167
  archivos a la familia más difícil. **Decisión de Romina y Cappo.**
- Toca la explicación del pliegue `pdf` en **dejar-un-tipo-fuera** (§subsec:exp2c_tipos), que
  es el de menos familias (25) y menor macro-F1 (0,836). La explicación escrita ---las
  familias que renombran no tienen tipo asignable--- **está incompleta**: además faltaban los
  PDF de las dos que no renombran.

---

### P2bal — EL REPARTO DE P2 ARREGLADO. Cifra de cabecera del frente de notas (2026-09-22)

`protocolo_p2bal.py --n-semillas 50`, **preregistro en el docstring, commiteado ANTES de correr**
(`a936397`, 2026-09-22 23:17:21) — precisamente lo que le faltó a LOGO. Log `_log_p2bal_149.txt`,
salidas en `resultados_protocolo_p2bal_149/`.

**Qué es.** `StratifiedGroupKFold` optimiza un balance global y **no garantiza que cada familia
tenga una plantilla en entrenamiento en cada pliegue**: deja 3,86 familias por pliegue con F1 = 0
forzado, ceros que entran al macro sin que el método haya fallado. P2bal reparte las plantillas
**dentro de cada familia** (las chicas eligen primero; empates sorteados con la semilla). Mismo
número de pliegues, **mismo tamaño de entrenamiento (49,5 plantillas por pliegue, idéntico)** y la
misma garantía, verificada: **0 violaciones** de plantilla en train y test a la vez, en los dos
protocolos.

| Protocolo | Capa | macro-F1 (30) | sd | IC 95 % | CV(F1) | sobre 28 evaluables | exact. | bal. | MCC | ¿>0,50? |
|---|---|---|---|---|---|---|---|---|---|---|
| P2 | texto | 0,4593 | 0,075 | [0,438; 0,481] | 0,164 | 0,4921 | 0,579 | 0,520 | 0,561 | NO |
| P2 | M.6 | 0,5191 | 0,079 | [0,497; 0,542] | 0,152 | 0,5562 | 0,660 | 0,578 | 0,645 | IC toca |
| **P2bal** | **texto** | **0,6551** | 0,034 | **[0,6454; 0,6648]** | 0,052 | **0,7019** | 0,719 | 0,703 | 0,709 | **SÍ** |
| **P2bal** | **M.6** | **0,7417** | 0,031 | **[0,7328; 0,7505]** | 0,042 | **0,7946** | 0,812 | 0,780 | 0,804 | **SÍ** |

Δ pareado: texto **+0,1959** [+0,174; +0,217], M.6 **+0,2225** [+0,199; +0,246], **50/50 semillas**.
Familias ≥ 0,50 con M.6: **25/30**; ≥ 0,70: 21/30. Bajo 0,50 quedan BADRABBIT y CRYPTOLOCKER
(plantilla única, cero estructural), HELLOKITTY 0,293, RYUK 0,358 y JIGSAW 0,499.

**Veredicto del preregistro: 6 de 7 predicciones se cumplen. La que falla es mía y es aritmética.**
Predije «2 familias sin entrenamiento por pliegue bajo P2bal» y da **1,00**. El valor correcto era
1,00 desde el principio: cada familia de plantilla única está ausente del train en **uno** de los
dos pliegues, no en los dos, así que 2 familias × ½ = 1,0 por pliegue. El protocolo se comporta
como debía; la predicción estaba mal calculada. **Dato que sale de ahí:** de las 3,86 familias sin
entrenamiento que pierde P2 por pliegue, **1,00 es inevitable** (plantilla única) y **2,86 son el
defecto del reparto**. Eso es lo que arregla P2bal, y es el 96 % del Δ.

Se cumplen: texto ≥ 0,60 con IC por encima de 0,50 (P1); tamaño de entrenamiento idéntico, o sea
**la ganancia no viene de entrenar con más** (P3); las 5 familias de 2 plantillas suben entre
+0,41 y +0,56 (P4); la sd cae de 0,075 a 0,034, se va la lotería de ceros (P5); P2bal < LOGO
0,6747, la diferencia por duplicar el entrenamiento es de solo +0,020 (P6); y la puerta de entrada
reproduce el canónico al cuarto decimal, 0,4593 / 0,5191 (P7).

**Reproducción independiente:** la revisión del 17/9 midió P2bal 0,6651 ± 0,032 a 20 semillas con
otra implementación; acá da 0,6551 ± 0,034 a 50. Diferencia 0,010, dentro de la variación por
semilla. Dos implementaciones separadas dan lo mismo.

**Cómo se cita (obligatorio).** «macro-F1 0,6551 con texto y 0,7417 con la cascada, sobre plantilla
no vista **según el criterio de casi-duplicado por coseno de caracteres 0,90**». Ese criterio **no
detecta contención**: con contención ≥ 0,8 el corpus pasa de 99 a 81 plantillas y de 28 a 21
familias evaluables, y la cifra baja (ver `REVISION_LOGO_2026-09-17_informe.md`). La limitación va
pegada al número, no en una nota al pie lejana.

**Decisión: P2bal pasa a ser el protocolo de cabecera del frente de notas.** P2 se reporta al lado
como lo que fue, y la diferencia entre los dos se explica como defecto de reparto, no como método.
LOGO no se escribe en la tesis (descartado el 17/9).


---

### ★★★ LAS DOS MEDICIONES QUE FALTABAN BAJO P2bal — CERRADAS (2026-09-26, local)

Las dos que pedía `HANDOFF_2026-09-26_cerrar_con_la_cascada.md` §3. **Preregistro commiteado
ANTES de correr**: `7debb3b`, 2026-09-26 19:38:50 (el código, no los resultados). Las dos tienen
**puerta de entrada** y las dos **reprodujeron exacto** la cifra de cabecera de P2bal
(cascada 0,8123 · texto 0,7191, diferencia 0,0000), así que no son una corrida distinta: son la
misma corrida mirada por dentro.

#### 1. Abstención (M.3) bajo P2bal — ya hay frase de despliegue citable

`abstencion_notas.py --protocolo P2bal --n-semillas 50`. Log `_log_m3_149_p2bal.txt`, salidas en
`resultados_abstencion_149_p2bal/`. **149 notas · 30 familias · 99 plantillas.**

| umbral | cobertura | **acierto donde contesta** | macro-F1 de las respondidas | por regla | por texto | se abstiene |
|---|---|---|---|---|---|---|
| 0,00 | 1,0000 | 0,8123 | 0,7417 | 80,3 | 68,7 | 0,0 |
| 0,20 | 0,8509 | 0,9003 | 0,8445 | 80,3 | 46,5 | 22,2 |
| **0,50** | **0,7718** | **0,9324** | 0,8725 | **80,3** | **34,7** | **34,0** |
| 1,00 | 0,6658 | 0,9864 | 0,9544 | 80,3 | 18,9 | 49,8 |

**La frase para la tesis, con su métrica pegada:** con umbral de margen 0,50 el sistema
**contesta el 77,18 % de las notas y, donde contesta, acierta el 93,24 %**; de cada 149 notas,
80,3 las resuelve la capa de reglas, 34,7 el texto y se abstiene en 34,0. Base: P2bal, 149 notas,
30 familias, 50 semillas, plantilla no vista **según coseno char 0,90**.

Bajo P2 la misma frase era 0,6459 / 0,9000. **Contesta más y acierta más**: las dos cosas a la
vez, que es lo que predecía A4.

**Acierto de cada capa por separado** (columna nueva del CSV): la capa de reglas acierta
**0,9928** y no depende del umbral —el umbral solo filtra al texto—; el texto solo acierta
0,6025 sin umbral y 0,7966 con umbral 0,50.

**VEREDICTO DEL PREREGISTRO: A1, A2, A4 y A5 cumplen; A6 FALLA.** A6 decía que la cobertura de
la capa de reglas no cambiaría entre P2 y P2bal (~0,4546). Midió **0,5389**, o sea +0,084. El
razonamiento preregistrado estaba incompleto: se miró solo el **tamaño** del entrenamiento (49,5
plantillas en los dos protocolos) y no la **cobertura de familias**. P2bal le garantiza a toda
familia con 2+ plantillas al menos una en entrenamiento, así que el diccionario de marcadores
cubre más familias y la regla encuentra coincidencia más seguido. Se reporta el fallo tal cual.

#### 2. Desglose por parecido con el entrenamiento, bajo P2bal

`similitud_vs_acierto_p2bal.py --n-semillas 50` (script nuevo, no toca ningún canónico). Log
`_log_similitud_p2bal_149.txt`, salidas en `resultados_similitud_p2bal_149/`. El parecido se mide
por **contención de 3-shingles de palabras** contra las notas de **su propia familia que estaban
en el entrenamiento de ese pliegue** —no contra el corpus entero, como se había hecho bajo LOGO.

| tramo de contención | notas (media) | acierto texto | **acierto cascada** | brecha | coseno medio |
|---|---|---|---|---|---|
| sin plantilla propia en entrenamiento | 4,0 | 0,0000 | 0,0000 | 0,0000 | — |
| [0,0; 0,3) | 61,7 | 0,5032 | 0,6337 | +0,1305 | 0,4358 |
| [0,3; 0,5) | 29,4 | 0,7747 | 0,9774 | +0,2027 | 0,6956 |
| [0,5; 0,7) | 17,9 | 0,9977 | 0,9682 | **−0,0295** | 0,8025 |
| [0,7; 0,9) | 30,0 | 1,0000 | 1,0000 | 0,0000 | 0,8109 |
| [0,9; 1,0] | 6,1 | 1,0000 | 1,0000 | 0,0000 | 0,8680 |

**Corte binario, que es lo citable:**
- **Con hermana contenida ≥ 0,5 en el entrenamiento** (54,0 notas, el 36,2 % del corpus): la
  cascada acierta **0,9891** [0,9848; 0,9934].
- **Sin hermana parecida** (91,0 notas, el 61,1 %): la cascada acierta **0,7434**
  [0,7294; 0,7574], frente a **0,5839** del texto solo.
- **Sin ninguna plantilla de su familia en el entrenamiento** (4,0 notas): **0,0000**, con texto
  y con cascada. Es el control de sanidad B6 y salió exactamente 0: si hubiera dado más, habría
  fuga. Son BADRABBIT y CRYPTOLOCKER, las dos de plantilla única.

**Lo que hay que decir junto al número, porque un jurado lo va a mirar:** el coseno medio con el
entrenamiento en el tramo de contención alta es **0,8143** — por debajo del umbral 0,90. O sea
que esas notas **son plantillas distintas según el criterio declarado** y aun así están
contenidas en algo que el sistema vio. No es que la partición esté mal; es que **el coseno no ve
la contención**, que es justo lo que midió la revisión del 2026-09-17.

**VEREDICTO: B1, B2, B3, B4 y B6 cumplen; B5 FALLA por poco.** B5 predecía que la fracción con
hermana ≥ 0,5 bajaría a ≤ 0,35 al entrenar con la mitad de las plantillas; midió **0,3621**
(bajo LOGO era 0,4228). Bajó, pero no tanto como lo preregistrado.

#### ⚠ HALLAZGO NO PREVISTO: en el tramo fácil, la capa de reglas RESTA

En el tramo [0,5; 0,7) la cascada acierta **menos** que el texto solo (0,9682 vs 0,9977, brecha
**−0,0295**), y en el corte binario ≥ 0,5 también (0,9891 vs 0,9993). Es coherente con que la
regla acierte 0,9928 y no 1,0000: **cuando el texto ya iba a acertar, el ~0,7 % de error de la
regla pisa una decisión correcta.** Por familia se ve en CONTI (texto 1,0000 → cascada 0,9000) y
AVOSLOCKER (1,0000 → 0,9400).

No cambia la decisión de presentar la cascada —el balance global es +0,0866 de macro-F1 y en el
tramo difícil la brecha es +0,1595 a favor de la cascada— pero **hay que escribirlo**: es
exactamente el tipo de detalle que un jurado encuentra y que, si no está declarado, parece
escondido. Y sugiere una variante medible: dejar que la regla ceda cuando el margen del texto es
muy alto. **No está medida y no se reporta como si lo estuviera.**

#### Dónde vive cada cosa

- `2_codigo/abstencion_notas.py` — preregistro A1–A6 en el docstring, puerta A1 que aborta.
- `2_codigo/similitud_vs_acierto_p2bal.py` — preregistro B1–B6 en el docstring, puerta B1.
- CSV: `resultados_abstencion_149_p2bal/m3_curva_abstencion.csv`,
  `resultados_similitud_p2bal_149/{similitud_tramos,similitud_binario,similitud_por_familia,similitud_por_nota}_p2bal.csv`.


---

### ★★ PEDIDO DE CAPPO DEL 2026-09-09, CONTESTADO CON NÚMERO (medido 2026-09-26)

Del chat del 9 de septiembre, tres pedidos. Los tres tienen respuesta medida.

#### 1. «Ver si las notas tienen algún patrón entre las que dicen que son de la misma clase. Si no hay ningún patrón, se puede deducir que el rendimiento del clasificador no será bueno»

`patron_vs_acierto.py`, log `_log_patron_vs_acierto.txt`, salidas en
`resultados_similitud_p2bal_149/patron_vs_acierto_*.csv`. **⚠ ANÁLISIS POST-HOC, NO
PREREGISTRADO** — se escribió después de ver los resultados y se cita como tal.

El «patrón interno» de una familia se mide por **contención media de 3-shingles de palabras**
entre sus notas y las notas de su propia familia que estaban en el entrenamiento. Alto = sus
notas se repiten entre sí; bajo = cada nota dice algo distinto.

**La relación que propuso el tutor existe y es fuerte:**

| | Pearson r | p | Spearman ρ | p |
|---|---|---|---|---|
| patrón vs acierto, **texto solo** | **+0,834** | < 0,00001 | +0,857 | < 0,00001 |
| patrón vs acierto, **cascada** | +0,715 | 0,00002 | +0,702 | 0,00003 |

Sobre las **28 familias evaluables** (las de 1 plantilla no tienen patrón interno definido).

| nivel de patrón | familias | acierto texto | acierto cascada | lo que rescata la regla |
|---|---|---|---|---|
| sin patrón (< 0,10) | 6 | 0,4063 | 0,5702 | **+0,1639** |
| intermedio [0,10; 0,40) | 4 | 0,5213 | 0,6934 | **+0,1720** |
| con patrón (≥ 0,40) | 18 | 0,9209 | 0,9555 | +0,0346 |

**EL MATIZ QUE IMPORTA: la correlación BAJA con la cascada (+0,834 → +0,715).** No es ruido: es
el efecto buscado. Las capas de reglas aflojan la dependencia del patrón textual, y por eso
existen. Donde no hay patrón, la regla aporta +0,1639; donde ya hay patrón, solo +0,0346.

**Los tres casos que contradicen la regla del tutor, y explican por qué:**
- **CHIMERA** — patrón 0,0186 (sus 2 notas no se parecen entre sí) y aun así **acierto 1,0000**:
  la capa de reglas la resuelve el 100 % de las veces por marcadores propios.
- **CLOP** — patrón 0,0235, texto 0,610 → cascada 0,740 (regla en el 65,5 %).
- **RANSOMEXX** — patrón 0,0611 y texto 0,748: le alcanza con vocabulario distintivo aunque no
  repita frases.

**Y las tres que le dan la razón — el límite honesto del corpus:** HELLOKITTY (patrón 0,0043 →
0,2267), RYUK (0,0572 → 0,2867), JIGSAW (0,0157 → 0,4200). Sin patrón de texto y sin marcadores
reutilizables, no hay de dónde aprender. **Son exactamente las familias que quedan bajo 0,50 de
F1 en la corrida de cabecera.** La predicción del tutor acierta en los casos donde falla el
sistema, que es donde importa.

#### 2. «Si tu clasificador da ≤ 50 % es como tirar una moneda»

El criterio es de una tarea **binaria**. Acá son **30 clases**: acertar al azar es **1/30 =
0,033**, no 0,50. Un macro-F1 de 0,5191 (la cifra que estaba vigente el 9 de septiembre, bajo el
reparto viejo P2) es **15,6 veces el azar**, no una moneda.

**La métrica que zanja la discusión sin discutirla es el MCC**, que vale 0 para un clasificador
al azar y 1 para uno perfecto, sea binario o multiclase. Bajo P2bal la cascada da **MCC 0,8042**.

Cifras vigentes de la cascada bajo P2bal, todas de `_log_p2bal_149.txt`:

| métrica | cascada | azar |
|---|---|---|
| MCC | **0,8042** | 0,000 |
| exactitud | 0,8123 | 0,033 |
| exactitud balanceada | 0,7798 | 0,033 |
| macro-F1 (30 familias) | 0,7417 [0,7328; 0,7505] | ~0,002 |
| macro-F1 (28 evaluables) | 0,7946 | — |

Además, el chat del 9-09 es **anterior a P2bal** (22-09): la cifra que motivó el comentario era
0,5191 y hoy es **0,7417**, y el sistema no cambió — se corrigió el reparto de la partición.

Y sobre «¿el clasificador debe poder decir no sé?»: ahora está medido. Con umbral de margen 0,50
**contesta el 77,18 % y acierta el 93,24 % donde contesta**.

#### 3. «Las familias con una sola plantilla no se procesan. Por eso necesitamos métricas»

**Instrucción del tutor, y el proyecto ya tiene la cifra separada:** macro-F1 sobre las **28
evaluables** = **0,7946** (cascada) y 0,7019 (texto), contra 0,7417 y 0,6551 sobre las 30.

Hay además justificación medida para excluirlas, del control de sanidad B6: las 4 notas de
BADRABBIT y CRYPTOLOCKER dan **acierto 0,0000 exacto** porque su familia nunca tiene material en
el entrenamiento de ningún pliegue. **No es que el clasificador falle: la tarea no está definida
para ellas.**

**DECISIÓN ABIERTA para Romina + Cappo:** si «no se procesa» es la regla, la cifra de cabecera
del frente de notas pasa a ser **macro-F1 0,7946 sobre 28 familias** en vez de 0,7417 sobre 30.
Las dos están medidas; es cuestión de cuál se declara como principal. **Lo que no se puede es
mezclarlas sin decir la base.**


---

---

### ★★ BASE B: EL PARECIDO CON LO CONOCIDO, Y EL FILTRO POR DOMINIO (2026-09-28)

`extendido_parecido_y_filtro.py`, **preregistro Q1–Q7 commiteado antes de correr**. Log
`_log_extendido_parecido.txt`. **Los seis preregistros medibles CUMPLEN.**

#### Parte 1 — qué pasa cuando la nota se parece a lo que ya conoce

| tramo | notas | **Base B (106 fam.)** | Base A (30 fam.) | caída |
|---|---|---|---|---|
| con hermana contenida ≥ 0,5 | 193,4 (32,5 %) | **0,9587** | 0,9891 | **0,0304** |
| sin hermana parecida | 397,6 (66,7 %) | **0,6624** | 0,7434 | **0,0810** |
| sin plantilla propia en entrenamiento | 5,0 | 0,0000 | 0,0000 | — |

**Q3 CUMPLE y es el resultado:** la caída al pasar de 30 a 106 familias **se concentra en el
tramo sin parecido** (0,0810 contra 0,0304). Ante una variante de algo ya visto, el sistema con
106 familias sigue acertando **0,9587**.

> **Escalar el catálogo no daña el reconocimiento de variantes conocidas; daña la generalización
> a notas nuevas.** Es la lectura que corresponde y es la esperable.

#### Parte 2 — el filtro por DOMINIO manteniendo la URL como clave

| variante | macro-F1 | exactitud | cobertura de la regla | **acierto de la regla** |
|---|---|---|---|---|
| filtro actual | 0,6485 | 0,7529 | 0,5119 | 0,9427 |
| **filtro por dominio** | **0,6543** | **0,7672** | 0,4964 | **0,9706** |

**Δ macro-F1 +0,0059** [+0,0026; +0,0092], 16/20 semillas. **Δ exactitud +0,0143.**

La clave sigue siendo la URL completa —conserva su especificidad— pero el **filtro mira el
dominio**, así que si el dominio cruza familias se descartan **todas** sus variantes a la vez.
Es lo que le faltaba a la normalización por dominio, que unía las claves además de filtrarlas y
por eso **empeoraba** la Base B (−0,0080).

**Q6 (efecto sobre el núcleo de 30) NO se midió**, porque la decisión ya está tomada: el núcleo
queda como está. Sin esa medición la variante **no es adoptable en la Base A**.

#### 🐛 BUG PROPIO EN LA PRIMERA CORRIDA, corregido

La primera versión guardaba solo **si** cada nota se había acertado, no **qué** se había predicho,
y calculaba el macro-F1 con `f1_score(y, where(acertó, y, "__mal__"))`. Al mandar todos los
errores a una clase inexistente, **ninguna familia recibe falsos positivos**, la precisión sale 1
por construcción y el F1 refleja solo el recall: daba **0,7342** donde la Base B vale 0,6485.

**Lo delató que el número no cuadrara con una cifra ya conocida.** Es el cuarto caso del día en
que un control de coherencia atrapa un error que ningún test declarado habría detectado, y se
agregó esa verificación a la puerta del script.

**No estaban afectados** —se calculan sobre aciertos y no necesitan la predicción— **la exactitud,
el acierto de la capa de reglas y toda la Parte 1.**

---

## 📌 LAS DOS BASES DEL FRENTE DE NOTAS — NO SE MEZCLAN (decisión de Romina, 2026-09-28)

A partir de acá conviven dos conjuntos, **con cifras propias que no son comparables entre sí**.
Al citar cualquier número hay que decir de cuál sale.

### BASE A — NÚCLEO CANÓNICO. Es la de la tesis.

**149 notas · 99 plantillas · 30 familias · P2bal · 50 semillas.** Es el núcleo que empareja los
dos frentes del trabajo, y **no se toca**.

| | cascada | texto solo |
|---|---|---|
| macro-F1 (30 familias) | **0,7417** [0,7328; 0,7505] | 0,6551 |
| macro-F1 (28 evaluables) | **0,7946** | 0,7019 |
| exactitud | 0,8123 | 0,7191 |
| exactitud balanceada | 0,7798 | 0,7033 |
| MCC | 0,8042 | 0,7092 |
| IC por remuestreo de plantillas | [0,6585; 0,8187] | [0,5654; 0,7446] |

**En uso:** con abstención responde el **77,18 %** y acierta el **93,24 %**; la familia correcta
está entre las tres primeras el **86,6 %**; ante una familia fuera del catálogo **se abstiene el
79,1 %** (al costo de 18,8 % en las conocidas).

**Por escenario:** 0,9891 ante una nota parecida a una ya vista (36 % del corpus) · **0,7434**
ante una nota genuinamente nueva (61 %) · 0,0000 en las 4 notas de familias de plantilla única.

⚠️ Estas cifras usan la **normalización de URL original**. La corrección medida el 28-09 daría
0,7492, **y NO está aplicada**: se reporta aparte.

### BASE B — CORPUS EXTENDIDO. Experimento aparte, para trabajo futuro.

**596 notas · 106 familias**, sumando las fuentes públicas ya reunidas. **No es comparable con la
Base A** y nunca se cita sin declarar su base.

| conjunto | familias | notas | macro-F1 | exactitud |
|---|---|---|---|---|
| global | 106 | 596 | **0,6485** | 0,7529 |
| restringido a las 30 originales | 30 | 257 | **0,7419** | 0,7669 |
| solo las familias nuevas | 76 | 339 | 0,6502 | 0,7423 |

**Lo que dice:** las 30 originales **mantienen su macro-F1** (0,7419 contra 0,7417) con 76
familias más compitiendo y 108 notas nuevas en las propias familias viejas. **Lo que baja en el
global es la tarea, no el método: el sistema escala.** 42 de las 76 nuevas superan F1 0,70.

**Limitaciones que van pegadas siempre:** etiquetas de las fuentes **sin auditoría de
procedencia**; **sin nombres de archivo auditados**, así que la segunda capa de la cascada no
actúa sobre las nuevas; normalización de nombres de familia laxa; y 21 familias con conflicto de
etiqueta que **el catálogo MISP identificó como parentesco real, no como error**.

**Y una advertencia propia de esta base:** la corrección de normalización de URL que mejora la
Base A (+0,0076) **empeora la Base B** (−0,0080). El criterio del filtro de genéricos no escala:
con muchas familias haría falta un umbral relativo, que **no está medido**.

---

### ★★★ LA PRIMERA MEJORA REAL: NORMALIZAR LAS URL DEL DICCIONARIO (2026-09-28)

`filtro_genericos_url.py`, **preregistro N1–N6 commiteado antes de correr**. Logs
`_log_filtro_url.txt` y `_log_filtro_url_nucleo50.txt`.

#### El defecto

El filtro de genéricos descarta un marcador cuando aparece **en más de una familia del pliegue de
entrenamiento**. Pero el mismo sitio entra al diccionario **partido en varias claves**. Sobre el
corpus de 106 familias, `torproject.org` aparece como **siete claves distintas**:

| clave | familias |
|---|---|
| `https://www.torproject.org/download/` | 46 |
| `https://www.torproject.org/` | 45 |
| `https://torproject.org` | 19 |
| `https://torproject.org/` | 19 |
| `https://www.torproject.org` | 14 |

Una variante poco frecuente puede quedar en **una sola familia** de ese pliegue, **pasar el filtro
y hacer que la regla conteste con seguridad equivocada**. Es el mismo mecanismo que en mundo
abierto hacía que la regla reclamara las notas de CONTI para BLACKBASTA.

#### El resultado, sobre el núcleo de 30 (50 semillas)

| variante | macro-F1 | exactitud | acierto de la regla |
|---|---|---|---|
| A — actual | 0,7417 | 0,8123 | 0,9928 |
| **B — normalización suave** | **0,7492** | **0,8183** | **0,9977** |
| C — por dominio | 0,7480 | 0,8168 | 0,9928 |

**Δ macro-F1 de B: +0,0076** [+0,0049; +0,0102].

**Distribución del Δ por semilla, verificada aparte: 29 semillas con Δ exactamente 0, 21
positivas con media +0,0180, y CERO negativas. Nunca empeora.**

«Suave» es: minúsculas, sin esquema, sin `www.`, sin barra final. No hay parámetro que ajustar.

#### Por qué esta sí se adopta y la cesión por margen no

| | cesión por margen | normalización de URL |
|---|---|---|
| Δ | +0,0015 | **+0,0076** |
| parámetro a elegir | sí, umbral, y sobre el mismo conjunto | **ninguno** |
| ventana de funcionamiento | estrecha (0,75 pierde, 1,50 no hace nada) | no aplica |
| ¿empeora alguna vez? | no, pero el efecto es despreciable | **no, y el efecto es 5× mayor** |
| naturaleza | complica el método | **corrige un defecto** |

Que `https://www.torproject.org/` y `https://torproject.org` sean **la misma clave** no es una
decisión de diseño: es lo correcto. El sistema las trataba como valores distintos por un descuido
de normalización.

#### ⚠️ Y el hallazgo que va con él: en 106 familias, la MISMA corrección EMPEORA

| corpus | Δ macro-F1 de B | acierto de la regla |
|---|---|---|
| núcleo 30 | **+0,0076** | 0,9928 → **0,9977** |
| extendido 106 | **−0,0080** | 0,9427 → 0,9349 |

**N2 y N3 FALLAN**: se había predicho lo contrario. La explicación: con 30 familias la clave
unificada aparece en varias y **siempre se filtra**, así que normalizar solo quita ruido. Con 106
hay pliegues donde **sobrevive**, y entonces una sola clave genérica captura de golpe todas las
notas que antes se repartían entre siete — la cobertura sube (0,5119 → 0,5385) y el acierto baja.

**Consecuencia para cualquier extensión futura:** el criterio «aparece en más de una familia del
entrenamiento» **no escala**. Con muchas familias hace falta un umbral relativo, del tipo
«descartar si aparece en más del X % de las familias del pliegue». **No está medido.**

#### ✅ DECISIÓN TOMADA (Romina, 2026-09-28): NO se adopta en el canónico

**El canónico de 30 familias queda como está: macro-F1 0,7417 con la normalización original.**
`normalizacion_marcadores.py` **no se toca** y no se re-mide nada.

La corrección se reporta **aparte**, como mejora verificada y no aplicada: «se identificó un
defecto en la normalización de las URL del diccionario; corregirlo eleva el macro-F1 a 0,7492
[Δ +0,0076, IC +0,0049 a +0,0102, sin una sola semilla negativa en 50]». Es material para
trabajo futuro y para la discusión, no para las cifras del capítulo.

---

### ★★ LA EXTENSIÓN CON EL CATÁLOGO MISP DEL TUTOR — LA VÍA QUEDA CERRADA (2026-09-28)

`capa_extension_misp.py`, **preregistro M1–M6 commiteado antes de correr**. Log
`_log_capa_extension_misp.txt`.

**Qué se probó.** La capa de extensión había dado Δ = 0 porque la regla exige haber visto el
valor en otra plantilla del entrenamiento (techo duro 6/149). Se sustituyó ese diccionario
aprendido por el **catálogo MISP «Ransomware»** que envió el tutor el 2026-08-20: **735
extensiones distintas, 673 asociadas a una sola familia**.

**Resultado: cobertura 0,0000.** El catálogo no resuelve **ninguna** de las 149 notas. Δ = +0,0000.
**M2, M3 y M4 FALLAN.**

#### Por qué, verificado extensión por extensión

| extensión del corpus | ¿está en el MISP? |
|---|---|
| `.gacmw` · `.rfncw` · `.ibkfz` · `.lgzcfcr` · `.eebf08` | **NO** — son **aleatorias por víctima** |
| `.gdcb` · `.krab` · `.sz40` | **NO** — son fijas, pero **el catálogo no las tiene** |

El MISP registra para GandCrab solo `.Crab` y `.CRAB`; **le falta `.GDCB`, que es su extensión
conocida de la v1**.

**Dos causas independientes, y las dos cierran la vía:**

1. **La mayoría de las extensiones que las notas mencionan son específicas de la víctima.**
   `.gacmw`, `.ibkfz`, `.rfncw` son cadenas aleatorias generadas por campaña. **Ningún catálogo
   puede registrarlas, por diseño del ransomware.**
2. **El catálogo está incompleto incluso para las extensiones fijas.**

#### Consecuencia

La vía de la extensión de cifrado queda **cerrada por tres barreras medidas**, no por una:
las fuentes publican las notas saneadas (solo 12 de 149 la conservan), el corte por plantilla
exige ver el valor dos veces (techo 6/149, oráculo +0,0038), y **ahora: aun con un catálogo
externo de 735 extensiones, la cobertura es cero porque las extensiones son aleatorias**.

Es además un dato sobre la comparación con **ID Ransomware**, que se apoya en extensión y nombre
de nota: las extensiones aleatorias por víctima son un límite del enfoque, no de esta
implementación.

#### Los alias del MISP: solo dos fusiones confirmadas

Se cruzaron los 24 conflictos de etiqueta del corpus extendido contra los sinónimos del catálogo.
**El MISP confirma como misma familia solo dos pares: `alphv` = `blackcat` y `revil` =
`sodinokibi`.** Para el resto dice que son **familias distintas** o no tiene entrada.

**Eso corrige la lectura anterior:** los «conflictos de etiqueta» del corpus extendido **no son
mayormente errores de etiquetado, son parentesco real** — familias distintas que comparten el
molde de la nota, igual que CLOP–RYUK. Fusionarlas sería incorrecto.

---

### ★★★ EL MISMO SISTEMA SOBRE 106 FAMILIAS — ESCALA (2026-09-28)

`extension_familias_corpus.py --n-semillas 20`, **preregistro X1–X6 commiteado antes de correr**.
Log `_log_extension_familias.txt`, salidas en `resultados_extension_familias_149/`.
**Prueba exploratoria:** no busca mejorar, busca ver cómo se comporta el sistema al generalizarse.

| conjunto | familias | notas | macro-F1 | exactitud |
|---|---|---|---|---|
| núcleo solo | 30 | 149 | 0,7428 | 0,8161 |
| **extendido global** | **106** | 596 | **0,6485** | 0,7529 |
| **extendido, restringido a las 30 originales** | 30 | **257** | **0,7419** | 0,7669 |
| extendido, solo las familias nuevas | 76 | 339 | 0,6502 | 0,7423 |

#### El resultado: el sistema ESCALA

**Las 30 familias originales mantienen su macro-F1: 0,7419 contra 0,7417. Caída de −0,0002.**

Y es más fuerte de lo que parece, porque el conjunto «30 originales» del corpus extendido **no es
el mismo que el núcleo**: tiene **257 notas en vez de 149**, o sea **108 notas nuevas de esas
mismas familias traídas de otras fuentes** (DHARMA pasa de 19 a 38, CERBER de 18 a 37). Así que
el sistema sostiene su rendimiento aunque a la vez:

1. se le agregan 108 notas heterogéneas a las familias que ya tenía, y
2. compite con 76 familias más.

**Lo que baja en el global (0,7417 → 0,6485) es la TAREA, no el método.** Es el efecto trivial de
pasar de 30 a 106 clases, y por eso la cifra global no se cita sola.

**42 de las 76 familias nuevas superan F1 0,70**, varias por encima de 0,93 (risen 0,9944,
cryptowire 0,9900, ragnarlocker 0,9786, proxima 0,9364 con 33 plantillas). **10 de 76 quedan en
F1 = 0.**

#### La predicción que FALLA, y es la informativa

**X4: el acierto de la capa de reglas baja de 0,9928 a 0,9427.** La cobertura casi no se mueve
(0,5119 contra 0,5389, X5 cumple). **La causa no está medida** y no se afirma: la hipótesis es
que con 106 familias aparecen colisiones de marcadores que el filtro de genéricos no atrapa,
coherente con el hallazgo de mundo abierto sobre `torproject`. **Queda pendiente de explicar.**

#### 🚨 HALLAZGO DE DISEÑO: el criterio de «plantilla» NO es estable ante la ampliación

La primera corrida **abortó en la puerta X1** (el núcleo daba 0,7948 en vez de 0,7417). Al
diagnosticar aparecieron dos causas, y la segunda importa más allá de esta prueba:

`TfidfVectorizer` **ajusta el IDF sobre el corpus que recibe**, así que agregar 690 notas cambia
los pesos y con ellos los cosenos. Verificado sobre GANDCRAB: tres pares cruzan el umbral 0,90 al
ampliar —0,9116 → 0,8960 · 0,9098 → 0,8950 · 0,9036 → 0,8885— y la familia pasa de 4 a 5
plantillas; WASTEDLOCKER hace el camino inverso, de 3 a 2.

**Consecuencia: ampliar el corpus redefine qué es una plantilla, y con ello la partición y todas
las cifras.** Cualquier extensión futura tiene que preservar explícitamente la estructura del
núcleo, como se hizo acá: el núcleo se agrupa solo, y una nota de fuente que es casi-copia de una
canónica **hereda su grupo** para que no pueda caer en otro pliegue y abrir una fuga.

#### Limitaciones, que hacen la cifra NO comparable con la del núcleo

Etiquetas de las fuentes **sin auditoría de procedencia**; **15 familias nuevas están en
conflicto de etiqueta** (grupos que cruzan familias); las familias nuevas **no tienen nombre de
archivo auditado**, así que la segunda capa de la cascada no puede actuar sobre ellas;
normalización de nombres laxa; y el criterio de plantilla sigue siendo coseno 0,90.

---

### ★★★ MUNDO ABIERTO: QUÉ HACE EL SISTEMA ANTE UNA FAMILIA QUE NUNCA VIO (2026-09-28)

`mundo_abierto_familia_fuera.py --n-semillas 20`, **preregistro O1–O6 commiteado antes de
correr** (`da0b123`). Log `_log_mundo_abierto_149.txt`, salidas en
`resultados_mundo_abierto_149/`.

**Qué convierte en medición.** Hasta hoy el escenario «familia fuera del catálogo» tenía **un
solo caso real** (una campaña de pocos días que ninguna herramienta pudo nombrar). Ahora es
sistemático: **cada familia sale entera del entrenamiento por turno** y se mide si el sistema se
abstiene ante sus notas o se las asigna a otra con confianza.

**Diseño simétrico:** por cada familia que sale se retiene una plantilla de cada una de las otras
29, y el **mismo modelo** evalúa los dos conjuntos. Sin eso, la diferencia de abstención sería un
artefacto del tamaño de entrenamiento.

| umbral | **abstención ante DESCONOCIDA** | abstención ante conocida | separación | acierto en lo que contesta |
|---|---|---|---|---|
| 0,10 | 0,4215 | 0,0856 | +0,336 | 0,8734 |
| 0,30 | 0,6745 | 0,1538 | +0,521 | 0,9011 |
| **0,50** | **0,7909** | **0,1884** | **+0,603** | **0,9156** |
| 0,75 | 0,8409 | 0,2348 | +0,606 | 0,9403 |
| 1,00 | 0,8755 | 0,3063 | +0,569 | 0,9910 |

**LOS CINCO PREREGISTROS CUMPLEN.**

**Frase citable, con las dos cifras juntas siempre:** ante una familia fuera del catálogo el
sistema **se abstiene en el 79,1 %** de los casos (umbral 0,50), **al costo de abstenerse también
en el 18,8 %** de las notas de familia conocida, donde acierta 0,9156 sobre lo que sí contesta.

- **O1** — acierto sobre familia desconocida **0,000000 exacto**. Control de fuga: no puede
  acertar una clase que no tiene.
- **O4** — la capa de reglas aplica **0,0956** ante desconocidas contra **0,5138** ante
  conocidas. Los marcadores de una familia nueva no están en el diccionario, como se esperaba.
- **O5 CUMPLE, y es el más informativo** — las familias con **pariente de linaje** en el
  entrenamiento se rechazan **mucho menos**: **0,5097 contra 0,8421**. El hallazgo del linaje
  explica también el comportamiento en mundo abierto: el texto asigna la familia nueva **a su
  pariente** en vez de dudar.

#### 🚨 HALLAZGO NO PREVISTO: el filtro de genéricos depende del catálogo

CONTI es la familia que **menos se rechaza**: se abstiene solo **0,0875**, y la capa de reglas la
reclama en el **75 %** de los casos. Verificado nota por nota:

```
CONTI\conti1.txt  ->  la regla dice BLACKBASTA
   clave: ('[URL]', 'https://torproject.org')  ->  ['BLACKBASTA']
```

**La causa.** `https://torproject.org` está en BLACKBASTA **y** en CONTI. Con las 30 familias en
el catálogo, el filtro de genéricos lo descarta por aparecer en dos. **Con CONTI fuera, ese valor
queda como “privado” de BLACKBASTA** y la regla reclama sus notas **con confianza absoluta**.

Es lo peor que puede hacer un sistema en mundo abierto, porque **la capa de reglas no pasa por el
umbral de abstención**: contesta siempre.

**Y hay un defecto concreto detrás, más fino:** la normalización **no unifica la barra final**.

| clave | familias en que aparece |
|---|---|
| `https://torproject.org` | BLACKBASTA · CONTI |
| `https://torproject.org/` | BLACKCAT · DARKSIDE · LORENZ · NETWALKER · SODINOKIBI |

Son **el mismo valor** partido en dos claves. Unificadas estarían en **7 familias** y el filtro
las descartaría siempre, con catálogo completo o no.

**⚠️ IMPACTO SOBRE LAS CIFRAS ACTUALES: NINGUNO.** En mundo cerrado —todas las cifras de la
tesis— CONTI está en el catálogo y el valor ya se filtra. El problema aparece **solo** cuando
falta una familia. No hay que re-medir nada.

**Arreglo propuesto, NO aplicado (decisión de Romina):** una lista negra de valores genéricos del
ecosistema, fijada a priori e independiente del catálogo —empezando por los dominios de
torproject— más normalizar la barra final de las URL. Tocar `normalizacion_marcadores.py` es
tocar código canónico y no se hace sin decisión explícita.

#### Lo que este experimento le agrega a la tesis

Completa el cuadro de los tres escenarios con medición, no con anécdota:

| escenario | resultado |
|---|---|
| nota parecida a una ya vista (36 %) | **0,9891** |
| nota nueva de familia conocida (61 %) | **0,7434** |
| **familia fuera del catálogo** | **se abstiene en el 79,1 %** (al costo de 18,8 % en conocidas) |

Y responde de forma cuantitativa la limitación de mundo cerrado que el capítulo ya declaraba:
**el mecanismo de abstención funciona como detector de novedad**, con la salvedad medida de que
**falla justo donde hay un pariente de linaje en el catálogo**.

---

### ★★★ SÍNTESIS: CUATRO TÉCNICAS NUEVAS, CUATRO NEGATIVOS, CUATRO CAUSAS DISTINTAS (2026-09-28)

Se evaluaron en paralelo cuatro vías para mejorar el acierto del frente de notas, todas con
**preregistro commiteado antes de correr** y **puerta de entrada** que reprodujo 0,7417 / 0,8123.
**Ninguna aporta.** Lo valioso es que **fallan por razones distintas**, y juntas cierran el
argumento de dónde está el límite.

| técnica | Δ sobre la cascada | por qué no aporta |
|---|---|---|
| **Capa de contención** | **+0,0000** exacto, 0/50 | la señal **ya está explotada**: el TF-IDF la agota |
| **Extensión de cifrado** | **+0,0000** exacto, 0/50 | la señal **casi no está en el corpus**: 12 notas de 149 |
| **Ensemble de vistas** | −0,0015 a −0,0000 | la **concatenación ya era razonable** |
| **Forma del nombre** | ninguna aporta | la señal está **contaminada por la curaduría** |
| *(jerarquía por linaje, mismo día)* | −0,0011 | las clases **no son separables** por texto |

#### 1. Capa de contención — el cero más limpio del proyecto

**4.163 decisiones de la capa, cero en las que difiera del LinearSVC.** Contención alta significa
que el vocabulario de la nota es casi un subconjunto del de una de entrenamiento, y TF-IDF con
clasificador lineal **ya es un emparejador léxico**. Coinciden incluso en los errores.
Verificación independiente de esta sesión: la capa **aislada** sí puede contradecir al texto (1
caso en 342, `LORENZ/pcrisk_lorenz_1.txt` → SODINOKIBI), pero en su posición dentro de la cascada
esa nota ya la resuelve una regla. **Adelantar la capa sería dañino**: cambiaría un decisor de
0,9928 por uno de 0,8159 sobre las mismas notas.

#### 2. Extensión de cifrado — el techo oráculo es el resultado

Patrón limpio: **8 extensiones en 12 notas, cero basura** (barrer todo `.token` daba 135 tokens
con ~97 % de basura). Acierto **1,0000** donde aplica. Pero:

- **Las fuentes publican las notas saneadas** — `[snip]`, `${EXTENSION}` en BLACKCAT, `{EXT}` en
  SODINOKIBI. Solo 12 de 149 conservan la extensión real.
- Como el corte es **por plantilla**, la regla solo dispara si el mismo valor está en otra
  plantilla de entrenamiento: pasa solo con `.gacmw` y `.gdcb`. **Techo duro 6/149 = 0,0403**
  (verificado en esta sesión).
- **Techo oráculo**, regalándole la respuesta en las 12 notas: macro-F1 0,7417 → **0,7454**,
  Δ **+0,0038**. Ése es el máximo alcanzable con conocimiento perfecto.

**Ventaja de diseño que conviene declarar:** extrae la extensión del **texto** de la nota, no del
nombre del archivo, así que es inmune al artefacto de curaduría que afectó al frente de archivos.

#### 3. Ensemble de vistas — y un resultado metodológico transferible

Seis variantes (suma, voto, ponderado; con 2 y 3 vistas). En la cascada el peor |Δ| es 0,0015 y
**ninguna excluye el cero**, ni con IC 95 % ni con Bonferroni. En la capa de texto sola, **cinco
de las seis pierden** con IC que excluye el cero.

**Hallazgo metodológico:** el peso óptimo elegido por validación interna tiene un reparto **casi
plano** sobre los 100 pliegues — no es un peso elegido, es el promedio de un sorteo. La causa
está medida: la validación interna da macro-F1 **0,3419** contra 0,6551 del externo, porque
entrena con ~37 notas y **en ese régimen el criterio no discrimina**. Por eso el ponderado
(−0,0124) sale **peor que los pesos iguales** (−0,0083). **Con corpus de este tamaño, elegir
hiperparámetros internos de forma honesta agrega ruido, no señal.**

**Hallazgo lateral:** la vista `caracteres` sola pierde claro contra `combinado` en la capa de
texto (−0,0139, IC excluye el cero) pero **en la cascada la diferencia se evapora** (+0,0012, IC
cruza el cero). **La ventaja de concatenar existe solo donde las reglas de IOC no llegan.**

**Nota para D.1:** esto es *majority voting* **dentro** del frente de notas, entre vistas TF-IDF.
**NO** es el *majority voting entre los dos frentes* que pidió el tutor, que sigue sin poder
evaluarse por falta de muestras pareadas. Pero es un dato para esa decisión: en este corpus,
votar decisiones no agregó nada.

#### 4. Forma del nombre — falló la transferencia, y se encontró la misma fuga

Siete abstracciones del nombre. **Ninguna aporta.** La única con Δ global positivo (esqueleto
tipográfico, +0,0042) **no es una mejora**: su Δ de exactitud incluye el cero y su Δ **restringido
a las 64 notas con nombre es −0,0038** — negativo justo donde la capa puede actuar. El +0,0042 es
el macro-F1 reaccionando a una familia chica (HELLOKITTY, 3 notas).

**🚨 FUGA ENCONTRADA EN NUESTRO CORPUS, verificada en esta sesión.** La auditoría enmascaró el ID
de la víctima **con dos notaciones distintas**:

| notación | nombres | caracteres |
|---|---|---|
| `[]` | **21** | 2 |
| `[victim's_id]` | **1** (DARKSIDE) | 13 |

**El largo del nombre y su cantidad de tokens dependen del auditor, no del ransomware.** Es el
mismo problema del `-fromweb` de NapierOne en el frente de archivos, a menor escala. Dos hechos
lo confirman: la única abstracción con Δ positivo es del grupo contaminado, y al quitarle a la
firma los dos campos afectados **empeora** (−0,0069 contra +0,0003) — o sea, lo poco que aportaba
**vivía en los campos contaminados**.

**Interpretación de por qué la transferencia falla** (es interpretación, no medición): en
NapierOne **cada familia es una sola campaña**, de modo que la forma del nombre *es* la campaña;
en el corpus de notas **cada familia trae notas de varias campañas que renombran distinto**. Eso
explica a la vez por qué funcionó en archivos y por qué no acá.

**⚠️ DECISIÓN ABIERTA PARA ROMINA:** si conviene re-auditar los 22 nombres con ID enmascarado para
uniformar la notación. **No se tocó nada.** El impacto sobre las cifras actuales es nulo —la capa
no se adopta— pero afecta a cualquier análisis futuro del nombre.

#### Lo que esto le da a la tesis

Con estas cuatro se llega a **nueve resultados negativos convergentes** en el frente de notas:
hiperparámetros (840 configuraciones), embeddings multilingües, metadato, estilometría, cesión
por margen, jerarquía por linaje, contención, extensión de cifrado y ensemble de vistas.

Atacan lugares distintos —hiperparámetro, representación, estructura de clases, señal nueva,
señal exacta, agregación de decisiones— y **los nueve dan nulo**. Ya no es un argumento por
agotamiento: es un argumento por **convergencia**, y cada uno trae su propia explicación de por
qué no había nada que ganar ahí.

**El límite del frente de notas no está en el método. Está en el corpus**, y las cuatro causas
identificadas hoy dicen exactamente en qué: señal ya explotada, señal ausente, señal contaminada
por la curaduría y clases genuinamente no separables.

---

### ★★★ CAPA DE CONTENCIÓN — NO APORTA NADA, Y EL CERO ES EXACTO (2026-09-28)

`capa_contencion.py`, **preregistro K1–K7 commiteado antes de correr** (`c7c7228`). Log
`_log_capa_contencion.txt`, salidas en `resultados_capa_contencion_149/`.

**Qué se probó.** Una capa nueva en la cascada, entre el nombre y el texto: si la nota de prueba
está **contenida** (3-shingles de palabras) en alguna nota de entrenamiento por encima de un
umbral, se le asigna esa familia. La motivación era el hallazgo del 26-09: el agrupamiento usa
coseno y **no ve la contención**, de modo que parecía una señal sin explotar.

| umbral | cobertura | acierto donde aplica | **Δ macro-F1** | semillas + |
|---|---|---|---|---|
| 0,5 | 0,1672 (24,9 notas) | 0,8131 | **+0,0000** [+0,0000; +0,0000] | 0/50 |
| 0,6 | 0,1467 | 0,8074 | +0,0000 | 0/50 |
| 0,7 | 0,1177 | 0,7640 | +0,0000 | 0/50 |
| 0,8 | 0,0944 | 0,7142 | +0,0000 | 0/50 |
| 0,9 | 0,0328 | 0,8613 | +0,0000 | 0/50 |

#### El hallazgo: la capa nunca contradice al texto

**4.163 decisiones de la capa, cero en las que su respuesta difiera del LinearSVC.** No es que
aporte poco: aporta **exactamente nada**, y no cambia ni la exactitud balanceada ni el MCC.

La explicación es que **la señal ya estaba explotada**. Contención alta por 3-shingles significa
que el vocabulario de la nota de prueba es casi un subconjunto del de una nota de entrenamiento,
y TF-IDF con un clasificador lineal **ya es, en el fondo, un emparejador léxico**. La capa repite
lo que el texto hace.

Importante: esa coincidencia **incluye los errores**. Donde la capa aplica, los dos aciertan solo
0,71–0,86, y **se equivocan en las mismas notas y con la misma familia equivocada**.

#### Verificación independiente (esta sesión, no el subagente)

Se reimplementó la capa desde cero sobre 5 semillas. Resultado: **1 diferencia en 342
decisiones**, no cero — y el caso es `LORENZ/pcrisk_lorenz_1.txt`, donde la capa dice SODINOKIBI
y el texto acierta LORENZ. **Es el par LORENZ–SODINOKIBI**, que comparte 108 n-gramas exclusivos.

La discrepancia se explicó midiendo: esa nota cae en prueba 5 veces de 5 y **las 5 la resuelve
una regla antes de llegar a la capa de contención**. Las dos mediciones son correctas y miden
cosas distintas: el subagente midió la capa **en su posición dentro de la cascada** y la
verificación midió la capa **aislada**.

**El matiz importa:** la capa aislada **sí puede contradecir al texto**, y cuando lo hace **se
equivoca, por parentesco entre familias**. Refuerza con un caso concreto el corolario de abajo.

#### Por qué además no conviene adelantarla

Cobertura **bruta** (ignorando la posición en la cascada) en u=0,5: 0,4157 con acierto 0,8159.
Pero **el 59,6 % de esas notas ya las resolvía la capa de reglas**, que acierta 0,9928 sobre
ellas. **Adelantar la contención delante de las reglas cambiaría un decisor de 0,9928 por uno de
0,8159: sería dañino.**

#### Veredicto del preregistro

- **K1** puerta de entrada — CUMPLE (0,7417 / 0,8123 exacto).
- **K2** cobertura 0,10–0,30 y monótona — CUMPLE.
- **K3** acierto donde aplica ≥ 0,90 — **FALLA** (0,714–0,861, ninguno llega, y no es monótono).
  Causa identificada: se subestimó la **contención CRUZADA entre familias**; los pares de linaje
  comparten el molde y la regla de conflicto no los atrapa cuando el pliegue dejó una sola de las
  dos familias en entrenamiento.
- **K4** (la principal) — CUMPLE, en su versión más extrema: no «cerca de cero», **cero exacto**.
- **K5** el texto acierta ≥ 0,95 en esas notas — **FALLA** (0,714–0,861). La predicción se había
  anclado en el 0,9993 del corte por contención contra la **propia familia**; esta capa mira
  contra **cualquier** familia, que es una zona más difícil.
- **K6** ≥ 50 % de la cobertura bruta ya resuelta por reglas — CUMPLE (0,5961).
- **K7** control de no-circularidad — CUMPLE exacto (Δ restringido 0,000000, sin redondeo).

#### Consecuencia para la tesis

**Resultado negativo citable, y de los buenos:** la contención literal —la limitación conocida
del criterio de plantilla por coseno 0,90— **no es una señal sin explotar**. El TF-IDF ya la
agota. Esto **refuerza** el argumento de circularidad del informe del 28-09 en vez de
debilitarlo: el acierto alto en las notas contenidas no es un premio que el sistema se lleve por
una vía aparte, es el mismo emparejamiento léxico de siempre.

**Honestidad de procedimiento:** la columna `coincide_con_el_texto` no estaba en el preregistro;
se agregó después de que una corrida de humo diera Δ = 0 exacto, para explicar la causa. Está
declarada como posterior en el docstring y en el commit `5dc8a87`; ninguna de K1–K7 se tocó.

---

### ★★★ CASCADA JERÁRQUICA POR LINAJE — NO MEJORA, Y EL PORQUÉ ES EL RESULTADO (2026-09-28)

`cascada_jerarquica_linaje.py --n-semillas 50`, **preregistro H1–H6 commiteado antes de correr**
(`64c6839`). Log `_log_jerarquica_linaje_149.txt`, salidas en `resultados_jerarquica_linaje_149/`.

**Qué se probó.** Convertir el diagnóstico del linaje en mejora real: etapa 1 decide el grupo
(cascada entrenada con 27 clases fusionadas) y etapa 2, si cayó en un par, un **clasificador
binario entrenado solo con las notas de ese par** decide cuál de las dos. La hipótesis: si la
confusión viene de la interferencia de las otras 28 clases, un especialista la resuelve.

| sistema | macro-F1 | exactitud | bal. | MCC |
|---|---|---|---|---|
| cascada plana (30 clases) | **0,7417** | **0,8123** | 0,7798 | 0,8042 |
| cascada jerárquica (27 → binario) | 0,7415 | 0,8113 | 0,7782 | 0,8030 |

**Δ exactitud −0,0011 [−0,0035; +0,0013]**, 11/50 semillas. El intervalo incluye el cero:
**no hay diferencia**. **H4 FALLA.**

#### El desglose, que dice mucho más que el promedio

| par | decisiones | **acierto del binario dedicado** |
|---|---|---|
| BLACKBASTA–CONTI | 426 | **0,9225** |
| DHARMA–PHOBOS | 1127 | **0,8660** |
| **CLOP–RYUK** | 393 | **0,5878** |

**Dos de los tres pares SÍ se separan bien** con un clasificador dedicado. No mejoran el total
porque **la cascada plana ya los resolvía**: sus marcadores son privados y la capa de reglas los
desambigua sin ayuda. Por familia, el movimiento es mínimo (BLACKBASTA +0,0400, CONTI 0,0000,
DHARMA 0,0000, PHOBOS −0,0150).

**El caso CLOP–RYUK es el resultado de todo el experimento.** Un clasificador **dedicado
exclusivamente a separar esas dos familias, sin ninguna otra clase interfiriendo**, acierta
**0,5878** — contra 0,5000 de una moneda. **La información para separarlas no está en el texto.**

> **Es la demostración más directa que tiene el trabajo de que el techo, en ese caso, es del
> CORPUS y no del método.** No es que el clasificador de 30 clases se confunda por
> sobrecarga: es que las notas no contienen la señal.

Y pesa: las confusiones CLOP↔RYUK son **158 de los 1398 errores de la cascada, el 11,3 % del
error total**, y con un especialista dedicado seguirían fallando cerca del 41 % de esas
decisiones.

#### Veredicto del preregistro

- **H1 CUMPLE** (puerta: 0,7417 / 0,8123).
- **H2 FALLA por poco**: la etapa 1 entrenada con 27 clases da **0,8577**, contra **0,8585** de
  fusionar a posteriori las predicciones del clasificador de 30. **Entrenar con las etiquetas
  fusionadas no aporta nada**: el clasificador de 30 clases no está gastando capacidad en
  separar lo inseparable, como se había supuesto.
- **H3 FALLA**, pero por un solo par: 0,922 y 0,866 cumplen, CLOP–RYUK con 0,588 no.
- **H4 FALLA**: no mejora.
- **H6 CUMPLE**: CLOP–RYUK es el peor par, como se predijo.

⚠️ **La lectura automática que imprime el script es imprecisa** y hay que corregirla al citar:
dice «los binarios tampoco aciertan», pero **dos de los tres sí aciertan**. El desenlace real es
una mezcla de dos de los tres casos previstos en H5 — para dos pares, la plana ya los resolvía;
para el tercero, la señal no existe.

#### Consecuencia para la tesis

**La vía de mejora por reorganización del clasificador está agotada**, y ahora con evidencia
directa y no por descarte. Se suma a los otros resultados negativos convergentes
(hiperparámetros, abstracción de marcadores, embeddings, metadato, estilometría, cesión por
margen): **el techo del frente de notas no lo pone la arquitectura.**

La vía que sí queda abierta y ya está medida es la **abstención**: contestar el 77,18 % y acertar
el 93,24 %. Y una variante que se desprende de este resultado y **NO está medida**: cuando el
sistema no puede desambiguar dentro de un par, **responder el grupo** («CLOP o RYUK») en vez de
elegir al azar. A nivel de grupo el sistema acierta 0,8585, y para un analista esa respuesta es
útil. **No se midió y no se reporta como si lo estuviera.**

---

### ✅ RESUELTO: 49,5 vs 50,5 PLANTILLAS — las dos cifras son correctas (2026-09-28)

Había quedado como cabo suelto que `protocolo_p2bal.py` reportara **49,5** plantillas de
entrenamiento por pliegue y la curva **50,5**. **No hay error en ninguno de los dos: cuentan
unidades distintas**, y la diferencia son exactamente los dos grupos mixtos.

| | |
|---|---|
| plantillas distintas (grupos de casi-duplicado) | **99** |
| pares (familia, plantilla) | **101** |
| diferencia | **2** = los grupos 6 y 53 |

Verificado corriendo el reparto de P2bal sobre las 50 semillas y contando de las dos maneras:

- contando **grupos**: **49,50** por pliegue → lo que reporta `protocolo_p2bal.py`
- contando **pares (familia, plantilla)**: **50,50** → lo que reporta la curva

Al citar, decir cuál de las dos unidades se está usando. **No deben aparecer las dos cifras en
el mismo documento sin esa aclaración.**

#### Los dos grupos mixtos fueron el hilo de toda la jornada

El grupo 6 (BLACKBASTA 1 + CONTI 1) y el grupo 53 (DHARMA 11 + PHOBOS 1) —14 notas, el 9,4 % del
corpus— explicaron cuatro cosas distintas el mismo día:

1. **La discrepancia 49,5 / 50,5**, resuelta arriba.
2. **El error del estrato** en `revision_bootstrap_estratificado.py`: tomar la plantilla entera
   metía notas ajenas en el estrato de cada familia.
3. **El parentesco DHARMA–PHOBOS**, que es el par de linaje con más confusión del corpus (403
   errores con texto solo).

> **⚠️ CORRECCIÓN (2026-09-28).** Esta lista decía **cuatro** manifestaciones e incluía que el
> preregistro P2 de `protocolo_p2bal.py` hubiera fallado «porque una familia puede entrar al
> entrenamiento a través del grupo mixto de otra». **Es falso y está medido.** Sobre los 100
> pliegues (50 semillas × 2), las únicas familias ausentes del entrenamiento son **BADRABBIT y
> CRYPTOLOCKER, 50 veces cada una y ninguna otra jamás**; sus grupos (3 y 44) **no son mixtos**,
> y las cuatro familias que sí están en grupos mixtos tienen entre 3 y 6 plantillas, así que
> nunca se quedan sin material. La causa real de que P2 fallara es **aritmética y ya estaba bien
> explicada en su bloque original**: una familia de plantilla única cae en un solo pliegue, o sea
> que está ausente del entrenamiento en **uno** de los dos y no en los dos — 2 familias × ½ =
> 1,00 por pliegue. La predicción de 2,00 estaba mal calculada; el protocolo no hace nada raro.
> Lo detectó la sesión hermana al medirlo. Es el cuarto error del día y del tipo más difícil:
> plausible, coherente con el resto de la historia, y falso.

**Merecen un párrafo propio en el capítulo metodológico.** Un grupo de casi-duplicados que
contiene notas de dos familias distintas no es una rareza del corpus: es una consecuencia directa
de que el agrupamiento sea **no supervisado** —como debe ser, para no usar las etiquetas— y toca
el conteo de plantillas, el reparto de la partición, el remuestreo y la interpretación del
parentesco.

---

### ★★ REVISIÓN CRUZADA DEL BOOTSTRAP POR PLANTILLA — ESTRATIFICAR POR FAMILIA (2026-09-28)

La sesión hermana midió el IC de P2bal por remuestreo de plantillas
(`bootstrap_plantilla_p2bal.py`) y encontró un sesgo de −0,047: la media de las réplicas queda
muy por debajo del punto estimado. Esta revisión confirma su diagnóstico y muestra que **el
sesgo no hay que corregirlo: hay que no generarlo**. `revision_bootstrap_estratificado.py`, log
`_log_revision_bootstrap.txt`, salidas en `resultados_revision_bootstrap_149/`, 2000 réplicas.
**Revisión posterior al resultado, sin preregistro, declarada como tal.**

#### El punto de fondo

Remuestrear las 99 plantillas sin mirar la familia trata al **conjunto de familias como
aleatorio** — como si el corpus pudiera no tener CERBER. Pero **las 30 familias no son una
muestra**: están fijadas por NapierOne y son el núcleo canónico que empareja los dos frentes del
trabajo. Lo muestral es **qué plantillas se consiguieron de cada familia**.

La pregunta que corresponde es «¿y si hubiéramos conseguido **otras plantillas de estas mismas
30 familias**?», y se responde remuestreando **dentro de cada familia**.

#### Las tres convenciones, sobre exactamente las mismas predicciones

| método | capa | punto | media | **sesgo** | IC 95 % | ancho |
|---|---|---|---|---|---|---|
| (a) libre + labels=30 | texto | 0,6551 | 0,6119 | **−0,0432** | [0,5066; 0,7182] | 0,2116 |
| (a) libre + labels=30 | cascada | 0,7417 | 0,6945 | **−0,0471** | [0,5872; 0,7935] | 0,2063 |
| (b) libre + labels presentes | texto | 0,6551 | 0,6594 | +0,0043 | [0,5508; 0,7682] | 0,2174 |
| (b) libre + labels presentes | cascada | 0,7417 | 0,7484 | +0,0068 | [0,6437; 0,8508] | 0,2071 |
| **(c) ESTRATIFICADO + labels=30** | texto | 0,6551 | 0,6540 | **−0,0011** | **[0,5654; 0,7446]** | 0,1792 |
| **(c) ESTRATIFICADO + labels=30** | **cascada** | **0,7417** | 0,7409 | **−0,0008** | **[0,6585; 0,8187]** | 0,1602 |

**Familias perdidas por remuestra: libre 2,15 de 30 · estratificado 0,00 por construcción.**

**(c) es la única que no obliga a elegir**: tiene sesgo casi nulo **y** conserva las 30 etiquetas
fijas. No arrastra el sesgo de (a) ni el denominador variable de (b), que era el precio de la
alternativa propuesta por la sesión hermana.

#### Consecuencia

- **El límite inferior de la cascada sube de 0,5872 a 0,6585.** La frase «el intervalo entero
  supera el umbral de 0,50» pasa de tener 0,087 de margen a tener **0,159**. La conclusión no
  solo sobrevive: queda más firme.
- **El texto solo pasa de 0,5066 a 0,5654** y deja de estar pegado al umbral. Sigue siendo cierto
  que la cascada tiene más margen (0,159 contra 0,065), pero el matiz cambia.
- El intervalo estratificado es **más angosto** (0,160 contra 0,206). Eso **no es hacerlo más
  favorable por conveniencia**: es más angosto porque no incluye la variación de «qué familias
  hay en el corpus», que en este diseño no es una fuente de incertidumbre real. Si alguien
  objeta el ancho, la respuesta es que mide otra cosa y que esa otra cosa es la que corresponde.

#### Cómo reportarlo

**(c) como intervalo principal, declarando qué pregunta responde, y (a) al lado** como el
escenario más conservador que además trataría al conjunto de familias como muestral. Las dos son
defendibles; lo que no conviene es citar una sola sin decir qué pregunta contesta.

#### ⚠️ ERROR PROPIO EN LA PRIMERA VERSIÓN, Y LA CONVERGENCIA FINAL

La primera implementación de (c) estratificaba **por plantilla entera**, y eso está mal en los
**dos grupos de casi-duplicados que cruzan familias**: el grupo 6 (BLACKBASTA 1 + CONTI 1) y el
grupo 53 (DHARMA 11 + PHOBOS 1), **14 notas, el 9,4 % del corpus**. Cuando DHARMA sorteaba el
grupo 53 entraban las 12 notas —incluida la de PHOBOS— y cuando PHOBOS sorteaba ese mismo grupo
entraban otra vez las 12: **notas ajenas se colaban en el estrato y el tamaño de cada familia
cambiaba entre réplicas**. Lo detectó la sesión hermana al comparar implementaciones; la suya ya
tomaba la unidad correcta.

**La unidad del estrato es el par (familia, plantilla)**, no la plantilla: de un grupo mixto
entran solo las notas de la familia que lo sorteó. Corregido; el sesgo residual bajó de −0,0120
a **−0,0011**.

**Las dos implementaciones, escritas por separado y sobre predicciones recalculadas de forma
independiente, convergen a CUATRO DECIMALES en las tres convenciones** —incluida la (c)
corregida: texto [0,5654; 0,7446] y cascada [0,6585; 0,8187] en ambas—. Es la validación cruzada
más fuerte que tiene el frente de notas hasta ahora.

#### La lección metodológica, en su formulación conjunta

Dos casos del mismo día, con desenlaces opuestos:
- En el acierto por linaje, los cinco preregistros dieron CUMPLE **con un bug adentro**. Lo que
  delató el error fue que la cifra era físicamente imposible, no que un test fallara.
- En el bootstrap, la predicción F5 era **una puerta de salida sobre el resultado** —el sesgo no
  puede pasar de 0,02— y falló, y por eso se fue a buscar la causa.

**Un preregistro protege contra elegir la hipótesis después de ver los datos, pero no protege
contra medir mal. Las predicciones tienen que incluir al menos una sobre la coherencia interna
del cálculo, y no solo sobre el resultado sustantivo.**

---

### ★★ ACIERTO A NIVEL DE LINAJE — CON CONTROL DE FUSIÓN ALEATORIA (2026-09-28)

`acierto_por_linaje.py --n-semillas 50 --sorteos 200`, **preregistro G1–G5 commiteado antes de
correr** (`74bc943`). Log `_log_acierto_linaje_149.txt`, salidas en
`resultados_acierto_linaje_149/`.

**Qué contesta:** cuánto del error restante es «confundir dos familias emparentadas» y cuánto es
error real. Se mide tratando cada par como una sola clase.

#### ⚠️ La métrica sube SIEMPRE, y por eso lleva control

Fusionar dos clases cualesquiera sube el acierto sin que el sistema haya mejorado en nada. El
número solo no significa nada. El control fusiona **la misma cantidad de pares elegidos al azar**
entre familias sin parentesco, promediado sobre 200 sorteos: la pregunta no es «¿sube?» sino
«¿sube **más** que fusionando pares cualesquiera?».

| fusión | sistema | clases | exactitud | **azar** | **ventaja** |
|---|---|---|---|---|---|
| sin fusión | cascada | 30 | 0,8123 | — | — |
| **3 pares fuertes** | **cascada** | **27** | **0,8585** | 0,8132 [0,8123; 0,8168] | **+0,0453** |
| 3 pares fuertes | texto solo | 27 | 0,8056 | 0,7206 [0,7191; 0,7285] | +0,0850 |
| fuertes + débiles | cascada | 23 | 0,8736 | 0,8161 [0,8123; 0,8294] | +0,0575 |
| fuertes + débiles | texto solo | 23 | 0,8275 | 0,7254 [0,7191; 0,7457] | +0,1021 |

**VEREDICTO: G1, G2, G3, G4 y G5 CUMPLEN los cinco.**

#### El dato que hace fuerte al resultado

**La fusión aleatoria casi no sube nada: 0,8123 → 0,8132, es decir +0,0009.** El efecto «por
construcción» que obligaba a poner el control resultó ser minúsculo. En consecuencia, de la
ganancia bruta de +0,0462 al fusionar los tres pares emparentados, **+0,0453 es ventaja real
sobre el azar** — prácticamente toda. Los errores **se concentran genuinamente en esos pares** y
no es un artefacto de reducir el número de clases.

Traducido: de los 18,8 puntos de error de la cascada, **unos 4,6 son confundir dos familias que
comparten el molde de la nota** — cerca de un cuarto del error total del sistema.

**G3 también cumple y es coherente con todo lo demás:** la ganancia por fusionar es mayor con el
texto solo (+0,0866) que con la cascada (+0,0462), porque la cascada ya resuelve por marcadores
buena parte de la confusión de linaje (486 → 186 errores).

#### Cómo se cita, obligatoriamente

Las tres cosas juntas y nunca una sola: **cifra, número de clases y línea de base aleatoria.**
«0,8585 sobre 27 clases, contra 0,8132 de fusionar tres pares al azar» — nunca «0,8585» a secas,
que se leería como una mejora del sistema y no lo es.

#### 🐛 BUG ENCONTRADO Y CORREGIDO EN LA PRIMERA CORRIDA — vale como lección

La primera corrida dio **los cinco preregistros en CUMPLE**, incluido el control G2. Pero la
fusión aleatoria daba **0,8054 contra 0,8123 sin fusionar**, y eso es **imposible**: fusionar
clases solo puede subir la exactitud o dejarla igual, nunca bajarla.

**Causa:** el mapa de fusión se construía a partir de las etiquetas del array recibido. Con `y`
traía las 30 familias; con las predicciones de una semilla, solo las familias efectivamente
predichas. Un par cuya familia nunca se predijo quedaba fusionado en `y` y **sin fusionar** en la
predicción, y aciertos se convertían en errores. Pegaba sobre todo en el control aleatorio,
porque los pares al azar incluyen BADRABBIT y CRYPTOLOCKER, que el modelo no predice nunca.

**Corrección:** el mapa se construye una vez sobre el universo completo de las 30 familias y el
**mismo** mapa se aplica a etiquetas y predicciones. Se agregó un control de sanidad que **aborta**
si la fusión baja la exactitud en alguna semilla.

**Efecto de la corrección:** la cifra principal (0,8585) **no cambió** —los pares reales sí se
predicen todos—; lo que cambió fue el control, de 0,8054 a 0,8132, y con él la ventaja declarada,
de +0,0531 a +0,0453.

**La lección, que vale para todo el proyecto:** el veredicto automático de los cinco preregistros
decía CUMPLE y no sirvió de nada. Lo que delató el bug fue que **la cifra era físicamente
imposible**, no que algún test fallara. Un preregistro protege contra elegir la hipótesis después
de ver los datos; **no protege contra medir mal**. Para eso hacen falta controles de coherencia
física, del tipo «esta cantidad no puede bajar».

---

### ★★ LOS PARES DE BOILERPLATE, VERIFICADOS UNO POR UNO (2026-09-28)

Quedaban «detectados, pendientes de verificación» tras el barrido del 26-09. Se abrieron los
archivos. **Ninguno es error de etiqueta**, y el parentesco tiene **tres formas distintas** que
conviene no mezclar.

#### La prueba que cierra la cuestión: cada familia conserva sus marcadores

Se cruzaron los marcadores (correos, *onion*, monederos, URL) de cada par:

| par | marcadores compartidos | propios de A | propios de B |
|---|---|---|---|
| CLOP – RYUK | **0** | 14 | 9 |
| DHARMA – PHOBOS | **0** | 18 | 7 |
| MEDUZALOCKER – SODINOKIBI | **0** | 11 | 8 |
| LORENZ – SODINOKIBI | 1 | 4 | 7 |
| BLACKCAT – SODINOKIBI | 1 | 7 | 7 |
| BLACKBASTA – CONTI | 1 | 3 | 6 |

**El único marcador compartido, en los tres casos donde aparece, es `https://torproject.org/`** —
la URL de descarga del navegador Tor, que no es un contacto. Si dos familias fueran en realidad
la misma mal separada, compartirían contactos; **ninguna los comparte**.

Esto además **valida empíricamente el mecanismo de la cascada**, que era la predicción de B.3 y
de M.4 y hasta ahora se sostenía indirectamente: *los marcadores son privados de cada familia aun
cuando el texto sea compartido*. Y explica por qué el filtro de genéricos del diccionario
descarta `torproject`: es literalmente el único valor que cruza familias.

#### Tres formas de parentesco, no una

Reconstruyendo los tramos **contiguos** de texto compartido (no solo el conteo de n-gramas):

| par | tramos | palabras compartidas | tramo mayor | prefijo idéntico |
|---|---|---|---|---|
| **CLOP – RYUK** | **1** | 100 de 235 (43 %) | **100 pal.** | **423 car.** |
| LORENZ – SODINOKIBI | 7 | 185 de 297 (**62 %**) | 64 pal. | 6 car. |
| BLACKCAT – SODINOKIBI | 2 | 46 de 162 (28 %) | 33 pal. | 0 car. |

- **CLOP–RYUK es un molde de apertura**: un único bloque continuo de 100 palabras con el que
  las dos notas empiezan. Es el caso más fuerte y el único con prefijo común.
- **LORENZ–SODINOKIBI comparte más texto en total (62 %) pero repartido en siete tramos
  sueltos**, y las notas no empiezan igual. Es reutilización de bloques, no del molde. El tramo
  mayor (64 palabras) es la advertencia de no usar software de recuperación; el segundo (53) es
  el pasaje «just a business…», característico de REvil/Sodinokibi.
- **BLACKCAT–SODINOKIBI comparte dos bloques temáticos** (la enumeración de datos exfiltrados y
  la amenaza de publicación), que es práctica común de la doble extorsión más que parentesco.

**Consecuencia para la tesis:** el par fuerte es uno solo, CLOP–RYUK, y se suma a los dos ya
documentados. Los tres pares con SODINOKIBI y NOTPETYA–WANNACRY se reportan como **bloques
reutilizados**, con esa etiqueta y no como linaje.

#### Procedencias: descartan el error de una sola fuente

- LORENZ/`pcrisk_lorenz_1.txt` es de **PCrisk con URL verificada**; SODINOKIBI/`revil1.txt` es de
  **ThreatLabz**. **Fuentes independientes**, así que el texto compartido no puede ser un error
  de catalogación de un solo curador.
- BLACKCAT/`alphv2.txt` y SODINOKIBI/`revil3.txt` son las dos de ThreatLabz, que las separa.

---

### 🚨🚨 HALLAZGO (2026-09-26): HAY UN TERCER PAR DE LINAJE SIN DOCUMENTAR — **CLOP ↔ RYUK**

Salió del barrido de boilerplate (`boilerplate_compartido.py`) y lo confirmó, de forma
**independiente**, la matriz de confusión (`confusion_y_linaje.py`). Dos métodos que no
comparten nada señalan el mismo par.

#### La evidencia, medida

| | valor |
|---|---|
| n-gramas de 8 palabras compartidos CLOP–RYUK | **185** |
| de ellos, **exclusivos del par** (en ninguna otra familia) | **169 (91 %)** |
| coseno char 3-5 `CLOP/clop1.txt` vs `RYUK/ryuk.txt` | **0,8018** |
| **contención de `RYUK/ryuk.txt` dentro de `CLOP/clop1.txt`** | **0,811** |
| confusiones RYUK → CLOP (50 semillas) | **117** con texto · **116** con cascada |
| confusiones CLOP → RYUK | 42 |

Las dos notas **empiezan con el mismo texto palabra por palabra**: «Your network has been
penetrated. All files on each host in the network have been encrypted with a strong
algorithm…». Tres notas de RYUK (`ryuk.txt`, `note_pcrisk.txt`, `note_variant_email.txt`, que
son casi idénticas entre sí, coseno 0,986–0,9999, grupo 123) están **contenidas al 81 %** dentro
de `CLOP/clop1.txt`, que es más larga.

#### Por qué nadie lo había visto

**El coseno se queda en 0,8018, por debajo del umbral 0,90.** El agrupador de casi-duplicados
las deja en plantillas distintas (grupo 37 y grupo 123) y el protocolo las trata como material
independiente. Es exactamente la limitación que midió la revisión del 2026-09-17 —el coseno no
ve la contención— pero acá el efecto **cruza familias**, que es peor: no es una plantilla
repetida, es una familia cuyo texto vive dentro de otra.

#### Lo que explica

**RYUK es la segunda peor familia del corpus** (acierto 0,2867 con cascada; patrón interno
0,0572, el 3.º más bajo de las 28). Ahora se entiende: sus notas **no se parecen entre sí** —los
cosenos internos van de 0,15 a 0,24 salvo el trío del grupo 123— **y sí se parecen a CLOP**. El
clasificador hace lo razonable con lo que tiene.

**Y la cascada no lo arregla: 117 → 116.** A diferencia de DHARMA↔PHOBOS (403 → 139, −65 %) y
BLACKBASTA↔CONTI (72 → prácticamente 0), acá los IOCs no separan: la capa de reglas solo resuelve
el 20 % de las notas de RYUK.

#### ✅ RESUELTO EL MISMO Día: es PARENTESCO, no error de atribución

Se verificó yendo al archivo original. **Las dos notas salen del MISMO repositorio fuente**
(`fuente = ThreatLabz` en el manifiesto = el repo `ransomware_notes` de Zscaler ThreatLabz, que
está en `3_datos/fuentes_notas/ransomware_notes/`): `clop/clop1.txt` y `ryuk/ryuk.txt`, con esos
mismos nombres. O sea que **la separación en dos familias la hace la fuente, no el armado del
corpus de la tesis.**

Y cada nota lleva **sus propios marcadores**, que es lo que zanja la cuestión:

- `ryuk.txt` (771 car.) termina con sus contactos propios — `WayneEvenson@protonmail.com`,
  billetera BTC `14hVKm7Ft2rxDBFTNkkRC3kGstMGp2A4hk` — y con **la firma explícita de la
  familia: «Ryuk. No system is safe»**.
- `clop1.txt` (1429 car.) sigue por otro lado después del tronco común y lleva los suyos.
- **Prefijo idéntico: 423 caracteres, el 55 % de la nota de Ryuk.** La primera diferencia es
  tipográfica (un guión largo contra uno corto en «SHUTDOWN»).

**Conclusión: las etiquetas están bien.** Es un molde de nota compartido entre dos familias
distintas, cada una con sus propios contactos y su propia firma. **CLOP–RYUK es un tercer par de
linaje y se declara como tal en la tesis**, junto a BLACKBASTA–CONTI y DHARMA–PHOBOS. No hay
nada que corregir en el corpus.

**⚠ CORRECCIÓN de lo que se dijo antes en esta misma sesión:** se afirmó que «tres de las cuatro
notas de RYUK implicadas vienen de PCrisk». **Es falso.** Según el manifiesto: `ryuk.txt` es de
**ThreatLabz**; `note_pcrisk.txt` y `note_variant_email.txt` son de **«NapierOne/varios»** (la
procedencia débil que B.2 del plan manda auditar, pese a lo que sugiere el nombre del archivo); y
la única de PCrisk con URL verificada es `pcrisk_ryuk_1.txt`, que **no** es del grupo implicado.

#### Por qué la cascada no separa este par, si cada nota tiene IOCs propios

Porque la regla solo aplica cuando el IOC de la nota de prueba **ya se vio en entrenamiento**.
Las tres notas casi idénticas de RYUK están en el **mismo grupo 123**, así que cuando ese grupo
cae en prueba sus correos y su billetera salen con él y la regla se queda sin clave. Por eso
RYUK tiene `frac_resuelta_por_regla` = 0,20, la más baja entre las familias confundidas. **No es
que los IOCs no sirvan: es que en esta familia no se repiten entre plantillas.**

#### Los otros pares que salieron del mismo barrido

| par | compartidos | exclusivos | ¿lo confirma la confusión? |
|---|---|---|---|
| BLACKBASTA–CONTI *(conocido)* | 239 | 239 (100 %) | sí, 72 con texto |
| DHARMA–PHOBOS *(conocido)* | 193 | 148 (77 %) | sí, **403** con texto |
| **CLOP–RYUK** | 185 | **169 (91 %)** | **sí, 117** |
| **LORENZ–SODINOKIBI** | 145 | **108 (74 %)** | sí, SODINOKIBI→LORENZ 35 |
| **BLACKCAT–SODINOKIBI** | 39 | **39 (100 %)** | — |
| **MEDUZALOCKER–SODINOKIBI** | 48 | 21 (44 %) | sí, 37 |
| **NOTPETYA–WANNACRY** | 22 | 18 (82 %) | sí, WANNACRY→NOTPETYA 44 |
| BADRABBIT–NOTPETYA | 11 | 7 (64 %) | sí, **100** |

**La coincidencia entre las dos columnas de la derecha es el resultado metodológico:** el
boilerplate compartido **predice dónde se va a equivocar el clasificador**, sin mirar ni una
predicción. Pares como DHARMA–LOCKBIT o CLOP–DHARMA, que comparten 27 y 16 n-gramas pero **0
exclusivos**, no generan confusión: lo que comparten es el boilerplate del ecosistema.

---

### ★★ MATRIZ DE CONFUSIÓN BAJO P2bal Y CIERRE DE M.4 (2026-09-26)

`confusion_y_linaje.py --n-semillas 50`, **preregistro L1–L5 commiteado antes de correr**
(`9849f55`). Log `_log_confusion_linaje_149.txt`, salidas en `resultados_confusion_linaje_149/`
(matrices 30×30 completas para texto y cascada). Puerta L1: reproduce 0,8123 / 0,7191 exacto.

| sistema | errores (50 semillas) | de linaje | fracción | por semilla |
|---|---|---|---|---|
| texto solo | 2093 | 486 | 0,2322 | 41,9 |
| **cascada** | **1398** | **186** | **0,1330** | 28,0 |

**VEREDICTO: L1, L2, L3 y L4 CUMPLEN los cuatro.** La fracción de errores de linaje baja de
0,2322 a 0,1330 y en absoluto cae a menos de la mitad (486 → 186). **La predicción que M.4 hacía
y nunca se puso a prueba —que los IOCs separan familias que comparten texto— queda confirmada**,
y la hace la cascada sola, sin la etapa de linaje que M.4 proponía.

#### Cierre de M.4, con el matiz que impone el hallazgo de CLOP–RYUK

El criterio automático del script dice «todavía tiene material» (13,3 % > 10 %). Leído con el
hallazgo de arriba, la conclusión es más precisa y **M.4 se cierra igual**:

- **Donde hay IOCs privados, M.4 ya está hecho:** DHARMA→PHOBOS pasa de 403 a 139 errores
  (−65 %) y BLACKBASTA→CONTI de 72 a prácticamente 0. Una segunda etapa de linaje no agregaría
  nada sobre eso.
- **Donde no hay IOCs, ninguna etapa de linaje ayuda:** RYUK→CLOP pasa de 117 a 116. La capa de
  reglas solo alcanza al 20 % de las notas de RYUK. **El problema no es de desambiguación, es
  que no hay señal que desambiguar.**

**Conclusión: M.4 no se corre.** Lo que proponía ya está medido por vía de la cascada, y el
residuo que quedaría a su cargo es justamente el caso donde su mecanismo —IOCs privados— no
existe.

---

### ★★ ¿CONVIENE QUE LA REGLA CEDA CUANDO EL TEXTO ESTÁ MUY SEGURO? (2026-09-26)

`cascada_cede_por_margen.py --n-semillas 50`, **preregistro C1–C5 commiteado antes de correr**
(`bf1ffdb`). Log `_log_cascada_cede_149.txt`, salidas en `resultados_cascada_cede_149/`. Sale del
hallazgo del mismo día: en el tramo fácil la capa de reglas resta.

La variante es una sola línea: la regla cede si el margen del texto supera un umbral. Umbral
infinito = la cascada actual; umbral 0 = el texto solo. **Las dos puertas reprodujeron exacto**
(0,7417 / 0,8123 y 0,6551 / 0,7191).

| umbral de cesión | cede en | macro-F1 (30) | Δ vs cascada | IC Bonferroni | semillas + |
|---|---|---|---|---|---|
| 0,00 (texto solo) | 80,3 | 0,6551 | −0,0866 | [−0,0954; −0,0777] | 0/50 |
| 0,25 | 63,6 | 0,7095 | −0,0321 | [−0,0377; −0,0266] | 0/50 |
| 0,50 | 50,0 | 0,7297 | −0,0120 | [−0,0155; −0,0085] | 2/50 |
| 0,75 | 39,7 | 0,7354 | −0,0062 | [−0,0087; −0,0038] | 2/50 |
| **1,00** | 29,4 | **0,7432** | **+0,0015** | **[+0,0002; +0,0028]** | 9/50 |
| 1,50 | 4,3 | 0,7417 | 0,0000 | [0; 0] | 0/50 |
| inf (actual) | 0,0 | 0,7417 | — | — | — |

**VEREDICTO: C1, C2, C3, C4 y C5 CUMPLEN TODOS.** Por el criterio de adopción fijado antes de
correr, el umbral 1,00 **se adopta**: su IC de Bonferroni al 99,17 % excluye el cero por arriba.

**Distribución del Δ por semilla en el umbral 1,00** (recalculada aparte para entenderlo): **41
semillas dan exactamente 0, 9 son positivas (media +0,0084) y NINGUNA es negativa.** O sea: la
variante **nunca empeora**; en 4 de cada 5 particiones no cambia nada y en el resto mejora poco.

#### ⚠️ RECOMENDACIÓN: cumple el criterio, pero NO conviene adoptarlo. Es decisión de Romina.

Tres razones, y ninguna contradice el preregistro —el criterio se cumplió y así queda escrito:

1. **La magnitud es despreciable:** +0,0015 sobre 0,7417 es un 0,2 % relativo. No mueve ninguna
   conclusión de la tesis.
2. **El punto dulce es estrecho y eso huele a sobreajuste del umbral:** con 0,75 pierde
   (−0,0062), con 1,50 ya no cede casi nunca (Δ = 0). Solo funciona en una ventana angosta.
3. **El umbral se eligió sobre el mismo conjunto con el que se mide**, sin partición de
   validación aparte —está declarado en el propio preregistro—. Con 149 notas no alcanza para
   partir en tres.

Complicar la descripción del método por +0,0015 hace el sistema más difícil de defender y no
más fuerte.

#### EL RESULTADO QUE SÍ VA A LA TESIS: el barrido VALIDA la jerarquía de la cascada

Lo valioso no es el +0,0015 sino la columna entera: **hacer que la regla ceda antes empeora de
forma sistemática y monótona** (−0,0866 · −0,0321 · −0,0120 · −0,0062, con 0/50, 0/50, 2/50 y
2/50 semillas positivas). **La prioridad «la regla manda sobre el texto» no es una elección de
comodidad: es la configuración medida como mejor**, y ahora hay una tabla que lo muestra. Eso
responde por anticipado a la pregunta de un jurado sobre por qué las capas van en ese orden.

---

### ★★ TOP-K: EL SISTEMA COMO RANKING (2026-09-26) — idea de Romina, medida

`topk_notas.py --n-semillas 50`, **preregistro T1–T4 commiteado antes de correr** (`bf1ffdb`).
Log `_log_topk_149.txt`, salidas en `resultados_topk_p2bal_149/`. Sale de la pregunta de Romina
en el chat con el tutor del 2026-09-09: «o dejamos siempre un tipo ranking de porcentajes».

**Cómo se arma el ranking de la cascada, que hay que declarar:** si la capa de reglas aplica y
apunta a una sola familia, esa familia va **primera** y detrás va el orden del texto sin
repetirla; si no aplica, el ranking es el del texto. Es la traducción directa de la cascada a
una lista.

| sistema | top-1 | top-2 | top-3 | top-5 | top-10 |
|---|---|---|---|---|---|
| texto solo | 0,7191 | 0,8192 | 0,8446 | 0,8787 | 0,9258 |
| **cascada** | **0,8123** | 0,8493 | **0,8663** | 0,8899 | 0,9286 |

**Frase citable:** la familia correcta es la primera propuesta en el **81,2 %** de las notas y
está **entre las tres primeras en el 86,6 %** (P2bal, 149 notas, 30 familias, 50 semillas).

**VEREDICTO: T1, T3 y T4 cumplen; T2 FALLA.** T2 predecía top-3 ≥ 0,90 y midió **0,8663**. La
predicción era optimista: suponía que los errores del SVC dejan la clase correcta cerca del
tope, y no es así — cuando la cascada no acierta de una, la familia correcta queda en la
**posición mediana 6 de 30** (el texto solo, en la 4).

**LO QUE MUESTRA EL RANKING, y es coherente con lo demás:** la cascada gana solo +0,0540 al
pasar de top-1 a top-3, y el texto solo gana +0,1255 (T3, cumple). Es la otra cara del hallazgo
de la capa de reglas: **cuando la regla se equivoca, se equivoca fuerte** y empuja la familia
correcta hacia abajo en la lista; el texto, en cambio, falla por poco. Las dos cosas —que la
regla acierte 0,9928 y que cuando yerra yerra feo— son el mismo hecho.

**DONDE EL RANKING SÍ SIRVE MUCHO: en las familias difíciles.**

| familia | cascada top-1 | cascada top-3 | gana |
|---|---|---|---|
| JIGSAW | 0,4200 | **0,7850** | +0,3650 |
| HELLOKITTY | 0,2267 | 0,4400 | +0,2133 |
| RYUK | 0,2867 | 0,4867 | +0,2000 |
| WANNACRY | 0,7700 | 0,9600 | +0,1900 |

JIGSAW casi duplica. **Para las familias que el sistema no resuelve de una, ofrecer tres
candidatas cambia la utilidad práctica de la herramienta** — y es justo el argumento de
despliegue que sostiene presentar un ranking además del top-1.

⚠️ **Top-10 apenas llega a 0,9286**: hay un ~7 % de notas donde la familia correcta no aparece
arriba en ninguna posición razonable. De ese 7 %, 2,7 puntos son las 4 notas de BADRABBIT y
CRYPTOLOCKER, cuya clase no existe en el modelo (T4, control de sanidad: aparecen en 0,0000 de
los casos, como debía ser).

**AL CITAR: top-k no es exactitud.** Se escribe siempre con la k pegada y al lado del top-1.

---

### ★ AÑO DE APARICIÓN vs RENDIMIENTO POR FAMILIA — EL AÑO NO EXPLICA NADA (2026-09-26)

Pedido textual del tutor, anotado en `HANDOFF_2026-08-25` §3(d). `anio_vs_rendimiento.py`, log
`_log_anio_vs_rendimiento.txt`, salidas en `resultados_anio_vs_rendimiento/`.
**⚠ ANÁLISIS POST-HOC, no preregistrado.**

**Fuente del año:** hoja «Informacion sobre familias» de `Pruebas.xlsx` (las 30 con año, armada
por Romina y Carlos). **No sale de MISP**, que solo cubre 5 de 28 — ya estaba verificado.
Emparejamiento 30/30 con mapeo explícito a mano: `BLACKCAT/alphv`, `BLACKMATTER7`,
`MEDUSALOCKERb7`, `CRYPTOLOCKERc9`.

**Resultado, sobre las 28 evaluables:**

| relación | Pearson r | p |
|---|---|---|
| año vs F1 de la cascada | **+0,091** | 0,645 |
| año vs F1 del texto solo | +0,164 | 0,406 |
| plantillas vs F1 de la cascada | −0,113 | 0,566 |
| año vs cantidad de plantillas | −0,305 | 0,115 |
| **año vs F1 descontando las plantillas** (parcial) | **+0,060** | 0,763 |

**Ninguna es significativa. El año de aparición de la familia no predice el rendimiento**, ni
antes ni después de descontar el número de plantillas —que era la variable de confusión obvia y
tampoco resultó significativa por sí sola.

| tramo | familias | plantillas media | F1 texto | F1 cascada | patrón medio |
|---|---|---|---|---|---|
| viejas 2013-2016 | 5 | 4,80 | 0,6359 | 0,8362 | 0,2872 |
| medias 2017-2019 | 12 | 3,25 | 0,6915 | 0,7463 | 0,4347 |
| recientes 2020-2022 | 11 | 3,27 | 0,7432 | 0,8284 | 0,4475 |

**EL CONTRASTE ES EL RESULTADO, y es lo que hay que llevarle al tutor.** Se probaron dos
explicaciones del rendimiento por familia, las dos a pedido suyo:

| explicación candidata | r con el F1/acierto | p |
|---|---|---|
| **año de aparición** | +0,091 | 0,645 — **no explica nada** |
| **patrón interno de la familia** | **+0,834** (texto solo) | < 0,00001 — **explica mucho** |

Lo que decide si una familia se clasifica bien **no es cuándo apareció, es si sus notas se
parecen entre sí**. Los casos lo muestran: CHIMERA es de 2015 y BLACKBASTA de 2022, y las dos
tienen texto malo (0,2947 y 0,2020) rescatado por la cascada (0,8920 y 0,7456); TESLACRYPT
(2015) y BLACKCAT (2021) andan bien las dos. **Familias de todas las épocas aparecen arriba y
abajo de la tabla.**

⚠️ n = 28 es chico: un caso extremo mueve la correlación. Se cita como exploración post-hoc.

---

### INVENTARIO DE FAMILIAS DISPONIBLES PARA EXTENDER EL FRENTE DE NOTAS (2026-09-26, local)

Contesta la pregunta «¿podemos ampliar las familias?» con el criterio del proyecto, no con
cuántas carpetas traen las fuentes. `inventario_familias_fuentes.py`, log
`_log_inventario_fuentes.txt`, salidas en `inventario_familias_fuentes/`.

**Lo que hay en disco hoy** (`3_datos/fuentes_notas/`, 690 notas leídas):

| fuente | notas | familias | familias con 2+ plantillas | familias nuevas con 2+ |
|---|---|---|---|---|
| `ransomware_notes` | 332 | 222 | 48 | 40 |
| `RansomNoteFiles` | 157 | 67 | 18 | 14 |
| `f6dfir_ransom_notes` | 182 | 31 | 21 | 21 |
| `notas_pcrisk` | 19 | 13 | 6 | 0 |

**Unificando todo con el corpus canónico** (agrupamiento por coseno char 0,90 sobre todo junto):

- 319 familias distintas en total; **289 son nuevas** (no están entre las 30).
- **77 familias nuevas llegan a 2 o más plantillas** — el piso para que una familia no saque
  F1 = 0 estructural bajo corte por plantilla.
- 32 familias nuevas llegan a 3 o más; 17 llegan a 4 o más.
- **Total de familias con 2+ plantillas, canónicas más nuevas: 105.**
- De las 690 notas de fuentes, **131 colapsan** con una plantilla que el corpus ya tiene y
  **559 aportan texto distinto**.

**⚠ 25 grupos de casi-duplicados cruzan dos o más familias**, y hay que resolverlos a mano antes
de incorporar nada. El más grande junta **btcware | dharma | phobos en un solo grupo de 29
notas**. **Ojo: DHARMA↔PHOBOS no es hallazgo nuevo** — ya está registrado y analizado en este
mismo archivo (grupo 55/56, 11 notas de DHARMA + `pcrisk_phobos_1.txt`, y el grafo B.3 concluyó
que el vínculo es por contenido casi duplicado, no por marcadores). El corpus canónico tiene
**2 grupos mixtos y solo 2**: DHARMA+PHOBOS (12 notas) y BLACKBASTA+CONTI (2 notas), verificado
hoy corriendo el agrupamiento sobre las 149. Lo que agregan las fuentes nuevas a ese grupo es
**btcware**, familia externa. Otros grupos cruzados: `blackbasta|conti|monti`,
`cartel|hades|lapiovra|revil|sodinokibi|sugar` (6 familias en un grupo), `conti|zeon`,
`stop|stopdjvu`, `blacklock|dragonforce|eldorado`. Lista completa en
`conflictos_de_etiqueta.csv`. Es la trampa del homónimo, que en este proyecto ya pegó tres veces.

**Este inventario es un TECHO OPTIMISTA.** «Plantilla» es coseno 0,90 y ese criterio no detecta
contención: con contención ≥ 0,8 el corpus canónico pasa de 99 a 81 plantillas y de 28 a 21
familias evaluables. El número real de familias con dos plantillas de verdad independientes es
**menor que 77**, y no está medido.

---

# 📌 PREREGISTRO — Exp. 2e: rasgos estructurales para mejorar el frente de archivos (2026-09-22)

**Objetivo declarado: MEJORAR el 0,912, no medirlo otra vez.** Pedido explícito de Romina.

## De dónde sale la idea (no es una corazonada, son dos mediciones propias)

Del job 3630, ya registrado más arriba:
1. **El 79,6 % de la importancia está en la cola** (cabecera 20,4 %), y los doce
   desplazamientos más importantes son todos de cola.
2. **SUNCRYPT tiene entropía de cola 4,78 y NOTPETYA 6,58**, contra 7,44 del resto. Las dos
   **sí dejan algo estructurado al final** y aun así son la 1.ª y la 6.ª peor familia.

El capítulo ya explica por qué no alcanza: *«ese bloque varía en cada archivo: una clave, un
identificador de víctima o un contador. Es estructura, pero no es firma de familia»*.

**Ahí está la palanca.** La representación posicional aprende **valores de byte en posiciones
fijas**. Un pie cuyo contenido cambia en cada archivo es invisible para ella —aunque su
**presencia**, su **tamaño** y su **grado de aleatoriedad** sean constantes dentro de la
familia. «Hay 200 bytes poco aleatorios al final y el tamaño es múltiplo de 16» **es** un
rasgo de familia aunque esos 200 bytes sean distintos en cada archivo.

## Qué se agrega

44 rasgos que describen la **forma** del archivo, no su contenido: entropía a ocho
profundidades en cada extremo (16 a 4096), ocho bloques de 512 repartidos por el archivo,
**largo del bloque final no aleatorio**, tamaño y sus restos módulo 16/512/4096, χ² contra la
uniforme, bytes distintos, frecuencia máxima, ceros y fracción imprimible.

**Ninguno mira el nombre ni la extensión.** Es la diferencia con el Exp. 2d: si acá aparece
mejora, es del **contenido**, y **no arrastra la limitación de campaña**. Ese es el punto.

Tres columnas sobre la misma partición: (1) bytes canónico · (2) bytes + estructura ·
(3) solo estructura (control de complementariedad). Delta pareado e IC 95 %, 5 semillas, más
reporte por familia de (1) y (2) y tabla de delta por familia. Corre sobre el **corpus
corregido**, así que de paso mide cuánto mueve devolver los 310 `.pdf` cifrados.

## ⚠ PREDICCIONES REGISTRADAS ANTES DE CORRER

Se escriben para poder equivocarse en público, como en el Exp. 2d —donde la predicción falló
por cuarenta puntos y eso decidió la lectura del resultado.

| Qué | Predicción |
|---|---|
| (1) bytes canónico, exactitud, corpus corregido | **0,912 a 0,925** (sube o queda igual; NOTPETYA recupera 167 archivos) |
| (2) bytes + estructura, Δ macro-F1 contra (1) | **+0,010 a +0,030**, IC excluye el cero |
| (3) solo estructura, macro-F1 | **0,40 a 0,60** — no alcanza sola, 44 rasgos no separan 30 familias |
| Dónde cae la mejora | **delta medio mayor en las seis difíciles que en las otras 24** |
| Las dos que más suben | **SUNCRYPT y NOTPETYA** (son las que tienen cola de baja entropía) |

**Cómo se lee cada desenlace:**
- Si (2) mejora y cae en las difíciles → **mejora genuina del método por contenido**, se
  reporta como resultado principal y sube el número del capítulo.
- Si (2) mejora pero se reparte parejo → es mejora real pero **no cierra el residuo**; se
  reporta con esa lectura y sin la historia de las seis.
- Si (2) no mejora → **queda medido que el residuo no es de representación sino de ausencia
  de señal**, que es un resultado negativo fuerte y citable: refuerza la conclusión ya escrita
  de que la indistinguibilidad de esas seis familias es real.
- Si (3) sale alto (>0,80) → los rasgos estructurales son un método por sí mismos y hay que
  replantear, no solo agregar.

**Riesgo conocido:** si el pie de SUNCRYPT **también varía de largo** entre archivos, el rasgo
`largo_cola_no_aleatoria` tampoco lo captura y la predicción falla. No se puede saber sin
correr.

Código: `2_codigo/exp2e_estructura_bytes.py` + `slurm/job_exp2e.sh`, commiteados y pusheados.
Probado con un corpus sintético de tres familias; el extractor detecta un pie de 192 bytes
con exactitud.

---

---

# 📐 D.1 *majority voting*: el techo aritmético, calculado (2026-09-26)

Pregunta de Romina: «¿combinado con el frente de notas mejoraremos mucho?». Se calcula el
techo con las cifras vigentes de los dos frentes, para que el elemento de acción 3 del tutor
se conteste con una cuenta y no con una impresión.

**Puntos de partida, con su base y su métrica:**
- Archivos (Exp. 2c): **exactitud 0,912 ± 0,002**, macro-F1 0,911 · 30 familias · 15.000
  archivos de NapierOne · cobertura 1,00.
- Notas (M.6, sin abstención): **acierto 0,6601**, macro-F1 0,5191 · 30 familias · 149 notas
  · cobertura 1,00.

## El techo si existieran datos pareados

El clasificador de archivos falla en el **8,8 %** de los casos. Si las notas fueran
independientes y acertaran en el 66,01 % de esos, un **oráculo perfecto** —que supiera
exactamente cuándo el clasificador de archivos se equivoca— rescataría
0,088 × 0,6601 = **+5,8 puntos**, o sea 0,912 → **0,970**. Ese es el techo absoluto, y no es
alcanzable: supone saber de antemano dónde está el error.

**El riesgo del otro lado es 5× mayor.** Archivos acierta y notas falla en
0,912 × 0,3399 = **31,0 %** de los casos. Una regla que delegue mal ahí destruye más de lo
que gana.

**Punto de equilibrio:** delegar conviene solo sobre un subconjunto donde el clasificador de
archivos falle en más del **34 %** de los casos (de 0,3399/0,6601). Su tasa base de error es
8,8 %, así que la regla de delegación tiene que concentrar los errores casi **cuatro veces**
por encima de la base.

## ¿Es alcanzable? Sí, y por eso el número final es chico pero no cero

Hay una regla obvia que supera el umbral: **delegar cuando el clasificador de archivos predice
una de las seis familias difíciles.** Su F1 medio ahí es ~0,57 (SUNCRYPT 0,745 · WASTEDLOCKER
0,628 · CRYPTOLOCKER 0,605 · DARKSIDE 0,603 · JIGSAW 0,439 · NOTPETYA 0,394), o sea ~43 % de
error: por encima del 34 % de equilibrio.

Ganancia esperada con esa regla, si las seis son el 20 % de los casos:

    0,20 × (0,43 × 0,6601 − 0,57 × 0,3399) = 0,20 × 0,090 ≈ **+1,8 puntos**

**0,912 → ~0,930.** Es real, pero es «un par de puntos», no «mucho».

## Por qué igual no se puede reportar

**No hay muestras pareadas** (ya registrado en `PLAN_MEJORAS.md` D.1): las notas vienen de
repositorios públicos y los archivos de NapierOne, así que **no existe un incidente del que se
tengan los dos artefactos**. Emparejar al azar una nota de CERBER con un archivo de CERBER no
mide nada: fabrica una correlación que los datos no tienen, y el «resultado» sería aritmética
de las dos marginales, calculable sin correr nada —justamente lo que se acaba de hacer acá.

## Comparación de palancas, que es lo que decide la prioridad

| Palanca | Ganancia | ¿Medible? | Limitación que arrastra |
|---|---|---|---|
| Exp. 2d — forma del nombre | **+8,8 pts** (medido) | sí | campaña: no generaliza a campañas no vistas |
| Exp. 2e — rasgos estructurales | +1 a +3 pts (predicho) | sí | ninguna: es contenido |
| Corpus corregido (310 `.pdf`) | ? (a medir en 2e) | sí | ninguna |
| **D.1 combinar los frentes** | **~+1,8 pts** (calculado) | **NO** | requiere datos que no existen |

**Combinar es la palanca más chica y la única que no se puede medir.** Conclusión para el
elemento de acción 3: de las tres salidas que lista D.1 —implementarlo, proponerlo sin
evaluar, o argumentar por qué no— la que sostienen los números es **proponerlo como esquema de
despliegue con el techo calculado y la razón medida**, que es bastante más fuerte que un «no
se pudo».

## Dónde sí aportaría combinar, y no es la exactitud

1. **Cobertura de artefactos.** En un incidente real puede haber solo nota, solo archivos, o
   los dos. Un protocolo de dos frentes cubre los tres casos; un clasificador solo, no.
2. **Abstención.** Con M.3 a umbral 1,00 las notas dan **acierto 0,9687 donde contestan** con
   cobertura 0,5495. Como *confirmador* —no como votante— la nota es precisa. El caso real del
   22-08 lo mostró: el sistema **se abstuvo bien** ante una familia desconocida, y ese fue el
   comportamiento valioso, no la clasificación.

## ~~La única vía que desbloquearía D.1 de verdad~~ — FALSO (28-09): f6-dfir no tiene NINGUNA familia en común con las 30 de NapierOne

**f6-dfir**, si sus archivos cifrados vienen del mismo incidente que sus notas. Ya está como
decisión abierta (requiere descargar material cifrado ⇒ es de Romina y Cappo). Si están
pareados, D.1 pasa de «no evaluable» a evaluable, y es lo único que lo hace.

---

---

# ✅✅ EXP. 2e CERRADO — la estructura mejora el frente de archivos, y la mejora cae ENTERA en las seis difíciles (2026-09-26)

Job 4058, nodo c2, 1 h 47 min. Registrado de lo pegado por Romina. Salidas en
`/scratch/ralfonzo/tesis/resultados_exp2e_job4058`. **Sin nombre ni extensión: es contenido.**

## Las tres columnas (5 semillas, 15.000 archivos, 30 familias, corpus corregido)

| Columna | Exactitud | macro-F1 |
|---|---|---|
| (1) bytes canónico 512+512 | 0,9123 ± 0,0003 | 0,9114 ± 0,0004 |
| **(2) bytes + 44 rasgos estructurales** | **0,9357 ± 0,0005** | **0,9359 ± 0,0004** |
| (3) solo los 44 rasgos | 0,8699 ± 0,0030 | 0,8680 ± 0,0032 |

| Delta pareado por semilla | Δ | IC 95 % | Semillas |
|---|---|---|---|
| (2) − (1) exactitud | **+0,0234** | [+0,0228; +0,0241] | 5/5 |
| **(2) − (1) macro-F1** | **+0,0246** | **[+0,0237; +0,0254]** | **5/5** |
| (3) − (1) macro-F1 | −0,0434 | [−0,0473; −0,0394] | 0/5 |

**Cifra nueva del frente de archivos: macro-F1 0,9359 ± 0,0004**, contra 0,9114 de la
representación canónica sobre la misma partición. **Y sin mirar el nombre ni la extensión**,
así que —a diferencia del Exp. 2d— no arrastra la limitación de campaña.

## ⭐ Dónde cae la mejora: el resultado principal

| | Δ medio de F1 |
|---|---|
| **Las seis difíciles del Exp. 2c** | **+0,1186** |
| Las otras 24 familias | **+0,0006** |

**Doscientas veces más.** La mejora no se reparte: va entera al residuo que el capítulo ya
había identificado. Por familia (⚠ **semilla 0**, que es la única con reporte por familia; los
agregados de arriba son sobre las 5):

| Familia | Solo bytes | Con estructura | Δ |
|---|---|---|---|
| WASTEDLOCKER | 0,6397 | **0,8317** | **+0,1920** |
| JIGSAW | 0,4285 | 0,5711 | +0,1426 |
| DARKSIDE | 0,5982 | 0,7327 | +0,1345 |
| NOTPETYA | 0,3614 | 0,4839 | +0,1224 |
| SUNCRYPT | 0,7719 | 0,8394 | +0,0675 |
| CRYPTOLOCKER | 0,6026 | 0,6552 | +0,0526 |
| BADRABBIT | 0,9827 | **1,0000** | +0,0173 |

Las otras 23 se mueven entre +0,0050 y −0,0061. Ninguna se rompe: la peor caída es HELLOKITTY
con −0,0061.

**Media de las seis: 0,5671 → 0,6857.** WASTEDLOCKER cruza por encima de 0,75 y JIGSAW por
encima de 0,50. Con la salvedad de que es una semilla, el residuo del capítulo se **achica**,
no desaparece.

## Las cinco predicciones del preregistro: tres bien, dos mal

| Predicción | Resultado | |
|---|---|---|
| (1) exactitud 0,912 a 0,925 | 0,9123 | ✅ |
| (2) Δ macro-F1 +0,010 a +0,030 | +0,0246 | ✅ mitad superior |
| (3) solo estructura 0,40 a 0,60 | **0,8680** | ❌ por 27 puntos |
| Δ mayor en las seis que en las otras 24 | +0,1186 vs +0,0006 | ✅ y por 200× |
| **Las dos que más suben: SUNCRYPT y NOTPETYA** | **WASTEDLOCKER y JIGSAW** | ❌ |

## ⚠ La predicción fallada que importa: el mecanismo NO es el que razoné

Diseñé `largo_cola_no_aleatoria` pensando en SUNCRYPT (entropía de cola 4,78) y NOTPETYA
(6,58): las dos que **sí** dejan un pie poco aleatorio. Predije que serían las que más
subieran.

**Subieron menos que WASTEDLOCKER y DARKSIDE**, y esas dos son justamente las que el capítulo
describe como *«exactamente en el techo»* de entropía —7,591 y 7,585 WASTEDLOCKER, 7,593 en
cabecera DARKSIDE—, es decir **indistinguibles de datos aleatorios en ambos extremos**.

Entonces la señal que las rescata **no puede ser la entropía del pie**. Los candidatos que
quedan entre los 44 rasgos son los de **tamaño**: `tam`, `tam_mod16`, `tam_mod512`,
`tam_mod4096`. Todas las familias cifraron **el mismo conjunto base de documentos**, así que
las diferencias de tamaño entre familias son diferencias en **cuánto agrega cada una** —relleno
a bloque, pie de longitud fija—. Eso es una propiedad del **código**, invisible para una
representación que mira valores de byte en posiciones fijas.

**Es hipótesis, no medición.** Se resuelve con las importancias del bosque de la columna (2),
que el script no guardó. Es barato: una semilla, ~20 min. **No se corre sin que Romina lo
pida**, pero es la pregunta que un jurado va a hacer —«¿qué rasgo hace el trabajo?»— y
conviene tener la respuesta medida.

## Lo que (3) = 0,8680 significa

Estaba escrito en el preregistro: *«si (3) sale alto (>0,80) → los rasgos estructurales son un
método por sí mismos y hay que replantear, no solo agregar»*. **44 números dan 0,8680 contra
0,9114 de los 1.024 bytes posicionales: cuatro puntos menos con 23 veces menos
características.** No es un agregado, es una **representación alternativa**, compacta y
—argumentablemente— más robusta al cambio de campaña, porque el tamaño y la forma del pie los
fija el código y no la configuración. Eso último **no se puede medir sobre NapierOne**; se
propone y se declara.

## Efecto del corpus corregido: nulo en el agregado

0,9123 con las 310 muestras devueltas contra 0,9120 publicado: **+0,0003**. Devolver los
`.pdf` cifrados de BADRABBIT y NOTPETYA no movió el número global. Coherente: son 1,5 familias
de 30. **Lo que sí se ve es en BADRABBIT, que pasa a 1,0000 con estructura.** La corrección del
filtro era necesaria por honestidad en la descripción del corpus, no porque cambiara cifras.

---

---

# ✅ EXP. 2e-b — QUÉ rasgo hace el trabajo: el TAMAÑO, y mi rasgo estrella no aporta nada (2026-09-27)

Job 4059, COMPLETED en 6 min 2 s, MaxRSS 3,5 GB. Salidas en
`/scratch/ralfonzo/tesis/resultados_exp2e_rasgo_job4059`. Una semilla: es diagnóstico, no
cifra reportable.

## Ablación por grupo sobre «solo estructura» (44 rasgos, macro-F1 0,8716)

| Grupo quitado | n | macro-F1 | Caída | Seis difíciles | **Caída dif.** |
|---|---|---|---|---|---|
| **tamaño** | 5 | 0,7489 | −0,1227 | 0,3457 | **−0,2247** |
| entropía de cabecera | 8 | 0,8612 | −0,0103 | 0,5206 | −0,0498 |
| entropía del medio | 12 | 0,8634 | −0,0082 | 0,5400 | −0,0303 |
| distribución de bytes | 9 | 0,8542 | −0,0174 | 0,5471 | −0,0233 |
| entropía de cola | 8 | 0,8149 | −0,0566 | 0,5527 | −0,0177 |
| salto cabecera-cola | 1 | 0,8687 | −0,0029 | 0,5612 | −0,0092 |
| **pie no aleatorio** | 1 | 0,8716 | **+0,0001** | 0,5699 | **−0,0005** |

Base: seis difíciles 0,5704 con los 44 rasgos.

## Los tres hallazgos, en orden de importancia

**1. El tamaño es el grupo dominante, y por lejos.** Quitar los cinco rasgos de tamaño cuesta
**−0,2247 en las seis difíciles**: cuatro veces y media más que el segundo grupo. Es coherente
con el mecanismo que se sospechaba: en NapierOne **todas las familias cifraron el mismo
conjunto base de documentos** —ya está escrito en §subsec:exp2c_tipos—, de modo que las
diferencias de tamaño entre familias son diferencias en **cuánto agrega cada una**: relleno a
bloque, pie de longitud fija, cabecera propia. Es propiedad del **código** de la familia, e
invisible para una representación que mira valores de byte en posiciones fijas.

**2. Pero el tamaño solo NO alcanza.** Los cinco rasgos aislados dan macro-F1 **0,3272** y
**0,1151** en las seis difíciles. O sea: el tamaño es **necesario y no suficiente**. La señal
sale de la **interacción** entre el tamaño y el perfil de entropía; ninguno de los dos hace el
trabajo por su cuenta. Esto hay que escribirlo así, porque «el tamaño identifica a la familia»
sería falso.

**3. ❌ `largo_cola_no_aleatoria` no aporta NADA.** +0,0001 global, −0,0005 en las difíciles.
Es exactamente el rasgo que diseñé para este experimento, razonando desde la entropía de cola
4,78 de SUNCRYPT y 6,58 de NOTPETYA. **Es ruido.** La mejora de +0,0246 del Exp. 2e habría
salido igual sin él.

## Dos veces equivocado sobre el mismo mecanismo

- **Predicción del preregistro:** «las dos que más suben son SUNCRYPT y NOTPETYA». Fue
  WASTEDLOCKER (+0,1920) y JIGSAW (+0,1426).
- **Hipótesis del 26-09 al ver eso:** «entonces la señal es el tamaño». A medias: el tamaño es
  el grupo dominante, pero solo da 0,3272 aislado. La explicación correcta es la interacción.

Queda registrado porque el patrón importa: **el razonamiento de diseño acertó el resultado y
erró el mecanismo dos veces seguidas.** El experimento funcionó por una razón distinta de la
que lo motivó, y solo se supo porque se midió.

## Lo que cambia para la redacción

No se escribe «los rasgos de entropía del pie rescatan a las familias sin firma». Se escribe:

> El grupo de rasgos de tamaño es el que sostiene la mejora ---quitarlo cuesta 0,2247 de F1 en
> las seis familias difíciles, contra 0,0498 del segundo grupo--- pero no la explica por sí
> solo: los cinco rasgos de tamaño aislados alcanzan apenas 0,1151 sobre esas familias. La
> señal reside en la combinación del tamaño con el perfil de entropía.

Y el orden para las seis difíciles, citable: **tamaño −0,2247 · entropía de cabecera −0,0498 ·
entropía del medio −0,0303 · distribución −0,0233 · entropía de cola −0,0177 · salto −0,0092 ·
pie no aleatorio −0,0005**. Notar que **la cola casi no importa para las difíciles** aunque sea
el segundo grupo a nivel global (−0,0566): son poblaciones distintas y conviene no mezclarlas.

## (A) Importancias: el ranking se INVIERTE, y eso confirma el diseño

| Grupo | Importancia acumulada | Puesto por importancia | Puesto por ablación (difíciles) |
|---|---|---|---|
| entropía de cola | 0,0740 | 1.º | 5.º (−0,0177) |
| entropía de cabecera | 0,0690 | 2.º | 2.º (−0,0498) |
| distribución | 0,0606 | 3.º | 4.º (−0,0233) |
| **tamaño** | **0,0409** | **4.º** | **1.º (−0,2247)** |
| entropía del medio | 0,0138 | 5.º | 3.º (−0,0303) |
| salto cabecera-cola | 0,0055 | 6.º | 6.º (−0,0092) |
| pie no aleatorio | 0,0031 | **último** | **último** |

**Reparto global:** 1.024 bytes posicionales **0,7331** · 44 rasgos estructurales **0,2669**.
Los rasgos son el 4,1 % de las columnas y se llevan el 26,7 % de la importancia: **cada rasgo
estructural pesa 8,5 veces lo que un byte**.

**La contradicción es aparente y era predecible.** Está escrito en el docstring del script
antes de correr: *«la ablación es más informativa que las importancias cuando los rasgos están
correlacionados»*. Hay **ocho** entropías de cola midiendo casi lo mismo a ocho profundidades;
cada una recibe una tajada de importancia y la suma del grupo queda alta, pero **quitarlas
todas cuesta poco porque los otros grupos cubren la misma información**. El tamaño tiene solo
**cinco** rasgos y **nada lo sustituye**: por eso puntúa cuarto en importancia y primero en
ablación.

**Regla de lectura para la tesis: manda la ablación**, que mide qué pasa si el rasgo no está.
Las importancias sirven para **nombrar** el rasgo individual, no para ordenar grupos.

## ⭐ El rasgo individual más importante tiene nombre: `tam_mod16`

| Rasgo | Importancia | Grupo |
|---|---|---|
| **`tam_mod16`** | **0,0294** | tamaño |
| `H_cola_32` | 0,0230 | entropía de cola |
| `ascii_cola` | 0,0228 | distribución |
| `H_cab_64` | 0,0191 | entropía de cabecera |
| `H_cola_16` | 0,0172 | entropía de cola |
| `H_cab_16` | 0,0122 | entropía de cabecera |
| `maxfrec_cab` | 0,0119 | distribución |
| `tam_mod512` | 0,0082 | tamaño (11.º) |

**El resto del tamaño del archivo módulo 16** —el tamaño de bloque de AES— es el rasgo
estructural individual más informativo de los 44, un 28 % por encima del segundo.

**Y tiene una explicación mecánica que se puede escribir.** Todas las familias cifraron el
mismo conjunto base de documentos, así que el tamaño original es el mismo y lo que varía es la
transformación. Una familia que usa cifrado por bloques con relleno deja el tamaño en múltiplo
de 16 más el largo de su pie; una que usa cifrado de flujo deja el resto original intacto. **La
distribución de `tam_mod16` dentro de una familia es, entonces, una huella del modo de cifrado
y del tamaño del pie** — las dos cosas las fija el código, no la configuración de la campaña.
Es el mejor argumento disponible a favor de que esta representación resista el cambio de
campaña.

*Matiz técnico que conviene anotar:* las importancias por impureza de un bosque favorecen a
los rasgos continuos y de alta cardinalidad. `tam_mod16` tiene apenas 16 valores posibles y aun
así encabeza la lista **contra** ese sesgo, lo que refuerza el hallazgo en vez de debilitarlo.

## Doble confirmación de que mi rasgo diseñado es inútil

`pie_no_aleatorio` queda **último en las dos mediciones**: importancia 0,0031 (último GRUPO; que sea el rasgo individual más bajo de los 44 NO está verificado: solo se vio el top 12 y los totales por grupo — corrección del 28-09 —; lo más bajo de
las 44) y ablación +0,0001. No es que una medición lo salve y la otra no: las dos coinciden en
que **no aporta nada**.

---

---

# ✅ EL FRENTE DE ARCHIVOS QUEDA REDACTADO (2026-09-28)

Se agregaron a `resultados.tex` **cinco piezas**, todas con base y métrica declaradas. **Solo se
agregó: no se tocó una línea de lo ya escrito**, salvo la corrección de dos palabras que pidió
Romina (abajo). Respaldo previo en `resultados.tex.antes_2d2e`.

**Compila: 84 páginas, 0 errores, 0 referencias sin resolver** (venía de 74).

| Dónde | Qué | Etiqueta |
|---|---|---|
| dentro del Exp. 2c | Censo de integridad de las 30 carpetas + CERBER como cifrado parcial | `subsec:exp2c_integridad_censo` |
| dentro del Exp. 2c | El sesgo del filtro `.pdf` y su corrección (310 muestras) | `subsec:exp2c_sesgo_pdf` |
| dentro del Exp. 2c | Curva de aprendizaje por archivos/familia (A.3) | `subsec:exp2c_curva` |
| sección nueva | **Experimento 2d** — aporte del nombre, con los dos controles | `sec:exp2d` |
| sección nueva | **Experimento 2e** — rasgos estructurales, con la ablación por grupo | `sec:exp2e` |

Etiquetas nuevas que definen: `tab:censo_integridad` · `tab:curva_archivos` · `tab:exp2d` ·
`subsec:exp2d_limitacion` · `tab:exp2e` · `tab:exp2e_dificiles` · `tab:exp2e_ablacion`.

## La corrección de dos palabras

En `subsec:exp2c_limitacion` decía «…un conjunto de datos con múltiples campañas por familia,
**no disponible en la actualidad**». Eso afirmaba algo sobre el mundo que no se comprobó: existen
colecciones públicas de material cifrado cuyo contenido este trabajo no revisó. Reemplazado por
«…**del que no se dispuso para este trabajo**», que dice exactamente lo que se sabe y no obliga a
defender lo que no. Es el cambio que proponía `nota_limitacion_napierone.tex`, ahora aplicado.

## Decisiones de redacción que se tomaron, por si hay que revisarlas

1. **El 0,912 sigue siendo la cifra canónica del frente.** El 2e se presenta como mejora que se
   agrega, no como reemplazo. Si Cappo prefiere lo contrario, es cambiar el énfasis de la
   síntesis, no reescribir las secciones.
2. **La columna del nombre (0,9998) se reporta como resultado**, no como cota superior, porque el
   control de 0,5771 lo habilita. La limitación de campaña va en su propia subsección y el
   resultado se enuncia con las tres frases juntas.
3. **La extensión literal se declara como cota superior** y se dice explícitamente que un
   diccionario sin aprendizaje (0,9244) supera al método (0,9117), con la razón al lado.
4. **A.3 entró** ---Romina había dicho que no le interesaba--- porque contesta un pedido textual
   del tutor y ocupa una subsección con una tabla. Se saca borrando `subsec:exp2c_curva`, que no
   está referenciada desde ningún otro lado.
5. **Los fracasos se escriben.** La subsección de alcance del 2e dice que el rasgo diseñado para
   el caso no aporta nada y que las familias que más mejoran no son las previstas. Es lo que
   sostiene el resto.

## Lo que queda del documento

- **`resultados_notas_ampliacion.tex` sigue sin integrar**: son 14 subsecciones del frente de
  notas, 77 KB, y ningún `\input` las trae. Integrarlas es decisión de Romina y Carlos, porque
  toca el orden del capítulo.
- **«Campaña» sigue sin definirse** en marco teórico, metodología ni introducción, y ahora la
  palabra aparece además en las dos secciones nuevas. Propuesta pendiente de respuesta: un
  párrafo en metodología, junto a la presentación de NapierOne.
- Las figuras del capítulo no se regeneraron: `generar_figuras_cap4.py` tiene cifras viejas
  hardcodeadas (ver el punto 1 de «Hallazgos que piden acción» más arriba). Ninguna de las
  secciones nuevas usa figuras, así que no bloquea.

---

---

# ✓ Síntesis del frente de archivos actualizada + «campaña» definida (2026-09-28)

**Corrección a la decisión 1 del bloque anterior.** Dejar 0,912 como canónica *sin tocar la
síntesis* la había dejado **contradiciendo a las secciones que la preceden**: seguía diciendo
«seis de las treinta familias» y «0,912» después de que el 2e redujera el residuo a cuatro y
subiera el número a 0,936. Romina lo notó («¿qué?»). Arreglado **agregando**, sin tocar el
párrafo existente:

1. La frase de la figura ahora dice que recoge «los cuatro **primeros** enfoques» y remite a las
   Secciones 2d y 2e. La figura no se regeneró (`generar_figuras_cap4.py` tiene cifras viejas).
2. Cuatro párrafos de cierre después de la conclusión existente: 2d como resultado sobre
   NapierOne con la limitación de campaña; 2e como mejora por contenido sin esa limitación; y
   **las dos cifras, cada una con su lugar**: 0,912 ± 0,002 es la configuración canónica que
   atravesó toda la validación y la que se compara con notas; 0,936 es el mejor resultado sin
   metadatos y fija hasta dónde llega el contenido. Residuo final: **cuatro familias** —
   NOTPETYA, JIGSAW, CRYPTOLOCKER, DARKSIDE.

**«Campaña» definida** en `metodologia.tex`, subsección nueva `subsec:met_campana` antes de
«Datos de Archivos Encriptados» (Romina: «poné donde quieras»). Familia = el programa; campaña =
un despliegue con su configuración. Tabla de qué cambia y qué no. La frase clave: en NapierOne
familia y campaña están **completamente confundidas** y ninguna medición sobre el conjunto las
separa — no se arregla con semillas ni con protocolo, requiere otro conjunto. Cierra con el
paralelo campaña ↔ plantilla en notas, con `
ef{subsec:neardups}`.

**Compila: 85 páginas, 0 errores, 0 referencias sin resolver.** Respaldos:
`resultados.tex.antes_2d2e`, `metodologia.tex.antes_campana`.

---

---

# ✅ DECISIÓN DE ROMINA: el 0,936 es el canónico del frente de archivos (2026-09-28)

Textual: «quiero que el mejor resultado sea el canónico». Se interpreta **mejor resultado =
Exp. 2e (0,936, sin metadatos)**, no el 2d (0,9998), porque el 2d arrastra la limitación de
campaña y no se puede presentar como propiedad del método. Si Romina quería el 0,9998, hay que
revisarlo.

## Qué cambió en el documento (cuatro ediciones, mínimas)

1. **Tabla comparativa final** (`tab:comparacion_final`): fila nueva «Exp. 2e: Bytes + rasgos
   estructurales — 93,6 % acc. / 0,936 macro-F1 — Random Forest (posicional + estructura)». La
   negrita pasa de la fila del 2c a esta. El 2c queda en la tabla, sin negrita.
2. **Frase notas-vs-archivos** de la comparación con herramientas: «la señal de contenido alcanza
   0,912 con los bytes y 0,936 al incorporar la estructura del archivo».
3. **Cierre de la síntesis**: el párrafo «dos cifras, cada una con su lugar» se reemplazó por uno
   que dice que **la cifra del frente es 0,936 / 0,9357 ± 0,0005**, y que el 0,912 es la base
   sobre la que se construyó y la que atravesó la validación. Residuo: cuatro familias.
4. **Cita del censo**: la frase sobre cifrado parcial decía «técnica cuyo propósito documentado
   es reducir el tiempo de cifrado» **sin cita y atribuyendo un propósito que la fuente no
   afirma** (el resumen de arXiv habla de eludir la detección). Ahora dice «una forma de cifrado
   intermitente, técnica documentada en la literatura reciente» y cita `ineza2025intermittent`.
   Entrada agregada a `bibliography.bib` con autores verificados en arxiv.org el 28-09:
   Ineza, Jackson, Niyonkuru, Kevil, Serwadda; v1 oct-2025, v3 ago-2026.

Sin cambios: resumen, introducción y conclusión no mencionaban el 0,912. La figura de progresión
sigue mostrando cuatro enfoques; el texto lo declara.

## ⚠ Lo que esta decisión deja descubierto, y hay que cerrar

El 0,912 tiene detrás: búsqueda anidada de hiperparámetros, **10 semillas** de dispersión,
**dejar-un-tipo-fuera** sobre siete tipos de documento, ablación de ventana y análisis por
familia sobre 10 semillas. **El 0,936 tiene 5 semillas, los hiperparámetros heredados del 2c
sin volver a buscar, y NO pasó por dejar-un-tipo-fuera.** El reporte por familia es de una
semilla.

Un jurado que pregunte «¿validaron la configuración canónica sobre tipos de documento no
vistos?» hoy recibe un **no**. Y hay un riesgo concreto: los rasgos de tamaño correlacionan con
el tipo de documento, así que la representación estructural podría degradarse **más** que los
bytes al dejar un tipo fuera. Si pasa eso, el 0,936 sigue siendo válido pero deja de ser mejor
que el 0,912 en la dimensión que la tesis usa para descartar la objeción de «aprende el
documento y no el ransomware».

**Propuesta: un job** —dejar-un-tipo-fuera sobre la representación bytes + estructura, mismos
siete pliegues que la Tabla de tipos del 2c, más 10 semillas de la configuración completa—
para que el canónico tenga el mismo respaldo que el número al que reemplaza. Hasta que corra,
la síntesis dice que el 0,936 «se agrega sin alterar» las comprobaciones del 0,912, que es
cierto pero no es lo mismo que haberlas pasado.

---

---

# 📌 PREREGISTRO — Exp. 2e-c: dejar-un-tipo-fuera sobre el canónico nuevo (2026-09-28)

Consecuencia directa de la decisión «el 0,936 es el canónico». El 0,912 pasó por
dejar-un-tipo-fuera (promedio 0,879 de exactitud, 29 familias); el 0,936 no. Y los rasgos de
tamaño correlacionan con el tipo de documento, así que hay un riesgo real de que la estructura
esté aprendiendo el documento y no el ransomware.

Réplica exacta de los pliegues del 2c (tipo desde el nombre, >= 200 archivos por tipo, un
ajuste por pliegue, RF 300/20/2/0,3 semilla 42, macro-F1 sobre las familias presentes), con DOS
representaciones por pliegue para que el delta sea pareado. Base: 30 familias, corpus
corregido. Código `2_codigo/exp2e_validacion_tipos.py` + `slurm/job_exp2e_tipos.sh`,
**commiteados antes de correr**, predicciones en el docstring:

| | Predicción |
|---|---|
| P1 | bytes solos promedian 0,86-0,90 de exactitud (publicado 0,879 sobre 29 fam.) |
| P2 | Δ macro-F1 promedio en [0,000; +0,025]: positivo pero menor que el +0,0246 de VC aleatoria |
| P3 | los dos pliegues con menor Δ son pdf y jpg |
| P4 | ningún pliegue con Δ macro-F1 por debajo de −0,02 |

**Lectura acordada de antemano:** P2 y P4 cumplen → el 0,936 pasa la misma prueba que el 0,912
y se escribe como validación del canónico. Δ promedio negativo → la mejora del 2e es en parte
«aprender el documento», **el canónico vuelve a ser el 0,912** y el 2e queda como mejora con
limitación declarada. P4 falla en un pliegue o dos → el 0,936 se sostiene con la salvedad del
tipo al lado.

Humo: corpus sintético de 3 familias × 3 tipos, con el tamaño base dependiente del tipo a
propósito; el script produjo el veredicto «los rasgos estructurales perjudican bajo tipo no
visto», que es exactamente el modo de fallo que está hecho para detectar.

---

---

# ✅✅ EXP. 2e-c CERRADO — el 0,936 pasa dejar-un-tipo-fuera igual que el 0,912 (2026-09-28)

Job 4079, 24 min, nodo c2. Registrado de lo pegado. Salidas en
`/scratch/ralfonzo/tesis/resultados_exp2e_tipos_job4079`. Base: 30 familias, corpus corregido,
500/familia, semilla de muestreo 0, RF 300/20/2/0,3 semilla 42, un ajuste por pliegue.

## Por pliegue (28 familias en la prueba en los siete; BLACKMATTER y CERBER solo entrenan)

| Tipo excluido | n | Bytes exact. / F1 | Bytes+estructura exact. / F1 | Δ exact. | **Δ F1** |
|---|---|---|---|---|---|
| doc | 2071 | 0,8957 / 0,8843 | 0,9367 / 0,9337 | +0,0410 | **+0,0494** |
| docx | 2007 | 0,8974 / 0,8860 | 0,9113 / 0,9023 | +0,0139 | +0,0163 |
| jpg | 2401 | 0,6951 / 0,7987 | 0,6997 / 0,8052 | +0,0046 | +0,0065 |
| pdf | 1977 | 0,8012 / 0,7746 | 0,8336 / 0,8131 | +0,0324 | +0,0385 |
| pptx | 2056 | 0,8838 / 0,8691 | 0,9027 / 0,8893 | +0,0189 | +0,0202 |
| xls | 1974 | 0,8931 / 0,8835 | 0,9179 / 0,9037 | +0,0248 | +0,0202 |
| xlsx | 2007 | 0,8949 / 0,8846 | 0,9118 / 0,9047 | +0,0169 | +0,0201 |
| **Promedio** | | **0,8516 / 0,8544** | **0,8734 / 0,8789** | +0,0218 | **+0,0245** |

**Δ pareado por pliegue, n = 7:** exactitud **+0,0218 [+0,0106; +0,0330]**, macro-F1
**+0,0245 [+0,0110; +0,0379]**, **7/7 pliegues a favor**, mínimo +0,0065 (jpg).

## ⭐ El resultado que importa: la mejora no depende del tipo de documento

| | macro-F1 con VC aleatoria | macro-F1 con tipo no visto | pérdida |
|---|---|---|---|
| solo bytes | 0,9114 | 0,8544 | **−0,0570** |
| bytes + estructura | 0,9359 | 0,8789 | **−0,0570** |

**Las dos representaciones pierden exactamente lo mismo** al dejar un tipo fuera, y el delta
entre ellas bajo tipo no visto (**+0,0245**) es idéntico al delta bajo validación cruzada
aleatoria (**+0,0246**). Los rasgos estructurales **no aprenden el tipo de documento**: si lo
hicieran, la pérdida de bytes+estructura sería mayor y el delta se achicaría. El riesgo
concreto que motivaba el experimento —que el tamaño correlacione con el tipo— **no se
materializa**.

**Consecuencia: el 0,936 queda validado con la misma prueba que validó al 0,912.** La
decisión de Romina de hacerlo canónico tiene ahora el respaldo que le faltaba.

## Las seis difíciles bajo tipo no visto (F1 medio sobre los siete pliegues)

| Familia | Bytes | Bytes+estructura | Δ |
|---|---|---|---|
| WASTEDLOCKER | 0,3929 | **0,6418** | **+0,2489** |
| JIGSAW | 0,3400 | 0,5154 | +0,1754 |
| NOTPETYA | 0,1190 | 0,2675 | +0,1486 |
| SUNCRYPT | 0,5810 | 0,6360 | +0,0551 |
| CRYPTOLOCKER | 0,2566 | 0,2799 | +0,0232 |
| DARKSIDE | 0,5063 | 0,5015 | −0,0047 |

Mismo patrón que bajo VC aleatoria: la mejora cae en las difíciles, y en el mismo orden
(WASTEDLOCKER, JIGSAW, NOTPETYA). DARKSIDE es la única plana. Notar que bajo tipo no visto
las seis están **mucho** más abajo que bajo VC aleatoria (NOTPETYA 0,119 contra 0,361): el
tipo no visto castiga sobre todo a las familias sin firma, en las dos representaciones.

## Veredicto del preregistro: dos cumplen, dos fallan

| | Predicción | Resultado | |
|---|---|---|---|
| P1 | bytes promedian 0,86-0,90 | **0,8516** | ❌ por 0,008 debajo del piso |
| P2 | Δ F1 en [0,000; +0,025] | **+0,0245** | ✅ pegado al techo |
| P3 | menor Δ en pdf y jpg | jpg y **docx**; pdf es el 2.º mayor | ❌ |
| P4 | ningún pliegue con Δ < −0,02 | mínimo **+0,0065** | ✅ ningún pliegue negativo |

**P1 — por qué falló, y por qué no preocupa.** El 0,879 publicado se midió sobre 29 familias
y **sin los `.pdf` cifrados de BADRABBIT y NOTPETYA**, de modo que su pliegue `pdf` tenía **25**
familias. Ahora **los siete pliegues tienen 28**: entró BLACKBASTA, y entraron NOTPETYA y
BADRABBIT al pliegue `pdf` con sus 310 archivos devueltos. NOTPETYA bajo tipo no visto da F1
0,119: sumarla a un pliegue baja el promedio. **El 0,8516 es la misma medición sobre una base
más completa y más dura**, no una degradación del método. Las dos cifras se declaran con su
base y no se comparan entre sí. *Lo que sí llama la atención y queda anotado:* en `jpg` la
exactitud cae a 0,6951 mientras el macro-F1 se queda en 0,7987; esa divergencia dice que una o
dos familias con muchos jpg fallan fuerte en ese pliegue. Está en `tipos_por_familia.csv`; no
cambia la decisión y no se persigue salvo que Romina quiera.

**P3 — por qué falló.** Razoné que pdf y jpg serían los peores por tener distribución de
tamaño «distinta». jpg sí es el peor (+0,0065), pero **pdf es el segundo mejor** (+0,0385). El
razonamiento sobre «tamaño típico del tipo» era el mismo que ya había fallado en el 2e-b —el
rasgo que manda es `tam_mod16`, el resto módulo 16, que no depende del tamaño típico de nada—.
Tercera vez que acierto el resultado y erro el mecanismo por pensar en la entropía o el tamaño
absoluto en vez de en el resto.

## Cifras para la tesis, con su base pegada

> Bajo dejar-un-tipo-fuera (siete tipos, 28 familias por pliegue, 30 familias en
> entrenamiento), la configuración canónica de bytes + rasgos estructurales promedia
> **0,8734 de exactitud y 0,8789 de macro-F1**, frente a 0,8516 y 0,8544 de solo bytes sobre
> los mismos pliegues: **Δ = +0,0245 [+0,0110; +0,0379], 7/7**. La pérdida respecto de la
> validación cruzada aleatoria es de **0,057 de macro-F1 en las dos representaciones**, y el
> delta entre ellas se conserva (+0,0245 contra +0,0246): la mejora de los rasgos estructurales
> es independiente del tipo de documento.

**Escrito el mismo día:** subsección `subsec:exp2e_tipos` («Generalización a tipos de documento
nunca vistos») dentro de `sec:exp2e`, con la tabla `tab:exp2e_tipos`; la síntesis del frente
dice ahora que el canónico superó la comprobación. **Compila: 87 páginas, 0 errores.** El
informe para Cappo (`6_notas_trabajo/informe_2026-09-27_frente_archivos_para_cappo.md`) quedó
actualizado: validación agregada a la sección 3, decisión 2 marcada como tomada por Romina y
validada, predicción fallida 4 sumada a la sección 7.

**Informe breve de cierre en PDF** (pedido de Romina, 28-09):
`6_notas_trabajo/informe_cierre_2026-09-28_frente_archivos.pdf` (+ `.tex`, mismo preámbulo que
el del 18-08). Seis secciones en dos páginas: la cifra del frente, qué se probó, qué se encontró
en los datos, qué no salió como se predijo, qué queda abierto, estado del documento.

**Commit `33688aa` → `8a011b3`:** el mensaje había quedado con código Python por un cruce de
heredocs. Romina autorizó reescribirlo («subí el commit, te di permiso»); `amend` +
`push --force-with-lease` hechos el 28-09, `develop` en sincronía con `origin`. El archivo
temporal con el mensaje se borró. **Recordatorio
para Claude:** el `/scratch` de Romina no se limpia —está en la memoria desde el 17-08— y esta
sesión lo usó igual como argumento de urgencia tres veces. No repetir.

---

---

# 🔍 VERIFICACIÓN DE LO QUE VA A CAPPO — diez afirmaciones corregidas (2026-09-28)

Pedido de Romina: «verificá y reverificá lo que le dirás al tutor, no podemos fallar», y «no es
el documento lo que me importa sino lo que dice». Se revisó el **contenido** de los dos informes
(`informe_2026-09-27_..._para_cappo.md` e `informe_cierre_2026-09-28_...pdf`) y de las secciones
del cap. 4 escritas hoy. Diez afirmaciones no eran ciertas tal como estaban:

| # | Decía | Lo verdadero |
|---|---|---|
| 1 | entropía de cabecera 0,88-6,46 «en los 988 archivos de CERBER» | la firma se contó en los 988; la entropía, en **6** |
| 2 | el rasgo diseñado fue «el menos importante de los 44» | no verificado (solo se vio top 12 y grupos); lo verificado: ablación +0,0001 |
| 3 | combinar con notas: +1,8 pts, 0,912 → 0,930 | cifras viejas de los dos frentes; con 0,9357 y notas P2bal 0,8123: **+2,5 → ~0,961** (oráculo +5,2; equilibrio 18,8 %) |
| 4 | «quedan dos mediciones de notas» | hechas y escritas desde el 26-09 |
| 5 | «fallaron cuatro predicciones» | **cinco** (faltaba P1 del 2e-c) |
| 6 | «0,9120 → 0,9123» como antes/después | corridas no pareadas: 0,9120, 0,9128 y 0,9123 son **indistinguibles** |
| 7 | «el residuo se reduce de seis a cuatro» | mezclaba 2c (10 semillas) con 2e (1 semilla); en la misma corrida es **5 → 4** |
| 8 | cifrado parcial «para reducir el tiempo» | la fuente (arXiv 2510.15133) dice **para eludir la detección** |
| 9 | «cuatro familias sin resolver» como hecho | **una semilla**; DARKSIDE en 0,73, en el límite |
| 10 | «40 archivos sin cifrar» | + **2 PDF de JIGSAW** con cabecera en claro, ahora declarados |

**Segunda pasada (la de «reverificá»):** búsqueda automática de cada frase falsa en los tres
documentos → 9 limpias; las 2 que aparecieron eran un falso positivo de la búsqueda (texto ya
corregido) y **texto original del 2c** («residuo genuino», en el commit, con «presumiblemente»),
que la regla protege y el 2e ya retoma. **Para el pulido final:** considerar matizar ese
«residuo genuino» del §4.5.8.

**Otro error propio corregido:** este ESTADO decía que f6-dfir era «la única vía» para
desbloquear D.1. **Falso:** f6-dfir no tiene ninguna familia en común con las 30 de NapierOne
(son de 2023-2025). No sirve para campaña ni para combinación.

**Además:** el PDF del informe daba texto roto al copiar («QuØ», «campaæa»). Arreglado
(`cmap` + `lmodern`, fuentes Type 1 con Unicode). **La tesis tiene el mismo problema**
(`campan~a` al extraer texto): afecta búsqueda, copiado y los detectores de plagio que leen el
PDF. **Arreglado el mismo día** (aprobado por Romina, hecho por un subagente y verificado): tres
líneas en `preambulo.tex` — `\usepackage{cmap}`, `\usepackage[T1]{fontenc}`, `\usepackage{lmodern}`;
respaldo en `preambulo.tex.antes_fuentes`. 89 páginas, 0 errores; `main.aux` idéntico byte a byte, así
que ninguna sección, figura, tabla ni cita cambió de página. Capa de texto: «campaña» 0 → 45,
«configuración» 0 → 29, tildes sueltos 181 → 0. Cambio visible menor: ~80 líneas recortadas en 20
páginas y 3 cortes de página corridos (págs. 20/21, 25, 26/27); las comillas «» y el guion bajo ahora
son caracteres reales. **Quedan 4 acentos sueltos** en dos fórmulas («mín»/«máx» en modo matemático):
arreglarlos toca la configuración de babel, fuera de alcance.

Compila: tesis 88 páginas, 0 errores, 0 referencias sin resolver; informe 2 páginas.

---

---

# ⚠ CORRECCIÓN: el canónico NO era «el 0,936». Criterio real de Romina + PREREGISTRO Exp. 2f (2026-09-28)

**Error de Claude, no decisión de Romina.** Romina dijo «quiero que el mejor resultado sea el
canónico». Claude **interpretó** que era el 2e (0,936, sin metadatos) y no el 2d (0,9998), y lo
avisó como interpretación corregible; pero después escribió «la regla que elegiste», atribuyéndole
a Romina una decisión propia. El bloque de arriba titulado «DECISIÓN DE ROMINA: el 0,936 es el
canónico» **queda superado por este**.

**El criterio de Romina, textual:** «yo no descarté nunca ninguna técnica, todo sirve para mejorar
el resultado obviamente, no son técnicas separadas, son una secuencia hasta encontrar la forma más
óptima de clasificar». Consecuencia: **el canónico es el sistema que apila todas las capas**
(bytes + estructura + forma del nombre), y la limitación de campaña **se declara, no excluye**. Es
coherente con la tesis: el 0,912 de solo bytes ya está declarado como medición de «esa campaña».

**Esa combinación nunca se había medido** (2d midió bytes+nombre, 2e bytes+estructura). Script
`2_codigo/exp2f_sistema_completo.py` + `slurm/job_exp2f.sh`, **commiteados antes de correr**:
4 columnas (bytes · +estructura · +forma · +extensión literal), F1 por familia **en las 5 semillas**
(resuelve que «cuatro familias sin resolver» dependiera de una), y dejar-un-tipo-fuera.

| | Predicción |
|---|---|
| F1 | sistema (3) macro-F1 ≥ 0,9990 |
| F2 | la extensión literal no suma a (3): \|Δ\| ≤ 0,0010 |
| F3 | las 30 familias con F1 medio ≥ 0,99 en (3), incluidas las cuatro difíciles |
| F4 | bajo tipo no visto, Δ (3)−(2) > 0 en los siete pliegues (si falla: la forma del nombre codifica en parte el tipo del documento, y se declara) |

**Queda pendiente tras el 2f** —no antes, para no escribir dos veces—: actualizar la síntesis del
cap. 4 (hoy dice «la cifra del frente es 0,936»), la sección del 2d (hoy «resultado sobre NapierOne,
no propiedad del método»), la tabla comparativa final y los dos informes a Cappo.

---

---

# 🔬 DIAGNÓSTICO DE LAS DIFÍCILES — job 4082: cifrado determinista detectado, hipótesis de DARKSIDE descartada (2026-09-28)

Sin aprendizaje, mirando los bytes de los 29.948 archivos (96 s). Registrado de lo pegado.
Salidas en `/scratch/ralfonzo/tesis/resultados_diagnostico_dificiles_job4082`.

## (f) Los 310 PDF devueltos: VERIFICADOS uno por uno

BADRABBIT 143 · **0** con `%PDF` en claro. NOTPETYA 167 · **0**. JIGSAW 2 · **2** en claro.
La afirmación «310 muestras cifradas» —que se apoyaba en dos por familia— queda **verificada
exhaustivamente**. Y los dos PDF de JIGSAW están confirmados en claro (ya declarados).

## (a') ⭐ CRYPTOLOCKER cifra de forma DETERMINISTA

| Familia | doc | xls | jpg | pdf | xlsx | docx | pptx |
|---|---|---|---|---|---|---|---|
| **CRYPTOLOCKER** | **143 / 1** | **143 / 1** | 143 / 13 | 143 / 17 | 142 / 46 | 143 / 117 | 143 / 140 |
| NOTPETYA | 149 / 35 | 152 / 105 | — | 167 / 32 | 166 / 70 | 167 / 136 | 167 / 161 |
| JIGSAW, DARKSIDE, WASTEDLOCKER, SUNCRYPT | un prefijo distinto por archivo en **todos** los tipos |

(archivos / prefijos de 16 bytes distintos)

**Los 143 `.doc` de CRYPTOLOCKER empiezan con los mismos 16 bytes cifrados, y los 143 `.xls`
también.** Es la huella de un cifrado **con clave fija y sin vector de inicialización por
archivo**: dos documentos con el mismo comienzo en claro dan el mismo comienzo cifrado. Todos los
OLE (doc, xls) empiezan igual en claro → un solo prefijo. Los OOXML (docx, pptx, xlsx) llevan fecha
y CRC en la cabecera ZIP, así que su comienzo en claro varía → prefijos distintos. **El patrón
calza exacto con esa explicación.** NOTPETYA muestra lo mismo, parcial (doc: 35 prefijos para 149
archivos, el más común cubre el 77 %).

**Confirmación independiente en la tabla global:** CRYPTOLOCKER, BADRABBIT, MEDUZALOCKER y
RANSOMEXX tienen **exactamente** 332 prefijos distintos y el más común cubre **0,286**, las cuatro,
y **ninguno** de sus prefijos aparece en otra familia (`pre_en_otra_fam` = 0,000). Misma estructura
~~de clases que el corpus base en claro (CERBER: 388 / 0,272)~~ — **comparación FALSA, corregida el
mismo día: CERBER da 388 y estas 332, no coinciden.** Lo que sí sostiene la conclusión: cuatro familias
con estructura **idéntica entre sí** y **cero** colisiones con otras, que es lo esperable de un cifrado
determinista aplicado al mismo conjunto base con una clave distinta por familia.
BADRABBIT, MEDUZALOCKER y RANSOMEXX lo compensan con un **sufijo constante** (suf_top 0,965 / 1,000
/ 1,000); **CRYPTOLOCKER no tiene sufijo fijo** (997 sufijos distintos) y por eso depende solo del
prefijo, que no sirve para los OOXML.

**⚠ Esto refina una afirmación publicada.** El Exp. 2b (§4.4) dice que CRYPTOLOCKER «solo poseía
extensión propia», sin marca en el contenido. **Sí tiene una firma de contenido, pero por tipo de
documento**, y el criterio del 2b —un prefijo común al 90 % de TODA la familia— no podía verla por
construcción. No es una marca que escribe el programa sino una consecuencia de cómo cifra. Y depende
de la clave, que es de la campaña: otra campaña de CRYPTOLOCKER daría otros prefijos (la *propiedad*
de repetirlos sí sería del código). Va a la tesis como agregado, no reescribiendo el 2b.

## (e) ❌ DARKSIDE NO cifra de forma intermitente en este conjunto

Hipótesis previa (desde el arXiv 2510.15133, que la lista entre las intermitentes): DARKSIDE
dejaría bloques en claro en el medio. **Dato: solo el 5,8 % de sus archivos tiene algún bloque de
baja entropía, 0,6 % de los bloques en promedio.** Descartada. Coherente con que el modo
intermitente se aplica a archivos grandes y los de NapierOne son chicos. **Cuarta vez que la
hipótesis de mecanismo falla; esta vez se supo antes de construir nada.**

## Por qué se confunden, ahora medido y no supuesto

| Par | Qué comparten en lo que mide este diagnóstico |
|---|---|
| **DARKSIDE ↔ WASTEDLOCKER** | prefijo aleatorio, sufijo aleatorio, cifrado total, resto mód. 16 igual a la tasa base del corpus (~0,34). Indistinguibles en todo lo medido acá |
| **JIGSAW ↔ CRYPTOLOCKER (en OOXML)** | prefijo sin repetición, sufijo aleatorio, cifrado total, tamaño **siempre** múltiplo de 16 (0,998 y 1,000) |

Matiz obligatorio: WASTEDLOCKER sí mejoró a 0,83 con los rasgos del 2e, así que algo la separa de
DARKSIDE por contenido; está en los rasgos de tamaño que este diagnóstico no miró (mód. 512, mód.
4096, tamaño absoluto). «Indistinguibles» vale **para lo que se midió acá**, no en general.

## Una curiosidad, sin interpretar todavía

DHARMA y PHOBOS: **864 prefijos distintos, el más común cubre 0,137, y ese 13,7 % coincide entre
las dos**. Son parientes de código conocidos (Phobos deriva de Dharma/CrySiS). Si se confirma, el
contenido cifrado revela linaje, en paralelo a los moldes de nota compartidos del frente de notas.
**Es observación, no hallazgo**: habría que mirar cuál es ese prefijo antes de afirmar nada.
**No va a la tesis** mientras no se verifique.

**Verificado el mismo día (comando sobre `por_archivo.csv` del job 4082, pegado por Romina):** el
prefijo compartido es **dieciséis bytes en cero** (`0000…0000`), en **137 archivos de cada una**, con
**idéntica distribución por tipo** en las dos (pptx 81 · pdf 25 · docx 12 · xls 9 · xlsx 7 · doc 2 ·
jpg 1) — o sea, **los mismos documentos de origen**. Es el **único** prefijo que comparten. Y CERBER,
que deja la cabecera en claro, no lo tiene (su colisión con otras familias es 1,1 %), así que los
ceros **no vienen del original**: los producen DHARMA y PHOBOS al procesar esos archivos.

**Lectura honesta:** es un **comportamiento compartido** —las dos hacen lo mismo con los mismos
archivos—, coherente con el código común que se les atribuye. **No es una firma de linaje en
sentido fuerte**: el valor (ceros) no es discriminante, y no se sabe qué mecanismo lo produce (la
mayoría son pptx, que tienden a ser grandes; podría ser un tratamiento distinto de archivos
grandes, pero **no está medido** y no se afirma). **Sigue fuera de la tesis** salvo que Romina lo
quiera como observación; en ese caso, con esta redacción y sin mecanismo.

## Llevado a la tesis el mismo día (Romina: «corregí lo erróneo si hace falta»)

- **Dos frases corregidas** del cap. 4, las dos falsas para CRYPTOLOCKER: en §4.5.8 «no hay marca
  que aprender» → «no hay marca **añadida** que aprender, aunque CRYPTOLOCKER deja una regularidad de
  otro tipo»; en el análisis por familia «la única marca que esas familias dejan» → «la única marca
  que esas familias **escriben explícitamente**», más la regularidad de CRYPTOLOCKER. Las dos
  remiten a la subsección nueva. El Exp. 2b **no se tocó**: su clasificación es correcta bajo su
  criterio, y la subsección nueva la precisa.
- **Subsección nueva** `subsec:exp2c_determinista` («Qué distingue por contenido a las familias
  difíciles»), antes de la Limitación del 2c, con la tabla `tab:prefijos_tipo`. Cifrado
  determinista de CRYPTOLOCKER, DARKSIDE no intermitente, los dos pares que se confunden.
- **Cita verificada en la fuente** antes de usarla: la tabla I de arXiv 2510.15133 se titula
  «Modes of Intermittent Encryption Used by Ransomware Families» y DARKSIDE (2020) figura en ella.
  (Un primer resumen automático la había leído como «cifrado completo»; se revisó la fila.)

---

### CURVA DE APRENDIZAJE BAJO P2bal — el corte en 3 se CONFIRMA (2026-09-28)

`curva_aprendizaje_notas.py --solo-p2bal`, preregistro en el docstring commiteado antes de correr
(`b4d17e9`, 14:57:54). Log `_log_curva_p2bal_149.txt`, salidas en `resultados_curva_p2bal_149/`.
50 repeticiones.

**Origen de la duda.** Al descubrirse que el reparto de P2 dejaba 3,85 familias por pliegue sin
entrenamiento, surgió si la conclusión «el último tope que aporta son 3 por familia» dependía de
ese defecto. **Verificado ANTES de correr, en `resultados_curva_149/b1_curva_por_repeticion.csv`
(30fam, plantillas): esa conclusión sale de P2ret, y P2ret NO tiene el defecto** — sus familias sin
entrenamiento son **2,00 constantes en todo k**, y son exactamente BADRABBIT y CRYPTOLOCKER. P2
marca 3,85 en todo k; P1, 0,00. Esta corrida confirma, no re-deriva.

**Tope por NOTAS por familia (el eje con alcance):**

| k | macro-F1 | Δ pareado respecto del anterior | ¿significativo? |
|---|---|---|---|
| 1 | 0,5757 | — | — |
| 2 | 0,6503 | +0,0747 [+0,063; +0,087] | **sí** |
| 3 | **0,6610** | +0,0107 [+0,004; +0,017] | **sí** |
| 4 | 0,6607 | −0,0004 [−0,005; +0,004] | no |
| 6 | 0,6602 | −0,0005 | no |
| 8 | 0,6577 | −0,0025 | no |
| todo | 0,6551 | −0,0026 | no |

**El corte queda en 3, idéntico al de P2ret.** La conclusión escrita en el capítulo se sostiene bajo
el reparto corregido.

**Tope por PLANTILLAS: sin alcance, como se preregistró (E3).** 1→2 +0,0840 (sí), 2→3 +0,0009 (no),
3→4 **−0,0057 (significativo, NEGATIVO)**. Con 2 pliegues el entrenamiento tiene ~1,8 plantillas por
familia: en k=3 solo **1 familia** queda bajo el tope y en k=4, ninguna. El eje se agota
por construcción, no por saturación del aprendizaje. **Quien cite esta curva como techo de
aprendizaje la cita mal**; la curva informativa por plantillas sigue siendo P2ret.

**CORRECCIÓN (2026-09-28): la predicción P2 fallada NO se debe a los grupos mixtos.** La sesión
hermana la atribuyó a que «una familia puede entrar al entrenamiento a través del grupo mixto de
otra». **Verificado y es falso.** Medición directa sobre los 100 pliegues de P2bal (50 semillas × 2):

- Las **únicas** familias que alguna vez faltan del entrenamiento son **BADRABBIT y CRYPTOLOCKER**,
  cada una en **exactamente 50 de 100** pliegues. Ninguna otra falta nunca.
- **Ninguna de las dos pertenece a un grupo mixto.** Los mixtos son el 6 (BLACKBASTA 1 + CONTI 1) y
  el 53 (DHARMA 11 + PHOBOS 1), y esas cuatro familias tienen 4, 3, 6 y 4 plantillas, así que
  **siempre** tienen material de entrenamiento.

La causa es aritmética y ya estaba bien explicada: una familia de **plantilla única** cae en un solo
pliegue, así que está ausente del entrenamiento en **uno de los dos** pliegues, no en los dos.
2 familias × ½ = **1,00 por pliegue**. Mi predicción de 2,00 estaba mal calculada; el protocolo
siempre se comportó como debía.

**Consecuencia para el capítulo de metodología:** el hilo de los grupos mixtos tiene **tres**
manifestaciones verificadas, no cuatro — el conteo 49,5 / 50,5, el error de estrato en el bootstrap
y el parentesco DHARMA-PHOBOS. La predicción P2 fallada **no es una de ellas**.

**Aclaración de unidades (planteada por la sesión hermana, resuelta):** `protocolo_p2bal.py` reporta
**49,5** plantillas de entrenamiento por pliegue y la curva reporta **50,5**. No es un error de
ninguno de los dos: son unidades distintas. Hay **99 plantillas distintas** pero **101 pares
(familia, plantilla)**, porque los grupos **6 y 53 mezclan dos familias** y la curva los contabiliza
una vez por familia, a propósito (ver el comentario de `_por_familia_grupo`). 99/2 = 49,5 y
101/2 = 50,5. Al citar, decir la unidad.

**Veredicto del preregistro: 4 de 5 se cumplen.**
- E1 puerta ✔ — k=todo da 0,655106 contra 0,6551 de `protocolo_p2bal.py`, diferencia 5,8·10⁻⁶ (el
  redondeo a 4 decimales de la referencia). El cableado es el mismo al sexto decimal.
- E2 ✔ — P2bal por encima de P2 en todos los k: 0,576/0,660/0,661/0,655 contra
  0,436/0,471/0,468/0,468.
- **E3 ✘ (parcial)** — predije «3→4 no significativo» y por plantillas salió **significativo pero
  negativo**. La dirección refuerza la conclusión (más plantillas no solo no aportan: restan un
  poco), pero **la predicción literal falló y se reporta como fallada**. Por notas sí se cumple.
- E4 ✔ — el corte de P2ret no se mueve; el eje de notas bajo P2bal lo reproduce.
- E5 ✔ — familias sin entrenamiento 1,00 constante en todo k.

### IC DE P2bal POR REMUESTREO DE PLANTILLAS (2026-09-28) — la afirmación resiste el test duro

`bootstrap_plantilla_p2bal.py`, preregistro commiteado antes de correr (`c5d1018`); la convención
sin sesgo se agregó DESPUÉS de ver el resultado y está declarada como post hoc (`eef472f`).
Log `_log_bootstrap_plantilla_p2bal.txt`, salidas en `resultados_bootstrap_plantilla_p2bal_149/`.

**Por qué.** El IC publicado es **entre semillas**: dice cuánto se mueve la cifra al cambiar la
partición, no cuánto se movería con otro corpus. Para lo segundo hay que remuestrear, y la unidad
independiente es la **plantilla**, no la nota: las notas de una plantilla son casi copias.
Remuestrear notas finge un tamaño de muestra que no existe y **estrecha** el intervalo. Como P2bal
no es determinista, el remuestreo es de **dos niveles**: se sortea una semilla de las 50 y después
se remuestrean plantillas con reposición, entrando todas las notas de cada plantilla sorteada.

| Capa | Punto | IC entre semillas | **(c) ESTRATIFICADO por familia — PRINCIPAL** | (a) libre, labels=30 | (b) libre, labels presentes |
|---|---|---|---|---|---|
| texto | 0,6551 | [0,6454; 0,6648] | **[0,5654; 0,7446]** | [0,5066; 0,7182] | [0,5508; 0,7682] |
| cascada | 0,7417 | [0,7328; 0,7505] | **[0,6585; 0,8187]** | [0,5872; 0,7935] | [0,6437; 0,8508] |

**(c) es el que corresponde al diseño, y se adopta como principal.** Planteado por la sesión
hermana, verificado acá con predicciones propias. El remuestreo **libre** trata al conjunto de
familias como aleatorio, o sea admite réplicas donde una familia no existe. Pero **las 30 familias
no son una muestra**: están fijadas por el núcleo canónico de NapierOne, que es decisión de diseño
del trabajo. Lo muestral es **qué plantillas se consiguieron de cada familia**. El estratificado
remuestrea las plantillas **dentro** de cada familia conservando su cantidad: ninguna familia
desaparece, el sesgo **no se genera** (−0,001 y −0,001 contra −0,043 y −0,047) y conserva
`labels=30` fijo, sin pagar el denominador variable de (b). La unidad es el par (familia,
plantilla), porque los grupos 6 y 53 mezclan dos familias.

Es más **angosto** (0,160 contra 0,206 en la cascada) y eso hay que declararlo: no es una elección
de conveniencia, es que **no incluye la variación de «qué familias hay en el corpus»**, que en este
diseño no es una fuente de incertidumbre real. Se reportan las dos y se dice cuál pregunta contesta
cada una.

**Validación cruzada entre sesiones — la más fuerte que tiene el frente de notas.** Las dos sesiones
implementaron el remuestreo por separado y recalcularon las predicciones de forma independiente.
Tras corregir una diferencia, **las tres convenciones coinciden a 4 decimales**. La diferencia
inicial en (c) —la sesión hermana daba [0,5580; 0,7313] con sesgo −0,0120 contra [0,5654; 0,7446]
con sesgo −0,0011— venía de estratificar por **plantilla entera** en vez de por el par (familia,
plantilla), y es otra consecuencia de los **dos grupos mixtos**: el 6 con BLACKBASTA + CONTI y el 53
con DHARMA (11 notas) + PHOBOS (1), **14 notas, el 9,4 % del corpus**. Con la plantilla entera,
cuando DHARMA sorteaba el grupo 53 entraban también las notas de PHOBOS, y viceversa: notas ajenas
dentro del estrato y tamaño de familia variable entre réplicas. Con el par, no.

El intervalo correcto es casi **diez veces más ancho** que el de semillas. **Las dos capas mantienen
el límite inferior por encima de 0,50 en las tres variantes**, así que la frase «supera el umbral
con el intervalo entero» sobrevive al test más exigente. Con el principal, la cascada tiene **0,159
de margen** sobre el umbral y el texto **0,065**: la cascada sigue siendo la que aguanta, aunque con
el estratificado el texto ya **no queda pegado** al umbral como sugería la variante libre.

**Veredicto: 4 de 5, con F5 fallada y explicada.** F5 predecía sesgo ≤ 0,02 y dio −0,041 y −0,046.
**No es error de cálculo:** con `labels=30` fijo, una remuestra que no incluye ninguna plantilla de
una familia le asigna F1 = 0 y ese cero entra al macro; **una remuestra pierde en promedio 2,15
familias de 30**. Por eso el macro remuestreado está sesgado hacia abajo *por construcción*. Es el
mismo efecto que la revisión del 17-09 vio en LOGO. Con la convención de etiquetas presentes el
sesgo cae a +0,007 y +0,008. **Se reportan las dos y se cita la conservadora.**

**Al citar:** el IC por plantillas habla del **corpus**; el IC entre semillas habla de la
**partición**. No son intercambiables y no se mezclan en la misma frase.

### PARA EL CAPÍTULO DE METODOLOGÍA: los cuatro errores del 2026-09-28 y qué atrapa a cada uno

Los cuatro se detectaron y corrigieron el mismo día, entre dos sesiones que se revisaron
mutuamente. Ninguno lo encontró quien lo cometió. Ordenados por dificultad de detección:

| # | Error | Qué lo delató |
|---|---|---|
| 1 | Remapeo de etiquetas aplicado a un array y no al otro (acierto por linaje) | Que la cifra fuera **físicamente imposible**: fusionar clases no puede bajar la exactitud |
| 2 | Sesgo del remuestreo con `labels=30` | Una **predicción preregistrada sobre la coherencia del cálculo** (F5) que falló |
| 3 | Remuestreo libre de plantillas: contestaba «¿y si el corpus tuviera otras familias?», que el diseño no se pregunta | **Nada automático.** Solo discutir cuál era el estimando |
| 4 | Atribuir la predicción P2 fallada a los grupos mixtos | **Nada automático.** Solo ir a medir la causa |

**La distinción que vale para la tesis (formulada por la sesión hermana):** los errores 1 y 2 son de
**cálculo** y los atrapan controles automáticos —puertas de entrada, de salida, chequeos de
imposibilidad—. Los errores 3 y 4 son de **explicación**: un número bien calculado con una causa o
un estimando mal atribuidos. **Ningún control automático los detecta**, porque no hay nada que
falle. Y una sección de metodología está hecha casi toda de causas atribuidas.

De ahí las dos reglas que conviene dejar escritas:
1. Todo preregistro debe incluir **al menos una predicción sobre la coherencia interna del
   cálculo**, no solo sobre el resultado sustantivo (atrapa el tipo 2).
2. **Verificar, no recordar, se aplica también —y sobre todo— al armar la narrativa.** El error 4
   se cometió encajando una pieza que faltaba en una explicación que ya sonaba bien. Es cuando más
   tienta saltear la medición. La revisión entre pares es el único control que queda para los tipos
   3 y 4.

---

# ✅⚠ EXP. 2f CERRADO — el sistema completo da 0,9998 y resuelve las cuatro difíciles… pero F4 FALLA: la forma del nombre se desploma en jpg (2026-09-28)

Job 4083, 3 h 25 min, nodo c2. Registrado de lo pegado. Reemplaza al parcial de arriba.
Salidas en `/scratch/ralfonzo/tesis/resultados_exp2f_job4083`.

## (A) Validación cruzada — 5 semillas, 15.000 archivos, 30 familias

| Columna | Exactitud | macro-F1 |
|---|---|---|
| (1) bytes | 0,9123 ± 0,0003 | 0,9114 ± 0,0004 |
| (2) + estructura | 0,9357 ± 0,0005 | 0,9359 ± 0,0004 |
| **(3) + estructura + forma del nombre** | **0,9998 ± 0,0001** | **0,9998 ± 0,0001** |
| (4) + extensión literal | 0,9998 ± 0,0001 | 0,9998 ± 0,0001 |

Δ pareado (macro-F1): (3)−(2) **+0,0639** [+0,0634; +0,0644] 5/5 · (3)−(1) **+0,0884** [+0,0879;
+0,0890] 5/5 · (4)−(3) **+0,0000** [−0,0001; +0,0001] → la extensión literal no suma: queda
afuera por los datos.

**F1 por familia, media sobre las 5 semillas — las 30 quedan ≥ 0,99 en el sistema (3):**

| Familia | bytes | + estructura | **+ estructura + forma** |
|---|---|---|---|
| NOTPETYA | 0,3850 | 0,4811 | **0,9978** ± 0,0016 |
| JIGSAW | 0,4338 | 0,5739 | **0,9990** ± 0,0007 |
| CRYPTOLOCKER | 0,6128 | 0,6576 | **1,0000** |
| DARKSIDE | 0,5981 | **0,7515** | **1,0000** |
| WASTEDLOCKER | 0,6266 | 0,8352 | 1,0000 |
| SUNCRYPT | 0,7590 | 0,8296 | 1,0000 |

**Corrección que trae el promedio de 5 semillas:** con bytes + estructura (2e), DARKSIDE da
**0,7515**, por encima de 0,75. En la semilla 0 daba 0,7327. O sea que bajo solo contenido
quedan **tres** familias por debajo de 0,75, no cuatro — el «en el límite» que se había
agregado era exactamente esto. La síntesis del cap. 4 hay que actualizarla.

## (B) ❌ Dejar-un-tipo-fuera: la forma del nombre COLAPSA en jpg

| Tipo excluido | + estructura | + estructura + forma | Δ |
|---|---|---|---|
| doc | 0,9337 | 0,9943 | +0,0606 |
| docx | 0,9023 | 1,0000 | +0,0977 |
| **jpg** | **0,8052** | **0,2167** | **−0,5885** |
| pdf | 0,8131 | 0,9791 | +0,1660 |
| pptx | 0,8893 | 1,0000 | +0,1107 |
| xls | 0,9037 | 0,9995 | +0,0958 |
| xlsx | 0,9047 | 1,0000 | +0,0953 |
| Promedio | 0,8789 | 0,8842 | +0,0054 [−0,2386; +0,2493], 6/7 |

**Veredicto: F1, F2 y F3 cumplen; F4 FALLA.** Lectura acordada de antemano: «si falla, la forma
del nombre codifica en parte el tipo del documento de origen, y hay que declararlo».

**Y no es «en parte»: en jpg es catastrófico.** Con el tipo visto en entrenamiento, el nombre
lleva a 0,9998; con jpg nunca visto, **empeora 59 puntos** respecto de no usarlo. Los seis tipos
de ofimática mejoran fuerte (+0,06 a +0,17); jpg se hunde.

**Causa probable, a verificar antes de construir encima:** `forma_del_nombre()` mira el nombre
**entero**, incluida la parte que viene del corpus base de NapierOne. Según el docstring de
`tipo_documento()` —escrito por quien miró los archivos—, los jpg se llaman
`0001-jpg-fromweb.jpg.<ext>` y los documentos `0001-doc.doc.<ext>`: la base de los jpg es más
larga y tiene otra composición. Con jpg fuera del entrenamiento, el modelo nunca vio nombres de
esa forma y los manda a las familias equivocadas. Si es así, los rasgos de forma aprendieron
**cómo nombró NapierOne sus archivos**, además de cómo renombra cada familia.

## Qué significa para lo que va a Cappo

- El **0,9998 es real bajo validación cruzada** y resuelve las cuatro difíciles: se reporta.
- **No es robusto a un tipo de documento no visto**: la prueba que la tesis usa para descartar
  «aprende el documento y no el ransomware» lo tumba en jpg. Presentarlo sin esto sería decirle al
  tutor algo más fuerte que la evidencia.
- **Vía de arreglo, propuesta a Romina:** rasgos de nombre calculados **solo sobre la extensión
  final** —lo que agrega el ransomware— y no sobre la base heredada de NapierOne. Medir CV + tipos.
  Si mantiene ≥ 0,99 y no colapsa en jpg, ese es el sistema robusto.

---

---

# 📌 PREREGISTRO — Exp. 2g: nombre robusto (solo la extensión final) (2026-09-28)

**Causa del colapso del 2f, VERIFICADA MIRANDO** (`ls AVOSLOCKER-small`, pegado por Romina):
`0001-doc.doc.avos2` · `0001-pdf.pdf.avos2` · **`0001-jpg-fromweb.jpg.avos2`**. La base del nombre
es herencia del corpus de NapierOne y en los jpg lleva «-fromweb» (base de 20 caracteres contra
12). `forma_del_nombre()` mira el nombre entero y aprendió esa herencia. Quinta hipótesis de
mecanismo de la semana, y la primera confirmada **antes** de construir encima.

**Arreglo:** `forma_de_la_extension()` — 14 rasgos solo sobre la extensión final (lo que agrega el
ransomware) + puntos del nombre + marca «la extensión es de un tipo de documento» (identifica a
las que no renombran). Script `2_codigo/exp2g_nombre_robusto.py` + `slurm/job_exp2g.sh`,
**commiteados antes de correr.**

| | Predicción |
|---|---|
| G1 | (5) bytes + estructura + extensión ≥ 0,995 de macro-F1 en CV |
| G2 | pliegue jpg sin colapso: Δ (5)−(2) ≥ −0,02 |
| G3 | Δ (5)−(2) > 0 en los siete pliegues de tipo |
| G4 | NOTPETYA, JIGSAW, CRYPTOLOCKER, DARKSIDE ≥ 0,95 con (5) |

**Lectura acordada:** G1-G3 cumplen → (5) es el sistema completo y robusto, **cifra canónica
del frente**. G2 falla → tampoco la extensión es robusta a tipos no vistos; se declara. G4 falla
→ hay familias con extensiones de la misma forma; se dice cuáles.

---

### ❌ CAPA DE EXTENSIÓN DE CIFRADO EN LAS NOTAS — no aporta, y la causa es el corpus (2026-09-28)

`2_codigo/capa_extension_cifrado.py`, preregistro E1–E8 en el docstring, **commiteado antes de
correr** (`b222884`). Log `4_resultados/_log_capa_extension.txt`, salidas en
`4_resultados/resultados_capa_extension_149/`. **149 notas, 30 familias, P2bal, 50 semillas.**

**La idea.** La extensión que el ransomware le pone a los archivos cifrados (`.locked`, `.GDCB`)
casi siempre está escrita en el texto de la nota, y hoy solo entra diluida en el TF-IDF. Se la
agregó como **una clave exacta más** al mismo diccionario de los IOCs, con el mismo filtro de
genéricos y la misma regla de unanimidad. Nada más de la cascada cambió.

**El patrón de extracción, verificado a mano ANTES de medir.** Barrer todo `.token` da 135 tokens
y ~97 % basura (TLD, rutas de URL, `.hta` de CERBER 210 veces, IP de CLOP, «AUTRE.ALORS»). El
patrón final: desofuscar (`[.]`, `hxxp`), borrar **URL espaciada → URL → ONION → EMAIL → BTC →
CLAVE → ID** (el orden importa: con ONION primero, `http://x.onion.cab/` deja `.cab` suelto;
TESLACRYPT escribe `https://en .wikipedia. org/` y dejaba `.wikipedia`), y cuatro rutas de
captura (ancla «extension» con y sin punto · token suelto · doble extensión · cadena
`.id-X.[mail].fam`). **Resultado: 8 extensiones distintas en 12 de 149 notas (8,05 %), CERO
capturas basura, CERO extensiones de documento coladas.**

| extensión | familia | notas | plantillas |
|---|---|---|---|
| `.gacmw` | GANDCRAB | 4 | 72, 75 |
| `.gdcb` | GANDCRAB | 2 | 73, 74 |
| `.ibkfz` · `.krab` · `.rfncw` | GANDCRAB | 1 c/u | 75 |
| `.eebf08` | NETWALKER | 1 | 107 |
| `.lgzcfcr` | SODINOKIBI | 1 | 129 |
| `.sz40` | LORENZ | 1 | 95 |

**Las tres columnas.** Cobertura de la capa **0,0250** [0,0194; 0,0306] (186 decisiones sobre
50 × 149) · acierto donde aplica **1,0000** (186/186) · efecto sobre el macro-F1 **+0,0000**
[+0,0000; +0,0000], **0/50 semillas positivas**. Δ exactitud idéntico. La capa **cambia la
respuesta en 0 de 7.450 decisiones**: todo lo que resuelve, la cascada ya lo resolvía.
Claves `[EXT]` en el diccionario: 4,62 por pliegue; **descartadas por el filtro de genéricos: 0**
(ninguna extensión se repite entre familias).

**Por qué no aporta — y es el corpus, no el método.** Las fuentes publican las notas saneadas
(`[snip]`, `${EXTENSION}` en BLACKCAT, `{EXT}` en SODINOKIBI), así que solo 12 notas traen la
extensión. Peor: como el corte de P2bal es **por plantilla**, la regla solo puede disparar si el
**mismo valor** está en otra plantilla de entrenamiento, y eso solo pasa con `.gacmw` (grupos
72|75) y `.gdcb` (73|74), las dos de GANDCRAB. **Techo duro de cobertura: 6/149 = 0,0403.** Y
GANDCRAB, LORENZ y NETWALKER ya están en recall 1,0000 sin la capa.

**Techo oráculo** (regalarle la respuesta en las 12 notas que sí traen extensión): macro-F1
0,7417 → **0,7454**, Δ **+0,0038** [+0,0023; +0,0052]. Ese es el máximo que la técnica podría dar
sobre este corpus.

**Veredicto del preregistro: E1–E7 CUMPLEN las siete.** Puerta de entrada reproducida exacta
(macro-F1 0,7417 / exactitud 0,8123). Resultado negativo informativo: **no se adopta**, y se
reporta como **limitación del corpus** con el techo pegado. La vía a explorar si alguna vez
interesa no es mejorar el patrón (no captura basura) sino **generalizar la clave** más allá del
valor exacto, o conseguir notas sin sanear.

**Respaldo de dominio (del OTRO frente, citado como argumento, no como cifra):** `exp2g` verificó
el 2026-09-28 que mirar el nombre **entero** del archivo hacía aprender cómo bautizó NapierOne sus
archivos (macro-F1 0,8052 → 0,2167 con los jpg fuera), y que la parte robusta es la **extensión
final**. Ventaja propia de esta capa: extrae la extensión del **texto** de la nota, no del nombre
del archivo, así que es inmune a ese artefacto de curaduría.

---

### ❌ ENSEMBLE DE VISTAS EN LA CAPA DE TEXTO — concatenar ya era suficiente (2026-09-28)

`2_codigo/ensemble_vistas.py`, preregistro H1–H7 en el docstring, **commiteado antes de correr**
(`c4686a4`). Log `4_resultados/_log_ensemble_vistas.txt`, salidas en
`4_resultados/resultados_ensemble_vistas_149/`. **149 notas, 30 familias, P2bal, 50 semillas.**

**La idea.** La última capa de la cascada es un LinearSVC sobre la vista `"combinado"`, que es un
`FeatureUnion`: **concatena** el TF-IDF de palabras y el de caracteres en un solo espacio. En un
espacio concatenado el peso relativo de cada bloque no se elige — lo fija cuántas dimensiones
aporta cada uno al producto interno. Se probó la alternativa: **un clasificador por vista y las
decisiones combinadas de forma explícita.** Es una pregunta distinta de la del hiperparámetro y de
la de los embeddings: **no cambia la representación, cambia cómo se agregan las decisiones.** Es la
única de esa familia que nunca se había probado.

**Seis formas de combinar**, 3 reglas × 2 conjuntos de vistas. `2v` = {palabras, caracteres} (la
alternativa **pura** a concatenar); `3v` = {palabras, caracteres, combinado}. Reglas: **suma** de
`decision_function` normalizada por vista · **voto** por mayoría con desempate por el margen mayor
(top1−top2) · **ponderada** con pesos por validación interna. Normalización por vista: z con media
y desvío del `decision_function` del **propio pliegue de entrenamiento** (normalizar con
estadísticos del pliegue de prueba sería transductivo).

**Cómo se eligieron los pesos (el punto donde es fácil hacer trampa).** Dentro de cada pliegue de
entrenamiento: se lo parte en 2 pliegues internos con el mismo `split_p2bal` (corte por plantilla,
rng propio `90_000 + 100·semilla + pliegue`), se entrenan las vistas en una mitad y se predice la
otra, y gana el juego de pesos con mayor macro-F1 **interno**; empate → el más cercano al uniforme.
Rejilla paso 0,1 (11 combinaciones con 2 vistas, 66 con 3). Recién entonces se reentrena con todo
el pliegue y se aplican esos pesos al de prueba. **El pliegue de prueba no se mira nunca.**

**Puerta H1 reproducida exacta:** `combinado` dio texto **0,6551 / 0,7191** y cascada
**0,7417 / 0,8123**, las cuatro cifras de cabecera.

| sistema | texto macro-F1 | texto exact. | cascada macro-F1 | cascada exact. |
|---|---|---|---|---|
| palabras (vista sola) | 0,6292 | 0,7024 | 0,7291 | 0,8044 |
| caracteres (vista sola) | 0,6412 | 0,6940 | **0,7429** | **0,8134** |
| **combinado (referencia)** | **0,6551** | **0,7191** | **0,7417** | **0,8123** |
| suma_2v | 0,6468 | 0,7078 | 0,7392 | 0,8109 |
| voto_2v | 0,6493 | 0,7118 | 0,7376 | 0,8119 |
| pond_2v | 0,6427 | 0,7067 | 0,7402 | 0,8122 |
| suma_3v | 0,6504 | 0,7123 | 0,7399 | 0,8115 |
| voto_3v | 0,6536 | 0,7173 | 0,7406 | 0,8123 |
| pond_3v | 0,6444 | 0,7087 | 0,7406 | 0,8122 |

**(a) Sobre la capa de TEXTO: las seis PIERDEN**, y cinco de las seis con IC 95 % que excluye el
cero. Δ macro-F1 pareado por semilla: suma_2v **−0,0083** [−0,0119; −0,0048] 10/50 · voto_2v
**−0,0058** [−0,0100; −0,0015] 16/50 · pond_2v **−0,0124** [−0,0179; −0,0069] 13/50 · suma_3v
**−0,0047** [−0,0070; −0,0024] 8/50 · voto_3v **−0,0015** [−0,0032; +0,0002] 15/50 · pond_3v
**−0,0107** [−0,0159; −0,0054] 16/50. **Ninguna mejora; el techo del ensemble es empatar.**

**(b) Sobre la CASCADA —la que decide— no cambia nada.** Δ exactitud: el peor |Δ| de las seis es
**0,0015**. voto_3v **−0,0000** [−0,0009; +0,0009] · pond_2v y pond_3v **−0,0001** · voto_2v
**−0,0004** · suma_3v **−0,0008** · suma_2v **−0,0015**. **Con Bonferroni sobre 6 variantes
(α 0,05/6 = 0,00833) ninguna excluye el cero por arriba: no hay nada que adoptar.** Por familia,
el mejor ensemble mueve a lo sumo **+0,008** (BLACKBASTA) y **−0,005** (JIGSAW); 24 de 30 familias
quedan exactamente iguales.

**Lo que sí se aprendió — los pesos NO tienen señal que elegir.** El reparto de w(caracteres) en
pond_2v sobre los 100 pliegues es **casi plano**: 0,0→13 · 0,1→11 · 0,2→7 · 0,3→6 · 0,4→7 ·
0,5→9 · 0,6→13 · 0,7→8 · 0,8→11 · 0,9→6 · 1,0→9. La media 0,482 no es un peso elegido: es el
promedio de un sorteo. **Causa: el macro-F1 de la validación interna es 0,3419**, contra 0,6551 del
externo — la mitad interna entrena con ~37 notas y en ese régimen el criterio no discrimina. Por eso
**pond_2v (−0,0124) es PEOR que suma_2v con pesos iguales (−0,0083)**: elegir pesos honestamente,
con este tamaño de corpus, cuesta más de lo que rinde. Es el resultado más transferible del
experimento.

**Hallazgo lateral, sin adoptar:** la vista **`caracteres` sola** iguala a la concatenación en la
cascada (macro-F1 0,7429, Δ **+0,0012** [−0,0037; +0,0062], 29/50 semillas; exactitud 0,8134, Δ
+0,0011 [−0,0026; +0,0047], 24/50). **No es significativo → no se adopta nada.** Lo interesante es
la forma: en la **capa de texto** `caracteres` pierde claro contra `combinado` (−0,0139 macro-F1,
−0,0251 exactitud, las dos significativas), y en la **cascada** la diferencia se evapora. La ventaja
de concatenar existe solo donde las reglas de IOC no llegan.

**Veredicto del preregistro: H1–H5 CUMPLEN, H6 falla.** H6 era una conjunción: `caracteres` > 
`palabras` en texto **sí** (0,6412 vs 0,6292), pero w(caracteres) ≥ 0,5 **no** (0,482) — y falla
porque el reparto de pesos es plano, no porque palabras domine. Lectura fijada por H7: **para este
corpus la concatenación ya es una forma razonable de combinar las dos vistas; el peso implícito no
estaba costando nada medible. NO ADOPTAR.**

**Dónde entra en el argumento.** Es el **cuarto negativo convergente** del frente de notas junto a
hiperparámetros (840 configs), embeddings multilingües y la cascada jerárquica por linaje
(−0,0011). Los cuatro atacan lugares distintos —hiperparámetro, representación, estructura de
clases, agregación de decisiones— y los cuatro dan nulo: el límite no está en el método sino en el
corpus. **Nota para D.1:** acá se midió *majority voting* **dentro** del frente de notas (entre
vistas TF-IDF), que **no** es el *majority voting* **entre los dos frentes** que pidió el tutor —
ese sigue sin poder evaluarse por falta de muestras pareadas. Pero es un dato para esa decisión: en
este corpus votar decisiones no agregó nada.

---

# 📋 PENDIENTE PARA EL FINAL: revisión científica independiente de la tesis completa (2026-09-28)

Pedido de Romina: una revisión «como si fuera un científico» que verifique que los datos sean
irrefutables y replicables, y que además **tome el proyecto como propio** para proponer mejores
formas, con sus resultados como precedente. **Se hace al finalizar, en un chat nuevo con Claude
Fable 5.1.** No antes: revisaría cifras que todavía cambian.

**Encargo completo y autocontenido:** `6_notas_trabajo/ENCARGO_REVISION_TESIS_COMPLETA.md`. Crear una
skill reutilizable (`.claude/skills/revisar-tesis/`), con inventario automático de cifras, cuatro
revisores en paralelo (archivos · notas · metodología · coherencia entre capítulos) y verificación
de cada hallazgo; lista de chequeo derivada de los errores reales de este proyecto. Precedente que
funcionó: la revisión de LOGO del 17-09.

---

# ❌ CAPA DE FORMA DEL NOMBRE EN NOTAS — NO APORTA. La transferencia del Exp. 2d FALLA (2026-09-28)

Script: `2_codigo/capa_forma_nombre.py` (preregistro commit `5b23894`, ampliación commit
`90e4097`). Log: `4_resultados/_log_capa_forma_nombre.txt`. CSV:
`4_resultados/resultados_capa_forma_nombre_149/`. **149 notas, 30 familias, P2bal, 50 semillas.**

**La pregunta.** La cascada ya usa el nombre genuino de la nota como **clave exacta**. No usa su
**forma** (el patrón). Se probaron siete abstracciones del nombre, cada una como **clave exacta
adicional** en el mismo diccionario, con el mismo filtro de genéricos y la misma regla de
unanimidad que los IOCs. Motivación: el Exp. 2d del frente de archivos.

**Puerta de entrada: PASA exacto.** La cascada sin la capa da macro-F1 **0,7417** y exactitud
**0,8123**, idénticos a la cifra de cabecera P2bal. Control interno adicional: el diccionario y la
regla locales dan predicciones **idénticas** a `protocolo_logo.dicc_privados` / `regla`.

## El techo, que es del corpus y no del método

**Solo 64 de las 149 notas tienen nombre genuino auditado = 42,95 %** (39 nombres distintos, 14 de
las 30 familias). Las otras 85 las renombró quien las recolectó y usarlas sería circular.
**Ninguna capa basada en el nombre puede cubrir más del 42,95 % del corpus.** Los repositorios
públicos renombran al catalogar: ese techo solo sube consiguiendo notas de fuentes que preserven
el nombre original.

## Resultado: todas las abstracciones EMPEORAN, salvo un positivo aparente que no lo es

Base de la capa de reglas: **cobertura 0,5391 · acierto donde aplica 0,9928.** Toda variante baja
la precisión donde aplica (a 0,935–0,979).

| Abstracción | Grupo | Cobertura agregada | Acierto en lo agregado | Δ macro-F1 global (IC 95 %) |
|---|---|---|---|---|
| ESQ esqueleto tipográfico | B | 0,0144 | **0,4299** | **+0,0042** [+0,0008; +0,0077] 35/50 |
| ESQC esqueleto colapsado | B | 0,0075 | 0,0179 | −0,0032 [−0,0055; −0,0009] 13/50 |
| FIRMA firma estructural | B | 0,0223 | 0,4217 | +0,0003 [−0,0041; +0,0046] 23/50 |
| DOM palabras del dominio | A | 0,0113 | **0,0000** | −0,0067 [−0,0086; −0,0049] 0/50 |
| EXT solo extensión (control) | A | 0,0099 | **0,0000** | −0,0020 [−0,0032; −0,0008] 0/50 |
| ROB ext + hay ID + dominio | A | 0,0110 | **0,0000** | −0,0034 [−0,0042; −0,0025] 1/50 |
| FROB firma sin largo ni tokens | A | 0,0197 | 0,3197 | −0,0069 [−0,0105; −0,0032] 14/50 |
| GRUPO_A combinado | A | 0,0361 | 0,1487 | −0,0157 [−0,0202; −0,0112] 8/50 |
| GRUPO_B combinado | B | 0,0337 | 0,3108 | −0,0058 [−0,0112; −0,0004] 19/50 |
| TODAS (las 4 del preregistro) | combo | 0,0337 | 0,2590 | −0,0110 [−0,0164; −0,0056] 14/50 |

**Cada vez que una capa de forma rompe la unanimidad, la base acertaba: `acierto_base_en_lo_roto`
= 1,000 en las ocho variantes que rompen algo** (DOM y ROB no rompen ninguna: su
`cobertura_rota` es 0,0000, pero tampoco aciertan nada de lo que agregan). La capa solo destruye.

**El +0,0042 de ESQ no es una mejora.** Donde agrega cobertura acierta **0,4299**; rompe 0,0193 de
decisiones donde la base acertaba 1,000; el **Δ exactitud global es +0,0005 con IC [−0,0011;
+0,0022]** (incluye el cero); y el **Δ macro-F1 restringido a las 64 notas con nombre es −0,0038**
— negativo justo donde la capa puede actuar. Se mueven **3 de 14 familias**: HELLOKITTY
0,2267 → 0,3800 (3 notas), RYUK −0,0134, LORENZ 1,0000 → 0,9000. Es el macro-F1 reaccionando a una
familia chica, no el sistema mejorando.

## Predicciones preregistradas que FALLARON

- **H5 (la principal) FALSADA.** Se predijo que ganaría DOM con Δ macro-F1 entre +0,005 y +0,040.
  DOM dio **−0,0067 con 0/50 semillas positivas** y **acierto 0,0000 donde agrega cobertura**.
- **H3b FALSADA:** la cobertura agregada de ESQC (0,0075) cae por debajo del piso predicho (0,010).
- **H10 quedó mal escrita.** La regla automática solo miraba «algún Δ macro-F1 con IC que excluye
  el cero» e imprime «TRANSFIERE». Es demasiado generosa: esa condición se cumple sin que la capa
  sirva. **La lectura que vale es la tercera rama de H10: la transferencia falla.**
- **A5 FALSADA (ampliación):** FROB (**−0,0069**) da MENOS que FIRMA (**+0,0003**). Sacarle a la
  firma estructural los dos campos contaminados la empeora ⇒ **lo poco que FIRMA aportaba vivía en
  los campos contaminados.**

## Fuga verificada en los nombres auditados — del mismo tipo que la del Exp. 2f

Al revisar por el aviso del Exp. 2g se encontró, **verificándolo sobre los nombres y no
razonándolo**, que el ID de la víctima lo enmascaró la auditoría **a mano y con dos notaciones
distintas**: `[]` en 15 nombres (2 caracteres) y `[victim's_id]` en 1 (13 caracteres,
`readme.[victim's_id].txt`, DARKSIDE). Por lo tanto **el largo de la cadena y la cantidad de
tokens no son propiedades del ransomware: dependen del auditor.** Efecto concreto: ese nombre cae
en el tramo «largo» y caería en «medio» con la máscara uniformada, y produce en FIRMA la clave
privada `.txt|none|1|no|SI|largo` → DARKSIDE, cuyo campo de largo es un artefacto.
**Es la misma forma de fuga que el `-fromweb` de NapierOne, a menor escala.**

Por eso las abstracciones se reportan en dos grupos: **A** (solo lo que pone el ransomware:
extensión, presencia del ID, palabras del dominio) y **B** (miran el nombre entero: largo,
tokens, tipografía). **El único Δ nominalmente positivo del experimento —ESQ— es del grupo B, el
sospechoso.** Ninguna del grupo A aporta: las tres que no miran largos aciertan **0,0000** donde
agregan cobertura.

## Veredicto

**NO SE ADOPTA.** La forma del nombre **no es firma de familia en notas**, ni en su versión
geométrica ni en la léxica. Explicación coherente con los dos frentes: **en NapierOne cada familia
es una sola campaña** —la forma del nombre *es* la campaña, que es la limitación ya declarada del
Exp. 2d— mientras que en el corpus de notas cada familia trae notas de **varias campañas**, que
renombran distinto. El nombre exacto ya captura lo poco que hay, y abstraerlo solo rompe
unanimidades correctas.

**Al citar:** toda cifra global está multiplicada por 64/149; el efecto restringido a las notas
con nombre es 2,328 veces el global en exactitud (identidad aritmética verificada, error máx.
0,000136).

⚠️ **NO citar los 0,9998 de «bytes + forma del nombre» como evidencia de que la forma del nombre
funciona:** esa cifra está comprometida por la fuga del Exp. 2f. La evidencia se está re-midiendo
en el Exp. 2g.

---

# ✅✅ EXP. 2g CERRADO — el sistema completo es ROBUSTO: bytes + estructura + extensión = 0,9998 y no colapsa en jpg. Cifra canónica del frente de archivos (2026-09-28)

Job 4091, 1 h 50 min (17:56 → 19:46), nodo c2. **Registrado de lo pegado.** Salidas en
`/scratch/ralfonzo/tesis/resultados_exp2g_job4091` (`cv_por_semilla.csv`,
`cv_por_familia_y_semilla.csv`, `cv_resumen.csv`, `cv_por_familia_resumen.csv`,
`tipos_por_pliegue.csv`, `manifiesto.json`). Base: NapierOne-small, 30 familias, 500/familia =
15.000 archivos, semillas 0-4, 5 pliegues, RF 300/20/2/0,3 (`class_weight="balanced"` en CV).

## Control de reproducibilidad: reproduce al 2f y al 2e-c al cuarto decimal

Sin estar programado como puerta, el control existe: **la columna (2) en CV** (0,9357 ± 0,0005 /
0,9359 ± 0,0004), **el F1 por familia de (2)** (NOTPETYA 0,4811 · JIGSAW 0,5739 · CRYPTOLOCKER
0,6576 · DARKSIDE 0,7515 · WASTEDLOCKER 0,8352 · SUNCRYPT 0,8296) y **las columnas (2) y (3) de
tipos no vistos** (doc 0,9337 / 0,9943 … jpg 0,8052 / 0,2167 … promedio 0,8789 / 0,8842) son
**idénticas** a las del 2f (job 4083), y la (2) de tipos también a la del 2e-c (job 4079). Mismo
muestreo, mismas particiones, mismos pliegues de prueba: **las comparaciones son pareadas de
verdad** y los n y familias por pliegue son los de la tabla del 2e-c.

## (A) Validación cruzada — 5 semillas

| Columna | Exactitud | macro-F1 |
|---|---|---|
| (2) bytes + estructura | 0,9357 ± 0,0005 | 0,9359 ± 0,0004 |
| **(5) bytes + estructura + forma de la EXTENSIÓN** | **0,9998 ± 0,0001** | **0,9998 ± 0,0001** |

Por semilla, macro-F1 (2) → (5): 0,9356 → 0,9999 · 0,9355 → 0,9998 · 0,9361 → 0,9999 ·
0,9360 → 0,9997 · 0,9365 → 0,9999. **Δ (5)−(2) = +0,0639 [+0,0634; +0,0644], 5/5.**

**Igual que la forma completa del 2f (0,9998 ± 0,0001), con las mismas semillas y particiones:
mirar solo la extensión final NO pierde nada en CV.** La salvedad de G1 («puede quedar algo por
debajo porque pierde la base») no se materializó.

**F1 por familia con (5), media de 5 semillas: las 30 ≥ 0,99.** Las más bajas: NOTPETYA 0,9978 ±
0,0008 · BADRABBIT 0,9988 ± 0,0008 · JIGSAW 0,9990 ± 0,0007 · BLACKBASTA 0,9996 · CUBA 0,9996. Las
otras 25 en 1,0000. **Solo CUBA baja respecto de (2)** (1,0000 → 0,9996 ± 0,0005).

| Familia | bytes (2f) | + estructura | **+ estructura + extensión** |
|---|---|---|---|
| NOTPETYA | 0,3850 | 0,4811 | **0,9978** |
| JIGSAW | 0,4338 | 0,5739 | **0,9990** |
| CRYPTOLOCKER | 0,6128 | 0,6576 | **1,0000** |
| DARKSIDE | 0,5981 | 0,7515 | **1,0000** |
| WASTEDLOCKER | 0,6266 | 0,8352 | **1,0000** |
| SUNCRYPT | 0,7590 | 0,8296 | **1,0000** |

**Observación, no mecanismo demostrado:** las dos más bajas con (5), NOTPETYA y BADRABBIT, son
las dos que no renombran lo que cifran (verificado en `subsec:exp2c_sesgo_pdf`): su extensión
final es la del documento y la marca «es de un tipo de documento» vale igual para las dos. Es
coherente con que ahí tengan que separar los bytes. No se midió. (No incluir a JIGSAW en esa
frase: el recuento del 2d la agrupa con ellas, pero sus nombres originales eran los de sus
archivos **sin cifrar**, que después se apartaron — a precisar en el pulido de §4.6.)

## (B) Dejar-un-tipo-fuera (semilla de muestreo 0, un ajuste por pliegue, RF semilla 42)

| Tipo excluido | (2) b + estr | (3) + forma completa (2f) | **(5) + extensión** | Δ (3)−(2) | **Δ (5)−(2)** |
|---|---|---|---|---|---|
| doc | 0,9337 | 0,9943 | 0,9943 | +0,0606 | +0,0606 |
| docx | 0,9023 | 1,0000 | 1,0000 | +0,0977 | +0,0977 |
| **jpg** | 0,8052 | **0,2167** | **0,9997** | **−0,5885** | **+0,1945** |
| pdf | 0,8131 | 0,9791 | 0,9800 | +0,1660 | +0,1669 |
| pptx | 0,8893 | 1,0000 | 1,0000 | +0,1107 | +0,1107 |
| xls | 0,9037 | 0,9995 | 0,9995 | +0,0958 | +0,0958 |
| xlsx | 0,9047 | 1,0000 | 1,0000 | +0,0953 | +0,0953 |
| **Promedio** | 0,8789 | 0,8842 | **0,9962** | | **+0,1174 [+0,0743; +0,1604], 7/7** |

**El colapso del 2f desaparece:** en jpg, la forma completa daba 0,2167 y la de la extensión da
**0,9997**. En los otros seis tipos (5) iguala a (3) (pdf: 0,9800 contra 0,9791). La columna (3)
reproduce el 2f al lado del arreglo, en la misma tabla.

**⭐ Lo que más dice:** la pérdida al pasar de VC aleatoria a tipo no visto es **−0,057 para bytes y
para bytes + estructura (2e-c), y −0,0036 para el sistema completo** (0,9998 → 0,9962). La
extensión que agrega el ransomware no depende del tipo del documento, así que no solo no
colapsa: **vuelve al sistema casi insensible al tipo**. El pliegue más bajo es pdf, 0,9800.

## Veredicto del preregistro: CUMPLEN LAS CUATRO

| | Predicción | Resultado | |
|---|---|---|---|
| G1 | (5) ≥ 0,995 en CV | **0,9998** | ✅ |
| G2 | jpg sin colapso: Δ (5)−(2) ≥ −0,02 | **+0,1945** (forma completa: −0,5885) | ✅ |
| G3 | Δ (5)−(2) > 0 en los 7 pliegues | mínimo **+0,0606** (doc) | ✅ |
| G4 | las cuatro difíciles ≥ 0,95 con (5) | NOTPETYA 0,9978 · JIGSAW 0,9990 · CRYPTOLOCKER 1,0000 · DARKSIDE 1,0000 | ✅ |

**Lectura acordada de antemano (docstring, commit `4477dbb`): «G1, G2 y G3 cumplen → (5) es el
sistema completo y robusto: la cifra canónica del frente».**

## ⭐ CIFRA CANÓNICA DEL FRENTE DE ARCHIVOS (criterio de Romina: el sistema que apila todas las capas)

> **Bytes + rasgos estructurales + forma de la extensión final (Exp. 2g): macro-F1 0,9998 ±
> 0,0001 y exactitud 0,9998 ± 0,0001** (15.000 archivos, 30 familias, 5 semillas × 5 pliegues),
> las 30 familias ≥ 0,99; bajo tipo de documento no visto, **0,9962 de macro-F1** promedio en siete
> pliegues, 7/7 por encima de bytes + estructura.
> **Limitación declarada, pegada:** una sola campaña por familia en NapierOne; la forma de la
> extensión puede ser del programa o de la campaña, y este conjunto no permite separarlo. **La
> parte que no arrastra esa limitación es la de contenido: 0,936** (Exp. 2e), que se reporta al lado.

## Qué queda superado y qué NO

- **Superado:** el 0,9998 de la forma COMPLETA del nombre (2d y 2f) como sistema: es real en CV
  pero aprende la base heredada de NapierOne (`-fromweb`) y colapsa en jpg. El aviso del bloque de
  la capa de forma del nombre en notas («no citar el 0,9998 de bytes + forma como evidencia… se
  está re-midiendo en el 2g») **queda resuelto: la evidencia robusta es la del 2g**, y es sobre la
  forma de la EXTENSIÓN, no del nombre entero.
- **Corrección de la síntesis:** con 5 semillas, bajo solo contenido (bytes + estructura) quedan
  **tres** familias por debajo de 0,75 (NOTPETYA, JIGSAW, CRYPTOLOCKER); DARKSIDE da 0,7515.
- **No cambia:** el 0,912 (solo bytes) y el 0,936 (contenido) siguen siendo mediciones válidas de su
  capa; el sistema completo las apila.

## ⚠ Verificación pendiente, disparada al preparar la tesis

La leyenda de `tab:exp2e_tipos` dice «BLACKMATTER y CERBER, que sustituyen el nombre completo,
participan solo del entrenamiento». **Sale de una etiqueta engañosa del script** del 2e-c:
`exp2e_validacion_tipos.py` imprime como «Familias que renombran por completo (solo
entrenamiento)» a toda familia con **algún** archivo sin tipo reconocible. BLACKMATTER renombra
13 de ~1.000 archivos (§4.5, `subsec:exp2c_renombrado`), así que sus demás archivos **sí** deberían
estar en la prueba. Los pliegues tienen 28 familias (medido), pero **cuáles son las dos que faltan
no está verificado.** Comando pedido a Romina sobre `tipos_por_familia.csv` del job 4079. Afecta
también la frase de `subsec:exp2c_renombrado` «su familia queda fuera del pliegue».

**Código corregido (commit `51c2d05`):** `exp2e_validacion_tipos.py` y `analisis_bytes.py` ahora
imprimen «Familias con archivos sin tipo reconocible (esos archivos solo entrenan)» y, aparte,
«Familias que no entran a ningún pliegue de prueba», calculadas. No cambia ningún resultado.

**✅ VERIFICADO el mismo día (comando sobre `tipos_por_familia.csv` del job 4079, pegado por Romina).**
Familias que NO están en la prueba de cada pliegue:

| Pliegue | Familias en la prueba | Fuera de la prueba |
|---|---|---|
| doc, docx, pdf, pptx, xls, xlsx | 28 | BLACKMATTER, CERBER |
| **jpg** | 28 | **CERBER, NOTPETYA** |

- **CERBER** no entra a ninguna prueba (sustituye el nombre: el tipo no se puede leer). ✔ lo que decía.
- **BLACKMATTER SÍ entra, pero solo al pliegue jpg.** La leyenda decía «solo entrenamiento»: **falso**.
- **NOTPETYA falta del pliegue jpg** (no tiene imágenes; coincide con el «—» del diagnóstico 4082). La
  leyenda no lo decía.
- Los 28 por pliegue eran correctos; lo que estaba mal era **cuáles**.

**Causa de lo de BLACKMATTER — HIPÓTESIS, no verificada:** su carpeta tendría casi solo imágenes.
Apoyos aritméticos: (1) el pliegue jpg tiene **2.401** archivos contra 1.974–2.071 de los demás; con
~71 por familia y tipo (500/7), las otras 27 familias darían ~1.920 y BLACKMATTER aportaría **~480
jpg de sus 500**; (2) el censo del 16-08 dice que BLACKMATTER renombra solo 13 archivos, así que no es
el renombrado lo que la saca de los pliegues de documentos; (3) explicaría la rareza anotada en el
2e-c: en jpg la exactitud cae a 0,6951 mientras el macro-F1 queda en 0,7987 — una familia con el 20 %
de los archivos del pliegue y casi sin entrenamiento (sin sus jpg le quedan ~16 archivos) baja mucho
la exactitud y poco el macro-F1. **Pedido a Romina un listado de la carpeta** (conteo por tipo) y el
F1 de BLACKMATTER en el pliegue jpg del 4079 para confirmarlo. Si se confirma, corregir también:
§4.5.5 (`subsec:exp2c_tipos`, «las familias que sustituyen por completo el nombre… no pueden
asignarse a ningún pliegue») y `subsec:exp2c_renombrado` («su familia queda fuera del pliegue»), que
atribuyen la ausencia de BLACKMATTER al renombrado.

**Corregido en la tesis ya** (compila, 93 págs., 0 errores):
- Leyenda de `tab:exp2e_tipos` con lo medido: CERBER solo entrena; BLACKMATTER solo entra a la
  prueba de jpg; NOTPETYA, sin imágenes, falta de ese pliegue. **Sin atribuirle causa a BLACKMATTER.**
- **«Configuración canónica» eliminado del frente de archivos:** en el 2d significaba solo bytes, en
  el 2e-c bytes + estructura, y ahora el canónico es el 2g. Se nombra cada una por lo que es (§4.5
  curva, §4.6 tres veces, §4.7.5 dos veces; «Adoptar esta configuración como resultado del frente» →
  «Incorporar esta configuración al sistema del frente»).

## Llevado a la tesis y a los informes el mismo día

**`resultados.tex`** (respaldo `resultados.tex.antes_2g`; CRLF preservado). Compila: **93 páginas,
0 errores, 0 referencias sin resolver**, sin desbordes en §4.6-§4.10.
- **Secciones nuevas** `sec:exp2f` («Experimento 2f: Sistema Completo») y `sec:exp2g` («Experimento
  2g: Nombre Robusto»), antes de la síntesis. Tablas `tab:exp2f` (CV de las cuatro columnas),
  `tab:exp2g_tipos` (tipos no vistos con las columnas + nombre y + extensión lado a lado) y
  `tab:exp2g_familias` (las seis difíciles a lo largo de las capas, 5 semillas). El 2f incluye el
  colapso en jpg con su causa y el valor metodológico (la validación por tipos detectó un artefacto
  de curaduría). El 2g cierra con «Alcance y limitación»: la limitación de campaña alcanza a todo el
  frente y la capa de la extensión es la más expuesta.
- **2d:** párrafo agregado al final de su limitación («un cuarto componente»), con remisión al 2f/2g.
- **2e:** tras «cuatro familias permanecen por debajo de 0,75» (una semilla) se agregó que con cinco
  semillas son tres.
- **Síntesis:** «los dos experimentos posteriores» → «los experimentos posteriores (2d a 2g)»; «La
  conclusión del frente… queda» → «Al cabo de esas etapas…»; las dos direcciones «culminan en un
  sistema que las apila»; frase del nombre robusto agregada; «tres con cinco semillas» agregado;
  **párrafo de la cifra reescrito**: 0,9998 del sistema completo, 0,9962 bajo tipo no visto, con la
  descomposición (contenido 0,936, bytes 0,912) y tres familias bajo 0,75 por contenido. **Precisión:**
  «la segunda concierne al contenido, y no arrastra esa limitación» → «y no depende del esquema de
  renombrado» (la limitación de campaña general alcanza también al contenido: §4.5.14 ya lo dice del
  0,912).
- **Tabla comparativa:** fila Exp. 2g en negrita; la del 2e pasa a «(solo contenido)» sin negrita;
  leyenda con la limitación. **Punto 2 de «Los resultados demuestran»** reescrito con las cifras
  vigentes. **Discusión:** una frase agregada (el sistema completo combina las dos señales) y el punto
  2 del procedimiento post-ataque actualizado (decía 91,2 % a secas).
- **Márgenes:** `tab:exp2e` y `tab:exp2e_tipos` (del 2e, de hoy) se salían 60 y 72 pt: envueltas en
  `\resizebox`. Quedan desbordes preexistentes que no son del frente de archivos.

**Informes a Cappo** (respaldos `.antes_2g`):
- `informe_2026-09-27_…_para_cappo.md`: recuadro con la cifra del frente al principio; §1 con
  actualización; §3 con la precisión de campaña; **§5 nuevo «El sistema completo — Exps. 2f y 2g»**;
  hallazgos pasan a §6 y suman **6.3 CRYPTOLOCKER determinista**; §7 (*majority voting*) empieza
  diciendo que con el sistema completo el margen es 0,0002; §8 suma la sexta predicción fallada
  (F4 del 2f); §9 decisión 1 reescrita — **la anterior atribuía a Romina la decisión «el 0,936 es
  canónico», que fue un error de Claude**; ahora dice su criterio real.
- `informe_cierre_2026-09-28_…_frente_archivos.pdf` (+ `.tex`): reescrito con las cifras vigentes.
  **2 páginas**, 0 errores, sin desbordes.

**Verificación de las cifras de los informes, contra el log pegado del 4091 y los bloques del 2f y
del 2e-c. Cinco imprecisiones propias corregidas antes de entregar:** (1) «de 0,912 a 0,9998 de
macro-F1» mezclaba métricas (0,912 es exactitud); (2) «catorce rasgos que miran solo la extensión»:
uno, la cantidad de puntos, mira el nombre; (3) CRYPTOLOCKER «clave fija sin IV» dicho como hecho →
«la huella de»; (4) **«techo de +2,5 puntos» (venía del informe anterior): +2,5 es la ganancia
estimada de delegar; el techo del oráculo es +5,2**; (5) «0,936» sin métrica en los dos informes.

**Para Romina, antes de mandar nada a Cappo:** con el sistema completo en 0,9998, el *majority
voting* con notas (elemento de acción 3) no puede sumar más de 0,0002: la cuenta del +2,5 vale
solo si no se dispone del nombre del archivo. Está así en los dos informes.

**Para el pulido final (no tocado):** el 2d dice que BADRABBIT, NOTPETYA **y JIGSAW** «son las
familias que no renombran»; los nombres originales de JIGSAW eran los de sus archivos **sin cifrar**,
apartados después. La figura de progresión (`generar_figuras_cap4.py`) sigue mostrando cuatro
enfoques. El «residuo genuino» del §4.5.8.

---

## 7. Reglas / convenciones (IMPORTANTES — respetar en todo chat)
- **★ FUENTES FIDEDIGNAS:** todo dato, archivo, métrica o afirmación que vaya a la tesis
  debe provenir de una **fuente verificable y citable** (paper, dataset oficial, repo con
  procedencia conocida). Registrar SIEMPRE el origen para poder citarlo. No usar nada sin fuente.
- **Solo 30 familias** (NapierOne). No ampliar el número de familias.
- Norma de citación: _(confirmar — el .bib sugiere estilo del template UNA)_
- Idioma de la tesis: español.
- Reportar métricas: accuracy + **balanced accuracy + macro-F1** + classification_report por familia.
