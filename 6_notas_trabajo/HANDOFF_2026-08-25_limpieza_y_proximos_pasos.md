# TRASPASO — 2026-08-25: corpus limpio en 149 notas y qué sigue

> **Para el chat nuevo:** leé este archivo y `ESTADO_TESIS.md`. Con eso retomás sin perder nada.
> Este documento es el resumen operativo de la sesión del 24-25 de agosto y **la lista de
> próximos pasos en orden**. El detalle con evidencia está en `ESTADO_TESIS.md`.

---

## ✅ 2026-09-22 — CIFRA VIGENTE DEL FRENTE DE NOTAS: **P2bal**

`protocolo_p2bal.py` (preregistro commiteado en el docstring, `a936397`): **texto 0,6551
[0,6454; 0,6648] · cascada M.6 0,7417 [0,7328; 0,7505]**, 50 semillas; sobre las 28 evaluables
0,7019 / 0,7946; MCC 0,709 / 0,804; 25/30 familias ≥ 0,50 con M.6. Es P2 con el reparto de
plantillas arreglado: mismo tamaño de entrenamiento (49,5 por pliegue) y misma garantía
(0 violaciones), pero sin los 2,86 ceros estructurales por pliegue que regalaba
`StratifiedGroupKFold`. Δ +0,196 / +0,223, 50/50 semillas. **Citar siempre como «plantilla no
vista según coseno char 0,90»**, criterio que no detecta contención (ver el aviso de abajo).
Detalle en `ESTADO_TESIS.md`, bloque «P2bal — EL REPARTO DE P2 ARREGLADO».

---

## ⛔ 2026-09-17 — REVISIÓN INDEPENDIENTE: **NO adoptar LOGO como cifra principal**

El bloque que sigue («P2-LOGO cambia la escala…») quedó **superado**. Un revisor independiente
(`6_notas_trabajo/REVISION_LOGO_2026-09-17_informe.md`, código `revision_logo.py`, salidas en
`4_resultados/resultados_revision_logo_149/`) encontró: (1) el salto P2→LOGO es sobre todo un
defecto del reparto de `StratifiedGroupKFold(2)`, que deja 3,9 familias por pliegue sin
entrenamiento; un 2 pliegues balanceado con la misma garantía (P2bal) ya da ~0,665 y LOGO solo
suma ~0,01 por duplicar el entrenamiento; (2) las 5 familias que llegan a 1,00 son pares de notas
**contenidas una en otra** (una de cada par viene de PCrisk), que el coseno de caracteres a 0,90 no
junta; 63 de 145 notas están contenidas ≥ 0,5 en otra plantilla propia y ahí LOGO acierta el 100 %;
fundiendo por contención ≥ 0,8, LOGO texto cae a 0,42 (30 fam.) / 0,60 (21 evaluables); (3) lo que
sí resistió: IC por bootstrap **por plantilla** [0,539; 0,722] texto y [0,636; 0,809] M.6, sin fuga
por código, no es circularidad de nombre. Leer el informe antes que el bloque de abajo. Las
correcciones a `ESTADO_TESIS.md` están listadas allí, **sin aplicar**.

---

## ⚠️⚠️ LEER PRIMERO — 2026-09-09: P2-LOGO cambia la escala de todo el frente de notas

El protocolo canónico P2 (2 pliegues) **descartaba la mitad del entrenamiento en cada corte**. La
justificación («hay familias con 2 notas») excluye k-fold con k≥3 pero **nunca excluyó
leave-one-template-out**, que es el protocolo estándar para este caso y nadie había probado.
Con LOGO, misma garantía (la plantilla de prueba nunca está en entrenamiento; **comprobado: coseno
máximo prueba-entrenamiento 0,8995 en los 99 pliegues, 0 violaciones**):

| | P2 (2 pliegues) | **LOGO** |
|---|---|---|
| texto solo | 0,4593 (50 sem.) | **0,6747** [0,567; 0,712] |
| M.6 | 0,5191 (50 sem., = canónico) | **0,7742** [0,663; 0,810] |
| M.3 abstención (umbral 0,50) | contesta 65 %, acierta 90 % | **contesta 85 %, acierta 94 %** |

Δ pareado +0,21 / +0,24, 10/10 semillas. **26 de 30 familias >0,50 bajo LOGO-M.6.** El umbral de
0,50 del tutor **se supera con IC entero**. Detalle, predicciones falladas y autocrítica en
`ESTADO_TESIS.md`, bloques «PREREGISTRO — P2-LOGO» y «RESULTADO P2-LOGO». Scripts:
`protocolo_logo.py`, `abstencion_notas.py --protocolo LOGO`.

**Consecuencias que cambian el plan de abajo:**
- **CORREGIDO en la misma sesión:** el 0,470 es la fila **P2** de B.1. B.1 ya tiene **P2ret**
  (1 plantilla por familia fuera, 70 en train): **0,643 ± 0,061** en k=todo, meseta en k=3–4 y
  leve baja después. LOGO (0,675, 98 en train) lo **confirma** (+0,03). **NO hace falta re-correr
  B.1 bajo LOGO.** Detalle: ESTADO_TESIS «PRECISIÓN a lo anterior». Hallazgo extra del CSV: bajo
  P2 quedan **3,9 familias/pliegue sin entrenamiento**; bajo P2ret/LOGO, 2 (BADRABBIT, CRYPTOLOCKER, plantilla
  única — la pregunta de Cappo).
- Reportar los tres protocolos juntos (P1 / LOGO / P2) diciendo qué mide cada uno.
- Autocrítica a escribir: P2 fue canónico meses con una justificación que no excluía LOGO.
- **HECHO (10/9): 50 semillas.** P2 texto 0,4593 (= canónico), P2 M.6 0,5191, LOGO texto 0,6747,
  LOGO M.6 0,7742. Δ LOGO−P2: texto +0,2155 [+0,194; +0,237], M.6 +0,2550 [+0,233; +0,278],
  50/50 semillas. Carpeta `resultados_protocolo_logo_149_50sem/`. Las 5 familias que suben a 1,00
  son las de 2 plantillas (SUNCRYPT, CUBA, NETWALKER, BLACKMATTER, DARKSIDE).

Lo que sigue abajo (limpieza 155→149, cifras P2, pasos) **sigue siendo válido**, pero léase con
LOGO en mente: las cifras P2 son correctas *para P2*.

---

## 1. Lo que cambió: el corpus pasó de 155 a 149 notas

**Se hizo una limpieza de integridad.** Seis notas retiradas a `3_datos/descartados_integridad/`
(**movidas, no borradas** — reversible, con README que documenta cada caso) y tres incorporadas
transcritas verbatim.

| Retiradas | Motivo |
|---|---|
| `HELLOKITTY/hellokitty_note2.txt` | **es de DHARMA** (confirmado por 3 evidencias independientes) |
| `HELLOKITTY/hellokitty_note1.txt` | linaje Dharma/Phobos — sospecha fuerte, **no confirmada** |
| `CHIMERA/chimera_note1.txt` y `note2.txt` | sintéticas; **las auténticas ya estaban en el corpus** |
| `CRYPTOLOCKER/cryptolocker_note2.txt` | sintética; CryptoLocker no dejaba archivo de nota |
| `LOCKBIT/lb20.txt` | duplicado por **defanging** de `lockbit2.txt` (mismo texto, 477 car. los dos) |

**Incorporadas** (verbatim desde Amigo-A, `tipo=transcripcion`), reemplazando material sintético:
`idr_badrabbit_pantalla_2017.txt`, `idr_badrabbit_readme_2017.txt`,
`idr_notpetya_readme_var2_2017.txt`.

**Deuda de procedencia: 47 → 0.** Las 149 notas tienen fuente citable. Hay un control que lo
mantiene: `python 2_codigo/validar_procedencia.py` (falla con exit 1 si entra una nota sin
fuente, si la deuda crece, o si manifiesto y disco no coinciden).

⚠️ **Cuidado al citar:** existen carpetas `_150` (`resultados_notas_150`,
`resultados_grafo_marcadores_150`) que están **SUPERADAS** — se corrieron antes de retirar
`lb20.txt`. **No usarlas.**

---

## 2. Las cifras vigentes (base 149)

| | Valor | Dónde |
|---|---|---|
| P2 base (texto solo) | **0,4593 ± 0,0752** | `resultados_cascada_combinada_149` |
| **M.6** (IOCs privados + nombre) | **0,5191** · Δ **+0,0599** [+0,0530; +0,0668] · 50/50 semillas | ídem |
| Sobre las **28 evaluables** | **0,5562** | calculado de `m6_por_familia.csv` |
| Clasificador canónico P2 | 0,468 ± 0,100 | `resultados_notas_149` |
| Clasificador canónico P1 | 0,799 ± 0,026 | ídem |
| Protocolo Lemmou (mundo cerrado) | 0,7767 | `resultados_protocolo_lemmou_149` |
| **M.3 abstención** | contesta **65 %**, acierta **90 %** (umbral 0,50) | `resultados_abstencion_149` |
| **M.1** cascada IOC→texto | **0,4959** · Δ **+0,0279** [+0,0124; +0,0434] · 10/10 semillas | `resultados_cascada_149` |
| **B.1** corte de la curva | **3 plantillas** por familia (antes 4 sobre 144) · faltan **11** en 9 familias | `resultados_curva_149`, `resumen_cap4_149` |
| **Techo estimado bajo P2** | macro-F1 **0,470** [0,411; 0,527] · **0,50 no alcanzable con más datos** | `resumen_cap4_149` |

**M.6 salió MÁS fuerte en el corpus limpio** (+0,0599 contra +0,0513 sobre 155).

---

## 3. PRÓXIMOS PASOS, en orden

### (a) ✅ COMPLETO (2026-08-25) — la re-medición sobre 149 está cerrada

Las tres corridas hechas, en `resultados_cascada_149/`, `resultados_curva_149/` y
`resumen_cap4_149/`. Resultados completos en `ESTADO_TESIS.md`. Resumen:

- **M.1:** veredictos idénticos a los de 155 — sin circularidad **SE ADOPTA** (Δ +0,0279
  IC 95 % [+0,0124; +0,0434], 10/10 semillas), con circularidad **NO**. El Δ **creció**
  respecto de 155 (+0,0188), igual que M.6.
- **Curva B.1:** el corte se mueve de **4 a 3 plantillas** por familia (los tres conjuntos
  concuerdan: el último paso significativo es 2→3). Cuesta **11 plantillas nuevas en 9
  familias**, contra las 33 en 19 que costaba llegar a 4 sobre 144 notas. **No es efecto de la
  limpieza: es otra base** (144/95 plantillas entonces, 149/99 ahora).
- **Techo estimado, la cifra más fuerte:** bajo **P2** el ajuste da techo **macro-F1 0,470**
  [0,411; 0,527] y **ni 0,50 es alcanzable agregando notas**. P1 llega a 0,80 con 2,9
  plantillas/familia.
- **Hallazgo nuevo:** pasado el corte, sumar material **baja** el macro-F1 de forma medible
  (cinco pasos con IC 95 % entero por debajo del cero). Estaban mal etiquetados como
  «indistinguible de cero».

**La advertencia de verificar los insumos se cobró cuatro arreglos de código** (todos en
develop): `cascada_ioc_notas.py` abortaba porque tenía la base de 155 clavada con tolerancia
0,003 → ahora `--base-desde` (`f09151e`, `4b0febc`); `resumen_para_capitulo4.py` imprimía «144
notas» fijo y no marcaba los ajustes con techo > 1 (`2c04e07`); el **azar de los subconjuntos
estaba escrito en el código** —imprimía 0,091 y 0,200 cuando sobre 149 son 0,067 y 0,333—
(`db9aab4`); y la etiqueta de significancia de los Δ tapaba los pasos negativos (`ad05071`).

⚠️ **Al citar la curva:** las etiquetas `11fam` y `5fam` son **nombres históricos**, no conteos.
Sobre 149 esos subconjuntos tienen **15 y 3 familias** (azar 0,067 y 0,333). Y los ajustes de
extrapolación de esos dos subconjuntos salieron **degenerados** (techo > 1): no citarlos.

⚠️ **Sigue valiendo para cualquier script que se corra de nuevo:** verificar de dónde lee sus
insumos antes de confiar en la salida. Ya aparecieron cinco casos del mismo patrón
(`techo_por_familia.py` fue el primero). El que queda identificado y sin tocar es
`cascada_combinada_notas.py`: tiene la base de 155 clavada con tolerancia 0,12, tan holgada que
no verifica nada — **las cifras de M.6 están bien** (su Δ es pareado contra su propia capa de
texto recomputada), pero esa puerta de entrada hoy es decorativa.

### (b) Decisión de reporte que hay que llevarle a Cappo — **la de mayor impacto**
**BADRABBIT y CRYPTOLOCKER dan F1 exactamente 0,0000** porque quedaron con **1 plantilla**: bajo
P2 nunca aparecen en entrenamiento y prueba a la vez. El propio `clasificador_notas_v2.py` emite
esa advertencia solo.

**Y en las DOS familias de 1 plantilla es propiedad REAL del malware, no hueco de
recolección:** BADRABBIT (las 2 variantes auténticas difieren solo en `key#1` vs `key#2`;
coseno 0,9490 → colapsan) · CRYPTOLOCKER (era una ventana; coseno 0,9883 → colapsan).

> ✅ **CORRECCIÓN 2026-08-25 (medida, chat siguiente).** Esta lista decía «las tres familias de
> 1 plantilla» e incluía a WASTEDLOCKER. **Son dos.** WASTEDLOCKER tiene **4 notas y 3
> plantillas**: sus cosenos internos son 0,8861 · 0,8879 · 0,8957 · 0,8977 · 0,8995 y 0,9742, y
> solo ese último par colapsa bajo el umbral de 0,90. Es el mismo molde a ojo —eso estaba bien—
> pero el agrupador la cuenta como 3, así que **es evaluable** y no es una de las que dan F1 = 0.
> Lo que NO cambia: las únicas familias con F1 = 0,0000 en `m6_por_familia.csv` son BADRABBIT y
> CRYPTOLOCKER (verificado en las 4 variantes de M.6), así que **las 28 evaluables y el +0,037
> siguen exactos**. Detalle y tablas en `ESTADO_TESIS.md`, bloque «CORRECCIÓN AL TRASPASO».
> WASTEDLOCKER sirve para otro argumento, más fuerte: es el caso de frontera del umbral 0,90.

| Conjunto | Familias | F1 base | F1 M.6 |
|---|---|---|---|
| Todas | 30 | 0,4593 | 0,5191 |
| **Evaluables** | **28** | **0,4921** | **0,5562** |

**Propuesta: reportar las dos cifras**, con la lista de inevaluables y la prueba de que su única
plantilla es real. Son **+0,037 de macro-F1** por dejar de promediar ceros estructurales. No es
un truco: es no contar como fracaso del método algo que el método no puede evaluar.

### (c) Donde hay más margen que en notas: el frente de ARCHIVOS
**Exp. 2d + A.3**, listos, **un solo job de clúster** (comparten la carga de datos). Es el
elemento de acción 2 del tutor. Ver `EXPERIMENTOS_PENDIENTES.md`.

### (d) Análisis cortos pendientes
- **Año de detección vs F1 por familia** — pedido textual de Cappo. Sale de `Pruebas.xlsx`, hoja
  «Informacion sobre familias». ⚠️ **NO sale de MISP**: verificado, solo cubre 5 de 28 familias.
- **Barrido de boilerplate compartido sobre las 149.** Es el método que encontró el error de
  HelloKitty en un paso, después de que tres tandas de búsqueda por texto no lo resolvieran.
  Buscar frases de cierre compartidas entre familias distintas. Si hay más notas mal
  clasificadas, las encuentra.
- **Extraer las 1.375 referencias de MISP** a un CSV por familia. Es lo más valioso del catálogo
  que queda sin usar, y destraba la bibliografía pendiente desde julio.

### (e) Decisiones que son de Romina + Cappo
1. **`hellokitty_note1.txt`**: se retiró por sospecha fuerte (comparte el boilerplate de cierre
   con 12 notas de DHARMA y 2 de PHOBOS, y con ninguna otra de HelloKitty) pero **ninguna fuente
   la atribuye explícitamente a Dharma**. Confirmar o reincorporar.
2. **Group-IB a la lista blanca**, si se aprueba la extensión a familias nuevas: 43 grupos
   activos con texto completo de nota y atribución. ⚠️ **No cubre ninguna de las 30**, así que
   sirve solo para la extensión.
3. **Las 4 reescrituras que se conservan** (`MAZE/malware_notes_maze.txt`,
   `RYUK/note_variant_email.txt`, `WASTEDLOCKER/note_pcrisk.txt`, `LOCKBIT/lb30b.txt`).
   ✅ **CORRECCIÓN 2026-08-25 (medida):** **tres de las cuatro no inflan** el conteo —RYUK
   0,9861, MAZE 0,9837 y LOCKBIT 0,9132, todas por encima de 0,90, y el agrupador las colapsa—
   pero **`WASTEDLOCKER/note_pcrisk.txt` SÍ: su coseno máximo con una hermana es 0,8977, queda
   por debajo del umbral y forma plantilla propia (grupo 145).** Es, ella sola, la razón por la
   que WASTEDLOCKER cuenta 3 plantillas y no 2. Se quedan las cuatro, pero hay que declarar que
   esa aporta una plantilla al conteo, porque la moneda de B.1 son las plantillas.

---

## 4. Cosas que NO hay que volver a intentar (verificadas y cerradas)

- **Buscar el origen de las notas sintéticas.** Cuatro pistas verificadas y descartadas: el repo
  `gitlab.com/kipziptie/ai_ransomware_note_detection`, la convención de placeholders publicada,
  el dataset de Kaggle y Group-IB. **Lo único que queda es preguntarle a Carlos**: `Pruebas.xlsx`
  es material compartido y el lote fundacional es anterior al seguimiento de procedencia.
- **El dataset de Kaggle como fuente de notas.** No tiene etiquetas de familia (las 769 líneas
  dicen `__label__ransomware`), así que no puede sumar plantillas. Y 72 de las 149 notas del
  corpus ya están ahí: no es independiente.
- **`date` de MISP** para el análisis de año: 5 de 28 familias. Inservible.
- **M.7 metadato de la nota** (extensión + tamaño): nulo/negativo. Cerrado.

---

## 5. Reglas de esta fase que conviene no olvidar

- **Toda cifra con su base pegada.** Ahora conviven 155 y 149; un número sin base es inútil.
- **Salidas a carpeta NUEVA siempre.** Las de 155 quedan intactas: se AGREGA, no se reemplaza.
- **Las erratas de la fuente se conservan.** `key#l` con ele minúscula y `waste your tine` están
  a propósito: corregirlas sería repetir el error que se acaba de limpiar.
- **Las truncaduras de la fuente (`*****`) no son placeholders del corpus.** Esa distinción es la
  que separa una transcripción legítima de una nota reconstruida.
- **Un F1 plano puede esconder una mejora grande.** La limpieza eliminó **166 confusiones**
  DHARMA/PHOBOS→HELLOKITTY (314 invasiones → 47) y el F1 de HELLOKITTY casi no se movió, porque
  mejoró la precisión y el recall sigue limitado por tener 3 plantillas poco cohesionadas.
  **Reportar precisión y recall por separado en las familias afectadas.**
