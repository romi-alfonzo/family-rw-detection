# Revisión independiente de la propuesta «LOGO» — informe (2026-09-17)

**VEREDICTO: NO ADOPTAR LOGO como cifra principal del frente de notas.** Reportarlo, si se quiere, como una fila más de la comparación de protocolos, pero con la explicación corregida: **el salto P2 → LOGO (+0,21 de macro-F1) no es «máximo entrenamiento» sino, en un 95 %, un artefacto de la estratificación de P2** (familias que quedan sin ninguna plantilla de entrenamiento en un pliegue). Un P2 de 2 pliegues con las plantillas repartidas de forma balanceada («P2bal», misma garantía, mismo tamaño de entrenamiento de ~49,5 plantillas por pliegue) da **0,6651 ± 0,032** de macro-F1 texto; LOGO agrega apenas +0,01 (0,6747). Además, la afirmación «supera 0,50 con IC entero» resiste el bootstrap por plantilla ([0,539; 0,722]) pero **no resiste el criterio de casi-duplicado**: las 5 familias que llegan a F1 1,00 son pares de notas donde una está contenida en la otra (contención 0,82–0,91 por 3-shingles de palabras) con coseno 0,45–0,87, y al fusionar esos pares LOGO-texto cae a 0,4186 sobre 30 familias.

Revisor: chat independiente, sin contexto del proponente. Corpus: 149 notas, 99 plantillas, 30 familias (verificado en la carga). Todo lo que sigue se recalculó desde el código y el corpus; nada se tomó de `ESTADO_TESIS.md` sin reproducirlo. Script: `2_codigo/revision_logo.py` (nuevo, commiteado). Salidas: `4_resultados/resultados_revision_logo_149/` y logs `4_resultados/_log_revision_logo_{A,B,B_colchon,B_anatomia,C}.txt`.

---

## 0. Lo que se reproduce y lo que no (control de partida)

| Afirmación del encargo / ESTADO | Verificación | Resultado |
|---|---|---|
| LOGO texto 0,6747, M.6 0,7742, cobertura regla 0,604 | `revision_logo.py --parte A`, línea «[1]» | **reproducido** exacto (0,6747 / 0,7742 / 0,6040) |
| LOGO idéntico entre semillas (premisa de reutilizarlo 50 veces) | semillas 0 y 1, comparación nota a nota | **reproducido**: 0 notas distintas en texto, 0 en M.6 |
| P2 a 10 semillas 0,4680 / 0,5318; a 50 semillas 0,4593 / 0,5191 | parte B umbral 0,90 (10 sem.); `logo_resumen.csv` de 50 sem. | reproducido a 10 sem. (0,4680 / 0,5318); el de 50 no se re-corrió (CSV canónico, no discutido) |
| IC bootstrap por notas [0,567; 0,712] / [0,663; 0,810] | misma semilla 7, misma convención | **reproducido** ([0,5674; 0,7118] / [0,6631; 0,8097]) |
| Coseno máximo prueba→entrenamiento 0,8995, 0 pliegues > 0,90 | `a_logo_predicciones_por_nota.csv`, columna `maxcos_train` | reproducido (máx 0,8995; el colchón 0,90 no quita ninguna nota) |
| «61 de 149 notas de prueba (41 %) tienen su casi-copia en train bajo P1, semilla 0» | recalculado | 61 si se cuenta coseno directo > 0,90; 63 (42,3 %) si se cuenta pertenencia al mismo grupo. Correcto tal como lo definieron |
| «3,9 familias por pliegue sin entrenamiento bajo P2» | recalculado con `StratifiedGroupKFold(2)`, 20 semillas | **reproducido**: 3,90 por pliegue (rango 2–8); 7,8 familias distintas por semilla (unión de los 2 pliegues). B.1 da 3,85 con 10 repeticiones |
| «26 de 30 familias > 0,50 bajo LOGO-M.6; 22 > 0,70» | `logo_por_familia_pivot.csv` (50 sem.) | reproducido (quedan bajo 0,50: BADRABBIT, CRYPTOLOCKER, HELLOKITTY, RYUK) |
| «Sobre 28 evaluables: 0,7229 / 0,8295» | recalculado | reproducido (es 0,6747·30/28: las 2 excluidas están en 0) |
| P2ret 0,6433 ± 0,061, «70,2 plantillas en train», «~45 notas evaluadas» | `b1_curva_por_repeticion.csv`, 30fam·P2ret·plantillas·k=todo, 100 rep. | reproducido: 0,6433 ± 0,0613; 70,2 plantillas; **45,3 notas evaluadas [34; 57]** |
| Bloque de preregistro escrito antes del resultado | `git log -S "PREREGISTRO — P2-LOGO" -- ESTADO_TESIS.md` | **no verificable por git**: `ESTADO_TESIS.md` no se commitea desde `750a069` (2026-08-20); ver pregunta 12 |

---

## 1. Evidencia por pregunta

### A. Legitimidad del protocolo

**P1. ¿LOGO por plantilla es estándar y defendible, o una elección optimista?**

Hecho: revisión del código (`sklearn.model_selection.LeaveOneGroupOut` en `protocolo_logo.py` L174) y de la literatura (referencias verificadas por búsqueda web, listadas al final).

- *Leave-one-group-out* es un protocolo estándar para datos con estructura de grupos: es el «leave-one-subject-out» de Saeb et al. (2017) y la «block cross-validation» de Roberts et al. (2017); ambos recomiendan exactamente esto cuando hay varias observaciones por unidad (aquí: varias notas por plantilla). La garantía «la plantilla de prueba nunca está en entrenamiento» es la misma que la de P2. **En eso el proponente tiene razón.**
- Lo que la literatura dice en contra: la validación con un solo grupo fuera tiene sesgo bajo pero **varianza alta** y no ofrece estimador entre particiones (Arlot & Celisse 2010, §5; Kohavi 1995; Hastie, Tibshirani & Friedman 2009, §7.10). Bengio & Grandvalet (2004) demuestran que no existe estimador insesgado de la varianza de la validación cruzada; Varoquaux (2018) mide barras de error de ±10 puntos con n≈100 y advierte que el error estándar entre pliegues las subestima. Con 99 grupos y familias de 2 plantillas, el F1 por familia bajo LOGO se decide con 1 o 2 notas: es un estimador legítimo pero **muy ruidoso por familia** (ver los IC por plantilla de `a_por_familia.csv`: CLOP [0,00; 1,00], GANDCRAB [0,00; 1,00], MAZE [0,00; 1,00]).
- Forman & Scholz (2010) muestran que el F1 **agrupado sobre todas las predicciones fuera de pliegue** (lo que hace `protocolo_logo.py`, L195–205: `f1_score(y, p)` sobre las 149 notas) y el F1 **promediado por pliegue** no son comparables y que con pliegues chicos el promedio se sesga. P2 y LOGO agrupan las predicciones igual, así que entre ellos la comparación es limpia; **P2ret no** (promedia 100 macro-F1 de ~45 notas cada uno), ver pregunta 9.

Conclusión: LOGO no es optimista *como protocolo*; lo optimista es (i) la explicación que se le dio al salto (pregunta 8) y (ii) la definición de «plantilla» sobre la que se apoya (preguntas 4 y 7). Citable: Saeb 2017, Roberts 2017, Arlot & Celisse 2010, Kohavi 1995, Varoquaux 2018, Bengio & Grandvalet 2004, Forman & Scholz 2010, Hastie 2009 (ya está en el `.bib` como `hastie2009`).

**P2. ¿Es válido el bootstrap por notas? Cluster bootstrap por plantilla.**

Hecho: `revision_logo.py --parte A`, función `bootstrap()`, B = 2000, dos semillas (7 y 2026), dos unidades de remuestreo (notas / plantillas) y dos convenciones de macro-F1. Salida: `a_bootstrap_ic.csv`.

| Remuestreo | Convención | LOGO texto: IC 95 % | media boot. | sd boot. | P(< 0,50) | LOGO M.6: IC 95 % |
|---|---|---|---|---|---|---|
| notas (la del script, semilla 7) | labels = 30, zero_division = 0 | **[0,5674; 0,7118]** | 0,6436 | 0,037 | 0,0000 | [0,6631; 0,8097] |
| notas | labels presentes en la remuestra | [0,6063; 0,7503] | 0,6736 | 0,037 | 0,0000 | [0,7095; 0,8481] |
| **plantillas (cluster)** | labels = 30 | **[0,5388; 0,7221]** | 0,6358 | **0,047** | **0,0040** | **[0,6364; 0,8088]** |
| plantillas (cluster) | labels presentes | [0,5952; 0,7772] | 0,6857 | 0,046 | 0,0000 | [0,7072; 0,8694] |
| plantillas, Δ M.6 − texto | labels = 30 | [0,0415; 0,1471] (P(Δ ≤ 0) = 0) | 0,093 | 0,028 | — | — |

(La segunda semilla da lo mismo a la tercera cifra; está en el CSV.)

- El bootstrap por notas **no es válido** para estos datos: las notas de una plantilla son casi idénticas (por construcción, coseno > 0,90) y remuestrearlas como independientes infla el n efectivo (Field & Welsh 2007 para el bootstrap de datos agrupados). El IC correcto es el de plantillas. Es **más ancho** (sd 0,047 frente a 0,037) y su límite inferior baja de 0,567 a **0,539**.
- **La afirmación «supera 0,50 con el IC entero» resiste el cluster bootstrap**: 0,50 queda fuera con las dos convenciones, y solo el 0,4 % de las remuestras por plantilla cae debajo de 0,50. Esto hay que decirlo con la misma claridad con que se dice lo que no resiste.
- Dos salvedades técnicas que el proponente no declaró: (a) la convención del script (`labels=familias, zero_division=0`, `protocolo_logo.py` L144) asigna F1 = 0 a toda familia que no cae en la remuestra, lo que **sesga la media bootstrap hacia abajo** (0,644 frente al punto 0,675; 0,636 por plantillas) y hace el IC asimétrico: no es un IC centrado en la estimación; (b) ningún bootstrap sobre predicciones fijas captura la variabilidad por cambio del conjunto de entrenamiento (Bengio & Grandvalet 2004; Varoquaux 2018), que es justamente lo que P2 sí muestra con su sd 0,075 entre semillas.
- El IC del Δ M.6 − texto, que el script deja degenerado ([+0,0994; +0,0994]), queda resuelto: **[+0,04; +0,15]** por plantilla.

**P3. Varianza oculta de LOGO y cómo reportar.**

Hecho: el «± 0» de LOGO en la tabla de ESTADO (`logo_resumen.csv`, columna `f1_macro_sd` = 0,0) es la sd entre semillas del clasificador, que es cero por convexidad de LinearSVC (verificado: 0 notas distintas entre semillas). No es incertidumbre del protocolo. La incertidumbre honesta de LOGO es la sd del cluster bootstrap: **0,047 (texto), 0,044 (M.6)**. Compárese con P2: sd 0,075 entre semillas (50 sem., `logo_resumen.csv`), y con P2bal (pregunta 8): sd 0,032 entre 20 semillas. Reportar LOGO como «0,6747 [0,539; 0,722], IC bootstrap por plantilla», nunca «± 0», y decir en la misma frase que ese IC no incluye la variabilidad por partición porque LOGO no tiene particiones alternativas.

### B. Fuga de información

**P4. Distribución del coseno máximo prueba→entrenamiento; umbral 0,80 / 0,70.**

Hecho: parte A (tramos de coseno, `a_coseno_tramos.csv`); parte B (re-agrupamiento con `agrupar_neardups(textos, u)`, `b_umbral_sensibilidad.csv`; y «LOGO con colchón»: grupos canónicos a 0,90, pero de cada pliegue de entrenamiento se saca toda nota con coseno > c con la plantilla de prueba, `b_logo_colchon_colchon.csv`).

Acierto de LOGO por tramo del coseno máximo (char 3-5, el espacio del agrupamiento) entre la nota de prueba y su vecino más cercano en entrenamiento:

| Tramo | n notas | acierto texto | acierto M.6 | acierto 1-NN |
|---|---|---|---|---|
| [0,0; 0,5) | 16 | 0,375 | 0,625 | 0,250 |
| [0,5; 0,6) | 22 | 0,818 | 0,818 | 0,409 |
| [0,6; 0,7) | 10 | 0,500 | 0,600 | 0,500 |
| [0,7; 0,8) | 36 | 0,583 | 0,889 | 0,583 |
| **[0,8; 0,9)** | **65** | **0,923** | **0,954** | **0,923** |

**65 de las 149 notas (44 %) tienen un vecino de entrenamiento a coseno entre 0,80 y 0,90**, y en ellas LOGO-texto acierta el 92 %; en las otras 84 acierta el 60 %. El «0 pliegues con coseno > 0,90» es verdad por construcción (`agrupar_neardups` usa `sim > umbral`, L172 de `clasificador_notas_v2.py`) y no dice nada sobre la banda inmediatamente inferior, que es donde vive el resultado. Un 1-NN sin entrenamiento acierta 99/149 = 0,664 bajo LOGO (LinearSVC: 0,738); cuando el vecino más cercano es de otra familia (50 notas), el SVC acierta 0,24.

Sensibilidad al umbral de agrupamiento (LOGO determinista; P2 a 10 semillas; macro-F1):

| Umbral | plantillas | fam. de 1 plantilla | evaluables | LOGO texto (30) | LOGO M.6 (30) | LOGO texto (evaluables) | P2 texto (30) | Δ LOGO−P2 texto |
|---|---|---|---|---|---|---|---|---|
| **0,90** (canónico) | 99 | 2 | 28 | **0,6747** | 0,7742 | 0,7229 | 0,4680 | +0,2067 |
| 0,85 | 83 | 6 | 24 | **0,5172** | 0,5873 | 0,6465 | 0,3310 | +0,1862 |
| 0,80 | 70 | 11 | 19 | **0,3379** | 0,4255 | 0,5336 | 0,2027 | +0,1352 |
| 0,75 | 66 | 12 | 18 | 0,2691 | 0,3926 | 0,4485 | 0,1989 | +0,0702 |
| 0,70 | 61 | 13 | 17 | 0,2348 | 0,3511 | 0,4144 | 0,1695 | +0,0653 |

LOGO con colchón (grupos a 0,90; se quitan del entrenamiento los vecinos > c de la plantilla de prueba):

| Colchón | notas quitadas por pliegue (media / máx) | pliegues evaluables que pierden toda plantilla propia | LOGO texto (30) | LOGO M.6 (30) | LOGO texto (28) |
|---|---|---|---|---|---|
| 0,90 | 0,00 / 0 | 0 | 0,6747 | 0,7742 | 0,7229 |
| **0,80** | 0,92 / 6 | 15 | **0,4721** | 0,5327 | 0,5059 |
| 0,70 | 1,49 / 12 | 21 | 0,3430 | 0,4275 | 0,3675 |

Conclusión: **la cifra se derrumba al bajar el umbral** (0,675 → 0,517 → 0,338 sobre 30 familias) y **baja de 0,50 con solo apartar del entrenamiento los vecinos a coseno > 0,80** (0,4721). Parte del derrumbe es estructural (a 0,80 hay 11 familias de una sola plantilla, con F1 = 0 por construcción), pero sobre las familias evaluables LOGO-texto a 0,80 es 0,5336: apenas roza el umbral del tutor. Dos matices que juegan a favor del proponente: (i) P2 se derrumba igual o más, de modo que el **orden LOGO > P2 se conserva en todos los umbrales**; (ii) la elección del 0,90 es anterior a LOGO y canónica en todo el frente. Es decir: el problema no es de LOGO sino de la definición de «plantilla nunca vista» sobre la que se apoya todo el frente de notas; lo que LOGO hace es volverlo visible, porque su cifra depende casi entera de la banda 0,80–0,90.

**P5. Grupos que mezclan familias.**

Hecho: recuento directo sobre `agrupar_neardups(textos, 0,90)`. Hay **dos** grupos mixtos: grupo 6 = `BLACKBASTA\blackbasta2.txt` + `CONTI\conti4.txt` (2 notas) y grupo **53** (no 55: el id es el índice raíz y cambió con el corpus de 149; el docstring de `curva_aprendizaje_notas.py` L136 dice 55, sobre 144) = 11 notas de DHARMA (`dharma.txt`, `Info__*.hta`) + `PHOBOS\pcrisk_phobos_1.txt`. LOGO los trata bien: `LeaveOneGroupOut` saca el grupo entero, así que la nota de PHOBOS y las 11 de DHARMA salen juntas y ninguna casi-copia queda en entrenamiento. Consecuencia visible: DHARMA-texto bajo LOGO tiene precisión 1,00 y **recall 0,42** (las 11 notas del grupo 53 se predicen PHOBOS cuando salen todas juntas); M.6 la rescata (recall 0,95) por los IOC. Eso es parentesco real DHARMA/PHOBOS, ya documentado en B.3, no fuga. Al bajar el umbral los grupos mixtos suben a 4 (0,80) y 6 (0,70): la etiqueta «familia» se vuelve ambigua para más notas.

**P6. Vectorizador y diccionario M.6 solo con entrenamiento.**

Hecho: lectura línea por línea.
- `protocolo_logo.py` L112–114: `vec = vectorizador("combinado")` nuevo por pliegue, `fit_transform(textos_arr[tr])`, `transform(textos_arr[te])`. Correcto.
- `protocolo_logo.py` L119: `d = dicc_privados(tr, iocs, nombres_nota, y)` — el diccionario IOC/nombre → familia se arma solo con índices de entrenamiento; L89–90 borra los valores que aparecen en más de una familia *del entrenamiento*. Correcto.
- `abstencion_notas.py` L142–146 y L156: idem para M.3 bajo LOGO. Correcto.
- `agrupar_neardups` (`clasificador_notas_v2.py` L155–177) ajusta el TF-IDF char sobre **todo** el corpus para agrupar: es un paso no supervisado previo a la partición (deduplicación), aceptable y declarado en el docstring.
- El nombre genuino de la nota de prueba se usa como rasgo de la nota (L96–97): es un observable legítimo, no la etiqueta.
- Dato nuevo: bajo LOGO la regla de M.6 contesta 90 notas y **acierta las 90** (`a_logo_predicciones_por_nota.csv`; coincide con M.3-LOGO umbral 1,50: 95 respondidas, acierto 1,000). Bajo P2 acierta ~97,6 %.

No hay fuga por código. Lo que hay es fuga **por definición de grupo** (preguntas 4 y 7).

### C. De dónde sale la ganancia

**P7. Las cinco familias de 2 plantillas que llegan a 1,00: ¿nombre de la familia u otro término circular?**

Hecho: (a) detección por nota de `_terminos_circulares(familia)` más patrones compuestos («BLACK Matter», «Dark Side»); (b) LOGO y P2 con esos términos **enmascarados** en el texto de las 30 familias (405 ocurrencias borradas, 68 notas modificadas) y el diccionario M.6 sin valores circulares (`excluir_circulares` de la cascada); (c) lectura de las notas; (d) parte C: contención por 3-shingles de palabras.

Resultado sobre la circularidad: **67 de 149 notas (18 familias) contienen el nombre o alias de su familia**. De las cinco: DARKSIDE 3/3 («Welcome to DarkSide» y dominios `darksidedxcftmqa.onion`), CUBA 3/4 (`cuba_support@exploit.im`, `cuba-supp.com`, `cuba4ikm…onion`), NETWALKER 2/3 («encrypted by Netwalker»), BLACKMATTER 1/2 («BLACK / Matter»), SUNCRYPT 0/2. **Pero el enmascaramiento no cambia LOGO-texto: 0,6747 → 0,6747, F1 por familia idéntico en las 30** (comprobado que el enmascaramiento sí se aplica: márgenes del SVC cambian en la tercera cifra; «darkside» ni siquiera entra al vocabulario de palabras por `max_features=5000`). P2-texto enmascarado: 0,4642 (10 sem.). **La ganancia de LOGO-texto no viene del nombre de la familia.** Sí afecta a M.6: LOGO-M.6 con filtro de circularidad 0,7742 → **0,7226** (CHIMERA 0,80 → 0,00 porque su único IOC compartido es `mega.nz/ChimeraDecrypter`; también bajan LORENZ, RYUK, TESLACRYPT, BLACKBASTA, JIGSAW, SODINOKIBI). Eso no es específico de LOGO: la cascada canónica adoptó la variante *sin* filtro de circularidad bajo P2, y bajo P2 el mismo filtro da 0,4976.

**De dónde viene entonces el 1,00 de las cinco** (`c_contencion_por_familia.csv`, `c_contencion_por_nota.csv`):

| Familia | par de plantillas | contención máx. (3-shingles) | coseno char entre ellas | qué son |
|---|---|---|---|---|
| SUNCRYPT | `suncrypt.html` → `note_pcrisk.txt` | **0,818** | **0,4516** | el HTML extrae **184 caracteres / 35 palabras** que son el final literal de la otra nota |
| NETWALKER | `note_pcrisk.txt` → `pcrisk_netwalker_2.txt` | **0,909** | 0,8112 | misma nota; la segunda agrega un párrafo de extorsión |
| DARKSIDE | `darkside.txt` → `note_pcrisk_variant.txt` | **0,911** | 0,8729 | misma nota; la segunda agrega la sección «Data leak» |
| CUBA | `pcrisk_cuba_1.txt` → `cuba.txt` | **0,870** | 0,7635 | misma nota recortada (56 frente a 103 palabras) con otro correo |
| BLACKMATTER | `blackmatter.txt` → `note_pcrisk.txt` | **0,869** | 0,8751 | misma nota; una agrega arte ASCII, la otra «Data leak includes» |

**El coseno char 3-5 no detecta contención**: una nota corta contenida en una larga da coseno bajo (SUNCRYPT: 0,45) porque la larga tiene mucho texto extra. Con la medida de contención, de las 145 notas que tienen otra plantilla de su familia, **63 están contenidas ≥ 0,5 en otra plantilla propia y LOGO-texto acierta las 63 (100 %); en las 82 restantes acierta 0,573.** Con contención ≥ 0,8: 26 notas / 20 plantillas / 13 familias, acierto 1,000.

LOGO y P2 con los grupos **fusionados por contención** (además del coseno 0,90; `c_fusion_contencion.csv`):

| Fusión | plantillas | fam. 1 plantilla | evaluables | LOGO texto (30) | LOGO M.6 (30) | LOGO texto (evaluables) | P2 texto (30) | las cinco (LOGO texto) |
|---|---|---|---|---|---|---|---|---|
| ninguna | 99 | 2 | 28 | 0,6747 | 0,7742 | 0,7229 | 0,4680 | todas 1,00 |
| ≥ 0,9 | 92 | 5 | 25 | 0,5598 | 0,6490 | 0,6718 | 0,3920 | NETWALKER 0, DARKSIDE 0 |
| **≥ 0,8** | 81 | 9 | 21 | **0,4186** | 0,4921 | 0,5980 | 0,3174 | **todas 0,00** |
| ≥ 0,7 | 73 | 9 | 21 | 0,3992 | 0,4721 | 0,5703 | 0,2556 | todas 0,00 |

Conclusión: la ganancia de las cinco no es circularidad de nombre; es que **su «segunda plantilla» es la misma nota con un bloque de más o de menos**. Bajo cualquier definición de plantilla que mire contención, esas familias pasan a tener una sola plantilla y F1 = 0 por construcción, y LOGO-texto queda en 0,42 sobre 30 familias / 0,60 sobre las 21 evaluables. Mismo comentario que en P4: esto afecta a P2 por igual (0,32), y el orden se conserva. El aporte real que sale de acá es para el corpus: **el criterio de casi-duplicado necesita una medida de contención además del coseno**; el proponente no lo miró y es un agujero que el jurado puede abrir con solo pedir ver las dos notas de SUNCRYPT.

**P8. ¿3,9 familias por pliegue sin entrenamiento? ¿Eso explica el hueco, o lo explica el tamaño de entrenamiento?**

Hecho: `revision_logo.py --parte B --solo anatomia --semillas-p2 20` (`b_p2_resumen.csv`, `b_p2_anatomia_por_familia.csv`, `b_p2_anatomia_por_semilla.csv`). Se recomputó `StratifiedGroupKFold(2)` con las semillas 0–19 y se implementó **P2bal**: dos pliegues por plantilla, con las plantillas de cada familia repartidas de forma balanceada entre los dos (una familia de 2 plantillas queda con una en cada pliegue; las de 1 plantilla siguen en 0). Mismo tamaño de entrenamiento que P2 (49,5 plantillas por pliegue), misma garantía de plantilla no vista.

| Protocolo (texto solo, macro-F1 30 fam.) | media | IC 95 % (t, semillas) | sd | M.6 |
|---|---|---|---|---|
| P2 canónico (20 sem.) | 0,4617 | [0,425; 0,499] | 0,079 | 0,5245 |
| P2 sin las familias estructurales de cada semilla (22,2 familias en promedio) | 0,6208 | [0,593; 0,648] | — | 0,7057 |
| LOGO sobre esas mismas familias | 0,7267 | — | — | 0,8280 |
| **P2bal** (mismo tamaño de train que P2, sin ceros estructurales salvo las 2 de 1 plantilla) | **0,6651** | **[0,650; 0,680]** | **0,032** | **0,7490** |
| LOGO (30 fam.) | 0,6747 | boot. plantilla [0,539; 0,722] | 0,047 | 0,7742 |

- «3,9 familias por pliegue sin plantilla de entrenamiento»: **verificado** (3,90; distribución por pliegue: 2 → 7 pliegues, 3 → 12, 4 → 8, 5 → 8, 6 → 2, 7 → 2, 8 → 1). Por semilla, la unión de los dos pliegues deja **7,8 familias** con F1 = 0 estructural en al menos un pliegue.
- **El hueco P2 → LOGO es casi todo estratificación, no tamaño de entrenamiento:** P2 → P2bal = **+0,203** con el mismo número de plantillas en entrenamiento; P2bal → LOGO = **+0,010** duplicando las plantillas de entrenamiento (49,5 → 98). Para M.6: +0,225 y +0,025.
- La prueba por familia refuta el «mecanismo global» que ESTADO propone (bloque «El mecanismo que las predicciones no vieron»): para las siete familias de 2 plantillas, cuando P2 las reparte en pliegues distintos su F1-texto ya es alto (**BLACKMATTER 0,90, CUBA 0,92, DARKSIDE 0,86, NETWALKER 0,90, SUNCRYPT 0,93**; NOTPETYA 0,58, CHIMERA 0,27), y cuando las deja juntas (40–55 % de las semillas) es 0,00. Los 0,40–0,56 «bajo P2» son la mezcla de ~0,9 y 0. LOGO no «afila las fronteras de las otras 29»: simplemente nunca cae en el caso «juntas».

Conclusión: el proponente encontró el dato correcto (3,9) y le dio la lectura equivocada («parte cuantificable»; «la diferencia P2→LOGO es tamaño de entrenamiento»). Es al revés: el tamaño de entrenamiento vale ~0,01 y la estratificación ~0,20. Lo que corresponde corregir no es el protocolo canónico por LOGO, sino la **partición** del protocolo canónico: `StratifiedGroupKFold(2)` de sklearn no garantiza que una familia de 2–4 plantillas tenga alguna en cada pliegue, y con 30 familias chicas falla en promedio 7,8 veces por semilla. P2bal conserva las 2 particiones, la varianza entre semillas que el tutor pide ver y la garantía de plantilla no vista, y supera 0,50 con IC entre semillas [0,650; 0,680] (con el criterio canónico de casi-duplicado; **no se corrió P2bal a otros umbrales ni con fusión por contención**, y lo que se vio en P4 y P7 le aplica igual).

**P9. P2ret (0,6433, ~45 notas por repetición) frente a LOGO (0,6747, 149 notas): ¿comparables?**

Hecho: `b1_curva_por_repeticion.csv` (30fam · P2ret · plantillas · k=todo): 100 repeticiones, macro-F1 0,6433 ± 0,0613, **45,3 notas evaluadas por repetición [34; 57]**, 70,2 plantillas en entrenamiento, 2,0 familias sin train. LOGO: 149 notas agrupadas, 98 plantillas en entrenamiento.

No son directamente comparables, por tres razones: (i) **denominador**: P2ret promedia 100 macro-F1 calculados sobre ~45 notas (una plantilla por familia; una familia de 2 notas se juzga con 1–2 notas por repetición) y LOGO calcula un macro-F1 sobre las 149 predicciones agrupadas (Forman & Scholz 2010 sobre por qué difieren); (ii) **diseño de la prueba**: en P2ret las 30 plantillas retenidas se prueban *a la vez* contra un modelo que no vio ninguna de las 30, de modo que las confusiones son entre plantillas desconocidas; en LOGO cada plantilla se prueba contra un modelo que conoce las otras 98, incluidas todas las plantillas confundibles de las otras familias. LOGO es el diseño más favorable de los dos; (iii) **tamaño de entrenamiento**: 70 frente a 98 plantillas, pero P2bal muestra que pasar de 49,5 a 98 vale ~0,01, así que el «+0,03 por 28 plantillas más» de ESTADO no se sostiene como explicación: es sobre todo (i) y (ii). Lo que sí es cierto y útil: los tres protocolos sin ceros estructurales (P2ret 0,643, P2bal 0,665, LOGO 0,675) coinciden en la banda 0,64–0,68; el que está solo es P2 (0,459), por el artefacto de la pregunta 8.

### D. Lo que pidió el tutor

**P10. Métrica por métrica del PDF, bajo LOGO.**

Hecho: se leyó el PDF (`C:\Users\Romina\Downloads\Ransomware - Notas de rescate insuficientes.pdf`, 9 pág., el 17/9 seguía ahí) y se calcularon las métricas (`a_metricas_logo_y_p2.csv`, `a_por_familia.csv`, `a_matriz_confusion_logo_{txt,m6}.csv`).

| El PDF pide | Bajo LOGO, estado | Cifra |
|---|---|---|
| Curvas de aprendizaje con fracciones crecientes | no se corrió bajo LOGO; existe B.1 bajo P2/P2ret/P1 | (P2ret meseta en 3–4 plantillas) |
| IC 95 % y dispersión entre particiones | **no existe «entre particiones» en LOGO** (partición única); se agrega IC bootstrap por plantilla | texto [0,539; 0,722], sd 0,047; M.6 [0,636; 0,809], sd 0,044 |
| Macro-F1 (no accuracy) | sí | texto 0,6747 (30) / 0,7229 (28); M.6 0,7742 / 0,8295 |
| Precisión, recall, F1 por familia | sí, calculado hoy, con IC por plantilla | `a_por_familia.csv` |
| **MCC** | **faltaba en `protocolo_logo.py`; calculado hoy** | texto **0,7301**, M.6 **0,8522** (P2 10 sem.: 0,562 ± 0,107 / 0,650 ± 0,091) |
| Balanced accuracy | sí | texto 0,7183, M.6 0,8096 |
| Matriz de confusión | calculada hoy | `a_matriz_confusion_logo_*.csv` |
| Dispersión N_i frente a F1_i | posible con `a_por_familia.csv` (n_plantillas, f1) | no graficada |
| **CV(F1_i) por familia** («la evidencia más fuerte») | **no existe bajo LOGO** (sin particiones repetidas); el sustituto es la sd bootstrap por plantilla | `f1_sd_boot_plantillas` en `a_por_familia.csv`: 0 en las de 2 plantillas (degenerado), 0,21–0,32 en CLOP, DHARMA, MAZE, PHOBOS, RYUK |
| Subsampling de familias grandes / N mínimo | no bajo LOGO (B.1 bajo P2ret) | — |
| Baseline char n-gram + SVM lineal | es el canónico | — |
| Transformer / S-BERT | Exp. 3e (P2) | — |
| Independencia: dedup, N efectivo | sí (99 plantillas de 149) — **pero ver P7: falta contención** | 63 notas contenidas ≥ 0,5 en otra plantilla propia |
| Imbalance ratio | 19/2 = 9,5 (notas) | verificado |

**¿«50 %» de qué métrica?** El tutor lo dijo sobre el informe del 18/8 (P2 texto 0,435, macro-F1 sobre 30 familias). Lo honesto es fijarlo como **macro-F1 sobre las 30 familias, texto solo y M.6 por separado**, y dar al lado la fila «sobre 28 evaluables». Tabla propuesta para mostrarle (macro-F1 texto / M.6, 30 familias, criterio de casi-duplicado canónico 0,90):

| Protocolo | qué mide | plantillas en train | texto | M.6 | incertidumbre |
|---|---|---|---|---|---|
| P1 | instancia nueva de plantilla conocida | 58,8 | 0,7889 ± 0,024 (50 sem.) | — | entre semillas |
| P2 canónico | plantilla no vista, 2 pliegues sklearn | 49,5 | 0,4593 ± 0,075 | 0,5191 ± 0,079 | entre semillas (50) |
| **P2bal** | plantilla no vista, 2 pliegues balanceados | 49,5 | **0,6651 ± 0,032** | **0,7490 ± 0,032** | entre semillas (20; **falta a 50**) |
| P2ret | una plantilla por familia fuera | 70,2 | 0,6433 ± 0,061 | — | entre repeticiones (100) |
| LOGO | una plantilla fuera, 99 pliegues | 98 | 0,6747 [0,539; 0,722] | 0,7742 [0,636; 0,809] | bootstrap por plantilla |
| + fila obligatoria | mismo LOGO con grupos fusionados por contención ≥ 0,8 | — | 0,4186 (30) / 0,5980 (21) | 0,4921 / 0,7030 | — |

**P11. Familias de una sola plantilla.**

Hecho: BADRABBIT y CRYPTOLOCKER están en el denominador del macro-F1 sobre 30 con F1 = 0,000 bajo P2, P2bal, P2ret y LOGO (`a_por_familia.csv`; `logo_por_familia_pivot.csv`). Sí debe reportarse también «sobre 28 evaluables» (0,7229 / 0,8295 bajo LOGO), diciendo cada vez cuál es cuál. La respuesta honesta a «no se procesa» es: correcto, en texto puro no hay con qué entrenar y el F1 es 0 por construcción, no por falla del clasificador (§K de `PENDIENTE_REDACCION.md` lo dice bien); **y hay que agregar** que, con la medida de contención, la lista de familias «de una plantilla» crece: con fusión ≥ 0,8 son **9** (las 2 más SUNCRYPT, CUBA, NETWALKER, BLACKMATTER, DARKSIDE y dos más; ver `c_fusion_contencion.csv`), y con re-agrupamiento a coseno 0,80 son 11. El «26 de 30 superan 0,50» de LOGO-M.6 es verdadero sobre el agrupamiento canónico y no sobrevive a esas definiciones.

### E. Riesgo de presentación

**P12. ¿Jardín de senderos? ¿Preregistro verificable?**

Hecho: `git log -S "PREREGISTRO — P2-LOGO" -- ESTADO_TESIS.md` devuelve **vacío**; el último commit de `ESTADO_TESIS.md` es `750a069` (2026-08-20); la copia de trabajo tiene +4.154 líneas sin commitear y `HEAD:ESTADO_TESIS.md` no contiene la palabra «LOGO». Lo único fechado por git es el commit `059cf5f` (2026-09-09 16:29:33, solo `protocolo_logo.py` y `estilometria_notas.py`), cuyo docstring (L26) cita el bloque de preregistro por su nombre; los primeros resultados en disco son de las 17:00:05 (`logo_resumen.csv`) y la primera corrida se cayó al imprimir la tabla por familia (`_log_logo_149.txt`, Traceback en L252 del script viejo). **El contenido del preregistro (rango 0,55–0,65, criterio de falsación) no es verificable**: existía *un* bloque con ese nombre a las 16:29, no se puede probar qué decía. A favor del proponente: dos de las cuatro predicciones se reportaron como falladas, con detalle, y la corrección «LOGO confirma B.1, no lo derriba» se escribió en la misma sesión contra el propio entusiasmo inicial. Eso es lo que se espera de un preregistro honesto.

El riesgo de «cambiar el protocolo hasta que el número pase» es **real y el orden de los hechos lo alimenta**: el 9/9 el tutor dice «si no superamos 50 % no sirve», y ese mismo día aparece un protocolo nuevo con el que se supera. Cawley & Talbot (2010) sobre el sesgo de selección al elegir entre alternativas después de ver el resultado; Gelman & Loken (2013) sobre los senderos que se bifurcan sin necesidad de mala fe. La única forma de presentarlo como aporte metodológico y no como excusa es:
1. Poner **primero el hallazgo que explica el hueco** (P8): el P2 canónico estaba deprimido ~0,20 por un artefacto de partición de sklearn, medible y reproducible; la corrección mínima (P2bal) conserva todo lo del canónico. Es un resultado *contra* el propio trabajo previo, y por eso creíble.
2. Presentar la tabla de protocolos completa (P10) con las filas que **no** superan 0,50 (P2, LOGO con contención, LOGO a coseno 0,80), no solo las que sí.
3. Declarar el orden temporal tal cual fue, incluido que el preregistro no quedó en git, y commitear `ESTADO_TESIS.md` de ahora en más para que el próximo sí quede.
4. No usar la palabra «máximo aprovechamiento / máximo entrenamiento» como explicación: es falsa (P8).

**P13. Lo que el proponente no miró.**

1. **El artefacto de estratificación de P2** (P8): es el hallazgo principal de esta revisión y cambia la lectura de todo el frente de notas, incluidos B.1 bajo P2 (techo 0,470 «no alcanzable») y todas las cifras P2 del cap. 4, que están deprimidas por la misma causa. Recomiendo correr P2bal a 50 semillas antes de decidir nada, y re-leer B.1 con ese dato.
2. **Contención** (P7): el corpus tiene pares «misma nota con un bloque de más» que el coseno char no junta. Afecta la cuenta de plantillas (99), el N efectivo del PDF del tutor y toda cifra «plantilla nunca vista», bajo cualquier protocolo. Requiere decisión sobre el criterio de casi-duplicado (coseno **o** contención ≥ umbral) y, si se adopta, re-correr el frente. El caso extremo (SUNCRYPT: HTML de 35 palabras) además sugiere revisar la extracción de texto de los `.html`.
3. La convención `labels=30, zero_division=0` en el bootstrap sesga la media hacia abajo (P2); no invalida el IC pero hay que decirlo si se reporta.
4. `b_p2_anatomia_por_familia.csv`: qué familias quedan sin entrenamiento y con qué frecuencia (CHIMERA y CUBA 55 % de las semillas, SUNCRYPT 50 %, DARKSIDE 45 %, BLACKMATTER y NETWALKER 40 %, SODINOKIBI 35 %, HELLOKITTY y JIGSAW 25 %). Es material directo para la discusión de «familias con pocas plantillas».
5. Bajo LOGO la regla de M.6 acierta 90/90; la cobertura 0,604 coincide con M.3-LOGO. Consistente, y vale como dato a favor de la capa de reglas.
6. El id del grupo mixto DHARMA/PHOBOS es 53 sobre 149 (el código dice 55): cosmético, pero un comentario incorrecto en código canónico se cita después como dato.
7. `a_metricas_logo_y_p2.csv` trae P2 a 10 semillas (la corrida de 50 de esta revisión se detuvo para no duplicar CPU); el MCC de P2 a 50 semillas que cita ESTADO (0,5610 / 0,6446) **no se re-verificó** aquí: no verificable en esta revisión, compatible con los 0,562 / 0,650 a 10 semillas.

---

## 2. Experimentos corridos (comandos, salidas, dónde quedaron)

Todo local, Python 3.11.2, scikit-learn 1.6.1 (el mismo del manifiesto de B.1). Un ajuste (vectorizador + LinearSVC sobre 148 notas) tarda ~3 s en esta máquina con varios procesos en paralelo; una pasada LOGO son 99 ajustes.

```
cd C:\Users\Romina\Tesis
python -u 2_codigo/revision_logo.py --parte A --semillas-p2 10          > 4_resultados/_log_revision_logo_A.txt
python -u 2_codigo/revision_logo.py --parte B                            > 4_resultados/_log_revision_logo_B.txt        (solo el bloque de umbral; se detuvo después)
python -u 2_codigo/revision_logo.py --parte B --solo colchon --colchones 0.80 0.70 > 4_resultados/_log_revision_logo_B_colchon.txt
python -u 2_codigo/revision_logo.py --parte B --solo anatomia --semillas-p2 20    > 4_resultados/_log_revision_logo_B_anatomia.txt
python -u 2_codigo/revision_logo.py --parte C                            > 4_resultados/_log_revision_logo_C.txt
```

Nota operativa: la primera corrida de las tres partes en paralelo se dio por muerta y se relanzó A en primer plano; en realidad seguían vivas y las dos A escribieron en la misma carpeta. No importa para las cifras: LOGO, el bootstrap (semillas fijas) y el enmascaramiento son deterministas e idénticos en ambas; la única diferencia es la fila «P2 (10 sem.)» de `a_metricas_logo_y_p2.csv`, que quedó a 10 semillas. El log `_log_revision_logo_A.txt` quedó parcialmente pisado; las tablas están en los CSV.

Salidas en `4_resultados/resultados_revision_logo_149/` (carpeta nueva; no se tocó ningún resultado existente):

| Archivo | Contenido |
|---|---|
| `a_logo_predicciones_por_nota.csv` | 149 filas: predicción texto/M.6, acierto, coseno máx. con train, vecino 1-NN, términos circulares hallados, nombre genuino |
| `a_bootstrap_ic.csv` | IC por notas y por plantilla, dos convenciones, dos semillas, Δ M.6−texto |
| `a_metricas_logo_y_p2.csv` | macro-F1 30/28, exactitud, bal. acc, **MCC**, F1 ponderado; LOGO y P2 (10 sem.) |
| `a_por_familia.csv` | P/R/F1 por familia, sd e IC bootstrap por plantilla, notas con nombre en el texto |
| `a_matriz_confusion_logo_txt.csv`, `..._m6.csv` | matrices 30×30 |
| `a_coseno_tramos.csv` | acierto por tramo de coseno máximo prueba→train |
| `a_circularidad_por_familia.csv`, `a_enmascarado_resumen.csv`, `a_enmascarado_por_familia.csv` | circularidad y corrida enmascarada |
| `b_umbral_sensibilidad.csv` | re-agrupamiento a 0,90/0,85/0,80/0,75/0,70; LOGO y P2 |
| `b_logo_colchon_colchon.csv` | LOGO con colchón 0,80 / 0,70 |
| `b_p2_resumen.csv`, `b_p2_anatomia_por_semilla.csv`, `b_p2_anatomia_por_familia.csv` | anatomía de P2 y P2bal (20 semillas) |
| `c_contencion_por_nota.csv`, `c_contencion_tramos.csv`, `c_contencion_por_familia.csv`, `c_fusion_contencion.csv` | contención por 3-shingles y LOGO/P2 con grupos fusionados |

Comprobaciones sueltas (no guardadas, reproducibles en segundos): P1 semilla 0 → 61 notas con vecino de train a coseno > 0,90 (63 por pertenencia a grupo); enmascaramiento efectivo en el pliegue de `darkside.txt` (márgenes 0,442/−0,706 → 0,439/−0,698); tiempo por ajuste 3,23 s.

Código: `2_codigo/revision_logo.py`, commit en `develop` con push (sin coautoría). No se modificó ningún script canónico ni ningún `.tex`.

**No corrido / no verificable en esta revisión:** P2bal a 50 semillas; P2bal a otros umbrales y con fusión por contención; MCC de P2 a 50 semillas; el contenido del bloque de preregistro antes de las 17:00 del 9/9 (no está en git); B.1 bajo LOGO o P2bal.

---

## 3. Lista de correcciones a `ESTADO_TESIS.md` (NO aplicadas; las aplica el chat original o Romina)

Bloques «RESULTADO P2-LOGO (2026-09-09)», «RESULTADO P2-LOGO a 50 semillas (2026-09-10)», «PRECISIÓN a lo anterior» y el encabezado de `HANDOFF_2026-08-25…md`:

1. **Borrar o invertir la explicación del mecanismo.** Donde dice «Lo que cambia no es su entrenamiento: es el de las otras 29… efecto global… la precisión se dispara» y «La diferencia P2→LOGO es tamaño de entrenamiento, no método ni fuga»: es falso. P2bal (mismo tamaño de entrenamiento, 49,5 plantillas por pliegue) da 0,6651; LOGO 0,6747. El hueco es el artefacto de `StratifiedGroupKFold(2)` que deja 3,9 familias por pliegue (7,8 por semilla) sin plantilla de entrenamiento. Para las familias de 2 plantillas, P2 «separadas» ya da 0,86–0,93 de F1 y «juntas» da 0.
2. «3,9 familias por pliegue… Es una parte cuantificable del hueco»: reemplazar por «es el 95 % del hueco» (+0,203 de +0,213).
3. «LOGO (0,675) está +0,03 por encima [de P2ret], lo esperable por 28 plantillas más de entrenamiento»: no se sostiene; el efecto del tamaño de entrenamiento medido es ~0,01 (P2bal→LOGO); la diferencia con P2ret es de denominador y de diseño de la prueba (P9).
4. Tabla de 50 semillas: la columna «± sd» de LOGO no puede decir 0. Poner sd bootstrap por plantilla 0,047 / 0,044 e IC por plantilla **[0,539; 0,722] / [0,636; 0,809]**, y dejar el IC por notas solo como nota al pie (no es válido con notas correlacionadas).
5. «⚠️ El IC de este último [M.6 − texto bajo LOGO] sale degenerado… queda como pendiente menor»: resuelto, **[+0,04; +0,15]** por bootstrap por plantilla (P(Δ ≤ 0) = 0).
6. «Comprobación directa: coseno máximo 0,8995… 0 pliegues con casi-copia»: agregar la frase que falta: **65 de 149 notas tienen vecino de entrenamiento en [0,80; 0,90) y ahí LOGO acierta 0,923; con colchón 0,80 LOGO-texto baja a 0,4721**; re-agrupando a 0,85 da 0,5172 y a 0,80 da 0,3379.
7. «Las cinco que llegan a 1,00 son familias de 2 plantillas… Es la evidencia por familia del mecanismo»: reemplazar por la tabla de contención (P7). Las cinco son pares de notas contenidas una en otra (0,82–0,91); con fusión por contención ≥ 0,8 las cinco dan 0 y LOGO-texto queda en 0,4186 (30) / 0,5980 (21 evaluables). Y agregar: SUNCRYPT `suncrypt.html` extrae 35 palabras que son el final literal de `note_pcrisk.txt`.
8. «El “50 %” se supera con holgura bajo LOGO… IC bootstrap entero por encima de 0,50»: condicionar explícitamente al criterio de casi-duplicado canónico (coseno char 0,90, sin contención). Es verdad bajo ese criterio y falso bajo 0,85, 0,80 o contención ≥ 0,8.
9. «Reportar los tres protocolos juntos… P1 / LOGO / P2»: agregar P2bal y P2ret a la tabla, y las filas que no superan 0,50.
10. «Autocrítica que hay que escribir: … nunca excluyó LOGO, que es el protocolo estándar para exactamente este caso»: reescribir la autocrítica hacia lo que realmente falló: la partición canónica de P2 no garantizaba entrenamiento por familia, y nadie lo midió hasta el 9/9 (y se le dio la lectura equivocada).
11. Bloque «PREREGISTRO — P2-LOGO»: anotar que **no quedó en git** (`ESTADO_TESIS.md` sin commit desde `750a069`, 2026-08-20) y que la única marca temporal es el docstring de `059cf5f` (16:29:33) frente a `logo_resumen.csv` (17:00:05). Pedir a Romina el commit del documento para que los próximos preregistros sean verificables.
12. Bloque «M.3 bajo LOGO»: agregar que las 90 notas resueltas por la regla son 90 aciertos (acierto de la regla 1,000 bajo LOGO), y que la «frase para la defensa» hereda las dos salvedades de arriba (umbral de agrupamiento y contención).
13. Bloque «EL TUTOR PIDE MÉTRICAS»: agregar MCC bajo LOGO (0,7301 / 0,8522) y aclarar que **CV(F1) por familia no existe bajo LOGO** (partición única); el sustituto es la sd bootstrap por plantilla.
14. `curva_aprendizaje_notas.py` L136 y L209 (código, no ESTADO): «grupo 55 = DHARMA + PHOBOS» es 53 sobre el corpus de 149; el id depende del orden del corpus. Cambiar el comentario para que no se cite como dato.
15. HANDOFF, encabezado «⚠️⚠️ LEER PRIMERO — P2-LOGO cambia la escala»: mismas correcciones 1, 4, 6, 7 y 8; y «El umbral de 0,50 del tutor se supera con IC entero» pasa a llevar la condición de la corrección 8.

---

## 4. Para el tutor (cinco líneas)

Se pidió una revisión independiente de LOGO y no resistió como cifra principal, aunque sí resistió en dos cosas: no hay fuga por código (vectorizador y diccionario se ajustan solo con entrenamiento) y el 0,6747 supera 0,50 incluso con un bootstrap por plantilla, que es el correcto ([0,539; 0,722]). Lo que no resistió es la explicación: el salto de 0,46 a 0,67 no es «más entrenamiento» sino que la partición de 2 pliegues de sklearn dejaba en cada corte ~4 familias sin ninguna plantilla para entrenar, con F1 = 0 forzado; repartiendo las plantillas de forma balanceada, con la misma mitad de entrenamiento, P2 da 0,665 ± 0,032, y LOGO agrega solo +0,01. Tampoco resistió la definición de «plantilla nunca vista»: las cinco familias que llegan a 1,00 son pares de notas donde una es la otra con un párrafo de más (contención 0,82–0,91 que el coseno no ve), y al juntarlas LOGO cae a 0,42 sobre 30 familias. La propuesta concreta es corregir la partición del protocolo canónico (P2bal, a 50 semillas), declarar la sensibilidad al criterio de casi-duplicado con la tabla completa de protocolos, y no presentar LOGO como «el protocolo que nadie probó» sino como una fila más.

---

## 5. Referencias verificadas (búsqueda web del 17/9; título, revista, volumen y páginas confirmados)

- Kohavi, R. (1995). *A study of cross-validation and bootstrap for accuracy estimation and model selection.* IJCAI-95, pp. 1137–1145.
- Arlot, S. & Celisse, A. (2010). *A survey of cross-validation procedures for model selection.* Statistics Surveys 4, 40–79. doi:10.1214/09-SS054.
- Bengio, Y. & Grandvalet, Y. (2004). *No unbiased estimator of the variance of K-fold cross-validation.* JMLR 5, 1089–1105.
- Varoquaux, G. (2018). *Cross-validation failure: small sample sizes lead to large error bars.* NeuroImage 180, 68–77. doi:10.1016/j.neuroimage.2017.06.061.
- Roberts, D. R. et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.* Ecography 40(8), 913–929. doi:10.1111/ecog.02881.
- Saeb, S. et al. (2017). *The need to approximate the use-case in clinical machine learning.* GigaScience 6(5), gix019.
- Forman, G. & Scholz, M. (2010). *Apples-to-apples in cross-validation studies: pitfalls in classifier performance measurement.* ACM SIGKDD Explorations 12(1), 49–57.
- Field, C. A. & Welsh, A. H. (2007). *Bootstrapping clustered data.* JRSS-B 69(3), 369–390.
- Cawley, G. C. & Talbot, N. L. C. (2010). *On over-fitting in model selection and subsequent selection bias in performance evaluation.* JMLR 11, 2079–2107.
- Gelman, A. & Loken, E. (2013). *The garden of forking paths: why multiple comparisons can be a problem, even when there is no "fishing expedition" or "p-hacking" and the research hypothesis was posited ahead of time.* Manuscrito, Columbia University (verificado el manuscrito; la versión de American Scientist 2014 **no se verificó**: citar el manuscrito).
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, 2.ª ed., Springer, §7.10 (ya en `bibliography.bib` como `hastie2009`; la sección se cita de memoria, verificar página antes de usarla).
