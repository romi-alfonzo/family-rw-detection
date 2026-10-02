# Encargo: revisión independiente y a fondo de la propuesta «LOGO» (2026-09-17)

**Para quien lee esto en un chat nuevo.** Sos un revisor **independiente y escéptico**. Otro chat
(el que escribió este archivo) propuso adoptar un protocolo de evaluación nuevo, «LOGO», que hace
que el frente de notas supere el umbral del 50 % que puso el tutor. Tu trabajo es **intentar
tirarlo abajo**. Si la propuesta resiste, mejor; si no resiste, decirlo con evidencia es el mejor
resultado posible, porque lo alternativo es que lo tire abajo el tutor o el jurado.

Reglas de la casa (leer `CLAUDE.md` primero; mandan sobre este encargo):
- Verificar, no recordar: **no confíes en ninguna cifra de `ESTADO_TESIS.md` ni de este archivo**
  sin abrir el archivo de resultados o el script que la produce. Este encargo puede tener errores.
- Español en todo. Nunca `Co-Authored-By` en commits. Solo el código de `2_codigo/` se commitea
  (a `develop`, con push); los documentos no. Nunca commitear datos. No tocar los `.tex`.
- No tocar el corpus (`3_datos/`). No descargar nada. Scripts de análisis nuevos: en el scratchpad
  o, si merecen quedar, en `2_codigo/` con commit.
- No le preguntes nada al chat original: la independencia es el punto. Si algo no se puede
  verificar, escribilo como «no verificable» en el informe.

---

## 1. Qué se está proponiendo (la afirmación a revisar)

Tesis de grado FP-UNA (Romina Alfonzo y Carlos Urdapilleta, tutor Prof. Cristian Cappo):
clasificación de familia de ransomware a partir de notas de rescate. Frente de notas: **149 notas,
99 plantillas (grupos de casi-duplicados), 30 familias** (las de NapierOne). Clasificador
congelado: TF-IDF vista «combinado» + LinearSVC(C=1, class_weight=balanced). Métrica principal:
macro-F1 sobre 30 clases (azar ≈ 0,033).

El tutor dijo (9/9/2026): «si no superamos 50 % no sirve el experimento ni el proyecto». Y por
WhatsApp: «¿y qué pasa con las familias que tienen una sola plantilla? No se procesa. Por eso
necesitamos métricas». Hay un PDF suyo con una lista de métricas (estaba en
`C:\Users\Romina\Downloads\Ransomware - Notas de rescate insuficientes.pdf` el 9/9; puede haberse
movido a `5_bibliografia/` o `6_notas_trabajo/`; buscarlo).

**Protocolos en juego** (todos sobre el mismo corpus, misma vista, mismo clasificador):
- **P1**: StratifiedKFold 2 pliegues por nota. Las casi-copias cruzan la partición. ~0,80. Inflado.
- **P2** (canónico hasta ahora): StratifiedGroupKFold 2 pliegues, grupo = plantilla. Texto
  **0,4593 ± 0,075**, cascada M.6 **0,5191** (50 semillas).
- **P2ret** (dentro de la curva de aprendizaje B.1): retiene UNA plantilla por familia al test,
  100 repeticiones; texto **0,6433 ± 0,061** en k=todo.
- **LOGO** (lo nuevo): `LeaveOneGroupOut` por plantilla, 99 pliegues deterministas. Texto
  **0,6747**, IC bootstrap por notas [0,567; 0,712]; M.6 **0,7742** [0,663; 0,810].
- **L**: mundo cerrado tipo Lemmou (1-NN sin partición). ~0,78. No comparable.

**La propuesta concreta:** reportar P1, LOGO y P2 juntos, con **LOGO como cifra principal** del
frente de notas, y escribirlo en la tesis como sección que se agrega. Argumentos del proponente:
(a) LOGO conserva la misma garantía que P2 (la plantilla de prueba nunca está en entrenamiento;
comprobado: coseno máximo prueba-entrenamiento 0,8995 en los 99 pliegues, 0 violaciones del
umbral 0,90); (b) P2 entrena con la mitad y deja **3,9 familias por pliegue sin ninguna plantilla
de entrenamiento**; (c) P2ret ya daba 0,64 y LOGO lo confirma (+0,03 por 28 plantillas más);
(d) Δ pareado LOGO−P2 = +0,2155 [+0,194; +0,237] texto y +0,2550 [+0,233; +0,278] M.6, 50/50
semillas; (e) 26 de 30 familias > 0,50 bajo LOGO-M.6; (f) con abstención (M.3) bajo LOGO:
contesta el 85 % y acierta el 94 % (umbral 0,50).

---

## 2. Dónde está todo

Código (`2_codigo/`):
- `protocolo_logo.py` — el script de la propuesta (commits `059cf5f`, `7dd6db9`, `84187b7`).
- `clasificador_notas_v2.py` — canónico: `agrupar_neardups` (umbral 0,90, char_wb 3-5),
  `vectorizador`, `obtener_modelos`, `N_FOLDS`, `cargar_corpus`.
- `cascada_combinada_notas.py` — M.6 canónico (variante adoptada `privados_sin_circ_MAS_NOMBRE`).
- `abstencion_notas.py` — M.3, con `--protocolo LOGO`.
- `curva_aprendizaje_notas.py` — B.1, incluye P2ret (`_splits`).
- `grafo_marcadores.py` — `extraer_marcadores`, `_terminos_circulares` (B.3, circularidad).
- `validar_procedencia.py`, `anotar_manifiesto.py` — procedencia del corpus.

Resultados (`4_resultados/`):
- `resultados_protocolo_logo_149_50sem/` + `_log_logo_149_50sem.txt` (definitivo, 50 semillas).
- `resultados_protocolo_logo_149/` + `_log_logo_149.txt` (10 semillas; incluye la comprobación de
  fuga).
- `resultados_abstencion_149_LOGO/` + `_log_m3_149_LOGO.txt`; `resultados_abstencion_149/` (P2).
- `resultados_curva_149/b1_curva_por_repeticion.csv` (columnas `n_fam_sin_train`,
  `n_plantillas_train`, `n_notas_evaluadas`) + `_log_resumen_cap4_149.txt` (techos, P2ret).
- `resultados_cascada_combinada_149/` + `_log_m6_149.txt` (M.6 canónico, 50 semillas).
- `resultados_notas_149/` (P1/P2 canónicos a 50 semillas).
- `resultados_grafo_marcadores_149/` (circularidad, B.3).

Datos (solo lectura): `3_datos/manifiesto_corpus_v2.csv` (procedencia, `nombre_genuino`,
`origen_nombre`, `placeholders`, `redaccion_fuente`), `3_datos/nombres_notas/`.

Documentos: `ESTADO_TESIS.md` — buscar los bloques «PREREGISTRO — P2-LOGO», «RESULTADO P2-LOGO
(2026-09-09)», «RESULTADO P2-LOGO a 50 semillas (2026-09-10)», «PRECISIÓN a lo anterior», «M.3 bajo
LOGO», «Nota operativa». `PENDIENTE_REDACCION.md` §K (familias de una plantilla).
`6_notas_trabajo/HANDOFF_2026-08-25_limpieza_y_proximos_pasos.md` (encabezado). Tesis:
`1_documento/Plantilla_de_Tesis___Romina_Carlos/resultados_notas_ampliacion.tex` líneas ~389-419
(comparación de protocolos ya escrita: L / P1 / P2).

Todo corre local en minutos (LOGO a 50 semillas: ~10 min desde el commit `84187b7`).

---

## 3. Preguntas que tenés que contestar con evidencia (mínimo)

**A. Legitimidad del protocolo**
1. ¿Leave-one-group-out por plantilla es un protocolo estándar y defendible para este caso
   (familias con 1–2 plantillas), o es una elección optimista? Buscá literatura citable
   (validación cruzada agrupada, leave-one-subject-out, Varoquaux/Kohavi/etc.) y decí qué se
   puede citar y qué no. **No inventes referencias**: si no la podés verificar, no va.
2. ¿Es válido el **IC por bootstrap sobre notas** cuando las notas de una misma plantilla están
   correlacionadas? Lo correcto podría ser bootstrap **por plantilla** (cluster bootstrap). Calculalo
   y compará: si el IC por plantilla incluye 0,50, la afirmación «supera el 50 % con IC entero»
   se cae. Esto es probablemente **lo más importante del encargo**.
3. LOGO es una partición única y determinista: ¿la ausencia de varianza entre semillas oculta
   incertidumbre que P2 sí muestra (sd 0,075)? ¿Cómo se debe reportar honestamente?

**B. Fuga de información**
4. El máximo coseno prueba-entrenamiento fue 0,8995 con umbral 0,90: es «justo» por construcción
   (los grupos se definen con ≥ 0,90). Mirá la **distribución** del máximo coseno por pliegue:
   ¿cuántas notas de prueba tienen un vecino en entrenamiento con coseno 0,80–0,90? ¿Qué pasa con
   el macro-F1 de LOGO si se re-agrupa con umbral 0,80 o 0,70? (`agrupar_neardups` acepta umbral.)
   Si la cifra se derrumba al bajar el umbral, la ganancia era fuga blanda.
5. El agrupamiento es por componentes conexas (encadenamiento simple). ¿Hay grupos que mezclan
   familias (se mencionan grupos 6 y 55 en `curva_aprendizaje_notas.py`)? ¿Cómo los trata LOGO?
6. ¿El vectorizador se ajusta solo con el pliegue de entrenamiento? ¿El diccionario de la regla
   M.6 (IOC + nombre) se construye solo con entrenamiento? Verificalo en el código, línea por línea.

**C. De dónde sale la ganancia**
7. Las 5 familias que pasan a F1 = 1,00 bajo LOGO son todas de **2 plantillas** (SUNCRYPT, CUBA,
   NETWALKER, BLACKMATTER, DARKSIDE). Abrí sus notas: ¿la otra plantilla contiene el **nombre de
   la familia** u otro término circular (`_terminos_circulares`)? Si la ganancia de LOGO viene de
   que el texto dice «DarkSide», hay que decirlo y medir LOGO **con** filtro de circularidad
   (`excluir_circulares=True` está en la cascada; en texto puro habría que enmascarar términos).
8. ¿Es cierto que bajo P2 quedan 3,9 familias por pliegue sin entrenamiento? Verificalo desde
   `b1_curva_por_repeticion.csv` o recomputando con `StratifiedGroupKFold(2)`. ¿Y explica eso el
   hueco, o el hueco es sobre todo el tamaño de entrenamiento? Separá los dos efectos si podés
   (p. ej., P2 excluyendo de la métrica las familias sin train).
9. P2ret (0,6433, evalúa ~45 notas por repetición) vs LOGO (0,6747, evalúa las 149): ¿son
   comparables? ¿El +0,03 es tamaño de entrenamiento o denominadores distintos?

**D. Lo que pidió el tutor**
10. Leé su PDF y decí, métrica por métrica, qué está y qué falta **bajo LOGO** (MCC no está en la
    salida de `protocolo_logo.py`). ¿«50 %» de qué métrica? ¿macro-F1 sobre 30 clases, sobre 28
    evaluables, exactitud, por familia? Proponé la tabla exacta que hay que mostrarle.
11. Familias de una sola plantilla (BADRABBIT, CRYPTOLOCKER): ¿están en el denominador del
    macro-F1 (30 clases)? ¿Debe reportarse también «sobre 28 evaluables»? ¿Cómo se responde
    honestamente su «no se procesa»?

**E. Riesgo de presentación**
12. ¿Un jurado puede ver esto como «cambiar el protocolo hasta que el número pase» (jardín de
    senderos)? Revisá el bloque de preregistro en `ESTADO_TESIS.md`: ¿se escribió antes del
    resultado? ¿Se reportaron las predicciones falladas? ¿Cómo se debe presentar la comparación
    de protocolos para que sea un aporte metodológico y no una excusa?
13. Cualquier otra cosa que vos encuentres. Buscá activamente lo que el proponente no miró.

---

## 4. Entregable

Escribí `6_notas_trabajo/REVISION_LOGO_2026-09-17_informe.md` con:
1. **Veredicto en la primera línea**: «adoptar LOGO como cifra principal» / «adoptar con estas
   salvedades obligatorias» / «no adoptar». Sin diplomacia.
2. Por cada pregunta: qué hiciste, qué archivo/línea/número lo sostiene, y la conclusión.
3. Los experimentos nuevos que corriste (comandos, salidas, dónde quedaron). Si agregaste código
   a `2_codigo/`, commit y push a `develop` en español, sin coautoría.
4. **Lista de correcciones** a `ESTADO_TESIS.md` (no las apliques: listalas; el chat original o
   Romina las aplican).
5. Un párrafo de 5 líneas para que Romina le explique al tutor **lo que resistió y lo que no**.

Al terminar, agregá en `ESTADO_TESIS.md` una sola línea al final del bloque «RESULTADO P2-LOGO a
50 semillas»: `→ Revisión independiente: ver 6_notas_trabajo/REVISION_LOGO_2026-09-17_informe.md
(veredicto: …)`. Nada más en ese archivo.
