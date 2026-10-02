# Encargo: revisión científica independiente de la tesis, con el estándar de un paper internacional

**Preparado el 2026-09-28 y actualizado el mismo día.** Romina decidió crear la skill ahora, con un
rol preciso: **un científico que verifique los datos y que todo sea real y replicable, como en la
revisión de un paper internacional.**

**Para quien lee esto en un chat nuevo.** Este archivo tiene todo lo necesario; no hace falta el
chat que lo escribió. **No le preguntes nada a ese chat: la independencia es el punto.** Quien
escribió la tesis no es buen revisor de sí mismo (ver sección 4).

---

## 0. Qué está en movimiento (al 2026-09-29)

Leé los últimos bloques de `ESTADO_TESIS.md` (justo antes de `## 7. Reglas`) para ver si esto ya
cambió.

- **Exp. 2h: corrido (job 4096) e incorporado** a la tesis y a los informes el 29-09. Las tablas
  `tab:exp2e_tipos` y `tab:exp2g_tipos` ya son las ponderadas del 2h.
- **Frente de notas: cerrado** según su sesión (29-09), salvo una frase (≈ l. 1107 de
  `resultados.tex`) que espera el sí o el no de Romina. Tiene su verificador:
  `python 2_codigo/cifras_finales.py` audita 67 cifras contra sus archivos de origen.
- **Frente de archivos: texto cerrado, verificador pendiente.** Los resultados de 2e a 2h (jobs
  4058, 4059, 4079, 4083, 4091 y 4096) y el diagnóstico 4082 están **solo en el clúster**; `generar_figuras_cap4.py`
  usa para el 2e y el 2g cifras transcritas del log mientras no se bajen. Hasta que estén en
  `4_resultados/` y exista un verificador equivalente al de notas, esas cifras solo se pueden
  rastrear hasta el log, no hasta el CSV.
- **La conclusión no está escrita**, por decisión: se escribe al final. No es un hallazgo.
- **La skill ya está creada** (`.claude/skills/revisar-tesis/`, 29-09). **La revisión NO se corre
  hasta que la tesis esté cerrada** —decisión de Romina, 29-09—: el Exp. 2h incorporado, la
  conclusión escrita y el frente de notas terminado. Si llegás acá y algo de eso sigue abierto,
  decíselo a Romina y no empieces.

---

## 1. Qué construir

**Paso 1 — una skill reutilizable** en `.claude/skills/revisar-tesis/SKILL.md`, con el rol de la
sección 2, el diseño de la sección 3 y la lista de chequeo de la sección 4. Tiene que servir para
volver a correrla antes de entregar y cada vez que cambie una parte. Si está disponible la skill
`skill-creator`, usala para armarla.

**Paso 2 — correrla recién cuando la tesis esté cerrada** (sección 0) y entregar el informe de la
sección 6.

**Modelos:** los revisores con **Claude Fable 5.1** (en el tool de agentes, `model: "fable"`). La
verificación de hallazgos alcanza con **Opus 5.5** (`model: "opus"`). No usar Sonnet ni Haiku: un
error que llega al tribunal cuesta más que la diferencia.

**Orquestación:** subagentes en paralelo. Si Romina lo pide explícitamente («usá un workflow»), un
workflow es más robusto y se puede retomar; sin ese pedido, no lances workflows.

---

## 2. Tu rol

**Revisor de un paper internacional y comité de reproducibilidad.** Revisá como si la tesis fuera
un artículo enviado a una conferencia o revista internacional de seguridad —IEEE S&P, USENIX
Security, ACM CCS, NDSS; IEEE TIFS, *Computers & Security*— y como su comité de evaluación de
artefactos (las insignias ACM *Artifacts Available*, *Artifacts Evaluated – Functional* y *Results
Reproduced*). Para cada afirmación, tres preguntas:

1. **¿Es real?** ¿Sale de un archivo de resultados (CSV, log, manifiesto), no de un resumen?
2. **¿Se puede replicar?** ¿Hay script, datos, semillas, hiperparámetros y versión del corpus
   declarados, y alguien de afuera podría obtener el mismo número?
3. **¿La conclusión se sigue de la evidencia?** ¿Sin fugas ni atajos del conjunto de datos, con la
   métrica y la base correctas, con comparaciones pareadas, sin afirmar más de lo medido?

**Reproducir, no solo rastrear.** Donde se pueda, volvé a ejecutar y compará con lo publicado:
- **Frente de notas:** el corpus está en `3_datos/` y los scripts en `2_codigo/`, así que se puede
  re-ejecutar localmente.
- **Frente de archivos:** NapierOne está solo en el clúster. Juntá todo lo que haga falta reproducir
  en **UNA sola corrida**, porque Romina no puede esperar un job de dos horas por cada duda. Si podés,
  **reimplementá de forma independiente** el cálculo de la cifra de cabecera (0,9998) en vez de
  reusar el código de la tesis: es la prueba más fuerte de que el número es real.

**También dueño del proyecto.** Los resultados medidos son **precedente, no verdad**. Si ves una
forma mejor —un protocolo, una validación, una lectura distinta—, proponela, y si se puede medir con
los datos disponibles, **medila** antes de proponerla. El modelo a seguir es la revisión de LOGO del
17-09: además de tirar abajo LOGO, midió **P2bal**, que terminó siendo el protocolo canónico del
frente de notas. **Separá siempre lo verificado de lo propuesto.**

---

## 3. El diseño

**Paso 0 — inventario de cifras (script, no modelo).** Extraer cada número de los `.tex` del
documento (`resultados.tex` y lo que incluye, `metodologia.tex`, `introduccion.tex`, `resumen_*.tex`,
`conclusion.tex`), con archivo, línea y ~200 caracteres de contexto. Es la lista que los revisores
recorren: garantiza que ninguna cifra quede sin rastrear.

**Paso 0b — inventario de conclusiones (script, no modelo).** El inventario de cifras no ve los
errores de explicación, porque la mayoría no tiene número (ver «Dos clases de error», abajo).
Extraer también las oraciones que concluyen algo: las que llevan *porque, por lo tanto, por eso, es
decir, se debe a, lo que sostiene, demuestra, confirma, garantiza, resulta más estable, generaliza,
no depende, no arrastra, siempre, ninguna, todas, exactamente*, con archivo y línea.

**Paso 1 — cuatro revisores en paralelo**, cada uno sin el contexto de los otros:

| Revisor | Qué cubre |
|---|---|
| A. Frente de archivos | cap. 4, Experimentos 1 a 2h y la síntesis del frente |
| B. Frente de notas | Experimento 3 y todo lo integrado del frente de notas |
| C. Metodología y reproducibilidad | cap. 3: corpus, protocolos (P1, P2, P2bal), casi-duplicados, definición de campaña; y para todo el documento: ¿alcanza lo declarado para replicar cada experimento? |
| D. Coherencia | resumen, introducción, síntesis de cada frente, tabla comparativa, figuras contra texto: ¿dicen lo mismo con las mismas cifras? ¿se cumplen los objetivos planteados en la introducción? |

**Paso 2 — reproducción:** lo que se pueda re-ejecutar localmente, se re-ejecuta; lo del clúster,
en una sola corrida (sección 5).

**Paso 3 — verificación de cada hallazgo** por otro agente, antes de que entre al informe. Sirve
para descartar falsos positivos (en la sesión del 28-09 una búsqueda automática marcó como error
una frase que ya estaba corregida).

### Dos clases de error, dos formas de buscarlas

Distinción propuesta por la sesión de notas el 29-09, a partir de lo que encontraron las dos
sesiones principales ese día:

- **Errores de cálculo:** la cifra no coincide con su archivo de resultados. Se encuentran
  **corriendo los scripts** (`cifras_finales.py`, el inventario del paso 0, la reproducción del
  paso 2).
- **Errores de explicación:** la cifra está bien, pero lo que se concluye de ella nunca se midió.
  Solo se encuentran **leyendo y preguntando, oración por oración: «¿esto que se concluye, está
  medido? ¿qué medición lo sostiene?»**. Las frases corregidas en la revisión de interpretaciones
  del 29-09 eran todas de esta clase y ningún verificador automático habría dado con ellas
  (la frase de notas de la l. 1107 seguía escrita el 29-09 y `cifras_finales.py` daba 67 de 67):
  ejemplos en la fila «Conclusión medida»
  de la sección 4 y el detalle en `ESTADO_TESIS.md`, bloque «INTERPRETACIONES NO MEDIDAS DEL
  FRENTE DE ARCHIVOS».

**Sin sobrecorregir.** No todo lo interpretativo está sin medir: hay que separar, en cada frase,
la parte medida de la no medida. Ejemplo real: en la síntesis del frente de archivos, que el
tamaño del archivo sea necesario y no suficiente **sí está medido** (ablación del 2e: el tamaño
solo alcanza 0,3272 de macro-F1); lo único no medido es el **mecanismo** (modo de cifrado,
tamaño del añadido). Marcar la frase entera como «no medida» sería un falso positivo; lo correcto
es pedir que el mecanismo figure como explicación propuesta.

---

## 4. Lista de chequeo — derivada de errores REALES de este proyecto

No es una lista genérica: cada renglón es un error que se cometió y se encontró. Los del 28-09
los encontró la misma sesión que los había cometido, recién al re-verificar: la prueba de que hace
falta un revisor de afuera.

| Chequeo | Caso real |
|---|---|
| **Cada cifra rastreada a su archivo de resultados** (CSV, log, manifiesto), no a `ESTADO_TESIS.md` | un 0,9687 que venía de un protocolo reemplazado; el techo de combinar los frentes calculado con cifras viejas de los dos lados |
| **Figuras generadas de la misma corrida que cita el texto** | la barra del Exp. 2c decía 0,909 (corrida única) al lado de un texto con 0,912 ± 0,002 (diez semillas); la de firmas del 2b, 54 % de cobertura contra 57,2 % de la tabla |
| **Recuentos de la misma versión del conjunto y del mismo criterio** | «quince familias con firma» (unanimidad), «once con solo extensión» (base de 29) y la figura con 16 y 12 (criterio 0,90, 30 familias), en la misma página |
| **Métrica y base declaradas junto al número** (exactitud / macro-F1 / acierto donde contesta; familias, archivos o notas; semillas; protocolo) | «el residuo se reduce de seis a cuatro», que mezclaba una corrida de 10 semillas con una de 1; «de 0,912 a 0,9998 de macro-F1», cuando 0,912 es exactitud |
| **El mismo modelo en todas las validaciones que se comparan** | la validación por tipos del 2e-c, 2f y 2g se corrió SIN ponderación de clases; la publicada del 2c y todas las validaciones cruzadas, CON. El script decía «como allí», y era falso |
| **Composición del conjunto por familia antes de interpretar una validación** | BLACKMATTER son 988 imágenes de 1.000 muestras: al sacar las imágenes queda sin entrenamiento y su F1 es 0 en ese pliegue. Se atribuyó al renombrado sin mirar la carpeta |
| **Lo que imprime un script, verificado contra lo que el código calcula** | «Familias que renombran por completo (solo entrenamiento)» listaba cualquier familia con UN archivo sin tipo; pasó a la tesis como «BLACKMATTER solo entrena», y era falso |
| **Fugas y atajos del conjunto de datos** | los rasgos de nombre aprendieron el `-fromweb` que NapierOne puso en los jpg (`0001-jpg-fromweb.jpg.avos2`): 0,9998 en validación cruzada, 0,2167 con los jpg fuera. ¿Hay duplicados entre pliegues? (el Exp. 2h los busca por primera vez) |
| **La cifra canónica pasó las mismas validaciones que la que reemplaza** | el 0,936 se adoptó sin haber pasado dejar-un-tipo-fuera (cuando se midió, pasó); el 0,9998 del 2f **no** la pasó; el del 2g sí |
| **La afirmación no es más fuerte que la evidencia** | «los 988 archivos tienen entropía de cabecera 0,88–6,46», medido en 6; «residuo genuino», sin demostrar; «sin una sola excepción ≥ 0,99», con BADRABBIT en 0,978 |
| **Comparaciones pareadas de verdad** | «0,9120 → 0,9123» como antes/después, entre corridas con distintas semillas |
| **Un término, un significado** | «configuración canónica» quería decir solo bytes en el 2d, bytes + estructura en el 2e-c, y ya no era ninguna de las dos |
| **Lo citado está en la fuente, leída en la fuente** | un resumen automático dijo que un paper ponía a DARKSIDE con «cifrado completo»; la tabla original la lista entre las de cifrado intermitente |
| **Nada pendiente que ya esté hecho, nada hecho que siga pendiente** | «quedan dos mediciones» en un informe, cuando estaban hechas |
| **Conclusión medida, no solo cifra medida** (errores de explicación, sección 3) | 29-09: las firmas binarias «resultan más estables entre campañas que la extensión», con una campaña por familia; el contenido «no arrastra la limitación de campaña», cuando solo esquiva el renombrado; el mecanismo de tamaño y entropía escrito como hecho; una figura que ponía lado a lado dos ganancias medidas con distinta dureza (notas bajo P2bal, archivos en validación cruzada) sin decirlo |
| **Hipótesis de mecanismo presentadas como hechos** | cinco hipótesis de mecanismo del frente de archivos fallaron en una semana; la sexta (BLACKMATTER) se confirmó solo al mirar la carpeta |
| **Predicciones preregistradas: ¿están también las que fallaron?** | el proyecto preregistra en el docstring de cada script, commiteado antes de correr; revisar que los fallos estén en la tesis, no solo los aciertos |
| **Afirmaciones de la literatura previa, exactas** | Lemmou et al.: su F = 0,920 es la tarea binaria, no clasificación de familia (el tutor conoce el paper); el 71,93 % de ID Ransomware es sobre 57 notas de 22 familias, no sobre el corpus |
| **Replicable** | script en `2_codigo/`, job en `2_codigo/slurm/`, semillas, hiperparámetros, versión del corpus y del código (commit) declarados para cada experimento |

---

## 5. Reglas de la casa

Leé **`CLAUDE.md` primero**: manda sobre este encargo. Lo esencial:

- **Verificar, no recordar.** No confíes en ninguna cifra de `ESTADO_TESIS.md`, de los informes ni de
  este archivo sin abrir el archivo de resultados o el script que la produce. Este encargo puede
  tener errores.
- Todo en español. **Nunca `Co-Authored-By` en los commits** (es trabajo académico de Romina).
- Solo el código de `2_codigo/` se commitea, a `develop` con push; los scripts de reproducción que
  escribas, también. Los documentos y el LaTeX, no. **Nunca commitear datos**: el corpus son notas y
  archivos cifrados auténticos (malware real).
- **No edites los `.tex`.** El resultado es un informe; qué se corrige lo deciden Romina y Carlos.
- No toques el corpus (`3_datos/`) ni descargues nada.
- **Clúster:** todo lo que necesites ahí, en **una sola corrida**, con predicciones escritas en el
  docstring y commiteadas antes de correr, puerta de entrada contra una cifra publicada y guardado
  incremental. `--nodelist=c2`, `--mem=` explícito, finales de línea LF, archivos a subir en
  `PARA_SUBIR_AL_CLUSTER/`, y cada comando que le pases a Romina empieza con
  `cd /scratch/ralfonzo/tesis &&`. Probalo antes con datos sintéticos.
- Lo que no se pueda verificar se escribe como **«no verificable»**, no se suaviza.

---

## 6. Qué entregar

Un informe en `6_notas_trabajo/REVISION_TESIS_<fecha>_informe.md`, con la forma de una revisión de
paper internacional:

1. **Resumen de la contribución**, en tus palabras: qué afirma la tesis.
2. **Veredicto**, como en un comité de programa (aceptar / revisión menor / revisión mayor /
   rechazar), justificado en un párrafo.
3. **Fortalezas.**
4. **Debilidades y hallazgos**, ordenados por gravedad (primero lo que cambia una conclusión). Cada
   uno con: dónde (`archivo.tex:línea`), qué dice, qué es verdad, la evidencia (el archivo que lo
   muestra) y si cambia alguna conclusión.
5. **Informe de reproducibilidad:** qué re-ejecutaste, qué dio y si coincide con lo publicado, con
   el nivel alcanzado por experimento (disponible / funcional / reproducido).
6. **Preguntas para los autores**, las que haría un revisor externo.
7. **Propuestas de mejora**, separadas de los hallazgos: qué, por qué, si se midió y con qué resultado.
8. **Lo no verificable**, dicho como tal.

Registrá también el resultado en `ESTADO_TESIS.md`, como pide `CLAUDE.md`.

---

## 7. Dónde está todo

| Qué | Dónde |
|---|---|
| Contrato de trabajo | `CLAUDE.md` |
| Estado, decisiones, todos los resultados | `ESTADO_TESIS.md` (~10.500 líneas; lo más reciente, antes de `## 7. Reglas`) |
| Pendientes y planes | `EXPERIMENTOS_PENDIENTES.md`, `PLAN_MEJORAS.md` |
| Pedidos del tutor | `6_notas_trabajo/reunion_2026-08-12_revision_resultados.md` |
| Scripts, con su preregistro en el docstring | `2_codigo/` · jobs en `2_codigo/slurm/` |
| Resultados locales | `4_resultados/` · los del clúster, en `/scratch/ralfonzo/tesis/resultados_*`. El reglamento declara el `/scratch` temporal, pero en la práctica no se limpia (verificado el 17-08: hay archivos de más de un año). Lo que se vaya a verificar igual tiene que estar bajado, porque el revisor no tiene acceso al clúster |
| Corpus de notas (local) y manifiesto | `3_datos/corpus_v2/`, `3_datos/manifiesto_corpus_v2.csv` |
| Archivos cifrados | NapierOne-small, solo en el clúster: `/scratch/ralfonzo/Napierone-small` |
| La tesis | `1_documento/Plantilla_de_Tesis___Romina_Carlos/` (`main.tex`, `latexmk -pdf`; 127 págs. al 29-09) |
| Figuras del cap. 4 | `2_codigo/generar_figuras_cap4.py` (lee los resultados; dice de qué corrida sale cada cifra) |
| Informes a Cappo | `6_notas_trabajo/informe_*_para_cappo.*`, `informe_cierre_*`, `INFORME_CAPPO_2026-09-28_frente_de_notas.md` |
| Traspaso del frente de notas | `6_notas_trabajo/HANDOFF_2026-09-26_cerrar_con_la_cascada.md` |
| **El precedente de revisión que funcionó** | `6_notas_trabajo/REVISION_LOGO_2026-09-17_encargo.md` y `_informe.md` — leelos como modelo |
| Errores ya encontrados y corregidos | en `ESTADO_TESIS.md`: «VERIFICACIÓN DE LO QUE VA A CAPPO» y «ERROR PROPIO DETECTADO AL VERIFICARLO» (28-09) |

Identificación: tesis de grado de **Romina Alfonzo** y **Carlos Urdapilleta**, FP-UNA. Tutor: Prof.
**Cristian Cappo**. Título: *Detección de familias de ransomware en base a archivos encriptados y
notas de rescate*. Dos frentes que van **separados** —notas y archivos—, núcleo canónico de 30
familias de NapierOne.
