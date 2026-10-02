# Traspaso — cerrar el frente de notas con LA CASCADA como sistema (2026-09-26)

**Fase:** cierre y redacción. Romina decidió presentar **la cascada** como el sistema del frente
de notas. El tutor (Prof. Cristian Cappo) quiere cerrar. Leer `CLAUDE.md` primero: manda sobre
todo lo de acá.

---

## 1. Qué es el sistema que se presenta

**La cascada (nombre interno M.6, variante adoptada `privados_sin_circ_MAS_NOMBRE`).** Ante una
nota de rescate decide en tres pasos, como un analista:

1. **Indicadores propios de la nota** (correos de contacto, direcciones de pago, URLs, formato del
   identificador de víctima), extraídos con `grafo_marcadores.extraer_marcadores`. Se construye un
   diccionario valor → familia **solo con el pliegue de entrenamiento**, y se descartan los valores
   que en entrenamiento aparecen en más de una familia (filtro de genéricos).
2. **Nombre genuino del archivo de la nota**, cuando está auditado.
3. **El texto**, con TF-IDF vista «combinado» + LinearSVC(C=1, class_weight=balanced).

Los pasos 1 y 2 responden por unanimidad: si las claves apuntan a una sola familia, se contesta;
si hay conflicto o no hay coincidencia, decide el texto.

Código: `2_codigo/cascada_combinada_notas.py` (canónico, 4 variantes) y la implementación compacta
reusada por los scripts de protocolo en `2_codigo/protocolo_logo.py` (`dicc_privados`, `regla`,
`evaluar`), verificada equivalente a la variante adoptada.

---

## 2. Cifras vigentes (verificadas 2026-09-22 y 2026-09-26)

**Protocolo de cabecera: P2bal** — `2_codigo/protocolo_p2bal.py`, preregistro en el docstring,
commit `a936397` (2026-09-22 23:17:21, **antes** de correr). Log `4_resultados/_log_p2bal_149.txt`,
salidas en `4_resultados/resultados_protocolo_p2bal_149/`. 149 notas · 99 plantillas · 30 familias,
50 semillas.

| | macro-F1 (30) | IC 95 % | sobre 28 evaluables | exactitud | bal. | MCC |
|---|---|---|---|---|---|---|
| **Cascada** | **0,7417** | [0,7328; 0,7505] | 0,7946 | 0,8123 | 0,7798 | 0,8042 |
| Texto solo (desarme) | 0,6551 | [0,6454; 0,6648] | 0,7019 | 0,7191 | 0,7033 | 0,7092 |
| Cascada bajo P2 (reparto viejo) | 0,5191 | [0,4967; 0,5416] | 0,5562 | 0,6601 | 0,5778 | 0,6446 |

- **Aporte de las capas de reglas sobre el texto:** +0,0866 bajo P2bal.
- **Capa de reglas aislada** (medido bajo P2, `_log_m6_149.txt`): cobertura **0,4546** de las notas,
  **acierto 0,9755** donde aplica, Δ sobre texto +0,0599 [+0,0530; +0,0668], 50/50 semillas.
- **Familias:** 25/30 ≥ 0,50 y 21/30 ≥ 0,70 con la cascada. Bajo 0,50 quedan BADRABBIT 0,000 y
  CRYPTOLOCKER 0,000 (plantilla única, cero estructural), HELLOKITTY 0,293, RYUK 0,358, JIGSAW 0,499.
- **Δ P2bal − P2:** cascada +0,2225 [+0,199; +0,246], 50/50 semillas.

**Cómo se cita, siempre:** «sobre plantilla no vista **según el criterio de casi-duplicado por
coseno de caracteres 0,90**». Ese criterio no detecta **contención**; con contención ≥ 0,8 el corpus
pasa de 99 a 81 plantillas y de 28 a 21 familias evaluables. La limitación va pegada al número.

---

## 3. Lo que queda por medir antes de escribir (dos corridas, ~1-2 h de máquina)

1. **Abstención (M.3) bajo P2bal.** `2_codigo/abstencion_notas.py` tiene `--protocolo {P2,LOGO}`;
   **falta agregar P2bal** (importar `split_p2bal` de `protocolo_p2bal.py`). Hoy la frase de
   despliegue («contesta el X %, acierta el Y %») solo existe bajo P2 (0,646 / 0,900 con umbral
   0,50) y bajo LOGO, que está descartado. **Sin esto no hay frase de despliegue citable.**
2. **Desglose por similitud bajo P2bal.** Romina preguntó cuánto se acierta cuando la nota se
   parece a una conocida. Medido bajo LOGO (descartado): con hermana contenida ≥ 0,5 el acierto es
   1,000 (63 notas); sin hermana parecida, 0,793 con cascada y 0,573 con texto (82 notas). **Hay
   que rehacerlo bajo P2bal**: es lo primero que va a preguntar un jurado. El insumo está en
   `4_resultados/resultados_revision_logo_149/c_contencion_por_nota.csv` y el cálculo de contención
   en `2_codigo/revision_logo.py`.

Preregistrar ambas en el docstring del script y commitear **antes** de correr. Es la regla que
salió del incidente de LOGO.

---

## 4. Qué NO se escribe

- **LOGO no va a la tesis.** Descartado el 2026-09-17 por revisión independiente
  (`REVISION_LOGO_2026-09-17_informe.md`). No re-proponerlo.
- **No hay clasificador combinado** notas + archivos. Los dos frentes van separados.
- **La conclusión se escribe al final de todo.** Decisión tomada.
- **En la tesis solo se AGREGA.** No reescribir ni reordenar lo ya escrito.

---

## 5. Cómo presentar la cascada sin que parezca que se esconde algo

La cascada es **el sistema**; el texto solo es un **desarme interno** que muestra el aporte de cada
capa. Reportar las dos filas siempre juntas. Un jurado va a preguntar qué parte del acierto viene
de las reglas: la respuesta está medida (cobertura 0,4546, acierto 0,9755) y es una fortaleza, no
una debilidad, porque esas reglas se construyen **solo con entrenamiento** y se les aplica filtro
de genéricos.

Lo mismo con el salto 0,52 → 0,74: **el sistema no cambió**. La cascada se construyó el 24-08 y el
corpus cerró el 25-08; el 22-09 solo se corrigió el reparto de la partición, que regalaba F1 = 0 a
~2,86 familias por pliegue sin darles material de entrenamiento. Es corrección de la **medición**,
no mejora del **método**, y hay que decirlo con esas palabras.

---

## 6. Dónde escribir

- `1_documento/Plantilla_de_Tesis___Romina_Carlos/resultados_notas_ampliacion.tex` — ahí ya está la
  comparación de protocolos (L / P1 / P2, líneas ~389-419). La cascada y P2bal se **agregan** como
  secciones nuevas.
- `PENDIENTE_REDACCION.md` §K tiene el borrador sobre familias de plantilla única.
- Estado y decisiones: `ESTADO_TESIS.md`, bloque «P2bal — EL REPARTO DE P2 ARREGLADO».
- Recordar: el LaTeX y los documentos de estado se commitean **solo cuando Romina lo pide**; el
  código de `2_codigo/` se commitea y pushea a `develop` en el momento, en español, sin coautoría.
