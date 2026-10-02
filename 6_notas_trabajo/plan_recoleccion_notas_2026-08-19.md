# Plan de recolección de notas — decisión tomada el 2026-08-19

Romina decidió activar la rama **RECOLECTAR** del Sprint C. El objetivo sale de B.1 y es
finito: **llegar a 4 textos distintos (plantillas) por familia**. Fuente de los conteos:
`4_resultados/resumen_capitulo4/b1_familias_a_recolectar.csv` y
`b3_cohesion_por_familia.csv`.

## Prioridad 1 — donde duele: 17 notas en 9 familias

Familias con F1 por familia bajo (P2ret, 100 repeticiones) **y** plantillas faltantes. Ordenadas
por urgencia:

| Familia | Tiene | Faltan para 4 | F1 actual | Nota |
|---|---|---|---|---|
| **WASTEDLOCKER** | 1 | **3** | 0,00 | Con 1 plantilla es INEVALUABLE en P2: cualquier texto nuevo la vuelve medible. La más urgente. |
| MEDUZALOCKER | 3 | 1 | 0,00 | |
| MAZE | 2 | 2 | 0,00 | ⚠ cohesión 0,41 — buscar notas de la MISMA campaña/estilo que las actuales |
| CHIMERA | 2 | 2 | 0,00 | ⚠ cohesión 0,15, la peor de las 30 — la peor apuesta por nota invertida; igual se intenta, pero sin gastar horas |
| WANNACRY | 2 | 2 | 0,01 | |
| RYUK | 2 | 2 | 0,03 | ⚠ cohesión 0,44 y margen negativo |
| CRYPTOLOCKER | 3 | 1 | 0,41 | ojo: el original de 2013, no «Crypt0l0cker» |
| JIGSAW | 2 | 2 | 0,67 | |
| NOTPETYA | 2 | 2 | 0,70 | |

## Prioridad 2 — completar el objetivo de B.1: 16 notas más en 10 familias

Ya andan bien (F1 ≥ 0,75) pero les faltan plantillas para el 4: CUBA, BADRABBIT,
BLACKMATTER, DARKSIDE, NETWALKER, SUNCRYPT (2 c/u) · LORENZ, CONTI, SODINOKIBI,
AVOSLOCKER (1 c/u). Suma con la Prioridad 1: **33 plantillas nuevas en 19 familias.**

## Dónde NO gastar tiempo

BLACKBASTA (F1 0,11), HELLOKITTY (0,29), CLOP (0,32) y DHARMA (0,56) **ya tienen 4+
plantillas**: su problema no es cantidad sino cohesión baja o plantillas compartidas
(BLACKBASTA↔CONTI, DHARMA↔PHOBOS). B.3 midió que agregar notas ahí no cambia la cohesión.
Si aparece algo al paso, se guarda; salir a buscarlas, no.

## Fuentes

Las URLs por familia ya están en `mas_notas_descarga.md` (pcrisk por variante, fuentes
maestras) y `fuentes_notas_descarga.md`. OCR/transcripción **autorizado por el tutor**
(2026-08-16); etiquetar `transcripcion_ocr` en el manifiesto.

## Reglas de recolección (para que lo juntado sirva)

1. **Vale el texto distinto, no la nota.** Una nota que repite un texto ya presente no
   suma (B.1: sobre el eje de plantillas, curvas superpuestas; diferencia media 0,0052 de
   macro-F1). Antes de dar por cumplida una familia, pasar el chequeo de casi-duplicados.
2. **Juntar con margen**: parte de lo recolectado va a colapsar con plantillas existentes
   al deduplicar. Apuntar a ~1,5× el objetivo.
3. **Formato original siempre** (no copiar/pegar a .txt si el bruto existe), nombre
   original, y registrar en el momento: familia · URL de origen · fecha · tipo
   (bruto / transcripcion / transcripcion_ocr).
4. **Carpeta de aterrizaje:** `3_datos/recoleccion_2026-08/<FAMILIA>/` — fuera del repo
   (los datos no se commitean). La incorporación a `corpus_v2` la hace el pipeline con
   manifiesto, no a mano.
5. ⚠️ **Windows Defender**: los `.hta` corren riesgo de cuarentena (ya pasó el
   2026-08-04). La exclusión de `C:\Users\Romina\Tesis\3_datos` sigue pendiente y la
   configura Romina.

## Al volver con notas

Claude corre: deduplicación (cuáles cuentan como plantilla nueva) → actualización del
manifiesto → corrida canónica sobre la base nueva → punto nuevo de la curva B.1. Todo
local, minutos. Recién ahí se decide si few-shot sigue haciendo falta para las familias
donde la recolección no alcanzó.
