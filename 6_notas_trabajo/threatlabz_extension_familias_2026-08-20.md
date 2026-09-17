# ThreatLabz — qué familias sirven para extender el frente de notas (2026-08-20)

Nota de trabajo para **continuar la búsqueda de familias**. Todo lo de abajo está medido,
no de memoria. Solo se revisó/midió: **no** se tocó el corpus ni el manifiesto.

## De qué repo hablamos y dónde está

- Repo **ThreatLabz** (`github.com/ThreatLabz/ransomware_notes`) — el que a veces llamamos
  "theartbiz" de oído.
- Clon local: `C:\Users\Romina\Tesis\3_datos\fuentes_notas\ransomware_notes`
  (está dentro de `3_datos/`, o sea **datos, no se versiona** en la tesis).
- Tras el `git pull` del 2026-08-20: **HEAD `8765d0b` (2026-08-13, "added anubis")**;
  **222 carpetas de familia**, 221 con al menos una nota usable, **331 notas usables**.

## Criterio usado (el mismo del frente de notas)

Colapso de casi-duplicados: **coseno ≥ 0,90**, TF-IDF **char_wb 3-5**, componentes conexas
(`agrupar_neardups` de `clasificador_notas_v2.py`, umbral 0,90). Se cuenta **textos distintos**
por familia = grupos distintos entre sus notas, **no** archivos.

Umbral del clasificador (protocolo **P2ret**):
- **≥ 2 textos distintos** = mínimo para ser *evaluable* (si hay 1 solo, al retener su única
  plantilla para test no queda nada para entrenar → **F1 = 0 por construcción**).
- **≥ 4 textos distintos** = objetivo de buena cobertura (B.1).

## Resultado (solo ThreatLabz)

| Textos distintos | Familias | ¿Cubrible? |
|---|---:|---|
| ≥ 4 | **8** | sí, objetivo cumplido |
| 2–3 | **40** | sí, pero bajo objetivo |
| 1 | **173** | **no** (una sola nota) |

8 + 40 + 173 = 221. Es decir: **48 familias con datos suficientes (≥2)**, de las cuales
**8 llegan a 4**. Las 173 restantes no alcanzan con este repo solo.

## Familias NUEVAS candidatas para la extensión (no están en nuestras 30)

### Con ≥ 4 textos distintos — listas tal cual (4)
`inc`, `doppelpaymer`, `interlock`, `nightspire`

### Con 2–3 textos distintos (34) — número = textos distintos
`cactus`·3, `braincipher`·3, `nemty`·3, `noescape`·3, `play`·3, `tengu`·3, `blackbyte`·2,
`ransomhub`·2, `risen`·2, `8base`·2, `blacklock`·2, `chilelocker`·2, `cloak`·2, `hunters`·2,
`qilin`·2, `thegentlemen`·2, `abysslocker`·2, `bitpaymer`·2, `clearwater`·2, `darkangels`·2,
`diavol`·2, `dragonforce`·2, `eldorado`·2, `karakurt`·2, `knight`·2, `krypt`·2, `lynx`·2,
`moneymessage`·2, `nokoyawa`·2, `noname`·2, `ragnarlocker`·2, `ragnarok`·2, `raworld`·2,
`weaxor`·2

### Familias del repo que YA son nuestras (ignorar el conteo de arriba para estas)
- ≥4 en el repo: `blackbasta`, `ransomexx`, `alphv`(=BLACKCAT), `clop`.
- 2–3 en el repo: `cerber`, `conti`, `revil`(=SODINOKIBI), `gandcrab`, `lockbit`, `lorenz`.

## Advertencias antes de usar esto

1. **Es ThreatLabz SOLO.** Para nuestras 30 familias del núcleo el corpus real combina
   ThreatLabz + Lemmou + pcrisk + NapierOne, así que el conteo real es mayor (p. ej. acá
   `lockbit`=2 y `gandcrab`=2, pero en el corpus son 6 y 5). **Para las 30 vale la tabla del
   frente de notas, no esta.** Esta medición sirve para la **extensión**.
2. **Extensión = decisión del tutor 2026-08-20:** el frente de notas PUEDE sumar familias como
   experimento que se AGREGA, con su base declarada. No corrige nada de lo escrito sobre 30.
3. **Cada familia nueva necesita los chequeos de integridad del proyecto** antes de entrar:
   procedencia citable y sobre todo la **trampa campaña-vs-familia** (varias de estas son grupos
   recientes con renombres/sucesores). No es "agregar y listo".
4. **En archivos cifrados NO hay extensión posible** (no existen archivos públicos fuera de
   NapierOne). Esto es solo para el frente de **notas**.

## Cómo reproducir el conteo (si hace falta rehacerlo)

Método: importar `agrupar_neardups`, `TFIDF_CHAR`, `UMBRAL_NEARDUP`, `MIN_CHARS_NOTA` de
`clasificador_notas_v2.py` y `extraer_texto` de `extractor_notas.py`; recorrer las carpetas de
familia del clon, extraer texto de cada nota, colapsar globalmente y contar group-ids distintos
por familia. sklearn local disponible (1.6.1).

## Contexto extra de esta sesión (para que no viva solo en el chat)

- **Adiciones recientes de ThreatLabz (mayo–ago 2026):** todas de familias fuera de nuestras 30
  (`anubis`, `gentlemen`, `valencia`, `booba`, `nightspire`, `mortar`, `chaos`, `everest`,
  `lamashtu`, `aurora`, `m3rx`, `vect`). **A nuestras 30 no se les agregó ninguna nota nueva.**
- **LOCKBIT — hueco local (no del repo):** antes del pull, al clon le faltaban 3 archivos de
  `lockbit/` que el repo YA tenía en nuestro propio commit (probable cuarentena de Defender antes
  de excluir `3_datos`). Tras el pull, verificar que `lockbit/` tenga los 5 archivos
  (`ReadMeForDecrypt.txt`, `[id].README.txt`, `[rand].README.txt`, `lockbit2.txt`, `lockbit3.txt`);
  si falta alguno en disco: `git checkout -- lockbit` dentro del clon.
