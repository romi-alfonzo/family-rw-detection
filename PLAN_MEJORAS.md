# Plan de mejoras — frentes de notas y archivos por separado

_Fijado el 2026-08-05, después de completar el capítulo 4. Los dos frentes se mantienen
independientes: no hay clasificación combinada._

## Estado de la búsqueda de hiperparámetros (para no volver a dudar)

| Experimento | ¿Optimizado? | Referencia |
|---|---|---|
| Exp. 3 — notas | ✅ hecho (job 3548) | §4.7.5. No mejoró: el límite es el corpus |
| Exp. 2c — bytes de archivos | ✅ hecho (búsqueda anidada interna) | §4.5.1 |
| Exp. 2 — características estadísticas | ❌ **pendiente** | Declarado como limitación en §4.3.1 |

---

## SPRINT 1 — Sin cluster, se hace ya

### 1.1 Abstracción de marcadores variables en las notas *(Claude, minutos)*
**Hipótesis:** el modelo memoriza valores concretos (`juan123@protonmail.com`) que cambian
entre campañas. Reemplazándolos por su tipo (`[EMAIL]`, `[ONION]`, `[BTC]`, `[URL]`, `[ID]`)
el modelo aprendería el *patrón*, que sí es estable.

**Evidencia previa medida (2026-08-05):** el perfil de marcadores es característico por
familia — DHARMA solo usa emails (17, ningún onion); CERBER solo onion (18, ningún email);
RYUK email+BTC; PHOBOS solo email; CONTI solo onion.

**Fundamento bibliográfico:** Lemmou et al. sustituyen `[ext]`, `[id]`, `[random]` en su
pre-análisis; Trujillo (UPC) usa variables `##URL##` y `##EMAIL##`.

**Métrica de éxito:** que P2 suba desde 0,435. Si no sube, es igualmente reportable: indica
que la variabilidad entre plantillas es estructural y no solo de datos de contacto.

### 1.2 Restaurar las 2 notas en cuarentena *(Romina, manual)*
`DHARMA/Info__13.hta` e `Info__3.hta`. Seguridad de Windows → Historial de protección →
Restaurar, y agregar exclusión para `C:\Users\Romina\Tesis\3_datos`. Necesario para que
todos los resultados queden sobre la misma base (146 notas, no 144).

---

## SPRINT 2 — Una sola tanda de cluster (~1 hora)

### 2.1 Hiperparámetros de las características estadísticas *(cierra el hueco)*
Búsqueda anidada sobre el subconjunto de 19 características (el mejor, 0,603) y sobre las
275. Barato: 28 s por configuración. Cierra la única limitación de optimización declarada.
**Expectativa honesta:** puede subir a 0,65-0,70; no cambia las conclusiones porque el
Exp. 2c ya alcanza 0,910 sobre los mismos datos, pero elimina la objeción.

### 2.2 Análisis de robustez del clasificador de bytes *(un solo script, 4 análisis)*

| # | Análisis | Qué responde | Valor |
|---|---|---|---|
| a | **Generalización a tipos de archivo no vistos**: entrenar con doc/xls/ppt, evaluar sobre pdf/jpg | ¿La marca es del ransomware o del documento original? | **Crítico** — puede invalidar o confirmar el 0,910 |
| b | Importancia por posición de byte (offset 0 a 1023) | ¿En qué desplazamientos vive la información? | Figura excelente; conecta 2c con 2b |
| c | Ablación de ventana: 64/128/256/512, solo cabecera vs solo cola | ¿Cuántos bytes hay que leer? | Valor práctico para implementar |
| d | Diagnóstico de las 6 difíciles: confusión mutua + entropía de sus cabeceras | ¿Por qué fallan? | Convierte constatación en explicación |

**Por qué (a) es crítico:** cada familia cifró el mismo conjunto base de documentos
(`0001-doc`, `0001-pdf`, `0001-jpg`...). Si alguna hace cifrado parcial y conserva la
cabecera original, el modelo podría estar aprendiendo del documento y no del ransomware.
Es la única prueba de confundido que se puede hacer con NapierOne.

---

## SPRINT 3 — En paralelo, trabajo manual de Romina

### 3.1 Ampliar el corpus en plantillas
URLs ya identificadas en `6_notas_trabajo/mas_notas_descarga.md` para 14 familias.
**Objetivo correcto: ≥3 plantillas distintas por familia**, no "≥5 notas" — la métrica
sale del hallazgo de que 146 notas son solo 95 contenidos.
Es lo único que puede mover P2 de forma sustancial, porque ataca la causa.

### 3.2 Auditar las 37 notas de procedencia "NapierOne/varios"
Pista: el repositorio `kipziptie` que apareció en `Pruebas.xlsx`. Necesario para que la
sección de procedencia del capítulo 3 sea verificable.

---

## SPRINT 4 — Después del Sprint 3

### 4.1 Nombre de archivo como característica, solo los genuinos
Lemmou obtiene F = 0,920 clasificando únicamente por el nombre. Nosotras lo descartamos
porque el 51 % de los nombres los puso el curador. Los 42 del repositorio de Lemmou sí son
auténticos. Requiere marcar la procedencia de cada nombre en el manifiesto.

### 4.2 Re-correr todo sobre el corpus ampliado
Corrida canónica + gridsearch sobre la base final, para que no convivan cifras de 144, 146
y del corpus ampliado.

---

## SPRINT 5 — Cierre

- Actualizar el capítulo 4 con los resultados de los sprints 2 y 4 (solo agregar).
- Reunión con Cappo: llevar el informe de avance y las 4 consultas ya redactadas.
- **Bloque E (al final de todo):** conclusión completa, resumen/abstract, front matter,
  agradecimiento obligatorio al cluster del NIDTEC, y limpieza del `.bib`.

---

## Lo que NO está en el plan, y por qué

- **Clasificación combinada notas + archivos:** descartada por decisión de Romina; los dos
  frentes se mantienen separados.
- **Aprendizaje few-shot** para las familias con 1-2 plantillas (paper que sugirió Cappo el
  06/06/2024): queda como *trabajo futuro* en la tesis, salvo que él pida incorporarlo.
- **Perseguir más exactitud en archivos:** con 0,910 y seis familias probablemente sin señal,
  el rendimiento marginal es bajo. El Sprint 2.2 busca entender y blindar, no subir el número.
- **Resolver la limitación de campaña:** no es posible con NapierOne (una campaña por
  familia). Se declara como limitación en §4.5.4.
