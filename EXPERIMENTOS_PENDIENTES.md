# Experimentos pendientes — por frente

_Escrito 2026-08-22. Estado vigente y detalle completo en `ESTADO_TESIS.md`; el diseño de
cada uno en `PLAN_MEJORAS.md`. Este archivo es solo el índice de lo que falta correr._

## Estado de cada frente

| | Método | Corpus | Cifra vigente |
|---|---|---|---|
| **Notas** | congelado (TF-IDF + LinearSVC) | **149 notas · 99 plantillas · 30 familias** (limpieza del 2026-08-25) | **P2bal (cabecera desde 2026-09-22) texto 0,6551 · M.6 0,7417** · P2 0,4593 ± 0,075 · P1 0,789 ± 0,024 |
| _(base anterior, no borrada)_ | ídem | 155 notas · 106 plantillas · 30 familias | P1 0,812 ± 0,031 · P2 0,5265 ± 0,0490 |
| **Archivos** | congelado (bytes 512+512 + RandomForest) | NapierOne · 30 familias | Exp. 2c exactitud 0,912 ± 0,002 · macro-F1 0,911 ± 0,001 |

---

## FRENTE DE NOTAS

| # | Experimento | Estado | Qué mide | Requisito |
|---|---|---|---|---|
| **M.1** | Cascada IOC→texto | ✅ **CERRADO** sobre 155 **y** re-medido sobre 149 (2026-08-25) | Cobertura 0,2523 · acierto-donde-aplica 0,9561 · macro-F1 0,4959, Δ +0,0279 [+0,0124; +0,0434] 10/10 → **se adopta la variante sin filtro de circularidad**; la variante con filtro NO. Salidas en `resultados_cascada_149/` | — |

| **M.2 / D.2** | Nombre + extensión de la nota como vista | 🟡 bloqueado | La señal principal de ID Ransomware, con procedencia controlada | mapeo MISP + auditoría genuino/curador |
| **M.4** | Desambiguación dentro de linaje | 🟡 opcional | Ataca BLACKBASTA↔CONTI y DHARMA↔PHOBOS: mismo texto, IOCs distintos | — |
| **M.5** | Definición de «nota mínima» | 🔴 decisión | RYUK: ¿fragmentos de 6 tokens solo-contacto son notas? | Cappo |
| **Ext.** | Extensión a familias nuevas (ThreatLabz, ~38 candidatas) | 🔴 decisión | Si el método escala. **El macro-F1 VA A BAJAR** (más clases). Diseñada como validación fuera de muestra de la cohesión (ρ +0,69) | Cappo + método congelado |

**Cerrados, NO repetir:** **M.1 cascada IOC→texto (cerrado; re-medido sobre 149 el
2026-08-25)** · **M.3 abstención (cerrado 2026-08-25: contesta 65 % y acierta 90 % a umbral
0,50; la curva completa en `resultados_abstencion_149/`)** · hiperparámetros (840 configs) ·
abstracción de marcadores · control P3 · embeddings multilingües (Exp. 3e) · **recolectar más
allá de 3 plantillas** (B.1 re-medida sobre 149: el último paso significativo es 2→3; pasado el
corte el macro-F1 **baja** de forma medible, cinco pasos con IC 95 % entero bajo cero).
**Cuatro negativos de método convergentes ⇒ el techo lo pone el dato.** Y ahora con número:
bajo P2 el techo estimado es **macro-F1 0,470** [0,411; 0,527] y **0,50 no es alcanzable
agregando notas** (`resumen_cap4_149/b1_extrapolacion.csv`).

---

## FRENTE DE ARCHIVOS CIFRADOS

> **✅ 2026-08-28 — la REDACCIÓN del frente de archivos está hecha.** Nueve subsecciones
> nuevas en `resultados.tex` (§4.3.3, 4.4.4, 4.4.5 y 4.5.4 a 4.5.9), 74 páginas, 0 errores.
> Lo que sigue pendiente del frente de archivos es **medir**, no escribir: Exp. 2d, A.3,
> A.5a y A.5b. Detalle en `ESTADO_TESIS.md`, bloque «ESCRITO EN LA TESIS (2026-08-28)».

| # | Experimento | Estado | Qué mide | Requisito |
|---|---|---|---|---|
| **Exp. 2e** | Rasgos estructurales sobre los bytes (sin nombre ni extensión) | ✅ **MEDIDO** (job 4058, 26-09) | **Mejora del método por contenido**, macro-F1 **0,9114 → 0,9359** (Δ +0,0246, IC [+0,0237; +0,0254], 5/5 semillas). La mejora cae entera en las seis difíciles: **+0,1186** contra **+0,0006** en las otras 24. Solo los 44 rasgos, sin bytes: 0,8680. **No arrastra limitación de campaña.** Pendiente opcional: importancias del bosque para saber QUÉ rasgo hace el trabajo (~20 min) | ninguno: hecho |
| **Exp. 2d** | Bytes × forma del nombre × extensión literal | ✅ **MEDIDO** (job 3937, 11-09; registrado en ESTADO el 17-09) | Elemento de acción 2 del tutor. Cinco columnas, macro-F1: solo bytes **0,9117** · + forma del nombre **0,9998** (Δ +0,088, reportable con limitación de campaña) · + extensión literal 0,9699 (cota superior) · controles sin bytes: solo forma **0,5771**, solo extensión 0,9244. La predicción «solo forma ≈ 0,97-0,99» falló: es complementariedad, no reemplazo | ninguno: hecho |
| **A.3** | Curva de aprendizaje (rendimiento vs archivos/familia) | ✅ **MEDIDA** (job 3772, terminó 29-08; registrada en ESTADO el 17-09), **prioridad baja** | «¿Cuántas muestras hacen falta?» — **pedido textual de Cappo (12-08)**, pero **Romina decidió el 29-08 que no le interesa**: con 1.001 archivos por familia no cambia ninguna decisión. Se deja terminar porque ya está dentro del job. **Techo de esfuerzo: UN párrafo. No se re-corre por ningún motivo.** Si el resultado no aporta, no se escribe | ninguno: ya corre |
| **A.5a** | RYUK «HERMES»: volcado hexadecimal | 🟡 chico | Única discrepancia con ID Ransomware; ¿marcador a distancia variable del final? | — |
| **A.5b** | CONTI: ¿el sufijo `0000000000` roba archivos ajenos? | 🟡 chico | Contar predicciones a CONTI vs sus 50 reales | — |
| **A.5c** | Longitud REAL de las firmas binarias | 🟡 **texto corregido el 29-09; la medición exacta, opcional** (un job corto: subir la ventana y re-correr el detector) | Cuatro familias (CERBER, LOCKBIT, RANSOMEXX, TESLACRYPT) reportan **exactamente 64 bytes**, que es el tamaño de la ventana del detector: sospecha de **medición censurada**. Si la firma real es más larga, hay que corregir la cifra en §4.4.2. **Hecho sin correr:** la tesis dice ahora «de al menos 64 bytes, el máximo que mide el procedimiento» en §4.4.2 y en las dos menciones de CERBER (lo señaló la sesión de agosto, «Carpeta recordada») | subir `n_bytes_ventana` y re-correr |
| **A.6** | **Abstención: que el modelo diga cuándo NO sabe** (pedido de Romina, 29-08) | ⚪ **no se corrió; la limitación quedó escrita** en §4.9 (mundo cerrado, 29-09). Sigue disponible si Romina la quiere | Hoy el clasificador de bytes **asigna familia siempre**, aunque la probabilidad de la clase ganadora sea 0,04. Con un umbral sobre esa probabilidad puede contestar «no sé» y reportarse como cobertura vs acierto-donde-contesta. Se mide con la probabilidad del RandomForest y se reporta la curva cobertura vs acierto-donde-contesta. **Ataca directamente el problema de DARKSIDE**, que hoy actúa de clase de descarte (precisión 0,459 / recall 0,881): con abstención esos archivos caerían en «no sé» en vez de ensuciar a DARKSIDE | ninguno: se calcula sobre el modelo que ya existe |
| **A.7** | **Clase «sano»: reconocer un archivo NO cifrado** (pedido de Romina, 29-08) | ⚪ **no se corrió; declarado como mundo cerrado** en §4.9 (29-09). Trabajo futuro | Hoy el modelo es de mundo cerrado: metas lo que metas, sale una de 30 familias. Agregar una clase de archivos limpios lo vuelve utilizable de verdad. **Lo interesante es científico, no cosmético:** el caso difícil es el **ZIP, JPEG o MP4 sano**, que es casi ruido igual que un archivo cifrado — es justo el problema que señalan los papers de entropía. Un `.txt` sano lo distingue cualquiera; un `.zip` sano es el examen de verdad. Retoma el **Exp. 1** (binario, medido con 6 características sobre 1.600 archivos y nunca rehecho a escala, limitación D6) pero con el método actual de bytes | archivos limpios de NapierOne en el clúster (verificar que estén) |
| **A.8** | **Bytes posicionales + estadísticas de la misma ventana** | ✅ **CERRADO: es el Exp. 2e** (job 4058): 0,912 → 0,936 de macro-F1 | Ataca a SUNCRYPT y NOTPETYA. Las dos tienen **estructura medible al final del archivo** — entropía de cola **4,78** y **6,58** contra un techo de 7,59 — pero el modelo no la ve porque mira bytes en posiciones fijas y ahí el contenido varía archivo por archivo. Una **estadística** de esa ventana (entropía, proporción de ceros, racha más larga, tamaño) es invariante a qué bytes sean. Fallas complementarias: lo estadístico solo da 0,603, lo posicional 0,912, **la unión nunca se evaluó** (el manifiesto del job 3639 lista tres configuraciones y ninguna las combina) | ninguno: las dos representaciones ya están codificadas |
| **A.9** | HistGradientBoosting sobre la representación posicional | ✅ **CERRADO sin correr (29-09): sin margen.** El sistema completo da 0,9998 y la ablación del Exp. 2h muestra que con la extensión cualquiera de las dos capas de contenido alcanza | **Nunca se probó.** Sobre las características estadísticas le ganó al RandomForest (0,603 contra 0,601) con la vigésima parte del cómputo (160 s contra 666 s). Sobre 1.024 columnas la comparación está abierta | 1 job |
| **A.10** | Clasificador en dos etapas para el grupo difícil | ✅ **CERRADO sin correr (29-09): sin margen.** Las seis difíciles quedan por encima de 0,99 con el sistema completo (Exp. 2g) | Las seis se confunden **entre ellas el 97,2 % a 99,4 %** de las veces, y la séptima peor (BADRABBIT, 0,978) está a 0,26 de distancia: la etapa 1 «¿pertenece al grupo?» es casi trivial. La etapa 2 usa características propias del grupo | después de A.8 |

> ### 🎯 DÓNDE ESTÁ EL MARGEN DEL FRENTE DE ARCHIVOS (medido el 2026-08-29)
> Sobre las 10 semillas del job 3648:
>
> | | macro-F1 |
> |---|---|
> | Las **24 familias buenas** | **0,9966** de promedio |
> | Las **6 difíciles** | **0,569** de promedio |
> | **Total** | **0,9111** |
>
> **Las 24 ya están prácticamente perfectas: todo el déficit son seis familias.** Si esas seis
> llegaran a 0,98, el macro-F1 pasaría de **0,911 a 0,993 (+0,082)**; a 0,90 daría 0,977; a
> 0,80 daría 0,957. **No hay margen en ningún otro lado**, y por eso A.8, A.9 y A.10 apuntan
> todas al mismo grupo de seis.
>
> Y las seis fallan por **dos motivos distintos**, que piden palancas distintas:
> - **JIGSAW, DARKSIDE, CRYPTOLOCKER, WASTEDLOCKER** solo marcan el **nombre**, que el Exp. 2c
>   no mira ⇒ los ataca la **columna (2) del Exp. 2d**, corriendo ahora en el job 3772.
> - **SUNCRYPT y NOTPETYA** sí dejan estructura, pero **variable archivo por archivo** ⇒ los
>   ataca **A.8**.

> **➕ AGREGADO EL 2026-08-29 (ideas de Romina).** A.6 y A.7 salen de una misma pregunta suya:
> *que el modelo diga cuándo no sabe, y cuándo el archivo está sano.* Las dos son baratas y
> las dos apuntan al mismo hueco: **el clasificador de archivos es de mundo cerrado y no tiene
> forma de decir «esto no es ninguna de las 30» ni «esto no está cifrado».** A.5c sale del
> hallazgo de CERBER del mismo día (ver `ESTADO_TESIS.md`).
>
> ⚠️ **Estas tres NO se empezaron.** Quedan anotadas, como pidió Romina.

**Cerrados:** A.0 BLACKBASTA · A.1 ablación de ventana + bloque del medio · A.2 desvío
(job 3648) · A.4 desvío del 2b (10 semillas) · A.8 (= Exp. 2e) · A.9 y A.10 (sin margen tras el
Exp. 2g/2h) · Exps. 2f, 2g y 2h (28-29/09). **El frente de archivos está cerrado de medición**
(29-09); solo quedan opcionales A.5c (largo exacto de las firmas), A.6 y A.7. **No hay extensión posible a más familias:** no
existen archivos cifrados públicos fuera de NapierOne (límite externo, citable).

---

## Transversales (no son de un frente)

| # | Qué | Estado |
|---|---|---|
| Mapeo manual MISP familia→entrada | 🟢 ~30 min de Romina. **Destraba tres cosas**: M.2, la extensión, y la tabla citable de extensiones por familia | pendiente |
| Año de detección vs F1 por familia | 🟢 análisis corto (`Pruebas.xlsx` + MISP corrobora). Pedido textual de Cappo | pendiente |
| B.2 auditar 37 notas «NapierOne/varios» | 🟡 | pendiente |
| Majority voting notas + archivos | 🔴 decisión Cappo. **No hay muestras pareadas** ⇒ se puede proponer, no evaluar | pendiente |

---

## Orden recomendado

1. ~~Esperar **M.1**~~ ✅ cerrado el 2026-08-25, junto con la curva B.1 y el resumen del
   capítulo 4 sobre 149. **La re-medición del frente de notas está completa.**
2. Lanzar **Exp. 2d + A.3 juntos** — comparten la carga de datos, un solo job. **Es lo que
   queda con más margen.**
3. Mientras corren: **mapeo MISP**, **año vs F1**, **A.5a/A.5b**, barrido de boilerplate
   sobre las 149. (M.3 ya está cerrado.)
4. Después: **auditoría de nombres → M.2**.
5. **Escritura** (Sprint D, bloque K de `PENDIENTE_REDACCION.md`).
6. Con Cappo: extensión de familias, majority voting, «nota mínima».
7. **Bloque E al final de todo:** conclusión, resumen, front matter, agradecimiento NIDTEC.

## Reglas que aplican a cualquiera de estos

- Predicción **preregistrada** en `ESTADO_TESIS.md` antes de correr, con criterio de adopción
  (IC 95 % del Δ pareado excluye 0) y la lectura de cada resultado posible.
- Carpeta de salida **nueva** siempre; nunca pisar `resultados_canonicos/`.
- Toda cifra con **métrica y base** pegadas.
- Código se commitea y pushea en el momento; los `.md` solo cuando Romina lo pide.

---

## Fuentes de notas NUEVAS por explorar (halladas 2026-08-22, SIN verificar)

Independientes de las tres ya usadas (ThreatLabz · kipziptie · Lemmou), así que sus notas
tienen más chance de ser **plantillas nuevas** y no casi-copias de lo que ya hay:

1. **Kaggle — «Ransomware Note Dataset Collection»**
   `kaggle.com/datasets/abiprasanth/ransomware-note-dataset-collection`
   Colección curada de textos de notas por familia. ⚠️ Sin verificar: cuántas familias, cuántas
   notas, y **de dónde las sacó el curador** (si viene de ThreatLabz sería duplicado). Revisar
   procedencia antes de incorporar nada.

2. **Group-IB — «Notes From the Most Active Ransomware Groups»**
   `group-ib.com/resources/ransomware-notes/`
   Empresa de inteligencia; notas de los grupos más activos de 2024. Fuente reputable y
   citable, y con notas RECIENTES (lo que más falta). ⚠️ Sin verificar cobertura.

**Procedimiento si se exploran:** pasar todo por el verificador de casi-duplicados antes de
contarlas (una nota que colapsa con una plantilla existente NO suma — B.1), registrar
procedencia en el manifiesto, y recordar el techo medido: **3 plantillas por familia**
(re-medido sobre 149 el 2026-08-25; eran 4 sobre 144 notas). Más allá de eso no solo no mueve el
macro-F1: lo **baja** de forma medible.

**Ojo — NO son datasets de notas** (aparecen en las búsquedas pero son de ejecutables o
comportamiento, no sirven para este frente): MLRan, RanSMAP, MOTIF, MarauderMap, MIRAD.
