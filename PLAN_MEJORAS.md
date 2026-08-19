# Plan de mejoras — frentes de notas y archivos por separado

_Fijado el 2026-08-05, después de completar el capítulo 4._
_**Reestructurado el 2026-08-15** tras la reunión con el tutor del 12-08-2026
(`6_notas_trabajo/reunion_2026-08-12_revision_resultados.md`), que reordenó las prioridades._

Los dos frentes se mantienen independientes: no hay clasificación combinada.
**Salvo que Romina decida lo contrario** — el tutor pidió evaluar *majority voting*; ver la
sección «Decisiones abiertas» al final.

## Estado de la búsqueda de hiperparámetros (para no volver a dudar)

| Experimento | ¿Optimizado? | Resultado |
|---|---|---|
| Exp. 3 — notas | ✅ job 3548 | No mejoró (0,775/0,424 vs 0,783/0,427). El límite es el corpus |
| Exp. 2c — bytes de archivos | ✅ búsqueda anidada interna | §4.5.1 |
| Exp. 2 — características estadísticas | ✅ **cerrado** | 0,599–0,603; diferencia −0,000 |

Los tres huecos de optimización están cerrados. **Son tres resultados negativos convergentes:
ningún ajuste de método mueve la aguja en ninguno de los dos frentes.** Ese es, en sí mismo, el
argumento de que el techo lo pone el dato.

---

## Qué está cerrado

- **Sprint 1.1 — abstracción de marcadores en notas.** Ejecutado 2026-08-05. P2 −0,010,
  P1 −0,005: sin efecto. Conclusión: la variabilidad entre plantillas de una misma familia es
  estructural, no se reduce a datos de contacto.
- **Sprint 1.2 — notas en cuarentena.** Pendiente manual de Romina (`DHARMA/Info__13.hta` e
  `Info__3.hta`) + exclusión de `C:\Users\Romina\Tesis\3_datos` en Windows Defender. Mientras
  no se haga, conviven cifras sobre 144 y sobre 146 notas.
- **Sprint 2 completo.** Volvieron los dos trabajos: gridsearch de estadísticas (arriba) y el
  análisis de robustez del clasificador de bytes — (a) generalización a tipos de archivo no
  vistos **0,879 frente a 0,910, caída de solo 0,031 ⇒ el resultado principal queda
  confirmado**; (b) importancia por posición; (c) ablación de ventana; (d) diagnóstico de las
  seis difíciles.
  > ⚠️ **Matiz hallado el 2026-08-16 (ver `ESTADO_TESIS.md`):** los pliegues de (a) cubren
  > entre **25 y 28** de las 29 familias, no las 29 — CERBER y BLACKMATTER renombran el archivo
  > y quedan sin tipo de documento. La conclusión no cambia; la redacción sí.

---

## SPRINT A — Cluster, una sola tanda (frente de archivos)

Todo lo que el tutor pidió sobre archivos cifrados. Se corre junto porque comparte la carga
de datos.

> ⚠️ **Orden obligatorio: primero BLACKBASTA, después el resto.** Sumar la familia que falta
> cambia el frente de 29 a 30 familias y, con ello, todas las cifras. Si se lanza la ablación
> antes, hay que repetirla. Ver A.0.

### A.0 Subir BLACKBASTA al cluster — ✅ **CERRADO** (2026-08-15/16)
BLACKBASTA ya está en el cluster y los dos experimentos se rehicieron sobre 30 familias:
Exp. 2b (job 3632) y Exp. 2c (job 3633, **0,910 exactitud / 0,910 macro-F1**, sin cambio
respecto de las 29). **Los dos frentes están ahora sobre 30 familias** y desaparece la
asimetría que había que explicar en cada tabla. Falta `Z-Safe` (los benignos), que no hace
falta para la multiclase.

### A.1 Ablación de ventana extendida + bloque del medio — ✅ **CERRADO** (job 3630, 2026-08-16)
`2_codigo/ablacion_ventana_extendida.py` · `2_codigo/slurm/job_ablacion_extendida.sh`.
Resultados completos en `ESTADO_TESIS.md`. Los dos pedidos quedan contestados:
- **La curva satura en 512+512** (0,904 / 0,909 limpio) y baja levemente hasta 4096+4096
  (0,901 / 0,905). Ya hay saturación que mostrar en la figura.
- **El control sin relleno acompaña la curva** ~0,005 por encima ⇒ la mejora es real.
- **El bloque del medio da 0,056** (azar 0,033) y no aporta nada sumado a los extremos.
  La **cola** (0,756) vale más del doble que la **cabecera** (0,338), lo que converge con los
  11 sufijos contra 4 prefijos del Exp. 2b.

✅ CSV bajados y auditados el 2026-08-17 (43/43 cifras confirmadas; ver «AUDITORÍA DE LA
DESCARGA» en `ESTADO_TESIS.md`). `0_tamanos.csv` confirma el control: el subconjunto sin
relleno es constante en los 7 puntos (n=14.783 = archivos ≥8.192 B). Falta solo bajar los
logs `slurm-*.out` del cluster.

### A.2 Desvío en el frente de archivos — **listo para lanzar** (2026-08-17)
Hoy las notas se reportan como media ± desvío sobre 10 semillas y los archivos van sin error.
El tutor pidió incluirlo («relacionado a overfitting y underfitting»). Repetir la corrida
canónica con varias semillas.

**Bloqueo resuelto:** `clasificador_bytes.py` sobrescribía su carpeta de salida **y** tenía las
semillas clavadas (42 en la búsqueda, 7 en la etapa final, todos los `random_state` en 42), así
que diez semillas habrían dado diez veces el mismo número. Arreglado y probado local contra un
dataset sintético; detalle en `ESTADO_TESIS.md`, bloque de los dos defectos.

- Se lanza con `2_codigo/slurm/job_bytes_multisemilla.sh` (`--multisemilla 0,1,…,9`).
- **Las mismas diez semillas (0-9) que `clasificador_notas_v2.py`**, para que los dos frentes se
  reporten sobre la misma base y sea decible en la tesis.
- Hiperparámetros **fijos** (los de la búsqueda anidada del Exp. 2c): se mide dispersión de la
  estimación, no una nueva selección de modelo. Declararlo así.
- Salidas: `bytes_multisemilla.csv` (por semilla), `bytes_multisemilla_resumen.csv`
  (media/desvío/mín/máx) y **`bytes_multisemilla_por_familia.csv` — F1 por familia y por semilla,
  que es lo que permite reportar las seis difíciles con desvío.**
- Estimado 12-15 min por semilla ⇒ 2-2,5 h. Los CSV se reescriben en cada semilla: un corte por
  tiempo no pierde lo corrido.

### A.3 Curva de aprendizaje en archivos
Rendimiento contra cantidad de archivos por familia. Contesta «¿cuántas muestras se necesitan
para una clasificación relativamente confiable?». Barato: hay 500 por familia.

### A.4 Desvío del Exp. 2b — ✅ **CERRADO** (2026-08-17, 10 semillas)

**Cifras para el capítulo** (criterio declarado umbral 0,90, media ± desvío sobre semillas 1-10):
**combinado 0,932 ± 0,001 · solo firmas 0,563 ± 0,002 · 28/30 familias con marca en las diez.**
Contra el criterio viejo de unanimidad: combinado 0,900 ± 0,016 y entre 26 y 28 familias según el
sorteo. **El cambio de criterio reduce el desvío 14× y sube la media 0,032.** Detalle completo,
incluida la validación de la predicción hipergeométrica (1 de 10 observado contra 1,2 esperado),
en `ESTADO_TESIS.md`.

Pendiente menor: bajar los `log_estr_s*.txt` del cluster para promediar también el modo «solo
extensión» y las coberturas. **Hacerlo antes de que se limpie el `/scratch`.**

<details>
<summary>Planteo original de A.4 (registro)</summary>
El 2b se reporta hoy con cifras puntuales (0,867 / 0,533 / 0,933) y **tiene dispersión del mismo
orden que el 2c**: con el mismo criterio (umbral 0,90), la semilla 1 da combinado **0,899** y la
semilla 42 **0,933** — 0,034 de diferencia. La causa está completamente trazada: BADRABBIT pierde
su sufijo de 18 B con otra muestra y, como no cambia la extensión, se queda sin ninguna vía de
identificación (−50 archivos de 1.500). Detalle y aritmética en `ESTADO_TESIS.md`.

- **Correr el detector sobre 10 semillas y reportar media ± desvío** de: familias con marca,
  cobertura y exactitud de los tres modos. Cuesta ~3-6 min por semilla con el criterio nuevo. Ya es
  seguro hacerlo: la carpeta de salida lleva la semilla.
- **NO escribir el capítulo del 2b con cifras puntuales antes de esto.**
- ✅ **HECHO 2026-08-17 — el criterio de firma binaria ya no es unanimidad byte a byte.** El umbral
  de mayoría se aplica ahora también al prefijo y al sufijo, y cada marca se reporta con su
  cobertura. Verificado que con umbral 1,0 el algoritmo da idéntico al anterior (6.000
  comparaciones, 0 diferencias) ⇒ el job 3638 sigue reproducible. Detalle en `ESTADO_TESIS.md`.
- **Orden de las corridas:** primero **una** semilla (la 42) para ver las cifras nuevas del 2b y
  auditar que el umbral en las firmas no esté contando relleno como marca (precedente: CONTI con
  `0000000000`); después las 10.
- Sigue pendiente, y ahora es visible en el CSV gracias a las coberturas: `MIN_MARCA = 4` es un
  borde duro (BLACKBASTA lo cruza según la muestra, `00020000`) y el detector puede contar relleno
  como firma. Decidir si se exige contenido no nulo **después** de ver la corrida con mayoría.
</details>

### A.5 Dos cabos sueltos del detector, para decidir DESPUÉS de escribir el 2b
Ninguno invalida nada; los dos harían el resultado más fuerte y los dos obligan a re-correr, así
que **no tocarlos hasta que el 2b esté escrito con las cifras de A.4**.

1. **La ventana de 64 bytes trunca las firmas.** CUBA da prefijo de **64 B al 92 %** cuando antes
   daba 20 B, y 64 es el máximo que el detector mira; están contra el mismo techo TESLACRYPT,
   CERBER, LOCKBIT y RANSOMEXX. La firma de CUBA tiene **al menos** 64 bytes. Subir `N_BYTES`
   permitiría decir cuánto miden en realidad.
2. **El detector cuenta relleno como firma:** CONTI es identificada por `0000000000`, cinco ceros.
   De las 16 firmas, 15 tienen contenido no trivial. Exigir un mínimo de bytes no nulos lo
   corregiría. Mientras no se haga, **declararlo en el capítulo**.

---

## SPRINT B — El que decide el frente de notas

**Este es el sprint que destraba todo lo demás. Corre en paralelo al A; son frentes distintos.**

### B.1 Curva de aprendizaje de notas — ✅ **CERRADO** (2026-08-19, local, 100 repeticiones)
`2_codigo/curva_aprendizaje_notas.py` · agregados `resumen_para_capitulo4.py --solo b1`.
Contesta con un número el pedido textual del tutor (*«ver el porcentaje de error para saber la
cantidad de notas a necesitar»*) y absorbe lo que era el Sprint 3.1(a).

**Respuesta: 4 textos distintos por familia ⇒ 33 plantillas nuevas en 19 familias.** El paso
3→4 aporta **+0,0297 de macro-F1** (IC 95 % [+0,0039; +0,0554]) y el 4→5 **+0,0006**
(IC 95 % [−0,0207; +0,0219], no significativo), medido sobre las 5 familias con ≥ 5 plantillas.
Todas las cifras, las tres curvas (30 / 11 / 5 familias, azar macro-F1 0,033 / 0,091 / 0,200),
las diferencias pareadas y **los cinco límites declarados** están en `ESTADO_TESIS.md`.

- **Control de corrección:** el punto k=«todo el corpus» reproduce el evaluador canónico con
  diferencia **0,00e+00** en P1 y P2; el script aborta si no coincide.
- **Base 144 notas.** Las 2 notas en cuarentena cuestan 0,0096 de macro-F1 en P1 y 0,0143 en
  P2, por debajo del desvío entre semillas ⇒ **el capítulo 4 no se toca** y la cifra oficial
  sigue siendo la de 146 notas (P1 0,760 ± 0,029 · P2 0,435 ± 0,057 de macro-F1).
- **No confundir protocolos:** P2ret (retención de una plantilla, hasta n−1 en entrenamiento)
  da macro-F1 0,6164 ± 0,0450 sobre 30 familias y **no reemplaza** al P2 canónico de 2
  pliegues, que da 0,4210 ± 0,0508 sobre las mismas 144 notas.
- **Falta:** agregar la curva bajo **P3** cuando B.3 entregue el grafo (el script ya está
  cableado para recibir otro criterio de agrupamiento).

### B.1.bis Volver a correr la curva después de cada lote de recolección
La curva es el medidor de si la recolección rinde, y ahora se regenera con un comando. No es
un experimento nuevo: es el instrumento de control del Sprint C.

### B.2 Auditar las 37 notas de procedencia «NapierOne/varios»
Pista confirmada: el repositorio `kipziptie` que aparece en `Pruebas.xlsx`, junto con
threatlabz y el `RansomNoteFiles` de Lemmou. Necesario para que la sección de procedencia del
capítulo 3 sea verificable.

### B.3 Grafo de marcadores compartidos entre plantillas — ✅ **CORRIDO** (2026-08-19, local)
`2_codigo/grafo_marcadores.py` · salidas en `4_resultados/resultados_grafo_marcadores/`.
Todas las cifras y los límites, en `ESTADO_TESIS.md`, sección «B.3». Tres resultados:

1. **Lo que predice el F1 de una familia es cuánto se parecen sus plantillas entre sí
   (Spearman ρ +0,704, p = 2·10⁻⁵), NO cuántas plantillas tiene (ρ −0,108, p = 0,58).**
   No contradice a B.1 —B.1 quitó plantillas y midió la caída, es causal— pero reordena la
   recolección: sumar un texto no convierte a una familia de cohesión baja en una de cohesión
   alta. **CHIMERA es la peor apuesta de la lista** (cohesión 0,1538, la más baja de las 30,
   y un solo marcador). Contraejemplo a declarar: SUNCRYPT, cohesión 0,4529 y cero
   marcadores, F1 por familia 1,000.
2. **El macro-F1 de P2 NO viene de reconocer IOCs repetidos.** Con el control de azar:
   P2 0,4210 · azar con el mismo perfil de tamaños 0,2290 ± 0,0209 · P3 real 0,2293. La caída
   de 0,1917 es **0,1920 de agrupamiento grueso y −0,0003 de continuidad de IOCs.** Es la
   respuesta, con número y control, a la objeción «tu 0,435 es búsqueda de IOCs disfrazada».
   **P3 no reemplaza a P2:** se reporta P2, y P3 entra como el control.
3. **Sin filtrar los valores de infraestructura común el grafo colapsa** (una componente de
   39 nodos de 97, 13 familias inevaluables). Corre en cuatro variantes y reporta las cuatro.

**Pendiente de B.3:** (a) auditar a mano `b3_valores_excluidos.csv` antes de citar que el
criterio de circularidad elimina 21 de 146 aristas dentro de familia (15-18 %) — el criterio
es subcadena y «conti» está dentro de «continue»; (b) revisar las 7 aristas entre familias que
sobreviven al filtro, para ver si reencuentran BLACKBASTA/CONTI y DHARMA/PHOBOS; (c) el
control de azar solo está corrido sobre la variante más estricta.

<details>
<summary>Planteo original de B.3 (registro)</summary>
Corre JUNTO con B.1: mismos datos, sin cluster, y su salida es insumo de B.1. Contesta el
elemento de acción 1 del tutor («hallar algún patrón entre plantillas para detectar una no
conocida») y es **diagnóstico** de C.bis: si las plantillas de una misma familia casi no
comparten valores, la vista de marcadores no puede funcionar bajo P2 y se sabe antes de
escribirla.

- **Grafo:** nodos = las 95 plantillas; aristas = valores exactos de marcador compartidos
  (emails, onions, BTC, URLs, IDs).
- **Protocolo P3 (nuevo, sale gratis del grafo):** definir el grupo del
  `StratifiedGroupKFold` como la **componente conexa** del grafo, en vez de la plantilla.
  El modelo nunca ve un IOC del test ⇒ el número mide «generaliza a un clúster de campaña
  nunca visto». Más estricto que P2; se reporta junto a P2 y la diferencia dice cuánto
  aportaba la continuidad de IOCs.
- **Criterio de circularidad:** «¿la feature sobreviviría si la familia se cambiara el nombre
  mañana?». Excluir todo token que contenga el nombre de la familia como subcadena (sin
  mayúsculas, en cualquier posición: parte local, dominio, onion, ruta, extensión mencionada)
  + alias conocidos (BLACKCAT/ALPHV, SODINOKIBI/REvil, MEDUZALOCKER/MedusaLocker).
  `lockbitsupp@…` afuera; `abc123@protonmail.com` adentro. **Correr con y sin la exclusión y
  reportar ambos** — habilita una afirmación cuantitativa sobre el 181/182 de Lemmou, cuya
  lista de keywords incluye el nombre de familia.

> ⚠️ **Corrección al diseño, medida el 2026-08-19:** la frase «la diferencia P2 − P3 dice
> cuánto aportaba la continuidad de IOCs» **es falsa tal como está escrita**. Bajo P3 los
> grupos bajan de 95 a 65 y eso solo ya empeora el resultado, por dos vías que no tienen nada
> que ver con IOCs: 7 familias quedan enteras en un pliegue (F1 = 0 por construcción) y cada
> pliegue entrena con menos unidades independientes. **Hace falta el control de azar** —
> agrupamientos aleatorios con el mismo perfil de tamaños— y ese control mostró que el aporte
> de la continuidad de IOCs es **−0,0003 de macro-F1**, o sea nulo. El diseño original habría
> llevado a afirmar que casi la mitad del 0,4210 era búsqueda de IOCs. Ya está implementado en
> `grafo_marcadores.py` (`--repeticiones-azar`).

</details>

---

## SPRINT C — ✅ **BIFURCACIÓN RESUELTA POR B.1 (2026-08-19): SE RECOLECTA**

> **La curva NO está plana.** Sobre las 5 familias con ≥ 5 plantillas, la diferencia pareada de
> macro-F1 es significativa hasta el paso 3→4 (**+0,0297**, IC 95 % [+0,0039; +0,0554]) y deja
> de serlo en 4→5 (**+0,0006**, IC 95 % [−0,0207; +0,0219]). Sobre las 11 con ≥ 4 plantillas
> sigue subiendo hasta el final (3→todo **+0,0385**, IC 95 % [+0,0206; +0,0565]).
>
> **Objetivo acotado y finito: 4 textos distintos por familia = 33 plantillas nuevas en 19
> familias**, o sea al menos 33 notas nuevas, cada una de contenido distinto. Detalle,
> métricas y límites en `ESTADO_TESIS.md`, sección «B.1 CERRADO». **Plan operativo de la
> recolección** (prioridades por familia, fuentes, reglas y qué se corre al volver):
> `6_notas_trabajo/plan_recoleccion_notas_2026-08-19.md`.
>
> **La chatura de la curva de 30 familias era AGOTAMIENTO del corpus, no saturación del
> aprendizaje** (`n_fam_bajo_tope` cae de 16,8 en k=1 a 5,0 en k=3 y a 0,0 en k=7). No usar
> esa chatura como argumento contra la recolección.
>
> **Regla de presupuesto, medida:** contar **textos distintos**, no notas. Al mismo número de
> plantillas por familia las dos formas de recolectar dan el mismo macro-F1 dentro de 0,0109
> (contra ± 0,0486 de desvío típico), y una nota que repite un contenido ya presente no mueve
> la métrica de forma medible.

**Qué se recolecta:** ampliar el corpus de forma oportunista, priorizando **las 19 familias que
están por debajo de 4 plantillas** (la lista sale de `4_resultados/resumen_capitulo4/`
+ `manifiesto_b1.json`). URLs ya identificadas en `6_notas_trabajo/mas_notas_descarga.md` para
14 familias. Si para algunas no aparecen plantillas nuevas, ese hecho se documenta como
resultado, no como tarea incumplida.

> **Ajuste 2026-08-16:** la recolección pcrisk/OCR **arranca YA en paralelo** (autorizada por
> el tutor; consume tiempo de reloj de Romina y no bloquea nada), pero con un lote chico —
> una o dos familias — midiendo el rendimiento por hora. Con ese dato, B.1 decide si escalar.
> Salir a recolectar todo de una es lo que el plan viejo hacía mal.
> **Actualización 2026-08-19:** B.1 ya dio el dato, y el lote objetivo es chico (33 textos).
> Al terminar cada lote, **re-correr `curva_aprendizaje_notas.py`**: la curva es el medidor de
> si la recolección está rindiendo, y ahora es reproducible con un comando.

**Few-shot** (paper sugerido por Cappo el 06/06/2024, arXiv 1908.06750) **deja de ser la única
salida y pasa a ser el complemento** para las familias donde la recolección no consiga llegar a
4 plantillas. Sigue en el plan; ya no es la rama principal.

**Notas sintéticas** (punto 4 del tutor): **AUTORIZADAS por el tutor** («se puede generar
datos sintéticos si es necesario», reunión 2026-08-12; OCR/transcripción también autorizado —
confirmado por Romina 2026-08-16). Siguen siendo el último recurso *metodológico*: primero
recolección real + OCR.

**Protocolo fijado 2026-08-16** (si se llega a hacer):
- **La generación va DENTRO del pliegue de entrenamiento, no antes.** Generar de una vez al
  principio a partir de notas que después caen en test es fuga, aunque las sintéticas solo
  estén en train. Generar por pliegue.
- Evaluar **únicamente** contra notas reales; el conjunto de test se mantiene **byte a byte
  igual** al actual, para que los números sigan comparables con 0,760 / 0,435.
- **Declarar el método de generación y dejarlo reproducible**: semilla; si es un LLM, cuál y
  con qué prompt (es una dependencia a declarar en metodología).
- **Control negativo:** generar también para una familia que YA tiene muchas plantillas y
  verificar que su desempeño no mejora. Si mejora, la ganancia es artefacto del generador.
- **No mezclar aumentación con rebalanceo:** «solo para familias con <3 plantillas» confunde
  las dos cosas y después no se puede decir cuál produjo la mejora. O se aumentan todas a la
  misma cantidad, o se corren las dos variantes.
- Etiqueta `sintetica` en el manifiesto, separable, igual que las transcripciones.

### C.bis Marcadores como características *(diseño fijado 2026-08-16; va DESPUÉS de B.1)*
Candidato con evidencia propia: usar los **marcadores como características**, en vez de
quitarlos. El Sprint 1.1 los *eliminó* y no mejoró; esto es el experimento opuesto. Está
respaldado por el perfil de marcadores medido, que es característico por familia — DHARMA solo
emails (17, ningún onion); CERBER solo onion (18, ningún email); RYUK email+BTC; PHOBOS solo
email; CONTI solo onion.

**Tres bloques de features separados, con ablación** (eso lo convierte en resultado y no en
corazonada). Si hay que elegir uno: **FORMA**, porque el Sprint 1.1 ya dio la evidencia (la
vista de caracteres fue la que más perdió al uniformar los marcadores ⇒ la señal estaba en la
forma):
1. **Tipos** — conteo de emails/onions/BTC/URLs/IDs por nota.
2. **Valores** — el valor exacto (hasheado).
3. **Forma** — longitud de la .onion (v2 = 16 caracteres, v3 = 56; v2 discontinuada en 2021,
   así que **fecha la campaña**), formato de la dirección BTC (`1…` P2PKH / `3…` P2SH /
   `bc1…` bech32), TLD y longitud del dominio del email, parte local aleatoria o pronunciable,
   formato del ID (largo, hex vs base64, delimitadores). Son rasgos del **herramental** de la
   familia, y el herramental se mantiene entre campañas — justo lo que P2 necesita.

**Condiciones declaradas por separado** (nunca mezclar bajo el nombre «P2»):
| Condición | Qué mide |
|---|---|
| P2 texto (hoy: 0,435) | línea base |
| P2 + forma, sin valores exactos | **la mejora honesta de generalización** |
| P2 + valores exactos | continuidad de IOCs — es búsqueda de IOC, no generalización lingüística; legítima y comparable con Lemmou, llamándola por su nombre |
| **P3** (grupos = componentes conexas del grafo de B.3) | generalización a clúster de campaña nunca visto — elimina de raíz la discusión de fuga |

Circularidad: mismo criterio y exclusiones que B.3, con corrida con/sin reportando ambas.

---

## SPRINT D — Escritura

- **PRIMERO: reescribir `generar_figuras_cap4.py` para que LEA los CSV bajados** en vez de
  cifras hardcodeadas. Hoy tiene 29 familias (falta BLACKBASTA), F1 de una corrida superada y
  el Exp. 2b pre-corrección: regenerar las figuras con el script actual produciría números
  viejos en el documento. Detalle en la auditoría del 2026-08-17 (`ESTADO_TESIS.md`).
- Bloque A1–A8 de `PENDIENTE_REDACCION.md`: los ocho resultados medidos y no escritos.
- Los cuatro puntos del tutor que **ya se contestan con datos existentes**:
  - *validación separada train/test* → es P1/P2 con `StratifiedGroupKFold` por plantilla;
    solo falta decirlo con esas palabras.
  - *justificar ML frente a búsqueda de firma directa* → firmas exactas cubren 53,3 % de las
    familias (97,0 % de acierto donde aplican) y el ML sobre bytes llega a 0,910 cubriendo el
    100 %. Se refuerza con `Pruebas.xlsx`: ID Ransomware, motor de firmas en producción,
    reconoce **9 de 30 familias** cuando se le cambia el nombre al archivo.
  - *año de detección de cada familia* → está en la hoja «Informacion sobre familias» de
    `Pruebas.xlsx`. Las seis difíciles van de 2013 a 2020, así que «son las más viejas» no se
    sostiene; calcular la relación en serio y reportarla aunque dé negativa.
  - *por qué el atacante deja rastro* → discusión con fuentes, no experimento: la marca le
    sirve al propio ransomware para no cifrar dos veces, para guardar el ID de víctima o el
    blob de clave, y para que el desencriptador que venden funcione. **Buscar bibliografía;
    no sostenerlo con razonamiento propio.**
- Corregir la ficha de `lee2022` y sumar las 9 referencias identificadas.

---

## SPRINT E — Cierre (al final de todo)

Conclusión completa, resumen/abstract, front matter, agradecimiento obligatorio al cluster del
NIDTEC, limpieza del `.bib`.

---

## Decisiones abiertas (son de Romina, no técnicas)

### D.1 *Majority voting* entre los dos frentes
El tutor lo pidió como elemento de acción 3. Choca con la decisión de mantener los frentes
independientes, y además hay un obstáculo real: **no hay muestras pareadas** — las notas vienen
de repositorios públicos y los archivos cifrados de NapierOne, así que no existe un incidente
del que se tengan los dos artefactos. Se puede *proponer* como esquema de despliegue (cada
clasificador entrega probabilidades y se votan), pero **no se puede evaluar** sin datos
pareados. Paper de referencia ya entregado por el tutor:
`5_bibliografia/reunion 02-05-2024/Majority Voting Approach to Ransomware Detection.pdf`.
Tres salidas posibles: implementarlo, proponerlo sin evaluar, o argumentar por qué no.

### D.2 Nombre de archivo como característica *(era el Sprint 4.1)*
Se descartó porque el 51 % de los nombres los puso el curador; los 42 del repositorio de
Lemmou sí son auténticos. Requiere marcar la procedencia de cada nombre en el manifiesto.

> **Corregido 2026-08-15.** La justificación anterior decía que Lemmou obtiene F = 0,920
> clasificando la familia por el nombre. **Es falso.** Ese 0,920 es de su tarea *binaria*
> —nombre de nota frente a nombre benigno, Random Forest, exactitud 98,32 %—. Su
> identificación de familia es un prototipo por reglas y marcadores + LSA como búsqueda de
> casi-duplicados, 181/182 en mundo cerrado y sin train/test. La evidencia real de que el
> nombre lleva señal de familia es **propia**: el modo «solo extensión» alcanza 0,828 con una
> cobertura del 83 %.

---

## Lo que NO está en el plan, y por qué

- **Perseguir más exactitud en archivos:** con 0,910 y seis familias probablemente sin señal,
  el rendimiento marginal es bajo. El Sprint A busca entender y blindar, no subir el número.
- **Resolver la limitación de campaña:** no es posible con NapierOne. El paper indica que cada
  familia se ejecutó **una sola vez** en una máquina preparada, de modo que más archivos son
  más archivos de la misma campaña. **Bajar una escala mayor del dataset no lo arregla** —
  conviene decírselo al tutor, que pidió «probar con el dataset grande». Sirve para estabilizar
  las métricas, no para probar generalización entre campañas. Se declara como limitación
  en §4.5.4.
- **Ampliar el número de familias más allá de las 30 de NapierOne:** decisión tomada.
