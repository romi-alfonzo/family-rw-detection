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

### B.1 Curva de aprendizaje de notas
Rendimiento P1 y P2 contra cantidad de plantillas disponibles por familia. Contesta con un
número el pedido textual del tutor: *«ver el porcentaje de error para saber la cantidad de
notas a necesitar»*. Absorbe y mejora lo que era el Sprint 3.1(a): en vez de reportar el
desempeño estratificado, se reporta la curva completa.

**Por qué va antes que salir a buscar notas:** el plan anterior ponía la recolección manual
primero. Es al revés — se pueden perder días recolectando para descubrir que no movía la
aguja. La curva dice si vale la pena y cuánto falta.

### B.2 Auditar las 37 notas de procedencia «NapierOne/varios»
Pista confirmada: el repositorio `kipziptie` que aparece en `Pruebas.xlsx`, junto con
threatlabz y el `RansomNoteFiles` de Lemmou. Necesario para que la sección de procedencia del
capítulo 3 sea verificable.

### B.3 Grafo de marcadores compartidos entre plantillas *(diseño fijado 2026-08-16)*
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

---

## SPRINT C — Bifurca según lo que diga B.1

**Si la curva sigue subiendo** → ampliar el corpus de forma oportunista. URLs ya identificadas
en `6_notas_trabajo/mas_notas_descarga.md` para 14 familias. Si para algunas no aparecen
plantillas nuevas, ese hecho se documenta como resultado, no como tarea incumplida.

> **Ajuste 2026-08-16:** la recolección pcrisk/OCR **arranca YA en paralelo** (autorizada por
> el tutor; consume tiempo de reloj de Romina y no bloquea nada), pero con un lote chico —
> una o dos familias — midiendo el rendimiento por hora. Con ese dato, B.1 decide si escalar.
> Salir a recolectar todo de una es lo que el plan viejo hacía mal.

**Si la curva está plana** → documentar el límite y pasar a aprendizaje **few-shot** para las
familias con una o dos plantillas. Es la única vía metodológica cuando la recolección no puede
aportar más ejemplos. Paper sugerido por Cappo el 06/06/2024 (arXiv 1908.06750).

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
