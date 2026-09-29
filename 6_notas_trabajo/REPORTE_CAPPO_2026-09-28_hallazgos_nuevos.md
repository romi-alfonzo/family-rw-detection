# Frente de notas — lo nuevo desde el último reporte

**Fecha:** 2026-09-28 · Romina Alfonzo y Carlos Urdapilleta · FP-UNA

> Complementa el resumen general (`RESUMEN_CAPPO_2026-09-28_frente_notas.md`). Acá va **solo lo
> descubierto después**. Base, salvo aviso: **149 notas · 99 plantillas · 30 familias · P2bal ·
> 50 semillas**.

---

## 1. El sistema sabe cuándo no sabe

Hasta ahora el escenario «familia fuera del catálogo» tenía **un solo caso real**. Ahora está
medido sistemáticamente: se retira una familia entera del entrenamiento y se evalúa sobre sus
notas, repitiendo con las treinta.

> **Ante una familia que nunca vio, el sistema se abstiene en el 79,1 % de los casos**, al costo
> de abstenerse también en el **18,8 %** de las notas de familias conocidas —donde acierta
> **0,9156** sobre lo que responde.

Las dos cifras van juntas: una tasa de rechazo alta se consigue trivialmente subiendo el umbral
hasta no contestar nunca.

**Dos controles respaldan la medición:** el acierto sobre la familia ausente es **exactamente
cero** (si diera más, habría fuga), y la capa de reglas —que responde sin pasar por el umbral—
**casi no se activa** ante lo desconocido: 0,0956 contra 0,5138.

**Dónde falla, y era previsible:** las familias que tienen **un pariente en el catálogo** se
rechazan mucho menos (**0,5097 contra 0,8421**). Ante una campaña nueva emparentada con una
conocida, el sistema no duda: le atribuye la familia del pariente.

---

## 2. El intervalo de confianza honesto

El intervalo que veníamos reportando mide solo la variación entre particiones. El que además
contempla el corpus se obtiene **remuestreando plantillas** —no notas, porque las notas de una
plantilla son casi copias y no son observaciones independientes— y **estratificando por
familia**, porque las 30 familias no son una muestra sino un conjunto fijado por NapierOne.

| | estimación | entre semillas | **por plantilla** |
|---|---|---|---|
| texto solo | 0,6551 | [0,6454; 0,6648] | [0,5654; 0,7446] |
| **cascada** | **0,7417** | [0,7328; 0,7505] | **[0,6585; 0,8187]** |

Es **diez veces más ancho**. Aun así, **el intervalo entero de la cascada permanece sobre 0,50**,
con 0,159 de margen. La afirmación se sostiene bajo la estimación más exigente disponible.

El cálculo se hizo **dos veces, con implementaciones independientes**, y coinciden hasta el
cuarto decimal.

---

## 3. Un cuarto del error es confusión entre familias emparentadas

Tratando como una sola clase los tres pares que comparten molde de nota, el acierto pasa de
0,8123 a **0,8585**. Esa métrica **sube siempre por construcción**, así que se midió contra un
control: fusionar tres pares **al azar** solo llega a **0,8132**.

> De los 18,8 puntos de error del sistema, **unos 4,6 son confundir dos familias que comparten el
> molde de la nota** — cerca de una cuarta parte del total.

**Hallazgo asociado:** apareció un **tercer par no documentado, CLOP–RYUK**. Comparten un bloque
de apertura de 100 palabras y 169 secuencias de ocho palabras exclusivas del par. El criterio de
casi-duplicado no lo detectaba (coseno 0,8018, bajo el umbral 0,90). **Las etiquetas son
correctas**: misma fuente, y cada nota conserva sus propios contactos.

---

## 4. Cinco vías de mejora evaluadas, ninguna aporta — y por qué

Todas con predicción registrada antes de correr y prueba de entrada que exige reproducir la cifra
de referencia.

| vía | Δ macro-F1 | causa del resultado nulo |
|---|---|---|
| clasificación jerárquica por linaje | −0,0011 | un clasificador **dedicado solo a CLOP y RYUK acierta 0,5878**, contra 0,5000 de una moneda |
| capa de contención literal | +0,0000 | la señal ya está explotada: 4.163 decisiones sin **una sola** discrepancia con el texto |
| extensión de cifrado como clave | +0,0000 | solo **12 de 149** notas la conservan: las fuentes publican las notas saneadas |
| combinación de vistas por votación | −0,0015 | la concatenación vigente ya es adecuada |
| forma del nombre del archivo | sin aporte | señal contaminada por convenciones de catalogación |

**Lo más concluyente:** un clasificador binario entrenado **exclusivamente** con las notas de
CLOP y RYUK, sin interferencia de las otras 28 familias, acierta **0,5878**. Es la evidencia más
directa de que **la información para separarlas no está en el texto**. No es sobrecarga del
clasificador: la señal no existe. Esas confusiones son el 11,3 % del error total.

Con los resultados negativos previos —hiperparámetros, abstracción de marcadores, embeddings,
metadatos, estilometría— suman **nueve evaluaciones convergentes** sobre aspectos distintos del
método. **El límite del frente de notas no está en el método.**

---

## 5. Sobre el catálogo MISP que usted nos envió

Se usó para dos cosas, y **las dos dieron resultado negativo informativo**:

**Como diccionario de extensiones** (735 registradas, 673 de una sola familia): **no resuelve
ninguna de las 149 notas**. Verificado extensión por extensión: `.gacmw`, `.rfncw`, `.ibkfz`,
`.lgzcfcr` y `.eebf08` son **cadenas generadas aleatoriamente por víctima** y no son registrables
en ningún catálogo; `.gdcb`, `.krab` y `.sz40` son fijas pero **el catálogo no las tiene** —para
GandCrab registra solo `.Crab` y `.CRAB`.

> Esto delimita también el alcance de **ID Ransomware**, que se apoya en la extensión: las
> extensiones aleatorias por víctima son un límite del enfoque, no de nuestra implementación.

**Como fuente de alias**, para resolver los conflictos de etiqueta del corpus ampliado: confirma
**solo dos fusiones** (`alphv`=`blackcat`, `revil`=`sodinokibi`) y declara **familias distintas**
al resto. Eso corrige una hipótesis que teníamos: **los conflictos no son errores de etiquetado
sino parentesco real**, el mismo fenómeno que CLOP–RYUK.

---

## 6. Qué pasa al ampliar el catálogo a 106 familias

Se corrió el mismo sistema sobre un corpus ampliado con las fuentes públicas ya reunidas: **596
notas, 106 familias**. Es una **base distinta**, con etiquetas de las fuentes sin auditoría de
procedencia, y **su cifra no es comparable** con la del núcleo.

| conjunto | familias | macro-F1 |
|---|---|---|
| global | 106 | 0,6485 |
| **restringido a las 30 originales** | 30 | **0,7419** |

**Las 30 familias del núcleo mantienen su rendimiento** (0,7419 contra 0,7417) con 76 familias
más compitiendo **y** 108 notas nuevas incorporadas a esas mismas familias. Lo que baja en el
global es la dificultad de la tarea, no el método: **el sistema escala**. 42 de las 76 familias
nuevas superan F1 0,70.

**Y qué se degrada exactamente:**

| | Base B (106) | Base A (30) | caída |
|---|---|---|---|
| nota parecida a una ya vista | **0,9587** | 0,9891 | 0,030 |
| nota sin ningún parecido | 0,6624 | 0,7434 | **0,081** |

> Ampliar el catálogo **no deteriora el reconocimiento de variantes conocidas; deteriora la
> generalización a notas nuevas.**

---

## 7. Un defecto de implementación, y el margen que queda

El filtro que descarta marcadores compartidos entre familias operaba sobre el valor literal de
cada dirección, de modo que **el mismo sitio entraba al diccionario fragmentado en varias
claves**: la dirección del navegador Tor aparece bajo **siete formas** distintas. Una variante
poco frecuente puede quedar asociada a una sola familia del pliegue, superar el filtro y hacer
que la regla responda con certeza injustificada.

Unificarlas eleva el macro-F1 a **0,7492** (Δ +0,0076, IC [+0,0049; +0,0102], **sin una sola
semilla desfavorable en 50**) y el acierto de la capa de reglas a 0,9977.

**La corrección no se incorporó a las cifras del capítulo**, que se reportan con la
implementación original. Se documenta porque delimita el margen restante: **lo que queda por
ganar está en el detalle de implementación, no en el método.**

---

## 8. Nota metodológica

Todas las mediciones se hicieron con **predicción registrada y confirmada en el control de
versiones antes de ejecutarse** —la marca temporal es verificable— y con **prueba de entrada**:
el procedimiento se interrumpe si no reproduce la cifra de referencia. **Las predicciones
fallidas se reportan igual que las cumplidas.**

Durante la jornada se detectaron y corrigieron **cinco errores propios**. Ninguno lo encontró
quien lo había cometido, y **cuatro se detectaron porque una cifra no coincidía con otra ya
conocida**, no porque una prueba fallara.

De ahí una distinción que conviene dejar asentada en la metodología: los **errores de cálculo**
los detectan las pruebas automáticas; los **errores de explicación** —un número correctamente
calculado con una causa mal atribuida— **no los detecta ninguna**. Y una sección de metodología
está compuesta casi por entero de causas atribuidas.
