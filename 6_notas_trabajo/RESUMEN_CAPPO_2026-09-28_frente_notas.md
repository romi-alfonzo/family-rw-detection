# Frente de notas — resumen para el Prof. Cappo

**Fecha:** 2026-09-28 · **Autoras/es:** Romina Alfonzo y Carlos Urdapilleta · FP-UNA

**Base de todas las cifras de este documento**, salvo donde se diga otra cosa:
**149 notas · 99 plantillas · 30 familias · 50 semillas**, protocolo **P2bal** (corte por
plantilla: la plantilla evaluada nunca está en entrenamiento). Azar multiclase = 1/30 = **0,033**.

> Este documento resume. Cada cifra indica de qué corrida sale; la trazabilidad completa está en
> `ESTADO_TESIS.md` y en `4_resultados/`.

---

## 1. Qué es el sistema y cuánto rinde

El sistema del frente de notas es una **cascada de tres capas** que decide como lo haría un
analista:

1. **Indicadores propios de la nota** — correos de contacto, direcciones de pago, URL, formato
   del identificador de víctima. El diccionario se arma **solo con el pliegue de entrenamiento**
   y se descartan los valores que allí aparecen en más de una familia.
2. **Nombre genuino del archivo**, cuando está verificado contra la fuente.
3. **El texto**, con TF-IDF y clasificador lineal de márgenes máximos.

Las dos primeras responden **solo por unanimidad**: si las claves apuntan a una sola familia se
contesta; ante conflicto o ausencia, decide el texto.

| métrica | cascada | texto solo | azar |
|---|---|---|---|
| **macro-F1 (30 familias)** | **0,7417** [0,7328; 0,7505] | 0,6551 [0,6454; 0,6648] | ~0,002 |
| macro-F1 (28 evaluables) | **0,7946** | 0,7019 | — |
| exactitud | 0,8123 | 0,7191 | 0,033 |
| exactitud balanceada | 0,7798 | 0,7033 | 0,033 |
| **MCC** | **0,8042** | 0,7092 | 0,000 |

*Fuente: `4_resultados/_log_p2bal_149.txt`.*

### Por qué la cifra cambió desde septiembre, cuando era 0,5191

**El sistema no cambió.** La cascada y el corpus quedaron fijados antes de esta medición. Lo que
se corrigió fue **el reparto de la partición**: `StratifiedGroupKFold` optimiza un balance global
y no garantiza que cada familia tenga al menos una plantilla en entrenamiento, de modo que dejaba
**3,86 familias por pliegue sin ningún ejemplo con que aprender**. Esas familias sacaban F1 = 0
forzado y ese cero entraba al promedio macro.

El reparto corregido (**P2bal**) reparte las plantillas *dentro* de cada familia. Controles
verificados: **0 plantillas** presentes en entrenamiento y prueba a la vez, **49,5 plantillas de
entrenamiento por pliegue en los dos repartos** (o sea, no entrena con más material), y las
familias sin entrenamiento bajan de 3,86 a 1,00.

> **Es corrección de la medición, no mejora del método.** Las cifras anteriores no son erróneas:
> responden a una partición peor construida.

### Intervalo de confianza honesto

El intervalo entre semillas mide solo cuánto se mueve la cifra al cambiar la partición. El que
además contempla el corpus se obtiene **remuestreando plantillas** —no notas, porque las notas de
una misma plantilla no son observaciones independientes—, estratificando por familia:

| | intervalo |
|---|---|
| **cascada** | **[0,6585; 0,8187]** |
| texto solo | [0,5654; 0,7446] |

El intervalo **entero** de la cascada supera 0,50, con 0,159 de margen.
*Fuente: `_log_revision_bootstrap.txt`.*

---

## 2. Los pedidos del tutor, contestados con número

### 2.1 «Ver si las notas de una misma clase tienen algún patrón» (2026-09-09)

Medido, y **la relación existe y es fuerte**. El patrón interno de cada familia se midió como la
contención media entre sus notas y las notas de su propia familia vistas en entrenamiento.

| | Pearson r | p |
|---|---|---|
| patrón vs acierto, **texto solo** | **+0,834** | < 0,00001 |
| patrón vs acierto, **cascada** | +0,715 | 0,00002 |

| nivel de patrón | familias | texto solo | cascada | aporte de las reglas |
|---|---|---|---|---|
| bajo (< 0,10) | 6 | 0,4063 | 0,5702 | **+0,1639** |
| intermedio | 4 | 0,5213 | 0,6934 | **+0,1720** |
| alto (≥ 0,40) | 18 | 0,9209 | 0,9555 | +0,0346 |

**El dato más informativo es el descenso de la correlación al pasar del texto a la cascada
(+0,834 → +0,715):** es el efecto que la arquitectura persigue, porque las capas de reglas
**debilitan la dependencia del patrón textual**. CHIMERA lo ilustra: patrón 0,0186 —sus notas no
se parecen entre sí— y acierto 1,000, resuelto íntegramente por marcadores.

**Y las familias donde la hipótesis se cumple sin atenuantes** son las que carecen a la vez de
patrón y de marcadores reutilizables: HELLOKITTY (0,0043 → 0,2267), RYUK (0,0572 → 0,2867),
JIGSAW (0,0157 → 0,4200). Son **exactamente las que quedan por debajo de 0,50**. La predicción
del tutor acierta justo donde el sistema falla, que es donde importa.

*Análisis post-hoc, no preregistrado. Fuente: `_log_patron_vs_acierto.txt`.*

### 2.2 «Si da ≤ 50 % es como tirar una moneda»

Ese criterio corresponde a una tarea **binaria**. Aquí son **30 clases**: acertar al azar es
**1/30 = 0,033**, no 0,50. El 0,5191 vigente en septiembre era 15,6 veces el azar.

La métrica que hace la comparación justa —y que el propio tutor pidió al decir «por eso
necesitamos métricas»— es el **coeficiente de correlación de Matthews (MCC)**, que vale **0 para
un clasificador al azar y 1 para uno perfecto**, sea binario o multiclase.

> **MCC de la cascada: 0,8042.**

### 2.3 «Las familias con una sola plantilla no se procesan»

La cifra separada ya existe: **macro-F1 0,7946 sobre las 28 familias evaluables**, contra 0,7417
sobre las 30.

Y hay **justificación medida** para excluirlas: las 4 notas de BADRABBIT y CRYPTOLOCKER obtienen
**acierto 0,0000 exacto**, porque su familia nunca tiene material de entrenamiento en ningún
pliegue. Se usó como control de sanidad: un valor mayor que cero habría indicado una fuga.
**No es que el clasificador falle — la tarea no está definida para ellas.**

### 2.4 Año de detección vs F1 por familia

**No explica nada.** Sobre las 28 evaluables:

| relación | r | p |
|---|---|---|
| año vs F1 de la cascada | +0,091 | 0,645 |
| año vs F1 descontando nº de plantillas | +0,060 | 0,763 |
| nº de plantillas vs F1 | −0,113 | 0,566 |

**El contraste es el resultado:** lo que determina si una familia se clasifica bien **no es cuándo
apareció, sino si sus notas se parecen entre sí** (+0,834 contra +0,091). Familias de todas las
épocas aparecen en los dos extremos: CHIMERA es de 2015 y BLACKBASTA de 2022, y ambas tienen un
texto poco distintivo que la cascada rescata.

*Fuente del año: hoja «Informacion sobre familias» de `Pruebas.xlsx`. **No** sale de MISP, que
solo cubre 5 de 28 familias. Análisis post-hoc. Fuente: `_log_anio_vs_rendimiento.txt`.*

---

## 3. El sistema en uso real

| forma de uso | resultado |
|---|---|
| **Con abstención** (umbral de margen 0,50) | responde el **77,18 %** de las notas y acierta el **93,24 %** donde responde |
| **Lista de 3 candidatas** | la familia correcta está entre las tres primeras el **86,6 %** de las veces |

**Desarme por capas** — es el dato que describe el sistema entero:

| capa que resuelve | notas | acierto sobre ellas |
|---|---|---|
| reglas exactas | 80,3 de 149 (0,5389) | **0,9928** |
| clasificador de texto | 68,7 de 149 | 0,6025 |

**Las reglas son casi infalibles pero alcanzan a poco más de la mitad; el texto alcanza a todas
pero acierta seis de cada diez.** La cascada existe para que cada capa opere donde es competente.

El valor de la lista de candidatas está sobre todo en las familias difíciles: JIGSAW pasa de
0,4200 a **0,7850**, HELLOKITTY de 0,2267 a 0,4400, RYUK de 0,2867 a 0,4867.

*Fuentes: `_log_m3_149_p2bal.txt`, `_log_topk_149.txt`.*

---

## 4. Hallazgo nuevo: un tercer par de familias que comparte el molde de la nota

Los errores del sistema **no se reparten al azar**: se concentran en pares de familias cuyas notas
comparten texto. Dos ya estaban documentados (BLACKBASTA–CONTI y DHARMA–PHOBOS). Apareció un
tercero, **CLOP–RYUK**:

- comparten **185 secuencias de 8 palabras, de las cuales 169 (91 %) son exclusivas del par**;
- **un bloque de apertura continuo de 100 palabras**, con prefijo idéntico de 423 caracteres;
- el criterio de casi-duplicado **no lo detectaba**: coseno 0,8018, por debajo del umbral 0,90.

**Las etiquetas son correctas.** Se verificó contra los archivos originales: las dos notas
provienen del mismo repositorio fuente, que las clasifica por separado, y cada una conserva sus
propios marcadores —la de RYUK termina con sus contactos, su monedero y la firma explícita de la
familia. **Es un molde de nota compartido, no un error de catalogación.**

Explica por qué RYUK es la segunda familia más difícil: sus notas **no se parecen entre sí**
(cosenos internos de 0,15 a 0,24) **y sí se parecen a las de CLOP**.

### Cuánto pesa el parentesco en el error total

| sistema | errores | dentro de un par | fracción |
|---|---|---|---|
| texto solo | 2093 | 486 | 0,2322 |
| **cascada** | **1398** | **186** | **0,1330** |

La capa de marcadores **reduce la confusión de linaje a menos de la mitad**, lo que confirma el
mecanismo: los marcadores son privados de cada familia aun cuando el texto sea compartido. Se
verificó directamente: **en los seis pares emparentados, las familias comparten cero contactos**
—el único valor cruzado es el enlace de descarga del navegador Tor.

Tratando los tres pares fuertes como una sola clase, la cascada acierta **0,8585 sobre 27
clases**. Contra un control de fusionar tres pares **al azar**, que solo llega a **0,8132**. Es
decir: **de los 18,8 puntos de error del sistema, unos 4,6 son confundir dos familias
emparentadas** — cerca de un cuarto del error total.

*Fuentes: `_log_boilerplate_149.txt`, `_log_confusion_linaje_149.txt`, `_log_acierto_linaje_149.txt`.*

---

## 5. Lo que el sistema no puede, declarado

- **En el 36,2 % del corpus hay un parecido literal fuerte** con material visto, y allí el sistema
  acierta 0,9891. **Ese acierto es del parecido, no del método.** **Sin ningún parecido —el
  61,1 %— la cascada acierta 0,7434** [0,7294; 0,7574] contra 0,5839 del texto solo. Esa segunda
  cifra es la que mide el método.
- El coseno medio del tramo de parecido alto es **0,8143**, por debajo del umbral 0,90: esas notas
  **son plantillas distintas según el criterio declarado** y aun así están contenidas en material
  visto. El criterio de casi-duplicado opera sobre coseno y **no captura la contención**.
- El corpus es **público, auditado y trazable, pero no una muestra aleatoria** del fenómeno.
- **Efecto local en contra, que se reporta:** sobre las notas más fáciles la capa de reglas
  **resta** (cascada 0,9891 contra 0,9993 del texto). Es la consecuencia aritmética de que la
  regla acierte 0,9928 y no 1: cuando el texto ya iba a acertar, el error residual de la regla se
  impone sobre una decisión correcta.

---

## 6. Nota metodológica

Todas las mediciones de esta jornada se hicieron con **preregistro commiteado en git antes de
correr**, de modo que la marca temporal es verificable, y con **puerta de entrada**: el script
aborta si no reproduce la cifra de cabecera. **Las predicciones falladas se reportan igual que las
cumplidas**; hubo cuatro y están documentadas.

El trabajo se hizo entre **dos sesiones que se revisaron mutuamente**. Se detectaron y corrigieron
cuatro errores el mismo día, y **ninguno lo encontró quien lo había cometido**:

| error | qué lo delató |
|---|---|
| remapeo de etiquetas aplicado a un array y no al otro | que la cifra fuera **físicamente imposible** |
| sesgo del remuestreo con etiquetas fijas | una **predicción preregistrada sobre la coherencia del cálculo**, que falló |
| remuestreo que contestaba una pregunta que el diseño no se hace | **nada automático**: solo discutir cuál era el estimando |
| atribuir una predicción fallada a una causa equivocada | **nada automático**: solo ir a medir la causa |

**Los dos primeros son errores de cálculo y los atrapan controles automáticos. Los dos últimos son
errores de explicación —un número bien calculado con una causa mal atribuida— y ningún control
automático los detecta, porque no hay nada que falle.** Una sección de metodología está hecha casi
toda de causas atribuidas, de modo que ese es el tipo de error contra el que hay que protegerse
deliberadamente. La revisión cruzada fue el único control que los atrapó.

Como validación adicional, el intervalo por remuestreo se calculó **dos veces, con
implementaciones independientes y predicciones recalculadas desde cero**, y ambas coinciden en los
**cuatro decimales**.

---

## 7. Dos decisiones que corresponden al tutor

1. **¿La cifra de cabecera va sobre 30 familias (macro-F1 0,7417) o sobre las 28 evaluables
   (0,7946)?** Las dos están medidas. Si «las familias de una sola plantilla no se procesan» es
   la regla, corresponde la segunda. Lo que no puede hacerse es mezclarlas sin declarar la base.

2. **¿Se extiende el frente de notas a más familias?** Hay material ya reunido para **77 familias
   nuevas** con al menos dos plantillas (las de una sola no son evaluables). Tres advertencias:
   el conteo usa el criterio de coseno, que es un **techo optimista**; hay **25 conflictos de
   etiqueta** que exigen revisión manual; y **el macro-F1 va a bajar al escalar**, lo cual es el
   resultado esperable y no un fracaso. En el frente de archivos no hay extensión posible: no
   existen archivos cifrados públicos fuera de NapierOne.
