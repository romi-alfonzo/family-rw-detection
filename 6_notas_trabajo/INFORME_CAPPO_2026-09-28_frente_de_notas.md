# Informe final — frente de notas de rescate

**Tesis:** Detección de familias de ransomware en base a archivos encriptados y notas de rescate.
Romina Alfonzo y Carlos Urdapilleta. Tutor: Prof. Cristian Cappo. FP-UNA.
**Fecha:** 2026-09-28. **Alcance:** frente de notas únicamente. El frente de archivos cifrados se
informa por separado, como corresponde a dos sistemas independientes.

> **Estado del documento:** todas las cifras están verificadas contra los archivos de resultados y
> revisadas de forma cruzada por la sesión que corrió las mediciones de cierre. La única medición
> pendiente está declarada en la sección 8.

---

## 1. Qué se construyó

Un identificador de familia que, ante una nota de rescate, decide en tres pasos, igual que un
analista:

1. **Indicadores propios de la nota** — correos de contacto, direcciones de pago, URLs y formato
   del identificador de víctima. El diccionario valor → familia se arma **solo con el conjunto de
   entrenamiento**, y se descarta todo valor que en entrenamiento aparezca en más de una familia.
2. **Nombre genuino del archivo de la nota**, cuando está auditado.
3. **El texto**, con TF-IDF sobre caracteres y palabras y un clasificador lineal de margen máximo.

Los dos primeros pasos responden solo por unanimidad; ante conflicto o ausencia de coincidencia,
decide el texto.

**Corpus:** 149 notas auténticas de 30 familias (las mismas 30 de NapierOne, que emparejan los dos
frentes), agrupadas en 99 plantillas por casi-duplicado. **Cero notas sin fuente citable**, y el
manifiesto cuadra uno a uno con el disco.

---

## 2. Cómo se evalúa, y una corrección que hicimos en el camino

La pregunta que importa en un despliegue real no es si el sistema reconoce una nota que ya vio,
sino si reconoce una **plantilla nunca vista**. Por eso el corpus se parte **por plantilla**: todas
las notas de una plantilla caen del mismo lado, de modo que el texto de prueba nunca estuvo en
entrenamiento.

Durante el cierre detectamos que nuestro repartidor de plantillas, si bien respetaba esa garantía,
**no aseguraba que cada familia tuviera al menos un ejemplo en entrenamiento en cada pliegue**.
Dejaba 3,86 familias por pliegue sin material alguno, que sacaban F1 = 0 sin que el método hubiera
fallado, y esos ceros entraban al promedio macro.

El reparto corregido (P2bal) distribuye las plantillas **dentro de cada familia**. Mantiene todo lo
demás igual, y se verificó:

| Control | P2 (anterior) | P2bal |
|---|---|---|
| Plantilla en entrenamiento y prueba a la vez | 0 | 0 |
| Plantillas de entrenamiento por pliegue | 49,5 | 49,5 |
| Familias sin entrenamiento por pliegue | 3,86 | 1,00 |

De esas 3,86 familias, **1,00 es inevitable** (las de plantilla única, ausentes en uno de los dos
pliegues) y **2,86 eran defecto del reparto**. El tamaño de entrenamiento es idéntico: la mejora no
viene de entrenar con más datos.

**Es importante decirlo con precisión: el sistema no cambió.** Se construyó el 24 de agosto y el
corpus se cerró el 25. La corrección del 22 de septiembre es de la **medición**, no del método. El
preregistro de esta corrección, con siete predicciones falsificables y una puerta de entrada que
aborta si no reproduce el evaluador anterior, quedó **commiteado en el repositorio antes de correr
el experimento**, de modo que la marca temporal es verificable.

---

## 3. Resultados

Corpus de 149 notas, 30 familias, 50 semillas, plantilla nunca vista.

| | macro-F1 (30 familias) | IC 95 % | 28 evaluables | Exactitud | Exact. balanceada | MCC |
|---|---|---|---|---|---|---|
| **Sistema completo (cascada)** | **0,7417** | [0,7328; 0,7505] | **0,7946** | 0,8123 | 0,7798 | 0,8042 |
| Solo el texto (desarme interno) | 0,6551 | [0,6454; 0,6648] | 0,7019 | 0,7191 | 0,7033 | 0,7092 |
| Cascada con el reparto anterior | 0,5191 | [0,4967; 0,5416] | 0,5562 | 0,6601 | 0,5778 | 0,6446 |

**Sobre los intervalos.** Los de la tabla miden cuánto se mueve la cifra al cambiar la partición.
Medimos además el intervalo que corresponde para hablar del **corpus**, remuestreando **plantillas**
y no notas, porque las notas de una misma plantilla son casi copias y tratarlas como observaciones
independientes estrecha artificialmente el intervalo. Ese intervalo es unas diez veces más ancho, y
la conclusión se sostiene igual:

| Capa | Intervalo al 95 % | Escenario más conservador |
|---|---|---|
| **Sistema completo** | **[0,659; 0,819]** | [0,587; 0,794] |
| Solo el texto | [0,565; 0,745] | [0,507; 0,718] |

La columna principal remuestrea las plantillas **dentro de cada familia**, que es la pregunta que
corresponde al diseño: las treinta familias están fijadas por el conjunto de referencia, y lo que
podría haber salido distinto es qué plantillas conseguimos de cada una. La columna de la derecha
remuestrea sin distinguir familia, lo que además admitiría corpus sin alguna de ellas; la incluimos
porque es el escenario más desfavorable. **La conclusión es la misma en las dos: el intervalo entero
queda por encima del umbral**, y el sistema completo conserva más margen que el texto solo.

Este cálculo se hizo **dos veces, por separado**, con implementaciones escritas de forma
independiente y volviendo a calcular las predicciones desde cero en cada una. Las dos coinciden en
los cuatro decimales, en las dos columnas. Lo mencionamos porque es el número que más se va a mirar
y conviene saber que no depende de una sola implementación.

- **25 de 30 familias** superan 0,50 y **21 de 30** superan 0,70.
- Mejora del reparto corregido: **+0,2225** [+0,199; +0,246], favorable en **las 50 semillas**.
- Aporte de las capas de reglas sobre el texto solo: **+0,0866**.
- La capa de reglas, aislada, **acierta 0,9928** y resuelve **80,3 de las 149 notas (53,9 %)**.
  Con el reparto anterior acertaba 0,9755 sobre 67,7 notas: sube porque el diccionario alcanza a
  más familias, no porque entrene con más datos, que son idénticos en los dos repartos.

### Por qué las reglas deciden antes que el texto

No es una elección de comodidad: es la configuración medida como mejor. Si se hace que las reglas
**cedan** ante el texto cuando el texto está muy seguro, el sistema empeora de forma sistemática y
monótona, y la degradación es mayor cuanto antes ceden.

| Las reglas ceden… | Cambio en macro-F1 | Semillas a favor |
|---|---|---|
| siempre (equivale a texto solo) | −0,0866 | 0 de 50 |
| con confianza alta del texto | −0,0321 | 0 de 50 |
| con confianza muy alta | −0,0120 | 2 de 50 |
| casi nunca | −0,0062 | 2 de 50 |

Se exploró además una variante de cesión en el extremo, que da +0,0015. **No se adopta**: la
ganancia es del 0,2 % relativo, el punto favorable es estrecho, y el umbral se habría elegido sobre
el mismo conjunto con el que se mide. Queda documentada como explorada y descartada.

Hay además una verificación directa de por qué esas reglas funcionan. Se tomaron los seis pares de
familias que más texto comparten entre sí y se cruzaron sus marcadores —correos de contacto,
servicios ocultos, monederos y enlaces—. **Ninguno de los seis comparte un solo dato de contacto.**
El único valor que aparece en dos familias a la vez es el enlace de descarga del navegador Tor, que
no identifica a nadie, y que nuestro filtro de genéricos descarta automáticamente por aparecer en
más de una familia. Es decir: **aunque dos familias copien el texto una de la otra, sus marcadores
siguen siendo propios.** Eso es exactamente el supuesto sobre el que se apoya la primera capa de la
cascada, y hasta ahora se sostenía de forma indirecta.

### Comportamiento en uso real

El sistema puede **abstenerse** cuando la decisión del texto es dudosa. Las capas de reglas siempre
responden, porque su acierto ya es 0,99.

| Umbral | Responde | Acierta donde responde |
|---|---|---|
| Sin abstención | 100 % | 0,8123 |
| 0,50 | 77,2 % | **0,9324** |
| 1,00 | 66,6 % | **0,9864** |

Y como apoyo a un analista, que es el uso previsto, la familia correcta es la **primera propuesta
en el 81,2 %** de las notas y está **entre las tres primeras en el 86,6 %**.

---

## 4. Dónde falla, y por qué

**Las dos familias de plantilla única, BADRABBIT y CRYPTOLOCKER, dan F1 = 0.** Es estructural: con
una sola plantilla, evaluar sobre plantilla nunca vista implica no tener nada con que entrenar.
Ningún protocolo honesto lo evita, y lo declaramos así. Responde directamente a su observación de
que esas familias «no se procesan». Por eso informamos también sobre las **28 evaluables**.

Las otras tres por debajo de 0,50 son HELLOKITTY (0,293), RYUK (0,358) y JIGSAW (0,499).

**Errores de linaje.** Buena parte del error restante es confusión entre familias emparentadas, no
error arbitrario. La cascada baja los errores totales de 2.093 a 1.398, y los de linaje de 486 a
186: en fracción, del 23,2 % al **13,3 %** de los errores. La confusión más frecuente del texto es
DHARMA con PHOBOS, dos familias de linaje común, con 8,1 casos por semilla.

Lo cuantificamos con un control: si se contabilizan como acierto las confusiones dentro de los pares
emparentados, el acierto del sistema sube de 0,8123 a 0,8585 sobre 27 clases. Como fusionar clases
sube el resultado por sí solo, comparamos contra fusionar la misma cantidad de pares **elegidos al
azar** entre familias sin parentesco, promediado sobre doscientos sorteos: el azar solo llega a
0,8132. Es decir que **casi toda la ganancia es real**, y se traduce así: de los 18,8 puntos de
error del sistema, unos 4,6 son confundir dos familias que comparten el molde de la nota, cerca de
una cuarta parte del error total.

---

## 5. Limitaciones declaradas

**1. El criterio de plantilla no detecta contención.** Agrupamos por similitud de caracteres, que
no ve cuando una nota está **contenida** dentro de otra. Medido: 54 de las 149 notas (36,2 %)
tienen una hermana de su familia contenida en el entrenamiento, aunque el criterio las considere
plantillas distintas. El efecto es grande y lo informamos pegado a la cifra:

| Situación | Acierto de la cascada | Solo texto |
|---|---|---|
| Con hermana contenida en entrenamiento (54 notas) | 0,9891 | — |
| **Sin hermana parecida (91 notas)** | **0,7434** | 0,5839 |

Toda cifra de este informe corresponde a «plantilla no vista **según el criterio de casi-duplicado
declarado**». Con un criterio de contención más estricto, el corpus baja de 99 a 81 plantillas y de
28 a 21 familias evaluables, y las cifras bajan en consecuencia.

**2. Texto compartido entre familias.** Un barrido por n-gramas de ocho palabras detectó notas que
comparten texto con familias ajenas. Los seis pares con más texto en común **se verificaron uno por
uno abriendo los archivos, y ninguno es un error de etiquetado.** Lo que decide si el texto
compartido importa no es la cantidad sino la exclusividad: existe un molde de ecosistema, frases
que aparecen en seis familias distintas y cuyo uso compartido no dice nada. El par DHARMA-LOCKBIT
comparte veintisiete n-gramas, ninguno exclusivo, y no genera confusión alguna.

Los pares verificados son además **dos fenómenos distintos**, y los nombramos distinto:

- **Un molde de nota compartido**, el caso de CLOP y RYUK: las dos notas **empiezan con el mismo
  bloque continuo** de cien palabras y comparten un prefijo literal de 423 caracteres. Provienen de
  repositorios públicos que ya las separan en dos familias, y cada una conserva sus propios
  contactos, su monedero y su firma. Nuestro agrupador no las unió porque su similitud de
  caracteres queda por debajo del umbral declarado, aunque la contención llega a 0,81: otro ejemplo
  de la limitación del punto anterior.
- **Bloques de texto reutilizados**, el caso de los pares con SODINOKIBI y el de NOTPETYA con
  WANNACRY: comparten pasajes sueltos —la advertencia de no usar software de recuperación, la
  enumeración de datos exfiltrados, la amenaza de publicación— pero **no el comienzo de la nota**.
  Es práctica común del sector, no parentesco. Además provienen de fuentes independientes entre sí,
  lo que descarta que sea un artefacto de un solo recolector.

Esta distinción importa: si se llamara linaje a todos los casos, una misma familia quedaría
emparentada con tres a la vez y el concepto perdería sentido.

**3. Redundancia del corpus público.** Las fuentes públicas de notas se copian entre sí. Es un
límite del material disponible, no del método, y está documentado nota por nota.

**4. Procedencia por auditar en un subconjunto.** Un grupo de notas quedó registrado con
procedencia agregada en vez de individual. Todas tienen fuente citable, pero la auditoría fina de
ese subconjunto sigue pendiente y lo declaramos.

---

## 6. Respuestas a sus pedidos

**Métricas.** Además de macro-F1 informamos exactitud, exactitud balanceada, **coeficiente de
Matthews (0,8042)**, coeficiente de variación del F1 entre semillas (0,042), intervalos de
confianza al 95 %, F1 por familia y matriz de confusión.

**«¿Y las familias de una sola plantilla?»** Son dos, están identificadas, dan cero por
construcción, y por eso reportamos en paralelo sobre las 28 evaluables.

**Cuánto material hace falta por familia, que fue un pedido suyo.** El aporte de sumar notas se
agota pronto. Lo medimos con deltas apareados **bajo el mismo protocolo de la cifra de cabecera**,
sobre 50 repeticiones:

| Notas por familia en entrenamiento | macro-F1 | ¿el salto aporta? |
|---|---|---|
| 1 | 0,5757 | — |
| 2 | 0,6503 | sí |
| **3** | **0,6610** | **sí, y es el último que aporta** |
| 4 | 0,6607 | no |
| 6 y más | 0,660 y baja | no |

**A partir de la tercera nota por familia, sumar material no mejora el resultado.**

La medición equivalente contando **textos distintos** en vez de notas, y no notas que se repiten,
da el mismo corte en tres, pero proviene de un esquema de evaluación que entrena con un 42 % más de
material que el de la cifra de cabecera. Lo señalamos porque los dos números no salen del mismo
régimen y no conviene leerlos como si fueran uno solo. Lo que sí es común a los dos: el corte está
en tres y no se mueve, de modo que la conclusión no dependía del defecto de partición que
corregimos. Tiene una consecuencia práctica: el límite de este
frente no se levanta recolectando más notas de las familias que ya tenemos, sino consiguiendo
textos **genuinamente distintos** de las familias que hoy tienen uno o dos.

**Año de aparición frente a rendimiento.** No hay correlación significativa: r = +0,091 con
p = 0,645 sobre las 28 familias evaluables; descontando el número de plantillas, r = +0,060. Es un
análisis posterior y con 28 casos, así que se informa como exploratorio.

---

## 7. Aporte de este frente

Además del sistema, la tesis aporta la **medición de cuánto del rendimiento reportado en esta línea
de trabajo es artefacto de evaluación**. Mostramos, con números, tres cosas: que casi la mitad de
las notas de un corpus público tiene una copia casi idéntica de sí misma; que evaluar sin separar
esas copias infla el resultado; y cuánto rendimiento queda al retirar cada fuente de optimismo. La
escalera completa de protocolos, desde el escenario de mundo cerrado hasta el más estricto, se
informa en el capítulo de resultados.

---

## 8. Estado y lo que falta

El capítulo de resultados **ya tiene escritas** la cascada y el protocolo corregido, en seis
subsecciones nuevas con sus tablas. Solo se agregó: las cifras anteriores no se tocaron, y el texto
declara que corresponden a una partición peor construida.

Las mediciones del frente de notas están **cerradas**. Los dos controles que quedaban —intervalo de
confianza por remuestreo de plantillas y acierto contabilizado a nivel de linaje— se completaron y
están incorporados a este informe, en las secciones 3 y 4.

