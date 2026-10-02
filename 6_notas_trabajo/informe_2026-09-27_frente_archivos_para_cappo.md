# Informe de avance — frente de archivos cifrados (27 de septiembre de 2026, actualizado el 29)

Para: Prof. Cristian Cappo · De: Romina Alfonzo y Carlos Urdapilleta

Este informe cubre el **frente de archivos cifrados** desde el informe del 18 de agosto. Queda
cerrado: los dos experimentos pendientes de la reunión del 12/08 están medidos, se agregaron
experimentos que **mejoran el resultado publicado** hasta el sistema completo del frente, y
aparecieron hallazgos sobre el conjunto de datos que corresponde declarar.

> **La cifra del frente (28/09): 0,9998 ± 0,0001 de macro-F1 y de exactitud** con el sistema
> que apila las tres capas —bytes, rasgos estructurales y forma de la extensión— (Exp. 2g),
> 15.000 archivos, 30 familias, 5 semillas; las 30 familias por encima de 0,99 de F1 y **0,9983 ±
> 0,0011** de macro-F1 bajo tipo de documento no visto, en cinco semillas. Se reporta con su descomposición: **0,936** de
> macro-F1 empleando solo el contenido (Exp. 2e) y **0,912** de exactitud con los bytes solos (Exp. 2c). La
> capa de la extensión es la más expuesta a la limitación de campaña (Sección 5).

Todas las cifras salen de corridas en el clúster del NIDTEC (jobs 3772, 3937, 4058, 4059, 4079,
4082, 4083, 4091 y 4096) y quedaron registradas con su manifiesto de parámetros y semillas. **El frente
de notas va por carril separado** y no se trata acá; su estado está al final, en una línea.

---

## 1. Elemento de acción 2 — aporte del nombre del archivo (Experimento 2d)

Pedido: cruzar los bytes con el nombre y la extensión, para las familias que solo marcan el
nombre. Medido con **cinco columnas sobre la misma partición**, 15.000 archivos, 30 familias,
5 semillas, 5 pliegues, hiperparámetros del Exp. 2c.

| Columna | Exactitud | macro-F1 |
|---|---|---|
| Solo forma del nombre (control, sin bytes) | 0,5830 ± 0,0024 | **0,5771 ± 0,0033** |
| Solo extensión literal (control, sin bytes) | 0,9380 ± 0,0011 | **0,9244 ± 0,0036** |
| Solo bytes (referencia del Exp. 2c) | 0,9128 ± 0,0012 | 0,9117 ± 0,0011 |
| **Bytes + forma del nombre** | **0,9998 ± 0,0002** | **0,9998 ± 0,0002** |
| Bytes + extensión literal | 0,9718 ± 0,0020 | 0,9699 ± 0,0024 |

Delta pareado por semilla contra «solo bytes», macro-F1: **+0,0880 [+0,0866; +0,0894]** para la
forma del nombre y **+0,0581 [+0,0555; +0,0608]** para la extensión, 5/5 semillas en ambos.

**Dos resultados, y conviene no mezclarlos.**

**(a) La extensión es un diccionario, y le gana al método.** Un recuento previo al
entrenamiento muestra que **25 de las 30 familias tienen una sola extensión** y que solo **5 de
905 extensiones** aparecen en más de una familia —`.doc`, `.docx`, `.pptx`, `.xls`, `.xlsx`,
compartidas por las tres familias que no renombran—. Una **tabla de consulta sin aprendizaje**
que solo mira la extensión acierta **0,9724 dentro de la muestra**, y validada da **0,9244 de
macro-F1: más que los bytes (0,9117)**. Se reporta como cota superior declarada, no como
método: es memorización de identificadores de campaña.

**(b) La forma del nombre sí aporta, pero la mejora es de la combinación.** El control descarta
la explicación fácil: la forma del nombre **por sí sola alcanza apenas 0,5771**, cuarenta puntos
por debajo de lo que se esperaba. La forma define grupos gruesos de familias con muchas
colisiones y los bytes separan dentro de cada grupo; ninguna parte hace el trabajo de la otra.

**Limitación, que va pegada al número.** En NapierOne cada familia es una sola campaña. Los
controles descartan que «el nombre solo lo haga», pero **no pueden descartar que la forma del
nombre sea una huella de la campaña**. Que el esquema de renombrado lo fije el código del
malware o el operador no se puede medir con este conjunto. La redacción propuesta enuncia las
tres cosas juntas: la mejora, que la forma sola da 0,577, y que no se afirma generalización a
campañas no vistas.

**Actualización 28/09.** La validación por tipos de documento del Exp. 2f mostró que la forma
del nombre **completo** aprende también la base que NapierOne puso a cada archivo, y colapsa
con las imágenes fuera del entrenamiento. El Exp. 2g la reemplaza por la forma de la
**extensión final** —lo que agrega el ransomware— y conserva el 0,9998 (Sección 5).

---

## 2. Curva de aprendizaje del frente de archivos (A.3)

Pedido: cuántas muestras hacen falta, en los dos frentes. Subconjuntos anidados, solo bytes,
3 semillas de submuestreo.

| Archivos/familia | 10 | 25 | 50 | 100 | 200 | 350 | 500 |
|---|---|---|---|---|---|---|---|
| macro-F1 | 0,7809 | 0,8464 | 0,8723 | 0,8880 | **0,9045** | 0,9096 | 0,9117 |

> **Con 200 archivos por familia el frente ya está en 0,9045 de macro-F1. Los 300 adicionales
> hasta 500 aportan +0,0072 en total, y ningún paso individual de ahí en adelante se distingue
> de cero.** Con 10 archivos por familia —300 en total— ya se alcanza 0,7809.

Con tres semillas los intervalos son anchos (t = 4,303): el paso 50→100 (+0,0158) queda
indistinguible mientras 100→200 (+0,0164) sí aporta, con deltas casi iguales. «Indistinguible»
acá significa **no medible con tres semillas**, no «cero».

La corrida con 500 archivos/familia da **exactitud 0,9124 ± 0,0005**, que **reproduce el
0,9120 ± 0,0016 publicado** y funciona como control de que la medición es la misma.

---

## 3. Experimento 2e — una mejora del método, sin metadatos

Este experimento **no estaba pedido**. Nació de cruzar dos mediciones anteriores: el 79,6 % de
la importancia está en la cola del archivo, y SUNCRYPT (entropía de cola 4,78) y NOTPETYA
(6,58) **sí dejan algo estructurado al final** y aun así no se identifican. El capítulo ya
explicaba por qué: ese bloque «varía en cada archivo».

Ahí está la palanca. La representación posicional aprende **valores de byte en posiciones
fijas**, de modo que un pie cuyo contenido cambia es invisible para ella **aunque su presencia,
su tamaño y su aleatoriedad sean constantes dentro de la familia**. Se agregaron 44 rasgos que
describen la **forma** del archivo y no su contenido: entropía a ocho profundidades en cada
extremo, ocho bloques repartidos, tamaño y sus restos módulo 16, 512 y 4.096, χ² contra la
uniforme, bytes distintos, frecuencia máxima, ceros y fracción imprimible.

**Ninguno mira el nombre ni la extensión.** Esa es la diferencia con el Exp. 2d: la mejora es
del contenido y **no depende del esquema de renombrado**, que es la parte más expuesta a la
limitación de campaña.

| Columna (5 semillas, 15.000 archivos, 30 familias) | Exactitud | macro-F1 |
|---|---|---|
| Bytes canónico 512+512 | 0,9123 ± 0,0003 | 0,9114 ± 0,0004 |
| **Bytes + 44 rasgos estructurales** | **0,9357 ± 0,0005** | **0,9359 ± 0,0004** |
| Solo los 44 rasgos (control) | 0,8699 ± 0,0030 | 0,8680 ± 0,0032 |

**Δ pareado = +0,0246 de macro-F1, IC 95 % [+0,0237; +0,0254], 5/5 semillas.**

### La mejora cae entera en las seis familias difíciles

| | Δ medio de F1 |
|---|---|
| **Las seis difíciles del Exp. 2c** | **+0,1186** |
| Las otras 24 familias | **+0,0006** |

Por familia (semilla 0, que es la que lleva reporte por familia):

| Familia | Solo bytes | Con estructura | Δ |
|---|---|---|---|
| WASTEDLOCKER | 0,6397 | **0,8317** | +0,1920 |
| JIGSAW | 0,4285 | 0,5711 | +0,1426 |
| DARKSIDE | 0,5982 | 0,7327 | +0,1345 |
| NOTPETYA | 0,3614 | 0,4839 | +0,1224 |
| SUNCRYPT | 0,7719 | 0,8394 | +0,0675 |
| CRYPTOLOCKER | 0,6026 | 0,6552 | +0,0526 |
| BADRABBIT | 0,9827 | **1,0000** | +0,0173 |

Las otras 23 se mueven entre +0,0050 y −0,0061: **ninguna se rompe**. La media de las seis pasa
de 0,5671 a 0,6857; WASTEDLOCKER cruza por encima de 0,75 y JIGSAW por encima de 0,50. El
residuo que el capítulo identifica **se achica, no desaparece**.

### Un resultado colateral que puede importar más que el delta

**Los 44 rasgos solos alcanzan 0,8680 de macro-F1**, cuatro puntos por debajo de los 1.024
bytes posicionales, **con 23 veces menos características**. No es un agregado: es una
**representación alternativa** del problema, compacta y explicable. Y —esto es argumento, no
medición— debería resistir mejor el cambio de campaña: una campaña nueva cambia claves e
identificadores de víctima, que son *valores* de byte, pero no el tamaño ni el perfil de
entropía, que los fija el código. **Sobre NapierOne no se puede comprobar.**

### Validación sobre tipos de documento nunca vistos (agregado el 28/09)

Se sometió el 0,936 a la misma prueba que validó al 0,912 —hoy es la parte de contenido del
sistema completo (Sección 5)—:
entrenar sin un tipo de documento y evaluar sobre él, con las dos representaciones sobre cada
uno de los siete pliegues, **con el mismo modelo que la validación cruzada** (ponderación de clases;
medido en el Exp. 2h). Base: 30 familias, conjunto corregido, 28 familias en cada prueba.

| | Exactitud | macro-F1 | macro-F1 sin BLACKMATTER en `jpg` |
|---|---|---|---|
| Solo bytes | 0,8757 | 0,8617 | 0,8621 |
| **Bytes + rasgos estructurales** | **0,8786** | **0,8814** | **0,8859** |

**Δ = +0,0238 de macro-F1 [+0,0108; +0,0368], 7/7 pliegues a favor**, sin el único caso
degenerado: BLACKMATTER son **988 imágenes de sus 1.000 muestras**, así que en el pliegue `jpg` se
queda con 7 de entrenamiento, y ahí la estructura la reconoce peor que los bytes (F1 0,285 contra
0,868). Con ese caso adentro, Δ = +0,0197 [−0,0014; +0,0408], 6/7. Respecto de la validación
cruzada aleatoria, **las dos representaciones pierden lo mismo, 0,049 y 0,050 de macro-F1**, y el
delta entre ellas se conserva (+0,0238 contra +0,0246). Si los rasgos de tamaño estuvieran
aprendiendo el tipo de documento ---el riesgo concreto de esta representación--- la pérdida sería
mayor y el delta se achicaría; no ocurre ninguna de las dos cosas. **La mejora es independiente
del tipo de documento.**

La exactitud de solo bytes (0,8757) queda cerca del 0,879 publicado; lo que falta está en los
pliegues cuya composición cambió: `pdf` incorpora a NOTPETYA y BADRABBIT, y `jpg` ya no tiene a
CERBER. **Una primera versión de esta prueba se había corrido sin ponderación de clases**
(detectado el 28/09): ahí BLACKMATTER quedaba en F1 0. Con el modelo correcto solo cambia ese
pliegue (Sección 6).

---

## 4. Experimento 2e-b — qué rasgo hace el trabajo

Diagnóstico pedido por nosotros mismos, porque un jurado lo va a preguntar. Ablación por grupo
sobre «solo estructura», con validación cruzada, quitando un grupo por vez:

| Grupo quitado | n | Caída global | **Caída en las seis difíciles** |
|---|---|---|---|
| **tamaño** | 5 | −0,1227 | **−0,2247** |
| entropía de cabecera | 8 | −0,0103 | −0,0498 |
| entropía del medio | 12 | −0,0082 | −0,0303 |
| distribución de bytes | 9 | −0,0174 | −0,0233 |
| entropía de cola | 8 | −0,0566 | −0,0177 |
| salto cabecera-cola | 1 | −0,0029 | −0,0092 |
| pie no aleatorio | 1 | +0,0001 | −0,0005 |

**El tamaño es el grupo dominante**, cuatro veces y media el segundo. Pero **solo no alcanza**:
los cinco rasgos aislados dan 0,3272 global y 0,1151 en las difíciles. Es **necesario y no
suficiente**; la señal sale de la interacción entre tamaño y perfil de entropía. Escribir «el
tamaño identifica la familia» sería falso.

**El rasgo individual más importante es `tam_mod16`** —el resto del tamaño módulo 16, el tamaño
de bloque de AES—, un 28 % por encima del segundo. Tiene explicación mecánica: todas las
familias cifraron el mismo conjunto base de documentos, así que una familia con cifrado por
bloques y relleno deja el tamaño en múltiplo de 16 más su pie, y una con cifrado de flujo deja
el resto original intacto. **La distribución de `tam_mod16` en una familia es una huella del
modo de cifrado y del tamaño del pie**, y las dos las fija el código.

Las importancias del bosque ordenan los grupos distinto —el tamaño queda cuarto— y esa
discrepancia es esperable y está explicada: **ocho entropías de cola miden casi lo mismo y se
reparten el crédito**, mientras que los cinco rasgos de tamaño no tienen sustituto. **Manda la
ablación**, que mide qué pasa cuando el rasgo falta.

---

## 5. El sistema completo — Experimentos 2f y 2g (28/09)

Las técnicas del frente no son alternativas: son capas de una misma secuencia, y cada una agrega
información que las anteriores no tienen. Hasta acá se habían medido de a pares contra los bytes;
la combinación de las tres, nunca. El Exp. 2f la midió, y el 2g corrigió lo que el 2f encontró.

| Columna (5 semillas, 15.000 archivos, 30 familias) | Exactitud | macro-F1 |
|---|---|---|
| Bytes (Exp. 2c) | 0,9123 ± 0,0003 | 0,9114 ± 0,0004 |
| + rasgos estructurales (Exp. 2e) | 0,9357 ± 0,0005 | 0,9359 ± 0,0004 |
| + estructura + forma del nombre completo (Exp. 2f) | 0,9998 ± 0,0001 | 0,9998 ± 0,0001 |
| **+ estructura + forma de la extensión final (Exp. 2g)** | **0,9998 ± 0,0001** | **0,9998 ± 0,0001** |

Δ pareado sobre bytes + estructura: **+0,0639 [+0,0634; +0,0644], 5/5 semillas**, con las dos
formas. La extensión literal no suma nada sobre el sistema (+0,0000 [−0,0001; +0,0001]): queda
afuera por los datos.

### El 2f falló la validación por tipos, y la causa estaba en el conjunto de datos

Con la forma del nombre **completo**, dejar un tipo fuera mejoró seis pliegues (+0,06 a +0,17) y
**hundió el de `jpg` de 0,8052 a 0,2167**. La causa se verificó mirando los archivos:

    0001-doc.doc.avos2        0001-pdf.pdf.avos2        0001-jpg-fromweb.jpg.avos2

La base del nombre la puso NapierOne al armar su conjunto, y en las imágenes lleva un `-fromweb`
(20 caracteres contra 12). Los rasgos de forma miraban el nombre entero y aprendieron **cómo nombró
NapierOne sus archivos**, además de cómo renombra cada familia. La validación cruzada aleatoria no
lo podía ver: reparte las imágenes entre entrenamiento y prueba.

### El 2g lo corrige: solo la extensión final

Catorce rasgos que no miran la base: trece sobre la extensión final (largo, composición,
proporción hexadecimal, entropía, si es la extensión de un tipo de documento) y la cantidad de
puntos del nombre. Predicciones commiteadas antes de correr: **se cumplieron las cuatro**.

| Tipo excluido (macro-F1) | Bytes + estr. | + nombre completo (2f) | **+ extensión (2g)** |
|---|---|---|---|
| doc | 0,9321 | 0,9947 | 0,9943 |
| docx | 0,9071 | 1,0000 | 1,0000 |
| **jpg** | 0,8155 | **0,4047** | **1,0000** |
| pdf | 0,8137 | 0,9834 | 0,9800 |
| pptx | 0,8935 | 1,0000 | 1,0000 |
| xls | 0,9046 | 0,9995 | 0,9995 |
| xlsx | 0,9031 | 1,0000 | 1,0000 |
| **Promedio** | 0,8814 | 0,9118 | **0,9963** |

(Con ponderación de clases, Exp. 2h. En la primera versión, sin ponderar, la forma completa daba
0,2167 en `jpg`: el colapso no depende de la ponderación.)

Δ (2g) − (bytes + estructura) por pliegue: **+0,1149 [+0,0743; +0,1554], 7/7**. Con cinco semillas
de muestreo, el sistema completo bajo tipo no visto promedia **0,9983 ± 0,0011**; el pliegue más
bajo es siempre `pdf` (0,980 a 0,996). Al pasar de la validación aleatoria a un tipo no visto, las
representaciones de contenido pierden unas **cinco centésimas** de macro-F1; el sistema completo,
**0,0015**. La extensión que agrega el ransomware no depende del tipo
del documento, así que la capa del nombre, en vez de acoplar el clasificador al tipo, lo vuelve
casi insensible a él.

**F1 por familia, media de 5 semillas:** las 30 por encima de 0,99; las más bajas NOTPETYA 0,9978,
BADRABBIT 0,9988 y JIGSAW 0,9990. Bajo solo contenido, con el mismo promedio, DARKSIDE da 0,7515:
las familias por debajo de 0,75 son **tres** (NOTPETYA 0,48, JIGSAW 0,57, CRYPTOLOCKER 0,66), no
las cuatro que salían de una sola semilla.

**Limitación, pegada al número.** La validación por tipos acredita que el sistema no depende del
tipo de documento; **no que resista un cambio de campaña**. La limitación de campaña alcanza a todo
el frente, y la capa de la extensión es la más expuesta: en NapierOne 25 de 30 familias tienen una
sola extensión, así que esa capa aprende el esquema de renombrado de cada campaña, y si lo fija el
código o el operador no se puede medir con este conjunto. Por eso el 0,9998 va siempre con su
parte de contenido, 0,936, al lado.

---

## 6. Verificaciones de cierre — Experimento 2h (29/09)

Una sola corrida (job 4096), con 16 predicciones commiteadas antes y tres puertas de entrada (las
tres ✔), para las dudas que quedaban: la composición del conjunto, archivos duplicados, el modelo
de la validación por tipos, qué aporta cada capa y la estabilidad frente a la semilla. **Se
cumplieron 13 de las 16 predicciones**; las tres que no, en la Sección 9.

**Qué aporta cada capa** (macro-F1; validación cruzada con 5 semillas; tipo no visto, semilla 0):

| Representación | Rasgos | Validación cruzada | Tipo no visto |
|---|---|---|---|
| Solo forma de la extensión | 14 | 0,8781 ± 0,0013 | 0,8630 |
| Bytes + estructura (Exp. 2e) | 1.068 | 0,9359 ± 0,0004 | 0,8814 |
| Estructura + extensión | 58 | 0,9997 ± 0,0001 | 0,9980 |
| Bytes + extensión | 1.038 | 0,9999 ± 0,0001 | 0,9990 |
| **Sistema completo (Exp. 2g)** | 1.082 | **0,9998 ± 0,0001** | **0,9963** |

- **La extensión sola no alcanza** (0,878, menos que el contenido solo): no hace el trabajo del sistema.
- **Con la extensión, cualquiera de las dos capas de contenido basta**: 58 rasgos de estructura +
  extensión dan 0,9997. En el sistema completo la estructura no suma sobre bytes + extensión
  (+0,0001, no significativo), y bajo tipo no visto bytes + extensión es algo mejor por un solo
  pliegue (`pdf`: 0,9995 contra 0,9800). La cifra del frente sigue siendo la del sistema que apila
  todo, con esta ablación declarada al lado.
- **Por qué se complementan, medido familia por familia:** con la extensión sola, CHIMERA
  (`.crypt`), WANNACRY (`.wncry`), CONTI (`.mrbny`) y TESLACRYPT (`.micro`) tienen exactamente el
  mismo vector de rasgos (cinco minúsculas distintas, un solo carácter hexadecimal), y BADRABBIT y
  NOTPETYA comparten las extensiones de documento: a esas las separa el contenido. A la inversa,
  JIGSAW, CRYPTOLOCKER, DARKSIDE, WASTEDLOCKER y SUNCRYPT, que el contenido confunde, tienen
  extensiones de forma única (0,9988 a 1,0000 con la extensión sola).

---

## 7. Cinco hallazgos sobre el conjunto de datos

### 7.1 CERBER cifra parcialmente

Un censo por firma de tipo sobre los **29.676 archivos** de las treinta carpetas encontró que
**los 988 archivos de CERBER empiezan con la cabecera del documento original**, con firmas
distintas entre sí (JPEG, ZIP/OOXML, OLE, PDF) y con el nombre ya sustituido por CERBER. Un
perfil de entropía sobre una muestra de seis archivos lo confirma: cabecera **0,88 a 6,46** bits/byte, cuerpo y cola **7,58 a
7,64**, que es el techo de una ventana de 512 bytes.

No es un defecto del conjunto: **CERBER cifra parcialmente y preserva el comienzo del archivo**,
una forma de cifrado intermitente, documentada en la literatura reciente (Ineza et al., arXiv 2510.15133). Es un resultado sobre la familia. Obliga
a una precisión: como las demás familias sí cifran su cabecera —entropía media 7,03—, **CERBER
es la única cuya cabecera resulta legible**, de modo que esa cabecera discrimina. Su F1 de 1,000
queda sobredeterminado: lo explican tanto su sufijo constante de 64 bytes como la cabecera
preservada.

### 7.2 Un sesgo en la selección de archivos, corregido

El mismo censo destapó un defecto del procedimiento de carga. Cada carpeta trae un
`<FAMILIA>.pdf` con la documentación de NapierOne, y para excluirlo los guiones descartaban
**todo** archivo con extensión `.pdf`. Correcto para 28 familias, cuyos PDF cifrados reciben la
extensión de la familia. **Incorrecto para BADRABBIT y NOTPETYA, que no cambian la extensión**:
sus PDF cifrados siguen llamándose `.pdf`.

Se descartaban **310 muestras cifradas** —143 de BADRABBIT y 167 de NOTPETYA—, todas de las dos
únicas familias que no renombran y todas del mismo tipo de documento. NOTPETYA, la familia con
menor F1, quedaba evaluada sobre el 83 % de sus archivos.

**Las cifras publicadas no son incorrectas** —están medidas sobre un conjunto determinado y
reproducible— pero la descripción de ese conjunto debía precisarse. Corregido el criterio y
vuelto a medir: la exactitud es **0,9123**, frente a 0,9128 del mismo procedimiento sin corregir y 0,912 publicado: las tres son indistinguibles. El efecto en el
agregado es nulo; la corrección era necesaria por honestidad en la descripción, no por las
cifras.

También se apartaron **40 archivos sin cifrar** (32 de NOTPETYA y 8 de JIGSAW, el 0,13 % del
conjunto), del mismo tipo que los 12 JPEG de CERBER ya declarados, con el mismo procedimiento
reversible. Quedan en el conjunto dos PDF de JIGSAW que empiezan con firma PDF en claro: el filtro por extensión los ocultaba y el control de integridad los detectó al incorporarlos. Se declaran, no se apartaron.

### 7.3 CRYPTOLOCKER cifra de forma determinista (diagnóstico, job 4082)

Mirando los bytes sin aprendizaje: **los 143 `.doc` de CRYPTOLOCKER empiezan con los mismos 16
bytes cifrados, y los 143 `.xls` también**. Es la huella de un cifrado con clave fija y sin vector
de inicialización por archivo: documentos con el mismo comienzo en claro (todos los OLE) dan el
mismo comienzo cifrado; los OOXML, que llevan fecha y CRC en la cabecera, dan prefijos distintos.
Precisa lo que dice el Exp. 2b —que CRYPTOLOCKER «solo poseía extensión propia»—: sí tiene una
regularidad de contenido, pero **por tipo de documento**, y el criterio del 2b no podía verla. Y
depende de la clave, que es de la campaña. Está en la tesis como agregado.

### 7.4 BLACKMATTER son solo imágenes (censo, Exp. 2h)

Sobre las 29.948 muestras: **BLACKMATTER son 988 `jpg` y 12 archivos con el nombre sustituido**.
Por eso solo aparece en el pliegue `jpg` de la validación por tipos, y ahí queda con 7 muestras de
entrenamiento. El censo confirma además lo demás que el capítulo cita: CERBER sustituye el nombre
en sus 988 muestras; NOTPETYA no tiene imágenes; 25 de 30 familias usan una sola extensión, y
MAZE, junto con SUNCRYPT, usa extensiones aleatorias (573 distintas en 1.000 archivos).

### 7.5 Tres documentos repetidos: seis familias cifran de forma determinista (Exp. 2h)

Por SHA-256: **16 pares de archivos idénticos, todos dentro de una familia, ninguno entre
familias**. Son siempre los mismos tres pares de documentos de origen de NapierOne (`0066`/`0067`
doc, `0098`/`0100` jpg, `0134`/`0136` xls), en BADRABBIT, CRYPTOLOCKER, MEDUZALOCKER, RANSOMEXX y
TESLACRYPT (los tres) y NOTPETYA (uno). **Esas seis familias cifran de forma determinista**: dos
documentos iguales dan el mismo archivo cifrado. Efecto en las cifras: despreciable (en la
validación por tipos las dos copias caen en el mismo pliegue; en la cruzada, del orden de una
diezmilésima). Se declara.

---

## 8. Elemento de acción 3 — *majority voting* entre los dos frentes

**Con el sistema completo no hay margen:** archivos tiene exactitud 0,9998, así que ningún
esquema de combinación puede sumar más de 0,0002. La cuenta que sigue vale para el caso en que
**solo se dispone del contenido** del archivo, sin su nombre: archivos en exactitud 0,9357
(Exp. 2e) y notas en acierto 0,8123 con cobertura total (cascada bajo P2bal). El clasificador de
archivos falla entonces en el 6,4 % de los casos: un **oráculo perfecto** que
supiera dónde está el error rescataría **+5,2 puntos**. El riesgo opuesto es más de tres veces mayor:
archivos acierta y notas falla en el 17,6 % de los casos. Delegar conviene solo sobre un subconjunto
donde archivos falle en más del **18,8 %**, contra una tasa base del 6,4 %.

Ese umbral es alcanzable: delegar cuando el clasificador predice una de las seis difíciles, donde
su error ronda el 31 % (aproximación: 1 − F1 medio de esas seis). Ganancia esperada: **≈ +2,5 puntos**,
de 0,936 a ~0,961.

**Pero no se puede medir.** No hay muestras pareadas: las notas vienen de repositorios públicos
y los archivos de NapierOne, de modo que no existe un incidente del que se tengan los dos
artefactos. Emparejar al azar fabricaría una correlación que los datos no tienen, y el resultado
sería aritmética de las dos marginales —justamente la que se acaba de hacer—.

**Propuesta:** presentarlo como esquema de despliegue con el techo calculado y la razón medida,
en vez de implementarlo sin poder evaluarlo. Donde sí aportaría combinar no es la exactitud sino
la **cobertura de artefactos** —en un incidente real puede haber solo nota, solo archivos, o los
dos— y la **abstención**: con umbral 1,00 las notas dan acierto 0,9864 donde contestan (cobertura 0,6658, bajo P2bal), de modo
que la nota sirve como confirmador y no como votante.

---

## 9. Lo que no salió como se esperaba

Cada experimento llevó sus predicciones escritas y commiteadas **antes** de correr. Nueve
fallaron, y se dejan asentadas porque cambian la lectura:

1. **Exp. 2d:** se predijo que la forma del nombre sola daría 0,97–0,99, lo que la habría vuelto
   una segunda cota superior. Dio **0,5771**. La regla de decisión estaba escrita con sus dos
   ramas, y el resultado cayó en la que convierte a la columna en reportable.
2. **Exp. 2e:** se predijo que los 44 rasgos solos darían 0,40–0,60. Dieron **0,8680**.
3. **Exp. 2e:** se predijo que las dos familias que más subirían serían SUNCRYPT y NOTPETYA, por
   ser las que dejan cola de baja entropía. Fueron **WASTEDLOCKER y JIGSAW**, que están en el
   techo de entropía en ambos extremos. El rasgo diseñado específicamente para el caso
   —`largo_cola_no_aleatoria`— resultó de **aporte nulo**: retirarlo cambia el macro-F1 en +0,0001.

4. **Validación por tipos (28/09):** se predijo que los pliegues con menor mejora serían `pdf` y
   `jpg`, por tener distribución de tamaño distinta. `jpg` sí; `pdf` resultó el segundo con
   **mayor** mejora. El razonamiento sobre el tamaño típico del tipo era incorrecto: el rasgo que
   manda es el resto módulo 16, que no depende del tamaño típico de nada.

5. **Validación por tipos (28/09):** se predijo que solo bytes promediaría 0,86–0,90 de exactitud
   bajo tipo no visto. Dio **0,8516**, pero esa corrida no tenía la ponderación de clases de la
   medición publicada. Con el modelo correcto (Exp. 2h) da **0,8757**, dentro del rango: la falla
   era del método, no de la predicción (Sección 3).

6. **Exp. 2f (28/09):** se predijo que, bajo tipo no visto, sumar la forma del nombre mejoraría a
   bytes + estructura en los siete pliegues. En `jpg` lo hundió de **0,8052 a 0,2167**: la forma
   del nombre completo había aprendido la base que puso NapierOne (Sección 5). El Exp. 2g, que lo
   corrigió, cumplió sus cuatro predicciones.

7. **Exp. 2h (29/09):** se predijo que habría duplicados solo en CRYPTOLOCKER y NOTPETYA.
   Aparecieron en **seis** familias, siempre los mismos tres pares de documentos (Sección 7.5).

8. **Exp. 2h:** se predijo que, con ponderación, la estructura mejoraría a los bytes en los siete
   pliegues. En `jpg` los empeora en **0,0264**, por BLACKMATTER con 7 ejemplos de entrenamiento;
   sin ese caso, la mejora es positiva en los siete.

9. **Exp. 2h:** se predijo que el sistema completo superaría 0,99 en los siete pliegues. En `pdf`,
   con la semilla 0, queda en **0,980** (entre 0,992 y 0,996 con las otras cuatro). El umbral estaba
   mal fijado: el 2g ya daba 0,980 ahí. La afirmación que queda es más modesta: **0,98 o más en
   todos los pliegues y 0,9983 de promedio**.

El Exp. 2e funcionó, pero **por una razón distinta de la que lo motivó**, y el 2f falló por una
que ninguna validación cruzada aleatoria podía mostrar; las dos cosas solo se supieron porque se
midió en vez de suponerlo.

---

## 10. Decisiones que pedimos

1. **La cifra del frente.** Criterio de Romina (28/09): las técnicas son capas de una secuencia,
   y la cifra del frente es la del sistema que las apila —**0,9998**, validado con
   dejar-un-tipo-fuera (Sección 5)—, con la limitación de campaña declarada y no usada para
   excluir una capa. La parte de contenido (0,936) y la base de solo bytes (0,912) se reportan al
   lado, con la ablación de la Sección 6. ¿De acuerdo con ese encuadre, o prefiere presentar la
   capa de la extensión como cota superior, como se había propuesto para el Exp. 2d?
2. **CERBER**: declarar el cifrado parcial como resultado sobre la familia, sin tocar el diseño
   de la ventana.
3. **A.3**: la curva de aprendizaje ya está en el documento, en una subsección. Se saca sin afectar
   nada si prefiere dejarla fuera.
4. **La definición de «campaña»** ya está agregada en metodología, en una subsección nueva junto a
   la presentación de NapierOne. Pedimos que la revise: sostiene la limitación principal del frente.

---

## 11. Frente de notas

Va por carril separado, con su propio traspaso. Cifra vigente bajo el protocolo P2bal
(149 notas, 99 plantillas, 30 familias, 50 semillas): **macro-F1 0,7417 [0,7328; 0,7505] con la
cascada y 0,6551 [0,6454; 0,6648] con el texto solo**, sobre plantilla no vista según el
criterio de casi-duplicado por coseno de caracteres 0,90. Las dos mediciones que faltaban ya están hechas: con umbral de margen 0,50 el sistema
**contesta el 77,2 % de las notas y acierta el 93,2 % donde contesta**; y cuando la nota tiene una
hermana de su familia contenida en el entrenamiento, la cascada acierta 0,989. El frente está
redactado.
