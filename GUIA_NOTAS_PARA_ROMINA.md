# El frente de notas, explicado — guía para Romina

> Documento de contexto, no de la tesis. Para que puedas leer tu propio capítulo sabiendo qué
> mira cada parte, y para que puedas explicarlo con tus palabras. Las páginas son las del PDF
> compilado (121 páginas, `main.pdf`).
>
> **Está escrito para leerse de arriba abajo una vez.** Después funciona como referencia.

---

## 1. Qué hace el frente de notas, en tres frases

Tenés 149 notas de rescate de 30 familias de ransomware. El sistema recibe una nota que **nunca
vio** y tiene que decir **de qué familia es**. No es detectar que es una nota de rescate: es
decir cuál de las 30 la escribió.

Eso es todo. El resto del capítulo es cómo se mide eso de forma honesta y qué sale.

---

## 2. La idea sobre la que se apoya todo: **plantilla**, no nota

Esto es lo primero que tenés que poder explicar, porque **todas las cifras cambian según cómo se
cuente**.

Las notas de rescate **se repiten**. Una familia usa el mismo texto una y otra vez, cambiando
solo el correo de contacto y el identificador de la víctima. Si tenés 18 notas de CERBER, no
tenés 18 ejemplos distintos: tenés **8 textos distintos** repetidos.

A cada texto distinto lo llamamos **plantilla**. Dos notas son la misma plantilla si se parecen
más de **0,90** en coseno de caracteres — una medida de cuánto comparten secuencias de 3 a 5
letras seguidas.

**Por qué importa:** si entrenás con una nota y evaluás con otra que es casi idéntica, no estás
midiendo si el sistema *aprendió*; estás midiendo si *reconoce una copia*. Por eso el corte entre
entrenamiento y prueba se hace **por plantilla**: todas las notas de una plantilla van juntas de
un lado o del otro.

**149 notas → 99 plantillas → 30 familias.**

👉 **Léelo en §4.11.1, pág. 58.** Es corto y es la base de todo.

---

## 3. Los protocolos: por qué la misma tesis tiene cifras tan distintas

Esta es la parte que más confunde y la que un jurado va a preguntar. Hay **cuatro formas de
partir los datos**, y cada una responde una pregunta distinta:

| protocolo | qué pregunta responde | macro-F1 |
|---|---|---|
| **L** (el de la literatura previa) | ¿reconoce una nota buscándola en un catálogo que la contiene? | 0,777 |
| **P1** | ¿reconoce una **instancia nueva de una plantilla que ya vio**? | 0,789 |
| **P2** | ¿reconoce una **plantilla que nunca vio**? | 0,459 |
| **P2bal** | lo mismo que P2, **con el reparto arreglado** | **0,7417** |

**No son cuatro resultados: es el mismo sistema respondiendo cuatro preguntas.** La honesta —y
la que usa la tesis— es P2bal: plantilla nunca vista.

### Qué pasó entre P2 y P2bal, que es el salto más grande

`StratifiedGroupKFold`, el repartidor que usaba P2, no garantiza que **cada familia tenga al
menos un ejemplo en entrenamiento**. Dejaba **3,86 familias por pliegue sin nada con que
aprender**. Esas familias sacaban F1 = 0 **forzado** —no porque el método fallara, sino porque no
se les dio material— y ese cero entraba al promedio.

P2bal reparte las plantillas **dentro de cada familia**, así toda familia con 2 o más pone al
menos una de cada lado. **Mismo tamaño de entrenamiento (49,5 plantillas por pliegue), misma
garantía.** Lo único que cambia es que deja de sortear ceros.

> **La frase exacta para decirlo:** «es una corrección de la **medición**, no una mejora del
> **método**». El sistema es idéntico; lo que estaba mal era cómo se partían los datos.

👉 **Léelo en §4.11.14, pág. 74.** Si entendés esta subsección, entendés por qué las cifras
viejas del capítulo dicen 0,52 y las nuevas 0,74.

---

## 4. El sistema: la cascada, capa por capa

El sistema **no es un clasificador**. Son **tres capas** que deciden en orden, como lo haría un
analista.

### Capa 1 — marcadores propios de la nota

La nota trae **correos de contacto, direcciones .onion, billeteras de Bitcoin, URLs**. Durante el
entrenamiento se arma un diccionario: *este correo → esta familia*.

Cuando llega una nota nueva, si contiene un marcador del diccionario **y todas sus coincidencias
apuntan a una sola familia**, se contesta esa. Si hay conflicto, no se contesta y pasa a la
siguiente capa.

Dos cosas que hacen que esto no sea trampa:
- el diccionario se arma **solo con el pliegue de entrenamiento**;
- se descarta todo valor que en entrenamiento aparezca **en más de una familia** (filtro de
  genéricos), porque ésos son infraestructura compartida, no señal.

**Resuelve 80,3 de las 149 notas y acierta 0,9928 sobre ellas.**

### Capa 2 — el nombre del archivo de la nota

Si el nombre original está verificado (`Info.hta`, `HOW_TO_DECRYPT.txt`), se usa como clave
exacta, igual que un marcador. **Solo 64 de 149 notas conservan su nombre genuino**: los
repositorios públicos renombran al catalogar.

### Capa 3 — el texto

Si ninguna regla aplicó, decide un clasificador: **TF-IDF + LinearSVC**.

- **TF-IDF** convierte cada nota en un vector de números, donde cada dimensión es «cuántas veces
  aparece esta palabra o esta secuencia de letras», pesado por qué tan rara es en el corpus. Lo
  raro pesa más, porque es lo que distingue.
- **LinearSVC** traza fronteras entre las familias en ese espacio y clasifica según de qué lado
  cae la nota nueva.

**Resuelve las otras 68,7 notas y acierta 0,6025 sobre ellas.**

> **El contraste entre las capas es el sistema entero en una frase:** las reglas son casi
> infalibles pero alcanzan a poco más de la mitad; el texto alcanza a todas pero acierta seis de
> cada diez. La cascada existe para que cada una trabaje donde sirve.

👉 **Léelo en §4.11.9 (pág. 66) y §4.11.15 (pág. 76).** La segunda tiene el desarme por capas.

### Por qué las reglas van primero, y no es arbitrario

Se probó dejar que la regla **cediera** ante el texto cuando el texto está muy seguro. Empeora de
forma sistemática. **El orden de las capas es la configuración medida como mejor**, y está en la
tabla de §4.11.15.

---

## 5. Las métricas: por qué 0,74 y 0,81 son el mismo resultado

Esto lo vas a tener que explicar sí o sí.

| métrica | qué cuenta | valor |
|---|---|---|
| **exactitud** | de cada 100 notas, cuántas acierta | **0,8123** |
| **macro-F1** | promedio del F1 **de cada familia**, todas con el mismo peso | **0,7417** |
| macro-F1 sobre 28 | igual, pero sin las 2 familias imposibles | **0,7946** |
| exactitud balanceada | promedio del acierto por familia | 0,7798 |
| **MCC** | 0 = azar, 1 = perfecto. Sirve para comparar con el azar | **0,8042** |

**Por qué el macro-F1 es más bajo:** le da **el mismo peso a CERBER (18 notas) que a BADRABBIT
(2)**. Las familias chicas que fallan lo arrastran. Y además incluye **dos familias que sacan 0
forzado** porque tienen una sola plantilla.

> **El macro-F1 de 0,7417 es la cifra más dura que tenés, no la que mejor describe el sistema.**
> Está bien reportarla —es la conservadora— pero al lado van la exactitud y el MCC.

**Y sobre el «≤ 50 % es como tirar una moneda»:** ese criterio es de una tarea **binaria**. Acá
son 30 clases y **el azar es 1/30 = 0,033**. El MCC de 0,8042 zanja la discusión sin tener que
discutirla.

---

## 6. Qué tan seguro es el número: los dos intervalos

| | |
|---|---|
| **entre semillas** | [0,7328; 0,7505] — mide cuánto se mueve al cambiar la partición |
| **por remuestreo de plantillas** | **[0,6585; 0,8187]** — mide qué pasaría con otro corpus |

El segundo es **diez veces más ancho** y es el honesto. Se remuestrean **plantillas y no notas**,
porque las notas de una plantilla son casi copias y contarlas por separado finge un tamaño de
muestra que no existe.

**Aun con el intervalo honesto, el extremo inferior queda en 0,6585: el intervalo entero supera
0,50.**

👉 **§4.11.20, pág. 83.**

---

## 7. Los tres escenarios: esto es lo que hay que saber contestar

El número global esconde tres situaciones muy distintas:

| situación | cuánto del corpus | acierto |
|---|---|---|
| la nota **se parece** a una ya vista | 36 % | **0,9891** |
| la nota es **genuinamente nueva** | 61 % | **0,7434** |
| la familia **no está en el catálogo** | — | **se abstiene el 79,1 %** |

**El primero hay que declararlo vos antes de que lo pregunten:** ese 0,9891 es del parecido, no
del método. **El segundo es el que mide el método de verdad.**

El tercero es el resultado más vendible: **ante una campaña que nunca vio, el sistema se calla en
4 de cada 5 casos** en vez de inventar. El costo es abstenerse también en el 18,8 % de las
conocidas.

👉 **§4.11.17 (pág. 79) y §4.11.21 (pág. 84).**

---

## 8. Las tres formas de usarlo

| modo | contesta | acierto |
|---|---|---|
| siempre una familia | 100 % | 81,23 % |
| siempre **tres candidatas** | 100 % | **86,6 %** |
| **puede abstenerse** | 77,18 % | **93,24 %** |

El umbral de abstención es **un parámetro de despliegue, no un hiperparámetro ajustado a los
datos**: se reporta la curva completa y el punto se elige según cuánto cuesta equivocarse.

👉 **§4.11.16, pág. 78.**

---

## 9. El límite: por qué no se puede mejorar

Hay **tres pares de familias que comparten el molde de la nota**: BLACKBASTA–CONTI,
DHARMA–PHOBOS y **CLOP–RYUK** (este último lo descubrimos ahora; el criterio de coseno no lo
veía, porque da 0,8018 y el umbral es 0,90).

**Una cuarta parte del error del sistema es confundir familias de esos pares.**

Y la prueba más fuerte de que ahí no hay nada que hacer: un clasificador **dedicado
exclusivamente a separar CLOP de RYUK**, sin ninguna otra familia interfiriendo, acierta
**0,5878** — contra 0,5000 de una moneda. **La información para separarlas no está en el texto.**

> **Esa es la frase de cierre del frente:** el límite no está en el método, está en el corpus. Y
> no es por agotamiento: se probaron **nueve vías distintas** —hiperparámetros, representación,
> estructura de clases, señales nuevas, señales exactas, agregación de decisiones— y las nueve
> dieron nulo, **cada una con su explicación**.

👉 **§4.11.19 (pág. 81), §4.11.22 (pág. 85) y §4.11.23 (pág. 86).**

---

## 10. Lo que tenés que declarar antes de que lo pregunten

1. **«Plantilla no vista» es según coseno 0,90**, y ese criterio **no detecta contención** — una
   nota puede estar contenida dentro de otra y contar como plantilla distinta. Con contención
   ≥ 0,8 el corpus pasa de 99 a 81 plantillas.
2. **El corpus es público y auditado, pero no es una muestra aleatoria** del fenómeno.
3. **Dos familias sacan cero estructural** (BADRABBIT, CRYPTOLOCKER): tienen una sola plantilla y
   nunca pueden estar en entrenamiento y prueba a la vez. Por eso se reporta también el macro-F1
   sobre las 28 evaluables.
4. **En el tramo fácil, la capa de reglas resta** un poco (0,9891 contra 0,9993 del texto solo),
   porque acierta 0,9928 y no 1: a veces pisa una decisión que el texto tenía bien.

---

## 11. Qué leer, y en qué orden

Si tenés poco tiempo, **en este orden**:

1. **§4.11.14, pág. 74** — de P2 a P2bal. Es lo que explica por qué las cifras cambiaron, y la
   frase «corrección de la medición, no mejora del método» la tenés que poder decir sola.
2. **§4.11.15, pág. 76** — la cascada por capas. Es el sistema.
3. **§4.11.17, pág. 79** — cuánto depende del parecido. **Es lo que un jurado ataca primero.**
4. **§4.11.21, pág. 84** — familias fuera del catálogo. Es tu mejor resultado.
5. **§4.11.23, pág. 86** — las vías descartadas. Es el argumento de dónde está el límite.
6. **§4.11.1, pág. 58** — la definición de plantilla, si en algún momento te perdés.

**Lo que podés saltear por ahora:** §4.11.5 (hiperparámetros, pág. 60), §4.11.11 (embeddings,
pág. 70) y §4.11.13 (evolución del corpus, pág. 72). Son resultados negativos ya cerrados: sirven
como respaldo, no hace falta que los tengas frescos.

**Y lo que decidís vos, no yo:**

- si la cifra de cabecera va **sobre 30 familias (0,7417) o sobre las 28 evaluables (0,7946)**.
  Cappo sugirió que las de una plantilla «no se procesan»; si eso es la regla, corresponde la
  segunda.
- si el corpus de **106 familias** entra como trabajo futuro (yo diría que sí, con una tabla
  chica: dice que el método escala).
- si la **corrección de normalización de URL** (que llevaría 0,7417 a 0,7492) se menciona en la
  discusión. Está medida pero **no aplicada**.

---

## 12. Dónde está cada cosa

| qué | dónde |
|---|---|
| todas las cifras, verificadas contra su fuente | `python 2_codigo/cifras_finales.py` |
| el detalle de cada medición | `ESTADO_TESIS.md` |
| el resumen para Cappo | `6_notas_trabajo/RESUMEN_CAPPO_2026-09-28_frente_notas.md` |
| lo nuevo de la última jornada | `6_notas_trabajo/REPORTE_CAPPO_2026-09-28_hallazgos_nuevos.md` |
| el sistema | `2_codigo/cascada_combinada_notas.py` y `protocolo_p2bal.py` |
