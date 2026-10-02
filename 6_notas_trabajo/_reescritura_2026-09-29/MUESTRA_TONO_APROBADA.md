# Muestra de tono aprobada para la reescritura de la tesis

**Aprobada por Romina el 2026-09-29, para toda la tesis.** Pregunta: «¿El tono de la muestra (el
párrafo del 2e) es el que querés para toda la tesis?». Respuesta: «Así está bien». Ella vio y aprobó
el **párrafo 1**. El párrafo 2 es un ejemplo adicional con las mismas reglas: ella no lo vio.

Registro: resultados (§4.7, Exp. 2e), un párrafo explicativo que motiva el experimento. Está a
mitad de camino entre lo descriptivo y lo de resultados.

## Párrafo 1

**Original** (`resultados.tex`, ≈ l. 726):

> Dos mediciones previas, tomadas juntas, señalan una limitación de la representación y no del
> problema. La primera: el 79,6\,\% de la importancia del clasificador se concentra en la cola del
> archivo. La segunda (Tabla~\ref{tab:exp2c_dificiles}): de las seis familias que quedan por debajo
> de 0,75 de F1, \textbf{dos sí depositan estructura al final} ---SUNCRYPT presenta una entropía de
> cola de 4,78 bits/byte y NOTPETYA de 6,58, frente a 7,44 del resto--- y aun así no se identifican.
> La explicación ofrecida allí es que ese bloque \textit{varía en cada archivo}: una clave, un
> identificador de víctima o un contador.

**Aprobado:**

> Dos resultados del experimento anterior apuntan a que el límite está en la representación y no en
> el problema. Por un lado, el 79,6\,\% de la importancia del clasificador se concentra en la cola
> del archivo. Por otro, de las seis familias que quedan por debajo de 0,75 de F1, dos sí dejan
> estructura al final y aun así no se identifican: SUNCRYPT tiene una entropía de cola de 4,78
> bits/byte y NOTPETYA de 6,58, frente a 7,44 en el resto (Tabla~\ref{tab:exp2c_dificiles}). La
> explicación que se propuso entonces es que ese bloque cambia de un archivo a otro, como ocurriría
> con una clave, un identificador de la víctima o un contador.

## Párrafo 2 (ejemplo, no visto por Romina)

**Original:**

> Ahí reside la oportunidad. La representación posicional aprende \textbf{valores de byte en
> desplazamientos fijos}, de modo que un bloque cuyo contenido cambia en cada archivo le resulta
> invisible \textit{aunque su presencia, su tamaño y su grado de aleatoriedad sean constantes dentro
> de la familia}. Dicho de otro modo: «hay doscientos bytes poco aleatorios al final y el tamaño es
> múltiplo de dieciséis» constituye un rasgo de familia aun cuando esos doscientos bytes sean
> distintos en cada archivo, y ninguna representación por valores posicionales puede expresarlo.

**Reescrito:**

> Esto abre una posibilidad. La representación posicional aprende qué valor de byte aparece en cada
> desplazamiento, de modo que no puede ver un bloque cuyo contenido cambia en cada archivo, aunque su
> presencia, su tamaño y su grado de aleatoriedad sean constantes dentro de la familia. Una
> descripción como «hay doscientos bytes poco aleatorios al final y el tamaño es múltiplo de
> dieciséis» constituye un rasgo de familia aun cuando esos doscientos bytes sean distintos en cada
> archivo, y ninguna representación por valores posicionales puede expresarla.

## Qué se hizo y qué no

- **Fuera:** la negrita y la cursiva de énfasis, las rayas, «La primera: / La segunda:», «Ahí reside» y
  «Dicho de otro modo».
- **Verbos más llanos:** depositan → dejan, presenta → tiene.
- **Queda idéntico:** todas las cifras, las referencias y el alcance de cada afirmación. «Apuntan a»,
  neutro, en lugar de «señalan». La fuerza de «constituye» y de «ninguna… puede» no se tocó. El
  ejemplo entre comillas va palabra por palabra.
- **El cuidado clave:** «una clave, un identificador o un contador» sigue siendo un EJEMPLO («como
  ocurriría con»). Escribirlo como «porque contiene…» lo habría vuelto una causa afirmada: el error
  de explicación de la revisión del 29-09, metido por el estilo.

## Reglas del pase (acordadas entre las tres sesiones y aprobadas por Romina)

1. Es un pase de **forma**, no de contenido: no se toca ningún número, métrica, base, referencia ni
   matiz («debería», «cota superior», «con una campaña por familia»). Las correcciones de contenido
   aprobadas son aparte: Filiz et al. y Lemmou en el marco teórico, y el A.2 al final.
2. Negritas solo para un término que se define; en el texto corrido, ninguna.
3. Rayas solo cuando hagan falta; si no, comas, paréntesis u otra oración.
4. Sin las muletillas «La primera: / La segunda:», «Dicho de otro modo», «Ahí reside», «Cabe destacar»,
   ni «no es X, sino Y» en serie.
5. En positivo («Carpeta recordada»): variar el largo de las oraciones y no abrir dos párrafos seguidos
   con la misma estructura.
6. La misma voz impersonal de ahora. No se reordenan capítulos, y en la tesis se agrega y se pule, no se
   borra contenido.
7. Control con números, antes y después: `inventario_estilo.py --comparar` contra la línea de base de
   `antes/`. Las negritas, rayas y muletillas tienen que bajar; las cautelas y las páginas por capítulo
   no pueden bajar. `inventario_cifras.py`: las mismas cifras. Las oraciones con marcador de
   conclusión nuevas o cambiadas se leen una por una.
8. Edición de a una sesión por vez: avisar «edito resultados.tex, §X–Y» y al terminar «libre». Para
   compilar, «compilo» y «listo». La Discusión la edita solo «Carpeta recordada»; los frentes le mandan
   sus párrafos por mensaje.

Línea de base de los conteos: `antes/estilo.csv`, crudo, como `wc -w` y `grep -o`, con comentarios.
Usar esa y no un conteo propio, para que el antes y el después se midan igual.
