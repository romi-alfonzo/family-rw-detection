# Guía para entender el código de la tesis

_Escrita el 2026-08-05. Pensada para leer en orden: cada archivo se apoya en el anterior._

## Cómo leer un archivo de código

Todos los scripts tienen la misma estructura, y **el orden de lectura dentro de cada uno
debería ser este**:

1. **El comentario de arriba** (entre `"""` triples). Explica qué hace el archivo, por qué, y
   cómo se usa. Si solo lees eso ya entendés el 70 %.
2. **Las constantes en MAYÚSCULAS** (después de los `import`). Son los parámetros del
   experimento: tamaños, umbrales, rutas.
3. **`def main()`, al final.** Es la receta paso a paso: qué se hace primero, qué después.
4. **Las funciones sueltas**, solo si necesitás el detalle de un paso concreto.

Los `import` de arriba se pueden saltear: solo dicen qué herramientas se usan.

---

## Orden de lectura recomendado

### 1. `2_codigo/extractor_notas.py` — el más simple (~150 líneas)

**Qué hace:** convierte cualquier archivo de nota (`.txt`, `.html`, `.hta`, PDF, imagen) en
texto plano, listo para analizar.

**Por qué existe:** las notas de rescate vienen en formatos distintos. Sin esto habría que
tratar cada formato a mano.

**Qué mirar:**
- `_leer_texto_plano()`: acá está el arreglo del bug de codificación. Detecta si el archivo
  está en UTF-16 o cp1252 en lugar de asumir UTF-8. Es el arreglo que recuperó 13 notas.
- `_limpiar_html_regex()`: quita etiquetas HTML sin necesitar librerías externas (para el
  cluster, que no tiene Internet).
- `extraer_texto()`: la función que usan los demás scripts. Devuelve `(texto, metodo)` —
  el "método" queda registrado para poder decir en la tesis cómo se extrajo cada nota.

**Concepto clave:** un archivo es una secuencia de bytes; para leerlo como texto hay que
saber *con qué codificación* fue escrito. Adivinar mal no da error, da basura.

### 2. `2_codigo/deteccion_estructural.py` — Experimento 2b (~230 líneas)

**Qué hace:** descubre las marcas que cada familia de ransomware deja en los archivos que
cifra, y clasifica con esas marcas. **No usa aprendizaje automático**: es lógica pura, así
que es el más fácil de seguir.

**Qué mirar:**
- `prefijo_comun()` y `sufijo_comun()`: dado un montón de archivos, encuentran los bytes que
  todos comparten al principio (o al final). Así se descubre que WannaCry escribe `WANACRY!`.
- La sección `# ---- 2. Clasificación leave-one-out`: acá está la parte importante.
  **Leave-one-out** significa: para clasificar un archivo, las marcas se aprenden con los
  *demás* archivos, nunca con él mismo. Si no, sería tramposo (estaría "reconociendo" un
  archivo que ya vio).
- La variable `MODOS`: son las tres variantes de la ablación (solo extensión, solo firmas
  binarias, combinado). Comparar las tres es lo que reveló que la extensión hacía casi todo
  el trabajo.

**Concepto clave:** para que una evaluación sea honesta, el modelo nunca puede haber visto
el ejemplo que se le pide clasificar.

### 3. `2_codigo/clasificador_notas_v2.py` — el corazón de la tesis (~340 líneas)

**Qué hace:** clasifica la familia de ransomware a partir del texto de la nota.

**Qué mirar, en este orden:**

a) **Las constantes `TFIDF_WORD` y `TFIDF_CHAR`.** Definen cómo se convierte el texto en
   números. `ngram_range=(1,2)` significa "palabras solas y pares de palabras";
   `(3,5)` en modo carácter significa "trozos de 3 a 5 letras". Esos trozos son las
   *características* que el modelo usa.

b) **`agrupar_neardups()`.** Calcula qué notas son casi idénticas y las agrupa. Esto produjo
   el hallazgo de las plantillas: 146 notas = 95 contenidos distintos.

c) **`evaluar()`.** El núcleo. Tres cosas que importan:
   - `Pipeline([("tfidf", ...), ("clf", ...)])`: encadena "convertir a números" y
     "clasificar" en un solo objeto. Es fundamental: garantiza que el vocabulario se
     construya **solo** con los datos de entrenamiento. Sin esto había fuga de información.
   - `StratifiedGroupKFold` vs `StratifiedKFold`: son los protocolos **P2** y **P1**. El
     primero impide que dos notas del mismo grupo (plantilla) queden una en entrenamiento y
     otra en prueba; el segundo no. De ahí salen los dos números (0,760 y 0,435).
   - `for seed in range(N_SEMILLAS)`: repite todo con 10 particiones distintas y promedia.
     Con 146 notas, una sola partición puede dar un resultado engañoso por pura suerte.

d) **`main()`.** La receta completa: cargar → agrupar duplicados → evaluar todas las
   combinaciones → guardar CSV, figura y manifiesto.

**Conceptos clave:**
- **TF-IDF**: convierte texto en números. Cada trozo de texto es una columna; el valor es
  alto si ese trozo aparece mucho en *esta* nota y poco en las demás (o sea, si es
  distintivo).
- **Validación cruzada**: en lugar de partir los datos una vez, se parte varias veces y se
  promedia. Con pocos datos es la única forma de tener una estimación confiable.
- **macro-F1**: promedia el desempeño de las 30 familias por igual. La *accuracy* premia
  acertar las familias grandes; el macro-F1 no. Por eso es la métrica principal.

### 4. `2_codigo/gridsearch_notas.py` — búsqueda de hiperparámetros

**Qué hace:** prueba cientos de configuraciones (tamaño de vocabulario, rango de n-gramas,
regularización `C`) y encuentra la mejor.

**Qué mirar:**
- `grilla()`: las opciones que se prueban. Notá el comentario sobre `max_features`: ahí está
  explicado por qué quitar el "sin límite" (causaba que el cluster matara el trabajo).
- `busqueda_anidada()`: la parte metodológicamente importante. **Anidada** significa que la
  búsqueda de la mejor configuración ocurre *dentro* del conjunto de entrenamiento, y la
  medición final se hace en un conjunto que no participó de esa búsqueda. Si se eligiera la
  configuración mirando el conjunto de prueba, el resultado quedaría inflado — es uno de los
  errores más comunes y un revisor lo busca.

**Concepto clave:** elegir parámetros es *parte* del entrenamiento. Si los elegís mirando
los datos de prueba, ya no son datos de prueba.

### 5. `2_codigo/clasificador_bytes.py` — Experimento 2c

**Qué hace:** clasifica la familia usando los bytes de la cabecera y la cola de los archivos
cifrados, con aprendizaje automático (no reglas).

**Qué mirar:**
- El comentario de arriba: explica la lógica de por qué existe este experimento y qué
  limitación tiene.
- `leer_bytes()`: toma los primeros 512 y los últimos 512 bytes de cada archivo.
- `a_matriz_posicional()` y `a_texto_bytes()`: las dos formas de representar esos bytes.
  La segunda convierte cada byte en un símbolo para poder aplicarle TF-IDF de n-gramas —
  **es la misma técnica que usás con las notas, aplicada a bytes en lugar de letras.**
- El comentario largo dentro de `configuraciones()`: explica por qué NO se usa un modelo
  lineal sobre bytes crudos (los valores de byte son categóricos, no ordinales).

**Concepto clave:** la misma idea metodológica (n-gramas + clasificador lineal) sirve para
dos artefactos distintos. Esa coherencia es un punto a favor de la tesis.

### 6. `2_codigo/family-rw-detection/` — el análisis estadístico (Experimentos 1 y 2)

- `helpers.py`: las fórmulas de entropía de Shannon, chi-cuadrado, Monte Carlo y correlación
  serial. ⚠️ Tiene un error conocido en `calculate_chi_square` (divide por los bytes
  observados en vez de por 256). Está corregido en `advanced_features.py`.
- `advanced_features.py`: calcula 275 características por archivo. Mirá `get_feature_names()`
  para ver la lista completa de qué se mide.
- `train_advanced.py`: entrena y compara subconjuntos de características. Mirá el diccionario
  `subsets` en `evaluate_feature_sets()`: ahí se ve la comparación entre "solo entropía y
  tamaño" (el experimento original) y "las 275".

---

## Los scripts `.sh` de SLURM (`2_codigo/slurm/`)

No son código de análisis: son **instrucciones para el cluster**. Cada uno dice:

```bash
#SBATCH --cpus-per-task=8     # cuántos núcleos quiero
#SBATCH --mem=32G             # cuánta memoria quiero  <- si falta, el cluster da 2 GB y falla
#SBATCH --output=...           # dónde escribir la salida
python3.11 -u mi_script.py     # y finalmente, qué ejecutar
```

Las líneas `#SBATCH` parecen comentarios (empiezan con `#`) pero SLURM las lee. El `-u` hace
que la salida se vea mientras corre, en lugar de aparecer toda al final.

---

## Las tres preguntas que conviene poder responder en la defensa

Si entendés estas tres cosas, entendés el código:

1. **¿Cómo se convierte texto (o bytes) en números que un modelo pueda usar?**
   → TF-IDF de n-gramas. Cada trozo distintivo es una columna.

2. **¿Cómo sabés que el resultado no está inflado?**
   → Tres candados: el `Pipeline` (el vocabulario no ve los datos de prueba),
   `StratifiedGroupKFold` (las plantillas casi idénticas no se reparten entre entrenamiento y
   prueba), y la búsqueda anidada (los parámetros no se eligen mirando la prueba).

3. **¿Por qué el macro-F1 y no la accuracy?**
   → Porque dos familias concentran el 25 % del corpus. La accuracy premia acertar esas dos;
   el macro-F1 trata a las 30 familias por igual.

---

## Si querés experimentar sin romper nada

Todos los scripts tienen valores por defecto seguros y los datos originales nunca se
modifican (solo se leen). Para probar rápido, varios aceptan `--smoke`:

```bash
python 2_codigo/gridsearch_notas.py --smoke
```

Y si algo sale mal, los resultados viven en `4_resultados/`, separados de los datos: borrar
una carpeta de resultados nunca destruye el corpus.
