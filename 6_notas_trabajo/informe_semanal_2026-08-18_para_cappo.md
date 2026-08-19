# Informe de avance — semana del 12 al 18 de agosto de 2026

Para: Prof. Cristian Cappo · De: Romina Alfonzo y Carlos Urdapilleta

Esta semana se atendieron los pedidos de la reunión del 12/08 sobre el frente de archivos
cifrados. Todas las cifras salen de corridas en el clúster del NIDTEC y quedaron registradas
con su manifiesto de parámetros y semillas.

---

## 1. Pedidos resueltos

**El gráfico de bytes: la curva sí satura.** Se extendió la ablación de ventana de 512 a 1024,
2048 y 4096 bytes por extremo. El **macro-F1** por ventana es: 128 B → 0,790 · 256 B → 0,792 ·
512 B → 0,851 · **1.024 B → 0,905** · 2.048 B → 0,904 · 4.096 B → 0,903 · 8.192 B → 0,901.
El máximo está en 512+512 y a partir de ahí desciende levemente. Leer más de 1.024 bytes en
total no mejora el resultado: la información se agota antes.

Se agregó un control que no estaba previsto y conviene declarar. Al agrandar la ventana, los
archivos más cortos que ella se rellenan con ceros, y ese relleno codifica el tamaño del
archivo, que es una pista ajena al contenido. Por eso la curva se midió dos veces: sobre el
corpus completo (15.000 archivos) y sobre el subconjunto sin relleno ni solapamiento
(14.783 archivos, ≥ 8.192 B). Las dos curvas van paralelas, con el subconjunto limpio unos
0,005 de macro-F1 por encima, de modo que **la forma de la curva no es un artefacto del
relleno**.

**Los bytes del medio no aportan.** Un bloque de 1.024 bytes tomado del centro del archivo
alcanza un **macro-F1 de 0,058**, con azar en 0,033. Sumado a los extremos, el macro-F1 pasa de
0,905 a 0,903: no aporta y estorba levemente. La asimetría entre extremos es marcada —
**cola sola 0,746 · cabecera sola 0,348 de macro-F1** —, coherente con que once de las quince
firmas binarias halladas sean sufijos y solo cuatro prefijos. La marca la escribe el ransomware
al final del archivo.

**Error incluido en los dos frentes.** Se repitieron las corridas canónicas sobre diez semillas
de muestreo, con hiperparámetros fijos, de modo que lo que se mide es la dispersión de la
estimación y no una nueva selección de modelo:

| Experimento | Resultado |
|---|---|
| **2c** — aprendizaje sobre bytes | exactitud **0,912 ± 0,002** · macro-F1 **0,911 ± 0,001** |
| **2b** — detección estructural, modo combinado | exactitud **0,932 ± 0,001** |
| **3** — notas, P1 (plantilla conocida) | macro-F1 **0,760 ± 0,029** |
| **3** — notas, P2 (variante nunca vista) | macro-F1 **0,435 ± 0,057** |

**BLACKBASTA incorporada.** Los dos frentes se miden ahora sobre **30 familias**, con azar en
0,033. Desaparece la asimetría 30/29 que había que aclarar en cada tabla.

**Validación separada en train y test.** Ya estaba implementada y esta semana quedó verificada
en el código: el clasificador de bytes usa validación cruzada **anidada** —la búsqueda de
hiperparámetros se ajusta dentro del pliegue de entrenamiento y se evalúa en el pliegue externo,
dando 0,891 de exactitud— y el de notas usa `StratifiedGroupKFold` agrupando por plantilla, de
modo que bajo el protocolo P2 ninguna variante del conjunto de prueba aparece en el de
entrenamiento.

---

## 2. Hallazgos nuevos

**Se identificó la marca de BADRABBIT, que figuraba como familia sin marca.** Escribe la
palabra `encrypted` codificada en UTF-16 al final de los archivos que cifra, en **965 de 1.000
archivos (96,5 %)**. El detector no la encontraba porque exigía que la marca apareciera en
*todos* los archivos de la muestra, y con un 3,5 % de excepciones la probabilidad de que una
muestra de 50 archivos no contuviera ninguna era de apenas 0,17. **Las familias sin marca
detectable bajan de cuatro a dos: NOTPETYA y SUNCRYPT.**

**Se corrigió un defecto metodológico del detector.** El criterio de unanimidad byte a byte se
reemplazó por un umbral de mayoría declarado en 0,90, y cada marca se reporta con su cobertura.
Medido sobre diez semillas, el criterio nuevo **reduce el desvío catorce veces** (exactitud del
modo combinado 0,932 ± 0,001 frente a 0,900 ± 0,016) y además **sube la media 0,032**. Con el
criterio viejo, el número de familias con marca oscilaba entre 26 y 28 según el sorteo; con el
nuevo son 28 de 30 en las diez semillas.

**Queda explicado por qué fallan seis familias y no dos.** Cruzando el tipo de marca que deja
cada familia con su macro-F1 en el clasificador:

| Marca que deja la familia | Familias | Con macro-F1 ≥ 0,98 |
|---|---|---|
| Firma binaria y extensión | 15 | 15 |
| Solo firma binaria | 2 | 1 (BADRABBIT queda en 0,978) |
| **Solo extensión** | 11 | **7** |
| Sin marca alguna | 2 | 0 |

Las cuatro que fallan del grupo «solo extensión» son JIGSAW, DARKSIDE, CRYPTOLOCKER y
WASTEDLOCKER. Como el clasificador de bytes **no usa el nombre ni la extensión**, se queda sin
la única marca que esas familias dejan. Sumadas a NOTPETYA y SUNCRYPT dan las seis familias
difíciles, que son las mismas de la corrida anterior, ahora con desvío: NOTPETYA 0,394 ± 0,023 ·
JIGSAW 0,439 ± 0,014 · DARKSIDE 0,603 ± 0,011 · CRYPTOLOCKER 0,605 ± 0,016 · WASTEDLOCKER
0,628 ± 0,013 · SUNCRYPT 0,745 ± 0,016.

**La dispersión de cada frente es, en sí misma, un resultado.** El frente de archivos tiene un
desvío de ± 0,001 de macro-F1 sobre 15.000 muestras; el de notas, ± 0,057 sobre 95 plantillas.
Es una diferencia de un factor cercano a cuarenta, y refuerza por una vía independiente lo que
ya indicaban los tres resultados negativos de optimización: **el techo del frente de notas lo
impone la cantidad de datos, no el método.**

**Validación externa de las firmas.** Se recuperaron las anotaciones de la evaluación manual de
ID Ransomware y seis familias presentan coincidencia exacta entre la firma que el detector
descubre por su cuenta y los bytes que la herramienta reporta: WANNACRY, LORENZ, MAZE, GANDCRAB,
MEDUZALOCKER y TESLACRYPT. Son dos derivaciones independientes que llegan a los mismos bytes.

**Verificación de integridad del corpus.** Se detectaron 12 archivos sin cifrar —JPEG en claro,
confirmado por sus bytes de cabecera— dentro de la carpeta de CERBER. Excluirlos mueve la
exactitud del Experimento 2c en 0,003, dentro de la dispersión entre semillas. No se determinó
si se trata de un artefacto del empaquetado del conjunto o de archivos que la familia no cifró.

---

## 3. Observación sobre el conjunto de datos mayor

Se plantea una salvedad antes de invertir en la descarga. El artículo de NapierOne indica que
cada familia se obtuvo **ejecutando una sola vez** una muestra del ransomware en una máquina
preparada. Una escala mayor del conjunto aporta más archivos **de la misma campaña**, de modo
que sirve para estabilizar las métricas pero no para evaluar generalización entre campañas. La
limitación de una campaña por familia no se resuelve por volumen.

---

## 4. Próximos pasos

**Frente de notas, en curso.** Curva de aprendizaje del rendimiento contra cantidad de
plantillas por familia, que responde con un número cuántas notas harían falta, y análisis de los
marcadores compartidos entre plantillas de una misma familia para intentar detectar una variante
no vista.

**Cruce con el nombre del archivo (elemento de acción 2).** Diseñado, sin correr. Se propone
reportar tres condiciones comparables: solo bytes (0,911 de macro-F1, la referencia), bytes más
la *forma* del nombre —largo de la extensión, si es hexadecimal o pronunciable, si conserva el
nombre base, si incorpora un correo o un identificador—, y bytes más la extensión literal. La
segunda es la que puede sostenerse entre campañas; la tercera se reportaría como cota superior
declarada, porque la extensión identifica la campaña y no la familia.

**Consulta pendiente.** El *majority voting* combinando notas y archivos tiene un obstáculo que
conviene discutir: no existen muestras pareadas. Las notas provienen de repositorios públicos y
los archivos cifrados de NapierOne, de modo que no hay un incidente del que se tengan los dos
artefactos. El esquema puede proponerse como arquitectura de despliegue, pero no evaluarse con
los datos disponibles.
