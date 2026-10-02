# Reunión 2026-08-12 — «Revisión de resultados»

Fuente: `reunion_2026-08-12_revision_resultados.docx` (mismo directorio).
Asistentes: Romina Alfonzo, Carlos Urdapilleta. Revisado: Cristian Cappo.

Este archivo transcribe las notas del tutor **textualmente** y agrega, debajo de cada punto,
en qué estado está y cómo se aborda. La transcripción no se toca; lo agregado va marcado.

---

## Notas (textual)

1. Subir BLACKBASTA en arandu.
2. Ver el porcentaje de error para saber la cantidad de notas a necesitar, agregar también los datos en archivos encriptados.
3. Ver la validación por separado en train y test. Ver si mejora más los resultados de notas otra estrategia.
4. En caso de no encontrar notas de rescate, se puede generar datos sintéticos si es necesario.
5. En el gráfico de bytes parecería que ver más de 1024 bytes adelante y atrás encontraría el 100 % de accuracy, mostrar dónde la curva se corta en crecimiento y se vuelve constante o baja, o corregir ese dato.
6. Ver qué características tienen las familias no detectadas, si son más nuevas o más antiguas (colocar año de detección del ransomware).
7. Incluir error en la estadística de resultado (relacionado a overfitting y underfitting).
8. Justificar la metodología de uso de Machine Learning, por qué se opta por esta herramienta, qué mejora, qué ventaja tiene (contra el uso de búsqueda de signature directo).
9. ¿Por qué no se revisan los bytes del medio en archivos encriptados?
10. Porque un atacante deja ese rastro, por qué podría ser (buscar motivación).
11. Probar con el dataset grande (en arandu).
12. ¿Cuántas muestras se necesitan para obtener una clasificación relativamente confiable?

## Elementos de acción (textual)

1. **Para notas de rescate:** buscar más datasets de notas de rescate, ver si se puede hallar algún patrón de aprendizaje entre plantillas para poder detectar una no conocida.
2. **Para archivos encriptados:** buscar qué patrón se puede usar para identificar las familias críticas, si sirve cruzar con la detección por otra característica del archivo, como extensión o datos del nombre del archivo.
3. **En general:** ver si hay alguna característica común entre esos dos archivos que sirva para hacer un *majority voting* usando más características y combinar los resultados.
4. Actualizar el libro con la nueva información. Ir armando los capítulos.

---

## Cómo se aborda (agregado 2026-08-13)

### A. Ya está medido — solo falta escribirlo

| Punto | Dato que ya existe |
|---|---|
| **3** validación separada | Es exactamente P1/P2. P2 usa `StratifiedGroupKFold` agrupando por plantilla, así que ninguna variante del test aparece en train. P1 0,760 vs P2 0,435 de macro-F1. Falta explicarlo con esas palabras en el cap. 3. |
| **7** incluir el error | Las notas ya se reportan como media ± desvío sobre 10 semillas (0,760 ± 0,029 y 0,435 ± 0,057). **Falta** el equivalente en el frente de archivos, que hoy va sin desvío. |
| **8** justificar ML vs firmas | Ya está medido y es el argumento más fuerte de la tesis: la búsqueda de firma exacta cubre **53,3 %** de las familias (97,0 % de acierto donde aplica) y el ML sobre bytes llega a **0,910 cubriendo el 100 %**. Se refuerza con `Pruebas.xlsx`: ID Ransomware, que es un identificador por firmas en producción, solo reconoce **9 de 30 familias** cuando se le cambia el nombre al archivo. |
| **6** año de las familias | El año de detección de las 30 familias **ya lo tenemos**, en la hoja «Informacion sobre familias» de `Pruebas.xlsx` (2013 CRYPTOLOCKER → 2022 BLACKBASTA). Las seis difíciles son CRYPTOLOCKER 2013 · JIGSAW 2016 · NOTPETYA 2017 · SUNCRYPT 2019 · DARKSIDE 2020 · WASTEDLOCKER 2020: **van de 2013 a 2020**, así que la hipótesis «son las más viejas» no se sostiene sola. Hay que calcular la relación en serio y reportarla, sea positiva o negativa. También hay que sumar la entropía de cabecera/cola, donde SUNCRYPT (7,59/4,78) y NOTPETYA (7,34/6,58) sí se separan del resto. |

### B. Experimentos nuevos, baratos, en el cluster

| Prioridad | Punto | Qué correr |
|---|---|---|
| **1** | **5** | **El tutor tiene razón y la crítica apunta a un gráfico ya hecho.** La ablación actual es 64→0,795 · 128→0,793 · 256→0,852 · 512→**0,908**: la curva **sigue subiendo** en el último punto medido, así que el gráfico no muestra saturación. Correr 1024 y 2048 (y 4096 si el tiempo lo permite) hasta que se aplane o baje. Es el arreglo más urgente. |
| 2 | **9** | Bytes del medio. Hoy solo se miran 512 de cabecera + 512 de cola. Agregar una variante con un bloque del centro del archivo y comparar. Contesta una pregunta directa y además pone a prueba la conjetura de SUNCRYPT/NOTPETYA. |
| 3 | **1** | Subir BLACKBASTA. Es la familia que falta: `ESTADO_TESIS.md:549` ya registra que al dataset del cluster le faltan `BLACKBASTA-small` y `Z-Safe`, por eso el frente de archivos corre con 29 de 30. Con esto pasa a 30 y los dos frentes quedan con el mismo número. |
| 4 | **2, 12** | Curva de aprendizaje en los dos frentes: rendimiento vs cantidad de muestras por familia. Contesta «cuántas notas hacen falta» y «cuántas muestras para una clasificación confiable» con una sola figura por frente. En archivos es barato (hay 500/familia); en notas la curva va a ser corta y eso mismo es el resultado. |

### C. Requiere lectura o búsqueda bibliográfica

- **Punto 10** — por qué un atacante deja rastro. Es discusión, no experimento: el marcador le sirve al propio ransomware para reconocer qué ya cifró y no cifrar dos veces, para guardar el ID de víctima o el blob de clave, y para que el desencriptador que venden funcione. Hay que sostenerlo con fuentes, no con razonamiento propio.
- **Acción 1** — más datasets de notas. Ya hay URLs listadas en `fuentes_notas_descarga.md` y `mas_notas_descarga.md`.
- **Punto 4** — notas sintéticas. Anotar la propuesta; tiene el riesgo de que el modelo aprenda el generador. Si se hace, evaluar **solo** contra notas reales.

### D. Decisiones que son de Romina, no técnicas

- **Acción 3 — majority voting entre los dos frentes.** Choca con la decisión ya tomada de mantener los dos frentes independientes. Además hay un obstáculo real: **no hay muestras pareadas** — las notas vienen de repositorios públicos y los archivos cifrados de NapierOne, así que no existe un incidente del que se tengan los dos artefactos. Se puede *proponer* la combinación como esquema de despliegue (cada clasificador da probabilidades y se votan), pero **no se puede evaluar** sin datos pareados. El tutor ya entregó el paper de referencia: `5_bibliografia/reunion 02-05-2024/Majority Voting Approach to Ransomware Detection.pdf`. Hay que decidir si se implementa, se propone sin evaluar, o se argumenta por qué no.
- **Punto 11 — dataset grande.** Ojo: bajar una escala mayor de NapierOne **no arregla** la limitación de fondo. El paper dice (§4.4) que cada familia se ejecutó **una sola vez** en una máquina preparada, así que más archivos son más archivos de **la misma campaña**. Sirve para estabilizar las métricas, no para probar generalización entre campañas. Conviene decírselo al tutor.

### E. Escritura (acción 4)

Es el bloque A1–A8 de `PENDIENTE_REDACCION.md`, que ya está inventariado y no escrito.
