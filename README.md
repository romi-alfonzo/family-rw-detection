# Detección de familias de ransomware en base a archivos encriptados y notas de rescate

Tesis de grado — **Romina Alfonzo** y **Carlos Urdapilleta**
Tutor: Prof. Cristian Cappo · Facultad Politécnica, Universidad Nacional de Asunción

---

## Qué investiga este trabajo

Cuando una organización sufre un ataque de ransomware, el ejecutable que lo causó suele
haber sido eliminado. Lo que queda son dos artefactos: los **archivos cifrados** y la **nota
de rescate**. La pregunta de esta tesis es si esos artefactos permiten identificar
automáticamente qué familia de ransomware fue la responsable, como alternativa basada en
aprendizaje automático a herramientas de reglas manuales como ID Ransomware.

Los dos frentes se investigan **por separado**: no se propone un clasificador combinado.

## Resultados principales

### A partir de los archivos cifrados

| Enfoque | Exactitud | Cobertura | ¿Usa el nombre del archivo? |
|---|---|---|---|
| Entropía global + tamaño (2 características) | 0,166 | 100 % | no |
| Estadísticas regionales (19 características) | 0,603 | 100 % | no |
| Firmas binarias exactas | 0,517 (0,97 donde aplica) | 53 % | no |
| Extensión añadida | 0,828 | 83 % | sí — identifica la campaña |
| **Aprendizaje sobre bytes de cabecera y cola** | **0,910** | **100 %** | **no** |

*29 familias, azar = 0,034.* El hallazgo central es que la información que distingue
familias **no está en las propiedades criptográficas del contenido** —que son, en efecto,
indistinguibles— **sino en la estructura**: en los artefactos no aleatorios que cada familia
añade al archivo. La indistinguibilidad estadística resulta real, pero acotada a seis
familias de veintinueve.

### A partir de las notas de rescate

| Protocolo | Qué mide | macro-F1 | Exactitud |
|---|---|---|---|
| P1 — plantilla conocida | Identificar una nueva instancia de una plantilla ya catalogada | 0,760 | 0,818 |
| P2 — variante nunca vista | Generalizar a una plantilla que el modelo no vio | 0,435 | 0,551 |

*30 familias, azar = 0,033.* La distinción entre ambos protocolos surge de un hallazgo del
propio corpus: **las 146 notas corresponden a solo 95 contenidos distintos**, porque cada
familia reutiliza un repertorio reducido de plantillas cambiando únicamente los datos de
contacto. En el escenario comparable con las herramientas existentes (P1), el clasificador
supera el 71,93 % que obtiene ID Ransomware sobre el mismo tipo de entrada.

---

## Estructura del repositorio

```
1_documento/     La tesis en LaTeX. Se compila con: pdflatex → biber → pdflatex ×2
2_codigo/        Todo el código de los experimentos (ver más abajo)
4_resultados/    Métricas históricas. Los resultados actuales se regeneran ejecutando el código
6_notas_trabajo/ Bitácora de recolección del corpus y seguimiento de fuentes
```

Documentos de trabajo en la raíz:

| Archivo | Contenido |
|---|---|
| `ESTADO_TESIS.md` | Estado vigente, decisiones tomadas y resultados. **Punto de entrada** |
| `GUIA_CODIGO.md` | Guía de lectura del código, en orden de dificultad |
| `PLAN_MEJORAS.md` | Trabajo pendiente organizado en sprints |
| `DIAGNOSTICO_2026-07-27.md` | Auditoría completa del proyecto |
| `SERVIDOR_PASOS_AHORA.md` | Instrucciones para el clúster del NIDTEC |

## El código

| Script | Experimento | Qué hace |
|---|---|---|
| `extractor_notas.py` | — | Convierte notas de cualquier formato en texto. Detecta la codificación real (UTF-16, cp1252) |
| `clasificador_notas_v2.py` | 3 | Clasificador NLP de notas. Protocolos P1/P2, agrupamiento de casi-duplicados, macro-F1 |
| `gridsearch_notas.py` | 3 | Búsqueda anidada de hiperparámetros para el clasificador de notas |
| `normalizacion_marcadores.py` | 3 | Prueba si abstraer emails/onion/Bitcoin mejora la generalización (resultado: no) |
| `deteccion_estructural.py` | 2b | Descubre automáticamente las firmas de bytes de cada familia |
| `clasificador_bytes.py` | 2c | Clasifica por los bytes de cabecera y cola, sin usar el nombre del archivo |
| `family-rw-detection/` | 1 y 2 | Métricas estadísticas de archivos cifrados (275 características) |
| `generar_figuras_cap4.py` | — | Regenera las figuras del capítulo de resultados |
| `slurm/` | — | Scripts de encolado para el clúster del NIDTEC |

`family-rw-detection/` se versiona en su [propio repositorio](https://github.com/romi-alfonzo/family-rw-detection).

**Para entender el código, leer `GUIA_CODIGO.md`**: propone un orden de lectura y explica
qué mirar en cada archivo.

## Reproducibilidad

Cada ejecución guarda un manifiesto con los parámetros, las versiones de las librerías y las
semillas empleadas. Ninguna cifra del documento se transcribe a mano: todas las tablas se
regeneran a partir de los archivos de resultados.

```bash
pip install scikit-learn pandas numpy beautifulsoup4 matplotlib
python 2_codigo/clasificador_notas_v2.py          # corrida canónica del clasificador de notas
python 2_codigo/gridsearch_notas.py --smoke       # prueba rápida
```

Los experimentos sobre el conjunto completo de archivos cifrados se ejecutaron en el clúster
computacional del NIDTEC (FP-UNA); las instrucciones están en `SERVIDOR_PASOS_AHORA.md`.

## Sobre los datos

**Este repositorio no contiene datos.** El corpus está formado por notas de rescate y
archivos cifrados auténticos —es decir, malware real—, cuya publicación infringiría los
términos de uso de GitHub. Los `.gitignore` del proyecto están configurados para impedirlo.

Las fuentes son públicas y citables:

- **NapierOne** (Davies, Macfarlane y Buchanan, 2022) — archivos cifrados por 30 familias.
- **ThreatLabz / Zscaler** y **lemmou/RansomNoteFiles** (Lemmou, Lanet y Souidi, 2021) —
  notas de rescate en formato original.
- **PCRisk** — transcripciones documentadas, para las familias sin archivo bruto disponible.

Cada nota del corpus tiene registrada su procedencia en un manifiesto de trazabilidad.

## Agradecimientos

Los experimentos de mayor costo computacional se ejecutaron en el clúster del **Núcleo de
Investigación y Desarrollo Tecnológico (NIDTEC)** de la Facultad Politécnica de la
Universidad Nacional de Asunción, financiado por el proyecto LABO16-167 del programa
PROCIENCIA (CONACYT).
