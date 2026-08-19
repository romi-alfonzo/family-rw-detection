# Instrucciones para Claude — tesis de familias de ransomware

Este archivo se carga solo al abrir una sesión en esta carpeta. Es el contrato de trabajo.
No repite el contenido de los otros documentos: dice qué leer y qué no hacer nunca.

## Qué leer antes de responder

1. **`ESTADO_TESIS.md`** — estado vigente, decisiones tomadas, todos los resultados. Es el
   punto de entrada. Leerlo **siempre**, sin que haga falta pedirlo.
2. **`PLAN_MEJORAS.md`** — el trabajo pendiente, en sprints A–E.
3. **`6_notas_trabajo/reunion_2026-08-12_revision_resultados.md`** — los pedidos del tutor.
   Mandan sobre cualquier plan anterior.
4. Según el tema: `PENDIENTE_REDACCION.md` (lo medido y no escrito), `GUIA_CODIGO.md`,
   `SERVIDOR_PASOS_AHORA.md` (clúster), `DIAGNOSTICO_2026-07-27.md`.

## Reglas que no se negocian

- **Nunca agregar `Co-Authored-By` de Claude a un commit.** Es trabajo académico de Romina.
- **Cada cambio de código se commitea a `develop` y se pushea en el momento**, con una
  descripción breve en español y sin coautoría. No esperar a que Romina lo pida.
- **Nunca commitear datos.** El corpus son notas de rescate y archivos cifrados auténticos
  (malware real). Publicarlos infringe los términos de GitHub. Al repositorio va el **código**
  y el **documento**, nada más.
- **En la tesis solo se AGREGA.** El pulido es una pasada única al final. No reescribir ni
  reordenar capítulos por iniciativa propia.
- **La conclusión se escribe al final de todo**, nunca antes. Decisión tomada.
- **Los dos frentes —notas y archivos— van separados.** No hay clasificador combinado.
  (El tutor pidió evaluar *majority voting*; está como decisión abierta en `PLAN_MEJORAS.md`,
  sin resolver.)
- **Solo las 30 familias de NapierOne.** No ampliar el número de familias.
- **Toda cifra que vaya a la tesis necesita fuente verificable y citable**, y hay que dejar
  registrado de dónde salió.
- **Antivirus y configuración de Windows los toca Romina, no Claude.**

## Cómo trabajar

**Verificar, no recordar.** Antes de afirmar un número, abrir el archivo que lo contiene. Los
errores caros de este proyecto salieron todos de citar de memoria. Si un dato no se puede
verificar, decirlo explícitamente en vez de suavizar la frase.

**Nada importante vive solo en el chat.** Lo que se descubre o se decide se escribe en
`ESTADO_TESIS.md` en el momento, no al final de la sesión. Un chat comprimido pierde detalle;
un archivo no.

**Registrar desde lo pegado, no desde la descarga.** Cuando vuelve una corrida, Romina pega
la salida del terminal en el chat. Escribir el resultado en `ESTADO_TESIS.md` **en ese
momento**, a partir de lo pegado. La descarga de los CSV es respaldo y verificación, nunca
requisito para registrar. No condicionar el avance a que baje archivos.

Si a lo pegado le falta algo —está cortado, el log trae solo un extracto, hace falta una
cifra que no aparece— **pedir un bloque copiable concreto** (`tail -N`, `cat` de un CSV) y
pedirlo **en el momento**, no después: el `/scratch` se limpia y los scripts sobrescriben sus
carpetas de salida. Así se perdió el reporte por familia completo del job 3633.

**Toda cifra lleva su métrica pegada.** Nunca escribir «98 %» a secas: decir «98 % de
acierto donde la firma aplica», «macro-F1 0,911», «cobertura 0,572». En este proyecto conviven
exactitud, exactitud balanceada, macro-F1, cobertura y acierto-donde-aplica, y las tablas del
Exp. 2b tienen tres columnas que se confunden entre sí. Un número sin métrica ya causó
confusión más de una vez.

**Sesiones cortas.** Conviene abrir un chat nuevo por cada fase de trabajo antes que estirar
uno durante días.

**Comandos del clúster:** cada bloque copiable tiene que empezar con
`cd /scratch/ralfonzo/tesis &&`. Romina entra en `~` y copia bloques sueltos. Un comando por
bloque. Lanzar siempre con `--nodelist=c2` (c1 y c3 tienen `/scratch` degradado) y con
`--mem=` explícito (el cluster asigna 2 GB por defecto).

**Idioma:** todo en español — documento, código, comentarios, commits.

## Datos que se confunden fácil

| | Notas | Archivos cifrados |
|---|---|---|
| Familias | **30** | **29** hasta el 2026-08-15; **30** desde que se subió BLACKBASTA |
| Azar | 0,033 | 0,034 con 29 |
| Corpus | 146 notas, 95 contenidos distintos | NapierOne |

Las corridas del clúster anteriores al 2026-08-15 son sobre **144** notas y **29** familias de
archivos: dos notas quedaron en cuarentena de Windows Defender y faltaba BLACKBASTA. Al citar
una cifra, decir sobre qué base se midió.

**Lemmou et al. (2021):** sí identifican familia, pero por **reglas y marcadores** + LSA como
búsqueda de casi-duplicados (181/182, mundo cerrado, sin train/test). Su **F = 0,920 es la
tarea binaria** de nombre de nota frente a nombre benigno, **no** clasificación de familia por
nombre. Nunca escribir lo contrario: el tutor conoce el paper.

**El 71,93 % de ID Ransomware** son 41 aciertos sobre **57 notas de 22 familias**
(`7_compartido_carlos/Tesis Carlos y Romina/Pruebas.xlsx`), **no** sobre el corpus de la tesis.

## Identificación

Tesis de grado de **Romina Alfonzo** y **Carlos Urdapilleta**, FP-UNA.
Tutor: Prof. **Cristian Cappo**. Título: *Detección de familias de ransomware en base a
archivos encriptados y notas de rescate*.
Obligación del reglamento del NIDTEC: agradecer el clúster (proyecto LABO16-167,
PROCIENCIA/CONACYT) en la tesis.
