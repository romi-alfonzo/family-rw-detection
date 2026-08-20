---
name: sin-coautoria-en-commits
description: Nunca agregar coautoría de Claude en los commits de git de Romina
metadata: 
  node_type: memory
  type: feedback
  originSessionId: dbdaf708-7656-4b0a-906c-cef641b2edc3
  modified: 2026-08-07T02:25:33.198Z
---

**Nunca agregar la línea `Co-Authored-By: Claude ...` (ni ninguna otra atribución) en los
commits de git.** Indicado por Romina el 2026-08-05.

**Why:** es su trabajo académico —una tesis de grado— y la autoría del repositorio debe ser
exclusivamente de ella y de su compañero Carlos Urdapilleta. Una coautoría automática en el
historial de git sería incorrecta en ese contexto.

**How to apply:** al hacer commits en cualquier repositorio suyo, terminar el mensaje sin
ningún pie de atribución. Esto reemplaza la convención por defecto del entorno, que pide
agregar esa línea. Aplica a todos sus repos, incluidos
`family-rw-detection` y el repositorio de la tesis.

**Ampliado el 2026-08-19 — commit y push automáticos SOLO de código:** cada vez que se
cambie código (scripts de `2_codigo/`, jobs de SLURM), **commitear a `develop` y pushear en
el momento**, sin que Romina lo pida. Mensaje breve, en español, sin coautoría.
**Corregido el mismo día por Romina («solo código se commitea, no todo lo del contexto»):**
los documentos de estado (`ESTADO_TESIS.md`, `PLAN_MEJORAS.md`, notas de trabajo, informes)
y el LaTeX se commitean únicamente cuando ella lo pide — se siguen ACTUALIZANDO en el
momento, solo que no se commitean solos. Nunca commitear datos del corpus ni resultados;
los `.bak` tampoco.

Ver [[tesis-ransomware-contexto]] y [[preferencias-trabajo-tesis]].
