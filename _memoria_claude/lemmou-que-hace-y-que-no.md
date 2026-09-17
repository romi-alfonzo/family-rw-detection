---
name: lemmou-que-hace-y-que-no
description: "Lemmou et al. 2021 SÍ identifica familia (por reglas/marcadores, no ML); su F=0,920 es la tarea BINARIA nombre-de-nota vs benigno, NO clasificación de familia por nombre"
metadata: 
  node_type: memory
  type: reference
  originSessionId: dbdaf708-7656-4b0a-906c-cef641b2edc3
  modified: 2026-08-12T17:32:04.185Z
---

Lemmou, Lanet y Souidi (2021), *In-Depth Analysis of Ransom Note Files*, Computers 10(11):145
— el benchmark directo de la tesis. Dos tareas distintas que NO hay que confundir:

1. **Identificación de familia** — prototipo basado en **reglas y marcadores** extraídos del
   contenido (emails, direcciones Bitcoin/Bitmessage, URLs onion, nombres de familia,
   keywords) + LSA como búsqueda de casi-duplicados (umbral 0,99995) contra su base de 176
   notas / 62 familias. Resultado 181/182, pero en **mundo cerrado, sin train/test**. No es ML.
2. **Binaria nombre-de-nota vs nombre benigno** — Random Forest sobre el nombre del archivo:
   **F 0,920 / exactitud 98,32 %**.

**El 0,920 es de la tarea 2, no de la 1.** Nunca escribir «Lemmou obtiene F = 0,920
clasificando por familia según el nombre»: es falso y Cappo conoce el paper.

Tampoco afirmar «nadie clasificó familias por nota» — sí lo hicieron. La novedad de la tesis
se formula así: primer clasificador supervisado multiclase **sobre el contenido completo**,
con medición explícita de generalización a variantes no vistas (P1/P2) y macro-F1;
complementario al identificador por marcadores de Lemmou, que exige una base curada de IOCs
actualizada permanentemente.

Nota de precaución para cualquier experimento que reutilice los marcadores de Lemmou: su
lista de keywords incluye **el nombre de la familia**. Si la nota dice «CONTI», acertar la
familia es circular y explica buena parte de su 181/182. Hay que excluir ese marcador o
reportarlo aparte.

Pendiente de corregir (al 2026-08-12): la frase equivocada sigue escrita en
`PLAN_MEJORAS.md` (Sprint 4.1) y en `plan_trabajo_cappo_2026-08-12.tex` (Etapa 2) —
Romina pidió no tocarlos por ahora. `ESTADO_TESIS.md` ya lo tiene correcto.

Conteos que no hay que cruzar: el frente de **notas** son **30 familias** (146 notas, 95
grupos de contenido) y el de **archivos cifrados** son **29 familias**. El azar es 1/30 en
notas y 1/29 en archivos.

Relacionado: [[tesis-ransomware-contexto]], [[preferencias-trabajo-tesis]]
