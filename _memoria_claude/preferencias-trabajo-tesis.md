---
name: preferencias-trabajo-tesis
description: "Cómo prefiere trabajar Romina en la tesis — solo agregar al documento, pulir al final; trabajo pesado va al servidor de la facultad"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: fda9b4f2-6e59-405a-904e-3a4e3baf9d69
  modified: 2026-08-04T22:55:48.985Z
---

Sobre la tesis de ransomware (ver [[tesis-ransomware-contexto]]):

- **"En el documento solo agreguemos; al final vamos a perfeccionar"** (2026-07-28).
  **Why:** prefiere avanzar sumando contenido antes que iterar puliendo secciones.
  **How to apply:** al trabajar el LaTeX, agregar secciones/tablas/figuras nuevas; NO
  reescribir por estilo ni perfeccionar redacción existente salvo cifras incorrectas.
  El pulido fino (front matter, resumen, conclusión, estilo) se hace en una pasada final.

- El **servidor de la facultad** es el lugar para el cómputo pesado (hiperparámetros,
  features avanzadas). Tiene muchos núcleos y GPU (la GPU no sirve para sklearn — solo
  para un eventual experimento con transformers). Los trabajos preparados están en
  `C:\Users\Romina\Tesis\SERVIDOR_INSTRUCCIONES.md` y `gridsearch_notas.py`.

- **"La conclusión se hace SOLAMENTE al final de la tesis"** (2026-08-04).
  **Why:** convención académica que Romina sigue estrictamente: la conclusión se redacta
  cuando todos los resultados están cerrados, no antes.
  **How to apply:** NUNCA proponer editar/reescribir `conclusion.tex` como tarea intermedia,
  ni siquiera para corregir cifras — eso va en el bloque de cierre final (Bloque E de la
  hoja de ruta en ESTADO_TESIS.md). La versión actual tiene cifras superadas (100%, 15,4%)
  que se corrigen recién en esa pasada final.

- **«Cada vez que me des números, decime a qué se refiere»** (2026-08-19). Nunca escribir una
  cifra suelta: siempre con su métrica pegada.
  **Why:** en este proyecto conviven exactitud, exactitud balanceada, macro-F1, F1 por familia,
  cobertura y acierto-donde-aplica. Un «0,093» o un «98 %» sin etiqueta no le dice nada, y ya
  causó confusión varias veces — entre otras, confundir el 57 % de cobertura con el 98 % de
  acierto donde la firma aplica, y llamar «macro-F1» al F1 de una familia sola.
  **How to apply:** decir «macro-F1 0,911», «F1 de BLACKBASTA 0,215», «cobertura 0,572»,
  «acierto donde la firma aplica 98,4 %». Y aclarar la base: sobre cuántas familias, cuántas
  muestras, qué protocolo (P1/P2), cuántas semillas. Ojo especialmente con **macro-F1**, que es
  el promedio sobre TODAS las familias: no existe «el macro-F1 de una familia».

- El tutor es **Cristian Cappo** (FP-UNA). No confundir con Pont (autor de una tesis
  doctoral de referencia).
