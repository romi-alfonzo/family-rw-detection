---
name: proximo-experimento-2d
description: "Experimento 2d pendiente: cruzar bytes con nombre/extensión para las 4 familias críticas que solo marcan el nombre — es el elemento de acción 2 del tutor"
metadata: 
  node_type: memory
  type: project
  originSessionId: dbdaf708-7656-4b0a-906c-cef641b2edc3
  modified: 2026-08-19T02:32:10.736Z
---

Experimento acordado el 2026-08-18, todavía **sin correr**. Es textualmente el elemento de
acción 2 del tutor (acta del 2026-08-12): *«Para archivos encriptados: buscar qué patrón se
puede usar para identificar las familias críticas, si sirve cruzar con la detección por otra
característica del archivo, como extensión o datos del nombre del archivo.»*

**Por qué apunta al lugar exacto.** El cruce entre el Exp. 2b (tipo de marca) y el Exp. 2c
(macro-F1 por familia) muestra que de las 6 familias críticas, **4 tienen extensión propia y
el 2c no la mira** (no usa nombre ni extensión):

| Marca que deja (2b) | Familias | Con F1 ≥ 0,98 en el 2c |
|---|---|---|
| Firma binaria y extensión | 15 | 15 de 15 |
| Solo firma binaria | 2 | 1 de 2 (BADRABBIT 0,978) |
| **Solo extensión** | 11 | **7 de 11** |
| Sin marca | 2 | 0 de 2 |

Las 4 que fallan del grupo «solo extensión» son **JIGSAW, DARKSIDE, CRYPTOLOCKER y
WASTEDLOCKER**; más NOTPETYA y SUNCRYPT, que no dejan nada = las 6 difíciles.

**Diseño acordado: una corrida, tres columnas comparables.**
1. Solo bytes — es el 2c actual, macro-F1 0,911 ± 0,001. La referencia.
2. Bytes + **forma** del nombre (largo de la extensión, si es hexadecimal o pronunciable, si
   se conserva el nombre base, si incorpora email o ID). Es la contribución nueva y la
   defendible: son rasgos del herramental de la familia, que sobreviven a un cambio de
   campaña.
3. Bytes + extensión **literal**. Estimación previa: el macro-F1 subiría a ~0,97 (las 4
   familias pasarían de ~0,44-0,63 a ~1,00). **Se reporta como cota superior declarada, nunca
   como resultado principal**: la extensión identifica la *campaña*, no la familia, y el
   capítulo ya afirma que su rendimiento no es extrapolable. Escribirlo de otro modo sería
   incoherente con lo ya escrito.

Relacionado: [[tesis-ransomware-contexto]], [[cluster-nidtec]]
