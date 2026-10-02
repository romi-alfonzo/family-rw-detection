---
name: comparacion-id-ransomware
description: "Dónde está y qué mide de verdad la comparación con ID Ransomware y Crypto Sheriff: Pruebas.xlsx, hojas «Deteccion de notas» (41/57 = 71,93 %) y «Deteccion de archivos encriptad»"
metadata: 
  node_type: memory
  type: reference
  originSessionId: dbdaf708-7656-4b0a-906c-cef641b2edc3
  modified: 2026-08-13T16:26:22.981Z
---

Archivo: `7_compartido_carlos/Tesis Carlos y Romina/Pruebas.xlsx` (4 hojas: Resultados ·
Informacion sobre familias · Deteccion de archivos encriptad · Deteccion de notas).
Es el registro de las pruebas manuales contra las dos herramientas públicas.

**Notas — el famoso 71,93 %.** Es ID Ransomware sobre **57 notas de 22 familias**,
**41 aciertos**. Las notas se bajaron de tres repos públicos (threatlabz, kipziptie/gitlab
y el propio RansomNoteFiles de Lemmou) eligiendo las familias de las que hay archivos
cifrados. **No es el corpus de 146 notas / 30 familias de la tesis**: es un subconjunto
anterior y más chico. Al citarlo hay que decir «57 notas de 22 familias», no «el mismo
corpus». Fallos: WASTEDLOCKER 0/1, PHOBOS 0/1, DARKSIDE 1/3, RANSOMEXX 2/5, CERBER 3/6,
CUBA 1/2, TESLACRYPT 1/2, BLACKBASTA 2/4, BLACKMATTER 1/2, BLACKCAT 3/4. Perfectas 12,
entre ellas GANDCRAB 8/8 y CONTI 4/4.

**Archivos cifrados — 30 familias probadas.** Crypto Sheriff **5/30** (16,7 %); lo
descartaron por eso. ID Ransomware **20/30** (66,7 %) con el nombre original, pero solo
**9/30 (30 %)** siguen detectándose **al cambiarle el nombre al archivo** (marcados «SI*»
en la hoja). Esos 9 son GANDCRAB, LORENZ, MAZE, MEDUSALOCKER, PHOBOS, RYUK, SODINOKIBI,
TESLACRYPT y WANNACRY. Es la comparación fuerte del frente de archivos: el clasificador de
bytes de la tesis llega a exactitud 0,910 / macro-F1 0,908 en 29 familias **sin usar nombre
ni extensión**. Ojo con la métrica: lo de ID Ransomware es cobertura por familia (sí/no),
no exactitud por archivo — hay que enunciarlo así.

La hoja guarda además los `sample_bytes` que ID Ransomware reporta, y corroboran de forma
independiente el Experimento 2b: WANNACRY `[0x00-0x08] 0x57414E4143525921` («WANACRY!»),
RYUK `0x4845524D4553` («HERMES»), TESLACRYPT `[0x00-0x30]`, LORENZ `0x2E737A3430`.

Relacionado: [[tesis-ransomware-contexto]], [[lemmou-que-hace-y-que-no]]
