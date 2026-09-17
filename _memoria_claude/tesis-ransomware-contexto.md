---
name: tesis-ransomware-contexto
description: "Contexto central de la tesis de Romina (FP-UNA) — clasificación de familias de ransomware por notas de rescate; estado, reglas, resultados canónicos y pendientes"
metadata: 
  node_type: memory
  type: project
  originSessionId: fda9b4f2-6e59-405a-904e-3a4e3baf9d69
  modified: 2026-08-04T22:43:35.385Z
---

Tesis de grado: "Detección de familias de ransomware en base a archivos encriptados y notas de rescate" — Romina Alfonzo y Carlos Urdapilleta, tutor Cristian Cappo (FP-UNA). En español.

**AL INICIAR SESIÓN, leer en este orden (carpeta `C:\Users\Romina\Tesis\`):**
1. `LEEME_ESTRUCTURA.md` — mapa de carpetas (reorganizada el 2026-08-04).
2. `ESTADO_TESIS.md` — documento vivo, fuente de verdad; actualizarlo al cerrar cada sesión.
3. `DIAGNOSTICO_2026-07-27.md` — auditoría completa (redacción, código, fuentes) + plan E1-E15.

**Estructura (2026-08-04):** `1_documento/` tesis LaTeX · `2_codigo/` scripts · `3_datos/`
(corpus_v2, fuentes_notas, archivos_cifrados, manifiestos) · `4_resultados/` salidas ·
`5_bibliografia/` papers · `6_notas_trabajo/` .md · `7_compartido_carlos/` (Pruebas.xlsx,
actas de reuniones) · `_archivo/` superado (clasificador v1, latex_capitulos, Plantilla inicial).
Los scripts resuelven rutas desde la raíz y son portables (caen a ./corpus_v2 si no hay estructura).
⚠️ Los archivos de `3_datos/archivos_cifrados/SVM/Pruebas/Encr/` PARECEN corruptos porque están
cifrados a propósito: son el dataset del Exp. 1, NO borrar.

**Reglas duras:**
- SOLO 30 familias; expansión solo en profundidad (más notas por familia). Todo dato con fuente citable.
- Reportar SIEMPRE accuracy + balanced accuracy + macro-F1 (nunca solo weighted).
- Ninguna cifra tipeada a mano: las tablas se regeneran desde `resultados_canonicos\*.csv`.

**Estado al 2026-07-27 (todo detallado en ESTADO_TESIS.md §6.ter):**
- Script canónico: `clasificador_notas_v2.py` (v1 = solo registro histórico). Corre sobre `corpus_v2` (146 notas / 30 familias). Encoding UTF-16 ya arreglado en `extractor_notas.py`.
- Resultados canónicos: P1 "plantilla conocida" macro-F1 0,760 (char+LinearSVC); P2 "variante nunca vista" macro-F1 0,435 (StratifiedGroupKFold sobre grupos de casi-duplicados). Hallazgo: las notas son plantillas — 146 notas = 95 contenidos distintos.
- Cifras 16,67/66,67/71,93% (CryptoSheriff / ID Ransomware archivos / ID Ransomware notas) = experimentos PROPIOS en `Tesis Carlos y Romina\Pruebas.xlsx` — redactar como experimento propio, no citar como bibliografía.
- NapierOne = Davies et al. 2022 (no Pont). Benchmark de notas = Lemmou et al. 2021. La tesis de Pont NO clasifica notas (sirve para la parte estadística de archivos).
- OJO novedad (corregido 2026-07-28): Lemmou SÍ identifica familia por nota — con reglas/marcadores (emails, BTC, onion, keywords) + LSA casi-duplicados, 181/182 en mundo cerrado, ML solo binario de nombres. NUNCA afirmar "nadie clasificó familias por notas". Novedad real de la tesis: primer clasificador ML SUPERVISADO multiclase por contenido, con generalización medida (P1/P2) y macro-F1. Detalle en ESTADO_TESIS.md.
- Pendiente inmediato: rescatar `resultados.tex` (203 líneas) de `Plantilla inicial\images\Plantilla_de_Tesis___Romina_Carlos\` y armar el capítulo 4 del documento vivo (`Plantilla_de_Tesis___Romina_Carlos\resultados.tex`, hoy vacío); los resultados viejos están mal ubicados en metodología §3.5-3.6.
- El .bib bueno es `Plantilla_de_Tesis___Romina_Carlos\bibliography.bib`; el `latex_capitulos\referencias.bib` tiene ≥6 autorías incorrectas — NO migrar. Entrada `lee2022` corrupta (citada 5 veces).
