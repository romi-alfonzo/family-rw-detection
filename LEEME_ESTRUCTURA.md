# Estructura de la carpeta Tesis

_Reorganizada el 2026-08-04. **Nada fue borrado**: lo obsoleto está en `_archivo/`._

## Mapa

| Carpeta | Qué contiene | Notas |
|---|---|---|
| `1_documento/` | La tesis en LaTeX (`Plantilla_de_Tesis___Romina_Carlos/`) | Acá se compila: `pdflatex main` → `biber main` → `pdflatex main` ×2 |
| `2_codigo/` | Scripts Python + `family-rw-detection/` | Ver abajo qué hace cada uno |
| `3_datos/` | Corpus de notas, archivos cifrados y manifiestos | `corpus_v2/` = corpus canónico (146 notas / 30 familias) |
| `4_resultados/` | Salidas de todos los experimentos | Las tablas de la tesis se regeneran de acá |
| `5_bibliografia/` | PDFs de papers, por origen | `Leido/` = ya analizados |
| `6_notas_trabajo/` | Notas `.md` de fuentes y pendientes | Bitácora de recolección del corpus |
| `7_compartido_carlos/` | `Tesis Carlos y Romina/` intacta | Contiene `Pruebas.xlsx` (experimento de herramientas) y las actas de reuniones |
| `_archivo/` | Material superado, conservado por trazabilidad | No borrar sin revisar |

En la raíz quedan solo los tres documentos vivos: `ESTADO_TESIS.md` (estado y decisiones),
`DIAGNOSTICO_2026-07-27.md` (auditoría completa) y `SERVIDOR_INSTRUCCIONES.md` (trabajos para el servidor).

## `2_codigo/` — qué hace cada script

| Script | Función | Entrada → Salida |
|---|---|---|
| `clasificador_notas_v2.py` | **Corrida canónica** del clasificador NLP (protocolos P1/P2, macro-F1) | `3_datos/corpus_v2/` → `4_resultados/resultados_canonicos/` |
| `extractor_notas.py` | Módulo de extracción multiformato (detecta UTF-16/cp1252, HTML/HTA, PDF, OCR) | importado por los demás |
| `gridsearch_notas.py` | Búsqueda de hiperparámetros anidada (para el servidor) | `3_datos/corpus_v2/` → `4_resultados/resultados_gridsearch/` |
| `deteccion_estructural.py` | Experimento 2b: magic bytes / metadatos por familia | recibe la ruta del dataset como argumento → `4_resultados/resultados_estructural/` |
| `family-rw-detection/` | Análisis estadístico de archivos cifrados (`advanced_features.py`, `train_advanced.py`) | requiere NapierOne (ver `SERVIDOR_INSTRUCCIONES.md`) |

Los scripts resuelven sus rutas desde la raíz del proyecto, así que se ejecutan desde cualquier lugar:

```bash
python "C:/Users/Romina/Tesis/2_codigo/clasificador_notas_v2.py"
```

## `3_datos/`

- `corpus_v2/` — **canónico**: 146 notas, 30 familias, extensiones originales preservadas.
- `ransom_notes_corpus/` — corpus original (156 notas, 24 % duplicadas). Solo referencia histórica.
- `manifiesto_corpus_v2.csv` — procedencia de cada nota (fuente, tipo, extensión). Trazabilidad para citar.
- `fuentes_notas/` — repositorios crudos: `RansomNoteFiles/` (Lemmou), `ransomware_notes/` (ThreatLabz), `notas_pcrisk/`, `imagenes_notas/`.
- `archivos_cifrados/SVM/Pruebas/` — dataset binario: `Encr/` (cifrados) y `Legit/`.
  ⚠️ Los archivos de `Encr/` **parecen corruptos porque están cifrados a propósito**. No borrar.

## `4_resultados/`

- `resultados_canonicos/` — corrida vigente del Exp. 3 (resumen, por familia, matriz de confusión, manifiesto).
- `resultados_experimentos/` — Exp. 1 y 2 (binaria, multiclase, figuras).
- `resultados_estructural/` — Exp. 2b (marcas por familia, clasificación LOO).
- `resultados_gridsearch/` — hiperparámetros (se llena al volver del servidor).
- `_historico/` — métricas viejas superadas (`resultados_nlp*.csv`), origen del antiguo «87,58 %».

**Regla:** ninguna cifra de la tesis se tipea a mano; toda tabla se regenera desde estos CSV.

## `_archivo/` — qué hay y por qué se archivó

| Elemento | Motivo |
|---|---|
| `latex_capitulos/` | Borrador de abril nunca integrado. Su contenido ya se incorporó al capítulo 4. Su `referencias.bib` tiene ≥6 autorías incorrectas: **no reutilizar**. |
| `Plantilla inicial/` | Copia antigua del template. Contenía el `resultados.tex` perdido, ya rescatado. |
| `Plantilla_de_Tesis___Romina_Carlos.zip` | Backup viejo del template. |
| `backup_pre_cap4/` | Respaldo de `metodologia.tex` y `resultados.tex` previo a la reconstrucción del capítulo 4. |
| `clasificador_notas_ransomware.py` | Clasificador v1 (con fuga de vocabulario y bug de codificación). Se conserva como registro de la corrida histórica. |
| `tesis_avance_preview.pdf` | PDF de avance de mayo, superado por `1_documento/.../main.pdf`. |

## Mantener el orden

- Los auxiliares de LaTeX (`main.aux`, `.log`, `.bbl`, `.toc`, …) se regeneran en cada compilación; se pueden borrar sin riesgo.
- Hay ~6 MB en PDFs duplicados entre `5_bibliografia/Leido/` y `7_compartido_carlos/.../Papers/` (mismos papers en ambas ubicaciones). Se dejaron a propósito: cada carpeta tiene su lógica.
