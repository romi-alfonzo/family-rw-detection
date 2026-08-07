# Instrucciones para el servidor de la facultad

_Preparado: 2026-07-28. Dos trabajos: (A) hiperparámetros del clasificador de notas,
(B) features avanzadas de archivos cifrados para blindar el Experimento 2._

## Nota sobre GPU
La búsqueda de hiperparámetros usa **scikit-learn, que corre solo en CPU** — la GPU no
acelera este trabajo. Lo que sí aprovecha del servidor es la **cantidad de núcleos**
(`n_jobs=-1` paraleliza la grilla completa). La GPU quedaría para un experimento futuro
(p. ej. comparación con un modelo transformer, o cuML/RAPIDS si está instalado).

---

## Trabajo A — Hiperparámetros del clasificador de notas

### Archivos a copiar al servidor (los tres .py en la misma carpeta)
```
2_codigo/gridsearch_notas.py
2_codigo/clasificador_notas_v2.py
2_codigo/extractor_notas.py
3_datos/corpus_v2/            (carpeta completa, 146 notas)
```
⚠️ En el servidor, poner los tres `.py` juntos y el corpus donde se quiera, indicándolo con
`CORPUS_DIR=/ruta/a/corpus_v2` (los scripts esperan la estructura `Tesis/2_codigo` + `Tesis/3_datos`
solo como valor por defecto).

### Dependencias
```bash
pip install scikit-learn pandas beautifulsoup4
```
(Verificado con scikit-learn 1.6.1 / Python 3.11.)

### Prueba rápida (~2 min) — verificar que todo está bien antes de la corrida larga
```bash
python gridsearch_notas.py --smoke
```
Debe terminar con "BÚSQUEDA COMPLETADA" y crear `resultados_gridsearch/` con
`score_interno` sin NaN en `gridsearch_hiperparams.csv`.

### Corrida completa (dejar corriendo con nohup)
```bash
nohup python gridsearch_notas.py > gridsearch.log 2>&1 &
tail -f gridsearch.log     # para mirar el avance
```
- Qué hace: búsqueda **anidada** (GridSearch dentro del fold de entrenamiento,
  evaluación en el fold externo — sin sesgo de selección), con los dos protocolos
  P1/P2, 10 semillas, scoring macro-F1. Grilla: ~360 configuraciones × 2 vistas
  × 2 modelos. Al final evalúa un "combinado" con los mejores hiperparámetros.
- Duración estimada: varias horas (depende de los núcleos; en 32 núcleos, ~2-6 h).
- Hay **checkpoint tras cada bloque**: si se corta, los CSV parciales quedan.

### Qué traer de vuelta
```
resultados_gridsearch/gridsearch_resumen.csv       ← métricas externas (la tabla para la tesis)
resultados_gridsearch/gridsearch_hiperparams.csv   ← qué hiperparámetros ganaron en cada fold
resultados_gridsearch/gridsearch_manifiesto.json   ← trazabilidad
```
Cómo leerlos: comparar `f1_macro_mean` contra la corrida canónica sin tuning
(P1 0,760 / P2 0,435). La configuración recomendada para la tesis = la moda de
hiperparámetros en `gridsearch_hiperparams.csv` (el script ya la usa para el combinado).

---

## Trabajo B — Features avanzadas de archivos cifrados (Experimento 2)

**Objetivo:** demostrar que la indistinguibilidad multiclase persiste incluso con
275 features (hoy el 9,9 % se apoya en solo 2 features — flanco débil ante un revisor).

### Archivos
```
family-rw-detection/advanced_features.py    (extractor de 275 features, ya corregido el chi²)
family-rw-detection/train_advanced.py       (evaluación 5 subconjuntos × RF/KNN/GB, SKF 5-fold)
```

### Requisito: dataset NapierOne en el servidor
Los scripts esperan la estructura `Pruebas2/<FAMILIA>-tiny/` con los archivos cifrados
por familia (la versión tiny/extra-small de NapierOne, ~50 archivos por familia).
Si el dataset no está en el servidor, descargarlo de https://napierone.com/
(citar como Davies, Macfarlane & Buchanan 2022).

### Corrida
```bash
pip install scikit-learn pandas numpy
python advanced_features.py     # genera advanced_features.csv (tarda: lee todos los archivos)
nohup python train_advanced.py > train_advanced.log 2>&1 &
```

### Qué traer de vuelta
`advanced_features.csv` + los resultados/log de `train_advanced.py`.

**⚠️ NO usar** los scripts `*_iterations.py` viejos (eligen test_size mirando el test =
sesgo optimista) ni `helpers.calculate_chi_square` (bug: divide por bytes observados
en vez de 256; `advanced_features.py` ya lo hace bien).

---

## Trabajo C — Clasificación estructural (magic bytes / metadatos) — Experimento 2b

**Objetivo:** demostrar que aunque la estadística no discrimina familias (Trabajo B),
los artefactos ESTRUCTURALES deliberados sí lo hacen para un subconjunto — el pedido
del tutor del 11/01/25 y la validación automática de las firmas anotadas en Pruebas.xlsx.

### Archivo
```
deteccion_estructural.py
```
(Probado localmente: descubre solo el magic number JPEG en archivos legítimos — funciona.)

### Corrida (usa el mismo dataset NapierOne del Trabajo B)
```bash
python deteccion_estructural.py /ruta/a/Pruebas2 --max-archivos 50
```
Rápido (solo lee 128 bytes por archivo). Hace dos cosas:
1. **Descubre** el prefijo/sufijo binario común y la extensión propia de cada familia.
2. **Clasifica** con leave-one-out usando esas marcas → exactitud multiclase comparable
   con el 9,9 % estadístico.

### Qué traer de vuelta
```
resultados_estructural/marcas_por_familia.csv
resultados_estructural/clasificacion_loo.csv
resultados_estructural/manifiesto.json
```
Validación esperada: las 9 familias con "SI*" en Pruebas.xlsx (GANDCRAB, LORENZ, MAZE,
MEDUSALOCKER, PHOBOS, RYUK, SODINOKIBI, TESLACRYPT, WANNACRY) deberían aparecer con
marca detectable (ej. WANNACRY: prefijo `57414e4143525921` = "WANACRY!").

---

## Al volver del servidor
1. Copiar `resultados_gridsearch/` (y lo del Trabajo B) a `C:\Users\Romina\Tesis\`.
2. En el próximo chat decir: *"lee ESTADO_TESIS.md, volvieron los resultados del servidor"*
   → se agregan las tablas nuevas al capítulo 4 (sección de hiperparámetros) sin tocar lo demás.
