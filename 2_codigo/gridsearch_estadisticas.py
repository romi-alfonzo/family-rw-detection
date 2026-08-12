#!/usr/bin/env python3
"""
gridsearch_estadisticas.py -- Búsqueda de hiperparámetros para el Experimento 2
(clasificación multiclase por características estadísticas de los archivos cifrados).

MOTIVO
Es el único de los tres experimentos que se evaluó con la configuración por defecto de
cada algoritmo. Está declarado como limitación en la tesis (§4.3.1). Este script cierra
esa objeción: si tras una búsqueda sistemática la exactitud sigue por debajo de la del
clasificador sobre bytes, la conclusión del Experimento 2 queda establecida sin reservas.

EXPECTATIVA HONESTA
La exactitud puede subir de 0,603 a algo del orden de 0,65-0,70. No alteraría las
conclusiones, porque el Experimento 2c alcanza 0,910 sobre los mismos archivos; el valor
de esta corrida es eliminar la objeción, no mejorar el resultado principal.

DISEÑO
Búsqueda ANIDADA: RandomizedSearchCV dentro del subconjunto de entrenamiento de cada
partición externa, evaluación sobre la partición externa que no participó de la selección.
Se evalúan los dos subconjuntos de características relevantes: las 19 seleccionadas por
criterio experto (el mejor con parámetros por defecto) y las 275 completas.

Uso:
    python3.11 gridsearch_estadisticas.py advanced_features.csv
    python3.11 gridsearch_estadisticas.py advanced_features.csv --smoke
Salidas en 4_resultados/resultados_gridsearch_estadisticas/
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# La consola de Windows usa cp1252; forzar UTF-8 evita fallos de codificacion
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or -1
_AQUI = Path(__file__).resolve().parent
OUT_DIR = (_AQUI.parent / "4_resultados" / "resultados_gridsearch_estadisticas"
           if (_AQUI.parent / "4_resultados").is_dir()
           else _AQUI / "resultados_gridsearch_estadisticas")

# Mismos subconjuntos que en train_advanced.py, para que las cifras sean comparables
SUBCONJUNTOS = {
    "estadisticas + derivadas (19)": list(range(9)) + list(range(265, 275)),
    "todas (275)": None,
}


def cargar(filepath):
    X, y = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for fila in r:
            X.append([float(v) for v in fila[:-1]])
            y.append(int(fila[-1]))
    return np.asarray(X, dtype=np.float32), np.asarray(y)


def espacios(smoke):
    if smoke:
        return {"RandomForest": (RandomForestClassifier(random_state=42, n_jobs=1),
                                 {"model__n_estimators": [50, 100]})}
    return {
        "RandomForest": (
            RandomForestClassifier(random_state=42, n_jobs=1),
            {"model__n_estimators": [100, 200, 400],
             "model__max_depth": [None, 10, 20, 40],
             "model__min_samples_leaf": [1, 2, 5],
             "model__max_features": ["sqrt", "log2", 0.3, 0.6],
             "model__criterion": ["gini", "entropy"]},
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(random_state=42),
            {"model__max_iter": [100, 200, 300],
             "model__learning_rate": [0.05, 0.1, 0.2],
             "model__max_leaf_nodes": [15, 31, 63],
             "model__min_samples_leaf": [10, 20, 50],
             "model__l2_regularization": [0.0, 0.5, 1.0]},
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="advanced_features.csv")
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--folds-externos", type=int, default=3)
    ap.add_argument("--folds-internos", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_iter = 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)

    log("=" * 74)
    log(f"  HIPERPARÁMETROS -- CARACTERÍSTICAS ESTADÍSTICAS{' [SMOKE]' if args.smoke else ''}")
    log("=" * 74)
    X, y = cargar(args.csv)
    clases = np.unique(y)
    azar = 1.0 / len(clases)
    log(f"Datos: {X.shape[0]} archivos · {X.shape[1]} características · "
        f"{len(clases)} clases (azar {azar:.3f})")
    log(f"Núcleos: {N_JOBS}")
    log(f"Referencia sin ajustar: 0,603 (19 características + Random Forest)\n")

    filas, elegidos = [], {}
    for nombre_sub, cols in SUBCONJUNTOS.items():
        Xs = X if cols is None else X[:, cols]
        for nombre_modelo, (modelo, espacio) in espacios(args.smoke).items():
            log(f"[{nombre_sub} | {nombre_modelo}]")
            t0 = time.time()
            cv_ext = StratifiedKFold(args.folds_externos, shuffle=True, random_state=42)
            yp = np.empty_like(y)
            for k, (tr, te) in enumerate(cv_ext.split(Xs, y)):
                busq = RandomizedSearchCV(
                    Pipeline([("scaler", StandardScaler()), ("model", modelo)]),
                    espacio, n_iter=args.n_iter, scoring="accuracy",
                    cv=StratifiedKFold(args.folds_internos, shuffle=True, random_state=k),
                    n_jobs=N_JOBS, random_state=42, refit=True)
                busq.fit(Xs[tr], y[tr])
                yp[te] = busq.predict(Xs[te])
                log(f"    fold {k+1}/{args.folds_externos}: interno {busq.best_score_:.3f}"
                    f" | {busq.best_params_}")
                if k == 0:
                    elegidos[f"{nombre_sub} | {nombre_modelo}"] = {
                        p: str(v) for p, v in busq.best_params_.items()}
            m = dict(subconjunto=nombre_sub, modelo=nombre_modelo,
                     accuracy=round(accuracy_score(y, yp), 4),
                     balanced_accuracy=round(balanced_accuracy_score(y, yp), 4),
                     f1_macro=round(f1_score(y, yp, average="macro", zero_division=0), 4),
                     segundos=round(time.time() - t0))
            filas.append(m)
            log(f"  => exactitud {m['accuracy']:.3f} | macro-F1 {m['f1_macro']:.3f}"
                f" | {m['segundos']}s\n")
            pd.DataFrame(filas).to_csv(
                OUT_DIR / "gridsearch_estadisticas_resumen.csv", index=False)

    mejor = max(filas, key=lambda r: r["accuracy"])
    log("=" * 74)
    log(f"  MEJOR: {mejor['subconjunto']} + {mejor['modelo']} -> {mejor['accuracy']:.3f}")
    log("=" * 74)
    delta = mejor["accuracy"] - 0.603
    log(f"  Sin ajustar: 0,603   ·   Ajustado: {mejor['accuracy']:.3f}   "
        f"·   Diferencia: {delta:+.3f}")
    log(f"  Comparación con el Exp. 2c (bytes, mismos archivos): 0,910")
    if mejor["accuracy"] < 0.85:
        log("  => Aun optimizado, el enfoque estadístico queda muy por debajo del")
        log("     clasificador sobre bytes. La conclusión del Experimento 2 se sostiene.")
    else:
        log("  => La optimización acerca el enfoque estadístico al de bytes: revisar la")
        log("     redacción del Experimento 2, que asume una brecha amplia.")

    (OUT_DIR / "gridsearch_estadisticas_manifiesto.json").write_text(json.dumps(dict(
        fecha=str(date.today()), csv_entrada=str(args.csv),
        n_archivos=int(X.shape[0]), n_caracteristicas=int(X.shape[1]),
        n_clases=int(len(clases)), azar=round(azar, 4),
        referencia_sin_ajustar=0.603, referencia_bytes_2c=0.910,
        n_iter=args.n_iter, mejor=mejor, hiperparametros=elegidos,
        sklearn=sklearn.__version__, python=sys.version.split()[0],
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\nSalidas en: {OUT_DIR}")


if __name__ == "__main__":
    main()
