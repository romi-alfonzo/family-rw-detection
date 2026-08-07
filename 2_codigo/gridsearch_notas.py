#!/usr/bin/env python3
"""
gridsearch_notas.py — Búsqueda de hiperparámetros HONESTA (anidada) para el
clasificador de notas de rescate. Pensado para el servidor de la facultad.

Diseño:
  - Evaluación ANIDADA: la búsqueda de hiperparámetros (GridSearchCV) ocurre
    DENTRO del fold de entrenamiento de cada partición externa. El fold de
    prueba externo nunca participa de la selección => la métrica reportada
    no está sesgada por la búsqueda (a diferencia de tunear sobre el test).
  - Protocolo externo idéntico al de clasificador_notas_v2.py: 2 folds
    repetidos con N semillas, bajo P1 (StratifiedKFold) y P2
    (StratifiedGroupKFold con grupos de casi-duplicados).
  - Métrica de selección y reporte principal: macro-F1.
  - Se tunean LinearSVC y Regresión Logística (los ganadores claros de la
    corrida canónica) sobre las vistas 'palabras' y 'caracteres'. Al final se
    evalúa una configuración 'combinado' construida con los mejores
    hiperparámetros encontrados por vista.
  - Paraleliza con todos los núcleos (n_jobs=-1). NO usa GPU: scikit-learn es
    CPU; en el servidor lo que aprovechamos son los núcleos.

Uso:
    python gridsearch_notas.py                  # corrida completa (servidor)
    python gridsearch_notas.py --smoke          # prueba rápida (~2 min, local)
    python gridsearch_notas.py --semillas 5     # menos repeticiones
    CORPUS_DIR=/ruta/corpus_v2 python gridsearch_notas.py

Salidas en .\resultados_gridsearch\:
    gridsearch_resumen.csv        métricas externas por (protocolo, vista, modelo)
    gridsearch_hiperparams.csv    hiperparámetros elegidos en cada fold externo
    gridsearch_manifiesto.json    parámetros de la corrida (trazabilidad)
"""

import argparse
import json
import sys
import warnings
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

# Reutiliza carga y agrupamiento de la corrida canónica (misma base de datos)
from clasificador_notas_v2 import (CORPUS_DIR, N_JOBS, TFIDF_CHAR, TFIDF_WORD,
                                   UMBRAL_NEARDUP, agrupar_neardups,
                                   cargar_corpus)

warnings.filterwarnings("ignore", message="The least populated class")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

_AQUI = Path(__file__).resolve().parent
OUT_DIR = (_AQUI.parent / "4_resultados" / "resultados_gridsearch"
           if (_AQUI.parent / "4_resultados").is_dir()
           else _AQUI / "resultados_gridsearch")
# Scorer nativo de sklearn: compatible con etiquetas de texto en GridSearchCV
SCORER = "f1_macro"


# ----------------------------------------------------------------------------
# Grillas de búsqueda
# ----------------------------------------------------------------------------
def grilla(vista, modelo, smoke=False):
    if smoke:
        tfidf = {"tfidf__min_df": [1]}
        clf = {"clf__C": [0.1, 1.0, 10.0]}
        return {**tfidf, **clf}

    if vista == "palabras":
        tfidf = {
            "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
            "tfidf__min_df": [1, 2],
            "tfidf__max_df": [1.0, 0.9],
            # Sin tope (None) el vocabulario de char n-gramas sobre las notas más
            # largas (hasta 67 000 caracteres) supera el millón de términos; el
            # coef_ denso de 30 clases × 1 M features por cada worker paralelo
            # agota la memoria del nodo (job 3539 murió por OOM el 2026-08-04).
            # Además, con 144 documentos un vocabulario ilimitado es sobreajuste.
            "tfidf__max_features": [5000, 10000, 20000],
            "tfidf__sublinear_tf": [True, False],
        }
    else:  # caracteres
        tfidf = {
            "tfidf__ngram_range": [(2, 4), (3, 5), (2, 5), (3, 6)],
            "tfidf__min_df": [1, 2],
            "tfidf__max_df": [1.0, 0.9],
            # Sin tope (None) el vocabulario de char n-gramas sobre las notas más
            # largas (hasta 67 000 caracteres) supera el millón de términos; el
            # coef_ denso de 30 clases × 1 M features por cada worker paralelo
            # agota la memoria del nodo (job 3539 murió por OOM el 2026-08-04).
            # Además, con 144 documentos un vocabulario ilimitado es sobreajuste.
            "tfidf__max_features": [5000, 10000, 20000],
            "tfidf__sublinear_tf": [True, False],
        }
    clf = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
    return {**tfidf, **clf}


def pipeline_base(vista, modelo, seed):
    params = TFIDF_WORD if vista == "palabras" else TFIDF_CHAR
    clf = (LinearSVC(max_iter=10000, random_state=seed, class_weight="balanced")
           if modelo == "LinearSVC" else
           LogisticRegression(max_iter=5000, solver="lbfgs", random_state=seed,
                              class_weight="balanced"))
    return Pipeline([("tfidf", TfidfVectorizer(**params)), ("clf", clf)])


# ----------------------------------------------------------------------------
# Búsqueda anidada
# ----------------------------------------------------------------------------
def busqueda_anidada(textos, y, grupos, protocolo, vista, modelo,
                     n_semillas, n_folds, smoke, log):
    """Por cada semilla y fold externo: GridSearch en train, evaluación en test."""
    textos = np.array(textos, dtype=object)
    familias = np.unique(y)
    metricas, elecciones = [], []

    for seed in range(n_semillas):
        if protocolo == "grupos":
            cv_ext = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                          random_state=seed)
            splits = cv_ext.split(textos, y, groups=grupos)
        else:
            cv_ext = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                     random_state=seed)
            splits = cv_ext.split(textos, y)

        y_pred = np.empty_like(y)
        for k, (tr, te) in enumerate(splits):
            if protocolo == "grupos":
                cv_int = StratifiedGroupKFold(n_splits=2, shuffle=True,
                                              random_state=seed)
                fit_kw = {"groups": grupos[tr]}
            else:
                cv_int = StratifiedKFold(n_splits=2, shuffle=True,
                                         random_state=seed)
                fit_kw = {}

            gs = GridSearchCV(
                pipeline_base(vista, modelo, seed),
                grilla(vista, modelo, smoke),
                scoring=SCORER, cv=cv_int, n_jobs=N_JOBS, refit=True,
            )
            gs.fit(textos[tr], y[tr], **fit_kw)
            y_pred[te] = gs.predict(textos[te])
            elecciones.append(dict(protocolo=protocolo, vista=vista,
                                   modelo=modelo, semilla=seed, fold=k,
                                   score_interno=round(gs.best_score_, 4),
                                   **{p: str(v) for p, v in gs.best_params_.items()}))

        metricas.append(dict(
            accuracy=accuracy_score(y, y_pred),
            balanced_accuracy=balanced_accuracy_score(y, y_pred),
            f1_macro=f1_score(y, y_pred, average="macro", zero_division=0),
            f1_weighted=f1_score(y, y_pred, average="weighted", zero_division=0),
        ))
        log(f"    semilla {seed}: macro-F1 {metricas[-1]['f1_macro']:.3f}")

    df = pd.DataFrame(metricas)
    resumen = {f"{c}_{s}": v for c in df.columns
               for s, v in (("mean", df[c].mean()), ("std", df[c].std()))}
    return resumen, elecciones


def params_mas_elegidos(elecciones, vista, modelo):
    """Moda de cada hiperparámetro entre los folds externos (para el combinado
    final y para reportar en la tesis la configuración recomendada)."""
    sel = [e for e in elecciones if e["vista"] == vista and e["modelo"] == modelo]
    claves = [k for k in sel[0] if k.startswith(("tfidf__", "clf__"))]
    return {k: Counter(e[k] for e in sel).most_common(1)[0][0] for k in claves}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="prueba rápida local: grilla mínima, 2 semillas")
    ap.add_argument("--semillas", type=int, default=10)
    ap.add_argument("--folds", type=int, default=2)
    args = ap.parse_args()
    n_semillas = 2 if args.smoke else args.semillas

    OUT_DIR.mkdir(exist_ok=True)
    log = lambda m: print(m, flush=True)

    log("=" * 70)
    log(f"  BÚSQUEDA DE HIPERPARÁMETROS ANIDADA {'(SMOKE)' if args.smoke else ''}")
    log("=" * 70)
    log(f"Corpus: {CORPUS_DIR}")
    textos, y, archivos, metodos = cargar_corpus(Path(CORPUS_DIR))
    grupos, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    grupos = np.array(grupos)
    log(f"Notas: {len(textos)} | Familias: {len(np.unique(y))} | "
        f"Grupos de contenido: {len(set(grupos))}")

    vistas = ["caracteres"] if args.smoke else ["palabras", "caracteres"]
    modelos = ["LinearSVC"] if args.smoke else ["LinearSVC", "Regresión Logística"]

    filas, todas_elecciones = [], []
    for protocolo in ("grupos", "estratificado"):
        for vista in vistas:
            for modelo in modelos:
                log(f"\n[{protocolo} | {vista} | {modelo}]")
                resumen, elecciones = busqueda_anidada(
                    textos, y, grupos, protocolo, vista, modelo,
                    n_semillas, args.folds, args.smoke, log)
                filas.append(dict(protocolo=protocolo, vista=vista,
                                  modelo=modelo, **resumen))
                todas_elecciones += elecciones
                log(f"  => macro-F1 {resumen['f1_macro_mean']:.3f} "
                    f"± {resumen['f1_macro_std']:.3f} | "
                    f"acc {resumen['accuracy_mean']:.3f}")
                # checkpoint tras cada bloque (corridas largas en servidor)
                pd.DataFrame(filas).to_csv(OUT_DIR / "gridsearch_resumen.csv",
                                           index=False)
                pd.DataFrame(todas_elecciones).to_csv(
                    OUT_DIR / "gridsearch_hiperparams.csv", index=False)

    # ---- Evaluación final 'combinado' con los hiperparámetros más elegidos ----
    if not args.smoke:
        log("\n[combinado con mejores hiperparámetros por vista]")
        for protocolo in ("grupos", "estratificado"):
            ew = params_mas_elegidos(
                [e for e in todas_elecciones if e["protocolo"] == protocolo],
                "palabras", "LinearSVC")
            ec = params_mas_elegidos(
                [e for e in todas_elecciones if e["protocolo"] == protocolo],
                "caracteres", "LinearSVC")

            def aplicar(base, elegidos):
                p = dict(base)
                for k, v in elegidos.items():
                    if k.startswith("tfidf__"):
                        p[k.split("__", 1)[1]] = eval(v) if v[0] in "([0123456789NTF" else v
                return p

            textos_arr = np.array(textos, dtype=object)
            metricas = []
            for seed in range(n_semillas):
                cv = (StratifiedGroupKFold(args.folds, shuffle=True, random_state=seed)
                      if protocolo == "grupos" else
                      StratifiedKFold(args.folds, shuffle=True, random_state=seed))
                splits = (cv.split(textos_arr, y, groups=grupos)
                          if protocolo == "grupos" else cv.split(textos_arr, y))
                y_pred = np.empty_like(y)
                C = float([e for e in todas_elecciones
                           if e["protocolo"] == protocolo][0].get("clf__C", 1.0))
                for tr, te in splits:
                    pipe = Pipeline([
                        ("tfidf", FeatureUnion([
                            ("word", TfidfVectorizer(**aplicar(TFIDF_WORD, ew))),
                            ("char", TfidfVectorizer(**aplicar(TFIDF_CHAR, ec))),
                        ])),
                        ("clf", LinearSVC(C=C, max_iter=10000, random_state=seed,
                                          class_weight="balanced")),
                    ])
                    pipe.fit(textos_arr[tr], y[tr])
                    y_pred[te] = pipe.predict(textos_arr[te])
                metricas.append(dict(
                    accuracy=accuracy_score(y, y_pred),
                    balanced_accuracy=balanced_accuracy_score(y, y_pred),
                    f1_macro=f1_score(y, y_pred, average="macro", zero_division=0),
                    f1_weighted=f1_score(y, y_pred, average="weighted", zero_division=0),
                ))
            df = pd.DataFrame(metricas)
            resumen = {f"{c}_{s}": v for c in df.columns
                       for s, v in (("mean", df[c].mean()), ("std", df[c].std()))}
            filas.append(dict(protocolo=protocolo, vista="combinado-tuned",
                              modelo="LinearSVC", **resumen))
            log(f"  [{protocolo}] macro-F1 {resumen['f1_macro_mean']:.3f} "
                f"± {resumen['f1_macro_std']:.3f}")
        pd.DataFrame(filas).to_csv(OUT_DIR / "gridsearch_resumen.csv", index=False)

    (OUT_DIR / "gridsearch_manifiesto.json").write_text(json.dumps(dict(
        fecha=str(date.today()), corpus=str(CORPUS_DIR), n_notas=len(textos),
        n_semillas=n_semillas, n_folds=args.folds, smoke=args.smoke,
        seleccion="GridSearchCV anidado, scoring=f1_macro (zero_division=0)",
        sklearn=sklearn.__version__, python=sys.version.split()[0],
    ), indent=2, ensure_ascii=False), encoding="utf-8")

    log(f"\nSalidas en: {OUT_DIR}")
    log("BÚSQUEDA COMPLETADA")


if __name__ == "__main__":
    main()
