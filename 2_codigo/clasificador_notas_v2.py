#!/usr/bin/env python3
"""
clasificador_notas_v2.py — Corrida canónica del clasificador NLP de notas de rescate.
Tesis: "Detección de familias de ransomware en base a archivos encriptados y notas de rescate"
Autores: Romina Alfonzo, Carlos Urdapilleta — Tutor: Cristian Cappo (FP-UNA)

Correcciones metodológicas respecto de clasificador_notas_ransomware.py (v1),
según auditoría del 2026-07-27 (ver DIAGNOSTICO_2026-07-27.md):

  C1. DECODIFICACIÓN: usa extractor_notas.extraer_texto(), que detecta UTF-16/cp1252.
      En v1, 13/146 notas de corpus_v2 entraban como texto ilegible (bytes NUL).

  C2. SIN FUGA DE VOCABULARIO: el TfidfVectorizer vive DENTRO de un Pipeline, por lo
      que el vocabulario y el IDF se ajustan solo con el fold de entrenamiento.
      En v1 se hacía fit_transform sobre TODO el corpus antes de la validación cruzada,
      de modo que el vectorizador "veía" el conjunto de prueba.

  C3. CASI-DUPLICADOS CONTROLADOS: las notas con similitud coseno de caracteres > 0.90
      se agrupan (componentes conexas) y StratifiedGroupKFold garantiza que un grupo
      nunca quede repartido entre entrenamiento y prueba. En v1, versiones casi
      idénticas de la misma nota podían caer una en train y otra en test, inflando
      las métricas. Se reporta también el protocolo sin grupos para cuantificar el efecto.

  C4. REPETICIÓN: la validación cruzada se repite con 10 semillas distintas y se
      reporta media ± desvío entre semillas. En v1 había una sola partición (una
      semilla), por lo que el "±" no reflejaba la variabilidad real.

  C5. MÉTRICAS PARA MULTICLASE DESBALANCEADA: se reportan accuracy, balanced accuracy,
      macro-F1 y weighted-F1, más el reporte por familia y la matriz de confusión.
      En v1 solo se calculaban métricas weighted, que las familias grandes dominan.

Uso:
    python clasificador_notas_v2.py            # usa .\corpus_v2
    CORPUS_DIR=... python clasificador_notas_v2.py
Salidas en .\resultados_canonicos\
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from extractor_notas import extraer_texto

# ============================================================
# CONFIGURACIÓN
# ============================================================
# Rutas relativas a la raíz del proyecto (este script vive en Tesis/2_codigo/).
# Portable: si no existe la estructura del proyecto (p. ej. en el servidor, con los
# archivos sueltos), cae a ./corpus_v2 y ./resultados_canonicos junto al script.
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
_estructura = (RAIZ / "3_datos").is_dir()

if "CORPUS_DIR" in os.environ:
    CORPUS_DIR = Path(os.environ["CORPUS_DIR"])
elif _estructura:
    CORPUS_DIR = RAIZ / "3_datos" / "corpus_v2"
else:
    CORPUS_DIR = AQUI / "corpus_v2"

OUT_DIR = (RAIZ / "4_resultados" / "resultados_canonicos" if _estructura
           else AQUI / "resultados_canonicos")

# Paralelismo: en un cluster con SLURM hay que usar SOLO los núcleos asignados al job,
# no todos los del nodo (n_jobs=N_JOBS los tomaría todos y sobresuscribiría el nodo).
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or -1

N_FOLDS = 2          # limitado por las familias con 2 notas (honesto: se declara)
N_SEMILLAS = 10      # repeticiones de la CV con distintas particiones
UMBRAL_NEARDUP = 0.90  # similitud coseno char(3-5) para considerar casi-duplicado
MIN_CHARS_NOTA = 10

TFIDF_WORD = dict(analyzer="word", ngram_range=(1, 2), max_features=5000,
                  sublinear_tf=True, min_df=1, strip_accents="unicode", lowercase=True)
TFIDF_CHAR = dict(analyzer="char_wb", ngram_range=(3, 5), max_features=5000,
                  sublinear_tf=True, min_df=1, strip_accents="unicode", lowercase=True)


def vectorizador(vista):
    """Vectorizador TF-IDF para cada vista. Se instancia NUEVO por pipeline (C2)."""
    if vista == "palabras":
        return TfidfVectorizer(**TFIDF_WORD)
    if vista == "caracteres":
        return TfidfVectorizer(**TFIDF_CHAR)
    if vista == "combinado":
        return FeatureUnion([
            ("word", TfidfVectorizer(**TFIDF_WORD)),
            ("char", TfidfVectorizer(**TFIDF_CHAR)),
        ])
    raise ValueError(vista)


def obtener_modelos(seed):
    return {
        "LinearSVC": LinearSVC(C=1.0, max_iter=10000, random_state=seed,
                               class_weight="balanced"),
        "Regresión Logística": LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs",
                                                  class_weight="balanced",
                                                  random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed,
                                                class_weight="balanced", n_jobs=N_JOBS),
        "KNN": KNeighborsClassifier(n_neighbors=3, metric="cosine", n_jobs=N_JOBS),
    }


# ============================================================
# CARGA DEL CORPUS (C1)
# ============================================================
def cargar_corpus(corpus_dir):
    textos, etiquetas, archivos, metodos = [], [], [], []
    if not corpus_dir.exists():
        sys.exit(f"ERROR: no existe el corpus: {corpus_dir}")
    for familia_dir in sorted(corpus_dir.iterdir()):
        if not familia_dir.is_dir():
            continue
        for nota in sorted(p for p in familia_dir.iterdir() if p.is_file()):
            texto, metodo = extraer_texto(nota)
            if metodo.startswith("error") or len(texto.strip()) < MIN_CHARS_NOTA:
                print(f"  ADVERTENCIA: {nota} omitida (metodo={metodo}, "
                      f"chars={len(texto.strip())})")
                continue
            textos.append(texto)
            etiquetas.append(familia_dir.name)
            archivos.append(str(nota.relative_to(corpus_dir)))
            metodos.append(metodo)
    return textos, np.array(etiquetas), archivos, metodos


# ============================================================
# AGRUPAMIENTO DE CASI-DUPLICADOS (C3)
# ============================================================
def agrupar_neardups(textos, umbral):
    """Agrupa notas con similitud coseno char(3-5) > umbral (componentes conexas).

    Devuelve un array de ids de grupo. Es un paso de preparación de datos NO
    supervisado (no usa las etiquetas ni participa del entrenamiento), análogo a
    la deduplicación: solo evita que la misma nota (o una variante trivial) esté
    a la vez en entrenamiento y prueba.
    """
    X = TfidfVectorizer(**TFIDF_CHAR).fit_transform(textos)
    sim = (X @ X.T).toarray()
    n = len(textos)
    padre = list(range(n))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    pares = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > umbral:
                pares.append((i, j, sim[i, j]))
                ri, rj = raiz(i), raiz(j)
                if ri != rj:
                    padre[rj] = ri
    grupos = np.array([raiz(i) for i in range(n)])
    return grupos, pares


# ============================================================
# EVALUACIÓN (C2, C3, C4, C5)
# ============================================================
def evaluar(textos, y, grupos, vista, nombre_modelo, protocolo, familias):
    """CV repetida con predicciones fuera-de-fold. Devuelve métricas por semilla,
    P/R/F1 por familia (promedio entre semillas) y la matriz de confusión agregada."""
    metricas_semilla = []
    prfs = []  # (precision[], recall[], f1[]) por semilla
    conf_total = np.zeros((len(familias), len(familias)), dtype=int)
    textos = np.array(textos, dtype=object)

    for seed in range(N_SEMILLAS):
        if protocolo == "grupos":
            cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            splits = cv.split(textos, y, groups=grupos)
        else:
            cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            splits = cv.split(textos, y)

        y_pred = np.empty_like(y)
        for tr, te in splits:
            pipe = Pipeline([("tfidf", vectorizador(vista)),
                             ("clf", obtener_modelos(seed)[nombre_modelo])])
            pipe.fit(textos[tr], y[tr])
            y_pred[te] = pipe.predict(textos[te])

        metricas_semilla.append(dict(
            accuracy=accuracy_score(y, y_pred),
            balanced_accuracy=balanced_accuracy_score(y, y_pred),
            f1_macro=f1_score(y, y_pred, average="macro", zero_division=0),
            f1_weighted=f1_score(y, y_pred, average="weighted", zero_division=0),
        ))
        p, r, f, _ = precision_recall_fscore_support(
            y, y_pred, labels=familias, zero_division=0)
        prfs.append((p, r, f))
        conf_total += confusion_matrix(y, y_pred, labels=familias)

    df = pd.DataFrame(metricas_semilla)
    resumen = {f"{c}_{s}": v for c in df.columns
               for s, v in (("mean", df[c].mean()), ("std", df[c].std()))}
    p = np.mean([x[0] for x in prfs], axis=0)
    r = np.mean([x[1] for x in prfs], axis=0)
    f = np.mean([x[2] for x in prfs], axis=0)
    return resumen, (p, r, f), conf_total


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser(
        description="Corrida canónica del clasificador de notas de rescate.")
    ap.add_argument("--salida", type=Path, default=None,
                    help="carpeta de salida (por defecto, 4_resultados/resultados_canonicos). "
                         "Usar una carpeta NUEVA para no pisar la corrida canónica.")
    args = ap.parse_args()
    if args.salida is not None:
        OUT_DIR = args.salida
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("  CORRIDA CANÓNICA — clasificador de notas v2")
    print("=" * 70)

    # ---- Carga (C1)
    print(f"\nCorpus: {CORPUS_DIR}")
    textos, y, archivos, metodos = cargar_corpus(CORPUS_DIR)
    familias = np.unique(y)
    conteo = Counter(y)
    print(f"Notas: {len(textos)} | Familias: {len(familias)}")
    print(f"Extracción: {dict(Counter(metodos))}")

    # ---- Casi-duplicados (C3)
    grupos, pares = agrupar_neardups(textos, UMBRAL_NEARDUP)
    n_grupos = len(set(grupos))
    print(f"\nCasi-duplicados (coseno char > {UMBRAL_NEARDUP}): "
          f"{len(pares)} pares -> {n_grupos} grupos para {len(textos)} notas")

    pd.DataFrame({
        "archivo": archivos, "familia": y, "grupo": grupos, "metodo_extraccion": metodos,
    }).to_csv(OUT_DIR / "grupos_neardup.csv", index=False)

    # Familias cuyo contenido distinto es 1 solo grupo: no evaluables con grupos
    grupos_por_familia = {fam: len({g for g, ff in zip(grupos, y) if ff == fam})
                          for fam in familias}
    monogrupo = [f for f, k in grupos_por_familia.items() if k < 2]
    if monogrupo:
        print(f"ADVERTENCIA: familias con un único grupo de contenido (con el "
              f"protocolo 'grupos' nunca aparecen en train y test a la vez, "
              f"su F1 tenderá a 0): {monogrupo}")

    # ---- Evaluación (C2..C5)
    filas_resumen = []
    mejor = None  # (f1_macro, protocolo, vista, modelo, porfam, conf)
    for protocolo in ("grupos", "estratificado"):
        for vista in ("palabras", "caracteres", "combinado"):
            for nombre_modelo in obtener_modelos(0):
                print(f"\n[{protocolo} | {vista} | {nombre_modelo}] ...", end="", flush=True)
                resumen, porfam, conf = evaluar(
                    textos, y, grupos, vista, nombre_modelo, protocolo, familias)
                print(f" macro-F1 {resumen['f1_macro_mean']:.3f} "
                      f"± {resumen['f1_macro_std']:.3f} | "
                      f"bal.acc {resumen['balanced_accuracy_mean']:.3f} | "
                      f"acc {resumen['accuracy_mean']:.3f}")
                filas_resumen.append(dict(protocolo=protocolo, vista=vista,
                                          modelo=nombre_modelo, **resumen))
                if protocolo == "grupos" and (
                        mejor is None or resumen["f1_macro_mean"] > mejor[0]):
                    mejor = (resumen["f1_macro_mean"], protocolo, vista,
                             nombre_modelo, porfam, conf)

    df_resumen = pd.DataFrame(filas_resumen)
    df_resumen.to_csv(OUT_DIR / "corrida_canonica_resumen.csv", index=False)

    # ---- Reporte por familia de la mejor configuración honesta
    _, protocolo, vista, nombre_modelo, (p, r, f), conf = mejor
    df_fam = pd.DataFrame({
        "familia": familias,
        "n_notas": [conteo[x] for x in familias],
        "n_grupos_contenido": [grupos_por_familia[x] for x in familias],
        "precision": p.round(3), "recall": r.round(3), "f1": f.round(3),
    }).sort_values("f1")
    df_fam.to_csv(OUT_DIR / "corrida_canonica_por_familia.csv", index=False)

    print("\n" + "=" * 70)
    print(f"  MEJOR CONFIGURACIÓN HONESTA: {vista} + {nombre_modelo} "
          f"(protocolo {protocolo})")
    print("=" * 70)
    print(df_fam.to_string(index=False))

    # ---- Matriz de confusión (agregada sobre las 10 semillas)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        conf_norm = conf / conf.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(13, 11))
        im = ax.imshow(conf_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(familias)), familias, rotation=90, fontsize=7)
        ax.set_yticks(range(len(familias)), familias, fontsize=7)
        ax.set_xlabel("Familia predicha")
        ax.set_ylabel("Familia real")
        ax.set_title(f"Matriz de confusión normalizada — {vista} + {nombre_modelo}\n"
                     f"(agregada sobre {N_SEMILLAS} semillas, protocolo {protocolo})")
        fig.colorbar(im, shrink=0.8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig_confusion_canonica.png", dpi=200)
        print(f"\nFigura: {OUT_DIR / 'fig_confusion_canonica.png'}")
    except Exception as e:
        print(f"(figura omitida: {e})")

    # ---- Manifiesto de la corrida (trazabilidad total)
    manifiesto = dict(
        fecha=str(date.today()),
        corpus=str(CORPUS_DIR),
        n_notas=len(textos),
        n_familias=int(len(familias)),
        n_grupos_neardup=int(n_grupos),
        pares_neardup=len(pares),
        familias_monogrupo=monogrupo,
        n_folds=N_FOLDS,
        n_semillas=N_SEMILLAS,
        umbral_neardup=UMBRAL_NEARDUP,
        tfidf_word=TFIDF_WORD, tfidf_char=TFIDF_CHAR,
        sklearn=sklearn.__version__,
        python=sys.version.split()[0],
        mejor_config=dict(protocolo=protocolo, vista=vista, modelo=nombre_modelo),
    )
    (OUT_DIR / "manifiesto_corrida.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifiesto: {OUT_DIR / 'manifiesto_corrida.json'}")
    print("\nCORRIDA CANÓNICA COMPLETADA")


if __name__ == "__main__":
    main()
