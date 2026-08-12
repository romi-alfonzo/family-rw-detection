#!/usr/bin/env python3
"""
clasificador_bytes.py -- Experimento 2c: clasificación de familias de ransomware
mediante APRENDIZAJE AUTOMÁTICO sobre los bytes de cabecera y cola de los archivos
cifrados, con búsqueda de hiperparámetros.

MOTIVACIÓN
El Experimento 2 mostró que las propiedades estadísticas del cifrado no discriminan
familias (~10 %, azar 3,4 %). El Experimento 2b mostró que los artefactos estructurales
sí lo hacen, pero con reglas de coincidencia exacta: la firma binaria identifica al 97 %
únicamente el 53 % de los archivos (el resto no tiene un prefijo/sufijo común a TODOS los
archivos de su familia, así que la regla no aplica).

Este experimento reemplaza la regla exacta por un clasificador entrenado. Ventajas:
  * COBERTURA 100 %: siempre produce una predicción, no depende de que exista una firma
    idéntica en todos los archivos de la familia.
  * Tolera variabilidad: aprende patrones parciales o desplazados que la regla exacta pierde.
  * SOLO CONTENIDO: no usa el nombre ni la extensión del archivo. Esto es deliberado --
    en el Exp. 2b la extensión aportaba el 82,8 % pero es un identificador de campaña
    (constante dentro de NapierOne, variable en la práctica). Acá se mide qué información
    hay en los BYTES.

REPRESENTACIONES COMPARADAS (análogas a las del clasificador de notas)
  * "posicional": los primeros H y últimos T bytes como valores numéricos. Captura marcas
    en posiciones fijas (p. ej. WannaCry escribe "WANACRY!" en el offset 0).
  * "ngramas": los mismos bytes tratados como una secuencia de símbolos, con TF-IDF de
    n-gramas de bytes. Captura marcas en posición variable. Es el equivalente directo de
    los n-gramas de caracteres que se usan con las notas de rescate.

METODOLOGÍA
Búsqueda ANIDADA: RandomizedSearchCV dentro del fold de entrenamiento, evaluación en el
fold externo que nunca participó de la selección. Se reportan exactitud, balanced accuracy
y macro-F1. Para acotar el costo, la búsqueda usa una submuestra estratificada y la mejor
configuración se re-evalúa sobre un conjunto mayor.

LIMITACIÓN A DECLARAR: NapierOne representa cada familia con una sola campaña, de modo que
un acierto alto mide identificación de esa campaña. La generalización a campañas nuevas de
la misma familia no puede evaluarse con este dataset. Es el mismo fenómeno que las
plantillas en las notas de rescate.

Uso:
    python3.11 clasificador_bytes.py /ruta/Napierone-small
    python3.11 clasificador_bytes.py /ruta --por-familia 200 --n-iter 12
    python3.11 clasificador_bytes.py /ruta --smoke
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or -1

_AQUI = Path(__file__).resolve().parent
OUT_DIR = (_AQUI.parent / "4_resultados" / "resultados_bytes"
           if (_AQUI.parent / "4_resultados").is_dir()
           else _AQUI / "resultados_bytes")

N_HEAD = 512   # bytes de cabecera
N_TAIL = 512   # bytes de cola


def leer_bytes(path, n_head=N_HEAD, n_tail=N_TAIL):
    """Devuelve (cabecera, cola) con relleno a longitud fija si el archivo es corto."""
    with open(path, "rb") as f:
        head = f.read(n_head)
        f.seek(0, 2)
        n = f.tell()
        f.seek(max(0, n - n_tail))
        tail = f.read(n_tail)
    return head.ljust(n_head, b"\x00"), tail.ljust(n_tail, b"\x00")


def cargar(raiz, por_familia, seed=42):
    """Lee hasta `por_familia` archivos de cada carpeta <FAMILIA>[-small|-tiny]."""
    rng = np.random.default_rng(seed)
    Xb, y, familias = [], [], []
    raiz = Path(raiz)
    for d in sorted(p for p in raiz.iterdir() if p.is_dir()):
        fam = d.name.upper()
        for suf in ("-SMALL", "_SMALL", "-TINY", "_TINY"):
            fam = fam.removesuffix(suf)
        archivos = sorted(p for p in d.iterdir() if p.is_file())
        # excluir archivos que no son muestras cifradas (p. ej. el PDF descriptivo)
        archivos = [p for p in archivos if p.suffix.lower() != ".pdf"]
        if len(archivos) < 6:
            print(f"  ADVERTENCIA: {fam} tiene {len(archivos)} archivos, omitida",
                  flush=True)
            continue
        sel = [archivos[i] for i in rng.permutation(len(archivos))[:por_familia]]
        for p in sel:
            h, t = leer_bytes(p)
            Xb.append(h + t)
            y.append(fam)
        familias.append(fam)
        print(f"  {fam:<15} {len(sel):>4} archivos", flush=True)
    return Xb, np.array(y), sorted(set(familias))


def a_matriz_posicional(Xb):
    """Bytes como valores numéricos: (n, N_HEAD+N_TAIL)."""
    return np.frombuffer(b"".join(Xb), dtype=np.uint8).reshape(
        len(Xb), N_HEAD + N_TAIL).astype(np.float32)


def a_texto_bytes(Xb):
    """Cada byte -> un carácter en el rango Latin-1, para TF-IDF de n-gramas de bytes."""
    return [b.decode("latin-1") for b in Xb]


def configuraciones(smoke):
    """(nombre, representación, pipeline, espacio de búsqueda)."""
    tfidf = TfidfVectorizer(analyzer="char", lowercase=False, sublinear_tf=True)
    if smoke:
        return [
            ("n-gramas de bytes + LinearSVC", "ngramas",
             Pipeline([("tfidf", tfidf),
                       ("model", LinearSVC(max_iter=5000, random_state=42,
                                           class_weight="balanced"))]),
             {"tfidf__ngram_range": [(2, 4)], "model__C": [1.0, 10.0]}),
            ("posicional + RandomForest", "posicional",
             Pipeline([("model", RandomForestClassifier(random_state=42, n_jobs=1,
                                                        class_weight="balanced"))]),
             {"model__n_estimators": [100]}),
        ]
    # NOTA: no se incluye un modelo lineal sobre la representación posicional cruda.
    # Los valores de byte son CATEGÓRICOS, no ordinales: para un modelo lineal, "byte 87"
    # no es mayor ni menor que "byte 70", así que la relación de orden que asume es
    # espuria. Además 1.016 de las 1.024 posiciones son ruido que diluye la señal
    # (verificado con datos sintéticos: 0,24 de exactitud donde debía dar ~1,0).
    # Sobre bytes en posiciones fijas se usan árboles, que parten por valor exacto; y
    # para capturar las marcas se usan n-gramas de bytes, el análogo directo de los
    # n-gramas de caracteres del clasificador de notas.
    return [
        ("posicional + RandomForest", "posicional",
         Pipeline([("model", RandomForestClassifier(random_state=42, n_jobs=1,
                                                    class_weight="balanced"))]),
         {"model__n_estimators": [100, 200, 300],
          "model__max_depth": [None, 10, 20, 40],
          "model__min_samples_leaf": [1, 2, 5],
          "model__max_features": ["sqrt", "log2", 0.3]}),
        ("n-gramas de bytes + LinearSVC", "ngramas",
         Pipeline([("tfidf", tfidf),
                   ("model", LinearSVC(max_iter=10000, random_state=42,
                                       class_weight="balanced"))]),
         {"tfidf__ngram_range": [(1, 2), (2, 3), (2, 4), (3, 4)],
          "tfidf__max_features": [5000, 20000, 50000],
          "tfidf__min_df": [1, 2],
          "model__C": [0.1, 1.0, 10.0, 100.0]}),
        ("n-gramas de bytes + LogReg", "ngramas",
         Pipeline([("tfidf", tfidf),
                   ("model", LogisticRegression(max_iter=3000, solver="lbfgs",
                                                class_weight="balanced",
                                                random_state=42))]),
         {"tfidf__ngram_range": [(2, 3), (2, 4)],
          "tfidf__max_features": [20000, 50000],
          "model__C": [0.1, 1.0, 10.0]}),
    ]


def metricas(y, yp):
    return dict(accuracy=round(accuracy_score(y, yp), 4),
                balanced_accuracy=round(balanced_accuracy_score(y, yp), 4),
                f1_macro=round(f1_score(y, yp, average="macro", zero_division=0), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz", help="carpeta con subcarpetas <FAMILIA>[-small]")
    ap.add_argument("--por-familia", type=int, default=200)
    ap.add_argument("--n-iter", type=int, default=12)
    ap.add_argument("--folds-externos", type=int, default=3)
    ap.add_argument("--folds-internos", type=int, default=3)
    ap.add_argument("--por-familia-final", type=int, default=500,
                    help="archivos por familia para la re-evaluación final")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.por_familia, args.n_iter, args.por_familia_final = 20, 2, 30

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)

    log("=" * 74)
    log(f"  EXPERIMENTO 2c -- ML sobre bytes de cabecera/cola{' [SMOKE]' if args.smoke else ''}")
    log(f"  Ventana: {N_HEAD} bytes de cabecera + {N_TAIL} de cola. SIN nombre ni extensión.")
    log("=" * 74)
    log(f"Cargando desde {args.raiz} (hasta {args.por_familia} archivos por familia)...")
    Xb, y, familias = cargar(args.raiz, args.por_familia)
    azar = 1.0 / len(familias)
    log(f"\nTotal: {len(Xb)} archivos | {len(familias)} familias | azar = {azar:.4f}")
    log(f"Núcleos asignados: {N_JOBS}\n")

    reps = {"posicional": a_matriz_posicional(Xb), "ngramas": a_texto_bytes(Xb)}

    filas, mejores = [], {}
    for nombre, rep, pipe, espacio in configuraciones(args.smoke):
        X = reps[rep]
        log(f"[{nombre}] búsqueda anidada...")
        t0 = time.time()
        cv_ext = StratifiedKFold(args.folds_externos, shuffle=True, random_state=42)
        yp = np.empty_like(y)
        Xarr = X if rep == "posicional" else np.array(X, dtype=object)
        for k, (tr, te) in enumerate(cv_ext.split(Xarr, y)):
            busq = RandomizedSearchCV(
                pipe, espacio, n_iter=args.n_iter, scoring="f1_macro",
                cv=StratifiedKFold(args.folds_internos, shuffle=True, random_state=k),
                n_jobs=N_JOBS, random_state=42, refit=True)
            busq.fit(Xarr[tr], y[tr])
            yp[te] = busq.predict(Xarr[te])
            log(f"    fold {k+1}/{args.folds_externos}: interno {busq.best_score_:.3f}"
                f" | {busq.best_params_}")
            if k == 0:
                mejores[nombre] = {p: str(v) for p, v in busq.best_params_.items()}
        m = metricas(y, yp)
        m.update(configuracion=nombre, representacion=rep, cobertura=1.0,
                 n_archivos=len(y), segundos=round(time.time() - t0),
                 etapa="busqueda")
        filas.append(m)
        log(f"  => exactitud {m['accuracy']:.3f} | balanced {m['balanced_accuracy']:.3f}"
            f" | macro-F1 {m['f1_macro']:.3f} | {m['segundos']}s\n")

    import pandas as pd
    pd.DataFrame(filas).to_csv(OUT_DIR / "bytes_resumen.csv", index=False)

    # ---- Re-evaluación de la mejor configuración con más archivos por familia ----
    mejor = max(filas, key=lambda r: r["f1_macro"])
    log("=" * 74)
    log(f"  ETAPA FINAL -- {mejor['configuracion']} con {args.por_familia_final} "
        f"archivos/familia")
    log("=" * 74)
    Xb2, y2, fam2 = cargar(args.raiz, args.por_familia_final, seed=7)
    cfg = next(c for c in configuraciones(args.smoke) if c[0] == mejor["configuracion"])
    _, rep, pipe, _ = cfg
    X2 = a_matriz_posicional(Xb2) if rep == "posicional" else np.array(
        a_texto_bytes(Xb2), dtype=object)
    params = {k: eval(v) if v[0] in "([0123456789" or v in ("None", "True", "False")
              else v for k, v in mejores[mejor["configuracion"]].items()}
    pipe.set_params(**params)
    log(f"  Hiperparámetros: {params}")
    t0 = time.time()
    yp2 = cross_val_predict(pipe, X2, y2, cv=StratifiedKFold(5, shuffle=True,
                            random_state=42), n_jobs=N_JOBS)
    mf = metricas(y2, yp2)
    mf.update(configuracion=mejor["configuracion"], representacion=rep, cobertura=1.0,
              n_archivos=len(y2), segundos=round(time.time() - t0), etapa="final")
    filas.append(mf)
    log(f"  => exactitud {mf['accuracy']:.3f} | balanced {mf['balanced_accuracy']:.3f}"
        f" | macro-F1 {mf['f1_macro']:.3f}  ({mf['segundos']}s)")

    rep_txt = classification_report(y2, yp2, zero_division=0)
    (OUT_DIR / "bytes_por_familia.txt").write_text(rep_txt, encoding="utf-8")
    log("\nReporte por familia (extracto):")
    log("\n".join(rep_txt.splitlines()[:8]))

    pd.DataFrame(filas).to_csv(OUT_DIR / "bytes_resumen.csv", index=False)
    (OUT_DIR / "bytes_manifiesto.json").write_text(json.dumps(dict(
        fecha=str(date.today()), raiz=str(args.raiz), n_head=N_HEAD, n_tail=N_TAIL,
        usa_nombre_o_extension=False, n_familias=len(familias), azar=round(azar, 4),
        por_familia_busqueda=args.por_familia, por_familia_final=args.por_familia_final,
        n_iter=args.n_iter, mejores_hiperparametros=mejores, resultado_final=mf,
        sklearn=sklearn.__version__, python=sys.version.split()[0],
        comparacion=dict(estadisticas_2_features="0.099",
                         firmas_exactas_donde_aplican="0.970 (cobertura 0.533)",
                         extension_sola="0.828 (identificador de campaña)"),
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\nSalidas en: {OUT_DIR}")


if __name__ == "__main__":
    main()
