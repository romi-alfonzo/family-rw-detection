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
    python3.11 clasificador_bytes.py /ruta --multisemilla 1,2,3,4,5   # A.2: desvío
    python3.11 clasificador_bytes.py /ruta --smoke

SALIDA: una carpeta por corrida dentro de 4_resultados/, nombrada con la semilla y el
job de SLURM (`resultados_bytes_s42_job3639`). Antes se escribía siempre en
`resultados_bytes/` y cada corrida pisaba la anterior; así se perdió el reporte por
familia del job 3633. Si la carpeta ya tiene resultados, el script aborta salvo
que se le pase --forzar.
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
BASE_SALIDA = (_AQUI.parent / "4_resultados"
               if (_AQUI.parent / "4_resultados").is_dir() else _AQUI)

N_HEAD = 512   # bytes de cabecera
N_TAIL = 512   # bytes de cola

# Hiperparámetros elegidos por la búsqueda anidada del Exp. 2c (jobs 3557/3633/3639).
# Se dejan fijos en el modo multisemilla: lo que se mide ahí es la dispersión de la
# métrica entre semillas, no la selección de modelo (igual que en el clasificador de
# notas, que reporta 10 semillas con la configuración ya elegida).
HIPER_2C = dict(n_estimators=300, max_depth=20, min_samples_leaf=2, max_features=0.3)


def carpeta_salida(args):
    """Carpeta de salida ÚNICA por corrida.

    Hasta el 2026-08-17 este script escribía siempre en `resultados_bytes/`, de modo
    que cada corrida pisaba la anterior: el job 3639 borró los CSV del job 3633 y su
    reporte por familia completo se perdió. El nombre lleva ahora la semilla y, en el
    cluster, el identificador de trabajo de SLURM.
    """
    if args.salida:
        return BASE_SALIDA / args.salida
    partes = ["resultados_bytes"]
    partes.append("multisemilla" if args.multisemilla else f"s{args.semilla}")
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        partes.append(f"job{job}")
    return BASE_SALIDA / "_".join(partes)


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


def corrida_posicional(raiz, por_familia, semilla, hiper, folds, log):
    """Una evaluación completa de `posicional + RandomForest` con semilla explícita.

    La semilla gobierna las TRES fuentes de azar: qué archivos se muestrean, cómo se
    parten los pliegues y la aleatoriedad interna del bosque. Devuelve
    (métricas, y, y_predicho, familias)."""
    Xb, y, familias = cargar(raiz, por_familia, seed=semilla)
    X = a_matriz_posicional(Xb)
    pipe = Pipeline([("model", RandomForestClassifier(
        random_state=semilla, n_jobs=1, class_weight="balanced", **hiper))])
    t0 = time.time()
    yp = cross_val_predict(pipe, X, y, n_jobs=N_JOBS,
                           cv=StratifiedKFold(folds, shuffle=True,
                                              random_state=semilla))
    m = metricas(y, yp)
    m.update(semilla=semilla, n_archivos=len(y), n_familias=len(familias),
             segundos=round(time.time() - t0))
    log(f"  semilla {semilla:>3}: exactitud {m['accuracy']:.4f} | "
        f"balanced {m['balanced_accuracy']:.4f} | macro-F1 {m['f1_macro']:.4f}"
        f"  ({m['segundos']}s)")
    return m, y, yp, familias


def modo_multisemilla(args, out_dir, log):
    """A.2 del plan: dispersión de la métrica en el frente de archivos.

    El tutor pidió (12-08-2026) reportar el desvío también en archivos, que hasta ahora
    iba sin error mientras las notas se reportan como media ± desvío sobre 10 semillas.
    Se repite SOLO la evaluación final, con los hiperparámetros ya elegidos por la
    búsqueda anidada (HIPER_2C) — declararlo así en la tesis: es dispersión de la
    estimación, no una nueva selección de modelo."""
    import pandas as pd

    semillas = [int(s) for s in args.multisemilla.split(",") if s.strip()]
    log("=" * 74)
    log(f"  EXP. 2c -- DISPERSIÓN SOBRE {len(semillas)} SEMILLAS  {semillas}")
    log(f"  Hiperparámetros fijos: {HIPER_2C}")
    log(f"  {args.por_familia_final} archivos/familia | {args.folds_finales} pliegues")
    log("=" * 74)

    filas, por_familia = [], []
    for s in semillas:
        m, y, yp, familias = corrida_posicional(
            args.raiz, args.por_familia_final, s, HIPER_2C, args.folds_finales, log)
        filas.append(m)
        (out_dir / f"bytes_por_familia_s{s}.txt").write_text(
            classification_report(y, yp, zero_division=0), encoding="utf-8")
        rep = classification_report(y, yp, zero_division=0, output_dict=True)
        for fam in familias:
            r = rep[fam]
            por_familia.append(dict(semilla=s, familia=fam,
                                    precision=round(r["precision"], 4),
                                    recall=round(r["recall"], 4),
                                    f1=round(r["f1-score"], 4),
                                    soporte=int(r["support"])))
        # se guarda en cada iteración: si el trabajo se corta por tiempo, lo ya
        # corrido no se pierde (fue justamente lo que pasó con el job 3633)
        pd.DataFrame(filas).to_csv(out_dir / "bytes_multisemilla.csv", index=False)
        pd.DataFrame(por_familia).to_csv(
            out_dir / "bytes_multisemilla_por_familia.csv", index=False)

    df = pd.DataFrame(filas)
    resumen = []
    for col in ("accuracy", "balanced_accuracy", "f1_macro"):
        resumen.append(dict(metrica=col, media=round(df[col].mean(), 4),
                            desvio=round(df[col].std(ddof=1), 4),
                            minimo=round(df[col].min(), 4),
                            maximo=round(df[col].max(), 4), n_semillas=len(df)))
    df_res = pd.DataFrame(resumen)
    df_res.to_csv(out_dir / "bytes_multisemilla_resumen.csv", index=False)

    log("\n" + "=" * 74)
    for r in resumen:
        log(f"  {r['metrica']:<18} {r['media']:.4f} ± {r['desvio']:.4f}"
            f"   [{r['minimo']:.4f}; {r['maximo']:.4f}]")
    log("=" * 74)

    pf = pd.DataFrame(por_familia).groupby("familia")["f1"].agg(["mean", "std"])
    log("\nF1 por familia (media ± desvío), peores 8:")
    for fam, r in pf.sort_values("mean").head(8).iterrows():
        log(f"  {fam:<15} {r['mean']:.3f} ± {r['std']:.3f}")

    (out_dir / "bytes_manifiesto.json").write_text(json.dumps(dict(
        fecha=str(date.today()), raiz=str(args.raiz), modo="multisemilla",
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), semillas=semillas,
        n_head=N_HEAD, n_tail=N_TAIL, usa_nombre_o_extension=False,
        por_familia_final=args.por_familia_final, folds=args.folds_finales,
        hiperparametros_fijos={k: str(v) for k, v in HIPER_2C.items()},
        origen_hiperparametros="búsqueda anidada del Exp. 2c (jobs 3557/3633/3639)",
        resumen=resumen, por_semilla=filas,
        sklearn=sklearn.__version__, python=sys.version.split()[0],
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\nSalidas en: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz", help="carpeta con subcarpetas <FAMILIA>[-small]")
    ap.add_argument("--por-familia", type=int, default=200)
    ap.add_argument("--n-iter", type=int, default=12)
    ap.add_argument("--folds-externos", type=int, default=3)
    ap.add_argument("--folds-internos", type=int, default=3)
    ap.add_argument("--por-familia-final", type=int, default=500,
                    help="archivos por familia para la re-evaluación final")
    ap.add_argument("--folds-finales", type=int, default=5)
    ap.add_argument("--semilla", type=int, default=42,
                    help="semilla del muestreo de la etapa de búsqueda y de sus CV")
    ap.add_argument("--semilla-final", type=int, default=7,
                    help="semilla del muestreo de la etapa final (7 en los jobs "
                         "3633/3639: dejarla así para reproducirlos)")
    ap.add_argument("--multisemilla", default="",
                    help="A.2: lista de semillas separadas por coma (p. ej. "
                         "1,2,3,4,5). Corre SOLO la evaluación final, con los "
                         "hiperparámetros ya elegidos, y reporta media ± desvío")
    ap.add_argument("--salida", default="",
                    help="nombre de la carpeta de salida dentro de 4_resultados/ "
                         "(por defecto se arma con la semilla y el job de SLURM)")
    ap.add_argument("--forzar", action="store_true",
                    help="permitir escribir en una carpeta que ya tiene resultados")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.por_familia, args.n_iter, args.por_familia_final = 20, 2, 30

    out_dir = carpeta_salida(args)
    previos = sorted(out_dir.glob("bytes_*")) if out_dir.is_dir() else []
    if previos and not args.forzar:
        sys.exit(f"ABORTA: {out_dir} ya tiene {len(previos)} archivo(s) de una corrida "
                 f"anterior ({previos[0].name}...). Usá --salida OTRO_NOMBRE, o "
                 f"--forzar si de verdad querés sobrescribirlos.")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)

    if args.multisemilla:
        modo_multisemilla(args, out_dir, log)
        return

    log("=" * 74)
    log(f"  EXPERIMENTO 2c -- ML sobre bytes de cabecera/cola{' [SMOKE]' if args.smoke else ''}")
    log(f"  Ventana: {N_HEAD} bytes de cabecera + {N_TAIL} de cola. SIN nombre ni extensión.")
    log("=" * 74)
    log(f"Cargando desde {args.raiz} (hasta {args.por_familia} archivos por familia)...")
    Xb, y, familias = cargar(args.raiz, args.por_familia, seed=args.semilla)
    azar = 1.0 / len(familias)
    log(f"\nTotal: {len(Xb)} archivos | {len(familias)} familias | azar = {azar:.4f}")
    log(f"Núcleos asignados: {N_JOBS}\n")

    reps = {"posicional": a_matriz_posicional(Xb), "ngramas": a_texto_bytes(Xb)}

    filas, mejores = [], {}
    for nombre, rep, pipe, espacio in configuraciones(args.smoke):
        X = reps[rep]
        log(f"[{nombre}] búsqueda anidada...")
        t0 = time.time()
        cv_ext = StratifiedKFold(args.folds_externos, shuffle=True,
                                 random_state=args.semilla)
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
    pd.DataFrame(filas).to_csv(out_dir / "bytes_resumen.csv", index=False)

    # ---- Re-evaluación de la mejor configuración con más archivos por familia ----
    mejor = max(filas, key=lambda r: r["f1_macro"])
    log("=" * 74)
    log(f"  ETAPA FINAL -- {mejor['configuracion']} con {args.por_familia_final} "
        f"archivos/familia")
    log("=" * 74)
    Xb2, y2, fam2 = cargar(args.raiz, args.por_familia_final, seed=args.semilla_final)
    cfg = next(c for c in configuraciones(args.smoke) if c[0] == mejor["configuracion"])
    _, rep, pipe, _ = cfg
    X2 = a_matriz_posicional(Xb2) if rep == "posicional" else np.array(
        a_texto_bytes(Xb2), dtype=object)
    params = {k: eval(v) if v[0] in "([0123456789" or v in ("None", "True", "False")
              else v for k, v in mejores[mejor["configuracion"]].items()}
    pipe.set_params(**params)
    log(f"  Hiperparámetros: {params}")
    t0 = time.time()
    yp2 = cross_val_predict(pipe, X2, y2, n_jobs=N_JOBS,
                            cv=StratifiedKFold(args.folds_finales, shuffle=True,
                                               random_state=args.semilla))
    mf = metricas(y2, yp2)
    mf.update(configuracion=mejor["configuracion"], representacion=rep, cobertura=1.0,
              n_archivos=len(y2), segundos=round(time.time() - t0), etapa="final")
    filas.append(mf)
    log(f"  => exactitud {mf['accuracy']:.3f} | balanced {mf['balanced_accuracy']:.3f}"
        f" | macro-F1 {mf['f1_macro']:.3f}  ({mf['segundos']}s)")

    rep_txt = classification_report(y2, yp2, zero_division=0)
    (out_dir / "bytes_por_familia.txt").write_text(rep_txt, encoding="utf-8")
    log("\nReporte por familia (extracto):")
    log("\n".join(rep_txt.splitlines()[:8]))

    pd.DataFrame(filas).to_csv(out_dir / "bytes_resumen.csv", index=False)
    # NOTA: acá había un bloque `comparacion` con las cifras de los Exp. 2/2b
    # tipeadas a mano. Quedaron viejas al corregirse el detector estructural
    # (traía 0.533/0.828, superados por el job 3638) y el manifiesto las propagaba
    # como si fueran medición de esta corrida. Un manifiesto describe SU corrida;
    # las comparaciones entre experimentos se arman leyendo los CSV de cada uno.
    (out_dir / "bytes_manifiesto.json").write_text(json.dumps(dict(
        fecha=str(date.today()), raiz=str(args.raiz), n_head=N_HEAD, n_tail=N_TAIL,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        semilla=args.semilla, semilla_final=args.semilla_final,
        folds_externos=args.folds_externos, folds_internos=args.folds_internos,
        folds_finales=args.folds_finales,
        usa_nombre_o_extension=False, n_familias=len(familias), azar=round(azar, 4),
        por_familia_busqueda=args.por_familia, por_familia_final=args.por_familia_final,
        n_iter=args.n_iter, mejores_hiperparametros=mejores, resultado_final=mf,
        sklearn=sklearn.__version__, python=sys.version.split()[0],
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\nSalidas en: {out_dir}")


if __name__ == "__main__":
    main()
