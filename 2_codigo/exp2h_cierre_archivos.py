#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""Exp. 2h -- CIERRE DEL FRENTE DE ARCHIVOS: todas las dudas abiertas en una sola corrida.

POR QUÉ UNA SOLA CORRIDA
------------------------
Pedido de Romina (2026-09-28): «analizá bien ya todo lo que quieras, no podemos esperar 2 h
todo el tiempo para cada duda». Se revisó el frente entero y quedaron cuatro cosas que solo se
resuelven con los datos del clúster. Van juntas, ordenadas para que lo más importante salga
primero en el log.

(A) CENSO DEL CONJUNTO, sin aprendizaje. Varias frases del cap. 4 citan recuentos hechos a mano
    en agosto, antes de la cuarentena y del arreglo del filtro de PDF («CERBER renombra 981»,
    «BLACKMATTER 13», «25 de 30 familias con una sola extensión», «compartidas por BADRABBIT,
    NOTPETYA y JIGSAW»). Ya se encontró uno mal: BLACKMATTER eran 12 más el PDF de documentación,
    y su carpeta son 988 imágenes. Se recuentan todos sobre el conjunto vigente, y se buscan
    ARCHIVOS DUPLICADOS por SHA-256 (dentro de cada familia y entre familias): nunca se miró, y un
    duplicado a ambos lados de un pliegue inflaría la validación cruzada.

(B) VALIDACIÓN POR TIPOS CON PONDERACIÓN DE CLASES. La publicada del 2c (analisis_bytes.py) y
    todas las validaciones cruzadas usan class_weight="balanced"; las de tipos del 2e-c, 2f y 2g
    se corrieron sin ponderar (exp2e_validacion_tipos.py decía «sin class_weight, como allí», y era
    falso). En validación cruzada da igual: con 400 archivos por familia en cada entrenamiento los
    pesos valen todos 1. En la de tipos no: al sacar las imágenes, BLACKMATTER se queda casi sin
    entrenamiento, y sin ponderar su F1 es 0 (job 4079). Se repiten las columnas con ponderación y
    se reporta también el macro-F1 sin las familias que tienen < 50 archivos de entrenamiento en
    el pliegue (esas no prueban la generalización a un tipo nuevo sino el aprendizaje sin ejemplos).

(C) ABLACIÓN DEL SISTEMA COMPLETO. El canónico (bytes + estructura + extensión, 0,9998) nunca se
    comparó contra sus partes: ¿cuánto da la forma de la extensión SOLA? (el 2d tuvo ese control
    para la forma del nombre completo, 0,577, y fue el que decidió la lectura). ¿Hacen falta los
    1.024 bytes si están la estructura y la extensión? ¿Hace falta la estructura si están los bytes
    y la extensión? Un jurado lo va a preguntar.

(D) SEMILLAS EN LA VALIDACIÓN POR TIPOS. El 0,9962 bajo tipo no visto sale de una sola semilla de
    muestreo; la validación cruzada tiene cinco. Se repite el sistema completo (5) con las semillas
    1 a 4.

COLUMNAS
  (1) bytes 512+512 · (2) bytes + estructura · (3) + forma del nombre completo (la del 2f)
  (5) bytes + estructura + forma de la extensión (el canónico, 2g)
  (6) solo forma de la extensión (14) · (7) estructura + extensión (58) · (8) bytes + extensión
  Mismo RF que todo el frente: 300 árboles, profundidad 20, hoja 2, max_features 0,3,
  class_weight="balanced".

PUERTAS
  - Censo: 29.948 muestras (el total del diagnóstico 4082). Aviso, no aborta.
  - (B): n y familias de cada pliegue idénticos al 2e-c (job 4079). Si no, aborta.
  - (C): la semilla 0 reproduce el macro-F1 del 2g: (2) 0,9356 y (5) 0,9999. Si no, aborta (C) y
    (D): los deltas se parean contra las cifras del 2g.

PREREGISTRO -- escrito y commiteado ANTES de correr (2026-09-28)
----------------------------------------------------------------
Censo
  A1. CERBER: 0 de sus muestras con un tipo de documento legible en el nombre.
  A2. Extensión final de documento solo en BADRABBIT, NOTPETYA y JIGSAW; en JIGSAW, exactamente
      2 (los dos PDF en claro ya declarados).
  A3. Firma de documento en claro: CERBER 988, JIGSAW 2, las demás 0.
  A4. Duplicados exactos entre familias distintas: 0.
  A5. Duplicados exactos dentro de una familia: solo en CRYPTOLOCKER o NOTPETYA (las de cifrado
      determinista), o en ninguna.
Tipos con ponderación (semilla 0)
  P1. BLACKMATTER obtiene F1 >= 0,50 en el pliegue jpg con solo bytes (1).
  P2. La exactitud media de (1) sube a >= 0,865 (sin ponderar: 0,8516).
  P3. Δ macro-F1 (2)-(1) > 0 en los siete pliegues.
  P4. (5) >= 0,99 de macro-F1 en los siete pliegues.
  P5. (3) sigue colapsando en jpg (< 0,50): el colapso es de la base del nombre, no de la ponderación.
  P6. (6) no depende del tipo: su promedio bajo tipo no visto no queda más de 0,05 por debajo de
      su valor en validación cruzada.
Ablación (validación cruzada, 5 semillas)
  A6. (6) solo extensión < 0,95 de macro-F1: la extensión sola no alcanza al sistema.
  A7. (7) estructura + extensión >= 0,995.
  A8. (8) bytes + extensión >= 0,999.
Semillas
  M1. (5) promedia >= 0,99 bajo tipo no visto en cada una de las cinco semillas.

Lectura acordada de antemano:
  - P3 y P4 cumplen -> las cifras con ponderación reemplazan a las tablas de tipos del 2e-c y del
    2g (misma medición que la publicada del 2c) y las conclusiones se mantienen.
  - A6 falla (la extensión sola >= 0,95) -> la capa del nombre hace casi todo el trabajo del sistema,
    y se declara con ese peso.
  - A7 o A8 cumplen -> en el sistema completo esa capa de contenido es prescindible, y se reporta:
    el canónico sigue siendo el sistema apilado (criterio de Romina), con su ablación al lado.
  - Cualquier recuento del censo distinto del de la tesis -> se corrige la frase con este número.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))
from exp2e_estructura_bytes import N_JOBS, _magia, es_documentacion  # noqa: E402
from exp2g_nombre_robusto import EXT_DOCUMENTO, HIPER, cargar, ic95, tipo_documento  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TIPOS = ["doc", "docx", "jpg", "pdf", "pptx", "xls", "xlsx"]
MIN_ARCHIVOS_TIPO, MIN_FAMILIAS_PRUEBA, MIN_ENTRENAMIENTO = 200, 10, 50
TOTAL_4082 = 29948
PUERTA_TIPOS = {"doc": (2071, 28), "docx": (2007, 28), "jpg": (2401, 28), "pdf": (1977, 28),
                "pptx": (2056, 28), "xls": (1974, 28), "xlsx": (2007, 28)}
# macro-F1 por semilla del 2g (job 4091): mismas semillas y particiones -> deltas pareados
REF_2G = {"2": [0.9356, 0.9355, 0.9361, 0.9360, 0.9365], "5": [0.9999, 0.9998, 0.9999, 0.9997, 0.9999]}
# macro-F1 por pliegue SIN ponderar: (1) y (2) del 2e-c (4079), (3) y (5) del 2g (4091)
SIN_PONDERAR = {
    "1": {"doc": 0.8843, "docx": 0.8860, "jpg": 0.7987, "pdf": 0.7746, "pptx": 0.8691, "xls": 0.8835, "xlsx": 0.8846},
    "2": {"doc": 0.9337, "docx": 0.9023, "jpg": 0.8052, "pdf": 0.8131, "pptx": 0.8893, "xls": 0.9037, "xlsx": 0.9047},
    "3": {"doc": 0.9943, "docx": 1.0000, "jpg": 0.2167, "pdf": 0.9791, "pptx": 1.0000, "xls": 0.9995, "xlsx": 1.0000},
    "5": {"doc": 0.9943, "docx": 1.0000, "jpg": 0.9997, "pdf": 0.9800, "pptx": 1.0000, "xls": 0.9995, "xlsx": 1.0000},
}
NOMBRE = {"1": "bytes", "2": "b+estr", "3": "+nombre", "5": "+extensión", "6": "solo ext",
          "7": "estr+ext", "8": "bytes+ext"}


def familia_de(d):
    fam = d.name.upper()
    for suf in ("-SMALL", "_SMALL", "-TINY", "_TINY"):
        fam = fam.removesuffix(suf)
    return fam


def columnas(Xb, Xe, Xf, Xr, cuales):
    todas = {"1": lambda: Xb, "2": lambda: np.hstack([Xb, Xe]), "3": lambda: np.hstack([Xb, Xe, Xf]),
             "5": lambda: np.hstack([Xb, Xe, Xr]), "6": lambda: Xr, "7": lambda: np.hstack([Xe, Xr]),
             "8": lambda: np.hstack([Xb, Xr])}
    return {k: todas[k]() for k in cuales}


# ============================================================ (A) censo
def censo(raiz, out, log):
    t0 = time.time()
    filas = []
    for d in sorted(p for p in Path(raiz).iterdir() if p.is_dir()):
        fam = familia_de(d)
        for p in sorted(q for q in d.iterdir() if q.is_file()):
            if es_documentacion(p, fam):
                continue
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                cab = fh.read(16)
                h.update(cab)
                for bloque in iter(lambda: fh.read(1 << 20), b""):
                    h.update(bloque)
            ext = p.name.rpartition(".")[2].lower() if "." in p.name else ""
            tp = tipo_documento(p.name)
            filas.append(dict(familia=fam, nombre=p.name, tipo=tp if tp in TIPOS else "sin_tipo",
                              ext=ext, ext_documento=ext in EXT_DOCUMENTO, en_claro=_magia(cab) or "",
                              tam=p.stat().st_size, sha=h.hexdigest()))
    c = pd.DataFrame(filas)
    c.to_csv(out / "censo_por_archivo.csv", index=False)
    log(f"  {len(c)} muestras en {c.familia.nunique()} carpetas · {round(time.time() - t0)} s")
    log(f"  PUERTA DEL CENSO: {len(c)} contra {TOTAL_4082} del diagnóstico 4082 "
        + ("✔" if len(c) == TOTAL_4082 else "✘ (AVISO: no es el mismo conjunto que el 4082; sigue)"))

    tabla = pd.crosstab(c.familia, c.tipo).reindex(columns=TIPOS + ["sin_tipo"], fill_value=0)
    tabla.insert(0, "total", c.groupby("familia").size())
    g = c.groupby("familia")
    tabla["ext_distintas"] = g.ext.nunique()
    tabla["ext_mas_comun"] = g.ext.agg(lambda s: s.value_counts().index[0])
    tabla["prop_mas_comun"] = g.ext.agg(lambda s: round(s.value_counts().iloc[0] / len(s), 3))
    tabla["ext_de_documento"] = g.ext_documento.sum()
    tabla["en_claro"] = g.en_claro.agg(lambda s: int((s != "").sum()))
    tabla.to_csv(out / "censo_por_familia.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    log("\n  Por familia (muestras por tipo leído del nombre; «sin_tipo»: nombre sustituido):")
    log(tabla.to_string())

    una = int((tabla.ext_distintas == 1).sum())
    compartidas = c.groupby("ext").familia.agg(lambda s: sorted(set(s)))
    compartidas = compartidas[compartidas.map(len) > 1]
    log(f"\n  Familias con una sola extensión final: {una} de {len(tabla)}")
    log(f"  Extensiones finales distintas en el conjunto: {c.ext.nunique()}; compartidas por más de "
        f"una familia: {len(compartidas)}")
    for e, fams in compartidas.items():
        log(f"    .{e}: {', '.join(fams)}")

    dup = c[c.duplicated("sha", keep=False)]
    grupos = dup.groupby("sha").familia.agg(lambda s: sorted(s))
    entre = grupos[grupos.map(lambda f: len(set(f)) > 1)]
    dentro = grupos[grupos.map(lambda f: len(set(f)) == 1)]
    log(f"\n  Duplicados exactos (SHA-256): {len(grupos)} grupos, {len(dup)} archivos")
    log(f"    entre familias distintas: {len(entre)} grupos"
        + (": " + "; ".join(" + ".join(f) for f in entre.head(10)) if len(entre) else ""))
    fam_dentro = defaultdict(lambda: [0, 0])
    for f in dentro:
        fam_dentro[f[0]][0] += 1
        fam_dentro[f[0]][1] += len(f)
    log(f"    dentro de una familia: {len(dentro)} grupos"
        + (": " + ", ".join(f"{k} {v[0]} grupos / {v[1]} archivos" for k, v in sorted(fam_dentro.items()))
           if len(dentro) else ""))
    dup.sort_values(["sha", "familia"]).to_csv(out / "censo_duplicados.csv", index=False)

    r = {}
    r["A1"] = int(tabla.loc["CERBER", TIPOS].sum()) if "CERBER" in tabla.index else None
    extdoc = tabla[tabla.ext_de_documento > 0].ext_de_documento.to_dict()
    r["A2"] = (set(extdoc) <= {"BADRABBIT", "NOTPETYA", "JIGSAW"} and extdoc.get("JIGSAW", 0) == 2)
    claro = tabla[tabla.en_claro > 0].en_claro.to_dict()
    r["A3"] = claro == {"CERBER": 988, "JIGSAW": 2}
    r["A4"] = len(entre) == 0
    r["A5"] = set(fam_dentro) <= {"CRYPTOLOCKER", "NOTPETYA"}
    r["_extdoc"], r["_claro"], r["_dentro"] = extdoc, claro, dict(fam_dentro)
    return r


# ============================================================ (B) y (D) tipos
def tipos_no_vistos(X_cols, y, tipos, semilla, log, porfam=None, cabecera=True):
    todas = sorted(set(y))
    cuenta = pd.Series(tipos).value_counts()
    candidatos = sorted(t for t, c in cuenta.items() if c >= MIN_ARCHIVOS_TIPO and t in TIPOS)
    filas = []
    if cabecera:
        log(f"  {'tipo':<5} {'n':>5} {'fam':>4}   " + "  ".join(f"{NOMBRE[k]:>11}" for k in X_cols)
            + "   (macro-F1/exactitud)")
    for tp in candidatos:
        t1 = time.time()
        te, tr = np.flatnonzero(tipos == tp), np.flatnonzero(tipos != tp)
        fams = np.unique(y[te])
        if len(fams) < MIN_FAMILIAS_PRUEBA:
            continue
        n_tr = pd.Series(y[tr]).value_counts()
        fams_ok = np.array([f for f in fams if n_tr.get(f, 0) >= MIN_ENTRENAMIENTO])
        pocas = [f for f in fams if n_tr.get(f, 0) < MIN_ENTRENAMIENTO]
        ok = np.isin(y[te], fams_ok)
        r = dict(semilla=semilla, tipo=tp, n=len(te), familias=len(fams),
                 fuera_de_la_prueba=";".join(sorted(set(todas) - set(fams))),
                 con_poco_entrenamiento=";".join(f"{f}:{int(n_tr.get(f, 0))}" for f in pocas))
        for k, X in X_cols.items():
            m = RandomForestClassifier(random_state=42, n_jobs=N_JOBS, class_weight="balanced",
                                       **HIPER).fit(X[tr], y[tr])
            yp = m.predict(X[te])
            r[f"acc_{k}"] = round(accuracy_score(y[te], yp), 4)
            r[f"f1_{k}"] = round(f1_score(y[te], yp, average="macro", labels=fams, zero_division=0), 4)
            r[f"f1ok_{k}"] = round(f1_score(y[te][ok], yp[ok], average="macro", labels=fams_ok,
                                            zero_division=0), 4)
            if porfam is not None:
                rep = classification_report(y[te], yp, labels=fams, zero_division=0, output_dict=True)
                porfam += [dict(semilla=semilla, tipo=tp, columna=k, familia=f,
                                n_prueba=int((y[te] == f).sum()), n_entrenamiento=int(n_tr.get(f, 0)),
                                f1=round(rep[f]["f1-score"], 4)) for f in fams]
        filas.append(r)
        log(f"  {tp:<5} {len(te):>5} {len(fams):>4}   "
            + "  ".join(f"{r[f'f1_{k}']:.4f}/{r[f'acc_{k}']:.2f}" for k in X_cols)
            + f"   ({round(time.time() - t1)} s)")
        if pocas and cabecera:
            log(f"        con menos de {MIN_ENTRENAMIENTO} archivos de entrenamiento: "
                + ", ".join(f"{f} ({int(n_tr.get(f, 0))})" for f in pocas))
    return pd.DataFrame(filas)


# ============================================================ (C) validación cruzada
def cv(X, y, s):
    cvk = StratifiedKFold(5, shuffle=True, random_state=s)
    clf = RandomForestClassifier(random_state=s, n_jobs=1, class_weight="balanced", **HIPER)
    yp = cross_val_predict(clf, X, y, cv=cvk, n_jobs=N_JOBS)
    rep = classification_report(y, yp, zero_division=0, output_dict=True)
    return (round(accuracy_score(y, yp), 4), round(f1_score(y, yp, average="macro", zero_division=0), 4),
            {f: round(rep[f]["f1-score"], 4) for f in sorted(set(y))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz")
    ap.add_argument("--por-familia", type=int, default=500)
    ap.add_argument("--semillas", default="0,1,2,3,4")
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--prueba", action="store_true")
    args = ap.parse_args()
    global MIN_ARCHIVOS_TIPO, MIN_FAMILIAS_PRUEBA, MIN_ENTRENAMIENTO
    if args.prueba:
        args.por_familia, args.semillas = 60, "0,1"
        MIN_ARCHIVOS_TIPO, MIN_FAMILIAS_PRUEBA, MIN_ENTRENAMIENTO = 20, 2, 5
    semillas = [int(x) for x in args.semillas.split(",") if x.strip()]

    base = _AQUI.parent / "4_resultados" if (_AQUI.parent / "4_resultados").is_dir() else _AQUI
    out = args.salida or base / ("resultados_exp2h_job" + os.environ.get("SLURM_JOB_ID", "local"))
    if out.exists() and any(out.iterdir()):
        sys.exit(f"ABORTA: {out} ya tiene resultados. Borrarla o pasar --salida.")
    out.mkdir(parents=True, exist_ok=True)
    reg = open(out / "log.txt", "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        reg.write(m + "\n")
        reg.flush()

    def titulo(t):
        log("\n" + "=" * 78)
        log("  " + t)
        log("=" * 78)

    veredicto = {}
    log("=" * 78)
    log("  EXP. 2h -- CIERRE DEL FRENTE DE ARCHIVOS")
    log("=" * 78)
    log(f"  {args.raiz} · {args.por_familia}/familia · semillas {semillas} · núcleos {N_JOBS}")
    log(f"  RF {HIPER}, class_weight='balanced' en todo")

    # ---------------------------------------------------------------- (A)
    titulo("(A) CENSO DEL CONJUNTO VIGENTE (sin aprendizaje)")
    rc = censo(args.raiz, out, log)

    # ---------------------------------------------------------------- (B)
    titulo("(B) DEJAR-UN-TIPO-FUERA CON PONDERACIÓN DE CLASES (semilla de muestreo 0)")
    t0 = time.time()
    Xb, Xe, Xf, Xr, y, tipos, _ = cargar(args.raiz, args.por_familia, semillas[0], log)
    log(f"  {len(y)} archivos · {len(set(y))} familias · carga {round(time.time() - t0)} s")
    cand = sorted(t for t, c in pd.Series(tipos).value_counts().items() if c >= MIN_ARCHIVOS_TIPO and t in TIPOS)
    obs = {tp: (int((tipos == tp).sum()), int(len(np.unique(y[tipos == tp])))) for tp in cand}
    if args.prueba:
        log(f"  (prueba: puerta salteada) {obs}")
    elif obs != PUERTA_TIPOS:
        log(f"  PUERTA FALLA: {obs}\n               en el 2e-c: {PUERTA_TIPOS}")
        sys.exit(1)
    else:
        log("  PUERTA: n y familias de los siete pliegues idénticos al 2e-c (job 4079) ✔")
    porfam = []
    rb = tipos_no_vistos(columnas(Xb, Xe, Xf, Xr, ["1", "2", "3", "5", "6", "7", "8"]), y, tipos,
                         semillas[0], log, porfam)
    rb.to_csv(out / "tipos_por_pliegue.csv", index=False)
    pf = pd.DataFrame(porfam)
    pf.to_csv(out / "tipos_por_familia.csv", index=False)
    log("\n  Promedio de los siete pliegues (macro-F1 · exactitud · macro-F1 sin las de "
        f"< {MIN_ENTRENAMIENTO} de entrenamiento):")
    for k in ["1", "2", "3", "5", "6", "7", "8"]:
        log(f"    ({k}) {NOMBRE[k]:<11} {rb[f'f1_{k}'].mean():.4f} · {rb[f'acc_{k}'].mean():.4f} · "
            f"{rb[f'f1ok_{k}'].mean():.4f}")
    for a, b in (("2", "1"), ("5", "2")):
        d = rb[f"f1_{a}"] - rb[f"f1_{b}"]
        m, lo, hi = ic95(d)
        log(f"  Δ ({a})-({b}) por pliegue: {m:+.4f} [{lo:+.4f}; {hi:+.4f}]  {int((d > 0).sum())}/{len(d)}")
    if not args.prueba:
        log("\n  Con ponderación menos sin ponderar (2e-c y 2g), macro-F1 por pliegue:")
        log(f"  {'tipo':<5} " + "  ".join(f"{NOMBRE[k]:>10}" for k in ["1", "2", "3", "5"]))
        for _, r in rb.iterrows():
            log(f"  {r.tipo:<5} " + "  ".join(f"{r[f'f1_{k}'] - SIN_PONDERAR[k][r.tipo]:>+10.4f}"
                                              for k in ["1", "2", "3", "5"]))
    bm = pf[(pf.familia == "BLACKMATTER") & (pf.tipo == "jpg")]
    if len(bm):
        log("\n  BLACKMATTER en el pliegue jpg (sin ponderar: F1 0,0 en bytes y en b+estr):")
        for _, r in bm.iterrows():
            log(f"    ({r.columna}) {NOMBRE[r.columna]:<11} F1 {r.f1:.4f} · {r.n_prueba} de prueba · "
                f"{r.n_entrenamiento} de entrenamiento")
    b1 = bm[bm.columna == "1"].f1
    veredicto["P1"] = (bool(len(b1)) and float(b1.iloc[0]) >= 0.50, float(b1.iloc[0]) if len(b1) else float("nan"))
    veredicto["P2"] = (rb.acc_1.mean() >= 0.865, rb.acc_1.mean())
    d21 = rb.f1_2 - rb.f1_1
    veredicto["P3"] = (bool((d21 > 0).all()), d21.min())
    veredicto["P4"] = (bool((rb.f1_5 >= 0.99).all()), rb.f1_5.min())
    j3 = rb[rb.tipo == "jpg"].f1_3
    veredicto["P5"] = (bool(len(j3)) and float(j3.iloc[0]) < 0.50, float(j3.iloc[0]) if len(j3) else float("nan"))
    log(f"  ({round(time.time() - t0)} s)")

    # ---------------------------------------------------------------- (C) y (D)
    titulo("(C) ABLACIÓN DEL SISTEMA EN VALIDACIÓN CRUZADA · (D) TIPOS CON LAS DEMÁS SEMILLAS")
    filas_cv, fam_cv, filas_d = [], [], []
    seguir = True
    for s in semillas:
        t0 = time.time()
        log(f"\n--- semilla {s} ---")
        if s != semillas[0]:
            Xb, Xe, Xf, Xr, y, tipos, _ = cargar(args.raiz, args.por_familia, s, log)
        if s == semillas[0]:
            for k in ["2", "5"]:
                acc, f1, _ = cv(columnas(Xb, Xe, Xf, Xr, [k])[k], y, s)
                ref = REF_2G[k][s] if s < len(REF_2G[k]) else None
                log(f"    ({k}) {NOMBRE[k]:<11} macro-F1 {f1:.4f}  (2g: {ref})")
                if not args.prueba and f1 != ref:
                    seguir = False
            if not seguir:
                log("  PUERTA CV FALLA: la semilla 0 no reproduce al 2g; (C) y (D) no se parean. Se corta acá.")
                break
            log("  PUERTA CV: la semilla 0 reproduce al 2g ✔" if not args.prueba else "  (prueba: puerta salteada)")
        for k, X in columnas(Xb, Xe, Xf, Xr, ["6", "7", "8"]).items():
            acc, f1, porf = cv(X, y, s)
            filas_cv.append(dict(semilla=s, columna=k, accuracy=acc, f1_macro=f1))
            fam_cv += [dict(semilla=s, columna=k, familia=f, f1=v) for f, v in porf.items()]
            log(f"    ({k}) {NOMBRE[k]:<11} exactitud {acc:.4f} | macro-F1 {f1:.4f}")
        pd.DataFrame(filas_cv).to_csv(out / "cv_ablacion_por_semilla.csv", index=False)
        pd.DataFrame(fam_cv).to_csv(out / "cv_ablacion_por_familia.csv", index=False)
        if s != semillas[0]:
            rd = tipos_no_vistos(columnas(Xb, Xe, Xf, Xr, ["5"]), y, tipos, s, log, None, cabecera=False)
            filas_d.append(dict(semilla=s, f1_5=rd.f1_5.mean(), minimo=rd.f1_5.min()))
            log(f"    (D) tipos no vistos, (5): promedio {rd.f1_5.mean():.4f}, mínimo {rd.f1_5.min():.4f}")
            pd.DataFrame(filas_d).to_csv(out / "tipos_semillas.csv", index=False)
        log(f"  ({round(time.time() - t0)} s)")

    if seguir and filas_cv:
        titulo("RESUMEN DE LA ABLACIÓN (validación cruzada, media ± desvío sobre las semillas)")
        dc = pd.DataFrame(filas_cv)
        log(dc.groupby("columna")[["accuracy", "f1_macro"]].agg(["mean", "std"]).round(4).to_string())
        log("  Referencias del 2g con las mismas semillas: (2) 0,9359 ± 0,0004 · (5) 0,9998 ± 0,0001")
        for k in ["6", "7", "8"]:
            v = dc[dc.columna == k].sort_values("semilla")
            if args.prueba:
                continue
            for ref in ("5", "2"):
                d = v.f1_macro.values - np.array([REF_2G[ref][s] for s in v.semilla])
                m, lo, hi = ic95(d)
                log(f"  Δ ({k})-({ref}) pareado por semilla: {m:+.4f} [{lo:+.4f}; {hi:+.4f}]  "
                    f"{int((d > 0).sum())}/{len(d)} a favor")
        fc = pd.DataFrame(fam_cv).groupby(["columna", "familia"]).f1.mean().unstack(0).round(4)
        fc.columns = [f"({c}) {NOMBRE[c]}" for c in fc.columns]
        log("\n  F1 por familia (media sobre semillas), ordenado por solo extensión:")
        log(fc.sort_values(fc.columns[0]).to_string())
        m6 = dc[dc.columna == "6"].f1_macro.mean()
        veredicto["A6"] = (m6 < 0.95, m6)
        veredicto["A7"] = (dc[dc.columna == "7"].f1_macro.mean() >= 0.995, dc[dc.columna == "7"].f1_macro.mean())
        veredicto["A8"] = (dc[dc.columna == "8"].f1_macro.mean() >= 0.999, dc[dc.columna == "8"].f1_macro.mean())
        veredicto["P6"] = (rb.f1_6.mean() >= m6 - 0.05, rb.f1_6.mean() - m6)
    if seguir and filas_d:
        dd = pd.DataFrame(filas_d)
        todas_s = [rb.f1_5.mean()] + list(dd.f1_5)
        log(f"\n  (D) sistema completo bajo tipo no visto, por semilla: "
            + ", ".join(f"{v:.4f}" for v in todas_s)
            + f"  ->  {np.mean(todas_s):.4f} ± {np.std(todas_s, ddof=1):.4f}")
        veredicto["M1"] = (min(todas_s) >= 0.99, min(todas_s))

    titulo("VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    textos = {
        "A1": ("CERBER sin muestras con tipo legible", rc["A1"] == 0, rc["A1"]),
        "A2": ("extensión de documento solo en BADRABBIT, NOTPETYA y JIGSAW (JIGSAW 2)", rc["A2"], rc["_extdoc"]),
        "A3": ("firma en claro: CERBER 988, JIGSAW 2, resto 0", rc["A3"], rc["_claro"]),
        "A4": ("0 duplicados entre familias", rc["A4"], ""),
        "A5": ("duplicados dentro de familia solo en CRYPTOLOCKER/NOTPETYA", rc["A5"], rc["_dentro"]),
    }
    for k, (txt, ok, val) in textos.items():
        log(f"  [{'CUMPLE' if ok else 'FALLA '}] {k} {txt}: {val}")
    desc = {"P1": "BLACKMATTER F1 >= 0,50 en jpg con bytes", "P2": "exactitud media de (1) >= 0,865",
            "P3": "Δ (2)-(1) > 0 en los 7 pliegues (mínimo)", "P4": "(5) >= 0,99 en los 7 pliegues (mínimo)",
            "P5": "(3) colapsa en jpg (< 0,50)", "P6": "(6) en tipos no queda > 0,05 bajo su CV (diferencia)",
            "A6": "(6) solo extensión < 0,95 en CV", "A7": "(7) estr+ext >= 0,995 en CV",
            "A8": "(8) bytes+ext >= 0,999 en CV", "M1": "(5) >= 0,99 en tipos en las 5 semillas (mínimo)"}
    for k, txt in desc.items():
        if k in veredicto:
            ok, val = veredicto[k]
            log(f"  [{'CUMPLE' if ok else 'FALLA '}] {k} {txt}: {val:.4f}")
        else:
            log(f"  [ ----  ] {k} {txt}: no se llegó a medir")

    (out / "manifiesto.json").write_text(json.dumps(dict(
        fecha=str(date.today()), raiz=str(args.raiz), por_familia=args.por_familia, semillas=semillas,
        hiper=HIPER, class_weight="balanced", min_entrenamiento=MIN_ENTRENAMIENTO,
        preregistro={k: bool(v[0]) for k, v in veredicto.items()} | {k: bool(textos[k][1]) for k in textos},
    ), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"\n  Salidas en: {out}")
    reg.close()


if __name__ == "__main__":
    main()
