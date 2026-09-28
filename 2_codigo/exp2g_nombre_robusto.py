#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""Exp. 2g -- el sistema completo con rasgos de nombre ROBUSTOS: solo lo que agrega el ransomware.

POR QUÉ HACE FALTA
------------------
El Exp. 2f (job 4083) midió el sistema completo --bytes + estructura + forma del nombre-- en
0,9998 de macro-F1, con las 30 familias por encima de 0,99. Pero su validación por tipos de
documento falló de forma catastrófica en un pliegue: con los jpg fuera del entrenamiento, sumar
la forma del nombre bajó el macro-F1 de 0,8052 a 0,2167 (−0,5885). En los otros seis tipos subió.

Causa VERIFICADA MIRANDO LOS ARCHIVOS (2026-09-28), no razonada:

    0001-doc.doc.avos2          0001-pdf.pdf.avos2          0001-jpg-fromweb.jpg.avos2

La base del nombre la puso NapierOne al armar su corpus, y en los jpg lleva un «-fromweb» que en
los documentos no está. `forma_del_nombre()` mira el nombre ENTERO -- longitud total, longitud de
la base, su composición --, así que aprendió también cómo nombró NapierOne sus archivos. Con jpg
fuera del entrenamiento, esa forma nunca vista empuja las predicciones a familias equivocadas.

EL ARREGLO
----------
Rasgos calculados SOLO sobre la extensión final -- lo que agrega el ransomware -- más la cantidad
de puntos del nombre, que no depende del tipo (los jpg y los documentos tienen los dos un punto
en la base). Y una marca de si la extensión final es la de un tipo de documento conocido, que
identifica a las familias que NO renombran (NOTPETYA, BADRABBIT) sin depender de cuál tipo sea.
La base heredada de NapierOne no se mira.

COLUMNAS
  CV (5 semillas):  (2) bytes + estructura · (5) bytes + estructura + forma de la EXTENSIÓN
  Tipos no vistos:  (2) · (3) bytes + estructura + forma COMPLETA (la del 2f) · (5)
La columna (3) en tipos no vistos reproduce el colapso del 2f al lado del arreglo, para que la
comparación esté en una sola tabla. En CV no se repite: el 2f ya la midió con las mismas
semillas, el mismo muestreo y las mismas particiones.

PREREGISTRO -- escrito y commiteado ANTES de correr (2026-09-28)
----------------------------------------------------------------
G1. (5) alcanza macro-F1 >= 0,995 en CV (puede quedar algo por debajo del 0,9998 de la forma
    completa: pierde información de la base que, en CV, sí ayudaba).
G2. Tipos no vistos, pliegue jpg: Δ (5)-(2) >= -0,02. Es decir, SIN colapso.
G3. Tipos no vistos: Δ (5)-(2) > 0 en los siete pliegues.
G4. CV, F1 por familia con (5): NOTPETYA, JIGSAW, CRYPTOLOCKER y DARKSIDE >= 0,95.

Lectura acordada de antemano:
  - G1, G2 y G3 cumplen -> (5) es el sistema completo y robusto: la cifra canónica del frente.
  - G2 falla -> la extensión final también arrastra el tipo; el nombre no es robusto a tipos no
    vistos en ninguna de sus formas, y así se declara.
  - G4 falla para alguna familia -> dos familias usan extensiones de la misma forma; el nombre
    robusto no las separa y se dice cuáles.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))
from exp2e_estructura_bytes import (  # noqa: E402
    N_HEAD, N_JOBS, N_TAIL, _magia, es_documentacion, leer, rasgos_estructura)
from exp2d_nombre_extension import forma_del_nombre  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HIPER = dict(n_estimators=300, max_depth=20, min_samples_leaf=2, max_features=0.3)
DIFICILES = ["NOTPETYA", "JIGSAW", "CRYPTOLOCKER", "DARKSIDE", "WASTEDLOCKER", "SUNCRYPT"]
MIN_ARCHIVOS_TIPO, MIN_FAMILIAS_PRUEBA = 200, 10
HEX = set("0123456789abcdefABCDEF")
EXT_DOCUMENTO = {"doc", "docx", "xls", "xlsx", "ppt", "pptx", "pdf", "jpg", "jpeg", "png", "gif",
                 "bmp", "tif", "tiff", "txt", "rtf", "csv", "odt", "ods", "odp", "html", "htm",
                 "xml", "zip", "mp3", "mp4", "avi", "wav"}
NOMBRES_EXT = ["len_ext", "dig_ext", "let_ext", "may_ext", "prop_dig", "prop_let", "prop_may",
               "prop_hex", "entropia_ext", "ext_toda_hex", "ext_solo_letras", "ext_con_digitos",
               "n_puntos", "ext_es_tipo_documento"]


def _entropia(s):
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def forma_de_la_extension(nombres):
    """Forma de la extensión FINAL solamente, más los puntos del nombre y si la extensión final
    es la de un tipo de documento. No mira la base del nombre, que es herencia de NapierOne."""
    filas = []
    for n in nombres:
        ext = n.rpartition(".")[2] if "." in n else ""
        L = max(1, len(ext))
        d = sum(c.isdigit() for c in ext)
        a = sum(c.isalpha() for c in ext)
        u = sum(c.isupper() for c in ext)
        h = sum(c in HEX for c in ext)
        filas.append([len(ext), d, a, u, d / L, a / L, u / L, h / L, _entropia(ext),
                      1.0 if ext and all(c in HEX for c in ext) else 0.0,
                      1.0 if ext and ext.isalpha() else 0.0,
                      1.0 if any(c.isdigit() for c in ext) else 0.0,
                      float(n.count(".")),
                      1.0 if ext.lower() in EXT_DOCUMENTO else 0.0])
    return np.asarray(filas, dtype=np.float32)


def tipo_documento(nombre):
    m = re.match(r"^\d+-([a-z0-9]+)", nombre.lower())
    return m.group(1) if m else "desconocido"


def cargar(raiz, por_familia, seed, log=print):
    rng = np.random.default_rng(seed)
    Xb, Xe, y, nombres, tipos = [], [], [], [], []
    sospechosos = defaultdict(int)
    for d in sorted(p for p in Path(raiz).iterdir() if p.is_dir()):
        fam = d.name.upper()
        for suf in ("-SMALL", "_SMALL", "-TINY", "_TINY"):
            fam = fam.removesuffix(suf)
        arch = sorted(p for p in d.iterdir() if p.is_file() and not es_documentacion(p, fam))
        if len(arch) < 6:
            log(f"  ADVERTENCIA: {fam} tiene {len(arch)} archivos, omitida")
            continue
        for p in (arch[i] for i in rng.permutation(len(arch))[:por_familia]):
            head, cola, cabp, colap, medios, tam = leer(p)
            if _magia(head):
                sospechosos[fam] += 1
            Xb.append(head + cola)
            Xe.append(rasgos_estructura(cabp, colap, medios, tam))
            y.append(fam)
            nombres.append(p.name)
            tipos.append(tipo_documento(p.name))
    if sospechosos:
        log("  ⚠ archivos con firma en claro: " + ", ".join(f"{f} {n}" for f, n in sorted(sospechosos.items())))
    Xb = np.frombuffer(b"".join(Xb), dtype=np.uint8).reshape(len(Xb), N_HEAD + N_TAIL).astype(np.float32)
    Xe = np.nan_to_num(np.asarray(Xe, dtype=np.float32))
    return Xb, Xe, forma_del_nombre(nombres), forma_de_la_extension(nombres), np.array(y), np.array(tipos), nombres


def ic95(d):
    from scipy import stats
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return d.mean(), np.nan, np.nan
    t = stats.t.ppf(0.975, len(d) - 1)
    ee = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - t * ee, d.mean() + t * ee


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz")
    ap.add_argument("--por-familia", type=int, default=500)
    ap.add_argument("--semillas", default="0,1,2,3,4")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--prueba", action="store_true")
    args = ap.parse_args()
    global MIN_ARCHIVOS_TIPO, MIN_FAMILIAS_PRUEBA
    if args.prueba:
        args.por_familia, args.semillas, args.folds = 60, "0,1", 2
        MIN_ARCHIVOS_TIPO, MIN_FAMILIAS_PRUEBA = 20, 2
    semillas = [int(s) for s in args.semillas.split(",") if s.strip()]

    base = _AQUI.parent / "4_resultados" if (_AQUI.parent / "4_resultados").is_dir() else _AQUI
    out = args.salida or base / ("resultados_exp2g_job" + os.environ.get("SLURM_JOB_ID", "local"))
    if out.exists() and any(out.iterdir()):
        sys.exit(f"ABORTA: {out} ya tiene resultados. Borrarla o pasar --salida.")
    out.mkdir(parents=True, exist_ok=True)
    reg = open(out / "log.txt", "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        reg.write(m + "\n")
        reg.flush()

    log("=" * 78)
    log("  EXP. 2g -- SISTEMA COMPLETO CON NOMBRE ROBUSTO (solo la extensión final)")
    log("=" * 78)
    log(f"  {args.raiz} · {args.por_familia}/familia · semillas {semillas} · {args.folds} pliegues")
    log(f"  rasgos de extensión: {len(NOMBRES_EXT)} · núcleos {N_JOBS}")

    filas, porfam, s0 = [], [], None
    for s in semillas:
        t0 = time.time()
        log(f"\n--- semilla {s} ---")
        Xb, Xe, Xf, Xr, y, tipos, nombres = cargar(args.raiz, args.por_familia, s, log)
        if s == semillas[0]:
            s0 = (Xb, Xe, Xf, Xr, y, tipos)
            largo = pd.DataFrame({"tipo": tipos, "base": [len(n.rpartition(".")[0]) for n in nombres]})
            log("  largo medio de la base del nombre por tipo (la herencia de NapierOne): " +
                ", ".join(f"{t} {v:.1f}" for t, v in largo.groupby("tipo").base.mean().items()))
        cv = StratifiedKFold(args.folds, shuffle=True, random_state=s)
        for nombre, X in (("2_bytes_estructura", np.hstack([Xb, Xe])),
                          ("5_bytes_estructura_extension", np.hstack([Xb, Xe, Xr]))):
            clf = RandomForestClassifier(random_state=s, n_jobs=1, class_weight="balanced", **HIPER)
            yp = cross_val_predict(clf, X, y, cv=cv, n_jobs=N_JOBS)
            f = dict(semilla=s, columna=nombre, accuracy=round(accuracy_score(y, yp), 4),
                     balanced_accuracy=round(balanced_accuracy_score(y, yp), 4),
                     f1_macro=round(f1_score(y, yp, average="macro", zero_division=0), 4))
            filas.append(f)
            rep = classification_report(y, yp, zero_division=0, output_dict=True)
            porfam += [dict(semilla=s, columna=nombre, familia=fam, f1=round(rep[fam]["f1-score"], 4))
                       for fam in sorted(set(y))]
            log(f"    {nombre:<30} exactitud {f['accuracy']:.4f} | macro-F1 {f['f1_macro']:.4f}")
        log(f"  ({round(time.time() - t0)} s)")
        pd.DataFrame(filas).to_csv(out / "cv_por_semilla.csv", index=False)
        pd.DataFrame(porfam).to_csv(out / "cv_por_familia_y_semilla.csv", index=False)

    df, pf = pd.DataFrame(filas), pd.DataFrame(porfam)
    log("\n" + "=" * 78)
    log("  (A) VALIDACIÓN CRUZADA -- media ± desvío sobre las semillas")
    log("=" * 78)
    res = df.groupby("columna")[["accuracy", "f1_macro"]].agg(["mean", "std"]).round(4)
    log(res.to_string())
    res.to_csv(out / "cv_resumen.csv")
    piv = df.pivot(index="semilla", columns="columna", values="f1_macro")
    m, lo, hi = ic95(piv["5_bytes_estructura_extension"] - piv["2_bytes_estructura"])
    log(f"\n  Δ (5)-(2) macro-F1: {m:+.4f} [{lo:+.4f}; {hi:+.4f}]")
    log("  Referencia del 2f, mismas semillas y particiones: (3) forma completa = 0,9998")

    log("\n  F1 POR FAMILIA con (5), media ± desvío sobre las semillas (las difíciles primero):")
    fr = pf[pf.columna == "5_bytes_estructura_extension"].groupby("familia").f1.agg(["mean", "std"])
    fr2 = pf[pf.columna == "2_bytes_estructura"].groupby("familia").f1.mean()
    orden = [f for f in DIFICILES if f in fr.index] + sorted(f for f in fr.index if f not in DIFICILES)
    t = pd.DataFrame({"b+estructura": fr2.loc[orden], "b+estr+extension": fr.loc[orden, "mean"],
                      "desvío": fr.loc[orden, "std"]}).round(4)
    log(t.to_string())
    t.to_csv(out / "cv_por_familia_resumen.csv")
    bajo = t[t["b+estr+extension"] < 0.99].index.tolist()
    log(f"\n  Familias con F1 medio < 0,99 en (5): {bajo if bajo else 'ninguna'}")

    # ---------------------------------------------------------------- (B) tipos no vistos
    log("\n" + "=" * 78)
    log("  (B) DEJAR-UN-TIPO-FUERA (semilla de muestreo 0, un ajuste por pliegue)")
    log("=" * 78)
    Xb, Xe, Xf, Xr, y, tipos = s0
    cols = {"2": np.hstack([Xb, Xe]), "3": np.hstack([Xb, Xe, Xf]), "5": np.hstack([Xb, Xe, Xr])}
    cuenta = pd.Series(tipos).value_counts()
    candidatos = sorted(t for t, c in cuenta.items() if c >= MIN_ARCHIVOS_TIPO and t != "desconocido")
    log(f"  {'tipo':<6} {'(2) b+estr':>11} {'(3) +forma':>11} {'(5) +ext':>10}   {'Δ(3)-(2)':>9} {'Δ(5)-(2)':>9}")
    rows = []
    for tp in candidatos:
        te, tr = np.flatnonzero(tipos == tp), np.flatnonzero(tipos != tp)
        fams = np.unique(y[te])
        if len(fams) < MIN_FAMILIAS_PRUEBA:
            continue
        r = dict(tipo=tp, n=len(te), familias=len(fams))
        for k, X in cols.items():
            mdl = RandomForestClassifier(random_state=42, n_jobs=N_JOBS, **HIPER).fit(X[tr], y[tr])
            r[f"f1_{k}"] = round(f1_score(y[te], mdl.predict(X[te]), average="macro", labels=fams,
                                          zero_division=0), 4)
        r["d3"], r["d5"] = round(r["f1_3"] - r["f1_2"], 4), round(r["f1_5"] - r["f1_2"], 4)
        rows.append(r)
        log(f"  {tp:<6} {r['f1_2']:>11.4f} {r['f1_3']:>11.4f} {r['f1_5']:>10.4f}   {r['d3']:>+9.4f} {r['d5']:>+9.4f}")
    rb = pd.DataFrame(rows)
    rb.to_csv(out / "tipos_por_pliegue.csv", index=False)
    m5, lo5, hi5 = ic95(rb.d5)
    log(f"\n  PROMEDIO  (2) {rb.f1_2.mean():.4f} · (3) {rb.f1_3.mean():.4f} · (5) {rb.f1_5.mean():.4f}")
    log(f"  Δ (5)-(2) por pliegue: {m5:+.4f} [{lo5:+.4f}; {hi5:+.4f}]  {int((rb.d5 > 0).sum())}/{len(rb)} a favor")

    log("\n" + "-" * 78)
    log("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    log("-" * 78)
    f5 = res.loc["5_bytes_estructura_extension", ("f1_macro", "mean")]
    g1 = f5 >= 0.995
    log(f"  [{'CUMPLE' if g1 else 'FALLA '}] G1 (5) macro-F1 >= 0,995 en CV             {f5:.4f}")
    jpg = rb[rb.tipo == "jpg"]
    g2 = bool(len(jpg)) and float(jpg.d5.iloc[0]) >= -0.02
    log(f"  [{'CUMPLE' if g2 else 'FALLA '}] G2 jpg sin colapso: Δ (5)-(2) >= -0,02      "
        f"{float(jpg.d5.iloc[0]) if len(jpg) else float('nan'):+.4f}  (con la forma completa: "
        f"{float(jpg.d3.iloc[0]) if len(jpg) else float('nan'):+.4f})")
    g3 = bool((rb.d5 > 0).all())
    log(f"  [{'CUMPLE' if g3 else 'FALLA '}] G3 Δ (5)-(2) > 0 en los {len(rb)} pliegues         "
        f"mínimo {rb.d5.min():+.4f} ({rb.loc[rb.d5.idxmin(), 'tipo']})")
    d4 = t.loc[[f for f in ("NOTPETYA", "JIGSAW", "CRYPTOLOCKER", "DARKSIDE") if f in t.index], "b+estr+extension"]
    g4 = bool((d4 >= 0.95).all())
    log(f"  [{'CUMPLE' if g4 else 'FALLA '}] G4 las cuatro difíciles >= 0,95 con (5)      "
        + ", ".join(f"{k} {v:.4f}" for k, v in d4.items()))

    (out / "manifiesto.json").write_text(json.dumps(dict(
        fecha=str(date.today()), raiz=str(args.raiz), por_familia=args.por_familia, semillas=semillas,
        folds=args.folds, hiper=HIPER, rasgos_extension=NOMBRES_EXT,
        preregistro=dict(G1=bool(g1), G2=bool(g2), G3=bool(g3), G4=bool(g4)),
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n  Salidas en: {out}")
    reg.close()


if __name__ == "__main__":
    main()
