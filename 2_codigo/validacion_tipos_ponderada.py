#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""Validación por tipos de documento con el MISMO modelo que la validación cruzada.

POR QUÉ HACE FALTA
------------------
La validación dejar-un-tipo-fuera publicada del Exp. 2c (`analisis_bytes.py`: 0,879 de exactitud,
0,861 de macro-F1) usó RandomForest con class_weight="balanced", igual que TODAS las validaciones
cruzadas del frente (2c, 2d, 2e, 2f, 2g). Las de tipos del 2e-c, 2f y 2g, en cambio, se corrieron
SIN ponderación de clases: `exp2e_validacion_tipos.py` decía «sin class_weight, como allí», y era
falso (verificado en el código el 2026-09-28). El 2f y el 2g copiaron ese modelo.

En validación cruzada la ponderación casi no cambia nada, porque las clases están equilibradas
(500 archivos cada una). En la de tipos sí: al sacar un tipo, cada familia pierde una cantidad
distinta de archivos de entrenamiento, y una la pierde casi entera.

Es BLACKMATTER. Verificado el 2026-09-28 (listado de la carpeta, pegado por Romina):
BLACKMATTER-small son 988 imágenes jpg, 12 archivos con el nombre sustituido y el PDF de
documentación. En el pliegue jpg se queda sin imágenes de entrenamiento, y sin ponderación el
bosque no la predice nunca: F1 0,0 en las dos columnas del 2e-c (`tipos_por_familia.csv`, job
4079). Es cerca del 20 % de ese pliegue, lo que explica que su exactitud caiga a 0,6951 mientras
el macro-F1 queda en 0,7987. En el 2c publicado, con ponderación, ese pliegue daba 0,874.

Este guion repite la validación por tipos del 2e-c y del 2g con class_weight="balanced", para que
sus cifras sean la misma medición que la publicada del 2c y que las de validación cruzada. Reporta
además el macro-F1 de cada pliegue sin las familias que tienen menos de 50 archivos de
entrenamiento en él (en la práctica, BLACKMATTER en jpg): esas no ponen a prueba la
generalización a un tipo nuevo sino el aprendizaje con casi ningún ejemplo.

COLUMNAS (mismo muestreo y pliegues que el 2e-c y el 2g: semilla de muestreo 0)
  (1) bytes · (2) bytes + estructura · (3) + forma del nombre completo · (5) + forma de la extensión

PUERTA: el n y la cantidad de familias de cada pliegue tienen que ser los del 2e-c. Si no, aborta:
no sería la misma prueba.

PREREGISTRO -- escrito y commiteado ANTES de correr (2026-09-28)
----------------------------------------------------------------
P1. Con ponderación, BLACKMATTER obtiene F1 >= 0,50 en el pliegue jpg con solo bytes (1).
P2. La exactitud media de (1) sobre los siete pliegues sube a >= 0,865 (el 2e-c dio 0,8516).
P3. Δ macro-F1 (2)-(1) > 0 en los siete pliegues.
P4. (5) alcanza macro-F1 >= 0,99 en los siete pliegues.
P5. (3) sigue colapsando en jpg (macro-F1 < 0,50): el colapso viene de la base del nombre que puso
    NapierOne, no de la ponderación.

Lectura acordada de antemano:
  - P3 y P4 cumplen -> estas cifras reemplazan a las tablas de tipos no vistos del 2e-c y del 2g
    (misma medición que la publicada del 2c) y las conclusiones se mantienen.
  - P1 falla -> lo de BLACKMATTER no es la ponderación: se reporta el pliegue jpg con y sin ella.
  - P4 falla -> el sistema completo no es robusto a tipos no vistos con el modelo de la validación
    cruzada, y se declara así.
"""

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))
from exp2e_estructura_bytes import N_JOBS  # noqa: E402
from exp2g_nombre_robusto import HIPER, cargar, ic95  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MIN_ARCHIVOS_TIPO, MIN_FAMILIAS_PRUEBA, MIN_ENTRENAMIENTO = 200, 10, 50
# n de prueba y familias por pliegue del 2e-c (job 4079): misma semilla de muestreo, mismo corpus
PUERTA = {"doc": (2071, 28), "docx": (2007, 28), "jpg": (2401, 28), "pdf": (1977, 28),
          "pptx": (2056, 28), "xls": (1974, 28), "xlsx": (2007, 28)}
# macro-F1 por pliegue SIN ponderación: (1) y (2) del 2e-c (job 4079), (3) y (5) del 2g (job 4091)
SIN_PONDERAR = {
    "1": {"doc": 0.8843, "docx": 0.8860, "jpg": 0.7987, "pdf": 0.7746, "pptx": 0.8691, "xls": 0.8835, "xlsx": 0.8846},
    "2": {"doc": 0.9337, "docx": 0.9023, "jpg": 0.8052, "pdf": 0.8131, "pptx": 0.8893, "xls": 0.9037, "xlsx": 0.9047},
    "3": {"doc": 0.9943, "docx": 1.0000, "jpg": 0.2167, "pdf": 0.9791, "pptx": 1.0000, "xls": 0.9995, "xlsx": 1.0000},
    "5": {"doc": 0.9943, "docx": 1.0000, "jpg": 0.9997, "pdf": 0.9800, "pptx": 1.0000, "xls": 0.9995, "xlsx": 1.0000},
}
NOMBRE_COL = {"1": "bytes", "2": "b+estr", "3": "+nombre", "5": "+extensión"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz")
    ap.add_argument("--por-familia", type=int, default=500)
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--prueba", action="store_true")
    args = ap.parse_args()
    global MIN_ARCHIVOS_TIPO, MIN_FAMILIAS_PRUEBA, MIN_ENTRENAMIENTO
    if args.prueba:
        args.por_familia = 60
        MIN_ARCHIVOS_TIPO, MIN_FAMILIAS_PRUEBA, MIN_ENTRENAMIENTO = 20, 2, 5

    base = _AQUI.parent / "4_resultados" if (_AQUI.parent / "4_resultados").is_dir() else _AQUI
    out = args.salida or base / ("resultados_tipos_ponderada_job" + os.environ.get("SLURM_JOB_ID", "local"))
    if out.exists() and any(out.iterdir()):
        sys.exit(f"ABORTA: {out} ya tiene resultados. Borrarla o pasar --salida.")
    out.mkdir(parents=True, exist_ok=True)
    reg = open(out / "log.txt", "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        reg.write(m + "\n")
        reg.flush()

    log("=" * 78)
    log("  VALIDACIÓN POR TIPOS CON PONDERACIÓN DE CLASES (el modelo de la validación cruzada)")
    log("=" * 78)
    log(f"  {args.raiz} · {args.por_familia}/familia · semilla de muestreo 0 · núcleos {N_JOBS}")
    log(f"  RF {HIPER}, class_weight='balanced', random_state=42, un ajuste por pliegue")

    t0 = time.time()
    Xb, Xe, Xf, Xr, y, tipos, _ = cargar(args.raiz, args.por_familia, 0, log)
    todas = sorted(set(y))
    log(f"  {len(y)} archivos · {len(todas)} familias · carga {round(time.time() - t0)} s")
    cols = {"1": Xb, "2": np.hstack([Xb, Xe]), "3": np.hstack([Xb, Xe, Xf]), "5": np.hstack([Xb, Xe, Xr])}
    cuenta = pd.Series(tipos).value_counts()
    candidatos = sorted(t for t, c in cuenta.items() if c >= MIN_ARCHIVOS_TIPO and t != "desconocido")

    obs = {tp: (int((tipos == tp).sum()), int(len(np.unique(y[tipos == tp])))) for tp in candidatos}
    if args.prueba:
        log(f"  (prueba: puerta salteada) pliegues {obs}")
    elif obs != PUERTA:
        log(f"  PUERTA FALLA: n y familias por pliegue {obs}")
        log(f"                 en el 2e-c eran          {PUERTA}")
        sys.exit(1)
    else:
        log("  PUERTA: n y familias de los siete pliegues idénticos al 2e-c (job 4079) ✔")

    filas, porfam = [], []
    log(f"\n  {'tipo':<5} {'n':>5} {'fam':>4}   " + "  ".join(f"{NOMBRE_COL[k]:>10}" for k in cols)
        + "     (macro-F1 / exactitud)")
    for tp in candidatos:
        t1 = time.time()
        te, tr = np.flatnonzero(tipos == tp), np.flatnonzero(tipos != tp)
        fams = np.unique(y[te])
        if len(fams) < MIN_FAMILIAS_PRUEBA:
            log(f"  {tp:<5} omitido (solo {len(fams)} familias en la prueba)")
            continue
        n_tr = pd.Series(y[tr]).value_counts()
        fams_ok = np.array([f for f in fams if n_tr.get(f, 0) >= MIN_ENTRENAMIENTO])
        pocas = [f for f in fams if n_tr.get(f, 0) < MIN_ENTRENAMIENTO]
        ok = np.isin(y[te], fams_ok)
        r = dict(tipo=tp, n=len(te), familias=len(fams),
                 fuera_de_la_prueba=";".join(sorted(set(todas) - set(fams))),
                 con_poco_entrenamiento=";".join(f"{f}:{int(n_tr.get(f, 0))}" for f in pocas))
        for k, X in cols.items():
            mdl = RandomForestClassifier(random_state=42, n_jobs=N_JOBS, class_weight="balanced",
                                         **HIPER).fit(X[tr], y[tr])
            yp = mdl.predict(X[te])
            r[f"acc_{k}"] = round(accuracy_score(y[te], yp), 4)
            r[f"f1_{k}"] = round(f1_score(y[te], yp, average="macro", labels=fams, zero_division=0), 4)
            r[f"f1ok_{k}"] = round(f1_score(y[te][ok], yp[ok], average="macro", labels=fams_ok,
                                            zero_division=0), 4)
            rep = classification_report(y[te], yp, labels=fams, zero_division=0, output_dict=True)
            porfam += [dict(tipo=tp, columna=k, familia=f, n_prueba=int((y[te] == f).sum()),
                            n_entrenamiento=int(n_tr.get(f, 0)), f1=round(rep[f]["f1-score"], 4))
                       for f in fams]
        filas.append(r)
        log(f"  {tp:<5} {len(te):>5} {len(fams):>4}   "
            + "  ".join(f"{r[f'f1_{k}']:.4f}/{r[f'acc_{k}']:.2f}" for k in cols)
            + f"   ({round(time.time() - t1)} s)")
        if pocas:
            log(f"        con menos de {MIN_ENTRENAMIENTO} archivos de entrenamiento en este pliegue: "
                + ", ".join(f"{f} ({int(n_tr.get(f, 0))})" for f in pocas))
        pd.DataFrame(filas).to_csv(out / "tipos_por_pliegue.csv", index=False)
        pd.DataFrame(porfam).to_csv(out / "tipos_por_familia.csv", index=False)

    rb, pf = pd.DataFrame(filas), pd.DataFrame(porfam)
    log("\n" + "=" * 78)
    log("  PROMEDIO DE LOS SIETE PLIEGUES")
    log("=" * 78)
    for k in cols:
        log(f"  ({k}) {NOMBRE_COL[k]:<11} macro-F1 {rb[f'f1_{k}'].mean():.4f} · exactitud "
            f"{rb[f'acc_{k}'].mean():.4f} · macro-F1 sin las de < {MIN_ENTRENAMIENTO} de "
            f"entrenamiento {rb[f'f1ok_{k}'].mean():.4f}")
    for a, b in (("2", "1"), ("5", "2")):
        d = rb[f"f1_{a}"] - rb[f"f1_{b}"]
        m, lo, hi = ic95(d)
        log(f"  Δ ({a})-({b}) macro-F1 por pliegue: {m:+.4f} [{lo:+.4f}; {hi:+.4f}]  "
            f"{int((d > 0).sum())}/{len(d)} a favor")

    if not args.prueba:
        log("\n  CON PONDERACIÓN menos SIN PONDERACIÓN (2e-c y 2g), macro-F1 por pliegue:")
        log(f"  {'tipo':<5} " + "  ".join(f"{NOMBRE_COL[k]:>10}" for k in cols))
        for _, r in rb.iterrows():
            log(f"  {r.tipo:<5} " + "  ".join(f"{r[f'f1_{k}'] - SIN_PONDERAR[k][r.tipo]:>+10.4f}" for k in cols))

    bm = pf[(pf.familia == "BLACKMATTER") & (pf.tipo == "jpg")]
    if len(bm):
        log("\n  BLACKMATTER en el pliegue jpg (sin ponderar, en el 2e-c: F1 0,0 en bytes y en b+estr):")
        for _, r in bm.iterrows():
            log(f"    ({r.columna}) {NOMBRE_COL[r.columna]:<11} F1 {r.f1:.4f} · {r.n_prueba} de prueba · "
                f"{r.n_entrenamiento} de entrenamiento")

    log("\n" + "-" * 78)
    log("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    log("-" * 78)
    b1 = bm[bm.columna == "1"].f1
    p1 = bool(len(b1)) and float(b1.iloc[0]) >= 0.50
    log(f"  [{'CUMPLE' if p1 else 'FALLA '}] P1 BLACKMATTER F1 >= 0,50 en jpg con bytes        "
        f"{float(b1.iloc[0]) if len(b1) else float('nan'):.4f}")
    a1 = rb.acc_1.mean()
    p2 = a1 >= 0.865
    log(f"  [{'CUMPLE' if p2 else 'FALLA '}] P2 exactitud media de (1) >= 0,865               {a1:.4f}")
    d21 = rb.f1_2 - rb.f1_1
    p3 = bool((d21 > 0).all())
    log(f"  [{'CUMPLE' if p3 else 'FALLA '}] P3 Δ (2)-(1) > 0 en los {len(rb)} pliegues            "
        f"mínimo {d21.min():+.4f} ({rb.loc[d21.idxmin(), 'tipo']})")
    p4 = bool((rb.f1_5 >= 0.99).all())
    log(f"  [{'CUMPLE' if p4 else 'FALLA '}] P4 (5) >= 0,99 en los {len(rb)} pliegues             "
        f"mínimo {rb.f1_5.min():.4f} ({rb.loc[rb.f1_5.idxmin(), 'tipo']})")
    j3 = rb[rb.tipo == "jpg"].f1_3
    p5 = bool(len(j3)) and float(j3.iloc[0]) < 0.50
    log(f"  [{'CUMPLE' if p5 else 'FALLA '}] P5 (3) sigue colapsando en jpg (< 0,50)         "
        f"{float(j3.iloc[0]) if len(j3) else float('nan'):.4f}")

    (out / "manifiesto.json").write_text(json.dumps(dict(
        fecha=str(date.today()), raiz=str(args.raiz), por_familia=args.por_familia, semilla_muestreo=0,
        hiper=HIPER, class_weight="balanced", random_state=42, min_entrenamiento=MIN_ENTRENAMIENTO,
        preregistro=dict(P1=p1, P2=bool(p2), P3=p3, P4=p4, P5=p5),
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n  Salidas en: {out}")
    reg.close()


if __name__ == "__main__":
    main()
