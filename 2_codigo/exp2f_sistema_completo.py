#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""Exp. 2f -- EL SISTEMA COMPLETO del frente de archivos: todas las técnicas apiladas.

POR QUÉ
-------
Las técnicas del frente de archivos no son alternativas sino capas de una misma secuencia:
cada una agrega información que las anteriores no tienen. Criterio de Romina (2026-09-28):
«no son técnicas separadas, son una secuencia hasta encontrar la forma más óptima de
clasificar». Hasta hoy cada capa se midió de a pares contra los bytes:

    bytes + forma del nombre ............ 0,9998 de macro-F1 (Exp. 2d)
    bytes + rasgos estructurales ........ 0,9359 de macro-F1 (Exp. 2e)
    bytes + estructura + forma .......... NUNCA SE MIDIÓ

Este experimento mide el sistema que las combina, y además:
  - guarda el F1 POR FAMILIA en las cinco semillas (no solo en una), para que el estado de
    cada familia difícil deje de depender de una única corrida;
  - valida el sistema sobre tipos de documento nunca vistos, con los mismos pliegues que el
    Exp. 2c y el Exp. 2e-c, para que el número final tenga el mismo respaldo que los otros.

COLUMNAS (misma partición, delta pareado por semilla)
  (1) bytes .............................. 512+512 posicional, la base del Exp. 2c
  (2) bytes + estructura ................. el Exp. 2e (44 rasgos)
  (3) bytes + estructura + forma ......... el sistema propuesto (+18 rasgos de forma del nombre)
  (4) todo, con extensión literal ........ (3) + la extensión como variable categórica

La columna (4) se mide para que la decisión sobre la extensión literal salga de los datos:
en el Exp. 2d, sumada a los bytes, rindió MENOS que la forma (0,9699 contra 0,9998) porque
novecientas columnas dispersas diluyen a las demás. Si acá tampoco suma, queda afuera por
evidencia y no por criterio.

LIMITACIÓN QUE VA PEGADA AL RESULTADO, NO LO EXCLUYE: en NapierOne cada familia es una sola
campaña. Afecta a todo el frente ---la tesis ya lo dice del 0,912 de solo bytes---, y la capa
del nombre es la más expuesta, porque el esquema de renombrado puede cambiar entre campañas.

PREREGISTRO -- escrito y commiteado ANTES de correr (2026-09-28)
----------------------------------------------------------------
F1. La columna (3) alcanza macro-F1 >= 0,9990 en validación cruzada. Razón: bytes + forma ya
    da 0,9998; agregar estructura no puede empeorarlo materialmente.
F2. La columna (4) no mejora a la (3): |Δ(4)-(3)| <= 0,0010 de macro-F1.
F3. Con la columna (3), las 30 familias quedan en F1 medio >= 0,99 sobre las cinco semillas,
    incluidas NOTPETYA, JIGSAW, CRYPTOLOCKER y DARKSIDE.
F4. Bajo tipo de documento no visto, el Δ (3)-(2) es POSITIVO en los siete pliegues. Si falla,
    la forma del nombre codifica en parte el tipo del documento de origen (el largo de la
    extensión «pdf» contra «docx», por ejemplo) y hay que declararlo junto al número.

Reutiliza la lectura y los rasgos de exp2e_estructura_bytes.py y la forma del nombre y la
extensión de exp2d_nombre_extension.py. No toca ningún canónico.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
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
from exp2d_nombre_extension import extension_literal, forma_del_nombre  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HIPER = dict(n_estimators=300, max_depth=20, min_samples_leaf=2, max_features=0.3)
DIFICILES = ["NOTPETYA", "JIGSAW", "CRYPTOLOCKER", "DARKSIDE", "WASTEDLOCKER", "SUNCRYPT"]
MIN_ARCHIVOS_TIPO, MIN_FAMILIAS_PRUEBA = 200, 10


def tipo_documento(nombre):
    m = re.match(r"^\d+-([a-z0-9]+)", nombre.lower())
    return m.group(1) if m else "desconocido"


def cargar(raiz, por_familia, seed, log=print):
    """Bytes, rasgos estructurales, nombre y tipo de documento de los MISMOS archivos."""
    rng = np.random.default_rng(seed)
    Xb, Xe, y, nombres, tipos = [], [], [], [], []
    sospechosos = defaultdict(list)
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
            m = _magia(head)
            if m:
                sospechosos[fam].append((p.name, m))
            Xb.append(head + cola)
            Xe.append(rasgos_estructura(cabp, colap, medios, tam))
            y.append(fam)
            nombres.append(p.name)
            tipos.append(tipo_documento(p.name))
    if sospechosos:
        log("  ⚠ archivos con firma en claro: " + ", ".join(
            f"{f} {len(v)}" for f, v in sorted(sospechosos.items())))
    Xb = np.frombuffer(b"".join(Xb), dtype=np.uint8).reshape(len(Xb), N_HEAD + N_TAIL).astype(np.float32)
    Xe = np.nan_to_num(np.asarray(Xe, dtype=np.float32))
    Xf = forma_del_nombre(nombres)
    Xx, _, _ = extension_literal(nombres)
    return Xb, Xe, Xf, Xx, np.array(y), np.array(tipos)


def columnas(Xb, Xe, Xf, Xx):
    return {
        "1_bytes": Xb,
        "2_bytes_estructura": np.hstack([Xb, Xe]),
        "3_bytes_estructura_forma": np.hstack([Xb, Xe, Xf]),
        "4_todo_con_extension": np.hstack([Xb, Xe, Xf, Xx]),
    }


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
    ap.add_argument("--sin-tipos", action="store_true", help="omite la validación por tipos")
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--prueba", action="store_true")
    args = ap.parse_args()
    global MIN_ARCHIVOS_TIPO, MIN_FAMILIAS_PRUEBA
    if args.prueba:
        args.por_familia, args.semillas, args.folds = 60, "0,1", 2
        MIN_ARCHIVOS_TIPO, MIN_FAMILIAS_PRUEBA = 20, 2
    semillas = [int(s) for s in args.semillas.split(",") if s.strip()]

    base = _AQUI.parent / "4_resultados" if (_AQUI.parent / "4_resultados").is_dir() else _AQUI
    out = args.salida or base / ("resultados_exp2f_job" + os.environ.get("SLURM_JOB_ID", "local"))
    if out.exists() and any(out.iterdir()):
        sys.exit(f"ABORTA: {out} ya tiene resultados. Borrarla o pasar --salida.")
    out.mkdir(parents=True, exist_ok=True)
    reg = open(out / "log.txt", "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        reg.write(m + "\n")
        reg.flush()

    log("=" * 78)
    log("  EXP. 2f -- SISTEMA COMPLETO: bytes + estructura + nombre")
    log("=" * 78)
    log(f"  {args.raiz} · {args.por_familia}/familia · semillas {semillas} · {args.folds} pliegues")
    log(f"  RF {HIPER} · núcleos {N_JOBS}")

    filas, porfam, datos_s0 = [], [], None
    for s in semillas:
        t0 = time.time()
        log(f"\n--- semilla {s} ---")
        Xb, Xe, Xf, Xx, y, tipos = cargar(args.raiz, args.por_familia, s, log)
        if s == semillas[0]:
            datos_s0 = (Xb, Xe, Xf, Xx, y, tipos)
        cv = StratifiedKFold(args.folds, shuffle=True, random_state=s)
        for nombre, X in columnas(Xb, Xe, Xf, Xx).items():
            clf = RandomForestClassifier(random_state=s, n_jobs=1, class_weight="balanced", **HIPER)
            yp = cross_val_predict(clf, X, y, cv=cv, n_jobs=N_JOBS)
            f = dict(semilla=s, columna=nombre, n_caracteristicas=X.shape[1],
                     accuracy=round(accuracy_score(y, yp), 4),
                     balanced_accuracy=round(balanced_accuracy_score(y, yp), 4),
                     f1_macro=round(f1_score(y, yp, average="macro", zero_division=0), 4))
            filas.append(f)
            rep = classification_report(y, yp, zero_division=0, output_dict=True)
            for fam in sorted(set(y)):
                porfam.append(dict(semilla=s, columna=nombre, familia=fam,
                                   f1=round(rep[fam]["f1-score"], 4)))
            log(f"    {nombre:<26} ({X.shape[1]:>4} carac.)  exactitud {f['accuracy']:.4f} | "
                f"macro-F1 {f['f1_macro']:.4f}")
        log(f"  ({round(time.time() - t0)} s)")
        pd.DataFrame(filas).to_csv(out / "cv_por_semilla.csv", index=False)
        pd.DataFrame(porfam).to_csv(out / "cv_por_familia_y_semilla.csv", index=False)

    df = pd.DataFrame(filas)
    pf = pd.DataFrame(porfam)
    log("\n" + "=" * 78)
    log("  (A) VALIDACIÓN CRUZADA -- media ± desvío sobre las semillas")
    log("=" * 78)
    res = df.groupby("columna")[["accuracy", "f1_macro"]].agg(["mean", "std"]).round(4)
    log(res.to_string())
    res.to_csv(out / "cv_resumen.csv")

    log("\n  Delta pareado por semilla (macro-F1):")
    piv = df.pivot(index="semilla", columns="columna", values="f1_macro")
    deltas = {}
    for a, b in (("2_bytes_estructura", "1_bytes"), ("3_bytes_estructura_forma", "2_bytes_estructura"),
                 ("3_bytes_estructura_forma", "1_bytes"), ("4_todo_con_extension", "3_bytes_estructura_forma")):
        m, lo, hi = ic95(piv[a] - piv[b])
        deltas[f"{a}-{b}"] = m
        log(f"    {a:<26} − {b:<26} {m:+.4f} [{lo:+.4f}; {hi:+.4f}]  "
            f"{int(((piv[a] - piv[b]) > 0).sum())}/{len(piv)}")

    log("\n  F1 POR FAMILIA, media ± desvío sobre las semillas (las difíciles primero):")
    fam_res = pf.groupby(["familia", "columna"]).f1.agg(["mean", "std"]).unstack("columna")
    orden = [f for f in DIFICILES if f in fam_res.index] + \
            sorted(f for f in fam_res.index if f not in DIFICILES)
    cols_m = [("mean", c) for c in ("1_bytes", "2_bytes_estructura", "3_bytes_estructura_forma")]
    tabla = fam_res.loc[orden, cols_m].round(4)
    tabla.columns = ["bytes", "b+estruct", "b+estr+forma"]
    tabla["desvío (3)"] = fam_res.loc[orden, ("std", "3_bytes_estructura_forma")].round(4)
    log(tabla.to_string())
    fam_res.to_csv(out / "cv_por_familia_resumen.csv")
    bajo = tabla[tabla["b+estr+forma"] < 0.99].index.tolist()
    log(f"\n  Familias con F1 medio < 0,99 en el sistema (3): {bajo if bajo else 'ninguna'}")

    # ---------------------------------------------------------------- (B) tipos no vistos
    resB = None
    if not args.sin_tipos and datos_s0 is not None:
        log("\n" + "=" * 78)
        log("  (B) DEJAR-UN-TIPO-FUERA (semilla de muestreo 0, un ajuste por pliegue)")
        log("=" * 78)
        Xb, Xe, Xf, Xx, y, tipos = datos_s0
        cols = columnas(Xb, Xe, Xf, Xx)
        cuenta = pd.Series(tipos).value_counts()
        candidatos = sorted(t for t, c in cuenta.items() if c >= MIN_ARCHIVOS_TIPO and t != "desconocido")
        rows = []
        for t in candidatos:
            te, tr = np.flatnonzero(tipos == t), np.flatnonzero(tipos != t)
            fams = np.unique(y[te])
            if len(fams) < MIN_FAMILIAS_PRUEBA:
                continue
            r = dict(tipo=t, n=len(te), familias=len(fams))
            for nombre in ("2_bytes_estructura", "3_bytes_estructura_forma"):
                m = RandomForestClassifier(random_state=42, n_jobs=N_JOBS, **HIPER).fit(cols[nombre][tr], y[tr])
                yp = m.predict(cols[nombre][te])
                r[f"f1_{nombre}"] = round(f1_score(y[te], yp, average="macro", labels=fams, zero_division=0), 4)
            r["delta"] = round(r["f1_3_bytes_estructura_forma"] - r["f1_2_bytes_estructura"], 4)
            rows.append(r)
            log(f"  {t:<6} n={len(te):>5} fam={len(fams):>3}   b+estructura {r['f1_2_bytes_estructura']:.4f}"
                f"   b+estr+forma {r['f1_3_bytes_estructura_forma']:.4f}   Δ {r['delta']:+.4f}")
        if rows:
            resB = pd.DataFrame(rows)
            resB.to_csv(out / "tipos_por_pliegue.csv", index=False)
            m, lo, hi = ic95(resB.delta)
            log(f"\n  PROMEDIO  b+estructura {resB.f1_2_bytes_estructura.mean():.4f} · "
                f"b+estr+forma {resB.f1_3_bytes_estructura_forma.mean():.4f}")
            log(f"  Δ (3)-(2) por pliegue: {m:+.4f} [{lo:+.4f}; {hi:+.4f}]  "
                f"{int((resB.delta > 0).sum())}/{len(resB)} pliegues a favor")

    # ---------------------------------------------------------------- veredicto
    log("\n" + "-" * 78)
    log("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    log("-" * 78)
    f3 = res.loc["3_bytes_estructura_forma", ("f1_macro", "mean")]
    v1 = f3 >= 0.9990
    log(f"  [{'CUMPLE' if v1 else 'FALLA '}] F1 sistema (3) macro-F1 >= 0,9990        {f3:.4f}")
    d43 = deltas["4_todo_con_extension-3_bytes_estructura_forma"]
    v2 = abs(d43) <= 0.0010
    log(f"  [{'CUMPLE' if v2 else 'FALLA '}] F2 la extensión literal no suma a (3)     Δ {d43:+.4f}")
    v3 = not bajo
    log(f"  [{'CUMPLE' if v3 else 'FALLA '}] F3 las 30 familias con F1 medio >= 0,99   "
        f"{'todas' if v3 else 'bajo 0,99: ' + ', '.join(bajo)}")
    if resB is not None:
        v4 = bool((resB.delta > 0).all())
        log(f"  [{'CUMPLE' if v4 else 'FALLA '}] F4 Δ (3)-(2) > 0 en los {len(resB)} pliegues      "
            f"mínimo {resB.delta.min():+.4f} ({resB.loc[resB.delta.idxmin(), 'tipo']})")
    else:
        v4 = None

    (out / "manifiesto.json").write_text(json.dumps(dict(
        fecha=str(date.today()), raiz=str(args.raiz), por_familia=args.por_familia,
        semillas=semillas, folds=args.folds, hiper=HIPER,
        preregistro=dict(F1=bool(v1), F2=bool(v2), F3=bool(v3), F4=v4),
        limitacion="una campaña por familia: afecta a todo el frente; la capa del nombre es la más expuesta",
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n  Salidas en: {out}")
    reg.close()


if __name__ == "__main__":
    main()
