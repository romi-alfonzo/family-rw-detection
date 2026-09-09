#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
protocolo_logo.py -- P2-LOGO: leave-one-TEMPLATE-out, el protocolo que nadie probo en notas.

POR QUE EXISTE. P2 usa StratifiedGroupKFold con 2 pliegues «porque hay familias con 2 notas».
Ese argumento vale contra k-fold con k >= 3, pero NO contra leave-one-group-out, que funciona
con cualquier numero de grupos. Con 2 pliegues cada familia entrena con ~k/2 plantillas y se
descarta la mitad del entrenamiento en cada corte; con LOGO entrena con k-1. Para las 12
familias de 4 plantillas es 2 -> 3; para CERBER (8) es 4 -> 7.

LA GARANTIA QUE IMPORTA SE CONSERVA: la plantilla de prueba NUNCA esta en entrenamiento. Los
grupos son los mismos de P2 (casi-duplicados a coseno char 3-5 >= 0,90), asi que la nota
evaluada y todas sus casi-copias salen juntas del entrenamiento.

QUE SE CORRE, sobre el MISMO corpus y con la MISMA vista y clasificador de la corrida canonica:
  - texto solo (LinearSVC, vista combinada) bajo LOGO
  - M.6 (IOCs privados + nombre genuino -> texto) bajo LOGO, con el diccionario armado SOLO con
    el pliegue de entrenamiento (todo menos la plantilla evaluada)
Y para que la comparacion sea limpia, tambien se re-corre P2 (2 pliegues) en el mismo proceso.

VARIANZA: LOGO es determinista en la particion (99 pliegues fijos). La variacion viene del
clasificador (semilla del LinearSVC) y se reporta con N semillas; ademas se da un IC bootstrap
sobre notas para el macro-F1 puntual.

PREDICCIONES PREREGISTRADAS: ESTADO_TESIS.md, bloque «PREREGISTRO — P2-LOGO (2026-09-09)».
Se reporta COMO PROTOCOLO ADICIONAL, junto a P2, nunca en su lugar.

Uso:  python protocolo_logo.py [--n-semillas 10] [--salida CARPETA]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, N_FOLDS, UMBRAL_NEARDUP, agrupar_neardups,
                                   cargar_corpus, obtener_modelos, vectorizador)
from grafo_marcadores import extraer_marcadores

RAIZ = _AQUI.parent
DIR_NOMBRES = RAIZ / "3_datos" / "nombres_notas"
OUT_DEF = RAIZ / "4_resultados" / "resultados_protocolo_logo"


def cargar_nombres():
    nombres = {}
    aud = DIR_NOMBRES / "auditoria_nombres_corpus.csv"
    if aud.is_file():
        with open(aud, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f, delimiter=";"):
                if r["nombre_para_m2"]:
                    nombres[(r["familia"], r["archivo_corpus"])] = r["nombre_para_m2"].lower()
    js = DIR_NOMBRES / "nombres_por_nota_2026-08-23.json"
    if js.is_file():
        with open(js, encoding="utf-8") as f:
            for r in json.load(f):
                nm = (r.get("nombre_archivo") or "").strip()
                if r.get("encontrado") and nm and nm != "SIN_ARCHIVO":
                    nombres[(r["familia"], r["archivo_corpus"])] = nm.lower()
    return nombres


def dicc_privados(tr, iocs, nombres_nota, y):
    d = defaultdict(set)
    for i in tr:
        for c in iocs[i]:
            d[c].add(y[i])
        if nombres_nota[i]:
            d[("[NOMBRE]", nombres_nota[i])].add(y[i])
    for k in [k for k, v in d.items() if len(v) > 1]:
        del d[k]
    return d


def regla(i, d, iocs, nombres_nota):
    claves = set(iocs[i])
    if nombres_nota[i]:
        claves.add(("[NOMBRE]", nombres_nota[i]))
    fams = set()
    for c in claves:
        if c in d:
            fams |= d[c]
    return next(iter(fams)) if len(fams) == 1 else None


def evaluar(cv_splits, textos_arr, y, iocs, nombres_nota, seed):
    """Una pasada completa (todos los pliegues) para una semilla. Devuelve (pred_txt, pred_m6, aplica)."""
    n = len(y)
    p_txt = np.empty(n, dtype=object)
    p_m6 = np.empty(n, dtype=object)
    aplica = np.zeros(n, dtype=bool)
    for tr, te in cv_splits:
        vec = vectorizador("combinado")
        Xtr = vec.fit_transform(textos_arr[tr])
        Xte = vec.transform(textos_arr[te])
        clf = obtener_modelos(seed)["LinearSVC"]
        clf.fit(Xtr, y[tr])
        pt = clf.predict(Xte)
        p_txt[te] = pt
        d = dicc_privados(tr, iocs, nombres_nota, y)
        for k, i in enumerate(te):
            r = regla(i, d, iocs, nombres_nota)
            if r is not None:
                p_m6[i] = r
                aplica[i] = True
            else:
                p_m6[i] = pt[k]
    return p_txt, p_m6, aplica


def ic95(d):
    n = len(d)
    m = float(np.mean(d))
    s = float(np.std(d, ddof=1)) if n > 1 else 0.0
    h = t_dist.ppf(0.975, n - 1) * (s / np.sqrt(n)) if n > 1 else 0.0
    return m, m - h, m + h


def bootstrap_macro_f1(y, pred, familias, B=2000, seed=7):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals.append(f1_score(y[idx], pred[idx], average="macro", labels=familias,
                             zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=10)
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  P2-LOGO -- LEAVE-ONE-TEMPLATE-OUT (protocolo adicional, junto a P2)")
    print("=" * 78)
    textos, y, archivos, _ = cargar_corpus(CORPUS_DIR)
    y = np.asarray(y)
    grupos, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    grupos = np.asarray(grupos)
    familias = np.unique(y)
    textos_arr = np.array(textos, dtype=object)
    n = len(textos)
    iocs = [set(extraer_marcadores(t)) for t in textos]
    nom = cargar_nombres()
    nombres_nota = [nom.get((f, Path(a).name)) for f, a in zip(y, archivos)]
    print(f"Notas: {n} | Familias: {len(familias)} | Plantillas (grupos): {len(set(grupos))}")
    print(f"Semillas del clasificador: {args.n_semillas}\n")

    # LOGO: particion fija (una por plantilla)
    logo_splits = list(LeaveOneGroupOut().split(textos_arr, y, groups=grupos))
    print(f"LOGO: {len(logo_splits)} pliegues (uno por plantilla)")

    res = {"P2": {"txt": [], "m6": [], "ap": []}, "LOGO": {"txt": [], "m6": [], "ap": []}}
    for s in range(args.n_semillas):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
        p2_splits = list(cv.split(textos_arr, y, groups=grupos))
        for nombre, splits in (("P2", p2_splits), ("LOGO", logo_splits)):
            pt, pm, ap_ = evaluar(splits, textos_arr, y, iocs, nombres_nota, s)
            res[nombre]["txt"].append(pt)
            res[nombre]["m6"].append(pm)
            res[nombre]["ap"].append(ap_)
        print(f"  semilla {s}: P2 txt {f1_score(y, res['P2']['txt'][-1], average='macro', zero_division=0):.4f} "
              f"m6 {f1_score(y, res['P2']['m6'][-1], average='macro', zero_division=0):.4f} | "
              f"LOGO txt {f1_score(y, res['LOGO']['txt'][-1], average='macro', zero_division=0):.4f} "
              f"m6 {f1_score(y, res['LOGO']['m6'][-1], average='macro', zero_division=0):.4f}")

    def met(preds):
        return (np.array([f1_score(y, p, average="macro", zero_division=0) for p in preds]),
                np.array([accuracy_score(y, p) for p in preds]),
                np.array([balanced_accuracy_score(y, p) for p in preds]),
                np.array([precision_recall_fscore_support(y, p, labels=familias,
                                                          zero_division=0)[2] for p in preds]))

    filas, filas_fam = [], []
    guard = {}
    for prot in ("P2", "LOGO"):
        for capa in ("txt", "m6"):
            f1s, accs, bals, fams = met(res[prot][capa])
            guard[(prot, capa)] = (f1s, fams)
            cob = float(np.mean([a.mean() for a in res[prot]["ap"]])) if capa == "m6" else 0.0
            if prot == "LOGO":
                # particion determinista y LinearSVC convexo: las semillas dan lo mismo.
                # La incertidumbre honesta es el bootstrap sobre NOTAS.
                lo_b, hi_b = bootstrap_macro_f1(y, res[prot][capa][0], familias)
                ic_txt = f"boot.notas [{lo_b:.4f}; {hi_b:.4f}]"
            else:
                # P2: la incertidumbre es entre semillas (particiones distintas).
                _, lo_b, hi_b = ic95(f1s)
                ic_txt = f"semillas [{lo_b:.4f}; {hi_b:.4f}]"
            filas.append(dict(protocolo=prot, capa="texto solo" if capa == "txt" else "M.6",
                              f1_macro=round(float(f1s.mean()), 4),
                              f1_macro_sd=round(float(f1s.std(ddof=1)), 4) if len(f1s) > 1 else 0.0,
                              ic95=ic_txt,
                              exactitud=round(float(accs.mean()), 4),
                              exactitud_balanceada=round(float(bals.mean()), 4),
                              cobertura_regla=round(cob, 4),
                              supera_050=("SI, IC bootstrap excluye" if lo_b > 0.50 else
                                          ("media si, IC toca" if f1s.mean() > 0.50 else "NO"))))
            for j, fam in enumerate(familias):
                filas_fam.append(dict(protocolo=prot, capa=capa, familia=fam,
                                      f1=round(float(fams[:, j].mean()), 4)))

    # Deltas pareados por semilla: LOGO - P2, misma capa
    deltas = []
    for capa in ("txt", "m6"):
        d = guard[("LOGO", capa)][0] - guard[("P2", capa)][0]
        m, lo, hi = ic95(d)
        deltas.append(dict(comparacion=f"LOGO - P2 ({'texto' if capa=='txt' else 'M.6'})",
                           delta=round(m, 4), ic95_bajo=round(lo, 4), ic95_alto=round(hi, 4),
                           semillas_positivas=f"{int((d>0).sum())}/{len(d)}"))
    d = guard[("LOGO", "m6")][0] - guard[("LOGO", "txt")][0]
    m, lo, hi = ic95(d)
    deltas.append(dict(comparacion="M.6 - texto, bajo LOGO", delta=round(m, 4),
                       ic95_bajo=round(lo, 4), ic95_alto=round(hi, 4),
                       semillas_positivas=f"{int((d>0).sum())}/{len(d)}"))

    pd.DataFrame(filas).to_csv(OUT / "logo_resumen.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(filas_fam).to_csv(OUT / "logo_por_familia.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(deltas).to_csv(OUT / "logo_deltas.csv", index=False, encoding="utf-8-sig")

    print("\n=== RESULTADO: mismo corpus, misma vista, mismo clasificador; cambia el protocolo ===")
    print(f"{'protocolo':<8}{'capa':<12}{'macro-F1':>10}{'±sd':>8}{'IC 95 %':>30}{'exact.':>8}{'bal.':>8}{'cob':>7}  ¿>0,50?")
    for r in filas:
        print(f"{r['protocolo']:<8}{r['capa']:<12}{r['f1_macro']:>10.4f}{r['f1_macro_sd']:>8.4f}"
              f"{r['ic95']:>30}{r['exactitud']:>8.4f}{r['exactitud_balanceada']:>8.4f}"
              f"{r['cobertura_regla']:>7.3f}  {r['supera_050']}")
    print("\n=== DELTAS pareados por semilla ===")
    for d in deltas:
        print(f"  {d['comparacion']:<28} Δ {d['delta']:+.4f} [{d['ic95_bajo']:+.4f}; {d['ic95_alto']:+.4f}] {d['semillas_positivas']}")

    ff = pd.DataFrame(filas_fam)
    piv = ff.pivot_table(index="familia", columns=["protocolo", "capa"], values="f1")
    piv.columns = [f"{p_}_{c_}" for p_, c_ in piv.columns]   # aplanar: evita el bug de formato
    piv["ganancia_txt"] = piv["LOGO_txt"] - piv["P2_txt"]
    print("\n=== familias que MAS ganan con LOGO (texto solo) ===")
    for fam, r in piv.sort_values("ganancia_txt", ascending=False).head(8).iterrows():
        print(f"  {fam:<14} P2 {r['P2_txt']:.3f} -> LOGO {r['LOGO_txt']:.3f}  (+{r['ganancia_txt']:.3f})")
    print("\n=== y las que NO se mueven (deben ser las de 1 plantilla y las heterogeneas) ===")
    for fam, r in piv.sort_values("ganancia_txt").head(5).iterrows():
        print(f"  {fam:<14} P2 {r['P2_txt']:.3f} -> LOGO {r['LOGO_txt']:.3f}  ({r['ganancia_txt']:+.3f})")
    piv.to_csv(OUT / "logo_por_familia_pivot.csv", encoding="utf-8-sig")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
