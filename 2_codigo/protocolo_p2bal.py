#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
protocolo_p2bal.py -- P2bal: el protocolo P2 con el reparto de plantillas ARREGLADO.

QUE PROBLEMA RESUELVE
P2 (el protocolo canonico del frente de notas) parte las plantillas en 2 pliegues con
StratifiedGroupKFold. Ese repartidor optimiza un objetivo global de balance de clases y NO
garantiza que cada familia tenga al menos una plantilla en entrenamiento en cada pliegue. En la
practica deja ~3,9 familias por pliegue sin ningun ejemplo para aprender: esas familias sacan
F1 = 0 forzado, no porque el metodo falle sino porque no se les dio nada con que entrenar. Esos
ceros entran al promedio macro y hunden la cifra.

P2bal reparte las plantillas DENTRO de cada familia: las familias con k>=2 plantillas ponen al
menos una en cada pliegue. Mismo numero de pliegues, mismo tamano de entrenamiento, y la MISMA
garantia que P2: el corte es por plantilla, de modo que la plantilla de prueba nunca esta en
entrenamiento. Lo unico que cambia es que el reparto deja de sortear ceros estructurales.

QUE NO ES
No es un protocolo mas permisivo. No entrena con mas datos (se verifica: plantillas de train por
pliegue practicamente iguales). No mira la nota de prueba para decidir el reparto. Las familias
de UNA sola plantilla (BADRABBIT, CRYPTOLOCKER) siguen dando 0: eso es estructural del corpus y
ningun protocolo honesto lo arregla.

=============================================================================================
PREREGISTRO -- escrito y COMMITEADO ANTES de correr (por eso vive en el docstring y no en un
documento de estado: el commit deja la marca temporal verificable en git). Lo que se cumpla y lo
que falle se reporta igual.

P1. P2bal texto da macro-F1 >= 0,60 y su IC 95 % entre semillas excluye 0,50 por arriba.
    (Medicion previa independiente, 20 semillas: 0,6651 [0,650; 0,680].)
P2. Familias sin plantilla de entrenamiento por pliegue: EXACTAMENTE 2 bajo P2bal (las de
    plantilla unica) frente a ~3,9 bajo P2.
P3. Plantillas de entrenamiento por pliegue: la diferencia entre P2 y P2bal es <= 1,0. Si P2bal
    entrenara con materialmente mas, la comparacion estaria confundida y la conclusion no valdria.
P4. Las 5 familias de 2 plantillas (SUNCRYPT, CUBA, NETWALKER, BLACKMATTER, DARKSIDE) suben cada
    una >= +0,30 de F1 al pasar de P2 a P2bal.
P5. El desvio entre semillas es MENOR bajo P2bal que bajo P2 (que es 0,0752): al quitar la
    loteria de ceros estructurales, la varianza tiene que bajar.
P6. P2bal texto < LOGO (0,6747), porque LOGO entrena con ~98 plantillas y P2bal con ~49,5. Si
    P2bal saliera IGUAL o MAYOR que LOGO, la explicacion por tamano de entrenamiento seria falsa
    y habria que escribirlo.
P7. CONTROL EXTERNO (puerta de entrada): la columna P2 de este script tiene que reproducir el
    evaluador canonico a 50 semillas -- texto 0,4593 y M.6 0,5191 (_log_m6_149.txt). Si no
    reproduce, el script ABORTA y no se reporta nada.

LIMITACION QUE P2bal HEREDA Y NO ARREGLA
La definicion de "plantilla" es coseno de caracteres > 0,90, que no detecta CONTENCION (una nota
contenida dentro de otra). La revision independiente del 2026-09-17 midio que con un criterio de
contencion >= 0,8 el corpus pasa de 99 a 81 plantillas y de 28 a 21 familias evaluables. Toda
cifra de este script es "plantilla no vista SEGUN EL CRITERIO DE COSENO 0,90" y hay que decirlo
asi. Ver 6_notas_trabajo/REVISION_LOGO_2026-09-17_informe.md.
=============================================================================================

METRICAS: macro-F1, exactitud, exactitud balanceada, MCC, CV(F1) entre semillas, F1 por familia,
y la cifra sobre las 28 familias evaluables ademas de sobre las 30. Delta pareado P2bal - P2 por
semilla con IC 95 % t-Student.

Uso:  python protocolo_p2bal.py [--n-semillas 50] [--salida CARPETA]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, N_FOLDS, UMBRAL_NEARDUP, agrupar_neardups,
                                   cargar_corpus)
from grafo_marcadores import extraer_marcadores
from protocolo_logo import cargar_nombres, evaluar

RAIZ = _AQUI.parent
OUT_DEF = RAIZ / "4_resultados" / "resultados_protocolo_p2bal"

# Control externo P7. Fuente: 4_resultados/_log_m6_149.txt (cascada canonica, 149 notas, 50 semillas).
CANON_P2_TXT, CANON_P2_M6, TOL_CANON = 0.4593, 0.5191, 0.02


def split_p2bal(y, grupos, familias, rng, n_folds=N_FOLDS):
    """Reparte las PLANTILLAS en n_folds pliegues balanceando DENTRO de cada familia.

    Recorre las familias de menor a mayor cantidad de plantillas: las chicas eligen primero,
    que es donde el reparto es critico (una familia de 2 plantillas tiene una sola forma de
    quedar bien repartida). Cada plantilla libre va al pliegue donde esa familia tiene menos,
    y los empates se sortean con rng, de modo que semillas distintas dan reparticiones
    distintas y el promedio entre semillas sigue teniendo sentido.

    Una plantilla que contiene notas de DOS familias (los grupos mixtos del corpus) se asigna
    una sola vez, cuando la trata la primera familia; la segunda la ve como ya ubicada y
    equilibra con las que le quedan libres. Asi ninguna nota aparece en dos pliegues.

    Devuelve la lista [(train_idx, test_idx), ...], con el corte SIEMPRE por plantilla: todas
    las notas de una plantilla caen del mismo lado.
    """
    fold_de = {}
    por_fam = {f: sorted(set(grupos[y == f])) for f in familias}
    for f in sorted(familias, key=lambda f: (len(por_fam[f]), f)):
        cuenta = Counter({k: 0 for k in range(n_folds)})
        cuenta.update(fold_de[g] for g in por_fam[f] if g in fold_de)
        libres = [g for g in por_fam[f] if g not in fold_de]
        libres = [libres[i] for i in rng.permutation(len(libres))]
        for g in libres:
            minimo = min(cuenta.values())
            candidatos = [k for k in range(n_folds) if cuenta[k] == minimo]
            k = int(candidatos[rng.integers(len(candidatos))])
            fold_de[g] = k
            cuenta[k] += 1
    folds = np.array([fold_de[g] for g in grupos])
    return [(np.where(folds != k)[0], np.where(folds == k)[0]) for k in range(n_folds)]


def controles_particion(splits, y, grupos, familias):
    """Verifica la garantia y cuenta ceros estructurales. Devuelve (violaciones, fam_sin_train
    por pliegue, plantillas de train por pliegue)."""
    viol, sin_train, plant_train = 0, [], []
    for tr, te in splits:
        g_tr, g_te = set(grupos[tr]), set(grupos[te])
        viol += len(g_tr & g_te)                      # una plantilla en ambos lados = fuga
        fam_tr = set(y[tr])
        sin_train.append(sum(1 for f in familias if f not in fam_tr))
        plant_train.append(len(g_tr))
    return viol, float(np.mean(sin_train)), float(np.mean(plant_train))


def ic_t(v):
    v = np.asarray(v, float)
    m, n = float(np.mean(v)), len(v)
    s = float(np.std(v, ddof=1)) if n > 1 else 0.0
    h = t_dist.ppf(0.975, n - 1) * (s / np.sqrt(n)) if n > 1 else 0.0
    return m, m - h, m + h, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=50)
    ap.add_argument("--sin-puerta", action="store_true",
                    help="NO usar salvo depuracion: saltea el control externo P7.")
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  P2bal -- P2 CON EL REPARTO DE PLANTILLAS ARREGLADO")
    print("=" * 78)
    textos, y, archivos, _ = cargar_corpus(CORPUS_DIR)
    grupos, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    grupos = np.asarray(grupos)
    textos_arr = np.array(textos, dtype=object)
    y = np.asarray(y)
    familias = np.unique(y)
    iocs = [set(extraer_marcadores(t)) for t in textos]
    nom = cargar_nombres()
    nombres_nota = [nom.get((f, Path(a).name)) for f, a in zip(y, archivos)]

    ppf = {f: len(set(grupos[y == f])) for f in familias}
    fam_una = sorted(f for f in familias if ppf[f] == 1)
    fam_dos = sorted(f for f in familias if ppf[f] == 2)
    evaluables = [f for f in familias if ppf[f] >= 2]
    print(f"Notas: {len(y)} | Familias: {len(familias)} | Plantillas: {len(set(grupos))}")
    print(f"Familias de 1 plantilla (cero estructural inevitable): {fam_una}")
    print(f"Familias de 2 plantillas ({len(fam_dos)}): {fam_dos}")
    print(f"Semillas: {args.n_semillas}\n")

    res = {p: {"txt": [], "m6": [], "sin_train": [], "plant_train": [], "viol": 0}
           for p in ("P2", "P2bal")}
    f1fam = {p: {"txt": [], "m6": []} for p in ("P2", "P2bal")}

    for s in range(args.n_semillas):
        rng = np.random.default_rng(20_000 + s)
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
        splits = {"P2": list(cv.split(textos_arr, y, groups=grupos)),
                  "P2bal": split_p2bal(y, grupos, familias, rng)}
        for p in ("P2", "P2bal"):
            v, st, pt_ = controles_particion(splits[p], y, grupos, familias)
            res[p]["viol"] += v
            res[p]["sin_train"].append(st)
            res[p]["plant_train"].append(pt_)
            p_txt, p_m6, _ = evaluar(splits[p], textos_arr, y, iocs, nombres_nota, s)
            for capa, pred in (("txt", p_txt), ("m6", p_m6)):
                res[p][capa].append(dict(
                    f1=f1_score(y, pred, average="macro", labels=familias, zero_division=0),
                    acc=accuracy_score(y, pred),
                    bal=balanced_accuracy_score(y, pred),
                    mcc=matthews_corrcoef(y, pred),
                    f1_eval=f1_score(y, pred, average="macro", labels=evaluables,
                                     zero_division=0)))
                f1fam[p][capa].append(precision_recall_fscore_support(
                    y, pred, labels=familias, zero_division=0)[2])
        if (s + 1) % 10 == 0:
            print(f"  {s+1}/{args.n_semillas} semillas")

    # ---------------- control externo P7 (puerta de entrada) ----------------
    p2_txt = float(np.mean([d["f1"] for d in res["P2"]["txt"]]))
    p2_m6 = float(np.mean([d["f1"] for d in res["P2"]["m6"]]))
    print("\n" + "-" * 78)
    print("  CONTROL EXTERNO P7 -- la columna P2 debe reproducir el evaluador canonico")
    print("-" * 78)
    print(f"  texto : {p2_txt:.4f}  vs canonico {CANON_P2_TXT}  (dif {abs(p2_txt-CANON_P2_TXT):.4f})")
    print(f"  M.6   : {p2_m6:.4f}  vs canonico {CANON_P2_M6}  (dif {abs(p2_m6-CANON_P2_M6):.4f})")
    ok = abs(p2_txt - CANON_P2_TXT) <= TOL_CANON and abs(p2_m6 - CANON_P2_M6) <= TOL_CANON
    if not ok and not args.sin_puerta:
        sys.exit("ABORTADO (P7): la columna P2 no reproduce el evaluador canonico. No se reporta nada.")
    print("  OK\n" if ok else "  FUERA DE TOLERANCIA (se sigue por --sin-puerta)\n")

    # ---------------- resumen ----------------
    filas = []
    for p in ("P2", "P2bal"):
        for capa in ("txt", "m6"):
            v = res[p][capa]
            m, lo, hi, sd = ic_t([d["f1"] for d in v])
            me, loe, hie, _ = ic_t([d["f1_eval"] for d in v])
            filas.append(dict(
                protocolo=p, capa="texto solo" if capa == "txt" else "M.6 (cascada)",
                f1_macro_30=round(m, 4), sd=round(sd, 4),
                ic95=f"[{lo:.4f}; {hi:.4f}]",
                cv_f1=round(sd / m, 4) if m else np.nan,
                f1_macro_28_evaluables=round(me, 4), ic95_28=f"[{loe:.4f}; {hie:.4f}]",
                exactitud=round(float(np.mean([d["acc"] for d in v])), 4),
                exact_balanceada=round(float(np.mean([d["bal"] for d in v])), 4),
                mcc=round(float(np.mean([d["mcc"] for d in v])), 4),
                supera_050="SI" if lo > 0.50 else ("media si, IC toca" if m > 0.50 else "NO")))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "p2bal_resumen.csv", index=False, encoding="utf-8-sig")

    print("=== RESULTADO: mismo corpus, misma vista, mismo clasificador; cambia el REPARTO ===")
    print(df.to_string(index=False))

    # ---------------- deltas pareados ----------------
    dfilas = []
    for capa, etq in (("txt", "texto"), ("m6", "M.6")):
        d = np.array([b["f1"] for b in res["P2bal"][capa]]) - \
            np.array([a["f1"] for a in res["P2"][capa]])
        m, lo, hi, _ = ic_t(d)
        dfilas.append(dict(comparacion=f"P2bal - P2 ({etq})", delta=round(m, 4),
                           ic95_bajo=round(lo, 4), ic95_alto=round(hi, 4),
                           semillas_positivas=f"{int((d > 0).sum())}/{len(d)}"))
    dd = pd.DataFrame(dfilas)
    dd.to_csv(OUT / "p2bal_deltas.csv", index=False, encoding="utf-8-sig")
    print("\n=== DELTAS pareados por semilla ===")
    for r in dfilas:
        print(f"  {r['comparacion']:<24} D {r['delta']:+.4f} "
              f"[{r['ic95_bajo']:+.4f}; {r['ic95_alto']:+.4f}] {r['semillas_positivas']}")

    # ---------------- por familia ----------------
    ffilas = []
    for j, f in enumerate(familias):
        fila = dict(familia=f, n_plantillas=ppf[f])
        for p in ("P2", "P2bal"):
            for capa in ("txt", "m6"):
                fila[f"{p}_{capa}"] = round(float(np.mean([a[j] for a in f1fam[p][capa]])), 4)
        fila["delta_txt"] = round(fila["P2bal_txt"] - fila["P2_txt"], 4)
        ffilas.append(fila)
    dfam = pd.DataFrame(ffilas).sort_values("delta_txt", ascending=False)
    dfam.to_csv(OUT / "p2bal_por_familia.csv", index=False, encoding="utf-8-sig")

    # ---------------- controles de particion ----------------
    ctrl = []
    for p in ("P2", "P2bal"):
        ctrl.append(dict(protocolo=p,
                         violaciones_plantilla_en_ambos_lados=res[p]["viol"],
                         familias_sin_train_por_pliegue=round(float(np.mean(res[p]["sin_train"])), 2),
                         plantillas_train_por_pliegue=round(float(np.mean(res[p]["plant_train"])), 2)))
    dc = pd.DataFrame(ctrl)
    dc.to_csv(OUT / "p2bal_controles.csv", index=False, encoding="utf-8-sig")
    print("\n=== CONTROLES DE LA PARTICION ===")
    print(dc.to_string(index=False))

    # ---------------- veredicto del preregistro ----------------
    bal_txt = next(r for r in filas if r["protocolo"] == "P2bal" and r["capa"] == "texto solo")
    bal_m6 = next(r for r in filas if r["protocolo"] == "P2bal" and r["capa"] == "M.6 (cascada)")
    p2_txt_sd = next(r for r in filas if r["protocolo"] == "P2" and r["capa"] == "texto solo")["sd"]
    st_bal = float(np.mean(res["P2bal"]["sin_train"]))
    st_p2 = float(np.mean(res["P2"]["sin_train"]))
    pt_bal = float(np.mean(res["P2bal"]["plant_train"]))
    pt_p2 = float(np.mean(res["P2"]["plant_train"]))
    lo_bal = float(bal_txt["ic95"].split(";")[0].strip("[ "))
    sube5 = {f: float(dfam.loc[dfam.familia == f, "delta_txt"].iloc[0]) for f in fam_dos
             if f in ("SUNCRYPT", "CUBA", "NETWALKER", "BLACKMATTER", "DARKSIDE")}

    print("\n" + "=" * 78)
    print("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    print("=" * 78)
    chk = [
        ("P1 texto >= 0,60 y IC excluye 0,50",
         bal_txt["f1_macro_30"] >= 0.60 and lo_bal > 0.50,
         f"{bal_txt['f1_macro_30']:.4f} {bal_txt['ic95']}"),
        ("P2 familias sin train = 2 (vs ~3,9 en P2)",
         abs(st_bal - 2.0) < 0.05, f"P2bal {st_bal:.2f} | P2 {st_p2:.2f}"),
        ("P3 plantillas de train casi iguales (dif <= 1,0)",
         abs(pt_bal - pt_p2) <= 1.0, f"P2bal {pt_bal:.2f} | P2 {pt_p2:.2f}"),
        ("P4 las 5 familias de 2 plantillas suben >= +0,30",
         all(v >= 0.30 for v in sube5.values()),
         ", ".join(f"{k} {v:+.3f}" for k, v in sorted(sube5.items()))),
        ("P5 sd menor que la de P2 (0,0752)",
         bal_txt["sd"] < p2_txt_sd, f"P2bal {bal_txt['sd']:.4f} | P2 {p2_txt_sd:.4f}"),
        ("P6 P2bal < LOGO (0,6747)",
         bal_txt["f1_macro_30"] < 0.6747, f"{bal_txt['f1_macro_30']:.4f} vs 0,6747"),
        ("P7 control externo (P2 = canonico)", ok, f"{p2_txt:.4f} / {p2_m6:.4f}"),
    ]
    for nombre, cumple, det in chk:
        print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<46} {det}")
    print(f"\n  Violaciones de la garantia (plantilla en train y test a la vez): "
          f"P2 {res['P2']['viol']} | P2bal {res['P2bal']['viol']}  (deben ser 0)")
    print(f"\n  CIFRA DE CABECERA P2bal: texto {bal_txt['f1_macro_30']:.4f} {bal_txt['ic95']} | "
          f"M.6 {bal_m6['f1_macro_30']:.4f} {bal_m6['ic95']}")
    print(f"  Sobre las 28 evaluables:  texto {bal_txt['f1_macro_28_evaluables']:.4f} | "
          f"M.6 {bal_m6['f1_macro_28_evaluables']:.4f}")
    print("\n  RECORDAR AL CITAR: 'plantilla no vista segun coseno char 0,90'. El criterio no "
          "detecta contencion (ver informe del 2026-09-17).")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
