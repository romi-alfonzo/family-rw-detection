#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
revision_bootstrap_estratificado.py -- revision cruzada de bootstrap_plantilla_p2bal.py.

QUE REVISA. El bootstrap por plantilla de la sesion hermana (2026-09-28) encontro que el
macro-F1 remuestreado queda 0,041-0,046 POR DEBAJO del punto estimado, y atribuyo el sesgo a
que con `labels=30` fijo una remuestra que pierde familias les asigna F1=0. El diagnostico es
correcto: se verifica aca que una remuestra pierde familias y cuantas.

LO QUE ESTE SCRIPT AGREGA. El sesgo no es un defecto del estimador que haya que corregir
despues: es sintoma de que **el remuestreo no estratificado esta contestando otra pregunta**.
Remuestrear las 99 plantillas libremente trata al CONJUNTO DE FAMILIAS como aleatorio -- como si
el corpus pudiera no tener CERBER. Pero las 30 familias NO son una muestra aleatoria: estan
fijadas por diseno, son las de NapierOne, y son el nucleo canonico que empareja los dos frentes
del trabajo. Lo que si es muestral es QUE PLANTILLAS se consiguieron de cada familia.

El remuestreo que corresponde a esa pregunta es ESTRATIFICADO POR FAMILIA: se remuestrean las
plantillas DENTRO de cada familia, con reposicion, conservando su cantidad. Asi ninguna familia
desaparece, el macro-F1 se calcula siempre sobre las mismas 30, y el estimando no cambia entre
replicas. Las familias de una sola plantilla conservan esa plantilla y su F1 sigue siendo 0, que
es lo correcto y es el cero estructural de siempre.

SE COMPARAN LAS TRES CONVENCIONES sobre exactamente las mismas predicciones:
  (a) libre + labels=30      -- la conservadora que reporta la sesion hermana
  (b) libre + labels presentes -- su alternativa sin sesgo, con denominador variable
  (c) ESTRATIFICADO POR FAMILIA + labels=30 -- la que propone este script

⚠️ REVISION POSTERIOR AL RESULTADO. No hay preregistro: es una revision de un resultado ya
obtenido, en la linea de la revision independiente del 2026-09-17. Se declara como tal. Lo unico
que se afirma es lo que se mide: si (c) tiene o no el sesgo, y cuanto cambia el intervalo.

Uso:  python revision_bootstrap_estratificado.py [--n-semillas 50] [--replicas 2000]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import obtener_modelos, vectorizador
from protocolo_logo import dicc_privados, regla
from protocolo_p2bal import split_p2bal
from revision_logo import cargar_todo

RAIZ = _AQUI.parent
OUT_DEF = RAIZ / "4_resultados" / "resultados_revision_bootstrap"
CANON = {"texto": 0.6551, "cascada": 0.7417}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=50)
    ap.add_argument("--replicas", type=int, default=2000)
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  REVISION CRUZADA: bootstrap libre vs ESTRATIFICADO POR FAMILIA")
    print("=" * 78)
    textos, textos_arr, y, archivos, grupos, iocs, nombres_nota = cargar_todo()
    familias = np.unique(y)
    n = len(y)
    plantillas = np.unique(grupos)
    idx_de = {g: np.where(grupos == g)[0] for g in plantillas}
    pl_de_fam = {f: np.unique(grupos[y == f]) for f in familias}
    print(f"Notas: {n} | Familias: {len(familias)} | Plantillas: {len(plantillas)}")
    print(f"Semillas: {args.n_semillas} | Replicas: {args.replicas}\n")

    pred = {"texto": np.empty((args.n_semillas, n), dtype=object),
            "cascada": np.empty((args.n_semillas, n), dtype=object)}
    print("Recalculando predicciones ...")
    for s in range(args.n_semillas):
        rng = np.random.default_rng(20_000 + s)
        for tr, te in split_p2bal(y, grupos, familias, rng):
            vec = vectorizador("combinado")
            Xtr = vec.fit_transform(textos_arr[tr])
            Xte = vec.transform(textos_arr[te])
            clf = obtener_modelos(s)["LinearSVC"]
            clf.fit(Xtr, y[tr])
            pt = clf.predict(Xte)
            pred["texto"][s, te] = pt
            d = dicc_privados(tr, iocs, nombres_nota, y)
            for k, i in enumerate(te):
                r = regla(i, d, iocs, nombres_nota)
                pred["cascada"][s, i] = pt[k] if r is None else r
        if (s + 1) % 10 == 0:
            print(f"  {s+1}/{args.n_semillas}")

    # ---------------- puerta ----------------
    punto = {}
    for capa in ("texto", "cascada"):
        punto[capa] = float(np.mean([f1_score(y, pred[capa][s], average="macro",
                                              labels=familias, zero_division=0)
                                     for s in range(args.n_semillas)]))
    print("\n  PUERTA: punto estimado vs cifra publicada")
    for capa in ("texto", "cascada"):
        print(f"    {capa:<9} {punto[capa]:.4f} vs {CANON[capa]}  "
              f"(dif {abs(punto[capa]-CANON[capa]):.6f})")
    if any(abs(punto[c] - CANON[c]) > 1e-3 for c in CANON):
        sys.exit("ABORTADO: no reproduce la cifra publicada.")
    print("    OK\n")

    # ---------------- remuestreos ----------------
    rng = np.random.default_rng(2026)
    n_pl = len(plantillas)
    res = {m: {c: [] for c in ("texto", "cascada")}
           for m in ("libre_30", "libre_pres", "estratificado_30")}
    perdidas_libre, perdidas_estr = [], []

    print("Remuestreando ...")
    for b in range(args.replicas):
        s = int(rng.integers(args.n_semillas))
        # (a,b) LIBRE: se remuestrean las 99 plantillas sin mirar la familia
        el_libre = plantillas[rng.integers(0, n_pl, n_pl)]
        i_libre = np.concatenate([idx_de[g] for g in el_libre])
        pres_libre = np.unique(y[i_libre])
        perdidas_libre.append(len(familias) - len(pres_libre))
        # (c) ESTRATIFICADO: se remuestrea DENTRO de cada familia, conservando su cantidad
        el_estr = np.concatenate([pl[rng.integers(0, len(pl), len(pl))]
                                  for pl in (pl_de_fam[f] for f in familias)])
        i_estr = np.concatenate([idx_de[g] for g in el_estr])
        perdidas_estr.append(len(familias) - len(np.unique(y[i_estr])))
        for capa in ("texto", "cascada"):
            p = pred[capa][s]
            res["libre_30"][capa].append(f1_score(y[i_libre], p[i_libre], average="macro",
                                                  labels=familias, zero_division=0))
            res["libre_pres"][capa].append(f1_score(y[i_libre], p[i_libre], average="macro",
                                                    labels=pres_libre, zero_division=0))
            res["estratificado_30"][capa].append(f1_score(y[i_estr], p[i_estr], average="macro",
                                                          labels=familias, zero_division=0))
        if (b + 1) % 500 == 0:
            print(f"  {b+1}/{args.replicas}")

    print(f"\n  Familias que pierde una remuestra, en promedio:")
    print(f"    libre         : {np.mean(perdidas_libre):.2f} de {len(familias)}")
    print(f"    estratificado : {np.mean(perdidas_estr):.2f} de {len(familias)}  "
          f"(tiene que ser 0,00 por construccion)")

    filas = []
    for m, etq in (("libre_30", "(a) libre + labels=30"),
                   ("libre_pres", "(b) libre + labels presentes"),
                   ("estratificado_30", "(c) ESTRATIFICADO + labels=30")):
        for capa in ("texto", "cascada"):
            v = np.array(res[m][capa])
            lo, hi = np.percentile(v, [2.5, 97.5])
            filas.append(dict(metodo=etq, capa=capa, punto=round(punto[capa], 4),
                              media_replicas=round(float(v.mean()), 4),
                              sesgo=round(float(v.mean() - punto[capa]), 4),
                              ic_bajo=round(float(lo), 4), ic_alto=round(float(hi), 4),
                              ancho=round(float(hi - lo), 4),
                              supera_050="SI" if lo > 0.50 else "NO"))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "comparacion_bootstrap.csv", index=False, encoding="utf-8-sig")
    print("\n=== LAS TRES CONVENCIONES SOBRE LAS MISMAS PREDICCIONES ===")
    print(df.to_string(index=False))

    e = {r["capa"]: r for r in filas if r["metodo"].startswith("(c)")}
    a = {r["capa"]: r for r in filas if r["metodo"].startswith("(a)")}
    print("\n" + "=" * 78)
    print("  LECTURA")
    print("=" * 78)
    print(f"  Sesgo del estratificado: texto {e['texto']['sesgo']:+.4f} | "
          f"cascada {e['cascada']['sesgo']:+.4f}")
    print(f"  Sesgo del libre:         texto {a['texto']['sesgo']:+.4f} | "
          f"cascada {a['cascada']['sesgo']:+.4f}")
    print(f"\n  Limite inferior de la CASCADA: libre {a['cascada']['ic_bajo']:.4f} | "
          f"estratificado {e['cascada']['ic_bajo']:.4f}")
    print(f"  Limite inferior del TEXTO:     libre {a['texto']['ic_bajo']:.4f} | "
          f"estratificado {e['texto']['ic_bajo']:.4f}")
    print("\n  El estratificado contesta «¿y si hubieramos conseguido OTRAS PLANTILLAS de estas")
    print("  mismas 30 familias?», que es la pregunta que corresponde a un corpus cuyo conjunto")
    print("  de familias esta fijado por NapierOne y no es muestral.")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
