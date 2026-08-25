#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
techo_por_familia.py -- El techo estructural de cada familia bajo P2.

LA PREGUNTA (de Romina, 2026-08-23). Si una familia tiene solo 2 plantillas, P2 pone una en
entrenamiento y la otra en prueba. Entonces la tarea es literalmente: «visto el texto A,
reconoce el texto B», donde A y B son plantillas DISTINTAS por construccion (coseno < 0,90,
que es el umbral de casi-duplicado). Si ademas A y B se parecen poco entre si, la tarea es
casi imposible por texto. Salvo que haya OTRA senal compartida: el nombre del archivo o los
marcadores.

Esto no es una intuicion: es el mecanismo que B.3 ya habia medido como correlacion entre
cohesion y F1 por familia (Spearman rho +0,704). Aca se convierte en un techo POR FAMILIA,
que es lo accionable.

QUE CALCULA, por familia:
  n_plantillas          cuantos textos distintos tiene (umbral 0,90)
  coseno_medio/min/max  similitud ENTRE sus plantillas (la cohesion de B.3)
  f1_base / f1_m6       lo que logra el texto solo y lo que logra M.6 (50 semillas)
  tiene_nombre          cuantas de sus notas tienen nombre de archivo auditado
  iocs_repetidos        cuantos valores de IOC aparecen en MAS DE UNA de sus plantillas
                        (es lo unico que le permite a la regla cruzar de una plantilla a otra)

LA CLAVE ES `iocs_repetidos` + `tiene_nombre`. Un IOC que aparece en UNA sola plantilla no
sirve bajo P2: si esa plantilla esta en prueba, su IOC no esta en el diccionario de
entrenamiento. Solo los marcadores o nombres que se REPITEN entre plantillas cruzan la
particion. Eso explica por que M.6 ayuda a DHARMA (+0,27) y no a las de 2 plantillas (+0,002).

Uso:  python techo_por_familia.py [--salida CARPETA]
Solo lee. No modifica nada.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, TFIDF_CHAR, UMBRAL_NEARDUP,
                                   agrupar_neardups, cargar_corpus)
from grafo_marcadores import extraer_marcadores

RAIZ = _AQUI.parent
AUDIT = RAIZ / "3_datos" / "nombres_notas" / "auditoria_nombres_corpus.csv"
M6 = (RAIZ / "4_resultados" / "resultados_cascada_combinada_155_50semillas"
      / "m6_por_familia.csv")
OUT_DEF = RAIZ / "4_resultados" / "resultados_techo_por_familia"


def main():
    global M6
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--m6", type=Path, default=None,
                    help="m6_por_familia.csv del que se leen las columnas F1. OJO: tiene que "
                         "ser el de la MISMA base que el corpus, o las columnas quedan viejas.")
    args = ap.parse_args()
    if args.m6:
        M6 = args.m6
    args.salida.mkdir(parents=True, exist_ok=True)

    textos, y, archivos, _ = cargar_corpus(CORPUS_DIR)
    grupos, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    n = len(textos)
    print(f"Corpus: {n} notas | {len(set(y))} familias | {len(set(grupos))} plantillas")

    # ---- un texto representativo por plantilla (el mas largo del grupo)
    porg = defaultdict(list)
    for i, g in enumerate(grupos):
        porg[g].append(i)
    rep = {g: max(idx, key=lambda i: len(textos[i])) for g, idx in porg.items()}

    # ---- matriz de similitud entre plantillas, mismo criterio canonico
    vec = TfidfVectorizer(**TFIDF_CHAR)
    ids = sorted(rep)
    X = vec.fit_transform([textos[rep[g]] for g in ids])
    S = cosine_similarity(X)
    pos = {g: k for k, g in enumerate(ids)}

    # ---- nombres auditados
    nombre_de = {}
    if AUDIT.is_file():
        with open(AUDIT, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f, delimiter=";"):
                if r["nombre_para_m2"]:
                    nombre_de[(r["familia"], r["archivo_corpus"])] = r["nombre_para_m2"].lower()

    # ---- F1 de M.6 (50 semillas, variante adoptada)
    f1b, f1m = {}, {}
    if M6.is_file():
        with open(M6, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["variante"] == "privados_sin_circ_MAS_NOMBRE":
                    f1b[r["familia"]] = float(r["f1_base"])
                    f1m[r["familia"]] = float(r["f1_variante"])

    # ---- IOCs por plantilla, para contar los que se REPITEN entre plantillas
    iocs_plant = defaultdict(set)
    for i, g in enumerate(grupos):
        iocs_plant[g] |= set(extraer_marcadores(textos[i]))

    fam_plant = defaultdict(set)
    fam_notas = defaultdict(list)
    for i, (f, g, a) in enumerate(zip(y, grupos, archivos)):
        fam_plant[f].add(g)
        fam_notas[f].append(Path(a).name)

    filas = []
    for fam in sorted(fam_plant):
        gs = sorted(fam_plant[fam])
        k = len(gs)
        if k >= 2:
            vals = [S[pos[a], pos[b]] for i, a in enumerate(gs) for b in gs[i + 1:]]
            cmed, cmin, cmax = float(np.mean(vals)), float(min(vals)), float(max(vals))
        else:
            cmed = cmin = cmax = float("nan")
        # IOCs que aparecen en mas de una plantilla de la familia
        cont = defaultdict(int)
        for g in gs:
            for v in iocs_plant[g]:
                cont[v] += 1
        rep_iocs = sum(1 for v, c in cont.items() if c >= 2)
        # nombres: cuantas notas tienen nombre auditado, y cuantos nombres se repiten
        noms = [nombre_de.get((fam, a)) for a in fam_notas[fam]]
        con_nom = sum(1 for x in noms if x)
        nom_por_plant = defaultdict(set)
        for a, g in zip(fam_notas[fam], [grupos[i] for i, f in enumerate(y) if f == fam]):
            nm = nombre_de.get((fam, a))
            if nm:
                nom_por_plant[nm].add(g)
        nom_rep = sum(1 for nm, s in nom_por_plant.items() if len(s) >= 2)
        filas.append(dict(
            familia=fam, n_notas=len(fam_notas[fam]), n_plantillas=k,
            coseno_medio_entre_plantillas=round(cmed, 4) if k >= 2 else "",
            coseno_min=round(cmin, 4) if k >= 2 else "",
            coseno_max=round(cmax, 4) if k >= 2 else "",
            notas_con_nombre_auditado=con_nom,
            nombres_que_cruzan_plantillas=nom_rep,
            iocs_que_cruzan_plantillas=rep_iocs,
            f1_base=round(f1b.get(fam, float("nan")), 4),
            f1_m6=round(f1m.get(fam, float("nan")), 4),
            delta_m6=round(f1m.get(fam, 0) - f1b.get(fam, 0), 4) if fam in f1b else ""))

    df = pd.DataFrame(filas)
    df.to_csv(args.salida / "techo_por_familia.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 108)
    print("  TECHO ESTRUCTURAL POR FAMILIA (ordenado por n_plantillas y coseno entre plantillas)")
    print("=" * 108)
    print(f"{'familia':<14}{'pl':>3}{'cos.med':>9}{'cos.max':>9}"
          f"{'nom':>5}{'nom-x':>6}{'ioc-x':>6}{'F1 base':>9}{'F1 M.6':>8}{'D':>8}")
    print("-" * 108)
    for _, r in df.sort_values(["n_plantillas", "coseno_medio_entre_plantillas"]).iterrows():
        print(f"{r.familia:<14}{r.n_plantillas:>3}"
              f"{str(r.coseno_medio_entre_plantillas):>9}{str(r.coseno_max):>9}"
              f"{r.notas_con_nombre_auditado:>5}{r.nombres_que_cruzan_plantillas:>6}"
              f"{r.iocs_que_cruzan_plantillas:>6}"
              f"{r.f1_base:>9.4f}{r.f1_m6:>8.4f}{str(r.delta_m6):>8}")

    d2 = df[df.n_plantillas == 2]
    print("\n" + "=" * 108)
    print("  LAS 8 FAMILIAS DE 2 PLANTILLAS: ¿por que M.6 no las ayuda?")
    print("=" * 108)
    print(f"  coseno medio entre sus 2 plantillas: "
          f"{np.mean([float(x) for x in d2.coseno_medio_entre_plantillas]):.4f}")
    print(f"  suma de nombres que cruzan plantillas: "
          f"{d2.nombres_que_cruzan_plantillas.sum()}")
    print(f"  suma de IOCs que cruzan plantillas   : "
          f"{d2.iocs_que_cruzan_plantillas.sum()}")
    resto = df[df.n_plantillas >= 4]
    print(f"\n  comparacion con las 17 de >=4 plantillas:")
    print(f"  coseno medio entre plantillas        : "
          f"{np.mean([float(x) for x in resto.coseno_medio_entre_plantillas]):.4f}")
    print(f"  nombres que cruzan plantillas (total): "
          f"{resto.nombres_que_cruzan_plantillas.sum()}")
    print(f"  IOCs que cruzan plantillas (total)   : "
          f"{resto.iocs_que_cruzan_plantillas.sum()}")

    # correlacion cohesion vs F1, replicando B.3 con la base de hoy
    sub = df[df.n_plantillas >= 2].copy()
    sub["cos"] = sub.coseno_medio_entre_plantillas.astype(float)
    from scipy.stats import spearmanr
    for col, et in (("f1_base", "texto solo"), ("f1_m6", "con M.6")):
        rho, pv = spearmanr(sub["cos"], sub[col])
        print(f"\n  Spearman(coseno entre plantillas ; F1 {et}) = {rho:+.4f}  (p = {pv:.2g})")
    print(f"\nSalida: {args.salida / 'techo_por_familia.csv'}")


if __name__ == "__main__":
    main()
