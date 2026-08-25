#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
abstencion_notas.py -- M.3: abstencion por umbral de confianza en el frente de notas.

QUE CONTESTA. La objecion «0,52 de macro-F1 es poco» supone que el sistema SIEMPRE tiene que
contestar. Un identificador de familia desplegado no tiene por que: puede decir «no reconozco
esto». Este experimento entrega la curva **precision-vs-cobertura**: «cuando contesta, acierta
X %; contesta el Y % de las veces».

⚠️ PREDICCION PREREGISTRADA (ya escrita en PLAN_MEJORAS.md §M.3, antes de implementar):
**NO sube el macro-F1. Cambia el reporte.** Si el macro-F1 sobre TODO el conjunto subiera, algo
esta mal: abstenerse no puede mejorar una metrica que se calcula sobre las notas no respondidas.
Lo que tiene que subir es el **acierto donde contesta**, a costa de cobertura.

COMO SE MIDE LA CONFIANZA. LinearSVC no da probabilidades. Se usa el **margen entre la primera
y la segunda clase** de `decision_function` (one-vs-rest): si la mejor clase le gana holgado a
la segunda, la decision es firme; si estan pegadas, es un empate disfrazado. Es la medida
natural para un SVC y no requiere calibrar nada.

LA ARQUITECTURA RESPETA M.6 (la cascada adoptada):
  1. Si la regla exacta aplica (IOC privado o nombre genuino visto en entrenamiento) -> se
     contesta SIEMPRE. La regla ya mide 0,9755 de acierto: abstenerse ahi seria tirar precision.
  2. Si no aplica -> decide el texto, y ahi SI se aplica el umbral de abstencion.
Se reporta tambien la variante «solo texto» para aislar el efecto.

SE REPORTA, por cada umbral:
  cobertura (que fraccion contesta) · acierto donde contesta · macro-F1 sobre las respondidas ·
  cuantas abstenciones · y el desglose entre las que resolvio la regla y las que resolvio el texto.

PROTOCOLO identico a M.6: P2 (grupos, StratifiedGroupKFold 2 pliegues), corpus actual,
LinearSVC(C=1, class_weight=balanced) sobre la vista combinada, 50 semillas.

Uso:  python abstencion_notas.py [--n-semillas 50] [--salida CARPETA]
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, N_FOLDS, UMBRAL_NEARDUP, agrupar_neardups,
                                   cargar_corpus, obtener_modelos, vectorizador)
from grafo_marcadores import _terminos_circulares, extraer_marcadores

RAIZ = _AQUI.parent
DIR_NOMBRES = RAIZ / "3_datos" / "nombres_notas"
OUT_DEF = RAIZ / "4_resultados" / "resultados_abstencion"
UMBRALES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50]


def cargar_nombres():
    """(familia, archivo) -> nombre genuino en minusculas. Solo nombres auditados."""
    nombres = {}
    csv_aud = DIR_NOMBRES / "auditoria_nombres_corpus.csv"
    if csv_aud.is_file():
        with open(csv_aud, encoding="utf-8-sig") as f:
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
    """Diccionario de la variante ADOPTADA de M.6: IOCs privados + nombre, sin filtro circ."""
    d = defaultdict(set)
    for i in tr:
        for clave in iocs[i]:
            d[clave].add(y[i])
        if nombres_nota[i]:
            d[("[NOMBRE]", nombres_nota[i])].add(y[i])
    for k in [k for k, v in d.items() if len(v) > 1]:
        del d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=50)
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  M.3 -- ABSTENCION POR UMBRAL DE CONFIANZA")
    print("=" * 78)
    textos, y, archivos, _ = cargar_corpus(CORPUS_DIR)
    grupos, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    textos_arr = np.array(textos, dtype=object)
    n = len(textos)
    iocs = [set(extraer_marcadores(t)) for t in textos]
    nom_aud = cargar_nombres()
    nombres_nota = [nom_aud.get((f, Path(a).name)) for f, a in zip(y, archivos)]
    print(f"Notas: {n} | Familias: {len(set(y))} | Plantillas: {len(set(grupos))}")
    print(f"Con nombre genuino: {sum(1 for x in nombres_nota if x)} | "
          f"Semillas: {args.n_semillas} | Umbrales: {UMBRALES}")

    # margen[i] por semilla, prediccion de texto, y si la regla aplico
    margen = np.zeros((args.n_semillas, n))
    pred_txt = np.empty((args.n_semillas, n), dtype=object)
    pred_regla = np.empty((args.n_semillas, n), dtype=object)
    aplica = np.zeros((args.n_semillas, n), dtype=bool)

    print("\nEvaluando ...")
    for s in range(args.n_semillas):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
        for tr, te in cv.split(textos_arr, y, groups=grupos):
            vec = vectorizador("combinado")
            Xtr = vec.fit_transform(textos_arr[tr])
            Xte = vec.transform(textos_arr[te])
            clf = obtener_modelos(s)["LinearSVC"]
            clf.fit(Xtr, y[tr])
            dec = clf.decision_function(Xte)
            clases = clf.classes_
            orden = np.argsort(-dec, axis=1)
            top1 = clases[orden[:, 0]]
            m = dec[np.arange(len(te)), orden[:, 0]] - dec[np.arange(len(te)), orden[:, 1]]
            pred_txt[s, te] = top1
            margen[s, te] = m
            # capa de reglas de M.6 (variante adoptada)
            d = dicc_privados(tr, iocs, nombres_nota, y)
            for k, i in enumerate(te):
                claves = set(iocs[i])
                if nombres_nota[i]:
                    claves.add(("[NOMBRE]", nombres_nota[i]))
                fams = set()
                for c in claves:
                    if c in d:
                        fams |= d[c]
                if len(fams) == 1:
                    aplica[s, i] = True
                    pred_regla[s, i] = next(iter(fams))
        if (s + 1) % 10 == 0:
            print(f"  {s+1}/{args.n_semillas} semillas")

    y_arr = np.asarray(y)
    filas = []
    for solo_texto in (False, True):
        for u in UMBRALES:
            cob, ac, f1s, n_reg, n_txt = [], [], [], [], []
            for s in range(args.n_semillas):
                if solo_texto:
                    contesta = margen[s] >= u
                    pred = pred_txt[s]
                else:
                    contesta = aplica[s] | (margen[s] >= u)
                    pred = np.where(aplica[s], pred_regla[s], pred_txt[s])
                cob.append(contesta.mean())
                if contesta.any():
                    ac.append(accuracy_score(y_arr[contesta], pred[contesta]))
                    f1s.append(f1_score(y_arr[contesta], pred[contesta],
                                        average="macro", zero_division=0))
                else:
                    ac.append(np.nan); f1s.append(np.nan)
                n_reg.append(int((aplica[s] & contesta).sum()) if not solo_texto else 0)
                n_txt.append(int((contesta & ~aplica[s]).sum()) if not solo_texto
                             else int(contesta.sum()))
            filas.append(dict(
                sistema="solo_texto" if solo_texto else "M.6 (reglas + texto)",
                umbral=u, cobertura=round(float(np.mean(cob)), 4),
                cobertura_sd=round(float(np.std(cob, ddof=1)), 4),
                acierto_donde_contesta=round(float(np.nanmean(ac)), 4),
                f1_macro_de_las_respondidas=round(float(np.nanmean(f1s)), 4),
                abstenciones=round(float((1 - np.mean(cob)) * n), 1),
                resueltas_por_regla=round(float(np.mean(n_reg)), 1),
                resueltas_por_texto=round(float(np.mean(n_txt)), 1)))

    df = pd.DataFrame(filas)
    df.to_csv(OUT / "m3_curva_abstencion.csv", index=False, encoding="utf-8-sig")

    for sistema in ("M.6 (reglas + texto)", "solo_texto"):
        print(f"\n=== {sistema} ===")
        print(f"{'umbral':>8}{'cobertura':>12}{'acierto':>10}{'F1 respond.':>13}"
              f"{'abstiene':>10}{'x regla':>9}{'x texto':>9}")
        for r in filas:
            if r["sistema"] != sistema:
                continue
            print(f"{r['umbral']:>8.2f}{r['cobertura']:>12.4f}"
                  f"{r['acierto_donde_contesta']:>10.4f}"
                  f"{r['f1_macro_de_las_respondidas']:>13.4f}"
                  f"{r['abstenciones']:>10.1f}{r['resueltas_por_regla']:>9.1f}"
                  f"{r['resueltas_por_texto']:>9.1f}")

    base = next(r for r in filas if r["sistema"].startswith("M.6") and r["umbral"] == 0.0)
    print("\n=== CONTROL DE LA PREDICCION PREREGISTRADA ===")
    print(f"Sin abstencion (umbral 0): cobertura {base['cobertura']:.4f} | "
          f"acierto {base['acierto_donde_contesta']:.4f}")
    print("La prediccion dice: el acierto donde contesta DEBE subir con el umbral, y la")
    print("cobertura DEBE bajar. Si el acierto no sube, la confianza del SVC no informa nada.")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
