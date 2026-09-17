#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
protocolo_lemmou.py -- ¿Cuánto da NUESTRO corpus bajo el protocolo de Lemmou et al. (2021)?

POR QUE ESTE EXPERIMENTO. Lemmou et al. reportan 181/182 en identificacion de familia. Esa
cifra NO es comparable con nuestro P2, y este script lo demuestra con numeros en vez de con
argumentos. Lo que hacen ellos (segun lo ya fijado en el proyecto) es **reglas y marcadores +
LSA como busqueda de casi-duplicados, en MUNDO CERRADO y SIN train/test**: cada nota se busca
contra una base que CONTIENE esa misma nota. Ojo tambien con su F = 0,920, que es la tarea
BINARIA nombre-de-nota frente a nombre-benigno, NO clasificacion de familia por nombre.

Nuestro P2 hace lo contrario a proposito: StratifiedGroupKFold sobre los grupos de
casi-duplicados, de modo que una nota de prueba NUNCA tiene a su gemela en entrenamiento. Esa
decision es la que baja el numero, y es metodologicamente la correcta.

SE MIDEN TRES PROTOCOLOS SOBRE EL MISMO CORPUS Y EL MISMO VECTORIZADOR, para que la unica
cosa que cambie sea el protocolo:

  L  (Lemmou)   1-NN por coseno, leave-one-out, SIN separar casi-duplicados. Es su busqueda
                de casi-duplicados: la nota mas parecida del resto del corpus le presta su
                familia. Variante extra con LSA (TruncatedSVD), que es literalmente lo que
                ellos usan.
  P1            StratifiedKFold: separa notas pero NO plantillas. Una nota de prueba puede
                tener a su casi-duplicada en entrenamiento.
  P2            StratifiedGroupKFold sobre grupos de casi-duplicados. El protocolo honesto,
                el que usa la tesis.

LO QUE ESTE SCRIPT **NO** DICE. Que el numero de L sea mas alto no significa que el metodo sea
mejor: significa que la tarea es mas facil. L no puede reconocer una plantilla que no esta en
la base. Reportar L como resultado seria exactamente el error que la tesis evita.

Uso:  python protocolo_lemmou.py [--n-semillas 50] [--salida CARPETA]
Solo lee el corpus. No modifica nada.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import normalize

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, N_FOLDS, UMBRAL_NEARDUP, agrupar_neardups,
                                   cargar_corpus, obtener_modelos, vectorizador)

OUT_DEF = _AQUI.parent / "4_resultados" / "resultados_protocolo_lemmou"


def metricas(y, yp):
    return dict(exactitud=accuracy_score(y, yp),
                exactitud_balanceada=balanced_accuracy_score(y, yp),
                macro_f1=f1_score(y, yp, average="macro", zero_division=0))


def uno_nn_loo(X, y, usar_lsa=False, dim=100, semilla=0):
    """1-NN por coseno con leave-one-out sobre TODO el corpus (protocolo de Lemmou).

    Es busqueda de casi-duplicados: a cada nota le presta la familia la nota mas parecida
    del resto del corpus. No hay entrenamiento ni particion.
    """
    M = X
    if usar_lsa:
        k = min(dim, min(M.shape) - 1)
        M = TruncatedSVD(n_components=k, random_state=semilla).fit_transform(M)
    M = normalize(M)
    S = (M @ M.T)
    S = np.asarray(S.todense()) if hasattr(S, "todense") else np.asarray(S)
    np.fill_diagonal(S, -np.inf)          # leave-one-out: nunca se compara consigo misma
    vecina = S.argmax(axis=1)
    return np.asarray(y)[vecina], S[np.arange(len(y)), vecina]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=50)
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  EL MISMO CORPUS BAJO TRES PROTOCOLOS (Lemmou / P1 / P2)")
    print("=" * 78)
    textos, y, archivos, _ = cargar_corpus(CORPUS_DIR)
    grupos, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    textos_arr = np.array(textos, dtype=object)
    n = len(textos)
    print(f"Notas: {n} | Familias: {len(set(y))} | Plantillas (grupos): {len(set(grupos))}")

    filas = []

    # ---------- Protocolo L (Lemmou): 1-NN leave-one-out, mundo cerrado
    vec = vectorizador("combinado")
    X = vec.fit_transform(textos_arr)      # mundo cerrado: se ajusta con todo, como ellos
    for etiqueta, lsa in (("L (Lemmou) 1-NN coseno, mundo cerrado", False),
                          ("L (Lemmou) 1-NN + LSA(SVD 100)", True)):
        yp, sim = uno_nn_loo(X, y, usar_lsa=lsa)
        m = metricas(y, yp)
        # ¿cuantos aciertos vienen de una vecina que es CASI-DUPLICADA (mismo grupo)?
        mismo_grupo = np.array([grupos[i] == grupos[j] for i, j in
                                enumerate(np.argmax(
                                    _sim_argmax(X, lsa), axis=1))]) if False else None
        filas.append(dict(protocolo=etiqueta, **{k: round(v, 4) for k, v in m.items()},
                          semillas=1, nota="sin train/test; la vecina puede ser su casi-duplicada"))
        print(f"\n{etiqueta}")
        print(f"   exactitud {m['exactitud']:.4f} | exact. balanceada "
              f"{m['exactitud_balanceada']:.4f} | macro-F1 {m['macro_f1']:.4f}")

    # ---------- Cuanto de eso lo explica tener a la gemela en la base
    yp, sim = uno_nn_loo(X, y, usar_lsa=False)
    M = normalize(X)
    S = np.asarray((M @ M.T).todense())
    np.fill_diagonal(S, -np.inf)
    vecina = S.argmax(axis=1)
    g = np.array(grupos)
    misma_plantilla = g[vecina] == g
    acierto = np.asarray(yp) == np.asarray(y)
    print(f"\n  De las {n} notas, {misma_plantilla.sum()} tienen como vecina mas cercana a una "
          f"nota de su MISMA plantilla ({100*misma_plantilla.mean():.1f} %).")
    print(f"  Acierto cuando la vecina es de su misma plantilla : "
          f"{acierto[misma_plantilla].mean():.4f}")
    print(f"  Acierto cuando la vecina es de OTRA plantilla     : "
          f"{acierto[~misma_plantilla].mean():.4f}  <-- esto es lo que mide P2")
    filas.append(dict(protocolo="L: acierto con vecina de su MISMA plantilla",
                      exactitud=round(float(acierto[misma_plantilla].mean()), 4),
                      exactitud_balanceada=np.nan, macro_f1=np.nan,
                      semillas=1, nota=f"{int(misma_plantilla.sum())} de {n} notas"))
    filas.append(dict(protocolo="L: acierto con vecina de OTRA plantilla",
                      exactitud=round(float(acierto[~misma_plantilla].mean()), 4),
                      exactitud_balanceada=np.nan, macro_f1=np.nan,
                      semillas=1, nota=f"{int((~misma_plantilla).sum())} de {n} notas"))

    # ---------- P1 y P2 con el clasificador de la tesis
    for nombre, hacer_cv in (("P1 (estratificado: separa notas, NO plantillas)", "p1"),
                             ("P2 (grupos: separa PLANTILLAS) -- el de la tesis", "p2")):
        f1s, accs, bals = [], [], []
        for seed in range(args.n_semillas):
            cv = (StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                  if hacer_cv == "p1" else
                  StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed))
            it = (cv.split(textos_arr, y) if hacer_cv == "p1"
                  else cv.split(textos_arr, y, groups=grupos))
            yp = np.empty_like(y)
            for tr, te in it:
                v = vectorizador("combinado")
                Xtr = v.fit_transform(textos_arr[tr])
                Xte = v.transform(textos_arr[te])
                clf = obtener_modelos(seed)["LinearSVC"]
                clf.fit(Xtr, y[tr])
                yp[te] = clf.predict(Xte)
            f1s.append(f1_score(y, yp, average="macro", zero_division=0))
            accs.append(accuracy_score(y, yp))
            bals.append(balanced_accuracy_score(y, yp))
        filas.append(dict(protocolo=nombre, exactitud=round(float(np.mean(accs)), 4),
                          exactitud_balanceada=round(float(np.mean(bals)), 4),
                          macro_f1=round(float(np.mean(f1s)), 4),
                          semillas=args.n_semillas,
                          nota=f"desvio macro-F1 {np.std(f1s, ddof=1):.4f}"))
        print(f"\n{nombre}  ({args.n_semillas} semillas)")
        print(f"   exactitud {np.mean(accs):.4f} | exact. balanceada {np.mean(bals):.4f} | "
              f"macro-F1 {np.mean(f1s):.4f} +/- {np.std(f1s, ddof=1):.4f}")

    pd.DataFrame(filas).to_csv(OUT / "comparacion_protocolos.csv", index=False,
                               encoding="utf-8-sig")
    print(f"\n{'=' * 78}")
    print("  LECTURA: la diferencia entre L y P2 es EL PROTOCOLO, no el metodo.")
    print("  L no puede reconocer una plantilla que no este en su base; P2 mide justamente eso.")
    print(f"\nSalida: {OUT / 'comparacion_protocolos.csv'}")


def _sim_argmax(X, lsa):
    return X


if __name__ == "__main__":
    main()
