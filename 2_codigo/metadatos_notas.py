#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metadatos_notas.py -- M.7: ¿el METADATO de la nota (extension + tamano) agrega algo al texto?

Pregunta: el analisis forense usa metadatos del archivo (extension, tamano) ademas del
contenido. El TF-IDF normaliza L2, asi que el LARGO de la nota es informacion que el
clasificador congelado NO tiene. ¿Sumarla mueve el macro-F1?

DOS CONFUNDIDOS MEDIDOS ANTES DE CORRER, y por eso este experimento se reporta con reservas:

 1. EL LARGO ESTA CONFUNDIDO CON EL METODO DE RECOLECCION. Medido sobre las 155 notas:
      bruto (archivos del repo Lemmou)   n=96  largo medio 4016 car.
      corpus-existente                   n=34  largo medio 1100
      transcripcion (pcrisk/idr)         n=25  largo medio 1032
    y IM(tipo ; familia) = 0,751 bits de 4,597 => el METODO ya predice el 16 % de la familia.
    Los 37.602 caracteres de la nota mas larga de CERBER son el markup del .hta, no el
    mensaje. Es decir: parte del "largo" no es comportamiento del malware, es COMO se
    recolecto la nota. Es el mismo tipo de circularidad que el nombre puesto por el curador.

 2. LA INFORMACION MUTUA CRUDA SOBRE 155 NOTAS ESTA INFLADA. Con etiquetas PERMUTADAS al
    azar, IM(extension+largo ; familia) ya da 1,394 bits de media (30 % de la entropia)
    contra 2,058 observados. Con 30 clases y 155 muestras, contar celdas sobreajusta. Por eso
    la unica medicion valida es macro-F1 bajo P2, no la IM.

PREDICCION PREREGISTRADA (escrita antes de correr): el metadato aporta POCO o NADA:
Delta macro-F1 < 0,01 con IC 95 % que incluye el cero. Razon: la extension es casi constante
(117 de 155 notas son .txt, y aparece en las 30 familias), y el largo esta contaminado por el
metodo. Si SUBIERA de forma significativa, hay que sospechar del confundido antes de
festejar: el control es mirar si la ganancia viene de familias cuyas notas son todas `bruto`.

VARIANTES
  texto                     base congelada (vista combinada, LinearSVC)
  texto+extension           one-hot de la extension original del artefacto
  texto+largo               largo en caracteres, escalado
  texto+extension+largo     las dos
Se reporta ademas `solo_metadato` (sin texto) como piso: cuanto se saca con metadato solo.

PROTOCOLO identico a la base: P2 (grupos, StratifiedGroupKFold 2 pliegues), corpus 155/106/30,
LinearSVC(C=1, class_weight=balanced), Delta pareado por semilla contra el texto solo de la
MISMA particion. Por defecto 50 semillas (la base resulto sensible a la semilla: 0,5265 con
0-9 vs 0,4958 con 0-49).

Uso:  python metadatos_notas.py [--n-semillas 50] [--salida CARPETA]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.stats import t as t_dist
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, N_FOLDS, UMBRAL_NEARDUP, agrupar_neardups,
                                   cargar_corpus, obtener_modelos, vectorizador)

RAIZ = _AQUI.parent
MANIFIESTO = RAIZ / "3_datos" / "manifiesto_corpus_v2.csv"
OUT_DEF = RAIZ / "4_resultados" / "resultados_metadatos_155"

VARIANTES = ["texto", "texto+extension", "texto+largo", "texto+extension+largo",
             "solo_metadato"]


def ic95(d):
    n = len(d)
    m = float(np.mean(d))
    s = float(np.std(d, ddof=1)) if n > 1 else 0.0
    h = t_dist.ppf(0.975, n - 1) * (s / np.sqrt(n)) if n > 1 else 0.0
    return m, m - h, m + h


def cargar_metadatos(y, archivos):
    """Extension original declarada en el manifiesto (no la del archivo en disco) y largo."""
    ext_man = {}
    with open(MANIFIESTO, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ext_man[(r["familia"], r["archivo"])] = (
                r.get("extension_original") or "").strip().lower()
    exts = []
    for fam, a in zip(y, archivos):
        e = ext_man.get((fam, Path(a).name), "") or (Path(a).suffix.lower() or "(sin)")
        exts.append(e)
    return exts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=50)
    ap.add_argument("--semilla-inicial", type=int, default=0)
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)
    semillas = list(range(args.semilla_inicial, args.semilla_inicial + args.n_semillas))

    print("=" * 78)
    print("  M.7 -- METADATO DE LA NOTA (extension + tamano) SOBRE EL TEXTO")
    print("=" * 78)
    textos, y, archivos, _ = cargar_corpus(CORPUS_DIR)
    grupos, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    familias = np.unique(y)
    textos_arr = np.array(textos, dtype=object)
    n = len(textos)
    exts = cargar_metadatos(y, archivos)
    largos = np.array([len(t) for t in textos], dtype=float)
    vocab_ext = sorted(set(exts))
    print(f"Notas: {n} | Familias: {len(familias)} | Plantillas: {len(set(grupos))}")
    print(f"Extensiones distintas: {len(vocab_ext)} -> {vocab_ext}")
    print(f"Largo: min {int(largos.min())} | mediana {int(np.median(largos))} | "
          f"max {int(largos.max())} caracteres")
    print(f"Semillas: {semillas[0]}-{semillas[-1]} ({len(semillas)})")

    idx_ext = {e: i for i, e in enumerate(vocab_ext)}
    M_ext = np.zeros((n, len(vocab_ext)))
    for i, e in enumerate(exts):
        M_ext[i, idx_ext[e]] = 1.0

    pred = {v: [] for v in VARIANTES}
    print("\nEvaluando ...")
    for seed in semillas:
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        yp = {v: np.empty_like(y) for v in VARIANTES}
        for tr, te in cv.split(textos_arr, y, groups=grupos):
            vec = vectorizador("combinado")
            Xtr_t = vec.fit_transform(textos_arr[tr])
            Xte_t = vec.transform(textos_arr[te])
            # el largo se escala con estadisticas del ENTRENAMIENTO, nunca del total
            mu, sd = largos[tr].mean(), largos[tr].std() or 1.0
            L = ((largos - mu) / sd).reshape(-1, 1)
            bloques = {
                "texto": (Xtr_t, Xte_t),
                "texto+extension": (hstack([Xtr_t, csr_matrix(M_ext[tr])]),
                                    hstack([Xte_t, csr_matrix(M_ext[te])])),
                "texto+largo": (hstack([Xtr_t, csr_matrix(L[tr])]),
                                hstack([Xte_t, csr_matrix(L[te])])),
                "texto+extension+largo": (
                    hstack([Xtr_t, csr_matrix(M_ext[tr]), csr_matrix(L[tr])]),
                    hstack([Xte_t, csr_matrix(M_ext[te]), csr_matrix(L[te])])),
                "solo_metadato": (csr_matrix(np.hstack([M_ext[tr], L[tr]])),
                                  csr_matrix(np.hstack([M_ext[te], L[te]]))),
            }
            for v in VARIANTES:
                Xa, Xb = bloques[v]
                clf = obtener_modelos(seed)["LinearSVC"]
                clf.fit(Xa, y[tr])
                yp[v][te] = clf.predict(Xb)
        for v in VARIANTES:
            pred[v].append(yp[v])

    def met(ps):
        return (np.array([f1_score(y, p, average="macro", zero_division=0) for p in ps]),
                np.array([accuracy_score(y, p) for p in ps]),
                np.array([balanced_accuracy_score(y, p) for p in ps]),
                np.array([precision_recall_fscore_support(
                    y, p, labels=familias, zero_division=0)[2] for p in ps]))

    f1_b, acc_b, bal_b, fam_b = met(pred["texto"])
    print("\n" + "-" * 78)
    print(f"Base (texto solo, {len(semillas)} semillas): macro-F1 {f1_b.mean():.4f} "
          f"+/- {f1_b.std(ddof=1):.4f}")

    filas, filas_fam = [], []
    for v in VARIANTES:
        f1_v, acc_v, bal_v, fam_v = met(pred[v])
        d = f1_v - f1_b
        m, lo, hi = ic95(d)
        filas.append(dict(variante=v, f1_macro=round(float(f1_v.mean()), 4),
                          f1_macro_sd=round(float(f1_v.std(ddof=1)), 4),
                          exactitud=round(float(acc_v.mean()), 4),
                          exactitud_balanceada=round(float(bal_v.mean()), 4),
                          delta_f1=round(m, 4), ic95_bajo=round(lo, 4), ic95_alto=round(hi, 4),
                          semillas_positivas=f"{int((d > 0).sum())}/{len(d)}",
                          significativo="SI" if (lo > 0 or hi < 0) else "NO"))
        if v != "texto":
            for j, fam in enumerate(familias):
                mf, lof, hif = ic95(fam_v[:, j] - fam_b[:, j])
                filas_fam.append(dict(variante=v, familia=fam,
                                      f1_base=round(float(fam_b[:, j].mean()), 4),
                                      f1_variante=round(float(fam_v[:, j].mean()), 4),
                                      delta=round(mf, 4), ic95_bajo=round(lof, 4),
                                      ic95_alto=round(hif, 4)))

    pd.DataFrame(filas).to_csv(OUT / "m7_resumen.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(filas_fam).to_csv(OUT / "m7_por_familia.csv", index=False,
                                   encoding="utf-8-sig")

    print("\n=== RESULTADO ===")
    for r in filas:
        if r["variante"] == "texto":
            print(f"  {r['variante']:<24} macro-F1 {r['f1_macro']:.4f} "
                  f"+/- {r['f1_macro_sd']:.4f}  (referencia)")
        else:
            print(f"  {r['variante']:<24} macro-F1 {r['f1_macro']:.4f} | "
                  f"D {r['delta_f1']:+.4f} [{r['ic95_bajo']:+.4f}; {r['ic95_alto']:+.4f}] "
                  f"{r['semillas_positivas']} | signif: {r['significativo']}")

    print("\n=== PREDICCION PREREGISTRADA: aporte < 0,01 con IC que incluye el cero ===")
    for r in filas:
        if r["variante"] in ("texto", "solo_metadato"):
            continue
        cumple = abs(r["delta_f1"]) < 0.01 and r["ic95_bajo"] <= 0 <= r["ic95_alto"]
        print(f"  {r['variante']:<24} -> se cumple: {'SI' if cumple else 'NO'}")

    print("\n=== CONTROL DEL CONFUNDIDO (si alguna subio, mirar aca) ===")
    ff = pd.DataFrame(filas_fam)
    if len(ff):
        top = ff[ff.variante == "texto+extension+largo"].nlargest(5, "delta")
        print("  familias que mas suben con extension+largo:")
        for _, r in top.iterrows():
            print(f"    {r.familia:<14} D {r.delta:+.4f} "
                  f"[{r.ic95_bajo:+.4f}; {r.ic95_alto:+.4f}]")
        print("  -> revisar en el manifiesto si sus notas son todas `bruto`: en ese caso el "
              "largo esta leyendo el METODO DE RECOLECCION, no la familia.")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
