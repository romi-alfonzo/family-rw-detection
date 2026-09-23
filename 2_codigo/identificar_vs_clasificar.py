"""Dos tareas distintas con requisitos de dato distintos: CLASIFICAR vs IDENTIFICAR.

LA IDEA, que es de Romina (2026-08-28): «una nota no nos sirve para clasificar, pero sí
para descubrir quién es, si el texto es similar».

Y es exacto, y explica los ceros estructurales:

  CLASIFICAR (lo que mide P2). Se entrena un modelo y se evalúa con StratifiedGroupKFold
  sobre los grupos de casi-duplicados. Una familia con UNA sola plantilla nunca está en
  entrenamiento y prueba a la vez => su F1 es 0 por construcción. Necesita >= 2 plantillas.

  IDENTIFICAR por similitud (lo que hace ID Ransomware, y el paso de LSA de Lemmou). No hay
  modelo entrenado: se busca el vecino más cercano en un catálogo de notas conocidas y se
  devuelve su familia. Alcanza con UNA nota conocida por familia. Es recuperación, no
  aprendizaje.

Este script mide la segunda tarea con el mismo corpus, dejando una NOTA afuera cada vez
(leave-one-out sobre notas, no sobre grupos), y la compara con el F1 por familia de P2. La
pregunta concreta: las familias que dan 0 clasificando, ¿se identifican?

⚠️ Lo que NO es: no es una métrica del método propuesto en la tesis ni reemplaza P2. Es la
medida de una tarea distinta, y sirve para declarar con precisión qué puede y qué no puede
hacer el sistema con una sola nota por familia. El mundo cerrado sigue valiendo: si la nota
es de una familia que no está en el catálogo, el vecino más cercano va a ser incorrecto —
por eso se reporta también la distribución del coseno del acierto y del error.

Uso:
    python identificar_vs_clasificar.py
    python identificar_vs_clasificar.py --por-familia PROTON --fuente <carpeta>
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, TFIDF_CHAR, UMBRAL_NEARDUP,
                                   agrupar_neardups, cargar_corpus)
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--f1-p2", type=Path, default=None,
                    help="corrida_canonica_por_familia.csv para comparar contra P2")
    args = ap.parse_args()
    corpus = args.corpus or CORPUS_DIR

    textos, y, archivos, _ = cargar_corpus(corpus)
    y = np.array(y)
    grupos = np.array(agrupar_neardups(textos, UMBRAL_NEARDUP)[0])
    fams = sorted(set(y))
    plant = {f: len(set(grupos[y == f])) for f in fams}
    notas = {f: int((y == f).sum()) for f in fams}

    X = TfidfVectorizer(**TFIDF_CHAR).fit_transform(textos)
    sim = (X @ X.T).toarray()
    np.fill_diagonal(sim, -1.0)          # nunca se compara consigo misma

    print("=" * 86)
    print("  IDENTIFICAR POR SIMILITUD (1-vecino, dejando una NOTA afuera) vs CLASIFICAR (P2)")
    print("=" * 86)
    print("corpus: %s" % corpus)
    print("        %d notas · %d familias · %d plantillas\n" % (len(textos), len(fams),
                                                                len(set(grupos))))

    acierto = defaultdict(list)
    cos_ok, cos_mal = [], []
    for i in range(len(textos)):
        j = int(np.argmax(sim[i]))
        ok = y[j] == y[i]
        acierto[y[i]].append(ok)
        (cos_ok if ok else cos_mal).append(float(sim[i, j]))

    global_ok = sum(sum(v) for v in acierto.values())
    print("ACIERTO GLOBAL identificando por vecino más cercano: %d de %d = %.4f"
          % (global_ok, len(textos), global_ok / len(textos)))
    print("  coseno del vecino cuando ACIERTA:  mediana %.3f | mínimo %.3f"
          % (np.median(cos_ok), min(cos_ok)))
    if cos_mal:
        print("  coseno del vecino cuando FALLA:    mediana %.3f | máximo %.3f"
              % (np.median(cos_mal), max(cos_mal)))

    f1_p2 = {}
    ruta = args.f1_p2 or (_AQUI.parent / "4_resultados" / "resultados_notas_149"
                          / "corrida_canonica_por_familia.csv")
    if ruta.is_file():
        import csv
        for r in csv.DictReader(open(ruta, encoding="utf-8-sig")):
            f1_p2[r["familia"]] = float(r["f1"])

    print("\n" + "-" * 86)
    print("  %-14s %6s %6s %10s %12s %10s" % ("familia", "notas", "plant",
                                              "F1 P2", "identifica", "¿cambia?"))
    print("-" * 86)
    filas = []
    for f in fams:
        a = acierto[f]
        tasa = sum(a) / len(a)
        p2 = f1_p2.get(f)
        filas.append((plant[f], f, notas[f], tasa, p2))
    for pl, f, nt, tasa, p2 in sorted(filas):
        marca = ""
        if p2 is not None and p2 == 0.0 and tasa > 0.5:
            marca = "★ 0 clasificando, se IDENTIFICA"
        elif p2 is not None and tasa - p2 > 0.30:
            marca = "identificar es mucho mejor"
        elif p2 is not None and p2 - tasa > 0.30:
            marca = "clasificar es mejor"
        print("  %-14s %6d %6d %10s %11.3f  %s"
              % (f, nt, pl, ("%.3f" % p2) if p2 is not None else "—", tasa, marca))

    print("-" * 86)
    inev = [f for f in fams if plant[f] < 2]
    if inev:
        print("\n★ LAS FAMILIAS INEVALUABLES BAJO P2 (%s):" % ", ".join(inev))
        for f in inev:
            a = acierto[f]
            print("    %-14s F1 clasificando %.3f  ->  identifica %d de %d = %.3f"
                  % (f, f1_p2.get(f, float("nan")), sum(a), len(a), sum(a) / len(a)))
        print("\n  Su F1 = 0 es una propiedad DEL PROTOCOLO de clasificación, no del dato:")
        print("  con una sola plantilla no puede haber train y test a la vez, pero sus notas")
        print("  SÍ se reconocen entre sí. Para identificar alcanza una nota conocida.")


if __name__ == "__main__":
    main()
