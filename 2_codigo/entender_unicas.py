#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entender_unicas.py -- de dónde salen las notas cuya plantilla aparece una sola vez.

Pregunta (Romina, 2026-10-01): bajo P1 la etiqueta «plantilla ya catalogada» solo vale para el
41 % de las decisiones, porque 76 de las 149 notas son únicas. Este script describe esas 76 sin
cambiar nada: de qué familia y de qué fuente son, y a qué distancia está su pariente más cercano
dentro de la misma familia (coseno de caracteres, el mismo del criterio de plantilla). Si muchas
tienen una hermana justo por debajo de 0,90, el problema es el umbral; si están lejos, es de datos.

No imprime texto de ninguna nota: solo nombres de archivo, fuentes y números.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstencion_notas as ab  # noqa: E402
import clasificador_notas_v2 as cl  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

textos, y, archivos, _ = ab.cargar_corpus(ab.CORPUS_DIR)
grupos, _ = ab.agrupar_neardups(textos, ab.UMBRAL_NEARDUP)
y, grupos = np.asarray(y), np.asarray(grupos)
tam = Counter(grupos)
unica = np.array([tam[g] == 1 for g in grupos])

# fuente de cada nota, desde el manifiesto
fuente = {}
with open(ab.RAIZ / "3_datos" / "manifiesto_corpus_v2.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        fuente[(r["familia"], Path(r["archivo"]).name)] = r["fuente"] or "(sin fuente)"
fte = np.array([fuente.get((fa, Path(a).name), "(no en manifiesto)") for fa, a in zip(y, archivos)])

X = TfidfVectorizer(**cl.TFIDF_CHAR).fit_transform(textos)
sim = (X @ X.T).toarray()
np.fill_diagonal(sim, -1)

print(f"Notas: {len(y)} | únicas: {unica.sum()} | en plantillas repetidas: {(~unica).sum()}\n")

print("--- únicas por fuente (y fracción de cada fuente que es única) ---")
tot_f = Counter(fte)
for fu, k in Counter(fte[unica]).most_common():
    print(f"  {fu:<38}{k:>4} de {tot_f[fu]:<4} ({k / tot_f[fu]:.0%})")

print("\n--- coseno de cada nota única con su pariente más cercano de LA MISMA familia ---")
cerc = []
for i in np.where(unica)[0]:
    misma = [j for j in range(len(y)) if j != i and y[j] == y[i]]
    cerc.append(max(sim[i, j] for j in misma) if misma else np.nan)
cerc = np.array(cerc)
for lo, hi, etq in ((0.85, 0.90, "0,85-0,90 (casi plantilla)"), (0.70, 0.85, "0,70-0,85"),
                    (0.50, 0.70, "0,50-0,70"), (-1, 0.50, "< 0,50 (texto propio)")):
    k = int(((cerc >= lo) & (cerc < hi)).sum())
    print(f"  {etq:<30}{k:>4}")
print(f"  {'sin otra nota en la familia':<30}{int(np.isnan(cerc).sum()):>4}")

print("\n--- familias: notas, plantillas, y cuántas de sus notas son únicas ---")
fams = defaultdict(lambda: [0, set(), 0])
for i in range(len(y)):
    fams[y[i]][0] += 1
    fams[y[i]][1].add(grupos[i])
    fams[y[i]][2] += int(unica[i])
print(f"  {'familia':<14}{'notas':>6}{'plant.':>8}{'únicas':>8}")
for fa, (n, pl, u) in sorted(fams.items(), key=lambda kv: -kv[1][2] / kv[1][0]):
    print(f"  {fa:<14}{n:>6}{len(pl):>8}{u:>8}")
