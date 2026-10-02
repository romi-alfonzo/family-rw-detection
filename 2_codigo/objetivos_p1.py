#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
objetivos_p1.py -- qué notas falla la cascada bajo P1, para saber dónde servirían datos nuevos.

Una copia controlada (otra víctima de la misma plantilla) solo puede ayudar si la nota de esa
plantilla hoy se falla. La copia de una plantilla que ya se acierta siempre no mueve nada: es lo
que pasó con 3 de las 6 copias de p1_copias_controladas.py. Este script lista, nota por nota, el
acierto de la cascada publicada bajo P1 (50 semillas) y si su plantilla tiene hermana en el corpus.
No imprime texto de notas: solo familia, archivo y números. Salida en
4_resultados/resultados_objetivos_p1/objetivos_p1.csv.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstencion_notas as ab  # noqa: E402
from p1_catalogada import evaluar  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

N_SEM = int(sys.argv[1]) if len(sys.argv) > 1 else 50
OUT = ab.RAIZ / "4_resultados" / "resultados_objetivos_p1"
OUT.mkdir(parents=True, exist_ok=True)

textos, y, archivos, _ = ab.cargar_corpus(ab.CORPUS_DIR)
grupos, _ = ab.agrupar_neardups(textos, ab.UMBRAL_NEARDUP)
textos_arr = np.array(textos, dtype=object)
y, grupos = np.asarray(y), np.asarray(grupos)
iocs = [set(ab.extraer_marcadores(t)) for t in textos]
nom_aud = ab.cargar_nombres()
nombres_nota = [nom_aud.get((f, Path(a).name)) for f, a in zip(y, archivos)]
tam = Counter(grupos)

aciertos = np.zeros(len(y))
errores_a = [Counter() for _ in range(len(y))]
for s in range(N_SEM):
    cv = StratifiedKFold(n_splits=ab.N_FOLDS, shuffle=True, random_state=s)
    idx, _, p_cas, _ = evaluar(list(cv.split(textos_arr, y)), textos_arr, y, iocs, nombres_nota, s)
    for i, p in zip(idx, p_cas):
        if p == y[i]:
            aciertos[i] += 1
        else:
            errores_a[i][p] += 1
    if (s + 1) % 10 == 0:
        print(f"  {s+1}/{N_SEM} semillas", flush=True)

ac = aciertos / N_SEM
total = ac.mean()
print(f"\nPuerta: exactitud media {total:.4f} (publicada 0,8866)")
filas = []
for i in range(len(y)):
    filas.append(dict(familia=y[i], archivo=Path(archivos[i]).name, tam_plantilla=tam[grupos[i]],
                      acierto=round(ac[i], 3), error_mas_comun=(errores_a[i].most_common(1)[0][0]
                                                                if errores_a[i] else "")))
filas.sort(key=lambda r: (r["acierto"], r["familia"]))
with open(OUT / "objetivos_p1.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0]))
    w.writeheader()
    w.writerows(filas)

perdidas = (1 - ac) * N_SEM
print(f"Errores totales en {N_SEM} semillas: {perdidas.sum():.0f}")
unicas_mal = [r for r in filas if r["tam_plantilla"] == 1 and r["acierto"] < 0.5]
print(f"\nNotas ÚNICAS que se fallan en más de la mitad de las semillas: {len(unicas_mal)}")
print(f"  {'familia':<14}{'archivo':<44}{'acierto':>8}  la confunde con")
for r in unicas_mal:
    print(f"  {r['familia']:<14}{r['archivo'][:42]:<44}{r['acierto']:>8.2f}  {r['error_mas_comun']}")
recuperable = sum(1 - r["acierto"] for r in filas if r["tam_plantilla"] == 1) / len(y)
print(f"\nTecho si TODAS las notas únicas pasaran a acertarse siempre: +{recuperable:.4f} de exactitud")
techo_obj = sum(1 - r["acierto"] for r in unicas_mal) / len(y)
print(f"Techo si se arreglaran solo las {len(unicas_mal)} de la lista: +{techo_obj:.4f}")
print(f"\nSalida: {OUT / 'objetivos_p1.csv'}")
