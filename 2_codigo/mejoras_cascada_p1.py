#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mejoras_cascada_p1.py -- dos mejoras de la cascada bajo P1, sin sacar ninguna familia.

POR QUÉ. Romina (2026-10-01) fijó la cascada como cifra principal en los dos escenarios y pidió
intentar mejorarla con plantilla «conocida», sin achicar familias. Diseño propuesto por la sesión
de revisión.

PROTOCOLO. P1 exacto: StratifiedKFold de 2 pliegues con mezcla, random_state = s, 50 semillas,
149 notas, 30 familias. Mismo clasificador y misma capa de reglas que abstencion_notas.py.
  V0  cascada actual: reglas -> texto.
  V1  + capa de catálogo por similitud, después de las reglas y antes del texto: si el coseno
      máximo entre la nota y alguna nota de entrenamiento es >= 0,90 (TF-IDF char_wb 3-5 con la
      configuración TFIDF_CHAR, ajustado SOLO con el pliegue de entrenamiento), se asigna la
      familia de esa nota. El 0,90 es el umbral que define «plantilla» y no se ajusta.
  V2  + normalización de URL en el filtro de genéricos, modo «B suave» de filtro_genericos_url.py
      (la que da 0,7492 bajo P2bal).
  V3  V1 + V2.
Exactitud y macro-F1 (labels = las 30 familias); diferencias pareadas contra V0 con IC t sobre 50
semillas; decisiones cambiadas por capa; desglose con o sin hermana en entrenamiento.

PUERTA. V0 tiene que reproducir la cascada publicada bajo P1: 0,8866 / 0,8592. Si no, se aborta.

PREREGISTRO (commiteado ANTES de correr, 2026-10-01). El techo de V1 sale de
_log_diagnostico_P1_50sem.txt: las decisiones con hermana en entrenamiento que hoy decide el
texto son 379, con acierto 0,8285, o sea unos 65 errores sobre 7450 decisiones.
  PM-1  V1 mueve la exactitud a lo sumo 0,009 en valor absoluto, y lo esperable es menos de
        +0,005: muchos de esos errores están en las plantillas mixtas, donde la nota parecida es
        de otra familia. Con el IDF ajustado solo al pliegue, V1 también puede dispararse en notas
        sin hermana, para bien o para mal.
  PM-2  V2 da entre 0 y +0,015 de macro-F1.
  PM-3  V3 queda a menos de 0,002 de V1 + V2 en exactitud, porque actúan sobre notas distintas.
LO QUE HAY QUE DECIR PASE LO QUE PASE. El 89,7 % del error de la cascada bajo P1 está en notas SIN
su plantilla en entrenamiento, decididas por el texto. Ninguna técnica de catálogo puede recuperar
más de alrededor de 0,9 puntos: el resto es el problema de «nunca vista».
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstencion_notas as ab  # noqa: E402
from clasificador_notas_v2 import TFIDF_CHAR  # noqa: E402
from filtro_genericos_url import marcadores_modo  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

N_SEM = int(sys.argv[1]) if len(sys.argv) > 1 else 50
UMBRAL = 0.90
TOL = 0.0005
VARIANTES = ("V0", "V1", "V2", "V3")

textos, y, archivos, _ = ab.cargar_corpus(ab.CORPUS_DIR)
grupos, _ = ab.agrupar_neardups(textos, ab.UMBRAL_NEARDUP)
textos_arr = np.array(textos, dtype=object)
y, grupos = np.asarray(y), np.asarray(grupos)
familias = np.unique(y)
n = len(y)
iocs = {"A": [set(ab.extraer_marcadores(t)) for t in textos],
        "B": [marcadores_modo(t, "B") for t in textos]}
nom_aud = ab.cargar_nombres()
nombres_nota = [nom_aud.get((f, Path(a).name)) for f, a in zip(y, archivos)]


def regla(i, d, iocs_i):
    claves = set(iocs_i)
    if nombres_nota[i]:
        claves.add(("[NOMBRE]", nombres_nota[i]))
    fams = set()
    for c in claves:
        if c in d:
            fams |= d[c]
    return next(iter(fams)) if len(fams) == 1 else None


pred = {v: np.empty((N_SEM, n), dtype=object) for v in VARIANTES}
capa = {v: np.empty((N_SEM, n), dtype=object) for v in VARIANTES}
hermana = np.zeros((N_SEM, n), dtype=bool)
for s in range(N_SEM):
    cv = StratifiedKFold(n_splits=ab.N_FOLDS, shuffle=True, random_state=s)
    for tr, te in cv.split(textos_arr, y):
        vec = ab.vectorizador("combinado")
        Xtr = vec.fit_transform(textos_arr[tr])
        Xte = vec.transform(textos_arr[te])
        clf = ab.obtener_modelos(s)["LinearSVC"]
        clf.fit(Xtr, y[tr])
        top1 = clf.classes_[np.argmax(clf.decision_function(Xte), axis=1)]
        vch = TfidfVectorizer(**TFIDF_CHAR).fit(textos_arr[tr])
        sim = (vch.transform(textos_arr[te]) @ vch.transform(textos_arr[tr]).T).toarray()
        mejor = sim.argmax(axis=1)
        dic = {m: ab.dicc_privados(tr, iocs[m], nombres_nota, y) for m in ("A", "B")}
        g_tr = set(grupos[tr])
        for k, i in enumerate(te):
            hermana[s, i] = grupos[i] in g_tr
            for v in VARIANTES:
                modo = "B" if v in ("V2", "V3") else "A"
                fam = regla(i, dic[modo], iocs[modo][i])
                if fam is not None:
                    pred[v][s, i], capa[v][s, i] = fam, "regla"
                elif v in ("V1", "V3") and sim[k, mejor[k]] >= UMBRAL:
                    pred[v][s, i], capa[v][s, i] = y[tr][mejor[k]], "catalogo"
                else:
                    pred[v][s, i], capa[v][s, i] = top1[k], "texto"
    if (s + 1) % 10 == 0:
        print(f"  {s+1}/{N_SEM} semillas", flush=True)

ac = {v: np.array([accuracy_score(y, pred[v][s]) for s in range(N_SEM)]) for v in VARIANTES}
f1 = {v: np.array([f1_score(y, pred[v][s], average="macro", labels=familias, zero_division=0)
                   for s in range(N_SEM)]) for v in VARIANTES}

print("\n" + "-" * 78 + "\n  PUERTA DE ENTRADA: V0 = cascada publicada bajo P1\n" + "-" * 78)
ok = abs(ac["V0"].mean() - 0.8866) <= TOL and abs(f1["V0"].mean() - 0.8592) <= TOL
print(f"  V0 exactitud {ac['V0'].mean():.4f} (0,8866) | macro-F1 {f1['V0'].mean():.4f} (0,8592)  "
      f"{'OK' if ok else 'NO REPRODUCE'}")
if not ok:
    sys.exit("ABORTADO: V0 no reproduce la cascada publicada. No se reporta nada.")


def ic(d):
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h


print("\n" + "=" * 78 + f"\n  P1, 149 notas, 30 familias, {N_SEM} semillas\n" + "=" * 78)
print(f"  {'variante':<10}{'exactitud':>10}{'macro-F1':>10}   delta exactitud [IC 95 %]      delta macro-F1 [IC 95 %]")
deltas = {}
for v in VARIANTES:
    da, la, ha = ic(ac[v] - ac["V0"])
    df, lf, hf = ic(f1[v] - f1["V0"])
    deltas[v] = (da, df)
    print(f"  {v:<10}{ac[v].mean():>10.4f}{f1[v].mean():>10.4f}   {da:+.4f} [{la:+.4f}; {ha:+.4f}]    "
          f"{df:+.4f} [{lf:+.4f}; {hf:+.4f}]")

print("\n  Decisiones que cambian respecto de V0 (en las 50 semillas):")
for v in ("V1", "V2", "V3"):
    cambia = pred[v] != pred["V0"]
    a_bien = cambia & (pred[v] == y)
    a_mal = cambia & (pred["V0"] == y)
    por_capa = Counter(capa[v][cambia])
    print(f"  {v}: {cambia.sum()} cambian | {a_bien.sum()} pasan a acierto, {a_mal.sum()} pasan a error | "
          f"por capa: {dict(por_capa)}")
    for h, etq in ((True, "con hermana"), (False, "sin hermana")):
        m = cambia & (hermana == h)
        print(f"      {etq}: {m.sum()} cambian, {(m & (pred[v] == y)).sum()} a acierto, "
              f"{(m & (pred['V0'] == y)).sum()} a error")
dispara = sum((capa["V1"] == "catalogo").sum(axis=1)) / N_SEM
print(f"\n  La capa de catálogo de V1 decide en promedio {dispara:.1f} de 149 notas por semilla")

print("\n  VEREDICTO DEL PREREGISTRO")
d1, d2, d3 = deltas["V1"][0], deltas["V2"][1], deltas["V3"][0]
print(f"  [{'CUMPLE' if abs(d1) <= 0.009 else 'FALLA '}] PM-1 |delta V1| <= 0,009 de exactitud   {d1:+.4f}"
      f"   (esperado < +0,005: {'sí' if d1 < 0.005 else 'no'})")
print(f"  [{'CUMPLE' if 0 <= d2 <= 0.015 else 'FALLA '}] PM-2 delta V2 entre 0 y +0,015 de macro-F1   {d2:+.4f}")
suma = deltas["V1"][0] + deltas["V2"][0]
print(f"  [{'CUMPLE' if abs(d3 - suma) < 0.002 else 'FALLA '}] PM-3 V3 a menos de 0,002 de V1 + V2   "
      f"{d3:+.4f} frente a {suma:+.4f}")
