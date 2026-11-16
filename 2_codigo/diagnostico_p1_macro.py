#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diagnostico_p1_macro.py -- macro-F1 de la cascada ante una casi-copia de una plantilla conocida (P1).

Pedido de Romina (2026-11-16): en la tesis, el caso de la plantilla nunca vista lleva macro-F1 y exactitud (0,7417 y
0,8123, P2bal), pero el de la casi-copia solo exactitud (0,9773), porque diagnostico_p1.py guardó solo aciertos y
decisiones. Este script repite el mismo bucle que diagnostico_p1.py (cascada de abstencion_notas.py bajo P1, mismas
semillas y particiones) y guarda la predicción de cada nota en cada semilla, para calcular también el macro-F1 de
cada grupo: con una hermana de su plantilla en entrenamiento (casi-copia) y sin ella.

No cambia ningún cálculo existente. Controles (deben coincidir con la tesis):
  - exactitud con hermana en entrenamiento = 3.015 / 3.085 = 0,9773; sin hermana = 3.590 / 4.365 = 0,8225;
  - macro-F1 de todas las decisiones, media de las semillas = 0,8592 (Tabla «cascada frente al texto solo», P1).
El macro-F1 de un grupo se calcula sobre las familias que tienen al menos una nota en ese grupo (se informa cuántas).
Salida: 4_resultados/resultados_diagnostico_P1_macro/ (predicciones por nota y semilla, y resumen).
Uso: python diagnostico_p1_macro.py [semillas=50]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstencion_notas as ab  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

N_SEM = int(sys.argv[1]) if len(sys.argv) > 1 else 50
OUT = ab.RAIZ / "4_resultados" / "resultados_diagnostico_P1_macro"
OUT.mkdir(parents=True, exist_ok=True)

textos, y, archivos, _ = ab.cargar_corpus(ab.CORPUS_DIR)
grupos, _ = ab.agrupar_neardups(textos, ab.UMBRAL_NEARDUP)
textos_arr = np.array(textos, dtype=object)
y = np.asarray(y)
grupos = np.asarray(grupos)
iocs = [set(ab.extraer_marcadores(t)) for t in textos]
nom_aud = ab.cargar_nombres()
nombres_nota = [nom_aud.get((f, Path(a).name)) for f, a in zip(y, archivos)]

filas = []
for s in range(N_SEM):
    cv = StratifiedKFold(n_splits=ab.N_FOLDS, shuffle=True, random_state=s)
    for tr, te in cv.split(textos_arr, y):
        vec = ab.vectorizador("combinado")
        Xtr = vec.fit_transform(textos_arr[tr])
        Xte = vec.transform(textos_arr[te])
        clf = ab.obtener_modelos(s)["LinearSVC"]
        clf.fit(Xtr, y[tr])
        top1 = clf.classes_[np.argmax(clf.decision_function(Xte), axis=1)]
        d = ab.dicc_privados(tr, iocs, nombres_nota, y)
        grupos_tr = set(grupos[tr])
        for k, i in enumerate(te):
            claves = set(iocs[i])
            if nombres_nota[i]:
                claves.add(("[NOMBRE]", nombres_nota[i]))
            fams = set()
            for c in claves:
                if c in d:
                    fams |= d[c]
            if len(fams) == 1:
                capa, pred = "regla", next(iter(fams))
            else:
                capa, pred = "texto", top1[k]
            filas.append(dict(semilla=s, nota=int(i), familia=y[i], prediccion=pred, capa=capa,
                              hermana_en_train=int(grupos[i] in grupos_tr)))

df = pd.DataFrame(filas)
df.to_csv(OUT / "predicciones_P1.csv", index=False, encoding="utf-8")


def macro(sub):
    fams = sorted(sub["familia"].unique())
    return f1_score(sub["familia"], sub["prediccion"], labels=fams, average="macro", zero_division=0), len(fams)


lineas = [f"Semillas: {N_SEM} | decisiones: {len(df)} (149 notas x semillas)", ""]
for nombre, sub in (("todas las decisiones (P1)", df), ("con hermana en entrenamiento (casi-copia)", df[df.hermana_en_train == 1]),
                    ("sin hermana en entrenamiento", df[df.hermana_en_train == 0])):
    ac = int((sub.familia == sub.prediccion).sum())
    mf_total, nf_total = macro(sub)
    por_sem = [macro(g)[0] for _, g in sub.groupby("semilla")]
    nf_sem = [macro(g)[1] for _, g in sub.groupby("semilla")]
    lineas += [f"{nombre}:",
               f"  exactitud {ac} / {len(sub)} = {ac / len(sub):.4f}",
               f"  macro-F1, media de las semillas = {np.mean(por_sem):.4f} (desvío {np.std(por_sem):.4f}); "
               f"familias por semilla: {min(nf_sem)} a {max(nf_sem)}",
               f"  macro-F1 con todas las semillas juntas = {mf_total:.4f}, sobre {nf_total} familias", ""]
texto = "\n".join(lineas)
print(texto)
(OUT / "resumen_P1_macro.txt").write_text(texto + "\n", encoding="utf-8")
print(f"Salida: {OUT}")
