#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p2bal_evaluables.py -- el escenario de plantilla NUNCA vista, sobre las familias donde existe.

POR QUÉ. Es el espejo de p1_catalogada.py. Allí se evaluaron solo las familias donde «plantilla
conocida» puede existir (17). Acá, solo las familias donde «plantilla nunca vista con la familia
representada» puede existir: las que tienen 2 o más plantillas (28). BADRABBIT y CRYPTOLOCKER
tienen una sola plantilla: cuando se evalúan no queda nada de su familia en entrenamiento, y su
cero es estructural (§4.12.5), no una medida de dificultad.

Mismo reparto P2bal publicado (protocolo_p2bal.py, rng 20_000 + s), mismo clasificador y misma
capa de reglas. Además descompone el error por capa y por las familias de las tres parejas de
linaje (DHARMA–PHOBOS, BLACKBASTA–CONTI, CLOP–RYUK).

PUERTA DE ENTRADA. Con este mismo código, P2bal sobre las 149 notas tiene que reproducir la
cascada en 0,8123 / 0,7417 y el texto en 0,7191 / 0,6551 (exactitud / macro-F1), y el macro-F1
publicado sobre las 28 evaluables, 0,7946 y 0,7019 (labels = evaluables, todas las notas).

PREREGISTRO (commiteado ANTES de correr, 2026-10-01). Las dos primeras son casi aritméticas: si
las 4 notas de plantilla única fallan siempre, la exactitud sobre las 145 restantes es la publicada
multiplicada por 149/145. Se registran igual, como control de consistencia.
  PU-1  Cascada, exactitud sobre las 145 notas evaluables: 0,835 ± 0,005.
  PU-2  Texto solo, exactitud sobre las 145 notas evaluables: 0,739 ± 0,005.
  PU-3  El texto aporta el 80 % o más del error de la cascada (bajo P1 fue el 97 %).
  PU-4  Las seis familias de las parejas de linaje aportan el 25 % o más del error (§4.12.16:
        alrededor de una cuarta parte del error es confusión dentro de un par).
ADVERTENCIA AL CITAR. El macro-F1 se promedia sobre 28 familias; no es comparable con el de P1cat
(17) ni con el de 30. La exactitud sí.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstencion_notas as ab  # noqa: E402
from p1_catalogada import evaluar  # noqa: E402
from protocolo_p2bal import split_p2bal  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PUERTA = {"cascada": (0.8123, 0.7417, 0.7946), "texto": (0.7191, 0.6551, 0.7019)}
TOL = 0.0005
LINAJE = {"DHARMA", "PHOBOS", "BLACKBASTA", "CONTI", "CLOP", "RYUK"}
N_SEM = int(sys.argv[1]) if len(sys.argv) > 1 else 50

textos, y, archivos, _ = ab.cargar_corpus(ab.CORPUS_DIR)
grupos, _ = ab.agrupar_neardups(textos, ab.UMBRAL_NEARDUP)
textos_arr = np.array(textos, dtype=object)
y, grupos = np.asarray(y), np.asarray(grupos)
familias = np.unique(y)
iocs = [set(ab.extraer_marcadores(t)) for t in textos]
nom_aud = ab.cargar_nombres()
nombres_nota = [nom_aud.get((f, Path(a).name)) for f, a in zip(y, archivos)]
plant_por_fam = {f: len(set(grupos[y == f])) for f in familias}
evaluables = [f for f in familias if plant_por_fam[f] >= 2]
ev = np.isin(y, evaluables)
print(f"Notas: {len(y)} | familias evaluables: {len(evaluables)} | notas evaluables: {ev.sum()}")

m = defaultdict(list)
err = Counter()
for s in range(N_SEM):
    splits = split_p2bal(y, grupos, familias, np.random.default_rng(20_000 + s))
    idx, p_txt, p_cas, reg = evaluar(splits, textos_arr, y, iocs, nombres_nota, s)
    orden = np.argsort(idx)
    idx, p_txt, p_cas, reg = idx[orden], p_txt[orden], p_cas[orden], reg[orden]
    assert (idx == np.arange(len(y))).all(), "P2bal tiene que evaluar cada nota una vez por semilla"
    for nombre, pred in (("texto", p_txt), ("cascada", p_cas)):
        m[nombre + "_ac149"].append(accuracy_score(y, pred))
        m[nombre + "_f1_30"].append(f1_score(y, pred, average="macro", labels=familias, zero_division=0))
        m[nombre + "_f1_28pub"].append(f1_score(y, pred, average="macro", labels=evaluables, zero_division=0))
        m[nombre + "_ac145"].append(accuracy_score(y[ev], pred[ev]))
        m[nombre + "_f1_145"].append(f1_score(y[ev], pred[ev], average="macro", zero_division=0))
    mal = ev & (p_cas != y)
    for i in np.where(mal)[0]:
        err["regla" if reg[i] else "texto"] += 1
        err["linaje" if y[i] in LINAJE else "resto"] += 1
    if (s + 1) % 10 == 0:
        print(f"  {s+1}/{N_SEM} semillas", flush=True)

r = {k: float(np.mean(v)) for k, v in m.items()}
print("\n" + "-" * 78 + "\n  PUERTA DE ENTRADA: P2bal publicado, con este mismo código\n" + "-" * 78)
ok = True
for sist, (ac0, f30, f28) in PUERTA.items():
    vals = (r[sist + "_ac149"], r[sist + "_f1_30"], r[sist + "_f1_28pub"])
    bien = all(abs(a - b) <= TOL for a, b in zip(vals, (ac0, f30, f28)))
    ok &= bien
    print(f"  {sist:<8} exactitud {vals[0]:.4f} ({ac0}) | macro-F1 30 {vals[1]:.4f} ({f30}) | "
          f"macro-F1 28 {vals[2]:.4f} ({f28})  {'OK' if bien else 'NO REPRODUCE'}")
if not ok:
    sys.exit("ABORTADO: el código no reproduce P2bal publicado. No se reporta nada.")

print("\n" + "=" * 78 + f"\n  PLANTILLA NUNCA VISTA, sobre las {len(evaluables)} familias evaluables "
      f"({ev.sum()} notas, {N_SEM} semillas)\n" + "=" * 78)
for sist in ("texto", "cascada"):
    print(f"  {sist:<8} exactitud {r[sist + '_ac145']:.4f} | macro-F1 {r[sist + '_f1_145']:.4f}")
tot = err["regla"] + err["texto"]
print(f"\n  Error de la cascada sobre las evaluables: {tot} decisiones erradas")
print(f"    por capa : texto {err['texto']} ({err['texto']/tot:.0%}) | regla {err['regla']} ({err['regla']/tot:.0%})")
print(f"    por familia: parejas de linaje {err['linaje']} ({err['linaje']/tot:.0%}) | resto {err['resto']} ({err['resto']/tot:.0%})")

print("\n  VEREDICTO DEL PREREGISTRO")
for cod, desc, val, cumple in (
        ("PU-1", "cascada 0,835 ± 0,005", r["cascada_ac145"], abs(r["cascada_ac145"] - 0.835) <= 0.005),
        ("PU-2", "texto 0,739 ± 0,005", r["texto_ac145"], abs(r["texto_ac145"] - 0.739) <= 0.005),
        ("PU-3", "texto >= 80 % del error", err["texto"] / tot, err["texto"] / tot >= 0.80),
        ("PU-4", "linaje >= 25 % del error", err["linaje"] / tot, err["linaje"] / tot >= 0.25)):
    print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {cod} {desc:<26} {val:.4f}")
