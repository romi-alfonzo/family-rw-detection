#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pareado_conocida_vs_nueva.py -- las MISMAS notas, con y sin su plantilla en el catálogo.

POR QUÉ. P1cat (plantilla en el catálogo) se evalúa sobre 17 familias y P2bal (plantilla nunca
vista) sobre 30 o 28: sus cifras no se pueden poner una al lado de la otra, porque cambian las
notas y las familias. Acá se toma el conjunto de notas que puede estar en los dos escenarios y se
lo mide dos veces por semilla, cambiando solo si la plantilla de la nota está en entrenamiento.

NOTAS PAREADAS. Las de plantillas con 2 o más notas, en familias con 2 o más plantillas (en las de
una sola plantilla, BADRABBIT y CRYPTOLOCKER, el escenario «nunca vista» no deja nada de la
familia en entrenamiento).
  - Conocida: reparto P1cat (rng 30_000 + s), una hermana siempre en entrenamiento.
  - Nueva:    reparto P2bal publicado (rng 20_000 + s), la plantilla entera fuera de entrenamiento.

PUERTA DE ENTRADA. Con este mismo código, P2bal sobre las 149 notas reproduce lo publicado
(cascada 0,8123 / 0,7417; texto 0,7191 / 0,6551) y P1cat reproduce lo medido hoy en
_log_p1_catalogada.txt (cascada 0,9740 / 0,9393; texto 0,9479 / 0,9081). Si no, se aborta.

PREREGISTRO (commiteado ANTES de correr, 2026-10-01). Se reporta lo que dé.
  PA-1  En las notas pareadas, la cascada acierta MÁS con la plantilla en el catálogo que sin
        ella, con una diferencia de exactitud entre +0,05 y +0,20.
  PA-2  La diferencia es MAYOR para el texto solo que para la cascada. Fundamento: la capa de
        reglas se apoya en marcadores que se repiten entre plantillas de la misma familia, así que
        depende menos de haber visto la plantilla (bajo P2bal la cascada suma +0,0932 sobre el
        texto; bajo P1cat, solo +0,0261).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstencion_notas as ab  # noqa: E402
from p1_catalogada import evaluar, particiones_p1cat  # noqa: E402
from protocolo_p2bal import split_p2bal  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

N_SEM = int(sys.argv[1]) if len(sys.argv) > 1 else 50
TOL = 0.0005
PUERTA = {("P2bal", "cascada"): (0.8123, 0.7417), ("P2bal", "texto"): (0.7191, 0.6551),
          ("P1cat", "cascada"): (0.9740, 0.9393), ("P1cat", "texto"): (0.9479, 0.9081)}

textos, y, archivos, _ = ab.cargar_corpus(ab.CORPUS_DIR)
grupos, _ = ab.agrupar_neardups(textos, ab.UMBRAL_NEARDUP)
textos_arr = np.array(textos, dtype=object)
y, grupos = np.asarray(y), np.asarray(grupos)
familias = np.unique(y)
iocs = [set(ab.extraer_marcadores(t)) for t in textos]
nom_aud = ab.cargar_nombres()
nombres_nota = [nom_aud.get((f, Path(a).name)) for f, a in zip(y, archivos)]
tam = Counter(grupos)
plant_fam = {f: len(set(grupos[y == f])) for f in familias}
par = np.array([tam[grupos[i]] >= 2 and plant_fam[y[i]] >= 2 for i in range(len(y))])
print(f"Notas pareadas: {par.sum()} de {len(y)}, en {len(set(y[par]))} familias")


def a_vector(idx, pred):
    v = np.empty(len(y), dtype=object)
    v[idx] = pred
    return v


m = {k: [] for k in ("gate", "par")}
res = {}
for s in range(N_SEM):
    esquemas = {"P1cat": particiones_p1cat(grupos, np.random.default_rng(30_000 + s)),
                "P2bal": split_p2bal(y, grupos, familias, np.random.default_rng(20_000 + s))}
    for prot, splits in esquemas.items():
        idx, p_txt, p_cas, _ = evaluar(splits, textos_arr, y, iocs, nombres_nota, s)
        for sist, pred in (("texto", p_txt), ("cascada", p_cas)):
            v = a_vector(idx, pred)
            ev = np.zeros(len(y), bool)
            ev[idx] = True
            res.setdefault(("gate", prot, sist), []).append(
                (accuracy_score(y[ev], v[ev]), f1_score(y[ev], v[ev], average="macro", zero_division=0)
                 if prot == "P1cat" else
                 f1_score(y, v, average="macro", labels=familias, zero_division=0)))
            res.setdefault(("par", prot, sist), []).append(
                (accuracy_score(y[par], v[par]), f1_score(y[par], v[par], average="macro", zero_division=0)))
    if (s + 1) % 10 == 0:
        print(f"  {s+1}/{N_SEM} semillas", flush=True)

print("\n" + "-" * 78 + "\n  PUERTA DE ENTRADA\n" + "-" * 78)
ok = True
for (prot, sist), (ac0, f10) in PUERTA.items():
    ac, f1 = np.mean(res[("gate", prot, sist)], axis=0)
    bien = abs(ac - ac0) <= TOL and abs(f1 - f10) <= TOL
    ok &= bien
    print(f"  {prot:<6} {sist:<8} exactitud {ac:.4f} ({ac0}) | macro-F1 {f1:.4f} ({f10})  "
          f"{'OK' if bien else 'NO REPRODUCE'}")
if not ok:
    sys.exit("ABORTADO: el código no reproduce lo publicado. No se reporta nada.")

print("\n" + "=" * 78 + f"\n  LAS MISMAS {par.sum()} NOTAS, CON Y SIN SU PLANTILLA EN EL CATÁLOGO "
      f"({N_SEM} semillas)\n" + "=" * 78)
print(f"  {'sistema':<9}{'conocida':>11}{'nueva':>9}{'diferencia':>12}   IC 95 % de la diferencia")
dif = {}
for sist in ("texto", "cascada"):
    con = np.array(res[("par", "P1cat", sist)])
    nue = np.array(res[("par", "P2bal", sist)])
    d = con[:, 0] - nue[:, 0]
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    dif[sist] = d.mean()
    print(f"  {sist:<9}{con[:, 0].mean():>11.4f}{nue[:, 0].mean():>9.4f}{d.mean():>+12.4f}   "
          f"[{d.mean() - h:+.4f}; {d.mean() + h:+.4f}]   (exactitud)")
    print(f"  {'':<9}{con[:, 1].mean():>11.4f}{nue[:, 1].mean():>9.4f}"
          f"{con[:, 1].mean() - nue[:, 1].mean():>+12.4f}   (macro-F1, {len(set(y[par]))} familias)")

print("\n  VEREDICTO DEL PREREGISTRO")
print(f"  [{'CUMPLE' if 0.05 <= dif['cascada'] <= 0.20 else 'FALLA '}] PA-1 cascada: diferencia "
      f"entre +0,05 y +0,20   {dif['cascada']:+.4f}")
print(f"  [{'CUMPLE' if dif['texto'] > dif['cascada'] else 'FALLA '}] PA-2 la diferencia del texto "
      f"supera a la de la cascada   {dif['texto']:+.4f} frente a {dif['cascada']:+.4f}")
