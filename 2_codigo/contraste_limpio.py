#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contraste_limpio.py -- conocer o no la plantilla, cambiando SOLO las hermanas de la nota.

POR QUÉ. El pareado (P1cat contra P2bal) difiere en tres cosas: si la plantilla de la nota está en
entrenamiento, el tamaño del entrenamiento y cuántas OTRAS plantillas de la familia quedan en él.
Igualar el tamaño corrige la segunda pero no la tercera. Diseño propuesto por la sesión de revisión
(2026-10-01): sobre las mismas 69 notas, y sin semillas,
  - conocida: se deja afuera SOLO esa nota (entrenamiento = las otras 148);
  - nueva:    se deja afuera TODA su plantilla (el pliegue de LeaveOneGroupOut).
Lo único que cambia entre los dos brazos son las hermanas de la propia plantilla. El NIVEL no es
comparable con P1 ni con P2bal, porque se entrena con unas 140 notas; el CONTRASTE sí queda limpio.
Límite conocido: el brazo «nueva» hereda la contención entre plantillas (al fusionar por contención
>= 0,8, el texto bajo LOGO bajaba de 0,6747 a 0,4186), lo que lo infla: el contraste es conservador.

MÉTRICAS. Exactitud, y macro-F1 con labels = las familias de las notas evaluadas (el reparo 3 de la
revisión: sin labels=, f1_score promedia sobre la unión de etiquetas verdaderas y predichas, y cada
familia de afuera que recibe una predicción entra con F1 0). Incertidumbre por remuestreo de las
plantillas (las notas de una plantilla no son independientes). Se informa también sin las dos
plantillas que mezclan familias (DHARMA–PHOBOS y BLACKBASTA–CONTI), cuyo error es ambigüedad de
etiqueta por construcción.

PUERTA. El brazo «nueva» sobre las 149 notas es LeaveOneGroupOut con la semilla 0 y tiene que
reproducir _log_m3_149_LOGO.txt: cascada 0,8591 / 0,7742, texto 0,7383 / 0,6747. Si no, se aborta.

PREREGISTRO (commiteado ANTES de correr, 2026-10-01). Se reporta lo que dé.
  PL-1  La cascada acierta más con la plantilla conocida que con la nueva (diferencia > 0).
  PL-2  La diferencia del texto solo es mayor que la de la cascada.
  PL-3  La diferencia de la cascada es MENOR que en el pareado (+0,1241), porque acá la familia
        conserva todas sus otras plantillas y el brazo «nueva» hereda la contención.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstencion_notas as ab  # noqa: E402
from p1_catalogada import evaluar  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TOL = 0.0005
S = 0
PUERTA = {"cascada": (0.8591, 0.7742), "texto": (0.7383, 0.6747)}

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
mixta = {g for g in set(grupos) if len(set(y[grupos == g])) > 1}
todos = np.arange(len(y))
print(f"Notas pareadas: {par.sum()} en {len(set(grupos[par]))} plantillas y {len(set(y[par]))} familias "
      f"| plantillas mixtas: {len(mixta)}")


def a_vector(idx, pred):
    v = np.empty(len(y), dtype=object)
    v[idx] = pred
    return v


# brazo «nueva»: LeaveOneGroupOut sobre las 149 notas (sirve también de puerta)
loto = [(todos[grupos != g], todos[grupos == g]) for g in sorted(set(grupos))]
idx, p_txt, p_cas, _ = evaluar(loto, textos_arr, y, iocs, nombres_nota, S)
nueva = {"texto": a_vector(idx, p_txt), "cascada": a_vector(idx, p_cas)}

print("\n" + "-" * 78 + "\n  PUERTA DE ENTRADA: LeaveOneGroupOut publicado, con este mismo código\n" + "-" * 78)
ok = True
for sist, (ac0, f10) in PUERTA.items():
    ac = accuracy_score(y, nueva[sist])
    f1 = f1_score(y, nueva[sist], average="macro", zero_division=0)
    bien = abs(ac - ac0) <= TOL and abs(f1 - f10) <= TOL
    ok &= bien
    print(f"  {sist:<8} exactitud {ac:.4f} ({ac0}) | macro-F1 {f1:.4f} ({f10})  {'OK' if bien else 'NO REPRODUCE'}")
if not ok:
    sys.exit("ABORTADO: el código no reproduce lo publicado. No se reporta nada.")

# brazo «conocida»: se deja afuera solo la nota
lono = [(todos[todos != i], np.array([i])) for i in np.where(par)[0]]
idx, p_txt, p_cas, _ = evaluar(lono, textos_arr, y, iocs, nombres_nota, S)
conocida = {"texto": a_vector(idx, p_txt), "cascada": a_vector(idx, p_cas)}

rng = np.random.default_rng(50_000)


def informar(mask, titulo):
    labs = sorted(set(y[mask]))
    plantillas = sorted(set(grupos[mask]))
    print("\n" + "=" * 78 + f"\n  {titulo}: {mask.sum()} notas, {len(plantillas)} plantillas, {len(labs)} familias\n"
          + "=" * 78)
    difs = {}
    for sist in ("texto", "cascada"):
        c, n_ = conocida[sist], nueva[sist]
        ac_c, ac_n = accuracy_score(y[mask], c[mask]), accuracy_score(y[mask], n_[mask])
        f_c = f1_score(y[mask], c[mask], average="macro", labels=labs, zero_division=0)
        f_n = f1_score(y[mask], n_[mask], average="macro", labels=labs, zero_division=0)
        # remuestreo de plantillas para el IC de la diferencia de exactitud
        acierto_c = (c == y) & mask
        acierto_n = (n_ == y) & mask
        boot = []
        for _ in range(2000):
            elegidas = rng.choice(plantillas, size=len(plantillas), replace=True)
            sel = np.concatenate([np.where((grupos == g) & mask)[0] for g in elegidas])
            boot.append(acierto_c[sel].mean() - acierto_n[sel].mean())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        difs[sist] = ac_c - ac_n
        print(f"  {sist:<8} conocida {ac_c:.4f} | nueva {ac_n:.4f} | diferencia {ac_c - ac_n:+.4f} "
              f"[{lo:+.4f}; {hi:+.4f}]   (exactitud)")
        print(f"  {'':<8} conocida {f_c:.4f} | nueva {f_n:.4f} | diferencia {f_c - f_n:+.4f}   "
              f"(macro-F1 sobre {len(labs)} familias)")
    return difs


d = informar(par, "LAS MISMAS NOTAS, CAMBIANDO SOLO SUS HERMANAS")
informar(par & ~np.isin(grupos, list(mixta)), "SIN LAS DOS PLANTILLAS MIXTAS")

print("\n  VEREDICTO DEL PREREGISTRO")
print(f"  [{'CUMPLE' if d['cascada'] > 0 else 'FALLA '}] PL-1 cascada: conocida > nueva   {d['cascada']:+.4f}")
print(f"  [{'CUMPLE' if d['texto'] > d['cascada'] else 'FALLA '}] PL-2 diferencia del texto > la de la cascada   "
      f"{d['texto']:+.4f} frente a {d['cascada']:+.4f}")
print(f"  [{'CUMPLE' if d['cascada'] < 0.1241 else 'FALLA '}] PL-3 diferencia de la cascada < la del pareado (+0,1241)   "
      f"{d['cascada']:+.4f}")
