#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pareado_tamano_igualado.py -- control del confusor de tamaño en pareado_conocida_vs_nueva.py.

EL CONFUSOR. En el pareado, el escenario «conocida» (reparto P1cat) entrena con todas las notas
únicas más la mitad de cada plantilla repetida, unas 112 notas; el escenario «nueva» (P2bal), con
unas 74. Parte de la diferencia (+0,1241 en la cascada) puede venir de tener más datos y no de
conocer la plantilla. Lo detectó la propia sesión que lo midió, después de reportarlo.

EL CONTROL. Igual que el pareado, pero en cada pliegue de «conocida» se sacan del entrenamiento
notas ÚNICAS al azar (rng 40_000 + s) hasta igualar el tamaño del entrenamiento del pliegue
correspondiente de P2bal en la misma semilla. Nunca se saca una nota de plantilla repetida, así
que la hermana de cada nota evaluada sigue en entrenamiento. Mismas 69 notas, mismo código.

PUERTA. P2bal sobre las 149 notas reproduce lo publicado (cascada 0,8123 / 0,7417; texto
0,7191 / 0,6551), y el escenario «nueva» sobre las 69 notas reproduce lo medido en el pareado
(cascada 0,8484; texto 0,7299). Si no, se aborta.

PREREGISTRO (commiteado ANTES de correr, 2026-10-01). Se reporta lo que dé.
  PB-1  Con el tamaño igualado, la cascada sigue acertando 0,95 o más con la plantilla conocida
        (sin igualar fue 0,9725). Fundamento: en P1cat el 85,5 % de las decisiones las resolvió
        la regla con acierto 1,0000, apoyada en los marcadores de la hermana, que no dependen de
        cuántas otras notas haya en el catálogo.
  PB-2  La diferencia conocida - nueva de la cascada sigue siendo de +0,08 o más (sin igualar,
        +0,1241): la mayor parte de la ventaja es conocer la plantilla, no el tamaño.
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

textos, y, archivos, _ = ab.cargar_corpus(ab.CORPUS_DIR)
grupos, _ = ab.agrupar_neardups(textos, ab.UMBRAL_NEARDUP)
textos_arr = np.array(textos, dtype=object)
y, grupos = np.asarray(y), np.asarray(grupos)
familias = np.unique(y)
iocs = [set(ab.extraer_marcadores(t)) for t in textos]
nom_aud = ab.cargar_nombres()
nombres_nota = [nom_aud.get((f, Path(a).name)) for f, a in zip(y, archivos)]
tam = Counter(grupos)
unica = np.array([tam[g] == 1 for g in grupos])
plant_fam = {f: len(set(grupos[y == f])) for f in familias}
par = np.array([tam[grupos[i]] >= 2 and plant_fam[y[i]] >= 2 for i in range(len(y))])
print(f"Notas pareadas: {par.sum()} | únicas disponibles para recortar: {unica.sum()}")


def a_vector(idx, pred):
    v = np.empty(len(y), dtype=object)
    v[idx] = pred
    return v


res = {}
tamanos = []
for s in range(N_SEM):
    nueva = split_p2bal(y, grupos, familias, np.random.default_rng(20_000 + s))
    conocida = particiones_p1cat(grupos, np.random.default_rng(30_000 + s))
    rng = np.random.default_rng(40_000 + s)
    igualada = []
    for (tr_c, te_c), (tr_n, _) in zip(conocida, nueva):
        sobran = len(tr_c) - len(tr_n)
        unicas_tr = [i for i in tr_c if unica[i]]
        assert 0 <= sobran <= len(unicas_tr), f"no se puede igualar: sobran {sobran}"
        quitar = set(rng.choice(unicas_tr, size=sobran, replace=False).tolist())
        tr_ig = np.array([i for i in tr_c if i not in quitar])
        assert len(tr_ig) == len(tr_n)
        igualada.append((tr_ig, te_c))
        tamanos.append((len(tr_c), len(tr_n)))
    for prot, splits in (("nueva", nueva), ("igualada", igualada)):
        idx, p_txt, p_cas, _ = evaluar(splits, textos_arr, y, iocs, nombres_nota, s)
        for sist, pred in (("texto", p_txt), ("cascada", p_cas)):
            v = a_vector(idx, pred)
            if prot == "nueva":
                res.setdefault(("p2bal", sist), []).append(
                    (accuracy_score(y, v), f1_score(y, v, average="macro", labels=familias, zero_division=0)))
            res.setdefault((prot, sist), []).append(accuracy_score(y[par], v[par]))
    if (s + 1) % 10 == 0:
        print(f"  {s+1}/{N_SEM} semillas", flush=True)

tc, tn = np.mean(tamanos, axis=0)
print(f"\nEntrenamiento medio: «conocida» sin igualar {tc:.1f} notas, «nueva» {tn:.1f}; "
      f"«conocida» igualada {tn:.1f}")

print("\n" + "-" * 78 + "\n  PUERTA DE ENTRADA\n" + "-" * 78)
ok = True
for sist, (ac0, f10), par0 in (("cascada", (0.8123, 0.7417), 0.8484), ("texto", (0.7191, 0.6551), 0.7299)):
    ac, f1 = np.mean(res[("p2bal", sist)], axis=0)
    acp = np.mean(res[("nueva", sist)])
    bien = abs(ac - ac0) <= TOL and abs(f1 - f10) <= TOL and abs(acp - par0) <= TOL
    ok &= bien
    print(f"  {sist:<8} P2bal {ac:.4f} / {f1:.4f} ({ac0} / {f10}) | «nueva» en las 69: {acp:.4f} "
          f"({par0})  {'OK' if bien else 'NO REPRODUCE'}")
if not ok:
    sys.exit("ABORTADO: el código no reproduce lo publicado. No se reporta nada.")

print("\n" + "=" * 78 + f"\n  LAS MISMAS {par.sum()} NOTAS, CON EL ENTRENAMIENTO DEL MISMO TAMAÑO "
      f"({N_SEM} semillas)\n" + "=" * 78)
dif = {}
for sist in ("texto", "cascada"):
    con = np.array(res[("igualada", sist)])
    nue = np.array(res[("nueva", sist)])
    d = con - nue
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    dif[sist] = (con.mean(), d.mean())
    print(f"  {sist:<8} conocida {con.mean():.4f} | nueva {nue.mean():.4f} | diferencia {d.mean():+.4f} "
          f"[{d.mean() - h:+.4f}; {d.mean() + h:+.4f}]   (exactitud)")

print("\n  VEREDICTO DEL PREREGISTRO")
print(f"  [{'CUMPLE' if dif['cascada'][0] >= 0.95 else 'FALLA '}] PB-1 cascada conocida >= 0,95   "
      f"{dif['cascada'][0]:.4f}")
print(f"  [{'CUMPLE' if dif['cascada'][1] >= 0.08 else 'FALLA '}] PB-2 diferencia de la cascada >= +0,08   "
      f"{dif['cascada'][1]:+.4f}")
