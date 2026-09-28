#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bootstrap_plantilla_p2bal.py -- IC de la cascada bajo P2bal por remuestreo DE PLANTILLAS.

QUE PROBLEMA RESUELVE. El IC que reportamos hoy para P2bal es ENTRE SEMILLAS: mide cuanto se
mueve la cifra al cambiar la particion. No mide la otra fuente de incertidumbre, que es el
CORPUS: tenemos 149 notas y podriamos haber tenido otras. Remuestrear NOTAS para eso esta mal,
porque las notas de una misma plantilla son casi copias y no son observaciones independientes;
remuestrearlas por separado finge un tamano de muestra que no existe y ESTRECHA el intervalo.
La unidad independiente es la PLANTILLA. Es el mismo criterio con el que se parte el corpus.

La revision independiente del 2026-09-17 ya mostro que la distincion cambia la lectura: para
LOGO, el IC por notas daba [0,567; 0,712] y por plantillas [0,539; 0,722], mas ancho.

COMO SE MIDE. P2bal no es determinista: la particion depende de la semilla. Asi que el
remuestreo es de DOS NIVELES y captura las dos fuentes a la vez:
  por cada replica b:  (1) se sortea una semilla de las 50 ya evaluadas
                       (2) se remuestrean PLANTILLAS con reposicion, y entran TODAS las notas
                           de cada plantilla sorteada (tantas veces como salga la plantilla)
                       (3) se calcula macro-F1 sobre esas notas con las predicciones de esa semilla
El IC es el percentil 2,5-97,5 de las B replicas. Se reporta al lado el IC entre semillas, que
es el que ya esta publicado, para que se vean las dos cosas y no se confundan.

⚠️ PREREGISTRO -- escrito y COMMITEADO ANTES de correr.
  F1. PUERTA: el punto estimado reproduce protocolo_p2bal.py -- texto 0,6551 y cascada 0,7417
      (tolerancia 1e-4, el redondeo de la referencia). Si no, ABORTA.
  F2. El IC por plantillas es MAS ANCHO que el IC entre semillas, en las dos capas. Es la razon
      de ser del experimento: si saliera mas angosto, algo esta mal.
  F3. El limite inferior del IC por plantillas de la CASCADA sigue por encima de 0,50. Si no, la
      frase «supera el umbral con el intervalo entero» hay que condicionarla y se reescribe.
  F4. El limite inferior del IC por plantillas del TEXTO SOLO sigue por encima de 0,50.
  F5. La media de las replicas no se aparta del punto estimado mas de 0,02 en ninguna capa
      (control de sesgo del remuestreo).

CONTROL DE SANIDAD (por el bug que reporto la sesion hermana el 2026-09-28: un remapeo de
etiquetas aplicado a un array y no al otro convirtio aciertos en errores). Aca TODAS las
metricas se calculan con labels=familias fijo, el mismo vector de 30 en los dos lados, y se
verifica que las predicciones guardadas reproduzcan exactamente la cifra publicada antes de
remuestrear nada.

Uso:  python bootstrap_plantilla_p2bal.py [--n-semillas 50] [--replicas 2000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, N_FOLDS, UMBRAL_NEARDUP, agrupar_neardups,
                                   cargar_corpus)
from grafo_marcadores import extraer_marcadores
from protocolo_logo import cargar_nombres, evaluar
from protocolo_p2bal import split_p2bal

RAIZ = _AQUI.parent
OUT_DEF = RAIZ / "4_resultados" / "resultados_bootstrap_plantilla_p2bal"
REF = {"texto": 0.6551, "cascada": 0.7417}
TOL = 1e-4


def ic_t(v):
    v = np.asarray(v, float)
    m, n = float(np.mean(v)), len(v)
    s = float(np.std(v, ddof=1)) if n > 1 else 0.0
    h = t_dist.ppf(0.975, n - 1) * (s / np.sqrt(n)) if n > 1 else 0.0
    return m, m - h, m + h, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=50)
    ap.add_argument("--replicas", type=int, default=2000)
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  IC DE P2bal POR REMUESTREO DE PLANTILLAS (dos niveles)")
    print("=" * 78)
    textos, y, archivos, _ = cargar_corpus(CORPUS_DIR)
    grupos, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    grupos = np.asarray(grupos)
    textos_arr = np.array(textos, dtype=object)
    y = np.asarray(y)
    familias = np.unique(y)                      # vector FIJO de 30, usado en TODAS las metricas
    iocs = [set(extraer_marcadores(t)) for t in textos]
    nom = cargar_nombres()
    nombres_nota = [nom.get((f, Path(a).name)) for f, a in zip(y, archivos)]
    plantillas = np.unique(grupos)
    idx_de_plantilla = {g: np.where(grupos == g)[0] for g in plantillas}
    print(f"Notas: {len(y)} | Familias: {len(familias)} | Plantillas: {len(plantillas)}")
    print(f"Semillas: {args.n_semillas} | Replicas bootstrap: {args.replicas}\n")

    # ---- predicciones por semilla (se guardan para remuestrear sobre ellas) ----
    pred = {"texto": [], "cascada": []}
    print("Evaluando ...")
    for s in range(args.n_semillas):
        rng = np.random.default_rng(20_000 + s)   # MISMA siembra que protocolo_p2bal.py
        splits = split_p2bal(y, grupos, familias, rng, N_FOLDS)
        p_txt, p_m6, _ = evaluar(splits, textos_arr, y, iocs, nombres_nota, s)
        pred["texto"].append(np.asarray(p_txt, dtype=object))
        pred["cascada"].append(np.asarray(p_m6, dtype=object))
        if (s + 1) % 10 == 0:
            print(f"  {s+1}/{args.n_semillas} semillas")

    # ---- puerta F1 ----
    punto, entre_sem = {}, {}
    print("\n" + "-" * 78)
    print("  PUERTA F1 -- las predicciones guardadas deben reproducir protocolo_p2bal.py")
    print("-" * 78)
    ok = True
    for capa in ("texto", "cascada"):
        v = [f1_score(y, p, average="macro", labels=familias, zero_division=0)
             for p in pred[capa]]
        m, lo, hi, sd = ic_t(v)
        punto[capa] = m
        entre_sem[capa] = (lo, hi, sd)
        d = abs(m - REF[capa])
        print(f"  {capa:<8} {m:.6f}  vs referencia {REF[capa]}  diferencia {d:.2e}  "
              f"{'OK' if d <= TOL else 'NO COINCIDE'}")
        ok = ok and d <= TOL
    if not ok:
        sys.exit("ABORTADO (F1): las predicciones no reproducen la cifra publicada.")

    # ---- bootstrap de dos niveles por plantilla ----
    rng_b = np.random.default_rng(2026)
    n_pl = len(plantillas)
    reps = {"texto": [], "cascada": []}
    for b in range(args.replicas):
        s = int(rng_b.integers(args.n_semillas))
        elegidas = plantillas[rng_b.integers(0, n_pl, n_pl)]
        idx = np.concatenate([idx_de_plantilla[g] for g in elegidas])
        for capa in ("texto", "cascada"):
            reps[capa].append(f1_score(y[idx], pred[capa][s][idx],
                                       average="macro", labels=familias, zero_division=0))

    filas = []
    print("\n=== RESULTADO ===")
    print(f"{'capa':<9}{'punto':>9}{'IC entre semillas':>26}{'IC por plantillas':>26}"
          f"{'sd boot':>9}{'>0,50':>8}")
    for capa in ("texto", "cascada"):
        v = np.array(reps[capa])
        lo_b, hi_b = np.percentile(v, [2.5, 97.5])
        lo_s, hi_s, sd_s = entre_sem[capa]
        ancho_s, ancho_b = hi_s - lo_s, hi_b - lo_b
        filas.append(dict(
            capa=capa, punto=round(punto[capa], 4),
            ic_entre_semillas=f"[{lo_s:.4f}; {hi_s:.4f}]", ancho_semillas=round(ancho_s, 4),
            ic_por_plantillas=f"[{lo_b:.4f}; {hi_b:.4f}]", ancho_plantillas=round(ancho_b, 4),
            sd_bootstrap=round(float(v.std(ddof=1)), 4),
            media_bootstrap=round(float(v.mean()), 4),
            frac_replicas_bajo_050=round(float((v < 0.50).mean()), 4),
            supera_050="SI" if lo_b > 0.50 else "NO"))
        print(f"{capa:<9}{punto[capa]:>9.4f}{f'[{lo_s:.4f}; {hi_s:.4f}]':>26}"
              f"{f'[{lo_b:.4f}; {hi_b:.4f}]':>26}{v.std(ddof=1):>9.4f}"
              f"{('SI' if lo_b > 0.50 else 'NO'):>8}")
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "bootstrap_plantilla_p2bal.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    print("=" * 78)
    chk = [("F1 puerta (reproduce protocolo_p2bal.py)", ok, "texto y cascada al 1e-4")]
    for capa in ("texto", "cascada"):
        r = next(x for x in filas if x["capa"] == capa)
        chk.append((f"F2 IC por plantillas mas ancho que entre semillas ({capa})",
                    r["ancho_plantillas"] > r["ancho_semillas"],
                    f"{r['ancho_plantillas']:.4f} vs {r['ancho_semillas']:.4f}"))
    rc = next(x for x in filas if x["capa"] == "cascada")
    rt = next(x for x in filas if x["capa"] == "texto")
    chk += [
        ("F3 cascada: limite inferior > 0,50", rc["supera_050"] == "SI", rc["ic_por_plantillas"]),
        ("F4 texto: limite inferior > 0,50", rt["supera_050"] == "SI", rt["ic_por_plantillas"]),
        ("F5 sesgo del remuestreo <= 0,02",
         all(abs(x["media_bootstrap"] - x["punto"]) <= 0.02 for x in filas),
         " | ".join(f"{x['capa']} {x['media_bootstrap']-x['punto']:+.4f}" for x in filas)),
    ]
    for nombre, cumple, det in chk:
        print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<52} {det}")
    print(f"\n  AL CITAR: el IC por plantillas es el que corresponde para hablar del CORPUS; el "
          f"IC entre semillas habla de la PARTICION. No son intercambiables.")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
