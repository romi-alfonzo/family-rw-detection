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

AGREGADO POST HOC (2026-09-28, DESPUES de ver el resultado -- se declara como tal).
La prediccion F5 FALLO: la media de las replicas quedo 0,041 (texto) y 0,046 (cascada) por DEBAJO
del punto estimado. La causa esta identificada y no es un error de calculo: con `labels=familias`
fijo en 30, una remuestra que no incluye NINGUNA plantilla de alguna familia le asigna F1=0 a esa
familia, y ese cero entra igual al promedio macro. Como toda remuestra con reposicion pierde
familias, el macro-F1 remuestreado esta sesgado hacia abajo POR CONSTRUCCION. Es el mismo efecto
que la revision del 2026-09-17 ya habia visto en LOGO (media 0,6436 contra punto 0,6747 con
labels=30; media 0,6736 con labels presentes).
Por eso se reportan AHORA LAS DOS convenciones: labels=30 fijo (CONSERVADORA, intervalo mas ancho
y limite inferior mas bajo) y labels presentes en la remuestra (sin el sesgo, pero con denominador
variable). La conclusion no depende de cual se elija. La que se cita en el informe es la
conservadora.

SEGUNDO AGREGADO POST HOC (2026-09-28, planteado por la sesion hermana -- declarado como tal).
El sesgo de F5 no hay que corregirlo: hay que EVITARLO, y para eso hay que remuestrear la unidad
correcta. Remuestrear las 99 plantillas sin mirar la familia trata al CONJUNTO DE FAMILIAS como
aleatorio, o sea admite replicas donde una familia no existe. Pero las 30 familias NO son una
muestra: estan fijadas por el nucleo canonico de NapierOne, que es una decision de diseno del
trabajo (CLAUDE.md). Lo muestral es QUE PLANTILLAS conseguimos DE CADA familia.
La pregunta que corresponde es: "y si hubieramos conseguido otras plantillas de estas mismas 30
familias?". Se responde con un bootstrap ESTRATIFICADO POR FAMILIA: dentro de cada familia se
remuestrean sus plantillas con reposicion, conservando su cantidad. Ninguna familia desaparece, el
estimando no cambia entre replicas y el sesgo no se genera. Ademas conserva labels=30 fijo, asi que
no hay que pagar el denominador variable de la convencion "labels presentes".
La unidad estratificada es el par (familia, plantilla) y no la plantilla sola: los grupos 6 y 53
mezclan dos familias, y de un par sorteado entran solo las notas de ESA familia. Asi cada familia
conserva su tamano y un grupo mixto no arrastra notas ajenas.
Se reportan las tres. La PRINCIPAL pasa a ser la estratificada; la libre queda al lado como
escenario mas conservador, que ademas trataria al conjunto de familias como muestral.

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
    pares_de_fam = {f: [(f, g) for g in np.unique(grupos[y == f])] for f in familias}
    idx_de_par = {(f, g): np.where((y == f) & (grupos == g))[0]
                  for f in familias for _, g in pares_de_fam[f]}
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
    reps_pres = {"texto": [], "cascada": []}   # convencion "labels presentes", sin el sesgo
    reps_estr = {"texto": [], "cascada": []}   # (c) estratificado por familia
    fam_perdidas = []
    for b in range(args.replicas):
        s = int(rng_b.integers(args.n_semillas))
        elegidas = plantillas[rng_b.integers(0, n_pl, n_pl)]
        idx = np.concatenate([idx_de_plantilla[g] for g in elegidas])
        presentes = np.unique(y[idx])
        fam_perdidas.append(len(familias) - len(presentes))
        # (c) estratificado por familia: se remuestrean los pares (familia, plantilla)
        # DENTRO de cada familia, conservando su cantidad. Ninguna familia desaparece.
        ie = np.concatenate([idx_de_par[pares_de_fam[f][j]]
                             for f in familias
                             for j in rng_b.integers(0, len(pares_de_fam[f]),
                                                     len(pares_de_fam[f]))])
        for capa in ("texto", "cascada"):
            reps[capa].append(f1_score(y[idx], pred[capa][s][idx],
                                       average="macro", labels=familias, zero_division=0))
            reps_estr[capa].append(f1_score(y[ie], pred[capa][s][ie],
                                            average="macro", labels=familias,
                                            zero_division=0))
            reps_pres[capa].append(f1_score(y[idx], pred[capa][s][idx],
                                            average="macro", labels=presentes,
                                            zero_division=0))

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
        vp = np.array(reps_pres[capa])
        lo_p, hi_p = np.percentile(vp, [2.5, 97.5])
        filas[-1].update(
            ic_labels_presentes=f"[{lo_p:.4f}; {hi_p:.4f}]",
            media_labels_presentes=round(float(vp.mean()), 4),
            sesgo_labels_presentes=round(float(vp.mean() - punto[capa]), 4))
        ve = np.array(reps_estr[capa])
        lo_e, hi_e = np.percentile(ve, [2.5, 97.5])
        filas[-1].update(
            ic_estratificado=f"[{lo_e:.4f}; {hi_e:.4f}]",
            ancho_estratificado=round(float(hi_e - lo_e), 4),
            media_estratificado=round(float(ve.mean()), 4),
            sesgo_estratificado=round(float(ve.mean() - punto[capa]), 4),
            estratificado_supera_050="SI" if lo_e > 0.50 else "NO")
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
    print()
    print(f"  SESGO DE labels=30: una remuestra pierde en promedio "
          f"{np.mean(fam_perdidas):.2f} familias de 30, y cada una aporta un F1=0 al macro. "
          f"Por eso la media de las replicas queda por debajo del punto (F5 fallada).")
    for r in filas:
        print(f"    {r['capa']:<9} labels=30 {r['ic_por_plantillas']} "
              f"(sesgo {r['media_bootstrap']-r['punto']:+.4f}) | labels presentes "
              f"{r['ic_labels_presentes']} (sesgo {r['sesgo_labels_presentes']:+.4f})")
    print()
    print("  (c) ESTRATIFICADO POR FAMILIA -- el que corresponde al diseno: las 30 familias",
          "estan fijadas, lo muestral son sus plantillas")
    for r in filas:
        print(f"    {r['capa']:<9} {r['ic_estratificado']}  ancho {r['ancho_estratificado']:.4f}"
              f"  sesgo {r['sesgo_estratificado']:+.4f}  >0,50: {r['estratificado_supera_050']}")
    print(f"\n  AL CITAR: el IC por plantillas es el que corresponde para hablar del CORPUS; el "
          f"IC entre semillas habla de la PARTICION. No son intercambiables.")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
