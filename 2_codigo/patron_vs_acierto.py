#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patron_vs_acierto.py -- ¿las notas de una misma familia tienen un patron, y eso predice el
rendimiento del clasificador?

DE DONDE SALE LA PREGUNTA. Pedido del tutor (Prof. Cristian Cappo, 2026-09-09): «lo que yo haria
es [ver] si las notas tienen algun patron entre los que dicen que son de la misma clase. Si no
hay ningun patron, entonces se puede deducir que el rendimiento del clasificador no sera bueno».
Este script convierte esa intuicion en un numero verificable.

⚠️ ANALISIS POST-HOC, NO PREREGISTRADO. Se escribe DESPUES de ver los resultados de
similitud_vs_acierto_p2bal.py y hay que reportarlo como tal: no es una prediccion que se puso a
prueba, es una relacion que se midio sobre datos ya vistos. No se cita como confirmacion de una
hipotesis previa. Lo unico que aporta es la magnitud de una relacion que el tutor propuso.

QUE MIDE. No re-corre nada: lee `similitud_por_familia_p2bal.csv`, que ya trae por familia
  - el PATRON INTERNO: contencion media de sus notas contra las notas de su propia familia que
    estaban en el entrenamiento (fraccion de 3-shingles de palabras compartidos). Alto = las
    notas de esa familia se repiten entre si; bajo = cada nota dice algo distinto.
  - el ACIERTO con texto solo y con la cascada, y que fraccion resolvio la capa de reglas.
Y calcula la correlacion entre patron y acierto sobre las 28 familias EVALUABLES.

Las 2 familias de UNA plantilla (BADRABBIT, CRYPTOLOCKER) quedan fuera por construccion: no
tienen con que comparar dentro de su familia, asi que su patron interno no esta definido. Es la
misma razon por la que el tutor dijo «no se procesa» (2026-09-09).

Uso:  python patron_vs_acierto.py [--entrada CSV] [--salida CARPETA]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parent.parent
ENT_DEF = RAIZ / "4_resultados" / "resultados_similitud_p2bal_149" / "similitud_por_familia_p2bal.csv"
OUT_DEF = RAIZ / "4_resultados" / "resultados_similitud_p2bal_149"
COL = "contencion_media_con_entrenamiento"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", type=Path, default=ENT_DEF)
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  ¿HAY PATRON DENTRO DE CADA FAMILIA? ¿Y PREDICE EL RENDIMIENTO?")
    print("  (pedido del tutor, 2026-09-09 -- ANALISIS POST-HOC, no preregistrado)")
    print("=" * 78)
    d = pd.read_csv(args.entrada, encoding="utf-8-sig")
    fuera = d[d.n_plantillas < 2]
    ev = d[d.n_plantillas >= 2].dropna(subset=[COL]).copy()
    print(f"\nFamilias evaluables (2+ plantillas): {len(ev)}")
    print(f"Fuera por tener UNA sola plantilla (patron interno indefinido): "
          f"{sorted(fuera.familia)} -- acierto 0,0000 por construccion\n")

    x = ev[COL].values
    filas = []
    for nom, col in (("texto solo", "acierto_texto"), ("cascada", "acierto_cascada")):
        yv = ev[col].values
        r, p = pearsonr(x, yv)
        rs, ps = spearmanr(x, yv)
        filas.append(dict(sistema=nom, n_familias=len(ev),
                          pearson_r=round(float(r), 3), pearson_p=round(float(p), 6),
                          spearman_rho=round(float(rs), 3), spearman_p=round(float(ps), 6),
                          r2=round(float(r) ** 2, 3)))
        print(f"  patron vs acierto, {nom:<11} Pearson r = {r:+.3f} (p = {p:.5f}) | "
              f"Spearman rho = {rs:+.3f} (p = {ps:.5f})")
    pd.DataFrame(filas).to_csv(args.salida / "patron_vs_acierto_correlacion.csv",
                               index=False, encoding="utf-8-sig")

    r_txt, r_cas = filas[0]["pearson_r"], filas[1]["pearson_r"]
    print(f"\n  LECTURA: la relacion que propuso el tutor EXISTE y es fuerte con el texto solo")
    print(f"  (r = {r_txt:+.3f}). Con la cascada BAJA a r = {r_cas:+.3f}: las capas de reglas")
    print("  aflojan la dependencia del patron textual, que es la razon de ser de la cascada.")

    bajo = ev[ev[COL] < 0.10]
    medio = ev[(ev[COL] >= 0.10) & (ev[COL] < 0.40)]
    alto = ev[ev[COL] >= 0.40]
    tfilas = []
    for etq, sub in (("sin patron (< 0,10)", bajo), ("patron intermedio [0,10; 0,40)", medio),
                     ("con patron (>= 0,40)", alto)):
        if not len(sub):
            continue
        tfilas.append(dict(grupo=etq, n_familias=len(sub),
                           patron_medio=round(float(sub[COL].mean()), 4),
                           acierto_texto=round(float(sub.acierto_texto.mean()), 4),
                           acierto_cascada=round(float(sub.acierto_cascada.mean()), 4),
                           rescate_de_la_regla=round(float(sub.acierto_cascada.mean()
                                                           - sub.acierto_texto.mean()), 4),
                           frac_resuelta_por_regla=round(float(sub.frac_resuelta_por_regla.mean()), 4),
                           familias="|".join(sorted(sub.familia))))
    t = pd.DataFrame(tfilas)
    t.to_csv(args.salida / "patron_vs_acierto_tramos.csv", index=False, encoding="utf-8-sig")
    print("\n=== POR NIVEL DE PATRON INTERNO ===")
    print(t.drop(columns=["familias"]).to_string(index=False))
    for f in tfilas:
        print(f"\n  {f['grupo']}: {f['familias']}")

    print("\n=== LAS QUE CONTRADICEN LA REGLA (las interesantes) ===")
    ev = ev.assign(rescate=ev.acierto_cascada - ev.acierto_texto)
    raras = ev[(ev[COL] < 0.10) & (ev.acierto_cascada >= 0.70)]
    print("  Sin patron de texto y aun asi bien clasificadas -- las salva la capa de reglas:")
    print(raras[["familia", COL, "acierto_texto", "acierto_cascada",
                 "frac_resuelta_por_regla"]].to_string(index=False) if len(raras) else "    (ninguna)")
    duras = ev[(ev[COL] < 0.10) & (ev.acierto_cascada < 0.50)]
    print("\n  Sin patron y sin marcadores utiles -- el limite honesto del corpus:")
    print(duras[["familia", "n_plantillas", COL, "acierto_texto",
                 "acierto_cascada"]].to_string(index=False) if len(duras) else "    (ninguna)")

    print(f"\n  RECORDAR AL CITAR: analisis POST-HOC sobre P2bal, 149 notas, 28 familias")
    print("  evaluables; el patron se mide por contencion de 3-shingles y el acierto es")
    print("  exactitud por familia promediada sobre 50 semillas.")
    print(f"\nSalidas en {args.salida}")


if __name__ == "__main__":
    main()
