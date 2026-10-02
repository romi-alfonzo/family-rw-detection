#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anio_vs_rendimiento.py -- ¿el año de aparición de la familia explica el rendimiento?

DE DONDE SALE. Pedido textual del tutor (Prof. Cristian Cappo), anotado en
`HANDOFF_2026-08-25_limpieza_y_proximos_pasos.md` §3(d): «año de detección vs F1 por familia».

⚠️ NO SALE DE MISP. Verificado: el catálogo MISP solo cubre 5 de las 28 familias evaluables. La
fuente del año es la hoja «Informacion sobre familias» de
`7_compartido_carlos/Tesis Carlos y Romina/Pruebas.xlsx`, que tiene las 30 con año, armada por
Romina y Carlos sobre NapierOne. Es la única fuente citable del proyecto para esta variable.

⚠️ ANALISIS POST-HOC, NO PREREGISTRADO. No hay predicción previa escrita; es una relación que se
explora a pedido. Se reporta como exploración, no como hipótesis puesta a prueba.

LA VARIABLE DE CONFUSION QUE HAY QUE CONTROLAR, y es la razón por la que este análisis no se
puede leer solo: **el número de plantillas**. Si las familias viejas están mejor documentadas,
tienen más plantillas recolectadas, y son las plantillas las que mueven el F1 (B.1). Una
correlación año-F1 podría ser enteramente un reflejo de eso. Por eso se reportan las tres
correlaciones -- año/F1, plantillas/F1 y año/plantillas -- y la **correlación parcial** de año
con F1 descontando las plantillas. Sin eso, el número engaña.

MAPEO DE NOMBRES: explícito y a mano, como en candidatas_de_fuentes_nuevas.py. La hoja escribe
BLACKCAT/alphv, BLACKMATTER7, MEDUSALOCKERb7 y CRYPTOLOCKERc9.

Uso:  python anio_vs_rendimiento.py [--salida CARPETA]
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
XLSX = RAIZ / "7_compartido_carlos" / "Tesis Carlos y Romina" / "Pruebas.xlsx"
F1_CSV = RAIZ / "4_resultados" / "resultados_protocolo_p2bal_149" / "p2bal_por_familia.csv"
SIM_CSV = RAIZ / "4_resultados" / "resultados_similitud_p2bal_149" / "similitud_por_familia_p2bal.csv"
OUT_DEF = RAIZ / "4_resultados" / "resultados_anio_vs_rendimiento"

# Mapeo EXPLICITO hoja -> familia del corpus. A mano a proposito.
ALIAS = {"BLACKCAT/alphv": "BLACKCAT", "BLACKMATTER7": "BLACKMATTER",
         "MEDUSALOCKERb7": "MEDUZALOCKER", "CRYPTOLOCKERc9": "CRYPTOLOCKER"}


def parcial(x, y, z):
    """Correlacion parcial de x con y descontando z (via residuos de regresion lineal)."""
    def resid(a, b):
        b1 = np.polyfit(b, a, 1)
        return a - np.polyval(b1, b)
    rx, ry = resid(np.asarray(x, float), np.asarray(z, float)), resid(np.asarray(y, float), np.asarray(z, float))
    return pearsonr(rx, ry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  AÑO DE APARICION vs RENDIMIENTO POR FAMILIA (pedido del tutor)")
    print("  ANALISIS POST-HOC, no preregistrado")
    print("=" * 78)

    d = pd.read_excel(XLSX, sheet_name="Informacion sobre familias")
    d.columns = ["N", "familia_xlsx", "anio", "tiene_nota", "x", "obs"]
    d = d[["familia_xlsx", "anio", "tiene_nota"]].dropna(subset=["familia_xlsx"])
    d["familia"] = d.familia_xlsx.map(lambda f: ALIAS.get(f, f))

    f1 = pd.read_csv(F1_CSV, encoding="utf-8-sig")[["familia", "n_plantillas", "P2bal_txt", "P2bal_m6"]]
    sim = pd.read_csv(SIM_CSV, encoding="utf-8-sig")[
        ["familia", "n_notas", "contencion_media_con_entrenamiento", "frac_resuelta_por_regla"]]
    m = d.merge(f1, on="familia", how="outer", indicator=True).merge(sim, on="familia", how="left")

    huerf = m[m._merge != "both"]
    if len(huerf):
        print("\n⚠ NO EMPAREJARON (revisar el mapeo antes de leer nada):")
        print(huerf[["familia_xlsx", "familia", "_merge"]].to_string(index=False))
        sys.exit("ABORTADO: hay familias sin emparejar. El mapeo tiene que ser 1 a 1.")
    m = m.drop(columns=["_merge"])
    print(f"\nFamilias emparejadas: {len(m)}/30  (mapeo explicito: {list(ALIAS)})")

    ev = m[m.n_plantillas >= 2].copy()
    print(f"Evaluables (2+ plantillas): {len(ev)}  -- las de 1 plantilla dan F1 0 por")
    print("construccion y meterlas confundiria la correlacion con el conteo de plantillas.\n")

    filas = []
    pares = [("anio", "P2bal_m6", "año vs F1 de la cascada"),
             ("anio", "P2bal_txt", "año vs F1 del texto solo"),
             ("n_plantillas", "P2bal_m6", "plantillas vs F1 de la cascada"),
             ("anio", "n_plantillas", "año vs cantidad de plantillas")]
    for a, b, etq in pares:
        r, p = pearsonr(ev[a], ev[b])
        rs, ps = spearmanr(ev[a], ev[b])
        filas.append(dict(relacion=etq, pearson_r=round(float(r), 3), pearson_p=round(float(p), 5),
                          spearman_rho=round(float(rs), 3), spearman_p=round(float(ps), 5)))
        print(f"  {etq:<34} Pearson r = {r:+.3f} (p = {p:.4f}) | Spearman rho = {rs:+.3f} (p = {ps:.4f})")

    rp, pp = parcial(ev.anio, ev.P2bal_m6, ev.n_plantillas)
    filas.append(dict(relacion="año vs F1 cascada, DESCONTANDO plantillas",
                      pearson_r=round(float(rp), 3), pearson_p=round(float(pp), 5),
                      spearman_rho=np.nan, spearman_p=np.nan))
    print(f"\n  año vs F1 DESCONTANDO el n de plantillas (parcial): r = {rp:+.3f} (p = {pp:.4f})")
    pd.DataFrame(filas).to_csv(OUT / "anio_correlaciones.csv", index=False, encoding="utf-8-sig")

    # por tramo de antiguedad
    tramos = [(2013, 2016, "viejas 2013-2016"), (2017, 2019, "medias 2017-2019"),
              (2020, 2022, "recientes 2020-2022")]
    tf = []
    for lo, hi, etq in tramos:
        sub = ev[(ev.anio >= lo) & (ev.anio <= hi)]
        if not len(sub):
            continue
        tf.append(dict(tramo=etq, n_familias=len(sub),
                       plantillas_media=round(float(sub.n_plantillas.mean()), 2),
                       f1_texto=round(float(sub.P2bal_txt.mean()), 4),
                       f1_cascada=round(float(sub.P2bal_m6.mean()), 4),
                       patron_medio=round(float(sub.contencion_media_con_entrenamiento.mean()), 4),
                       familias="|".join(sorted(sub.familia))))
    t = pd.DataFrame(tf)
    t.to_csv(OUT / "anio_por_tramo.csv", index=False, encoding="utf-8-sig")
    print("\n=== POR TRAMO DE ANTIGUEDAD (28 evaluables) ===")
    print(t.drop(columns=["familias"]).to_string(index=False))
    for f in tf:
        print(f"\n  {f['tramo']}: {f['familias']}")

    m.sort_values(["anio", "familia"]).to_csv(OUT / "anio_por_familia.csv", index=False,
                                              encoding="utf-8-sig")
    print("\n=== LAS 30, POR AÑO ===")
    print(m.sort_values(["anio", "familia"])[
        ["anio", "familia", "n_plantillas", "P2bal_txt", "P2bal_m6"]].to_string(index=False))

    r_anio = next(f for f in filas if f["relacion"] == "año vs F1 de la cascada")
    r_pl = next(f for f in filas if f["relacion"] == "plantillas vs F1 de la cascada")
    print("\n" + "=" * 78)
    print("  LECTURA")
    print("=" * 78)
    sig = "SI" if r_anio["pearson_p"] < 0.05 else "NO"
    print(f"  El año {'SI' if sig=='SI' else 'NO'} correlaciona significativamente con el F1 "
          f"(r = {r_anio['pearson_r']:+.3f}, p = {r_anio['pearson_p']:.4f}).")
    print(f"  Descontando el numero de plantillas queda r = {rp:+.3f} (p = {pp:.4f}).")
    print(f"  Para comparar: plantillas vs F1 da r = {r_pl['pearson_r']:+.3f} "
          f"(p = {r_pl['pearson_p']:.4f}).")
    print("\n  RECORDAR AL CITAR: analisis POST-HOC, 28 familias evaluables, F1 por familia de")
    print("  P2bal a 50 semillas, año de Pruebas.xlsx (hoja «Informacion sobre familias»).")
    print("  n = 28 es chico: un solo caso extremo mueve la correlacion.")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
