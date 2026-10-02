#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mundo_abierto_familia_fuera.py -- ¿que hace el sistema ante una familia que NUNCA VIO?

QUE CONTESTA. Todas las cifras del frente de notas se miden en MUNDO CERRADO: la familia de la
nota evaluada siempre esta en el catalogo, aunque su plantilla no. En un incidente real eso no
se cumple: aparecen campanas nuevas. Hoy el trabajo tiene UN solo caso de familia fuera del
catalogo -- una campana de pocos dias, contra un catalogo ampliado a 320 familias, que ninguna
clase reclamo (`_log_reconocer_todo.txt`). Es un caso real y vale, pero es anecdotico.

Este experimento lo convierte en medicion sistematica: **saca una familia entera del
entrenamiento y mide si el sistema se abstiene ante sus notas o se las asigna a otra con
confianza**. Se repite con las 30.

EL DISENO, y por que es simetrico. Para cada familia objetivo f:
  - se RETIENE una plantilla de cada una de las otras 29 familias;
  - se entrena con todo lo demas (o sea, sin NINGUNA nota de f);
  - se evalua el MISMO modelo sobre dos conjuntos:
      DESCONOCIDO = las notas de f, cuya familia el modelo no tiene;
      CONOCIDO    = las plantillas retenidas de las otras 29, cuya familia si tiene.
Los dos conjuntos pasan por el mismo modelo y el mismo umbral, asi que la comparacion es limpia.
Sin esa simetria, el conjunto conocido vendria de un modelo entrenado con mas material y la
diferencia de abstencion seria un artefacto del tamano de entrenamiento.

QUE SE MIDE. Para cada umbral de margen, en los dos conjuntos:
  - **tasa de abstencion**: que fraccion NO se contesta. En DESCONOCIDO es un ACIERTO del
    sistema (no hay respuesta correcta posible); en CONOCIDO es un COSTO.
  - sobre lo que si contesta en CONOCIDO, el acierto.
Es la curva de un detector de novedad: rechazo correcto frente a rechazo indebido.

=============================================================================================
PREREGISTRO -- escrito y COMMITEADO ANTES de correr (2026-09-28).

O1. PUERTA DE ENTRADA. Con umbral 0 (contesta siempre) el acierto sobre DESCONOCIDO es
    EXACTAMENTE 0,0000: la familia no esta en el modelo, asi que no puede acertarla. Si diera
    mas que cero hay una fuga y ABORTA.
O2. A todo umbral > 0, la tasa de abstencion en DESCONOCIDO es MAYOR que en CONOCIDO. Es la
    premisa del mecanismo: si no se cumpliera, el margen no distingue lo nuevo de lo conocido y
    la abstencion no sirve como detector de novedad.
O3. En el punto de operacion adoptado (umbral 0,50) el sistema se abstiene en >= 0,50 de las
    notas de familia desconocida.
O4. La capa de REGLAS casi nunca aplica ante familia desconocida: cobertura < 0,10. Razon: sus
    correos, billeteras y onions no estan en el diccionario, que se arma solo con entrenamiento.
    Si aplicara seguido, estaria coincidiendo por valores genericos y seria un problema.
O5. Las familias desconocidas que MENOS se rechazan son las que tienen un pariente de linaje en
    el entrenamiento (CLOP-RYUK, DHARMA-PHOBOS, BLACKBASTA-CONTI): el texto las asigna con
    confianza al pariente. Es la consecuencia directa del hallazgo del 2026-09-26 y la
    prediccion mas informativa de este preregistro.
O6. INTERPRETACION FIJADA DE ANTEMANO: la cifra que se reporte NO es "el sistema detecta
    novedad con X de acierto". Es "ante una familia fuera del catalogo se abstiene el X % de
    las veces, al costo de abstenerse tambien en el Y % de las conocidas". Las dos juntas o
    ninguna.

LIMITACION QUE SE DECLARA: 30 familias dan 30 puntos de medicion, y cada uno con pocas notas
(de 2 a 19). La curva es informativa, no precisa.
=============================================================================================

Uso:  python mundo_abierto_familia_fuera.py [--n-semillas 20] [--salida CARPETA]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import obtener_modelos, vectorizador
from protocolo_logo import dicc_privados, regla
from revision_logo import cargar_todo

RAIZ = _AQUI.parent
OUT_DEF = RAIZ / "4_resultados" / "resultados_mundo_abierto"
UMBRALES = [0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00]
PARIENTE = {"CLOP": "RYUK", "RYUK": "CLOP", "DHARMA": "PHOBOS", "PHOBOS": "DHARMA",
            "BLACKBASTA": "CONTI", "CONTI": "BLACKBASTA"}


def ic(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    m, nn = float(np.mean(v)), len(v)
    s = float(np.std(v, ddof=1)) if nn > 1 else 0.0
    h = t_dist.ppf(0.975, nn - 1) * (s / np.sqrt(nn)) if nn > 1 else 0.0
    return m, max(0.0, m - h), min(1.0, m + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=20)
    ap.add_argument("--sin-puerta", action="store_true")
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  MUNDO ABIERTO: ¿que hace el sistema ante una familia que nunca vio?")
    print("=" * 78)
    textos, textos_arr, y, archivos, grupos, iocs, nombres_nota = cargar_todo()
    familias = np.unique(y)
    n = len(y)
    pl_de_fam = {f: np.unique(grupos[y == f]) for f in familias}
    print(f"Notas: {n} | Familias: {len(familias)} | Semillas: {args.n_semillas}")
    print("Cada familia sale entera del entrenamiento, por turno.\n")

    # acumuladores: por (familia_fuera, semilla) guardamos margenes y si la regla aplico
    reg = []          # filas: familia_fuera, semilla, conjunto, margen, aplica_regla, acierto_txt
    print("Evaluando ...")
    for s in range(args.n_semillas):
        rng = np.random.default_rng(50_000 + s)
        for f_fuera in familias:
            # retener una plantilla de cada OTRA familia -> conjunto CONOCIDO
            retenidas = set()
            for g in familias:
                if g == f_fuera:
                    continue
                pl = pl_de_fam[g]
                retenidas.add(pl[rng.integers(len(pl))])
            es_fuera = (y == f_fuera)
            es_ret = np.array([g in retenidas for g in grupos])
            tr = np.where(~es_fuera & ~es_ret)[0]
            te_desc = np.where(es_fuera)[0]
            te_con = np.where(~es_fuera & es_ret)[0]
            if len(tr) == 0 or len(np.unique(y[tr])) < 2:
                continue

            vec = vectorizador("combinado")
            Xtr = vec.fit_transform(textos_arr[tr])
            clf = obtener_modelos(s)["LinearSVC"]
            clf.fit(Xtr, y[tr])
            clases = clf.classes_
            d = dicc_privados(tr, iocs, nombres_nota, y)

            for etq, te in (("DESCONOCIDO", te_desc), ("CONOCIDO", te_con)):
                if len(te) == 0:
                    continue
                dec = clf.decision_function(vec.transform(textos_arr[te]))
                orden = np.argsort(-dec, axis=1)
                top1 = clases[orden[:, 0]]
                marg = (dec[np.arange(len(te)), orden[:, 0]]
                        - dec[np.arange(len(te)), orden[:, 1]])
                for k, i in enumerate(te):
                    r = regla(i, d, iocs, nombres_nota)
                    reg.append((f_fuera, s, etq, float(marg[k]), r is not None,
                                bool(top1[k] == y[i]),
                                (r if r is not None else top1[k]) == y[i]))
        if (s + 1) % 5 == 0:
            print(f"  {s+1}/{args.n_semillas} semillas")

    df = pd.DataFrame(reg, columns=["familia_fuera", "semilla", "conjunto", "margen",
                                    "aplica_regla", "acierto_texto", "acierto_cascada"])
    df.to_csv(OUT / "mundo_abierto_por_nota.csv", index=False, encoding="utf-8-sig")
    desc = df[df.conjunto == "DESCONOCIDO"]
    con = df[df.conjunto == "CONOCIDO"]
    print(f"\nDecisiones: {len(desc)} sobre familia DESCONOCIDA | {len(con)} sobre CONOCIDA")

    # ---------------- puerta O1 ----------------
    ac0 = float(desc.acierto_cascada.mean())
    print("\n" + "-" * 78)
    print("  O1 -- PUERTA: el acierto sobre familia DESCONOCIDA debe ser 0,0000 exacto")
    print("-" * 78)
    print(f"  acierto cascada sobre desconocidas: {ac0:.6f}")
    ok = ac0 == 0.0
    if not ok and not args.sin_puerta:
        sys.exit("ABORTADO (O1): acerto una familia que no estaba en el modelo. Hay fuga.")
    print("  OK\n" if ok else "  *** FUGA ***\n")

    # ---------------- curva ----------------
    filas = []
    for u in UMBRALES:
        # se abstiene si la regla NO aplica y el margen no llega al umbral
        ab_d = ((~desc.aplica_regla) & (desc.margen < u)).mean()
        ab_c = ((~con.aplica_regla) & (con.margen < u)).mean()
        m_con = (con.aplica_regla) | (con.margen >= u)
        ac_c = float(con.loc[m_con, "acierto_cascada"].mean()) if m_con.any() else np.nan
        filas.append(dict(umbral=u,
                          abstencion_DESCONOCIDO=round(float(ab_d), 4),
                          abstencion_CONOCIDO=round(float(ab_c), 4),
                          separacion=round(float(ab_d - ab_c), 4),
                          acierto_en_lo_que_contesta_CONOCIDO=round(ac_c, 4)))
    dfc = pd.DataFrame(filas)
    dfc.to_csv(OUT / "mundo_abierto_curva.csv", index=False, encoding="utf-8-sig")
    print("=== CURVA DE RECHAZO (la abstencion como detector de novedad) ===")
    print(dfc.to_string(index=False))

    # ---------------- cobertura de la regla (O4) ----------------
    cob_d = float(desc.aplica_regla.mean())
    cob_c = float(con.aplica_regla.mean())
    print(f"\n  Cobertura de la capa de REGLAS: DESCONOCIDO {cob_d:.4f} | CONOCIDO {cob_c:.4f}")

    # ---------------- por familia, al umbral adoptado (O5) ----------------
    U = 0.50
    ffilas = []
    for f in familias:
        sub = desc[desc.familia_fuera == f]
        if not len(sub):
            continue
        ab = float(((~sub.aplica_regla) & (sub.margen < U)).mean())
        ffilas.append(dict(familia_fuera=f, n_decisiones=len(sub),
                           se_abstiene=round(ab, 4),
                           tiene_pariente_en_train="SI" if f in PARIENTE else "no",
                           pariente=PARIENTE.get(f, ""),
                           regla_aplica=round(float(sub.aplica_regla.mean()), 4)))
    dff = pd.DataFrame(ffilas).sort_values("se_abstiene")
    dff.to_csv(OUT / "mundo_abierto_por_familia.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== POR FAMILIA, al umbral {U} (las 12 que MENOS se rechazan) ===")
    print(dff.head(12).to_string(index=False))

    con_par = dff[dff.tiene_pariente_en_train == "SI"].se_abstiene.mean()
    sin_par = dff[dff.tiene_pariente_en_train == "no"].se_abstiene.mean()

    # ---------------- veredicto ----------------
    r50 = next(r for r in filas if r["umbral"] == U)
    mono = all(r["separacion"] > 0 for r in filas if r["umbral"] > 0)
    print("\n" + "=" * 78)
    print("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    print("=" * 78)
    chk = [
        ("O1 acierto sobre desconocida = 0,0000 exacto", ok, f"{ac0:.6f}"),
        ("O2 se abstiene mas ante lo desconocido, a todo umbral", mono,
         " · ".join(f"u{r['umbral']:.2f}:{r['separacion']:+.3f}" for r in filas if r["umbral"] > 0)),
        ("O3 al umbral 0,50 rechaza >= 0,50 de las desconocidas",
         r50["abstencion_DESCONOCIDO"] >= 0.50, f"{r50['abstencion_DESCONOCIDO']:.4f}"),
        ("O4 la regla casi no aplica ante desconocida (< 0,10)",
         cob_d < 0.10, f"{cob_d:.4f} vs {cob_c:.4f} en conocidas"),
        ("O5 las que tienen pariente se rechazan menos",
         con_par < sin_par, f"con pariente {con_par:.4f} | sin pariente {sin_par:.4f}"),
    ]
    for nombre, cumple, det in chk:
        print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<48} {det}")

    print(f"\n  FRASE CITABLE (O6: las dos cifras juntas o ninguna):")
    print(f"    Ante una familia FUERA del catalogo, el sistema se abstiene en el "
          f"{r50['abstencion_DESCONOCIDO']*100:.1f} % de los casos")
    print(f"    (umbral 0,50), al costo de abstenerse tambien en el "
          f"{r50['abstencion_CONOCIDO']*100:.1f} % de las notas de familia conocida,")
    print(f"    donde acierta {r50['acierto_en_lo_que_contesta_CONOCIDO']:.4f} sobre lo que contesta.")
    print("    Base: 30 familias, una fuera por turno, corte por plantilla en las 29 restantes.")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
