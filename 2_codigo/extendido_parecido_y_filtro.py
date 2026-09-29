#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extendido_parecido_y_filtro.py -- dos preguntas sobre la Base B (106 familias).

PARTE 1 -- ¿QUE PASA CUANDO LA NOTA SE PARECE A LO QUE YA CONOCE?
Sobre el nucleo de 30 esto esta medido: con una hermana de su familia contenida >= 0,5 en el
entrenamiento el sistema acierta **0,9891**; sin ninguna nota parecida, **0,7434**. Sobre las 106
no se midio nunca, y es la pregunta natural: cuando el catalogo crece, ¿se degrada el
reconocimiento de variantes conocidas, o solo la generalizacion a notas nuevas?

PARTE 2 -- ¿SE PUEDE ARREGLAR EL FILTRO DE GENERICOS EN LA BASE B?
El acierto de la capa de reglas cae de 0,9928 (30 familias) a 0,9427 (106). La causa esta
medida: el mismo sitio entra al diccionario partido en varias claves -- `torproject.org` aparece
como SIETE -- y una variante poco frecuente puede quedar en una sola familia del pliegue, pasar
el filtro y hacer que la regla conteste con seguridad equivocada.

Ya se probo normalizar la clave, y **empeora la Base B** (-0,0080): al unificar las variantes,
la clave que sobrevive al filtro captura de golpe todas las notas que antes se repartian.

LO QUE SE PRUEBA ACA ES OTRA COSA, y separa las dos funciones que hoy cumple el mismo valor:
  - la CLAVE sigue siendo la URL completa, que conserva toda su especificidad;
  - pero el FILTRO decide mirando el DOMINIO: si el dominio de esa URL aparece en mas de una
    familia del entrenamiento, se descartan TODAS sus variantes.
Asi `torproject.org` se filtra completo -- las siete variantes a la vez -- sin colapsar ninguna
clave privada. Es lo que le faltaba a la normalizacion por dominio, que unia las claves ademas
de filtrarlas.

=============================================================================================
PREREGISTRO -- escrito y COMMITEADO ANTES de correr (2026-09-28).

PARTE 1
Q1. PUERTA. El sistema sobre la Base B reproduce macro-F1 0,6485 y exactitud 0,7529
    (tolerancia 0,01). Si no, ABORTA.
Q2. El tramo «con hermana contenida >= 0,5» sigue acertando >= 0,90 sobre 106 familias. Razon:
    reconocer una variante casi identica no deberia depender de cuantas clases haya.
Q3. LA PREDICCION QUE IMPORTA: la caida del nucleo a la Base B se concentra en el tramo SIN
    parecido. Es decir, la diferencia (acierto_30 - acierto_106) es MAYOR en el tramo sin
    hermana que en el tramo con hermana. Si fuera al reves, lo que se degrada al escalar es el
    reconocimiento de copias, que seria un resultado muy distinto.

PARTE 2
Q4. El filtro por dominio sube el acierto de la capa de reglas por encima de 0,9427 sobre la
    Base B.
Q5. El macro-F1 de la Base B sube con IC 95 % que excluye el cero.
Q6. Sobre el NUCLEO de 30 el efecto NO es negativo (Delta >= 0 dentro del IC). Si empeorara el
    nucleo, no se adopta aunque mejore la Base B: son dos bases separadas y el nucleo manda.
Q7. La cobertura de la regla BAJA en la Base B: se descartan mas claves que antes.
=============================================================================================

Uso:  python extendido_parecido_y_filtro.py [--n-semillas 20] [--salida CARPETA]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.metrics import accuracy_score, f1_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import obtener_modelos, vectorizador
from filtro_genericos_url import cargar_extendido, norm_fam
from grafo_marcadores import extraer_marcadores
from protocolo_logo import cargar_nombres
from protocolo_p2bal import split_p2bal
from revision_logo import matriz_contencion

RAIZ = _AQUI.parent
OUT_DEF = RAIZ / "4_resultados" / "resultados_extendido_parecido"
CANON_B_F1, CANON_B_ACC, TOL = 0.6485, 0.7529, 0.01
CANON_B_AC_REGLA = 0.9427
# Base A, para comparar los tramos (fuente: _log_similitud_p2bal_149.txt)
A_CON_HERMANA, A_SIN_HERMANA = 0.9891, 0.7434


def dominio(valor):
    v = re.sub(r"^https?://", "", valor.strip().lower())
    v = re.sub(r"^www\.", "", v)
    return v.split("/")[0].split("?")[0]


def dicc_dominio(tr, iocs, nombres, y):
    """Clave = valor completo. Filtro = por DOMINIO en el caso de las URL."""
    d = defaultdict(set)
    fam_de_dom = defaultdict(set)
    for i in tr:
        for tipo, valor in iocs[i]:
            d[(tipo, valor)].add(y[i])
            if tipo == "[URL]":
                fam_de_dom[dominio(valor)].add(y[i])
        if nombres[i]:
            d[("[NOMBRE]", nombres[i])].add(y[i])
    fuera = [k for k, v in d.items() if len(v) > 1]
    for k in fuera:
        del d[k]
    # y ademas: toda URL cuyo DOMINIO cruza familias
    for k in [k for k in d if k[0] == "[URL]" and len(fam_de_dom[dominio(k[1])]) > 1]:
        del d[k]
    return d


def dicc_actual(tr, iocs, nombres, y):
    d = defaultdict(set)
    for i in tr:
        for c in iocs[i]:
            d[c].add(y[i])
        if nombres[i]:
            d[("[NOMBRE]", nombres[i])].add(y[i])
    for k in [k for k, v in d.items() if len(v) > 1]:
        del d[k]
    return d


def regla(i, d, iocs, nombres):
    claves = set(iocs[i])
    if nombres[i]:
        claves.add(("[NOMBRE]", nombres[i]))
    fams = set()
    for c in claves:
        if c in d:
            fams |= d[c]
    return next(iter(fams)) if len(fams) == 1 else None


def ic(v):
    v = np.asarray(v, float)
    m, nn = float(np.mean(v)), len(v)
    s = float(np.std(v, ddof=1)) if nn > 1 else 0.0
    h = t_dist.ppf(0.975, nn - 1) * (s / np.sqrt(nn)) if nn > 1 else 0.0
    return m, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=20)
    ap.add_argument("--sin-puerta", action="store_true")
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  BASE B (106 familias): el parecido con lo conocido, y el filtro por dominio")
    print("=" * 78)
    textos, y, grupos, arch, es_can, canonicas = cargar_extendido()
    ta = np.array(textos, dtype=object)
    familias = np.unique(y)
    n = len(y)
    iocs = [set(extraer_marcadores(t)) for t in textos]
    nom = cargar_nombres()
    inv = {norm_fam(f): f for f in {k[0] for k in nom}}
    nombres = [nom.get((inv[f], Path(a).name)) if f in inv else None for f, a in zip(y, arch)]
    print(f"Base B: {n} notas, {len(familias)} familias | nucleo: {int(es_can.sum())} notas\n")

    print("Matriz de contencion (3-shingles) ...")
    C = matriz_contencion(textos)

    cont = np.full((args.n_semillas, n), -1.0)
    ok = {m: np.zeros((args.n_semillas, n), bool) for m in ("actual", "dominio")}
    apl = {m: np.zeros((args.n_semillas, n), bool) for m in ("actual", "dominio")}
    print("Evaluando ...")
    for s in range(args.n_semillas):
        rng = np.random.default_rng(20_000 + s)
        for tr, te in split_p2bal(y, grupos, familias, rng):
            vec = vectorizador("combinado")
            Xtr = vec.fit_transform(ta[tr])
            clf = obtener_modelos(s)["LinearSVC"]
            clf.fit(Xtr, y[tr])
            pt = clf.predict(vec.transform(ta[te]))
            dd = {"actual": dicc_actual(tr, iocs, nombres, y),
                  "dominio": dicc_dominio(tr, iocs, nombres, y)}
            for k, i in enumerate(te):
                propias = tr[y[tr] == y[i]]
                if len(propias):
                    cont[s, i] = C[i, propias].max()
                for m in ("actual", "dominio"):
                    r = regla(i, dd[m], iocs, nombres)
                    ok[m][s, i] = (pt[k] if r is None else r) == y[i]
                    apl[m][s, i] = r is not None
        if (s + 1) % 5 == 0:
            print(f"  {s+1}/{args.n_semillas}")

    # ---------- puerta Q1 ----------
    acc_a = float(ok["actual"].mean())
    f1_a = np.array([f1_score(y, np.where(ok["actual"][s], y, "__mal__"), average="macro",
                              labels=familias, zero_division=0) for s in range(args.n_semillas)])
    print("\n" + "-" * 78)
    print("  Q1 -- PUERTA (exactitud de la Base B)")
    print("-" * 78)
    print(f"  exactitud {acc_a:.4f} vs {CANON_B_ACC}")
    okp = abs(acc_a - CANON_B_ACC) <= TOL
    if not okp and not args.sin_puerta:
        sys.exit("ABORTADO (Q1).")
    print("  OK\n" if okp else "  FUERA DE TOLERANCIA\n")

    # ---------- PARTE 1: por tramo de parecido ----------
    def tramo(mask_fn):
        v = []
        for s in range(args.n_semillas):
            m = mask_fn(s)
            v.append(float(ok["actual"][s, m].mean()) if m.any() else np.nan)
        return float(np.nanmean(v)), float(np.mean([mask_fn(s).sum() for s in range(args.n_semillas)]))

    con_h, n_con = tramo(lambda s: cont[s] >= 0.5)
    sin_h, n_sin = tramo(lambda s: (cont[s] >= 0) & (cont[s] < 0.5))
    sin_f, n_sf = tramo(lambda s: cont[s] < 0)
    filas = [
        dict(tramo="con hermana contenida >= 0,5", notas=round(n_con, 1),
             frac=round(n_con / n, 4), acierto_B=round(con_h, 4), acierto_A=A_CON_HERMANA,
             caida=round(A_CON_HERMANA - con_h, 4)),
        dict(tramo="sin hermana parecida", notas=round(n_sin, 1), frac=round(n_sin / n, 4),
             acierto_B=round(sin_h, 4), acierto_A=A_SIN_HERMANA,
             caida=round(A_SIN_HERMANA - sin_h, 4)),
        dict(tramo="sin plantilla propia en entrenamiento", notas=round(n_sf, 1),
             frac=round(n_sf / n, 4), acierto_B=round(sin_f, 4), acierto_A=0.0,
             caida=round(0.0 - sin_f, 4))]
    df1 = pd.DataFrame(filas)
    df1.to_csv(OUT / "parecido_base_b.csv", index=False, encoding="utf-8-sig")
    print("=== PARTE 1: ACIERTO SEGUN EL PARECIDO CON LO CONOCIDO (Base B vs Base A) ===")
    print(df1.to_string(index=False))

    # ---------- PARTE 2: filtro por dominio ----------
    f1 = {}
    for m in ("actual", "dominio"):
        f1[m] = np.array([f1_score(y, np.where(ok[m][s], y, "__mal__"), average="macro",
                                   labels=familias, zero_division=0)
                          for s in range(args.n_semillas)])
    acr = {m: float(np.mean([ok[m][s][apl[m][s]].mean() for s in range(args.n_semillas)
                             if apl[m][s].any()])) for m in ("actual", "dominio")}
    cobr = {m: float(apl[m].mean()) for m in ("actual", "dominio")}
    dm, lo, hi = ic(f1["dominio"] - f1["actual"])
    df2 = pd.DataFrame([
        dict(variante="filtro actual", macro_f1=round(float(f1["actual"].mean()), 4),
             exactitud=round(float(ok["actual"].mean()), 4),
             cobertura_regla=round(cobr["actual"], 4), acierto_regla=round(acr["actual"], 4)),
        dict(variante="filtro por DOMINIO", macro_f1=round(float(f1["dominio"].mean()), 4),
             exactitud=round(float(ok["dominio"].mean()), 4),
             cobertura_regla=round(cobr["dominio"], 4), acierto_regla=round(acr["dominio"], 4))])
    df2.to_csv(OUT / "filtro_dominio_base_b.csv", index=False, encoding="utf-8-sig")
    print("\n=== PARTE 2: FILTRAR POR DOMINIO, MANTENIENDO LA URL COMO CLAVE ===")
    print(df2.to_string(index=False))
    print(f"\n  Delta macro-F1: {dm:+.4f} [{lo:+.4f}; {hi:+.4f}]  "
          f"{int(((f1['dominio']-f1['actual']) > 0).sum())}/{args.n_semillas} semillas")

    print("\n" + "=" * 78)
    print("  VEREDICTO DEL PREREGISTRO")
    print("=" * 78)
    chk = [
        ("Q1 puerta de entrada", okp, f"{acc_a:.4f}"),
        ("Q2 con hermana, sigue acertando >= 0,90", con_h >= 0.90, f"{con_h:.4f}"),
        ("Q3 la caida se concentra en el tramo SIN parecido",
         (A_SIN_HERMANA - sin_h) > (A_CON_HERMANA - con_h),
         f"caida sin hermana {A_SIN_HERMANA - sin_h:+.4f} | con hermana {A_CON_HERMANA - con_h:+.4f}"),
        ("Q4 el filtro por dominio sube el acierto de la regla",
         acr["dominio"] > CANON_B_AC_REGLA,
         f"{acr['actual']:.4f} -> {acr['dominio']:.4f}"),
        ("Q5 el macro-F1 sube con IC que excluye el cero", lo > 0,
         f"{dm:+.4f} [{lo:+.4f}; {hi:+.4f}]"),
        ("Q7 la cobertura de la regla baja", cobr["dominio"] < cobr["actual"],
         f"{cobr['actual']:.4f} -> {cobr['dominio']:.4f}"),
    ]
    for nombre, cumple, det in chk:
        print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<48} {det}")
    print(f"\n  (Q6, el efecto sobre el nucleo de 30, se mide aparte con --solo-nucleo del")
    print("   script filtro_genericos_url.py si esta variante resulta adoptable.)")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
