#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
confusion_y_linaje.py -- ¿contra QUIEN se equivoca el sistema? Y el cierre de M.4.

DOS COSAS DE UNA:

(a) LA MATRIZ DE CONFUSION del frente de notas bajo el protocolo de cabecera, que el checklist
    del tutor pide y que todavia no existe sobre P2bal. Un macro-F1 dice cuanto falla; la matriz
    dice CONTRA QUE falla, que es lo que un jurado pregunta despues.

(b) EL CIERRE DE M.4. M.4 (PLAN_MEJORAS.md) proponia un clasificador de DOS ETAPAS: primero al
    grupo de linaje, despues desambiguar adentro por IOCs, «que son privados de cada familia aun
    cuando el texto sea compartido». Nunca se corrio. Mientras tanto la cascada adoptada hace la
    desambiguacion por IOCs **directamente**, sin la etapa de linaje. La pregunta que cierra el
    item es: **¿queda algo que la etapa de linaje pudiera arreglar?** Si los errores dentro del
    linaje ya son pocos con la cascada, M.4 no tiene material y se cierra sin correrlo.

LOS PARES DE LINAJE, declarados a mano y verificados (2026-09-26): el corpus canonico tiene
**exactamente 2 grupos de casi-duplicados que cruzan familias**, y son los dos que M.4 nombraba:
  - BLACKBASTA + CONTI      (2 notas en un grupo)
  - DHARMA + PHOBOS         (12 notas en un grupo)
Verificado corriendo agrupar_neardups sobre las 149. Ver ESTADO_TESIS.md, grafo B.3: el vinculo
es por contenido casi duplicado, no por marcadores.

=============================================================================================
PREREGISTRO -- escrito y COMMITEADO ANTES de correr (2026-09-26).

L1. PUERTA DE ENTRADA. Exactitud global 0,8123 (cascada) y 0,7191 (texto), tolerancia 0,01
    (fuente _log_p2bal_149.txt). Si no reproduce, ABORTA.
L2. La FRACCION de errores que cae dentro de los pares de linaje BAJA al pasar de texto solo a
    cascada. Razon: los IOCs son privados de cada familia aun cuando el texto sea compartido
    (B.3), asi que la capa de reglas separa justo donde el texto no puede. Es la prediccion que
    M.4 hacia y que nunca se puso a prueba.
L3. En terminos ABSOLUTOS, los errores de linaje con la cascada son menos de la MITAD que con
    el texto solo.
L4. La confusion mas frecuente del texto solo involucra a uno de los dos pares de linaje.
    Razon: son las unicas dos familias del corpus que comparten plantilla de texto.
L5. EXPLORATORIO, declarado como tal y sin prediccion: se listan las 15 confusiones mas
    frecuentes para ver si aparece algun par NO documentado. Si aparece uno fuerte, es material
    nuevo para el capitulo y hay que mirarlo a mano; no se cuenta como confirmacion de nada.
=============================================================================================

Uso:  python confusion_y_linaje.py [--n-semillas 50] [--salida CARPETA]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import obtener_modelos, vectorizador
from protocolo_logo import dicc_privados, regla
from protocolo_p2bal import split_p2bal
from revision_logo import cargar_todo, plantillas_por_familia

RAIZ = _AQUI.parent
OUT_DEF = RAIZ / "4_resultados" / "resultados_confusion_linaje"
CANON_ACC_M6, CANON_ACC_TXT, TOL = 0.8123, 0.7191, 0.01
PARES_LINAJE = [frozenset({"BLACKBASTA", "CONTI"}), frozenset({"DHARMA", "PHOBOS"})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=50)
    ap.add_argument("--sin-puerta", action="store_true")
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  MATRIZ DE CONFUSION Y CIERRE DE M.4 (linaje) -- protocolo P2bal")
    print("=" * 78)
    textos, textos_arr, y, archivos, grupos, iocs, nombres_nota = cargar_todo()
    familias = np.unique(y)
    n = len(y)
    ppf = plantillas_por_familia(y, grupos)
    print(f"Notas: {n} | Familias: {len(familias)} | Semillas: {args.n_semillas}")
    print(f"Pares de linaje declarados: {[sorted(p) for p in PARES_LINAJE]}\n")

    p_txt = np.empty((args.n_semillas, n), dtype=object)
    p_m6 = np.empty((args.n_semillas, n), dtype=object)

    print("Evaluando ...")
    for s in range(args.n_semillas):
        rng = np.random.default_rng(20_000 + s)
        for tr, te in split_p2bal(y, grupos, familias, rng):
            vec = vectorizador("combinado")
            Xtr = vec.fit_transform(textos_arr[tr])
            Xte = vec.transform(textos_arr[te])
            clf = obtener_modelos(s)["LinearSVC"]
            clf.fit(Xtr, y[tr])
            pt = clf.predict(Xte)
            p_txt[s, te] = pt
            d = dicc_privados(tr, iocs, nombres_nota, y)
            for k, i in enumerate(te):
                r = regla(i, d, iocs, nombres_nota)
                p_m6[s, i] = pt[k] if r is None else r
        if (s + 1) % 10 == 0:
            print(f"  {s+1}/{args.n_semillas} semillas")

    # ---------------- puerta L1 ----------------
    acc_m6 = float(np.mean([accuracy_score(y, p_m6[s]) for s in range(args.n_semillas)]))
    acc_txt = float(np.mean([accuracy_score(y, p_txt[s]) for s in range(args.n_semillas)]))
    print("\n" + "-" * 78)
    print("  L1 -- PUERTA DE ENTRADA")
    print("-" * 78)
    print(f"  cascada {acc_m6:.4f} vs {CANON_ACC_M6} | texto {acc_txt:.4f} vs {CANON_ACC_TXT}")
    ok = abs(acc_m6 - CANON_ACC_M6) <= TOL and abs(acc_txt - CANON_ACC_TXT) <= TOL
    if not ok and not args.sin_puerta:
        sys.exit("ABORTADO (L1): no reproduce la cabecera. No se reporta nada.")
    print("  OK\n" if ok else "  FUERA DE TOLERANCIA (--sin-puerta)\n")

    # ---------------- confusiones ----------------
    res = {}
    for etq, P in (("texto solo", p_txt), ("cascada", p_m6)):
        conf = Counter()
        for s in range(args.n_semillas):
            for i in range(n):
                if P[s, i] != y[i]:
                    conf[(y[i], P[s, i])] += 1
        total = sum(conf.values())
        linaje = sum(c for (v, p), c in conf.items()
                     if frozenset({v, p}) in PARES_LINAJE)
        res[etq] = dict(conf=conf, total=total, linaje=linaje,
                        frac=linaje / total if total else 0.0)
        M = np.zeros((len(familias), len(familias)), int)
        for s in range(args.n_semillas):
            M += confusion_matrix(y, P[s], labels=familias)
        pd.DataFrame(M, index=familias, columns=familias).to_csv(
            OUT / f"matriz_confusion_{'texto' if etq == 'texto solo' else 'cascada'}.csv",
            encoding="utf-8-sig")

    print("=== ERRORES Y CUANTOS SON CONFUSION DENTRO DEL LINAJE ===")
    print(f"{'sistema':<12}{'errores':>10}{'de linaje':>12}{'fraccion':>11}"
          f"{'por semilla':>13}")
    for etq in ("texto solo", "cascada"):
        r = res[etq]
        print(f"{etq:<12}{r['total']:>10}{r['linaje']:>12}{r['frac']:>11.4f}"
              f"{r['total']/args.n_semillas:>13.1f}")

    # ---------------- top de confusiones ----------------
    filas = []
    for etq in ("texto solo", "cascada"):
        for (v, p), c in res[etq]["conf"].most_common(15):
            filas.append(dict(sistema=etq, verdadera=v, predicha=p,
                              veces=c, por_semilla=round(c / args.n_semillas, 2),
                              es_linaje="SI" if frozenset({v, p}) in PARES_LINAJE else "no",
                              frac_de_los_errores=round(c / res[etq]["total"], 4)))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "top_confusiones.csv", index=False, encoding="utf-8-sig")
    for etq in ("texto solo", "cascada"):
        print(f"\n=== 15 CONFUSIONES MAS FRECUENTES -- {etq} (L5, exploratorio) ===")
        print(df[df.sistema == etq].drop(columns=["sistema"]).to_string(index=False))

    # ---------------- confusor principal por familia ----------------
    ffilas = []
    for f in familias:
        fila = dict(familia=f, n_notas=int((y == f).sum()), n_plantillas=ppf[f])
        for etq, k in (("texto solo", "txt"), ("cascada", "cas")):
            cs = [(p, c) for (v, p), c in res[etq]["conf"].items() if v == f]
            if cs:
                p, c = max(cs, key=lambda t: t[1])
                fila[f"confusor_{k}"] = p
                fila[f"veces_{k}"] = c
                fila[f"es_linaje_{k}"] = "SI" if frozenset({f, p}) in PARES_LINAJE else "no"
            else:
                fila[f"confusor_{k}"] = "-"; fila[f"veces_{k}"] = 0; fila[f"es_linaje_{k}"] = "-"
        ffilas.append(fila)
    dff = pd.DataFrame(ffilas).sort_values("veces_cas", ascending=False)
    dff.to_csv(OUT / "confusor_principal_por_familia.csv", index=False, encoding="utf-8-sig")
    print("\n=== CONTRA QUIEN SE EQUIVOCA CADA FAMILIA (10 peores con la cascada) ===")
    print(dff.head(10).to_string(index=False))

    # ---------------- veredicto ----------------
    ft, fc = res["texto solo"], res["cascada"]
    top_txt = ft["conf"].most_common(1)[0]
    print("\n" + "=" * 78)
    print("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    print("=" * 78)
    chk = [
        ("L1 puerta de entrada", ok, f"{acc_m6:.4f} | {acc_txt:.4f}"),
        ("L2 la fraccion de errores de linaje BAJA con la cascada",
         fc["frac"] < ft["frac"], f"texto {ft['frac']:.4f} -> cascada {fc['frac']:.4f}"),
        ("L3 en absoluto, menos de la mitad",
         fc["linaje"] < ft["linaje"] / 2, f"texto {ft['linaje']} -> cascada {fc['linaje']}"),
        ("L4 la confusion top del texto es de linaje",
         frozenset({top_txt[0][0], top_txt[0][1]}) in PARES_LINAJE,
         f"{top_txt[0][0]} -> {top_txt[0][1]} ({top_txt[1]} veces)"),
    ]
    for nombre, cumple, det in chk:
        print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<48} {det}")

    print("\n  CIERRE DE M.4:")
    print(f"    Con la cascada quedan {fc['linaje']} errores de linaje sobre {fc['total']} "
          f"({fc['frac']*100:.1f} % de los errores),")
    print(f"    o sea {fc['linaje']/args.n_semillas:.1f} notas por semilla de {n}. La etapa de")
    print("    linaje de M.4 tendria que arreglar ESO y nada mas.")
    if fc["frac"] < 0.10:
        print("    => M.4 SE CIERRA SIN CORRERLO: la cascada ya desambigua el linaje por IOCs,")
        print("       que es justo lo que M.4 proponia hacer en dos etapas.")
    else:
        print("    => M.4 TODAVIA TIENE MATERIAL: vale la pena correr las dos etapas.")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
