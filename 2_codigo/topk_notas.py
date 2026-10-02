#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topk_notas.py -- el sistema como RANKING: ¿esta la familia correcta entre las k primeras?

DE DONDE SALE. Propuesta de Romina en el chat con el tutor (2026-09-09): «o dejamos siempre un
tipo ranking de porcentajes». Es la forma en que se usa de verdad un identificador de familia:
un analista no necesita que la herramienta acierte de una, necesita una lista corta donde mirar.
Top-k es la metrica estandar para eso y se reporta junto a la exactitud, nunca en su lugar.

COMO SE CONSTRUYE EL RANKING (hay que declararlo, porque la cascada no es un clasificador plano):
  - TEXTO SOLO: orden descendente de `decision_function` del LinearSVC (one-vs-rest).
  - CASCADA: si la capa de reglas aplica y apunta a una sola familia, esa familia va PRIMERA y
    detras va el orden del texto sin repetirla. Si no aplica, el ranking es el del texto.
    Es la traduccion directa de la cascada a una lista: la regla manda, el texto ordena el resto.

⚠️ LO QUE NO ES. Top-3 alto NO es «el sistema acierta el 90 %»: es «la respuesta esta entre tres
candidatas». Se cita siempre con la k pegada y junto al top-1, que es la exactitud de siempre.

=============================================================================================
PREREGISTRO -- escrito y COMMITEADO ANTES de correr (2026-09-26).

T1. PUERTA DE ENTRADA. El top-1 tiene que ser la exactitud de la corrida de cabecera de P2bal:
    cascada 0,8123 y texto 0,7191 (tolerancia 0,01, fuente _log_p2bal_149.txt). Si no, ABORTA.
T2. Top-3 de la cascada >= 0,90. Razon: el texto solo ya acierta 0,7191 en top-1 y los errores
    del SVC suelen tener la clase correcta cerca del tope.
T3. La ganancia de top-1 a top-3 es MAYOR en el texto solo que en la cascada. Razon: donde la
    regla contesta, o acierta (0,9928) o se equivoca fuerte, y mirar mas abajo no la arregla;
    el texto, en cambio, falla por poco. Si sale al reves, el ranking de la cascada esta mal
    construido y hay que revisar la definicion de arriba.
T4. Las 4 notas de familias sin material de entrenamiento (BADRABBIT, CRYPTOLOCKER) siguen en
    0,0000 para TODA k: su clase no existe en el modelo, asi que no puede estar en ninguna
    posicion del ranking. Control de sanidad; si aparece, hay fuga.
=============================================================================================

Uso:  python topk_notas.py [--n-semillas 50] [--salida CARPETA]
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
from protocolo_logo import cargar_nombres, dicc_privados, regla
from protocolo_p2bal import split_p2bal
from revision_logo import cargar_todo, plantillas_por_familia

RAIZ = _AQUI.parent
OUT_DEF = RAIZ / "4_resultados" / "resultados_topk_p2bal"
CANON_TOP1_M6, CANON_TOP1_TXT, TOL = 0.8123, 0.7191, 0.01
KS = [1, 2, 3, 5, 10]


def ic_t(v):
    v = np.asarray(v, float)
    m, n = float(np.mean(v)), len(v)
    s = float(np.std(v, ddof=1)) if n > 1 else 0.0
    h = t_dist.ppf(0.975, n - 1) * (s / np.sqrt(n)) if n > 1 else 0.0
    return m, max(0.0, m - h), min(1.0, m + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=50)
    ap.add_argument("--sin-puerta", action="store_true")
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  TOP-K: ¿esta la familia correcta entre las k primeras? -- protocolo P2bal")
    print("=" * 78)
    textos, textos_arr, y, archivos, grupos, iocs, nombres_nota = cargar_todo()
    familias = np.unique(y)
    n, F = len(y), len(familias)
    ppf = plantillas_por_familia(y, grupos)
    print(f"Notas: {n} | Familias: {F} | Plantillas: {len(set(grupos))} | Semillas: {args.n_semillas}\n")

    # posicion (1 = primera) de la familia correcta en cada ranking; F+1 si no esta
    pos_txt = np.full((args.n_semillas, n), F + 1, dtype=int)
    pos_m6 = np.full((args.n_semillas, n), F + 1, dtype=int)
    aplico = np.zeros((args.n_semillas, n), bool)

    print("Evaluando ...")
    for s in range(args.n_semillas):
        rng = np.random.default_rng(20_000 + s)
        for tr, te in split_p2bal(y, grupos, familias, rng):
            vec = vectorizador("combinado")
            Xtr = vec.fit_transform(textos_arr[tr])
            Xte = vec.transform(textos_arr[te])
            clf = obtener_modelos(s)["LinearSVC"]
            clf.fit(Xtr, y[tr])
            dec = clf.decision_function(Xte)
            clases = list(clf.classes_)
            orden = np.argsort(-dec, axis=1)
            d = dicc_privados(tr, iocs, nombres_nota, y)
            for k, i in enumerate(te):
                rank_txt = [clases[j] for j in orden[k]]
                if y[i] in rank_txt:
                    pos_txt[s, i] = rank_txt.index(y[i]) + 1
                r = regla(i, d, iocs, nombres_nota)
                if r is None:
                    rank_m6 = rank_txt
                else:
                    aplico[s, i] = True
                    rank_m6 = [r] + [c for c in rank_txt if c != r]
                if y[i] in rank_m6:
                    pos_m6[s, i] = rank_m6.index(y[i]) + 1

        if (s + 1) % 10 == 0:
            print(f"  {s+1}/{args.n_semillas} semillas")

    # ---------------- puerta T1 ----------------
    top1_m6 = float((pos_m6 <= 1).mean())
    top1_txt = float((pos_txt <= 1).mean())
    print("\n" + "-" * 78)
    print("  T1 -- PUERTA DE ENTRADA: el top-1 debe ser la exactitud de P2bal")
    print("-" * 78)
    print(f"  cascada: {top1_m6:.4f} vs {CANON_TOP1_M6}  (dif {abs(top1_m6-CANON_TOP1_M6):.4f})")
    print(f"  texto  : {top1_txt:.4f} vs {CANON_TOP1_TXT}  (dif {abs(top1_txt-CANON_TOP1_TXT):.4f})")
    ok = abs(top1_m6 - CANON_TOP1_M6) <= TOL and abs(top1_txt - CANON_TOP1_TXT) <= TOL
    if not ok and not args.sin_puerta:
        sys.exit("ABORTADO (T1): el top-1 no reproduce la cabecera. No se reporta nada.")
    print("  OK\n" if ok else "  FUERA DE TOLERANCIA (se sigue por --sin-puerta)\n")

    # ---------------- tabla top-k ----------------
    filas = []
    for etq, pos in (("texto solo", pos_txt), ("cascada", pos_m6)):
        for k in KS:
            porsem = [(pos[s] <= k).mean() for s in range(args.n_semillas)]
            m, lo, hi = ic_t(porsem)
            filas.append(dict(sistema=etq, k=k, top_k=round(m, 4),
                              ic95=f"[{lo:.4f}; {hi:.4f}]",
                              ganancia_sobre_top1=round(m - (top1_txt if etq == "texto solo"
                                                             else top1_m6), 4)))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "topk_p2bal.csv", index=False, encoding="utf-8-sig")
    print("=== TOP-K (media de 50 semillas, P2bal, 149 notas / 30 familias) ===")
    print(df.to_string(index=False))

    # posicion mediana del acierto cuando NO esta primera
    for etq, pos in (("texto solo", pos_txt), ("cascada", pos_m6)):
        fallo = pos[(pos > 1) & (pos <= F)]
        print(f"\n  {etq}: cuando no acierta de una, la familia correcta queda en la posicion "
              f"mediana {np.median(fallo):.0f} (media {fallo.mean():.1f}) de {F}")

    # ---------------- por familia ----------------
    ffilas = []
    for f in familias:
        idx = np.where(y == f)[0]
        fila = dict(familia=f, n_notas=len(idx), n_plantillas=ppf[f])
        for k in (1, 3):
            fila[f"texto_top{k}"] = round(float((pos_txt[:, idx] <= k).mean()), 4)
            fila[f"cascada_top{k}"] = round(float((pos_m6[:, idx] <= k).mean()), 4)
        fila["gana_de_top1_a_top3_cascada"] = round(fila["cascada_top3"] - fila["cascada_top1"], 4)
        ffilas.append(fila)
    dff = pd.DataFrame(ffilas).sort_values("gana_de_top1_a_top3_cascada", ascending=False)
    dff.to_csv(OUT / "topk_por_familia_p2bal.csv", index=False, encoding="utf-8-sig")
    print("\n=== FAMILIAS QUE MAS GANAN AL MIRAR 3 CANDIDATAS EN VEZ DE 1 (cascada) ===")
    print(dff.head(10).to_string(index=False))

    # ---------------- veredicto ----------------
    t3_m6 = float((pos_m6 <= 3).mean())
    t3_txt = float((pos_txt <= 3).mean())
    sin_mat = [i for i in range(n) if ppf[y[i]] < 2]
    fuga = float((pos_m6[:, sin_mat] <= F).mean()) if sin_mat else 0.0
    print("\n" + "=" * 78)
    print("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    print("=" * 78)
    chk = [
        ("T1 puerta (top-1 = exactitud de cabecera)", ok, f"{top1_m6:.4f} | {top1_txt:.4f}"),
        ("T2 top-3 de la cascada >= 0,90", t3_m6 >= 0.90, f"{t3_m6:.4f}"),
        ("T3 el texto gana mas de top-1 a top-3 que la cascada",
         (t3_txt - top1_txt) > (t3_m6 - top1_m6),
         f"texto +{t3_txt-top1_txt:.4f} | cascada +{t3_m6-top1_m6:.4f}"),
        ("T4 sanidad: familias de 1 plantilla nunca en el ranking",
         fuga == 0.0, f"aparecen en {fuga:.4f} de los casos ({len(sin_mat)} notas)"),
    ]
    for nombre, cumple, det in chk:
        print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<48} {det}")
    print(f"\n  FRASE CITABLE (P2bal, 149 notas / 30 familias, 50 semillas):")
    print(f"    La familia correcta es la primera propuesta en el {top1_m6*100:.1f} % de las notas")
    print(f"    y esta entre las TRES primeras en el {t3_m6*100:.1f} %.")
    print("    RECORDAR: top-k no es exactitud; se cita con la k pegada y junto al top-1.")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
