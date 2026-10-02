#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cascada_cede_por_margen.py -- ¿conviene que la capa de reglas CEDA cuando el texto esta muy
seguro?

DE DONDE SALE. Hallazgo no previsto del 2026-09-26 (similitud_vs_acierto_p2bal.py): en el tramo
facil -- notas con una hermana contenida >= 0,5 en el entrenamiento -- la cascada acierta 0,9891
y el TEXTO SOLO 0,9993. La regla resta ahi. La explicacion es aritmetica: la regla acierta
0,9928 y no 1,0000, y tiene prioridad absoluta sobre el texto, asi que cuando el texto ya iba a
acertar y la regla se equivoca, se pierde. Se ve por familia en CONTI (texto 1,0000 -> cascada
0,9000) y AVOSLOCKER (1,0000 -> 0,9400).

QUE SE PRUEBA. Una sola modificacion, minima: **la regla cede si el margen del texto supera un
umbral**. Con umbral infinito la regla no cede nunca y es la cascada actual; con umbral 0 cede
siempre y queda el texto solo. Todo lo demas -- corpus, vista, clasificador, particion, semillas
-- es identico a la corrida de cabecera.

    prediccion(u) = texto        si la regla aplica Y margen_del_texto >= u
                  = regla        si la regla aplica Y margen_del_texto <  u
                  = texto        si la regla no aplica

=============================================================================================
PREREGISTRO -- escrito y COMMITEADO ANTES de correr (2026-09-26).

C1. PUERTA DE ENTRADA. Con umbral infinito (la regla nunca cede) tiene que reproducir la
    corrida de cabecera: macro-F1 0,7417 y exactitud 0,8123 (tolerancia 0,01, fuente
    _log_p2bal_149.txt). Si no reproduce, ABORTA y no se reporta nada.
C2. Existe al menos un umbral con Delta > 0 sobre la cascada actual. Es lo que hace esperar el
    hallazgo del tramo facil. Si NINGUN umbral mejora, el hallazgo no se traduce en metodo y
    hay que escribirlo asi.
C3. La ganancia es CHICA: Delta <= +0,03 de macro-F1 en el mejor umbral. Razon: la regla se
    equivoca en el 0,72 % de las notas que resuelve, y solo una parte de esas tiene ademas un
    margen de texto alto. Si diera mucho mas, habria que sospechar de un error.
C4. CRITERIO DE ADOPCION, FIJADO ANTES DE VER NADA. Se prueban 6 umbrales candidatos (0,25 ·
    0,50 · 0,75 · 1,00 · 1,50 · 2,00), asi que hay comparacion multiple. Se adopta la variante
    solo si el Delta pareado por semilla del mejor umbral tiene IC de Bonferroni al 99,17 %
    (alfa = 0,05/6) que EXCLUYE EL CERO POR ARRIBA. Si no lo excluye, NO SE ADOPTA y se reporta
    como variante explorada y descartada. No se va a elegir el umbral mirando el resultado.
C5. CONTROL DE COHERENCIA. Con umbral 0 (la regla cede siempre) el resultado tiene que ser el
    texto solo: macro-F1 0,6551 y exactitud 0,7191. Si no da eso, la implementacion esta mal.

NOTA DE HONESTIDAD: aunque se adopte, el umbral quedaria elegido sobre el MISMO conjunto con el
que se mide, sin particion de validacion aparte. Eso se declara al reportarlo; con 149 notas no
alcanza para partir en tres.
=============================================================================================

Uso:  python cascada_cede_por_margen.py [--n-semillas 50] [--salida CARPETA]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef

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
OUT_DEF = RAIZ / "4_resultados" / "resultados_cascada_cede"
CANON_F1, CANON_ACC, TOL = 0.7417, 0.8123, 0.01
CANON_F1_TXT, CANON_ACC_TXT = 0.6551, 0.7191
CANDIDATOS = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]     # los 6 que entran a Bonferroni
UMBRALES = [0.0] + CANDIDATOS + [np.inf]              # 0 e inf son controles, no candidatos
ALFA_BONF = 0.05 / len(CANDIDATOS)


def ic(v, alfa=0.05):
    v = np.asarray(v, float)
    m, n = float(np.mean(v)), len(v)
    s = float(np.std(v, ddof=1)) if n > 1 else 0.0
    h = t_dist.ppf(1 - alfa / 2, n - 1) * (s / np.sqrt(n)) if n > 1 else 0.0
    return m, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=50)
    ap.add_argument("--sin-puerta", action="store_true")
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  ¿CONVIENE QUE LA REGLA CEDA CUANDO EL TEXTO ESTA MUY SEGURO? -- P2bal")
    print("=" * 78)
    textos, textos_arr, y, archivos, grupos, iocs, nombres_nota = cargar_todo()
    familias = np.unique(y)
    n = len(y)
    ppf = plantillas_por_familia(y, grupos)
    evaluables = [f for f in familias if ppf[f] >= 2]
    print(f"Notas: {n} | Familias: {len(familias)} | Plantillas: {len(set(grupos))}")
    print(f"Semillas: {args.n_semillas} | Umbrales de cesion: {UMBRALES}")
    print(f"Candidatos a adopcion: {CANDIDATOS} -> Bonferroni alfa = {ALFA_BONF:.5f}\n")

    pred = {u: np.empty((args.n_semillas, n), dtype=object) for u in UMBRALES}
    cede = {u: np.zeros((args.n_semillas, n), bool) for u in UMBRALES}

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
            clases = clf.classes_
            orden = np.argsort(-dec, axis=1)
            pt = clases[orden[:, 0]]
            margen = (dec[np.arange(len(te)), orden[:, 0]]
                      - dec[np.arange(len(te)), orden[:, 1]])
            d = dicc_privados(tr, iocs, nombres_nota, y)
            for k, i in enumerate(te):
                r = regla(i, d, iocs, nombres_nota)
                for u in UMBRALES:
                    if r is None:
                        pred[u][s, i] = pt[k]
                    elif margen[k] >= u:
                        pred[u][s, i] = pt[k]
                        cede[u][s, i] = True
                    else:
                        pred[u][s, i] = r
        if (s + 1) % 10 == 0:
            print(f"  {s+1}/{args.n_semillas} semillas")

    def met(u):
        f1 = [f1_score(y, pred[u][s], average="macro", labels=familias, zero_division=0)
              for s in range(args.n_semillas)]
        f1e = [f1_score(y, pred[u][s], average="macro", labels=evaluables, zero_division=0)
               for s in range(args.n_semillas)]
        acc = [accuracy_score(y, pred[u][s]) for s in range(args.n_semillas)]
        bal = [balanced_accuracy_score(y, pred[u][s]) for s in range(args.n_semillas)]
        mcc = [matthews_corrcoef(y, pred[u][s]) for s in range(args.n_semillas)]
        return np.array(f1), np.array(f1e), np.array(acc), np.array(bal), np.array(mcc)

    M = {u: met(u) for u in UMBRALES}
    f1_inf = M[np.inf][0]

    # ---------------- puertas C1 y C5 ----------------
    print("\n" + "-" * 78)
    print("  C1 -- PUERTA: umbral infinito debe ser la cascada de cabecera")
    print("-" * 78)
    f1i, acci = float(f1_inf.mean()), float(M[np.inf][2].mean())
    print(f"  macro-F1 {f1i:.4f} vs {CANON_F1} | exactitud {acci:.4f} vs {CANON_ACC}")
    ok1 = abs(f1i - CANON_F1) <= TOL and abs(acci - CANON_ACC) <= TOL
    f10, acc0 = float(M[0.0][0].mean()), float(M[0.0][2].mean())
    print(f"  C5 -- umbral 0 debe ser el texto solo: macro-F1 {f10:.4f} vs {CANON_F1_TXT} | "
          f"exactitud {acc0:.4f} vs {CANON_ACC_TXT}")
    ok5 = abs(f10 - CANON_F1_TXT) <= TOL and abs(acc0 - CANON_ACC_TXT) <= TOL
    if not (ok1 and ok5) and not args.sin_puerta:
        sys.exit("ABORTADO (C1/C5): no reproduce los dos extremos. No se reporta nada.")
    print("  OK\n" if (ok1 and ok5) else "  FUERA DE TOLERANCIA (--sin-puerta)\n")

    # ---------------- barrido ----------------
    filas = []
    for u in UMBRALES:
        f1, f1e, acc, bal, mcc = M[u]
        d = f1 - f1_inf
        m, lo95, hi95 = ic(d)
        _, loB, hiB = ic(d, ALFA_BONF)
        etq = "inf (cascada actual)" if np.isinf(u) else ("0,00 (texto solo)" if u == 0 else f"{u:.2f}")
        filas.append(dict(
            umbral_cesion=etq,
            candidato="si" if u in CANDIDATOS else "no (control)",
            notas_en_que_cede=round(float(cede[u].sum() / args.n_semillas), 1),
            f1_macro_30=round(float(f1.mean()), 4),
            f1_macro_28=round(float(f1e.mean()), 4),
            exactitud=round(float(acc.mean()), 4),
            bal=round(float(bal.mean()), 4), mcc=round(float(mcc.mean()), 4),
            delta_vs_cascada=round(m, 4),
            ic95=f"[{lo95:+.4f}; {hi95:+.4f}]",
            ic_bonferroni=f"[{loB:+.4f}; {hiB:+.4f}]",
            semillas_positivas=f"{int((d > 0).sum())}/{args.n_semillas}",
            excluye_cero_bonf="SI" if loB > 0 else "no"))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "cascada_cede_barrido.csv", index=False, encoding="utf-8-sig")
    print("=== BARRIDO (P2bal, 149 notas, 50 semillas) ===")
    print(df.drop(columns=["ic95"]).to_string(index=False))

    # ---------------- veredicto ----------------
    cand = [f for f in filas if f["candidato"] == "si"]
    mejor = max(cand, key=lambda f: f["delta_vs_cascada"])
    adopta = mejor["excluye_cero_bonf"] == "SI"
    print("\n" + "=" * 78)
    print("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    print("=" * 78)
    chk = [
        ("C1 puerta: umbral inf = cascada de cabecera", ok1, f"{f1i:.4f} / {acci:.4f}"),
        ("C2 existe algun umbral con Delta > 0",
         any(f["delta_vs_cascada"] > 0 for f in cand),
         f"mejor {mejor['umbral_cesion']} con {mejor['delta_vs_cascada']:+.4f}"),
        ("C3 la ganancia es chica (Delta <= +0,03)",
         mejor["delta_vs_cascada"] <= 0.03, f"{mejor['delta_vs_cascada']:+.4f}"),
        ("C4 ADOPCION: IC de Bonferroni excluye el cero", adopta,
         f"{mejor['umbral_cesion']} -> {mejor['ic_bonferroni']}"),
        ("C5 coherencia: umbral 0 = texto solo", ok5, f"{f10:.4f} / {acc0:.4f}"),
    ]
    for nombre, cumple, det in chk:
        print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<48} {det}")

    print("\n  DECISION POR EL CRITERIO PREREGISTRADO:")
    if adopta:
        print(f"    SE ADOPTA el umbral de cesion {mejor['umbral_cesion']}: macro-F1 "
              f"{mejor['f1_macro_30']:.4f} (cascada actual {f1i:.4f}), Delta "
              f"{mejor['delta_vs_cascada']:+.4f} {mejor['ic_bonferroni']}, "
              f"{mejor['semillas_positivas']} semillas.")
        print("    AL REPORTARLO: el umbral se eligio sobre el mismo conjunto con el que se")
        print("    mide, sin validacion aparte. Con 149 notas no alcanza para partir en tres.")
    else:
        print(f"    NO SE ADOPTA. El mejor umbral ({mejor['umbral_cesion']}) da "
              f"{mejor['delta_vs_cascada']:+.4f} con IC de Bonferroni "
              f"{mejor['ic_bonferroni']}, que no excluye el cero.")
        print("    Se reporta como variante explorada y descartada: la cascada queda como esta.")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
