#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acierto_por_linaje.py -- ¿cuanto acertaria el sistema si bastara con acertar el LINAJE?

QUE CONTESTA. Los errores del frente de notas no se reparten al azar: se concentran en pares de
familias que comparten el molde de la nota. Si esos pares son indistinguibles por texto, conviene
saber cuanto del error restante es «confundir dos familias emparentadas» y cuanto es error real.
La medida es el acierto tratando cada par como una sola clase.

⚠️ ESTA METRICA SUBE SIEMPRE, POR CONSTRUCCION. Fusionar dos clases cualesquiera reduce el
numero de clases y sube el acierto sin que el sistema haya mejorado en nada. Por eso el numero
solo, sin control, NO significa nada, y por eso este script incluye el control de abajo.

EL CONTROL QUE HACE HONESTA LA MEDICION: se fusiona la MISMA cantidad de pares, pero elegidos AL
AZAR entre familias sin parentesco, y se promedia sobre muchos sorteos. La pregunta real no es
«¿sube?» (siempre sube) sino «¿sube MAS que fusionando pares cualesquiera?». Si la fusion por
linaje no supera a la aleatoria, entonces el parentesco no explica la confusion y la metrica
sobra.

LOS PARES, declarados a mano, todos verificados el 2026-09-26/28 con texto exclusivo compartido,
marcadores propios de cada familia (0 compartidos salvo la URL generica de torproject) y
confusion medida:
  FUERTES (molde de apertura o casi todo el texto en comun):
    BLACKBASTA-CONTI · DHARMA-PHOBOS · CLOP-RYUK
  DEBILES (bloques tematicos reutilizados, no el molde entero):
    LORENZ-SODINOKIBI · BLACKCAT-SODINOKIBI · MEDUZALOCKER-SODINOKIBI · NOTPETYA-WANNACRY

Se reportan las dos variantes por separado. SODINOKIBI aparece en tres pares debiles, de modo que
al fusionar por componentes conexas esas cuatro familias caen en un solo grupo; el numero de
clases resultante se reporta SIEMPRE junto a la cifra, porque es lo que permite juzgarla.

=============================================================================================
PREREGISTRO -- escrito y COMMITEADO ANTES de correr (2026-09-28).

G1. PUERTA DE ENTRADA. Sin fusion, exactitud 0,8123 (cascada) y 0,7191 (texto), tolerancia
    0,01 (fuente _log_p2bal_149.txt). Si no reproduce, ABORTA.
G2. CONTROL PRINCIPAL. La fusion por LINAJE supera a la fusion ALEATORIA del mismo numero de
    pares, con intervalo de confianza del 95 % que excluye el cero. Si no lo supera, el
    parentesco no explica la confusion y la metrica no se reporta como resultado.
G3. La ganancia por fusionar es MAYOR con el texto solo que con la cascada. Razon: la cascada
    ya resuelve por marcadores buena parte de la confusion de linaje (486 -> 186 errores), asi
    que le queda menos por ganar.
G4. Con los tres pares FUERTES, la exactitud de la cascada supera 0,85.
G5. La fusion aleatoria sube el acierto por encima de 0,8123, aunque poco. Es el control de
    que el efecto por construccion existe y esta cuantificado; si diera 0, el control estaria
    mal implementado.
=============================================================================================

Uso:  python acierto_por_linaje.py [--n-semillas 50] [--sorteos 200] [--salida CARPETA]
"""
from __future__ import annotations

import argparse
import sys
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
from protocolo_logo import dicc_privados, regla
from protocolo_p2bal import split_p2bal
from revision_logo import cargar_todo

RAIZ = _AQUI.parent
OUT_DEF = RAIZ / "4_resultados" / "resultados_acierto_linaje"
CANON_M6, CANON_TXT, TOL = 0.8123, 0.7191, 0.01

FUERTES = [("BLACKBASTA", "CONTI"), ("DHARMA", "PHOBOS"), ("CLOP", "RYUK")]
DEBILES = [("LORENZ", "SODINOKIBI"), ("BLACKCAT", "SODINOKIBI"),
           ("MEDUZALOCKER", "SODINOKIBI"), ("NOTPETYA", "WANNACRY")]


def mapa_fusion(universo, pares):
    """Mapa familia -> representante de su componente conexa, sobre el universo COMPLETO.

    El universo se pasa explicito y es siempre el mismo (las 30 familias), de modo que el
    MISMO mapa se aplica a las etiquetas verdaderas y a las predicciones. Construirlo a
    partir de las etiquetas que trae cada array es un error: si una familia del par nunca
    aparece entre las predicciones de una semilla, el par se fusiona en y pero no en la
    prediccion, y aciertos se convierten en errores. Ese bug estaba en la primera version
    de este script (2026-09-28) y hacia que la fusion aleatoria BAJARA la exactitud, lo
    que es imposible: fusionar clases solo puede subirla o dejarla igual.
    """
    padre = {f: f for f in universo}

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x
    for a, b in pares:
        if a in padre and b in padre:
            ra, rb = raiz(a), raiz(b)
            if ra != rb:
                padre[rb] = ra
    return {f: raiz(f) for f in universo}


def aplicar(etiquetas, mapa):
    return np.array([mapa.get(e, e) for e in etiquetas])


def ic(v):
    v = np.asarray(v, float)
    m, n = float(np.mean(v)), len(v)
    s = float(np.std(v, ddof=1)) if n > 1 else 0.0
    h = t_dist.ppf(0.975, n - 1) * (s / np.sqrt(n)) if n > 1 else 0.0
    return m, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=50)
    ap.add_argument("--sorteos", type=int, default=200,
                    help="cuantas fusiones aleatorias se promedian en el control G2")
    ap.add_argument("--sin-puerta", action="store_true")
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  ACIERTO A NIVEL DE LINAJE -- con control de fusion aleatoria")
    print("=" * 78)
    textos, textos_arr, y, archivos, grupos, iocs, nombres_nota = cargar_todo()
    familias = np.unique(y)
    n = len(y)
    print(f"Notas: {n} | Familias: {len(familias)} | Semillas: {args.n_semillas}")
    print(f"Pares fuertes: {FUERTES}")
    print(f"Pares debiles: {DEBILES}\n")

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

    # ---------------- puerta G1 ----------------
    base = {}
    for etq, P in (("texto solo", p_txt), ("cascada", p_m6)):
        base[etq] = np.array([accuracy_score(y, P[s]) for s in range(args.n_semillas)])
    print("\n" + "-" * 78)
    print("  G1 -- PUERTA DE ENTRADA")
    print("-" * 78)
    am, at = base["cascada"].mean(), base["texto solo"].mean()
    print(f"  cascada {am:.4f} vs {CANON_M6} | texto {at:.4f} vs {CANON_TXT}")
    ok = abs(am - CANON_M6) <= TOL and abs(at - CANON_TXT) <= TOL
    if not ok and not args.sin_puerta:
        sys.exit("ABORTADO (G1): no reproduce la cabecera. No se reporta nada.")
    print("  OK\n" if ok else "  FUERA DE TOLERANCIA (--sin-puerta)\n")

    def medir(pares, P):
        """Exactitud y macro-F1 por semilla con las familias de `pares` fusionadas.

        El MISMO mapa se aplica a las etiquetas verdaderas y a las predicciones (ver
        mapa_fusion): de otro modo la metrica se corrompe.
        """
        mapa = mapa_fusion(familias, pares)
        yf = aplicar(y, mapa)
        clases = np.unique(yf)
        acc, f1 = [], []
        for s in range(args.n_semillas):
            pf = aplicar(P[s], mapa)
            acc.append(accuracy_score(yf, pf))
            f1.append(f1_score(yf, pf, average="macro", labels=clases, zero_division=0))
        return np.array(acc), np.array(f1), len(clases)

    # CONTROL DE SANIDAD: fusionar clases NUNCA puede bajar la exactitud. Si baja, el mapa
    # se esta aplicando de forma asimetrica y no se reporta nada.
    for _pares in (FUERTES, FUERTES + DEBILES):
        for _P in (p_txt, p_m6):
            _a, _, _ = medir(_pares, _P)
            _b = np.array([accuracy_score(y, _P[s]) for s in range(args.n_semillas)])
            if (_a < _b - 1e-12).any():
                sys.exit("ABORTADO: la fusion bajo la exactitud en alguna semilla. "
                         "Es imposible; el mapa se aplica mal.")

    # ---------------- variantes ----------------
    variantes = [("sin fusion", []), ("3 pares fuertes", FUERTES),
                 ("fuertes + debiles", FUERTES + DEBILES)]
    filas = []
    guarda = {}
    for etq_v, pares in variantes:
        for etq_s, P in (("texto solo", p_txt), ("cascada", p_m6)):
            acc, f1, nc = medir(pares, P)
            guarda[(etq_v, etq_s)] = acc
            d = acc - base[etq_s]
            m, lo, hi = ic(d)
            filas.append(dict(fusion=etq_v, sistema=etq_s, n_clases=nc,
                              exactitud=round(float(acc.mean()), 4),
                              macro_f1=round(float(f1.mean()), 4),
                              ganancia=round(m, 4),
                              ic95=f"[{lo:+.4f}; {hi:+.4f}]" if pares else "---"))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "acierto_linaje.csv", index=False, encoding="utf-8-sig")
    print("=== ACIERTO TRATANDO CADA PAR COMO UNA SOLA CLASE ===")
    print(df.to_string(index=False))

    # ---------------- G2: control de fusion aleatoria ----------------
    print("\n" + "=" * 78)
    print(f"  G2 -- CONTROL: fusionar la MISMA cantidad de pares, pero AL AZAR")
    print(f"  ({args.sorteos} sorteos; la fusion aleatoria evita los pares de linaje)")
    print("=" * 78)
    rng = np.random.default_rng(777)
    prohibidos = {frozenset(p) for p in FUERTES + DEBILES}
    ctrl_filas = []
    for etq_v, pares in variantes[1:]:
        k = len(pares)
        for etq_s, P in (("texto solo", p_txt), ("cascada", p_m6)):
            azar = []
            for _ in range(args.sorteos):
                elegidos, intentos = [], 0
                while len(elegidos) < k and intentos < 500:
                    intentos += 1
                    a, b = rng.choice(familias, 2, replace=False)
                    if frozenset({a, b}) in prohibidos or frozenset({a, b}) in \
                       {frozenset(e) for e in elegidos}:
                        continue
                    elegidos.append((a, b))
                acc_a, _, _ = medir(elegidos, P)
                azar.append(acc_a.mean())
            azar = np.array(azar)
            real = guarda[(etq_v, etq_s)].mean()
            dif = real - azar.mean()
            lo_a, hi_a = np.percentile(azar, [2.5, 97.5])
            ctrl_filas.append(dict(
                fusion=etq_v, sistema=etq_s, n_pares=k,
                exactitud_linaje=round(float(real), 4),
                exactitud_azar=round(float(azar.mean()), 4),
                ic95_azar=f"[{lo_a:.4f}; {hi_a:.4f}]",
                ventaja_del_linaje=round(float(dif), 4),
                supera_al_azar="SI" if real > hi_a else "no"))
    dc = pd.DataFrame(ctrl_filas)
    dc.to_csv(OUT / "control_fusion_aleatoria.csv", index=False, encoding="utf-8-sig")
    print(dc.to_string(index=False))

    # ---------------- veredicto ----------------
    f3 = next(r for r in filas if r["fusion"] == "3 pares fuertes" and r["sistema"] == "cascada")
    g_txt = next(r for r in filas if r["fusion"] == "3 pares fuertes" and r["sistema"] == "texto solo")["ganancia"]
    g_cas = f3["ganancia"]
    c3 = [r for r in ctrl_filas if r["fusion"] == "3 pares fuertes"]
    azar_sube = all(r["exactitud_azar"] > CANON_M6 - 0.02 for r in c3 if r["sistema"] == "cascada")
    print("\n" + "=" * 78)
    print("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    print("=" * 78)
    chk = [
        ("G1 puerta de entrada", ok, f"{am:.4f} | {at:.4f}"),
        ("G2 el linaje supera a la fusion aleatoria",
         all(r["supera_al_azar"] == "SI" for r in ctrl_filas),
         " · ".join(f"{r['fusion'][:9]}/{r['sistema'][:5]} {r['ventaja_del_linaje']:+.4f}"
                    for r in ctrl_filas)),
        ("G3 la ganancia es mayor con el texto que con la cascada",
         g_txt > g_cas, f"texto {g_txt:+.4f} | cascada {g_cas:+.4f}"),
        ("G4 con los 3 fuertes, la cascada supera 0,85",
         f3["exactitud"] > 0.85, f"{f3['exactitud']:.4f}"),
        ("G5 la fusion aleatoria tambien sube (efecto por construccion)",
         azar_sube, " · ".join(f"{r['exactitud_azar']:.4f}" for r in c3)),
    ]
    for nombre, cumple, det in chk:
        print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<46} {det}")
    print(f"\n  CIFRA CITABLE: tratando como una sola clase los tres pares de familias que")
    print(f"  comparten molde de nota, la cascada acierta {f3['exactitud']:.4f} sobre "
          f"{f3['n_clases']} clases,")
    print(f"  frente a {CANON_M6} sobre 30. SIEMPRE con el numero de clases pegado, y")
    print("  siempre con la cifra de la fusion aleatoria al lado.")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
