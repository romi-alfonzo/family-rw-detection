#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cascada_jerarquica_linaje.py -- ¿se puede MEJORAR el acierto usando el hallazgo del linaje?

DE DONDE SALE. El 2026-09-28 se midio que los errores del frente de notas no se reparten al
azar: de los 18,8 puntos de error de la cascada, unos 4,6 son confundir dos familias que
comparten el molde de la nota (0,8585 tratando los pares como una sola clase, contra 0,8132 de
fusionar pares al azar). Eso es un DIAGNOSTICO, no una mejora: agrupar y reportar el linaje
cambia la pregunta, no mejora el sistema.

LA IDEA QUE SI PODRIA MEJORARLO. Un clasificador de 30 clases tiene que separar CLOP de RYUK
con los mismos pesos con que separa CLOP de WANNACRY. Un clasificador entrenado SOLO con las
notas de un par ve unicamente las diferencias entre esas dos familias: el vocabulario que
comparten deja de tener peso discriminante y el que las separa lo gana. Si la confusion de
linaje viene de la INTERFERENCIA de las otras 28 clases, un especialista la resuelve. Si viene
de que las notas son genuinamente indistinguibles, no.

Las dos posibilidades dan resultados opuestos y el experimento las separa.

EL SISTEMA QUE SE PRUEBA (dos etapas, todo entrenado solo con el pliegue de entrenamiento):
  Etapa 1: la cascada de siempre, pero entrenada con las etiquetas FUSIONADAS de los 3 pares
           (27 clases). Predice el GRUPO.
  Etapa 2: si el grupo predicho es uno de los 3 pares, un clasificador BINARIO entrenado solo
           con las notas de ese par decide cual de las dos familias. Se respeta la jerarquia de
           la cascada: si la capa de reglas apunta a una de las dos, manda la regla.
  Si el grupo predicho no es un par, la respuesta es directa.

Se compara contra la CASCADA PLANA de 30 clases, la misma particion y las mismas semillas.

PARES (verificados el 2026-09-26/28: texto exclusivo compartido, marcadores propios de cada
familia, cero contactos en comun):
  BLACKBASTA-CONTI · DHARMA-PHOBOS · CLOP-RYUK

=============================================================================================
PREREGISTRO -- escrito y COMMITEADO ANTES de correr (2026-09-28).

H1. PUERTA DE ENTRADA. El sistema PLANO reproduce la cifra de cabecera: macro-F1 0,7417 y
    exactitud 0,8123 (tolerancia 0,01, fuente _log_p2bal_149.txt). Si no, ABORTA.
H2. La etapa 1 entrenada con 27 clases acierta MAS que fusionar a posteriori las predicciones
    del clasificador de 30 (0,8585). Razon: no gasta capacidad separando lo que no se puede
    separar. Si sale igual o peor, la fusion no aporta nada al entrenamiento y hay que decirlo.
H3. El binario dentro de cada par acierta >= 0,75 sobre las notas de ese par. Es el numero que
    decide todo: la etapa 2 solo puede recuperar lo que acierta.
H4. LA PREGUNTA PRINCIPAL. El sistema jerarquico supera la exactitud de la cascada plana
    (0,8123) con Delta pareado por semilla cuyo IC 95 % excluye el cero. Es la prediccion que
    puede fallar, y es lo que se quiere saber.
H5. INTERPRETACION FIJADA DE ANTEMANO, para no racionalizar despues:
    - Si H4 se cumple, la confusion de linaje venia de la INTERFERENCIA de las otras clases.
    - Si H4 falla Y H3 falla, las notas del par son genuinamente indistinguibles por texto y el
      limite no es del metodo sino del corpus. Es un resultado, no un fracaso.
    - Si H4 falla pero H3 se cumple, el problema esta en la etapa 1: el enrutamiento pierde mas
      de lo que la etapa 2 recupera.
H6. El par CLOP-RYUK es el que menos se beneficia de los tres, porque sus marcadores no se
    repiten entre plantillas (fraccion resuelta por reglas 0,20, la mas baja) y porque
    comparten un bloque de apertura continuo de 100 palabras.

NOTA DE HONESTIDAD: los 3 pares se eligieron mirando los datos (barrido de boilerplate y matriz
de confusion del 2026-09-26). Por lo tanto este experimento NO es una validacion independiente
del criterio de emparejamiento: mide si, DADOS esos pares, la jerarquia ayuda. Se declara asi.
=============================================================================================

Uso:  python cascada_jerarquica_linaje.py [--n-semillas 50] [--salida CARPETA]
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
from revision_logo import cargar_todo

RAIZ = _AQUI.parent
OUT_DEF = RAIZ / "4_resultados" / "resultados_jerarquica_linaje"
CANON_F1, CANON_ACC, TOL = 0.7417, 0.8123, 0.01
CANON_FUSION_POSTERIOR = 0.8585      # fusionar a posteriori las predicciones de 30 clases
PARES = [("BLACKBASTA", "CONTI"), ("DHARMA", "PHOBOS"), ("CLOP", "RYUK")]


def ic(v):
    v = np.asarray(v, float)
    m, nn = float(np.mean(v)), len(v)
    s = float(np.std(v, ddof=1)) if nn > 1 else 0.0
    h = t_dist.ppf(0.975, nn - 1) * (s / np.sqrt(nn)) if nn > 1 else 0.0
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
    print("  CASCADA JERARQUICA: ¿desambiguar dentro del par mejora el acierto?")
    print("=" * 78)
    textos, textos_arr, y, archivos, grupos, iocs, nombres_nota = cargar_todo()
    familias = np.unique(y)
    n = len(y)

    # mapa de fusion: cada familia de un par -> etiqueta del grupo
    grupo_de = {f: f for f in familias}
    for a, b in PARES:
        grupo_de[b] = a                      # el representante es el primero del par
    y_fus = np.array([grupo_de[f] for f in y])
    clases_fus = np.unique(y_fus)
    en_par = {f for p in PARES for f in p}
    par_de = {a: (a, b) for a, b in PARES}   # representante -> par
    print(f"Notas: {n} | Familias: {len(familias)} | Clases fusionadas: {len(clases_fus)}")
    print(f"Pares: {PARES}")
    print(f"Notas que pertenecen a un par: {sum(1 for f in y if f in en_par)}\n")

    p_plano = np.empty((args.n_semillas, n), dtype=object)
    p_etapa1 = np.empty((args.n_semillas, n), dtype=object)   # en espacio fusionado
    p_jerar = np.empty((args.n_semillas, n), dtype=object)    # respuesta final, 30 familias
    ok_bin = {p: [] for p in PARES}                           # acierto del binario por par

    print("Evaluando ...")
    for s in range(args.n_semillas):
        rng = np.random.default_rng(20_000 + s)
        for tr, te in split_p2bal(y, grupos, familias, rng):
            # ---------- sistema PLANO (30 clases), el de siempre ----------
            vec = vectorizador("combinado")
            Xtr = vec.fit_transform(textos_arr[tr])
            Xte = vec.transform(textos_arr[te])
            clf = obtener_modelos(s)["LinearSVC"]
            clf.fit(Xtr, y[tr])
            pt = clf.predict(Xte)
            d = dicc_privados(tr, iocs, nombres_nota, y)
            for k, i in enumerate(te):
                r = regla(i, d, iocs, nombres_nota)
                p_plano[s, i] = pt[k] if r is None else r

            # ---------- ETAPA 1: la misma cascada, con etiquetas FUSIONADAS ----------
            vec1 = vectorizador("combinado")
            X1tr = vec1.fit_transform(textos_arr[tr])
            X1te = vec1.transform(textos_arr[te])
            clf1 = obtener_modelos(s)["LinearSVC"]
            clf1.fit(X1tr, y_fus[tr])
            pt1 = clf1.predict(X1te)
            d1 = dicc_privados(tr, iocs, nombres_nota, y_fus)
            g1 = {}
            for k, i in enumerate(te):
                r = regla(i, d1, iocs, nombres_nota)
                g1[i] = pt1[k] if r is None else r
                p_etapa1[s, i] = g1[i]

            # ---------- ETAPA 2: un binario por par, entrenado SOLO con ese par ----------
            binarios = {}
            for a, b in PARES:
                m_tr = tr[(y[tr] == a) | (y[tr] == b)]
                if len(np.unique(y[m_tr])) < 2:
                    binarios[a] = None        # el pliegue no tiene las dos: no se puede
                    continue
                v2 = vectorizador("combinado")
                X2 = v2.fit_transform(textos_arr[m_tr])
                c2 = obtener_modelos(s)["LinearSVC"]
                c2.fit(X2, y[m_tr])
                binarios[a] = (v2, c2)

            for k, i in enumerate(te):
                g = g1[i]
                if g not in par_de:
                    p_jerar[s, i] = g
                    continue
                a, b = par_de[g]
                # la regla manda tambien aca: si apunta a una de las dos, se usa
                r = regla(i, d, iocs, nombres_nota)
                if r in (a, b):
                    p_jerar[s, i] = r
                elif binarios[a] is None:
                    p_jerar[s, i] = a
                else:
                    v2, c2 = binarios[a]
                    p_jerar[s, i] = c2.predict(v2.transform(textos_arr[[i]]))[0]
                if y[i] in (a, b):
                    ok_bin[(a, b)].append(p_jerar[s, i] == y[i])
        if (s + 1) % 10 == 0:
            print(f"  {s+1}/{args.n_semillas} semillas")

    # ---------------- puerta H1 ----------------
    f1_pl = np.array([f1_score(y, p_plano[s], average="macro", labels=familias, zero_division=0)
                      for s in range(args.n_semillas)])
    ac_pl = np.array([accuracy_score(y, p_plano[s]) for s in range(args.n_semillas)])
    print("\n" + "-" * 78)
    print("  H1 -- PUERTA DE ENTRADA")
    print("-" * 78)
    print(f"  plano: macro-F1 {f1_pl.mean():.4f} vs {CANON_F1} | exactitud {ac_pl.mean():.4f} vs {CANON_ACC}")
    ok = abs(f1_pl.mean() - CANON_F1) <= TOL and abs(ac_pl.mean() - CANON_ACC) <= TOL
    if not ok and not args.sin_puerta:
        sys.exit("ABORTADO (H1): el sistema plano no reproduce la cabecera.")
    print("  OK\n" if ok else "  FUERA DE TOLERANCIA (--sin-puerta)\n")

    # ---------------- metricas ----------------
    f1_je = np.array([f1_score(y, p_jerar[s], average="macro", labels=familias, zero_division=0)
                      for s in range(args.n_semillas)])
    ac_je = np.array([accuracy_score(y, p_jerar[s]) for s in range(args.n_semillas)])
    ac_e1 = np.array([accuracy_score(y_fus, p_etapa1[s]) for s in range(args.n_semillas)])

    filas = [dict(sistema="cascada plana (30 clases)",
                  macro_f1=round(float(f1_pl.mean()), 4), exactitud=round(float(ac_pl.mean()), 4),
                  bal=round(float(np.mean([balanced_accuracy_score(y, p_plano[s])
                                           for s in range(args.n_semillas)])), 4),
                  mcc=round(float(np.mean([matthews_corrcoef(y, p_plano[s])
                                           for s in range(args.n_semillas)])), 4)),
             dict(sistema="cascada JERARQUICA (27 -> binario)",
                  macro_f1=round(float(f1_je.mean()), 4), exactitud=round(float(ac_je.mean()), 4),
                  bal=round(float(np.mean([balanced_accuracy_score(y, p_jerar[s])
                                           for s in range(args.n_semillas)])), 4),
                  mcc=round(float(np.mean([matthews_corrcoef(y, p_jerar[s])
                                           for s in range(args.n_semillas)])), 4))]
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "jerarquica_resumen.csv", index=False, encoding="utf-8-sig")
    print("=== RESULTADO (149 notas, P2bal, 50 semillas) ===")
    print(df.to_string(index=False))

    d_acc = ac_je - ac_pl
    d_f1 = f1_je - f1_pl
    ma, loa, hia = ic(d_acc)
    mf, lof, hif = ic(d_f1)
    print(f"\n  Delta exactitud: {ma:+.4f} [{loa:+.4f}; {hia:+.4f}]  "
          f"{int((d_acc > 0).sum())}/{args.n_semillas} semillas")
    print(f"  Delta macro-F1 : {mf:+.4f} [{lof:+.4f}; {hif:+.4f}]  "
          f"{int((d_f1 > 0).sum())}/{args.n_semillas} semillas")

    print(f"\n  Etapa 1 (27 clases, entrenada fusionada): {ac_e1.mean():.4f}")
    print(f"  Fusion a posteriori de las predicciones de 30 clases: {CANON_FUSION_POSTERIOR}")

    bfilas = []
    for p in PARES:
        v = np.array(ok_bin[p], dtype=float)
        bfilas.append(dict(par=f"{p[0]}-{p[1]}", n_decisiones=len(v),
                           acierto=round(float(v.mean()), 4) if len(v) else np.nan))
    dbin = pd.DataFrame(bfilas)
    dbin.to_csv(OUT / "jerarquica_binarios.csv", index=False, encoding="utf-8-sig")
    print("\n=== ACIERTO DE LA ETAPA 2 (sobre las notas que realmente son del par) ===")
    print(dbin.to_string(index=False))

    # por familia
    ffilas = []
    for f in familias:
        idx = np.where(y == f)[0]
        ffilas.append(dict(familia=f, en_par="SI" if f in en_par else "no", n_notas=len(idx),
                           plano=round(float(np.mean([(p_plano[s, idx] == f).mean()
                                                      for s in range(args.n_semillas)])), 4),
                           jerarquica=round(float(np.mean([(p_jerar[s, idx] == f).mean()
                                                           for s in range(args.n_semillas)])), 4)))
    dff = pd.DataFrame(ffilas)
    dff["delta"] = (dff.jerarquica - dff.plano).round(4)
    dff.sort_values("delta", ascending=False).to_csv(OUT / "jerarquica_por_familia.csv",
                                                     index=False, encoding="utf-8-sig")
    print("\n=== POR FAMILIA (las de los pares y las 5 que mas se mueven) ===")
    print(dff[dff.en_par == "SI"].to_string(index=False))

    # ---------------- veredicto ----------------
    h3 = all(r["acierto"] >= 0.75 for r in bfilas if r["n_decisiones"])
    h4 = loa > 0
    peor_par = min((r for r in bfilas if r["n_decisiones"]), key=lambda r: r["acierto"])
    print("\n" + "=" * 78)
    print("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    print("=" * 78)
    chk = [
        ("H1 puerta de entrada", ok, f"{f1_pl.mean():.4f} / {ac_pl.mean():.4f}"),
        ("H2 etapa 1 supera la fusion a posteriori (0,8585)",
         ac_e1.mean() > CANON_FUSION_POSTERIOR, f"{ac_e1.mean():.4f}"),
        ("H3 todos los binarios aciertan >= 0,75", h3,
         " · ".join(f"{r['par']} {r['acierto']:.3f}" for r in bfilas)),
        ("H4 la jerarquica SUPERA a la plana en exactitud", h4,
         f"D {ma:+.4f} [{loa:+.4f}; {hia:+.4f}]"),
        ("H6 CLOP-RYUK es el par con peor binario",
         peor_par["par"] == "CLOP-RYUK", f"peor: {peor_par['par']} {peor_par['acierto']:.3f}"),
    ]
    for nombre, cumple, det in chk:
        print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<48} {det}")

    print("\n  LECTURA SEGUN H5 (fijada antes de correr):")
    if h4:
        print("    La jerarquia MEJORA: la confusion de linaje venia de la INTERFERENCIA de las")
        print("    otras clases, y un especialista por par la resuelve.")
    elif not h3:
        print("    La jerarquia NO mejora y los binarios tampoco aciertan: las notas del par son")
        print("    GENUINAMENTE INDISTINGUIBLES por texto. El limite es del corpus, no del")
        print("    metodo. Es un resultado, no un fracaso.")
    else:
        print("    La jerarquia NO mejora pese a que los binarios aciertan: el problema esta en")
        print("    el ENRUTAMIENTO de la etapa 1, que pierde mas de lo que la etapa 2 recupera.")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
