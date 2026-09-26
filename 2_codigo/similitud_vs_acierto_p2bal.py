#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
similitud_vs_acierto_p2bal.py -- cuanto acierta el sistema segun cuanto se parece la nota a lo
que YA VIO, bajo el protocolo de cabecera P2bal.

QUE CONTESTA. Es la primera pregunta de un jurado: «¿el sistema reconoce FAMILIAS o reconoce
notas parecidas a las que le mostraron?». La respuesta honesta no es un numero, es un desglose:
separar las notas de prueba segun cuanto se parecen al ENTRENAMIENTO DE SU PROPIO PLIEGUE y dar
el acierto en cada tramo. Una nota puede estar en otra plantilla -- la garantia de P2bal, coseno
de caracteres 3-5 por debajo de 0,90 -- y aun asi estar casi CONTENIDA en una nota de
entrenamiento: el coseno no ve la contencion. Este script mide exactamente eso.

COMO SE MIDE EL PARECIDO. Contencion por 3-shingles de palabras (Broder), la misma medida de la
revision independiente del 2026-09-17: cont(i, j) = fraccion de los 3-shingles de palabras de i
que aparecen en j. Es asimetrica a proposito: una nota corta metida dentro de una larga da ~1.
Para cada nota de prueba i se toma el MAXIMO contra las notas de entrenamiento DE SU PROPIA
FAMILIA en ese pliegue: ese es el material con el que el sistema podria haberla reconocido. Se
reporta tambien el coseno char 3-5 maximo contra ese mismo conjunto, para que se vea que el
tramo alto de contencion convive con cosenos por debajo del umbral de casi-duplicado.

DIFERENCIA CON LA MEDICION PREVIA. El desglose ya se habia hecho bajo LOGO, que quedo DESCARTADO
el 2026-09-17: con hermana contenida >= 0,5 el acierto era 1,000 (63 notas) y sin hermana
parecida 0,793 con la cascada y 0,573 con el texto (82 notas). Bajo LOGO el entrenamiento son
TODAS las demas plantillas; bajo P2bal es la mitad. Ademas alla el parecido se media contra el
corpus entero y aca se mide contra el entrenamiento REAL de cada pliegue, que es lo que el
sistema vio. Por las dos razones hay que rehacerlo, y la foto tiene que cambiar.

=============================================================================================
PREREGISTRO -- escrito y COMMITEADO ANTES de correr (2026-09-26). Vive en el docstring para que
el commit deje la marca temporal verificable en git. Lo que se cumpla y lo que falle se reporta
igual.

B1. PUERTA DE ENTRADA (control externo). El acierto global tiene que reproducir la exactitud de
    la corrida de cabecera de P2bal: cascada 0,8123 y texto 0,7191 (tolerancia 0,01; fuente
    4_resultados/_log_p2bal_149.txt, commit a936397). Si no reproduce, el script ABORTA.
B2. El acierto crece con la contencion: tramo [0,9; 1,0] >= tramo [0,5; 0,9) >= tramo [0; 0,5),
    con la cascada y con el texto. Si no creciera, la contencion no explicaria nada.
B3. En el tramo alto (contencion >= 0,9 contra el entrenamiento propio) la cascada acierta
    >= 0,95. Es el tramo que un jurado va a llamar «casi la misma nota».
B4. La brecha cascada - texto es MAYOR en el tramo bajo (< 0,5) que en el alto (>= 0,9): las
    capas de reglas son las que sostienen el caso dificil, no las que inflan el facil. Bajo
    LOGO la brecha sin hermana parecida era +0,220 y con hermana ~0.
B5. La fraccion de notas con hermana contenida >= 0,5 en SU entrenamiento baja respecto de LOGO
    (63/149 = 0,4228): prediccion <= 0,35, porque P2bal entrena con la mitad de las plantillas.
B6. CONTROL DE SANIDAD. En las notas cuya familia no tiene NINGUNA plantilla en el
    entrenamiento de ese pliegue, el acierto es EXACTAMENTE 0,0000 con texto y con cascada: el
    clasificador no tiene esa clase y el diccionario no tiene ninguna clave suya. Si diera > 0
    habria una fuga y no se reporta nada de esta corrida.

LO QUE ESTA MEDICION NO DICE. No dice que las notas del tramo alto «no valgan»: son notas
reales, de plantillas distintas segun el criterio declarado. Dice que el sistema se apoya en el
parecido, cuanto, y que sin parecido sigue acertando algo. La cifra citable del frente de notas
sigue siendo la de P2bal; esta tabla va al lado, como desglose.
=============================================================================================

Uso:  python similitud_vs_acierto_p2bal.py [--n-semillas 50] [--salida CARPETA]
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
from protocolo_p2bal import split_p2bal
from revision_logo import cargar_todo, matriz_contencion, plantillas_por_familia, sim_char

RAIZ = _AQUI.parent
OUT_DEF = RAIZ / "4_resultados" / "resultados_similitud_p2bal"

# B1. Fuente: 4_resultados/_log_p2bal_149.txt (149 notas, 30 familias, 99 plantillas, 50 semillas)
CANON_ACC_M6, CANON_ACC_TXT, TOL_CANON = 0.8123, 0.7191, 0.01

# Tramos de contencion maxima contra el entrenamiento propio. El primero (cont < 0) es la fila
# aparte: familias que ese pliegue dejo sin ninguna plantilla de entrenamiento.
TRAMOS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
SIN_FAMILIA = "sin plantilla propia en entrenamiento"


def ic_t(v):
    """Media e IC 95 % t-Student entre semillas. Ignora los nan (tramos vacios)."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    m, nn = float(np.mean(v)), len(v)
    s = float(np.std(v, ddof=1)) if nn > 1 else 0.0
    h = t_dist.ppf(0.975, nn - 1) * (s / np.sqrt(nn)) if nn > 1 else 0.0
    return m, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=50)
    ap.add_argument("--sin-puerta", action="store_true",
                    help="NO usar salvo depuracion: saltea el control externo B1.")
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  PARECIDO CON EL ENTRENAMIENTO vs ACIERTO -- protocolo P2bal")
    print("=" * 78)
    textos, textos_arr, y, archivos, grupos, iocs, nombres_nota = cargar_todo()
    familias = np.unique(y)
    n = len(y)
    ppf = plantillas_por_familia(y, grupos)
    print(f"Notas: {n} | Familias: {len(familias)} | Plantillas: {len(set(grupos))}")
    print(f"Semillas: {args.n_semillas}")
    print("Parecido = contencion por 3-shingles de palabras contra el entrenamiento de SU familia\n")

    C = matriz_contencion(textos)
    S = sim_char(textos)

    cont = np.full((args.n_semillas, n), -1.0)
    cos = np.full((args.n_semillas, n), -1.0)
    ok_txt = np.zeros((args.n_semillas, n), bool)
    ok_m6 = np.zeros((args.n_semillas, n), bool)
    por_regla = np.zeros((args.n_semillas, n), bool)

    print("Evaluando ...")
    for s in range(args.n_semillas):
        rng = np.random.default_rng(20_000 + s)     # mismo reparto que protocolo_p2bal.py
        for tr, te in split_p2bal(y, grupos, familias, rng):
            vec = vectorizador("combinado")
            Xtr = vec.fit_transform(textos_arr[tr])
            Xte = vec.transform(textos_arr[te])
            clf = obtener_modelos(s)["LinearSVC"]
            clf.fit(Xtr, y[tr])
            pt = clf.predict(Xte)
            d = dicc_privados(tr, iocs, nombres_nota, y)
            for k, i in enumerate(te):
                r = regla(i, d, iocs, nombres_nota)
                pm = pt[k] if r is None else r
                por_regla[s, i] = r is not None
                ok_txt[s, i] = pt[k] == y[i]
                ok_m6[s, i] = pm == y[i]
                propias = tr[y[tr] == y[i]]          # entrenamiento de SU familia, ese pliegue
                if len(propias):
                    cont[s, i] = C[i, propias].max()
                    cos[s, i] = S[i, propias].max()
        if (s + 1) % 10 == 0:
            print(f"  {s+1}/{args.n_semillas} semillas")

    # ---------------- B1: puerta de entrada ----------------
    acc_m6 = float(ok_m6.mean())
    acc_txt = float(ok_txt.mean())
    print("\n" + "-" * 78)
    print("  B1 -- PUERTA DE ENTRADA: el acierto global debe reproducir P2bal")
    print("-" * 78)
    print(f"  cascada: {acc_m6:.4f} vs canonico {CANON_ACC_M6}  (dif {abs(acc_m6-CANON_ACC_M6):.4f})")
    print(f"  texto  : {acc_txt:.4f} vs canonico {CANON_ACC_TXT}  (dif {abs(acc_txt-CANON_ACC_TXT):.4f})")
    ok_puerta = (abs(acc_m6 - CANON_ACC_M6) <= TOL_CANON
                 and abs(acc_txt - CANON_ACC_TXT) <= TOL_CANON)
    if not ok_puerta and not args.sin_puerta:
        sys.exit("ABORTADO (B1): no reproduce la corrida de cabecera. No se reporta nada.")
    print("  OK\n" if ok_puerta else "  FUERA DE TOLERANCIA (se sigue por --sin-puerta)\n")

    # ---------------- tabla por tramo ----------------
    def resumen(mascara_de):
        """mascara_de(s) -> bool[n] con las notas del tramo en la semilla s."""
        ns, at, am, cs = [], [], [], []
        for s in range(args.n_semillas):
            m = mascara_de(s)
            ns.append(int(m.sum()))
            at.append(float(ok_txt[s, m].mean()) if m.any() else np.nan)
            am.append(float(ok_m6[s, m].mean()) if m.any() else np.nan)
            mc = m & (cos[s] >= 0)      # el coseno solo tiene sentido si hubo material propio
            cs.append(float(cos[s, mc].mean()) if mc.any() else np.nan)
        mt, lt, ht = ic_t(at)
        mm, lm, hm = ic_t(am)
        return dict(n_notas_medio=round(float(np.mean(ns)), 1),
                    frac_del_corpus=round(float(np.mean(ns)) / n, 4),
                    acierto_texto=round(mt, 4), ic95_texto=f"[{lt:.4f}; {ht:.4f}]",
                    acierto_cascada=round(mm, 4), ic95_cascada=f"[{lm:.4f}; {hm:.4f}]",
                    brecha_cascada_menos_texto=round(mm - mt, 4),
                    coseno_medio_con_entrenamiento=round(float(np.nanmean(cs)), 4)
                    if not np.all(np.isnan(cs)) else np.nan)

    filas = [dict(tramo_contencion=SIN_FAMILIA, **resumen(lambda s: cont[s] < 0))]
    for lo, hi in TRAMOS:
        filas.append(dict(tramo_contencion=f"[{lo:.1f}; {hi:.1f})",
                          **resumen(lambda s, lo=lo, hi=hi: (cont[s] >= lo) & (cont[s] < hi))))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "similitud_tramos_p2bal.csv", index=False, encoding="utf-8-sig")
    print("=== ACIERTO POR TRAMO DE PARECIDO CON EL ENTRENAMIENTO (media de 50 semillas) ===")
    print(df.to_string(index=False))

    # ---------------- corte binario, la frase citable ----------------
    bins = [("con hermana contenida >= 0,5 en entrenamiento", lambda s: cont[s] >= 0.5),
            ("sin hermana parecida (contencion < 0,5)", lambda s: (cont[s] >= 0) & (cont[s] < 0.5)),
            (SIN_FAMILIA, lambda s: cont[s] < 0)]
    fb = [dict(grupo=et, **resumen(f)) for et, f in bins]
    dfb = pd.DataFrame(fb)
    dfb.to_csv(OUT / "similitud_binario_p2bal.csv", index=False, encoding="utf-8-sig")
    print("\n=== CORTE BINARIO (el que se cita) ===")
    print(dfb.to_string(index=False))

    # ---------------- por familia ----------------
    ffilas = []
    for f in familias:
        idx = np.where(y == f)[0]
        cf = cont[:, idx]
        ffilas.append(dict(
            familia=f, n_notas=len(idx), n_plantillas=ppf[f],
            contencion_media_con_entrenamiento=round(float(cf[cf >= 0].mean()), 4)
            if (cf >= 0).any() else np.nan,
            frac_semillas_sin_material=round(float((cf < 0).mean()), 4),
            acierto_texto=round(float(ok_txt[:, idx].mean()), 4),
            acierto_cascada=round(float(ok_m6[:, idx].mean()), 4),
            frac_resuelta_por_regla=round(float(por_regla[:, idx].mean()), 4)))
    dff = pd.DataFrame(ffilas).sort_values("acierto_cascada")
    dff.to_csv(OUT / "similitud_por_familia_p2bal.csv", index=False, encoding="utf-8-sig")
    print("\n=== POR FAMILIA (ordenado por acierto de la cascada) ===")
    print(dff.to_string(index=False))

    # ---------------- por nota ----------------
    dfn = pd.DataFrame(dict(
        archivo=archivos, familia=y, grupo=grupos,
        n_plantillas_familia=[ppf[f] for f in y],
        contencion_media=np.round(np.nanmean(np.where(cont >= 0, cont, np.nan), 0), 4),
        coseno_medio=np.round(np.nanmean(np.where(cos >= 0, cos, np.nan), 0), 4),
        frac_semillas_sin_material=np.round((cont < 0).mean(0), 4),
        acierto_texto=np.round(ok_txt.mean(0), 4),
        acierto_cascada=np.round(ok_m6.mean(0), 4),
        frac_resuelta_por_regla=np.round(por_regla.mean(0), 4)))
    dfn.to_csv(OUT / "similitud_por_nota_p2bal.csv", index=False, encoding="utf-8-sig")

    # ---------------- veredicto del preregistro ----------------
    g = {r["tramo_contencion"]: r for r in filas}
    bajo = [r for r in fb if r["grupo"].startswith("sin hermana parecida")][0]
    alto = [r for r in fb if r["grupo"].startswith("con hermana")][0]
    sinfam = g[SIN_FAMILIA]
    t_alto, t_med, t_bajo = g["[0.9; 1.0)"], g["[0.5; 0.7)"], g["[0.0; 0.3)"]
    creciente = (t_alto["acierto_cascada"] >= t_med["acierto_cascada"] >= t_bajo["acierto_cascada"]
                 and t_alto["acierto_texto"] >= t_med["acierto_texto"] >= t_bajo["acierto_texto"])
    print("\n" + "=" * 78)
    print("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    print("=" * 78)
    chk = [
        ("B1 puerta de entrada (acierto global = P2bal)", ok_puerta,
         f"cascada {acc_m6:.4f} | texto {acc_txt:.4f}"),
        ("B2 el acierto crece con la contencion", creciente,
         f"cascada {t_bajo['acierto_cascada']:.4f} -> {t_med['acierto_cascada']:.4f} -> "
         f"{t_alto['acierto_cascada']:.4f}"),
        ("B3 tramo >= 0,9: cascada acierta >= 0,95",
         t_alto["acierto_cascada"] >= 0.95, f"{t_alto['acierto_cascada']:.4f}"),
        ("B4 la brecha es mayor en el tramo bajo que en el alto",
         bajo["brecha_cascada_menos_texto"] > t_alto["brecha_cascada_menos_texto"],
         f"bajo {bajo['brecha_cascada_menos_texto']:+.4f} | alto "
         f"{t_alto['brecha_cascada_menos_texto']:+.4f}"),
        ("B5 fraccion con hermana >= 0,5 baja a <= 0,35",
         alto["frac_del_corpus"] <= 0.35,
         f"{alto['frac_del_corpus']:.4f} vs 0,4228 bajo LOGO"),
        ("B6 sanidad: sin familia en entrenamiento, acierto 0,0000",
         (sinfam["acierto_cascada"] == 0.0 and sinfam["acierto_texto"] == 0.0)
         or sinfam["n_notas_medio"] == 0,
         f"cascada {sinfam['acierto_cascada']} | texto {sinfam['acierto_texto']} | "
         f"{sinfam['n_notas_medio']} notas"),
    ]
    for nombre, cumple, det in chk:
        print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<48} {det}")

    print("\n  FRASE CITABLE (P2bal, 149 notas / 30 familias, 50 semillas):")
    print(f"    Cuando la nota tiene una hermana de su familia CONTENIDA >= 0,5 en el "
          f"entrenamiento\n    ({alto['n_notas_medio']:.1f} notas de {n}, el "
          f"{alto['frac_del_corpus']*100:.1f} %), la cascada acierta "
          f"{alto['acierto_cascada']:.4f} {alto['ic95_cascada']}.")
    print(f"    Cuando no la tiene ({bajo['n_notas_medio']:.1f} notas), acierta "
          f"{bajo['acierto_cascada']:.4f} {bajo['ic95_cascada']}, frente a "
          f"{bajo['acierto_texto']:.4f} del texto solo.")
    print(f"    El coseno medio con el entrenamiento en el tramo alto es "
          f"{alto['coseno_medio_con_entrenamiento']:.4f}: por debajo de 0,90, o sea que esas")
    print("    notas son PLANTILLAS DISTINTAS segun el criterio declarado y aun asi estan")
    print("    contenidas. Es la limitacion del criterio, y va pegada al numero.")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
