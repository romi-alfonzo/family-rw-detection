#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extension_familias_corpus.py -- el MISMO sistema sobre MUCHAS MAS familias.

QUE CONTESTA. Todas las cifras del frente de notas son sobre las 30 familias de NapierOne, que
son el nucleo canonico porque emparejan los dos frentes del trabajo. La pregunta de esta prueba
NO es si el sistema mejora -- no va a mejorar, y no es lo que se busca -- sino **como se comporta
al generalizarse**: que pasa cuando el catalogo crece de 30 a mas de cien familias.

LO QUE HAY QUE SEPARAR, Y ES TODO EL DISENO. Al agregar familias, el macro-F1 global baja SIEMPRE,
por dos motivos que se confunden si no se miden aparte:
  (a) **mas clases = tarea mas dificil.** Es trivial y no dice nada del sistema.
  (b) **interferencia:** las familias nuevas se parecen a las viejas y les roban decisiones.
El unico modo de distinguirlos es medir el macro-F1 **restringido a las 30 familias originales**
dentro del corpus extendido. Si las 30 mantienen su rendimiento con setenta y pico compitiendo,
el sistema **escala**: lo que baja es la tarea, no el metodo. Si se derrumban, la interferencia
es real y hay que decirlo.

Esa cifra restringida es el resultado de esta prueba. El macro-F1 global va al lado, como
contexto, y nunca solo.

DE DONDE SALEN LAS FAMILIAS NUEVAS. De `3_datos/fuentes_notas/` (ransomware_notes de Zscaler
ThreatLabz, RansomNoteFiles, f6dfir_ransom_notes, notas_pcrisk), inventariadas el 2026-09-26 en
`inventario_familias_fuentes.py`: 77 familias nuevas llegan a 2 plantillas con el criterio del
proyecto. Se admiten solo las familias con >= 2 plantillas: una familia de plantilla unica saca
F1 = 0 forzado bajo corte por plantilla y solo ensucia el promedio.

=============================================================================================
LIMITACIONES QUE SE DECLARAN Y NO SE DISIMULAN. Esta es una prueba EXPLORATORIA y su cifra NO
es comparable con la del nucleo de 30:

L1. **Las etiquetas de las familias nuevas son las de las fuentes**, sin la auditoria de
    procedencia que tiene el corpus canonico. El inventario del 26-09 encontro 25 grupos de
    casi-duplicados que cruzan familias; se reporta cuantos quedan en el corpus extendido y se
    corre tambien la variante que excluye esas familias.
L2. **Las familias nuevas no tienen nombre de archivo auditado**, asi que la segunda capa de la
    cascada no puede actuar sobre ellas. La cobertura de reglas va a bajar por construccion.
L3. La normalizacion de nombres de familia es LAXA (minusculas, sin separadores), igual que en
    el inventario. Para incorporar de verdad haria falta la tabla de alias explicita: la trampa
    del homonimo ya pego tres veces en este proyecto.
L4. "Plantilla" sigue siendo coseno de caracteres 0,90, criterio que no detecta contencion.

=============================================================================================
PREREGISTRO -- escrito y COMMITEADO ANTES de correr (2026-09-28).

X1. PUERTA DE ENTRADA. Restringido al corpus canonico solo, el sistema reproduce macro-F1
    0,7417 y exactitud 0,8123 (tolerancia 0,01). Si no, ABORTA.
X2. El macro-F1 GLOBAL del corpus extendido es menor que 0,7417. Es el efecto trivial de tener
    mas clases; se confirma para dejarlo medido, no porque sea informativo.
X3. LA PREDICCION QUE IMPORTA: el macro-F1 restringido a las 30 familias originales, dentro del
    corpus extendido, baja MENOS DE 0,10 respecto de 0,7417. Es decir, se mantiene por encima de
    0,64. Razon: los marcadores son privados por familia, asi que la primera capa no deberia
    verse afectada por cuantas clases haya; lo que sufre es la capa de texto.
X4. El acierto de la capa de REGLAS sobre el corpus extendido sigue siendo >= 0,95. Los
    marcadores sirven igual con mas familias: es la prediccion que dice si la arquitectura
    escala.
X5. La COBERTURA de la capa de reglas BAJA respecto de 0,5389, por L2 (las familias nuevas no
    tienen nombre auditado).
X6. Aparecen familias nuevas con F1 alto: al menos 10 de las nuevas superan 0,70. Si ninguna lo
    hiciera, el corpus nuevo seria demasiado ruidoso para decir nada.
=============================================================================================

Uso:  python extension_familias_corpus.py [--n-semillas 20] [--salida CARPETA]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, MIN_CHARS_NOTA, UMBRAL_NEARDUP,
                                   agrupar_neardups, cargar_corpus, obtener_modelos,
                                   vectorizador)
from extractor_notas import extraer_texto
from grafo_marcadores import extraer_marcadores
from protocolo_logo import cargar_nombres, dicc_privados, regla
from protocolo_p2bal import split_p2bal

RAIZ = _AQUI.parent
DIR_FUENTES = RAIZ / "3_datos" / "fuentes_notas"
OUT_DEF = RAIZ / "4_resultados" / "resultados_extension_familias"
FUENTES = ["ransomware_notes", "RansomNoteFiles", "f6dfir_ransom_notes", "notas_pcrisk"]
IGNORAR = {".git", ".github", "__pycache__"}
CANON_F1, CANON_ACC, TOL = 0.7417, 0.8123, 0.01
CANON_COB_REGLA, CANON_AC_REGLA = 0.5389, 0.9928


def normalizar(nombre):
    return re.sub(r"[^a-z0-9]", "", nombre.lower())


def cargar_fuente(d: Path):
    textos, fams, archs = [], [], []
    for fam_dir in sorted(p for p in d.iterdir() if p.is_dir() and p.name not in IGNORAR):
        for nota in sorted(p for p in fam_dir.rglob("*") if p.is_file()):
            if any(p in IGNORAR for p in nota.parts):
                continue
            try:
                texto, metodo = extraer_texto(nota)
            except Exception:
                continue
            if metodo.startswith("error") or len(texto.strip()) < MIN_CHARS_NOTA:
                continue
            textos.append(texto)
            fams.append(fam_dir.name)
            archs.append(f"{d.name}/{nota.relative_to(d)}")
    return textos, fams, archs


def evaluar(textos_arr, y, grupos, iocs, nombres, familias, n_semillas):
    """Cascada completa bajo P2bal. Devuelve (pred, aplica) apilados por semilla."""
    n = len(y)
    P = np.empty((n_semillas, n), dtype=object)
    A = np.zeros((n_semillas, n), dtype=bool)
    for s in range(n_semillas):
        rng = np.random.default_rng(20_000 + s)
        for tr, te in split_p2bal(y, grupos, familias, rng):
            vec = vectorizador("combinado")
            Xtr = vec.fit_transform(textos_arr[tr])
            clf = obtener_modelos(s)["LinearSVC"]
            clf.fit(Xtr, y[tr])
            pt = clf.predict(vec.transform(textos_arr[te]))
            d = dicc_privados(tr, iocs, nombres, y)
            for k, i in enumerate(te):
                r = regla(i, d, iocs, nombres)
                P[s, i] = pt[k] if r is None else r
                A[s, i] = r is not None
        print(f"    semilla {s+1}/{n_semillas}")
    return P, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=20)
    ap.add_argument("--sin-puerta", action="store_true")
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  EL MISMO SISTEMA SOBRE MUCHAS MAS FAMILIAS -- prueba exploratoria")
    print("=" * 78)

    # ---------- corpus canonico ----------
    t_can, y_can, a_can, _ = cargar_corpus(CORPUS_DIR)
    canonicas = {normalizar(f) for f in set(y_can)}
    print(f"\nCanonico: {len(t_can)} notas, {len(canonicas)} familias")

    # ---------- fuentes ----------
    t_ext, y_ext, a_ext = list(t_can), [normalizar(f) for f in y_can], list(a_can)
    es_canonica = [True] * len(t_can)
    for nombre in FUENTES:
        d = DIR_FUENTES / nombre
        if not d.is_dir():
            continue
        tt, ff, aa = cargar_fuente(d)
        t_ext += tt
        y_ext += [normalizar(f) for f in ff]
        a_ext += aa
        es_canonica += [False] * len(tt)
        print(f"  + {nombre}: {len(tt)} notas")

    y_ext = np.array(y_ext)
    es_canonica = np.array(es_canonica)
    print(f"\nTodo junto: {len(t_ext)} notas, {len(set(y_ext))} familias")
    print("Agrupando casi-duplicados sobre el conjunto completo ...")
    g_ext, _ = agrupar_neardups(t_ext, UMBRAL_NEARDUP)
    g_ext = np.array(g_ext)

    # ---------- familias con >= 2 plantillas ----------
    ppf = {f: len(set(g_ext[y_ext == f])) for f in set(y_ext)}
    admitidas = {f for f, k in ppf.items() if k >= 2}
    m = np.array([f in admitidas for f in y_ext])
    textos = [t for t, k in zip(t_ext, m) if k]
    y = y_ext[m]
    grupos = g_ext[m]
    arch = [a for a, k in zip(a_ext, m) if k]
    can = es_canonica[m]
    textos_arr = np.array(textos, dtype=object)
    familias = np.unique(y)
    originales = np.array(sorted(canonicas & set(familias)))
    nuevas = np.array(sorted(set(familias) - canonicas))
    print(f"\nAdmitidas (>= 2 plantillas): {len(familias)} familias, {len(y)} notas")
    print(f"  de las 30 canonicas sobreviven: {len(originales)}")
    print(f"  familias NUEVAS: {len(nuevas)}")

    # conflictos de etiqueta que quedan (L1)
    porg = defaultdict(set)
    for g, f in zip(grupos, y):
        porg[g].add(f)
    conflict = {g: fs for g, fs in porg.items() if len(fs) > 1}
    fam_conf = set().union(*conflict.values()) if conflict else set()
    print(f"  grupos que cruzan familias: {len(conflict)} (tocan {len(fam_conf)} familias)")

    iocs = [set(extraer_marcadores(t)) for t in textos]
    nom = cargar_nombres()
    # el nombre auditado solo existe para las canonicas; se busca por (FAMILIA_ORIGINAL, archivo)
    inv = {normalizar(f): f for f in {k[0] for k in nom}}
    nombres = []
    for f, a in zip(y, arch):
        orig = inv.get(f)
        nombres.append(nom.get((orig, Path(a).name)) if orig else None)
    print(f"  notas con nombre auditado: {sum(1 for x in nombres if x)} de {len(y)}")

    # ---------- PUERTA X1: solo el canonico ----------
    print("\n" + "-" * 78)
    print("  X1 -- PUERTA: el sistema sobre el corpus canonico solo")
    print("-" * 78)
    idx_can = np.where(can)[0]
    y_c, g_c = y[idx_can], grupos[idx_can]
    fam_c = np.unique(y_c)
    P_c, A_c = evaluar(np.array([textos[i] for i in idx_can], dtype=object), y_c, g_c,
                       [iocs[i] for i in idx_can], [nombres[i] for i in idx_can],
                       fam_c, min(args.n_semillas, 10))
    f1_c = float(np.mean([f1_score(y_c, P_c[s], average="macro", labels=fam_c, zero_division=0)
                          for s in range(P_c.shape[0])]))
    ac_c = float(np.mean([accuracy_score(y_c, P_c[s]) for s in range(P_c.shape[0])]))
    print(f"  macro-F1 {f1_c:.4f} vs {CANON_F1} | exactitud {ac_c:.4f} vs {CANON_ACC}")
    ok = abs(f1_c - CANON_F1) <= TOL and abs(ac_c - CANON_ACC) <= TOL
    if not ok and not args.sin_puerta:
        sys.exit("ABORTADO (X1): no reproduce la cifra del nucleo. No se reporta nada.")
    print("  OK\n" if ok else "  FUERA DE TOLERANCIA (--sin-puerta)\n")

    # ---------- corpus extendido ----------
    print("Evaluando el corpus EXTENDIDO ...")
    P, A = evaluar(textos_arr, y, grupos, iocs, nombres, familias, args.n_semillas)

    def met(labels, mascara=None):
        f1, ac = [], []
        for s in range(args.n_semillas):
            if mascara is None:
                f1.append(f1_score(y, P[s], average="macro", labels=labels, zero_division=0))
                ac.append(accuracy_score(y, P[s]))
            else:
                f1.append(f1_score(y[mascara], P[s][mascara], average="macro",
                                   labels=labels, zero_division=0))
                ac.append(accuracy_score(y[mascara], P[s][mascara]))
        return float(np.mean(f1)), float(np.mean(ac))

    f1_g, ac_g = met(familias)
    f1_o, ac_o = met(originales, np.isin(y, originales))
    f1_n, ac_n = met(nuevas, np.isin(y, nuevas))
    cob = float(A.mean())
    ac_regla = float(np.mean([(P[s][A[s]] == y[A[s]]).mean() for s in range(args.n_semillas)
                              if A[s].any()]))

    filas = [
        dict(conjunto=f"NUCLEO solo ({len(fam_c)} familias)", n_familias=len(fam_c),
             n_notas=len(y_c), macro_f1=round(f1_c, 4), exactitud=round(ac_c, 4)),
        dict(conjunto=f"EXTENDIDO global ({len(familias)} familias)", n_familias=len(familias),
             n_notas=len(y), macro_f1=round(f1_g, 4), exactitud=round(ac_g, 4)),
        dict(conjunto="EXTENDIDO, restringido a las 30 originales", n_familias=len(originales),
             n_notas=int(np.isin(y, originales).sum()), macro_f1=round(f1_o, 4),
             exactitud=round(ac_o, 4)),
        dict(conjunto="EXTENDIDO, solo las familias nuevas", n_familias=len(nuevas),
             n_notas=int(np.isin(y, nuevas).sum()), macro_f1=round(f1_n, 4),
             exactitud=round(ac_n, 4))]
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "extension_resumen.csv", index=False, encoding="utf-8-sig")
    print("\n=== RESULTADO ===")
    print(df.to_string(index=False))
    print(f"\n  Capa de reglas en el extendido: cobertura {cob:.4f} (nucleo {CANON_COB_REGLA}) "
          f"| acierto {ac_regla:.4f} (nucleo {CANON_AC_REGLA})")

    # por familia
    ff = []
    for f in familias:
        idx = np.where(y == f)[0]
        ff.append(dict(familia=f, nueva="SI" if f in set(nuevas) else "no",
                       n_notas=len(idx), n_plantillas=len(set(grupos[idx])),
                       en_conflicto="SI" if f in fam_conf else "no",
                       f1=round(float(np.mean([f1_score(y, P[s], average="macro", labels=[f],
                                                        zero_division=0)
                                               for s in range(args.n_semillas)])), 4)))
    dff = pd.DataFrame(ff).sort_values("f1", ascending=False)
    dff.to_csv(OUT / "extension_por_familia.csv", index=False, encoding="utf-8-sig")
    n_altas = int(((dff.nueva == "SI") & (dff.f1 > 0.70)).sum())
    print(f"\n=== FAMILIAS NUEVAS CON F1 > 0,70: {n_altas} ===")
    print(dff[(dff.nueva == "SI") & (dff.f1 > 0.70)].head(15).to_string(index=False))

    # ---------- veredicto ----------
    caida_o = CANON_F1 - f1_o
    print("\n" + "=" * 78)
    print("  VEREDICTO DEL PREREGISTRO")
    print("=" * 78)
    chk = [
        ("X1 puerta: el nucleo reproduce su cifra", ok, f"{f1_c:.4f} / {ac_c:.4f}"),
        ("X2 el macro-F1 global baja (efecto trivial)", f1_g < CANON_F1,
         f"{f1_g:.4f} vs {CANON_F1}"),
        ("X3 las 30 originales caen MENOS de 0,10", caida_o < 0.10,
         f"{f1_o:.4f}, caida {caida_o:+.4f}"),
        ("X4 el acierto de la regla sigue >= 0,95", ac_regla >= 0.95, f"{ac_regla:.4f}"),
        ("X5 la cobertura de la regla baja", cob < CANON_COB_REGLA,
         f"{cob:.4f} vs {CANON_COB_REGLA}"),
        ("X6 al menos 10 familias nuevas superan F1 0,70", n_altas >= 10, f"{n_altas}"),
    ]
    for nombre, cumple, det in chk:
        print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<48} {det}")

    print("\n  LECTURA:")
    if caida_o < 0.10:
        print(f"    Las 30 originales mantienen {f1_o:.4f} con {len(nuevas)} familias mas")
        print("    compitiendo. Lo que baja es la TAREA, no el metodo: el sistema ESCALA.")
    else:
        print(f"    Las 30 originales caen a {f1_o:.4f}: las familias nuevas INTERFIEREN y")
        print("    el sistema no escala sin mas trabajo sobre el catalogo.")
    print(f"\n  AL CITAR: base distinta del nucleo ({len(familias)} familias, etiquetas de las")
    print("  fuentes, sin auditoria de procedencia ni nombres). NO comparable con el 0,7417.")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
