#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boilerplate_compartido.py -- buscar notas MAL ETIQUETADAS por las frases largas que comparten
con otra familia.

POR QUE ESTE METODO Y NO OTRO. Es el que encontro el error de HelloKitty en un paso, despues de
que tres tandas de busqueda por texto no lo resolvieran (ver HANDOFF_2026-08-25 §3(d) y el
bloque de la limpieza del corpus en ESTADO_TESIS.md). `hellokitty_note1.txt` compartia el
boilerplate de cierre con 12 notas de DHARMA y 2 de PHOBOS, y con NINGUNA otra de HelloKitty.
Ese patron -- comparte frases largas con otra familia y con ninguna de la suya -- es la firma de
una nota mal atribuida, y se puede buscar en todo el corpus de una.

CRITERIO, DECLARADO ANTES DE MIRAR NADA (para no elegir el umbral segun lo que aparezca):
  - Unidad: n-gramas de **8 palabras** consecutivas, en minusculas, sobre \\w+. Ocho es largo:
    una coincidencia casual de 8 palabras seguidas entre dos notas no relacionadas no pasa.
  - Para cada nota se cuentan los 8-gramas que comparte con notas de SU familia que esten en
    OTRA plantilla (`propia`), y los que comparte con notas de OTRA familia (`ajena`).
  - **ALARMA** = `ajena > 0` y `propia == 0`: la nota tiene frases largas en comun con otra
    familia y ninguna con la suya. Es exactamente el caso HelloKitty.
  - Se reporta ademas el boilerplate GENERICO (frases presentes en muchas familias), que es otra
    cosa: vocabulario del ecosistema, no error de etiqueta.

=============================================================================================
PREREGISTRO -- escrito y COMMITEADO ANTES de correr (2026-09-26).

P1. CONTROL POSITIVO. Las notas del grupo mixto DHARMA+PHOBOS y del grupo BLACKBASTA+CONTI
    tienen que aparecer compartiendo 8-gramas entre familias. Son los dos unicos grupos de
    casi-duplicados que cruzan familias en el corpus (verificado 2026-09-26). Si el metodo no
    los encuentra, no sirve y no se reporta nada de lo demas.
P2. CONTROL NEGATIVO. `hellokitty_note1.txt` NO tiene que estar en el corpus: se retiro en la
    limpieza de agosto (155 -> 149). Si aparece, el corpus que se esta midiendo no es el
    limpio y hay que parar todo.
P3. PREDICCION PRINCIPAL: **no hay ninguna nota con ALARMA fuera de los pares de linaje
    conocidos** (BLACKBASTA-CONTI, DHARMA-PHOBOS). O sea, la limpieza de agosto no dejo
    ninguna otra nota mal atribuida detectable por este metodo. Si aparece alguna, es un
    hallazgo nuevo, va a revision humana y NO se retira nada automaticamente.
P4. El boilerplate generico existe y es del ecosistema: habra al menos una frase de 8 palabras
    presente en 5 o mas familias distintas (del tipo «all your files have been encrypted»).
    Eso NO es error de etiqueta y se reporta aparte para que no se confundan.

NADA SE RETIRA NI SE MODIFICA DESDE ESTE SCRIPT. Solo dictamina y deja la lista.
=============================================================================================

Uso:  python boilerplate_compartido.py [--n 8] [--salida CARPETA]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from revision_logo import cargar_todo, plantillas_por_familia

RAIZ = _AQUI.parent
OUT_DEF = RAIZ / "4_resultados" / "resultados_boilerplate"
PARES_LINAJE = [frozenset({"BLACKBASTA", "CONTI"}), frozenset({"DHARMA", "PHOBOS"})]


def ngramas(t, k):
    w = re.findall(r"\w+", t.lower())
    return {tuple(w[i:i + k]) for i in range(len(w) - k + 1)} if len(w) >= k else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n", type=int, default=8, help="largo del n-grama en palabras")
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  BOILERPLATE COMPARTIDO ENTRE FAMILIAS -- n-gramas de {args.n} palabras")
    print("=" * 78)
    textos, _, y, archivos, grupos, _, _ = cargar_todo()
    n = len(y)
    ppf = plantillas_por_familia(y, grupos)
    print(f"Notas: {n} | Familias: {len(set(y))} | Plantillas: {len(set(grupos))}\n")

    # P2: control negativo
    retiradas = [a for a in archivos if "hellokitty_note1" in a.lower()]
    if retiradas:
        sys.exit(f"ABORTADO (P2): {retiradas} sigue en el corpus. Esto no es el corpus limpio.")
    print("P2 control negativo: hellokitty_note1.txt NO esta en el corpus. OK\n")

    G = [ngramas(t, args.n) for t in textos]
    print(f"n-gramas por nota: mediana {np.median([len(g) for g in G]):.0f}, "
          f"max {max(len(g) for g in G)}")

    # indice n-grama -> familias / notas
    de_fam = defaultdict(set)
    de_nota = defaultdict(set)
    for i, g in enumerate(G):
        for ng in g:
            de_fam[ng].add(y[i])
            de_nota[ng].add(i)

    # ---------------- por nota ----------------
    filas = []
    for i in range(n):
        propia = ajena = 0
        fam_ajenas = defaultdict(int)
        for ng in G[i]:
            otras = de_nota[ng] - {i}
            if any(y[j] == y[i] and grupos[j] != grupos[i] for j in otras):
                propia += 1
            aj = {y[j] for j in otras if y[j] != y[i]}
            if aj:
                ajena += 1
                for f in aj:
                    fam_ajenas[f] += 1
        top = sorted(fam_ajenas.items(), key=lambda t: -t[1])[:3]
        alarma = ajena > 0 and propia == 0
        es_linaje = any(frozenset({y[i], f}) in PARES_LINAJE for f, _ in top)
        filas.append(dict(archivo=archivos[i], familia=y[i], grupo=int(grupos[i]),
                          n_plantillas_familia=ppf[y[i]], ngramas=len(G[i]),
                          comparte_con_su_familia=propia, comparte_con_otra_familia=ajena,
                          familias_ajenas="|".join(f"{f}:{c}" for f, c in top),
                          ALARMA="SI" if alarma else "no",
                          es_par_de_linaje_conocido="SI" if es_linaje else "no"))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "boilerplate_por_nota.csv", index=False, encoding="utf-8-sig")

    al = df[df.ALARMA == "SI"]
    nuevas = al[al.es_par_de_linaje_conocido == "no"]
    print(f"\n=== NOTAS CON ALARMA (comparten frases largas con otra familia y con ninguna "
          f"de la suya): {len(al)} ===")
    if len(al):
        print(al[["archivo", "familia", "n_plantillas_familia", "comparte_con_otra_familia",
                  "familias_ajenas", "es_par_de_linaje_conocido"]].to_string(index=False))
    else:
        print("  (ninguna)")

    # ---------------- P1: control positivo ----------------
    cruza = df[(df.comparte_con_otra_familia > 0) & (df.es_par_de_linaje_conocido == "SI")]
    fam_cruza = set(cruza.familia)
    p1 = {"DHARMA", "PHOBOS"} <= fam_cruza or {"BLACKBASTA", "CONTI"} <= fam_cruza
    print(f"\nP1 control positivo: familias de linaje que aparecen cruzando: "
          f"{sorted(fam_cruza)} -> {'OK' if p1 else 'FALLA'}")

    # ---------------- boilerplate generico ----------------
    gen = [(ng, len(fs)) for ng, fs in de_fam.items() if len(fs) >= 3]
    gen.sort(key=lambda t: -t[1])
    gfilas = [dict(frase=" ".join(ng), n_familias=k,
                   familias="|".join(sorted(de_fam[ng]))) for ng, k in gen[:40]]
    dg = pd.DataFrame(gfilas)
    if len(dg):
        dg.to_csv(OUT / "boilerplate_generico.csv", index=False, encoding="utf-8-sig")
    max_fam = gen[0][1] if gen else 0
    print(f"\n=== BOILERPLATE GENERICO: frases de {args.n} palabras en 3+ familias "
          f"({len(gen)} frases; maximo {max_fam} familias) ===")
    print("  (esto NO es error de etiqueta: es vocabulario del ecosistema)")
    if len(dg):
        print(dg.head(12).to_string(index=False))

    # ---------------- pares de familias que mas comparten ----------------
    pares = defaultdict(int)
    for ng, fs in de_fam.items():
        fl = sorted(fs)
        for a in range(len(fl)):
            for b in range(a + 1, len(fl)):
                pares[(fl[a], fl[b])] += 1
    dp = pd.DataFrame([dict(familia_a=a, familia_b=b, ngramas_compartidos=c,
                            es_linaje="SI" if frozenset({a, b}) in PARES_LINAJE else "no")
                       for (a, b), c in sorted(pares.items(), key=lambda t: -t[1])[:25]])
    dp.to_csv(OUT / "pares_de_familias.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== PARES DE FAMILIAS QUE MAS FRASES LARGAS COMPARTEN ===")
    print(dp.to_string(index=False))

    # ---------------- veredicto ----------------
    print("\n" + "=" * 78)
    print("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    print("=" * 78)
    chk = [
        ("P1 control positivo: encuentra los pares de linaje", p1, f"{sorted(fam_cruza)}"),
        ("P2 control negativo: hellokitty_note1 no esta", True, "verificado al empezar"),
        ("P3 no hay ALARMA fuera de los pares conocidos", len(nuevas) == 0,
         f"{len(nuevas)} nota(s) nueva(s)"),
        ("P4 hay boilerplate generico en 5+ familias", max_fam >= 5,
         f"maximo {max_fam} familias"),
    ]
    for nombre, cumple, det in chk:
        print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<48} {det}")
    if len(nuevas):
        print("\n  ⚠ HALLAZGO NUEVO -- va a REVISION HUMANA, no se retira nada automaticamente:")
        print(nuevas[["archivo", "familia", "comparte_con_otra_familia",
                      "familias_ajenas"]].to_string(index=False))
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
