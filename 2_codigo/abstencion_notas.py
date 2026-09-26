#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
abstencion_notas.py -- M.3: abstencion por umbral de confianza en el frente de notas.

QUE CONTESTA. La objecion «0,52 de macro-F1 es poco» supone que el sistema SIEMPRE tiene que
contestar. Un identificador de familia desplegado no tiene por que: puede decir «no reconozco
esto». Este experimento entrega la curva **precision-vs-cobertura**: «cuando contesta, acierta
X %; contesta el Y % de las veces».

⚠️ PREDICCION PREREGISTRADA (ya escrita en PLAN_MEJORAS.md §M.3, antes de implementar):
**NO sube el macro-F1. Cambia el reporte.** Si el macro-F1 sobre TODO el conjunto subiera, algo
esta mal: abstenerse no puede mejorar una metrica que se calcula sobre las notas no respondidas.
Lo que tiene que subir es el **acierto donde contesta**, a costa de cobertura.

COMO SE MIDE LA CONFIANZA. LinearSVC no da probabilidades. Se usa el **margen entre la primera
y la segunda clase** de `decision_function` (one-vs-rest): si la mejor clase le gana holgado a
la segunda, la decision es firme; si estan pegadas, es un empate disfrazado. Es la medida
natural para un SVC y no requiere calibrar nada.

LA ARQUITECTURA RESPETA M.6 (la cascada adoptada):
  1. Si la regla exacta aplica (IOC privado o nombre genuino visto en entrenamiento) -> se
     contesta SIEMPRE. La regla ya mide 0,9755 de acierto: abstenerse ahi seria tirar precision.
  2. Si no aplica -> decide el texto, y ahi SI se aplica el umbral de abstencion.
Se reporta tambien la variante «solo texto» para aislar el efecto.

SE REPORTA, por cada umbral:
  cobertura (que fraccion contesta) · acierto donde contesta · macro-F1 sobre las respondidas ·
  cuantas abstenciones · y el desglose entre las que resolvio la regla y las que resolvio el texto.

PROTOCOLO identico a M.6: P2 (grupos, StratifiedGroupKFold 2 pliegues), corpus actual,
LinearSVC(C=1, class_weight=balanced) sobre la vista combinada, 50 semillas.

=============================================================================================
PREREGISTRO -- P2bal (escrito y COMMITEADO ANTES de correr, 2026-09-26)

POR QUE. La frase de despliegue («contesta el X %, y cuando contesta acierta el Y %») hoy solo
existe bajo P2, que reparte mal las plantillas y le regala F1 = 0 a ~2,86 familias por pliegue
sin darles material de entrenamiento. El protocolo de cabecera del frente de notas pasa a ser
P2bal (protocolo_p2bal.py, commit a936397), asi que la curva de abstencion hay que rehacerla
ahi. Se agrega --protocolo P2bal usando la MISMA semilla de reparto que la corrida de cabecera
(rng 20000+s), de modo que las particiones son identicas nota a nota y las dos corridas son
comparables.

A1. PUERTA DE ENTRADA (control externo). Con umbral 0 la cobertura es 1,0000 y el acierto tiene
    que reproducir la exactitud de la corrida de cabecera: P2bal cascada 0,8123 y texto 0,7191;
    P2 cascada 0,6601 y texto 0,5785 (tolerancia 0,01). Si no reproduce, el script ABORTA y no
    se reporta nada.
A2. El acierto donde contesta sube de forma monotona con el umbral y la cobertura baja. Si el
    acierto no sube, el margen del SVC no informa nada y M.3 no se reporta.
A3. El macro-F1 sobre las respondidas NO es una mejora del sistema y no se cita como tal: se
    calcula sobre un subconjunto cada vez mas facil. Va en la tabla para que eso se vea.
A4. Con umbral 0,50 la cascada bajo P2bal contesta MAS que bajo P2 (cobertura > 0,6459) y
    acierta al menos lo mismo (>= 0,9000). Razon: P2bal le garantiza material de entrenamiento
    a toda familia con 2 o mas plantillas, asi que los margenes del texto deberian ser mas
    firmes. Es la prediccion que puede fallar: si la cobertura sube pero el acierto baja, se
    reporta asi.
A5. Entre las notas que resuelve la capa de reglas, el acierto es >= 0,95 bajo P2bal (bajo P2
    fue 0,9755, medido aparte en _log_m6_149.txt). Se agregan al CSV las columnas de acierto
    separado por regla y por texto para poder verificarlo sin otra corrida.
A6. La cobertura de la capa de reglas (las que contesta sin mirar el margen) NO cambia
    materialmente entre P2 y P2bal: ambas ~0,45 (67,7/149 = 0,4546 bajo P2), porque el
    diccionario se arma con ~49,5 plantillas de entrenamiento en los dos protocolos.
    Diferencia esperada <= 0,05.

LIMITACION HEREDADA: «plantilla no vista» es segun coseno de caracteres 0,90, criterio que no
detecta CONTENCION. Ver 6_notas_trabajo/REVISION_LOGO_2026-09-17_informe.md. La limitacion va
pegada al numero, siempre.
=============================================================================================

Uso:  python abstencion_notas.py [--n-semillas 50] [--protocolo P2bal] [--salida CARPETA]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, N_FOLDS, UMBRAL_NEARDUP, agrupar_neardups,
                                   cargar_corpus, obtener_modelos, vectorizador)
from grafo_marcadores import _terminos_circulares, extraer_marcadores
from protocolo_p2bal import split_p2bal

RAIZ = _AQUI.parent
DIR_NOMBRES = RAIZ / "3_datos" / "nombres_notas"
OUT_DEF = RAIZ / "4_resultados" / "resultados_abstencion"
UMBRALES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50]

# PUERTA DE ENTRADA A1: con umbral 0 se contesta todo, asi que el acierto tiene que ser la
# exactitud de la corrida de cabecera de cada protocolo. Fuentes verificables:
#   P2    -> 4_resultados/_log_m6_149.txt y la fila P2 de _log_p2bal_149.txt
#   P2bal -> 4_resultados/_log_p2bal_149.txt (protocolo_p2bal.py, commit a936397)
# LOGO no tiene puerta: esta descartado (REVISION_LOGO_2026-09-17_informe.md).
CANON = {"P2": {"m6": 0.6601, "txt": 0.5785}, "P2bal": {"m6": 0.8123, "txt": 0.7191}}
TOL_CANON = 0.01


def cargar_nombres():
    """(familia, archivo) -> nombre genuino en minusculas. Solo nombres auditados."""
    nombres = {}
    csv_aud = DIR_NOMBRES / "auditoria_nombres_corpus.csv"
    if csv_aud.is_file():
        with open(csv_aud, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f, delimiter=";"):
                if r["nombre_para_m2"]:
                    nombres[(r["familia"], r["archivo_corpus"])] = r["nombre_para_m2"].lower()
    js = DIR_NOMBRES / "nombres_por_nota_2026-08-23.json"
    if js.is_file():
        with open(js, encoding="utf-8") as f:
            for r in json.load(f):
                nm = (r.get("nombre_archivo") or "").strip()
                if r.get("encontrado") and nm and nm != "SIN_ARCHIVO":
                    nombres[(r["familia"], r["archivo_corpus"])] = nm.lower()
    return nombres


def dicc_privados(tr, iocs, nombres_nota, y):
    """Diccionario de la variante ADOPTADA de M.6: IOCs privados + nombre, sin filtro circ."""
    d = defaultdict(set)
    for i in tr:
        for clave in iocs[i]:
            d[clave].add(y[i])
        if nombres_nota[i]:
            d[("[NOMBRE]", nombres_nota[i])].add(y[i])
    for k in [k for k, v in d.items() if len(v) > 1]:
        del d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=50)
    ap.add_argument("--protocolo", choices=["P2", "P2bal", "LOGO"], default="P2",
                    help="P2 = StratifiedGroupKFold 2 pliegues (el reparto viejo). P2bal = P2 "
                         "con el reparto de plantillas arreglado (protocolo de CABECERA del "
                         "frente de notas; usa el mismo rng 20000+s que protocolo_p2bal.py, de "
                         "modo que las particiones son identicas nota a nota). LOGO = leave-"
                         "one-template-out: DESCARTADO el 2026-09-17, se deja solo para "
                         "reproducir lo ya medido; es determinista, usar --n-semillas 1.")
    ap.add_argument("--sin-puerta", action="store_true",
                    help="NO usar salvo depuracion: saltea el control externo A1.")
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  M.3 -- ABSTENCION POR UMBRAL DE CONFIANZA")
    print(f"  protocolo: {args.protocolo}")
    print("=" * 78)
    textos, y, archivos, _ = cargar_corpus(CORPUS_DIR)
    grupos, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    textos_arr = np.array(textos, dtype=object)
    y_np, grupos_np = np.asarray(y), np.asarray(grupos)
    familias = np.unique(y_np)
    n = len(textos)
    iocs = [set(extraer_marcadores(t)) for t in textos]
    nom_aud = cargar_nombres()
    nombres_nota = [nom_aud.get((f, Path(a).name)) for f, a in zip(y, archivos)]
    print(f"Notas: {n} | Familias: {len(set(y))} | Plantillas: {len(set(grupos))}")
    print(f"Con nombre genuino: {sum(1 for x in nombres_nota if x)} | "
          f"Semillas: {args.n_semillas} | Umbrales: {UMBRALES}")

    # margen[i] por semilla, prediccion de texto, y si la regla aplico
    margen = np.zeros((args.n_semillas, n))
    pred_txt = np.empty((args.n_semillas, n), dtype=object)
    pred_regla = np.empty((args.n_semillas, n), dtype=object)
    aplica = np.zeros((args.n_semillas, n), dtype=bool)

    print("\nEvaluando ...")
    for s in range(args.n_semillas):
        if args.protocolo == "LOGO":
            splits = LeaveOneGroupOut().split(textos_arr, y, groups=grupos)
        elif args.protocolo == "P2bal":
            # mismo generador que protocolo_p2bal.py: las particiones coinciden semilla a semilla
            splits = split_p2bal(y_np, grupos_np, familias, np.random.default_rng(20_000 + s))
        else:
            cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
            splits = cv.split(textos_arr, y, groups=grupos)
        for tr, te in splits:
            vec = vectorizador("combinado")
            Xtr = vec.fit_transform(textos_arr[tr])
            Xte = vec.transform(textos_arr[te])
            clf = obtener_modelos(s)["LinearSVC"]
            clf.fit(Xtr, y[tr])
            dec = clf.decision_function(Xte)
            clases = clf.classes_
            orden = np.argsort(-dec, axis=1)
            top1 = clases[orden[:, 0]]
            m = dec[np.arange(len(te)), orden[:, 0]] - dec[np.arange(len(te)), orden[:, 1]]
            pred_txt[s, te] = top1
            margen[s, te] = m
            # capa de reglas de M.6 (variante adoptada)
            d = dicc_privados(tr, iocs, nombres_nota, y)
            for k, i in enumerate(te):
                claves = set(iocs[i])
                if nombres_nota[i]:
                    claves.add(("[NOMBRE]", nombres_nota[i]))
                fams = set()
                for c in claves:
                    if c in d:
                        fams |= d[c]
                if len(fams) == 1:
                    aplica[s, i] = True
                    pred_regla[s, i] = next(iter(fams))
        if (s + 1) % 10 == 0:
            print(f"  {s+1}/{args.n_semillas} semillas")

    y_arr = np.asarray(y)
    filas = []
    for solo_texto in (False, True):
        for u in UMBRALES:
            cob, ac, f1s, n_reg, n_txt, ac_reg, ac_txt = [], [], [], [], [], [], []
            for s in range(args.n_semillas):
                if solo_texto:
                    contesta = margen[s] >= u
                    pred = pred_txt[s]
                else:
                    contesta = aplica[s] | (margen[s] >= u)
                    pred = np.where(aplica[s], pred_regla[s], pred_txt[s])
                cob.append(contesta.mean())
                if contesta.any():
                    ac.append(accuracy_score(y_arr[contesta], pred[contesta]))
                    f1s.append(f1_score(y_arr[contesta], pred[contesta],
                                        average="macro", zero_division=0))
                else:
                    ac.append(np.nan); f1s.append(np.nan)
                n_reg.append(int((aplica[s] & contesta).sum()) if not solo_texto else 0)
                n_txt.append(int((contesta & ~aplica[s]).sum()) if not solo_texto
                             else int(contesta.sum()))
                # A5: acierto separado de las dos capas, entre las que efectivamente contesta
                m_reg = (aplica[s] & contesta) if not solo_texto else np.zeros(n, bool)
                m_txt = (contesta & ~aplica[s]) if not solo_texto else contesta
                ac_reg.append(accuracy_score(y_arr[m_reg], pred[m_reg]) if m_reg.any() else np.nan)
                ac_txt.append(accuracy_score(y_arr[m_txt], pred[m_txt]) if m_txt.any() else np.nan)
            filas.append(dict(
                sistema="solo_texto" if solo_texto else "M.6 (reglas + texto)",
                umbral=u, cobertura=round(float(np.mean(cob)), 4),
                cobertura_sd=round(float(np.std(cob, ddof=1)), 4),
                acierto_donde_contesta=round(float(np.nanmean(ac)), 4),
                f1_macro_de_las_respondidas=round(float(np.nanmean(f1s)), 4),
                abstenciones=round(float((1 - np.mean(cob)) * n), 1),
                resueltas_por_regla=round(float(np.mean(n_reg)), 1),
                resueltas_por_texto=round(float(np.mean(n_txt)), 1),
                acierto_de_la_regla=round(float(np.nanmean(ac_reg)), 4)
                if not np.all(np.isnan(ac_reg)) else np.nan,
                acierto_del_texto=round(float(np.nanmean(ac_txt)), 4)
                if not np.all(np.isnan(ac_txt)) else np.nan))

    df = pd.DataFrame(filas)
    df.to_csv(OUT / "m3_curva_abstencion.csv", index=False, encoding="utf-8-sig")

    for sistema in ("M.6 (reglas + texto)", "solo_texto"):
        print(f"\n=== {sistema} ===")
        print(f"{'umbral':>8}{'cobertura':>12}{'acierto':>10}{'F1 respond.':>13}"
              f"{'abstiene':>10}{'x regla':>9}{'x texto':>9}{'ac.regla':>10}{'ac.texto':>10}")
        for r in filas:
            if r["sistema"] != sistema:
                continue
            ar = r["acierto_de_la_regla"]
            print(f"{r['umbral']:>8.2f}{r['cobertura']:>12.4f}"
                  f"{r['acierto_donde_contesta']:>10.4f}"
                  f"{r['f1_macro_de_las_respondidas']:>13.4f}"
                  f"{r['abstenciones']:>10.1f}{r['resueltas_por_regla']:>9.1f}"
                  f"{r['resueltas_por_texto']:>9.1f}"
                  f"{('--' if ar != ar else format(ar, '.4f')):>10}"
                  f"{r['acierto_del_texto']:>10.4f}")

    base = next(r for r in filas if r["sistema"].startswith("M.6") and r["umbral"] == 0.0)
    base_txt = next(r for r in filas if r["sistema"] == "solo_texto" and r["umbral"] == 0.0)
    print("\n=== CONTROL DE LA PREDICCION PREREGISTRADA ===")
    print(f"Sin abstencion (umbral 0): cobertura {base['cobertura']:.4f} | "
          f"acierto {base['acierto_donde_contesta']:.4f}")
    print("La prediccion dice: el acierto donde contesta DEBE subir con el umbral, y la")
    print("cobertura DEBE bajar. Si el acierto no sube, la confianza del SVC no informa nada.")

    # ---------------- A1: puerta de entrada (control externo) ----------------
    ok = True
    if args.protocolo in CANON:
        c = CANON[args.protocolo]
        d_m6 = abs(base["acierto_donde_contesta"] - c["m6"])
        d_tx = abs(base_txt["acierto_donde_contesta"] - c["txt"])
        print("\n" + "-" * 78)
        print(f"  A1 -- PUERTA DE ENTRADA: umbral 0 debe reproducir la exactitud de {args.protocolo}")
        print("-" * 78)
        print(f"  cascada: {base['acierto_donde_contesta']:.4f} vs canonico {c['m6']}  (dif {d_m6:.4f})")
        print(f"  texto  : {base_txt['acierto_donde_contesta']:.4f} vs canonico {c['txt']}  (dif {d_tx:.4f})")
        ok = d_m6 <= TOL_CANON and d_tx <= TOL_CANON
        if not ok and not args.sin_puerta:
            sys.exit("ABORTADO (A1): el umbral 0 no reproduce la corrida de cabecera. No se reporta nada.")
        print("  OK" if ok else "  FUERA DE TOLERANCIA (se sigue por --sin-puerta)")

    # ---------------- veredicto del preregistro (solo P2bal) ----------------
    if args.protocolo == "P2bal":
        m6 = {r["umbral"]: r for r in filas if r["sistema"].startswith("M.6")}
        ac_u = [m6[u]["acierto_donde_contesta"] for u in UMBRALES]
        cob_u = [m6[u]["cobertura"] for u in UMBRALES]
        r50 = m6[0.50]
        ac_regla0 = m6[0.0]["acierto_de_la_regla"]
        cob_regla = m6[0.0]["resueltas_por_regla"] / n
        print("\n" + "=" * 78)
        print("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
        print("=" * 78)
        chk = [
            ("A1 puerta de entrada (umbral 0 = cabecera)", ok,
             f"cascada {base['acierto_donde_contesta']:.4f} | texto {base_txt['acierto_donde_contesta']:.4f}"),
            ("A2 acierto sube y cobertura baja con el umbral",
             all(v2 >= v1 - 1e-9 for v1, v2 in zip(ac_u, ac_u[1:]))
             and all(v2 <= v1 + 1e-9 for v1, v2 in zip(cob_u, cob_u[1:])),
             f"acierto {ac_u[0]:.4f} -> {ac_u[-1]:.4f} | cobertura {cob_u[0]:.4f} -> {cob_u[-1]:.4f}"),
            ("A4 umbral 0,50: cobertura > 0,6459 y acierto >= 0,9000",
             r50["cobertura"] > 0.6459 and r50["acierto_donde_contesta"] >= 0.9000,
             f"cobertura {r50['cobertura']:.4f} | acierto {r50['acierto_donde_contesta']:.4f}"),
            ("A5 acierto de la capa de reglas >= 0,95",
             ac_regla0 >= 0.95, f"{ac_regla0:.4f}"),
            ("A6 cobertura de la regla ~0,4546 (dif <= 0,05)",
             abs(cob_regla - 0.4546) <= 0.05, f"{cob_regla:.4f} vs 0,4546 en P2"),
        ]
        for nombre, cumple, det in chk:
            print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<48} {det}")
        print("\n  FRASE DE DESPLIEGUE CITABLE (umbral 0,50, P2bal, 149 notas / 30 familias):")
        print(f"    contesta el {r50['cobertura']*100:.1f} % de las notas y, donde contesta, "
              f"acierta el {r50['acierto_donde_contesta']*100:.1f} %.")
        print(f"    De cada {n} notas: {r50['resueltas_por_regla']:.1f} las resuelve la capa de "
              f"reglas y {r50['resueltas_por_texto']:.1f} el texto; se abstiene en "
              f"{r50['abstenciones']:.1f}.")
        print("    RECORDAR AL CITAR: 'plantilla no vista segun coseno char 0,90', criterio que")
        print("    no detecta contencion (REVISION_LOGO_2026-09-17_informe.md).")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
