#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capa_contencion.py -- ¿agrega algo una capa que resuelve por CONTENCION literal?

DE DONDE SALE. La cascada del frente de notas decide en tres pasos: (1) marcadores propios
(IOCs) y nombre genuino del archivo -- la capa de reglas --, y si no alcanza, (2) el texto con
TF-IDF + LinearSVC. El 2026-09-26 se midio que el parecido literal con el entrenamiento explica
buena parte del acierto: el 36,2 % de las notas (54 de 149) tiene una hermana de SU familia
CONTENIDA >= 0,5 en el entrenamiento de su pliegue, y ahi la cascada acierta 0,9891
[0,9848; 0,9934] contra 0,7434 [0,7294; 0,7574] donde no la tiene
(4_resultados/_log_similitud_p2bal_149.txt).

LA IDEA. El agrupamiento de casi-duplicados que define la «plantilla» usa coseno de caracteres
3-5 y NO detecta contencion: una nota corta metida entera dentro de una larga da coseno bajo y
contencion ~1. Ninguna capa de la cascada usa esa senal de forma directa. Esta capa la usa:
si los 3-shingles de palabras de la nota de prueba estan contenidos en una nota de
ENTRENAMIENTO por encima de un umbral u, se responde la familia de esa nota.

  cont(i, j) = fraccion de los 3-shingles de palabras de i que aparecen en j (Broder).
  Asimetrica a proposito. Implementada en revision_logo.matriz_contencion; se importa, no se
  reimplementa. No usa etiquetas: se calcula una vez sobre el corpus y solo se CONSULTAN las
  columnas del pliegue de entrenamiento, que es lo unico que el sistema vio.

DONDE VA. Despues de la capa de reglas y ANTES del texto:
    IOCs + nombre genuino  ->  CONTENCION >= u  ->  texto
Si varias notas de entrenamiento superan u y apuntan a familias DISTINTAS, la capa NO contesta
y la nota pasa al texto, exactamente como hace la capa de IOCs cuando una clave apunta a mas de
una familia.

BARRIDO: u in {0,5 · 0,6 · 0,7 · 0,8 · 0,9}.

=============================================================================================
EL RIESGO QUE ESTE SCRIPT TIENE QUE CONTROLAR

La contencion alta es JUSTAMENTE lo que el protocolo no separa al partir por plantilla (coseno
0,90). Una capa que se apoya en ella corre riesgo de circularidad: podria estar cobrando de la
limitacion del criterio de agrupamiento en vez de clasificar. Por eso todo se reporta en
columnas separadas -- cobertura y acierto-donde-aplica nunca se mezclan en un solo numero -- y
se corre un control duro de no-circularidad: en las notas donde la capa NO aplica, la cascada
nueva tiene que dar la MISMA prediccion que la vieja, nota por nota. Si no, la capa esta
tocando decisiones que no le corresponden y hay un bug.

=============================================================================================
PREREGISTRO -- escrito y COMMITEADO ANTES de correr (2026-09-28). Vive en el docstring para que
el commit deje la marca temporal verificable en git. Se reporta igual lo que falle y lo que
cumpla.

K1. PUERTA DE ENTRADA. La cascada SIN la capa reproduce la cifra de cabecera: macro-F1 0,7417 y
    exactitud 0,8123 (tolerancia 0,01; fuente 4_resultados/_log_p2bal_149.txt). Si no, ABORTA y
    no se reporta nada.

K2. COBERTURA. La cobertura NETA de la capa (fraccion del corpus que resuelve DENTRO de la
    cascada, o sea con la capa de reglas ya descartada) esta entre 0,10 y 0,30 en u = 0,5, y
    baja de forma monotona al subir u, hasta menos de 0,05 en u = 0,9.
    Base: 63 de 149 notas tienen contencion >= 0,5 con otra plantilla propia y 26 la tienen con
    alguna familia ajena (c_contencion_por_nota.csv, corpus entero); la regla ya resuelve 80,3
    notas de 149 por pliegue (_log_m3_149_p2bal.txt), y esas salen de la cuenta.

K3. ACIERTO DONDE APLICA >= 0,90 en los cinco umbrales, y >= 0,95 en u = 0,9. Razon: de las 63
    notas con contencion propia >= 0,5, en 55 ninguna familia ajena llega a 0,5, y la regla de
    conflicto saca buena parte de las 8 restantes.

K4. LA PREDICCION PRINCIPAL, y la que espero que FALLE. Se declara la direccion de antemano
    para que un resultado nulo no se pueda leer despues como exito: el Delta macro-F1 pareado
    por semilla del MEJOR umbral cae en [-0,005; +0,010] y su IC 95 % INCLUYE el cero.
    Razon: en las notas con hermana contenida >= 0,5 el TEXTO SOLO ya acierta 0,9993
    [0,9982; 1,0000] (_log_similitud_p2bal_149.txt). La capa va a coincidir con el texto casi
    siempre, y una capa que repite la respuesta que ya venia no mueve la metrica.
    Si el IC excluyera el cero POR ARRIBA, la capa aporta de verdad y hay que adoptarla; si lo
    excluyera por abajo, hace dano y se descarta.

K5. APORTE NETO. El acierto del TEXTO SOLO sobre las mismas notas que la capa resuelve es
    >= 0,95 en todos los umbrales. Es la prueba directa de redundancia: si el texto ya acierta
    ahi, la capa no tiene margen. Si diera < 0,80, la capa estaria trabajando en una zona donde
    el texto falla y el aporte seria real.

K6. SOLAPAMIENTO CON LAS CAPAS ANTERIORES. De las notas que la capa resolveria IGNORANDO su
    posicion en la cascada (cobertura BRUTA), al menos el 50 % ya las resuelve la capa de
    reglas en u = 0,5. Base: la regla cubre 80,3/149 = 0,539 del corpus por pliegue y las
    familias de plantillas casi contenidas (CERBER, GANDCRAB, TESLACRYPT) son tambien las de
    marcadores repetidos. Si esa fraccion es alta, la cobertura bruta es enganosa.

K7. CONTROL DE NO-CIRCULARIDAD (duro). En las notas donde la capa NO aplica, la cascada nueva y
    la vieja dan EXACTAMENTE la misma prediccion: cero notas distintas, Delta exactitud
    restringido 0,0000 y Delta macro-F1 restringido 0,0000. Si no da exacto, hay un bug y NO se
    reporta ningun resultado de la corrida.

AGREGADO DESPUES DEL PREREGISTRO (se declara para que el historial de git no enganie). La
corrida de humo de 3 semillas dio Delta EXACTAMENTE 0,0000 en los cinco umbrales, con cero notas
cambiadas. Para EXPLICAR por que, se agrego una columna descriptiva -- «coincide_con_el_texto»:
en que fraccion de sus decisiones la capa contesta lo mismo que el LinearSVC. No es una hipotesis
nueva ni cambia ninguna de K1..K7, que quedan como fueron commiteadas en c7c7228; es el
estadistico que faltaba para no reportar un cero sin causa.

LO QUE ESTE EXPERIMENTO NO DICE. No dice si la contencion «deberia» separarse en la particion:
esa es la limitacion ya declarada del criterio de plantilla (coseno char 0,90), y sigue igual.
Dice si, DADO el protocolo declarado, explotar la contencion de forma directa agrega acierto
por encima de lo que el texto ya saca de ella.
=============================================================================================

Uso:  python capa_contencion.py [--n-semillas 50] [--salida CARPETA]
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
from revision_logo import matriz_contencion, cargar_todo

RAIZ = _AQUI.parent
OUT_DEF = RAIZ / "4_resultados" / "resultados_capa_contencion_149"

# K1. Fuente: 4_resultados/_log_p2bal_149.txt (149 notas, 30 familias, 99 plantillas, 50 semillas)
CANON_F1, CANON_ACC, TOL = 0.7417, 0.8123, 0.01
UMBRALES = (0.5, 0.6, 0.7, 0.8, 0.9)


def ic(v):
    """Media e IC 95 % t-Student. Ignora los nan (semillas sin ninguna nota en el tramo)."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    m, nn = float(np.mean(v)), len(v)
    s = float(np.std(v, ddof=1)) if nn > 1 else 0.0
    h = t_dist.ppf(0.975, nn - 1) * (s / np.sqrt(nn)) if nn > 1 else 0.0
    return m, m - h, m + h


def respuesta_contencion(i, tr, C, y, u):
    """La capa: familias de las notas de ENTRENAMIENTO que contienen a i por encima de u.
    Devuelve (respuesta, n_candidatas, hubo_conflicto). Sin candidatas o con familias en
    conflicto, la respuesta es None y la nota sigue camino al texto."""
    cand = tr[C[i, tr] >= u]
    if len(cand) == 0:
        return None, 0, False
    fams = set(y[cand])
    if len(fams) == 1:
        return next(iter(fams)), len(cand), False
    return None, len(cand), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=50)
    ap.add_argument("--sin-puerta", action="store_true",
                    help="NO usar salvo depuracion: saltea la puerta de entrada K1.")
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)
    NS = args.n_semillas
    NU = len(UMBRALES)

    print("=" * 78)
    print("  CAPA DE CONTENCION: ¿aporta resolver por contencion literal antes del texto?")
    print("=" * 78)
    textos, textos_arr, y, archivos, grupos, iocs, nombres_nota = cargar_todo()
    familias = np.unique(y)
    n = len(y)
    print(f"Notas: {n} | Familias: {len(familias)} | Plantillas: {len(set(grupos))}")
    print(f"Semillas: {NS} | Umbrales: {list(UMBRALES)}")
    print("Cascada nueva: IOCs + nombre  ->  CONTENCION >= u  ->  texto\n")

    print("Contencion por 3-shingles de palabras (revision_logo.matriz_contencion) ...")
    C = matriz_contencion(textos)

    p_base = np.empty((NS, n), dtype=object)       # cascada actual: regla -> texto
    p_txt = np.empty((NS, n), dtype=object)        # texto solo, para medir el aporte
    p_new = np.empty((NU, NS, n), dtype=object)    # cascada con la capa, por umbral
    por_regla = np.zeros((NS, n), bool)
    aplica = np.zeros((NU, NS, n), bool)           # la capa DECIDE (regla no contesto)
    bruta = np.zeros((NU, NS, n), bool)            # la capa contestaria ignorando la posicion
    ok_bruta = np.zeros((NU, NS, n), bool)         # ... y esa respuesta bruta es correcta
    conflicto = np.zeros((NU, NS, n), bool)
    n_cand = np.zeros((NU, NS, n), int)

    print("Evaluando ...")
    for s in range(NS):
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
                p_txt[s, i] = pt[k]
                por_regla[s, i] = r is not None
                p_base[s, i] = pt[k] if r is None else r
                for ui, u in enumerate(UMBRALES):
                    resp, nc, conf = respuesta_contencion(i, tr, C, y, u)
                    n_cand[ui, s, i] = nc
                    conflicto[ui, s, i] = conf
                    bruta[ui, s, i] = resp is not None
                    ok_bruta[ui, s, i] = resp == y[i]
                    if r is not None:                      # manda la capa de reglas
                        p_new[ui, s, i] = r
                    elif resp is not None:                 # decide la capa de contencion
                        p_new[ui, s, i] = resp
                        aplica[ui, s, i] = True
                    else:                                  # sin respuesta: pasa al texto
                        p_new[ui, s, i] = pt[k]
        if (s + 1) % 10 == 0:
            print(f"  {s+1}/{NS} semillas")

    # ---------------- K1: puerta de entrada ----------------
    f1_base = np.array([f1_score(y, p_base[s], average="macro", labels=familias, zero_division=0)
                        for s in range(NS)])
    ac_base = np.array([accuracy_score(y, p_base[s]) for s in range(NS)])
    print("\n" + "-" * 78)
    print("  K1 -- PUERTA DE ENTRADA: la cascada SIN la capa debe dar la cifra de cabecera")
    print("-" * 78)
    print(f"  macro-F1 {f1_base.mean():.4f} vs {CANON_F1} (dif {abs(f1_base.mean()-CANON_F1):.4f}) | "
          f"exactitud {ac_base.mean():.4f} vs {CANON_ACC} (dif {abs(ac_base.mean()-CANON_ACC):.4f})")
    ok_puerta = abs(f1_base.mean() - CANON_F1) <= TOL and abs(ac_base.mean() - CANON_ACC) <= TOL
    if not ok_puerta and not args.sin_puerta:
        sys.exit("ABORTADO (K1): la cascada sin la capa no reproduce la cabecera. No se reporta nada.")
    print("  OK\n" if ok_puerta else "  FUERA DE TOLERANCIA (se sigue por --sin-puerta)\n")

    # ---------------- K7: control de no-circularidad ----------------
    # En las notas donde la capa NO aplica, las dos cascadas tienen que coincidir NOTA A NOTA.
    print("-" * 78)
    print("  K7 -- CONTROL DE NO-CIRCULARIDAD (el que no se puede omitir)")
    print("-" * 78)
    ctrl, max_abs_crudo = [], []
    for ui, u in enumerate(UMBRALES):
        distintas, d_acc, d_f1 = 0, [], []
        for s in range(NS):
            m = ~aplica[ui, s]
            distintas += int((p_new[ui, s, m] != p_base[s, m]).sum())
            d_acc.append(float((p_new[ui, s, m] == y[m]).mean() - (p_base[s, m] == y[m]).mean()))
            d_f1.append(f1_score(y[m], p_new[ui, s, m], average="macro", labels=familias,
                                 zero_division=0)
                        - f1_score(y[m], p_base[s, m], average="macro", labels=familias,
                                   zero_division=0))
        max_abs_crudo.append(max(float(np.max(np.abs(d_f1))), float(np.max(np.abs(d_acc)))))
        ctrl.append(dict(umbral=u, notas_distintas_donde_no_aplica=distintas,
                         delta_exactitud_restringido=round(float(np.mean(d_acc)), 6),
                         delta_macro_f1_restringido=round(float(np.mean(d_f1)), 6),
                         max_abs_delta_por_semilla=f"{max_abs_crudo[-1]:.3e}"))
    dctrl = pd.DataFrame(ctrl)
    dctrl.to_csv(OUT / "contencion_control_circularidad.csv", index=False, encoding="utf-8-sig")
    print(dctrl.to_string(index=False))
    ok_ctrl = bool((dctrl.notas_distintas_donde_no_aplica == 0).all()
                   and max(max_abs_crudo) == 0.0)
    if not ok_ctrl:
        sys.exit("ABORTADO (K7): la capa cambia decisiones donde NO deberia aplicar. Hay un bug; "
                 "no se reporta nada.")
    print("  OK: la capa solo toca las notas que ella misma resuelve.\n")

    # ---------------- barrido de umbral ----------------
    filas, dfilas, ic_f1 = [], [], []
    for ui, u in enumerate(UMBRALES):
        f1_u = np.array([f1_score(y, p_new[ui, s], average="macro", labels=familias,
                                  zero_division=0) for s in range(NS)])
        ac_u = np.array([accuracy_score(y, p_new[ui, s]) for s in range(NS)])
        cob, ac_ap, ac_tx_ap, cob_br, ya_regla, ac_br = [], [], [], [], [], []
        camb, gana, pierde, confl, coinc = [], [], [], [], []
        for s in range(NS):
            a = aplica[ui, s]
            b = bruta[ui, s]
            cob.append(float(a.mean()))
            cob_br.append(float(b.mean()))
            confl.append(float(conflicto[ui, s].mean()))
            ac_ap.append(float((p_new[ui, s, a] == y[a]).mean()) if a.any() else np.nan)
            ac_tx_ap.append(float((p_txt[s, a] == y[a]).mean()) if a.any() else np.nan)
            # descriptivo agregado tras la corrida de humo: ¿la capa dice lo mismo que el texto?
            coinc.append(float((p_new[ui, s, a] == p_txt[s, a]).mean()) if a.any() else np.nan)
            ac_br.append(float(ok_bruta[ui, s, b].mean()) if b.any() else np.nan)
            ya_regla.append(float(por_regla[s, b].mean()) if b.any() else np.nan)
            dif = p_new[ui, s] != p_base[s]
            camb.append(int(dif.sum()))
            gana.append(int((dif & (p_new[ui, s] == y) & (p_base[s] != y)).sum()))
            pierde.append(int((dif & (p_new[ui, s] != y) & (p_base[s] == y)).sum()))
        m_ap, lo_ap, hi_ap = ic(ac_ap)
        m_tx, _, _ = ic(ac_tx_ap)
        d_f1 = f1_u - f1_base
        d_ac = ac_u - ac_base
        mf, lof, hif = ic(d_f1)
        ma, loa, hia = ic(d_ac)
        ic_f1.append((lof, hif))
        filas.append(dict(
            umbral=u,
            cobertura=round(float(np.mean(cob)), 4),
            notas_que_resuelve=round(float(np.mean(cob)) * n, 1),
            acierto_donde_aplica=round(m_ap, 4) if not np.isnan(m_ap) else np.nan,
            ic95_acierto_donde_aplica=f"[{lo_ap:.4f}; {hi_ap:.4f}]" if not np.isnan(m_ap) else "--",
            acierto_del_texto_en_esas_notas=round(m_tx, 4) if not np.isnan(m_tx) else np.nan,
            coincide_con_el_texto=round(float(np.nanmean(coinc)), 4)
            if not np.all(np.isnan(coinc)) else np.nan,
            macro_f1=round(float(f1_u.mean()), 4),
            exactitud=round(float(ac_u.mean()), 4),
            delta_macro_f1=round(mf, 4),
            ic95_delta_f1=f"[{lof:+.4f}; {hif:+.4f}]",
            semillas_f1_positivas=f"{int((d_f1 > 0).sum())}/{NS}",
            delta_exactitud=round(ma, 4),
            ic95_delta_exactitud=f"[{loa:+.4f}; {hia:+.4f}]",
            semillas_exact_positivas=f"{int((d_ac > 0).sum())}/{NS}"))
        dfilas.append(dict(
            umbral=u,
            cobertura_bruta=round(float(np.mean(cob_br)), 4),
            acierto_bruto=round(float(np.nanmean(ac_br)), 4),
            frac_bruta_ya_resuelta_por_reglas=round(float(np.nanmean(ya_regla)), 4),
            cobertura_neta=round(float(np.mean(cob)), 4),
            frac_notas_en_conflicto=round(float(np.mean(confl)), 4),
            candidatas_medias=round(float(n_cand[ui].mean()), 3),
            notas_que_cambian=round(float(np.mean(camb)), 2),
            cambios_que_GANAN=round(float(np.mean(gana)), 2),
            cambios_que_PIERDEN=round(float(np.mean(pierde)), 2),
            bal=round(float(np.mean([balanced_accuracy_score(y, p_new[ui, s])
                                     for s in range(NS)])), 4),
            mcc=round(float(np.mean([matthews_corrcoef(y, p_new[ui, s])
                                     for s in range(NS)])), 4)))
    df = pd.DataFrame(filas)
    dd = pd.DataFrame(dfilas)
    df.to_csv(OUT / "contencion_barrido_umbral.csv", index=False, encoding="utf-8-sig")
    dd.to_csv(OUT / "contencion_aporte.csv", index=False, encoding="utf-8-sig")

    print("=" * 78)
    print(f"  BARRIDO DE UMBRAL (P2bal, {n} notas, 30 familias, {NS} semillas)")
    print("=" * 78)
    print(f"  Cascada SIN la capa (referencia): macro-F1 {f1_base.mean():.4f} | "
          f"exactitud {ac_base.mean():.4f}")
    print("  COBERTURA y ACIERTO-DONDE-APLICA son dos columnas distintas y no se mezclan.\n")
    print(df.to_string(index=False))

    print("\n" + "=" * 78)
    print("  ¿DE DONDE SALE LA COBERTURA? (bruta = ignorando la posicion en la cascada)")
    print("=" * 78)
    print(dd.to_string(index=False))

    # ---------------- por familia, en el mejor umbral ----------------
    mejor = int(np.argmax(df.delta_macro_f1.values))
    u_mejor = UMBRALES[mejor]
    ffilas = []
    for ui, u in enumerate(UMBRALES):
        for f in familias:
            idx = np.where(y == f)[0]
            ffilas.append(dict(
                umbral=u, familia=f, n_notas=len(idx),
                cobertura_capa=round(float(aplica[ui, :, idx].mean()), 4),
                acierto_base=round(float(np.mean([(p_base[s, idx] == f).mean()
                                                  for s in range(NS)])), 4),
                acierto_con_capa=round(float(np.mean([(p_new[ui, s, idx] == f).mean()
                                                      for s in range(NS)])), 4)))
    dff = pd.DataFrame(ffilas)
    dff["delta"] = (dff.acierto_con_capa - dff.acierto_base).round(4)
    dff.to_csv(OUT / "contencion_por_familia.csv", index=False, encoding="utf-8-sig")
    sub = dff[(dff.umbral == u_mejor) & (dff.delta.abs() > 1e-9)].sort_values("delta",
                                                                             ascending=False)
    print(f"\n=== POR FAMILIA en el umbral con mejor Delta (u = {u_mejor}): las que se mueven ===")
    print(sub.to_string(index=False) if len(sub) else "  ninguna familia cambia de acierto.")

    # ---------------- por nota ----------------
    dfn = pd.DataFrame(dict(
        archivo=archivos, familia=y, grupo=grupos,
        frac_semillas_resuelta_por_regla=np.round(por_regla.mean(0), 4),
        acierto_base=np.round(np.array([[p_base[s, i] == y[i] for i in range(n)]
                                        for s in range(NS)]).mean(0), 4),
        acierto_texto=np.round(np.array([[p_txt[s, i] == y[i] for i in range(n)]
                                         for s in range(NS)]).mean(0), 4)))
    for ui, u in enumerate(UMBRALES):
        dfn[f"aplica_u{u}"] = np.round(aplica[ui].mean(0), 4)
        dfn[f"acierto_u{u}"] = np.round(np.array([[p_new[ui, s, i] == y[i] for i in range(n)]
                                                  for s in range(NS)]).mean(0), 4)
    dfn.to_csv(OUT / "contencion_por_nota.csv", index=False, encoding="utf-8-sig")

    # ---------------- veredicto ----------------
    r5 = filas[0]                                   # u = 0,5
    a5 = dfilas[0]
    mejor_fila = filas[mejor]
    lo_mejor, hi_mejor = ic_f1[mejor]
    cobs = [f["cobertura"] for f in filas]
    monotona = all(cobs[i] >= cobs[i + 1] for i in range(len(cobs) - 1))
    k2 = 0.10 <= r5["cobertura"] <= 0.30 and monotona and cobs[-1] < 0.05
    k3 = all(f["acierto_donde_aplica"] >= 0.90 for f in filas
             if not np.isnan(f["acierto_donde_aplica"])) and \
        (np.isnan(filas[-1]["acierto_donde_aplica"]) or filas[-1]["acierto_donde_aplica"] >= 0.95)
    k4 = (-0.005 <= mejor_fila["delta_macro_f1"] <= 0.010) and lo_mejor <= 0 <= hi_mejor
    k5 = all(f["acierto_del_texto_en_esas_notas"] >= 0.95 for f in filas
             if not np.isnan(f["acierto_del_texto_en_esas_notas"]))
    k6 = a5["frac_bruta_ya_resuelta_por_reglas"] >= 0.50

    print("\n" + "=" * 78)
    print("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    print("=" * 78)
    chk = [
        ("K1 puerta de entrada (cascada sin capa = cabecera)", ok_puerta,
         f"{f1_base.mean():.4f} / {ac_base.mean():.4f}"),
        ("K2 cobertura 0,10-0,30 en u=0,5, monotona, <0,05 en 0,9", k2,
         " ".join(f"{u}:{c:.3f}" for u, c in zip(UMBRALES, cobs))),
        ("K3 acierto donde aplica >= 0,90 (y >= 0,95 en u=0,9)", k3,
         " ".join(f"{f['umbral']}:{f['acierto_donde_aplica']:.3f}" for f in filas
                  if not np.isnan(f["acierto_donde_aplica"]))),
        ("K4 Delta del mejor umbral en [-0,005; +0,010] con IC que INCLUYE 0", k4,
         f"u={u_mejor} D {mejor_fila['delta_macro_f1']:+.4f} {mejor_fila['ic95_delta_f1']}"),
        ("K5 el TEXTO ya acierta >= 0,95 en las mismas notas", k5,
         " ".join(f"{f['umbral']}:{f['acierto_del_texto_en_esas_notas']:.3f}" for f in filas
                  if not np.isnan(f["acierto_del_texto_en_esas_notas"]))),
        ("K6 >= 50 % de la cobertura BRUTA ya la resolvian las reglas", k6,
         f"u=0,5: {a5['frac_bruta_ya_resuelta_por_reglas']:.4f}"),
        ("K7 control de no-circularidad exacto", ok_ctrl,
         f"{int(dctrl.notas_distintas_donde_no_aplica.sum())} notas distintas donde no aplica"),
    ]
    for nombre, cumple, det in chk:
        print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<52} {det}")

    print("\n  DECISION (criterio fijado en K4, antes de correr):")
    if lo_mejor > 0:
        print(f"    SE ADOPTA la capa con umbral {u_mejor}: Delta macro-F1 "
              f"{mejor_fila['delta_macro_f1']:+.4f} {mejor_fila['ic95_delta_f1']}, "
              f"{mejor_fila['semillas_f1_positivas']} semillas. El IC excluye el cero por arriba.")
    elif hi_mejor < 0:
        print(f"    SE DESCARTA: la capa HACE DANO en todos los umbrales (mejor {u_mejor}: "
              f"{mejor_fila['delta_macro_f1']:+.4f} {mejor_fila['ic95_delta_f1']}).")
    else:
        print(f"    NO SE ADOPTA: el mejor umbral ({u_mejor}) da Delta "
              f"{mejor_fila['delta_macro_f1']:+.4f} {mejor_fila['ic95_delta_f1']}, IC que incluye")
        print(f"    el cero. La capa resuelve el {mejor_fila['cobertura']*100:.1f} % del corpus "
              f"con acierto {mejor_fila['acierto_donde_aplica']:.4f} donde aplica, pero el texto")
        print(f"    solo ya acertaba {mejor_fila['acierto_del_texto_en_esas_notas']:.4f} en esas "
              f"mismas notas y la capa le contesta lo mismo en el "
              f"{mejor_fila['coincide_con_el_texto']*100:.2f} % de sus")
        print("    decisiones: la senal ya estaba explotada por el TF-IDF, la capa solo la repite.")
    print(f"\n  RECORDAR AL CITAR: cobertura y acierto-donde-aplica son METRICAS DISTINTAS. "
          f"Base: {n} notas, 30 familias, P2bal, {NS} semillas.")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
