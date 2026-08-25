#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cascada_ioc_notas.py -- M.1: cascada IOC -> texto en el frente de notas.
Es el ESPEJO del Experimento 2b (firmas binarias en archivos cifrados) en las notas.

MECANISMO MEDIDO QUE LO SOSTIENE (B.3 sobre 155 notas)
Tras filtrar la infraestructura comun, ningun IOC operativo (email, .onion, BTC, URL propia)
se comparte entre familias distintas: los marcadores son PRIVADOS de cada familia. Por lo
tanto un IOC ya visto identifica la familia casi sin error, pero solo cubre las notas que
reutilizan infraestructura conocida. Igual que una firma binaria: alta precision donde
aplica, cobertura parcial.

LA REGLA, POR PLIEGUE
  1. Se extraen los IOCs SOLO de las notas de ENTRENAMIENTO (nunca del pliegue de prueba)
     con los patrones canonicos de normalizacion_marcadores.PATRONES -- los mismos que ya
     estan declarados en la tesis para el Sprint 1.1 y para el grafo B.3. No se inventa un
     criterio nuevo para este experimento.
  2. Se arma el diccionario valor_de_IOC -> familias que lo usan en entrenamiento.
  3. Para cada nota de PRUEBA: si contiene algun IOC del diccionario y TODAS sus
     coincidencias apuntan a UNA sola familia, se le asigna esa familia. Si hay conflicto
     (dos familias) o no hay coincidencia, cae al clasificador de texto entrenado en ESE
     MISMO pliegue.

DOS VARIANTES (las mismas de B.3), se reportan por separado
  sin_circularidad : el diccionario usa todos los valores.
  con_circularidad : se excluyen del diccionario los valores que contienen el nombre de la
     familia o un alias conocido (criterio: la feature sobreviviria si la familia se
     cambiara el nombre manana?). El filtro se aplica al CONSTRUIR el diccionario, que es
     donde la etiqueta se conoce legitimamente; en prueba solo se consulta. Asi el criterio
     nunca usa la etiqueta de la nota evaluada.

PROTOCOLO -- identico a la base, sin ninguna variacion
  P2 (protocolo "grupos"): StratifiedGroupKFold de N_FOLDS=2 sobre los grupos de
  casi-duplicados, las MISMAS 10 semillas, corpus de 155 notas / 106 plantillas / 30
  familias, y LinearSVC(C=1, class_weight=balanced) sobre la vista combinada como capa de
  respaldo. La capa de texto se entrena UNA sola vez por (semilla, pliegue) y se comparte
  entre las dos variantes: la base y las variantes salen de la MISMA particion y de las
  MISMAS predicciones de texto, de modo que el Delta pareado por semilla es exacto.

PUERTA DE ENTRADA (el script aborta si no se cumple)
  La capa de texto sola debe reproducir la base declarada. Por defecto esa base es la de
  155 notas: macro-F1 0,5265 +/- 0,0490 (LinearSVC, combinado, P2;
  4_resultados/resultados_extension_155/resultados_canonicos/corrida_canonica_resumen.csv).
  Con OTRO corpus en disco hay que pasar la base de ESE corpus con --base-desde, apuntando
  a su corrida_canonica_resumen.csv (de ahi se lee la fila grupos/combinado/LinearSVC), o
  declararla a mano con --base. Si no, la puerta aborta y no se reporta ninguna cifra: eso
  es lo correcto, porque una base de otro corpus invalida el Delta pareado.

SE REPORTA CON LAS TRES COLUMNAS DEL EXP. 2b
  cobertura de la regla | acierto donde aplica | macro-F1 y exactitud del combinado,
  media +/- desvio sobre 10 semillas y Delta pareado contra la base con IC 95 %.

Predicciones preregistradas ANTES de correr: ESTADO_TESIS.md, bloque
"PREDICCIONES PREREGISTRADAS -- M.1 CASCADA IOC->TEXTO (2026-08-22)".

Uso:
    python cascada_ioc_notas.py
    python cascada_ioc_notas.py --salida <carpeta>
    python cascada_ioc_notas.py --salida ../4_resultados/resultados_cascada_149         --base-desde ../4_resultados/resultados_notas_149/corrida_canonica_resumen.csv
Salida por defecto: 4_resultados/resultados_cascada_155/ (carpeta NUEVA; no toca
resultados_canonicos/ ni resultados_extension_155/). Con otro corpus, carpeta nueva y
--base-desde del mismo corpus: la base y el corpus tienen que ser de la misma base.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import t as t_dist
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, N_FOLDS, N_SEMILLAS, UMBRAL_NEARDUP,
                                   agrupar_neardups, cargar_corpus, obtener_modelos,
                                   vectorizador)
from grafo_marcadores import ALIAS, _terminos_circulares, extraer_marcadores
from normalizacion_marcadores import PATRONES

OUT_DIR_DEFAULT = (_AQUI.parent / "4_resultados" / "resultados_cascada_155"
                   if (_AQUI.parent / "4_resultados").is_dir()
                   else _AQUI / "resultados_cascada_155")

# Base declarada a reproducir por la capa de texto (puerta de entrada).
# Estos son los valores POR DEFECTO, de la base de 155 notas. Para medir sobre otro corpus
# hay que apuntar la base al resumen canonico de ESE corpus (--base-desde) o declararla a
# mano (--base / --base-std): con la base de 155 y el corpus de 149 en disco, la puerta
# aborta. Antecedente: techo_por_familia.py leia el F1 desde una ruta fija y mezclo datos
# de dos corpus distintos sin avisar.
BASE_MACRO_F1 = 0.5265
BASE_MACRO_F1_STD = 0.0490
BASE_FUENTE = ("resultados_extension_155/resultados_canonicos/"
               "corrida_canonica_resumen.csv")
TOL_BASE = 0.003

# Las dos variantes PREREGISTRADAS (con y sin filtro de circularidad, como en B.3).
VARIANTES = ("sin_circularidad", "con_circularidad")

# Variantes de DIAGNOSTICO POST-HOC, decididas DESPUES de ver los resultados y por eso
# etiquetadas como tales: NO se les aplica el criterio de adopcion preregistrado y NO son
# candidatas a adoptarse. Existen para una sola pregunta: cuanta cobertura bloquean las
# URLs de infraestructura comun (torproject) al contaminar la regla de unanimidad.
VARIANTES_POSTHOC = ("POSTHOC_sin_circ_solo_privados", "POSTHOC_con_circ_solo_privados")
CFG_VARIANTE = {
    "sin_circularidad": dict(excluir_circulares=False, solo_privados=False),
    "con_circularidad": dict(excluir_circulares=True, solo_privados=False),
    "POSTHOC_sin_circ_solo_privados": dict(excluir_circulares=False, solo_privados=True),
    "POSTHOC_con_circ_solo_privados": dict(excluir_circulares=True, solo_privados=True),
}

# Familias preregistradas: las de mayor continuidad de IOCs (item c del preregistro) y los
# dos controles negativos, que tienen cobertura 0,000 en el grafo y NO deben moverse.
FAM_PREREG = ["BLACKBASTA", "CLOP", "BLACKMATTER", "SODINOKIBI", "LORENZ", "NOTPETYA",
              "PHOBOS", "GANDCRAB", "NETWALKER", "TESLACRYPT"]
FAM_CONTROL = ["RYUK", "HELLOKITTY"]


def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"no serializable: {type(o)}")


def ic95_media(delta):
    """IC 95 % de la media de las diferencias pareadas (t de Student, df = n-1)."""
    n = len(delta)
    m = float(np.mean(delta))
    s = float(np.std(delta, ddof=1)) if n > 1 else 0.0
    se = s / np.sqrt(n) if n > 1 else 0.0
    h = t_dist.ppf(0.975, n - 1) * se if n > 1 else 0.0
    return m, s, m - h, m + h


def construir_diccionario(indices_train, iocs_por_nota, y, excluir_circulares,
                          solo_privados=False):
    """Diccionario valor_de_IOC -> familias, construido SOLO con el pliegue de entrenamiento.

    Con `excluir_circulares`, se descarta todo valor que contenga el nombre de la familia
    (o un alias) de la nota que lo aporta. Es el mismo criterio de B.3 y se aplica aca,
    donde la etiqueta se conoce legitimamente; en prueba solo se consulta el diccionario.

    Con `solo_privados` (variante de DIAGNOSTICO POST-HOC, no preregistrada) se descartan
    ademas los valores que en el entrenamiento aparecen en mas de una familia. Es el filtro
    de genericos de B.3 (MAX_FAMILIAS_VALOR) y sirve para separar dos cosas que la regla
    de unanimidad confunde: una nota con un IOC privado correcto Y una URL de Tor compartida
    queda bloqueada entera, aunque el IOC privado la identificaba sin ambiguedad.
    """
    dicc = defaultdict(set)
    n_excluidos = 0
    for i in indices_train:
        fam = y[i]
        terminos = _terminos_circulares(fam) if excluir_circulares else ()
        for tipo, val in iocs_por_nota[i]:
            if terminos and any(t in val for t in terminos):
                n_excluidos += 1
                continue
            dicc[(tipo, val)].add(fam)
    if solo_privados:
        ambiguos = [k for k, v in dicc.items() if len(v) > 1]
        for k in ambiguos:
            del dicc[k]
        n_excluidos += len(ambiguos)
    return dicc, n_excluidos


def aplicar_regla(i, dicc, iocs_por_nota):
    """Devuelve (familia_asignada, estado, n_coincidencias).

    estado: 'asignada' | 'conflicto' | 'sin_coincidencia'.
    """
    familias = set()
    n_coincidencias = 0
    for clave in iocs_por_nota[i]:
        if clave in dicc:
            familias |= dicc[clave]
            n_coincidencias += 1
    if not familias:
        return None, "sin_coincidencia", 0
    if len(familias) > 1:
        return None, "conflicto", n_coincidencias
    return next(iter(familias)), "asignada", n_coincidencias


def base_desde_csv(ruta: Path):
    """(media, desvio) del macro-F1 de la fila canonica P2 de un corrida_canonica_resumen.csv.

    La fila es protocolo=grupos, vista=combinado, modelo=LinearSVC: exactamente la
    configuracion de la capa de texto de esta cascada. Se lee del CSV en vez de escribirla
    a mano para que la base y el corpus medido no puedan quedar de bases distintas.
    """
    df = pd.read_csv(ruta)
    fila = df[(df["protocolo"] == "grupos") & (df["vista"] == "combinado")
              & (df["modelo"] == "LinearSVC")]
    if fila.empty:
        sys.exit(f"ABORTA: no hay fila grupos/combinado/LinearSVC en {ruta}")
    return float(fila["f1_macro_mean"].iloc[0]), float(fila["f1_macro_std"].iloc[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--base-desde", type=Path, default=None,
                    help="corrida_canonica_resumen.csv del MISMO corpus que se esta "
                         "midiendo, de donde leer la base a reproducir (fila "
                         "grupos/combinado/LinearSVC). Sin esto se usa la base de 155.")
    ap.add_argument("--base", type=float, default=None,
                    help="base macro-F1 declarada a mano; prioridad sobre --base-desde")
    ap.add_argument("--base-std", type=float, default=None,
                    help="desvio de la base declarada a mano (solo informativo)")
    ap.add_argument("--tol", type=float, default=TOL_BASE,
                    help="tolerancia de la puerta de entrada (por defecto 0.003)")
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    base, base_std, base_fuente = BASE_MACRO_F1, BASE_MACRO_F1_STD, BASE_FUENTE
    if args.base_desde is not None:
        base, base_std = base_desde_csv(args.base_desde)
        base_fuente = str(args.base_desde)
    if args.base is not None:
        base = args.base
        base_std = args.base_std if args.base_std is not None else float("nan")
        base_fuente = "declarada a mano en la linea de comandos (--base)"
    tol = args.tol

    print("=" * 78)
    print("  M.1 -- CASCADA IOC -> TEXTO (espejo del Exp. 2b en el frente de notas)")
    print("=" * 78)
    textos, y, archivos, _ = cargar_corpus(CORPUS_DIR)
    grupos, pares = agrupar_neardups(textos, UMBRAL_NEARDUP)
    familias = np.unique(y)
    textos_arr = np.array(textos, dtype=object)
    n = len(textos)
    print(f"Corpus: {CORPUS_DIR}")
    print(f"Notas: {n} | Familias: {len(familias)} | Plantillas: {len(set(grupos))}")
    print(f"Base a batir (texto solo, P2 grupos+combinado+LinearSVC): "
          f"macro-F1 {base:.4f} +/- {base_std:.4f} | tol {tol}")
    print(f"  fuente de la base: {base_fuente}")

    # ---- IOCs por nota, una sola vez (los patrones son deterministas y no usan etiquetas).
    # Que se extraigan de todo el corpus NO es fuga: el diccionario se arma solo con train.
    iocs_por_nota = [set(extraer_marcadores(t)) for t in textos]
    tot_tipo = Counter(tipo for s in iocs_por_nota for tipo, _ in s)
    sin_ioc = sum(1 for s in iocs_por_nota if not s)
    print("IOCs por tipo (corpus completo): " +
          " | ".join(f"{k.strip('[]')} {v}" for k, v in sorted(tot_tipo.items())))
    print(f"Notas sin ningun IOC: {sin_ioc} de {n}")

    # ---- Evaluacion: una pasada por semilla; la capa de texto se comparte entre variantes
    idx_fam = {f: i for i, f in enumerate(familias)}
    pred_texto_sem = []                      # [semilla] -> y_pred del texto solo
    todas_var = VARIANTES + VARIANTES_POSTHOC
    pred_casc_sem = {v: [] for v in todas_var}
    aplica_sem = {v: [] for v in todas_var}  # mascara booleana de "la regla aplico"
    diag = []                                # detalle nota por nota
    filas_pliegue = []

    print(f"\nEvaluando P2 (grupos, StratifiedGroupKFold {N_FOLDS} pliegues, "
          f"{N_SEMILLAS} semillas) ...")
    for seed in range(N_SEMILLAS):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        y_texto = np.empty_like(y)
        y_casc = {v: np.empty_like(y) for v in todas_var}
        aplica = {v: np.zeros(n, dtype=bool) for v in todas_var}

        for n_fold, (tr, te) in enumerate(cv.split(textos_arr, y, groups=grupos)):
            # --- capa de texto: identica a la corrida canonica (TF-IDF dentro del pliegue)
            vec = vectorizador("combinado")
            Xtr = vec.fit_transform(textos_arr[tr])
            Xte = vec.transform(textos_arr[te])
            clf = obtener_modelos(seed)["LinearSVC"]
            clf.fit(Xtr, y[tr])
            pred_t = clf.predict(Xte)
            y_texto[te] = pred_t

            # --- capa de IOCs, por variante
            for var in todas_var:
                dicc, n_exc = construir_diccionario(tr, iocs_por_nota, y,
                                                    **CFG_VARIANTE[var])
                pred_v = pred_t.copy()
                n_asig = n_conf = n_sin = 0
                for k, i in enumerate(te):
                    fam_ioc, estado, n_coin = aplicar_regla(i, dicc, iocs_por_nota)
                    if estado == "asignada":
                        pred_v[k] = fam_ioc
                        aplica[var][i] = True
                        n_asig += 1
                    elif estado == "conflicto":
                        n_conf += 1
                    else:
                        n_sin += 1
                    diag.append(dict(
                        variante=var, semilla=seed, pliegue=n_fold, archivo=archivos[i],
                        familia_real=y[i], n_iocs_nota=len(iocs_por_nota[i]),
                        estado=estado, n_coincidencias=n_coin,
                        familia_regla=fam_ioc if fam_ioc else "",
                        pred_texto=pred_t[k], pred_cascada=pred_v[k],
                        acierta_texto=bool(pred_t[k] == y[i]),
                        acierta_cascada=bool(pred_v[k] == y[i])))
                y_casc[var][te] = pred_v
                filas_pliegue.append(dict(
                    variante=var, semilla=seed, pliegue=n_fold, n_train=len(tr),
                    n_test=len(te), valores_en_diccionario=len(dicc),
                    valores_excluidos_por_circularidad=n_exc,
                    n_asignadas=n_asig, n_conflicto=n_conf, n_sin_coincidencia=n_sin))

        pred_texto_sem.append(y_texto)
        for var in todas_var:
            pred_casc_sem[var].append(y_casc[var])
            aplica_sem[var].append(aplica[var])
        print(f"  semilla {seed}: texto macro-F1 "
              f"{f1_score(y, y_texto, average='macro', zero_division=0):.4f} | " +
              " | ".join(
                  f"{v.split('_')[0]}+IOC "
                  f"{f1_score(y, y_casc[v], average='macro', zero_division=0):.4f} "
                  f"(cob {aplica[v].mean():.3f})" for v in VARIANTES))

    # ---- PUERTA DE ENTRADA: el texto solo debe reproducir la base declarada
    f1_texto = np.array([f1_score(y, p, average="macro", zero_division=0)
                         for p in pred_texto_sem])
    dif_base = abs(f1_texto.mean() - base)
    print("\n" + "-" * 78)
    print(f"PUERTA DE ENTRADA: capa de texto sola {f1_texto.mean():.4f} +/- "
          f"{f1_texto.std(ddof=1):.4f} vs base declarada {base:.4f} +/- "
          f"{base_std:.4f} | dif {dif_base:.4f}")
    if dif_base > tol:
        sys.exit(f"ABORTA: la capa de texto NO reproduce la base declarada "
                 f"(dif {dif_base:.4f} > tol {tol}). No se reporta ninguna cifra: "
                 f"si la particion no es la misma, el Delta pareado no es valido. "
                 f"Base usada: {base:.4f} ({base_fuente}). Si el corpus cambio, pasar "
                 f"--base-desde con el corrida_canonica_resumen.csv de ESTE corpus.")
    print("  OK: misma particion que la base; el Delta pareado por semilla es valido.")

    def metricas(preds):
        return (np.array([f1_score(y, p, average="macro", zero_division=0) for p in preds]),
                np.array([accuracy_score(y, p) for p in preds]),
                np.array([balanced_accuracy_score(y, p) for p in preds]),
                np.array([precision_recall_fscore_support(
                    y, p, labels=familias, zero_division=0)[2] for p in preds]))

    f1_b, acc_b, bal_b, fam_b = metricas(pred_texto_sem)

    # ---- Las TRES columnas del Exp. 2b, por variante
    filas_res, filas_fam, filas_sem = [], [], []
    for s_i in range(N_SEMILLAS):
        filas_sem.append(dict(variante="texto_solo (base)", semilla=s_i,
                              cobertura=0.0, acierto_donde_aplica=float("nan"),
                              f1_macro=round(float(f1_b[s_i]), 6),
                              accuracy=round(float(acc_b[s_i]), 6),
                              balanced_accuracy=round(float(bal_b[s_i]), 6)))

    for var in todas_var:
        f1_v, acc_v, bal_v, fam_v = metricas(pred_casc_sem[var])
        cob, ac_ap, ac_texto_ap, mejora_ap = [], [], [], []
        for s_i in range(N_SEMILLAS):
            m = aplica_sem[var][s_i]
            p_v, p_t = pred_casc_sem[var][s_i], pred_texto_sem[s_i]
            cob.append(m.mean())
            ac_ap.append(float((p_v[m] == y[m]).mean()) if m.any() else float("nan"))
            ac_texto_ap.append(float((p_t[m] == y[m]).mean()) if m.any() else float("nan"))
            mejora_ap.append(ac_ap[-1] - ac_texto_ap[-1])
            filas_sem.append(dict(variante=var, semilla=s_i,
                                  cobertura=round(float(cob[-1]), 6),
                                  acierto_donde_aplica=round(float(ac_ap[-1]), 6),
                                  f1_macro=round(float(f1_v[s_i]), 6),
                                  accuracy=round(float(acc_v[s_i]), 6),
                                  balanced_accuracy=round(float(bal_v[s_i]), 6)))
        cob, ac_ap = np.array(cob), np.array(ac_ap)
        d_f1, s_f1, lo_f1, hi_f1 = ic95_media(f1_v - f1_b)
        d_ac, s_ac, lo_ac, hi_ac = ic95_media(acc_v - acc_b)
        filas_res.append(dict(
            variante=var,
            cobertura_mean=round(float(cob.mean()), 4),
            cobertura_std=round(float(cob.std(ddof=1)), 4),
            notas_cubiertas_mean=round(float(cob.mean() * n), 1),
            acierto_donde_aplica_mean=round(float(np.nanmean(ac_ap)), 4),
            acierto_donde_aplica_std=round(float(np.nanstd(ac_ap, ddof=1)), 4),
            acierto_texto_en_cubiertas_mean=round(float(np.nanmean(ac_texto_ap)), 4),
            mejora_en_cubiertas_mean=round(float(np.nanmean(mejora_ap)), 4),
            f1_macro_base=round(float(f1_b.mean()), 4),
            f1_macro_base_std=round(float(f1_b.std(ddof=1)), 4),
            f1_macro_cascada=round(float(f1_v.mean()), 4),
            f1_macro_cascada_std=round(float(f1_v.std(ddof=1)), 4),
            delta_f1_macro=round(d_f1, 4), delta_f1_std=round(s_f1, 4),
            ic95_f1_inf=round(lo_f1, 4), ic95_f1_sup=round(hi_f1, 4),
            semillas_delta_f1_pos=int(np.sum(f1_v - f1_b > 0)),
            accuracy_base=round(float(acc_b.mean()), 4),
            accuracy_cascada=round(float(acc_v.mean()), 4),
            delta_accuracy=round(d_ac, 4),
            ic95_acc_inf=round(lo_ac, 4), ic95_acc_sup=round(hi_ac, 4),
            balanced_accuracy_base=round(float(bal_b.mean()), 4),
            balanced_accuracy_cascada=round(float(bal_v.mean()), 4),
            adopta=bool(lo_f1 > 0)))

        for fam in familias:
            j = idx_fam[fam]
            b, v = fam_b[:, j], fam_v[:, j]
            m, s, lo, hi = ic95_media(v - b)
            # El Delta de F1 de una familia mezcla DOS efectos y hay que poder separarlos:
            #  - recall propio: notas de la familia que la regla asigno (la toco de verdad);
            #  - precision ajena: notas de OTRAS familias que el texto le atribuia por error
            #    y la regla reasigno. Una familia con cero notas propias asignadas puede
            #    subir su F1 solo por esta segunda via. Sin esta descomposicion, un control
            #    negativo parece moverse cuando en realidad la regla nunca lo toco.
            propias = int(sum(
                np.sum(aplica_sem[var][s_i] & (y == fam)) for s_i in range(N_SEMILLAS)))
            fp_corr = int(sum(
                np.sum((pred_texto_sem[s_i] == fam) & (y != fam) &
                       (pred_casc_sem[var][s_i] != fam)) for s_i in range(N_SEMILLAS)))
            filas_fam.append(dict(
                variante=var, familia=fam,
                prereg=fam in FAM_PREREG, control_negativo=fam in FAM_CONTROL,
                notas_propias_asignadas=propias,
                falsos_positivos_ajenos_corregidos=fp_corr,
                tocada_por_la_regla=bool(propias > 0),
                f1_base_mean=round(float(b.mean()), 4),
                f1_base_std=round(float(b.std(ddof=1)), 4),
                f1_cascada_mean=round(float(v.mean()), 4),
                f1_cascada_std=round(float(v.std(ddof=1)), 4),
                delta_mean=round(m, 4), delta_std=round(s, 4),
                ic95_inf=round(lo, 4), ic95_sup=round(hi, 4),
                semillas_delta_pos=int(np.sum(v - b > 0))))

    df_res = pd.DataFrame(filas_res)
    df_fam = pd.DataFrame(filas_fam)
    df_res.to_csv(OUT / "cascada_resumen.csv", index=False)
    df_fam.to_csv(OUT / "cascada_por_familia.csv", index=False)
    pd.DataFrame(filas_sem).to_csv(OUT / "cascada_por_semilla.csv", index=False)
    pd.DataFrame(filas_pliegue).to_csv(OUT / "cascada_por_pliegue.csv", index=False)
    pd.DataFrame(diag).to_csv(OUT / "cascada_detalle_notas.csv", index=False)

    # ---- Reporte: las tres columnas del Exp. 2b
    print("\n" + "=" * 78)
    print("  LAS TRES COLUMNAS DEL EXP. 2b (media +/- desvio, 10 semillas, base 155)")
    print("=" * 78)
    print(f"  {'variante':18} {'cobertura':>16} {'acierto d/aplica':>18} "
          f"{'macro-F1 combinado':>22} {'exactitud':>10}")
    print(f"  {'texto solo (base)':18} {'0,000':>16} {'--':>18} "
          f"{f1_b.mean():>13.4f}+/-{f1_b.std(ddof=1):.4f} {acc_b.mean():>10.4f}")
    for r in [x for x in filas_res if x["variante"] in VARIANTES]:
        print(f"  {r['variante']:18} "
              f"{r['cobertura_mean']:>9.4f}+/-{r['cobertura_std']:.4f} "
              f"{r['acierto_donde_aplica_mean']:>11.4f}+/-{r['acierto_donde_aplica_std']:.4f} "
              f"{r['f1_macro_cascada']:>13.4f}+/-{r['f1_macro_cascada_std']:.4f} "
              f"{r['accuracy_cascada']:>10.4f}")
    print("\n  Delta pareado por semilla contra la base (texto solo):")
    for r in [x for x in filas_res if x["variante"] in VARIANTES]:
        print(f"    {r['variante']:18} macro-F1 {r['delta_f1_macro']:+.4f} "
              f"IC95 [{r['ic95_f1_inf']:+.4f}; {r['ic95_f1_sup']:+.4f}] "
              f"| sem+ {r['semillas_delta_f1_pos']}/10 "
              f"| exactitud {r['delta_accuracy']:+.4f} "
              f"IC95 [{r['ic95_acc_inf']:+.4f}; {r['ic95_acc_sup']:+.4f}]")
    print("\n  Diagnostico donde la regla aplica (acierto de la regla vs del texto ahi):")
    for r in [x for x in filas_res if x["variante"] in VARIANTES]:
        print(f"    {r['variante']:18} regla {r['acierto_donde_aplica_mean']:.4f} vs "
              f"texto {r['acierto_texto_en_cubiertas_mean']:.4f} "
              f"=> {r['mejora_en_cubiertas_mean']:+.4f}")

    # ---- Diagnostico POST-HOC: cuanta cobertura bloquean las URLs de Tor
    print("\n" + "-" * 78)
    print("  DIAGNOSTICO POST-HOC -- NO PREREGISTRADO, NO ADOPTABLE")
    print("  Misma cascada, descartando del diccionario los valores que en el entrenamiento")
    print("  aparecen en mas de una familia (filtro de genericos de B.3). Contesta una sola")
    print("  pregunta: cuanta cobertura pierde la regla de unanimidad por las URLs de Tor.")
    print("-" * 78)
    for r in [x for x in filas_res if x["variante"] in VARIANTES_POSTHOC]:
        print(f"  {r['variante']:32} cobertura {r['cobertura_mean']:.4f} | "
              f"acierto d/aplica {r['acierto_donde_aplica_mean']:.4f} | "
              f"macro-F1 {r['f1_macro_cascada']:.4f} "
              f"(Delta {r['delta_f1_macro']:+.4f}, IC95 [{r['ic95_f1_inf']:+.4f}; "
              f"{r['ic95_f1_sup']:+.4f}])")

    # ---- F1 por familia: preregistradas + controles
    print("\n" + "=" * 78)
    print("  F1 POR FAMILIA -- preregistradas y controles negativos "
          "(Delta pareado, IC 95 %)")
    print("=" * 78)
    for var in VARIANTES:
        sub = df_fam[df_fam.variante == var].set_index("familia")
        print(f"\n  [{var}]")
        print(f"    {'familia':13} {'base':>15} {'cascada':>15} {'Delta':>9} "
              f"{'IC95':>20} {'sem+':>6}")
        for grupo, etq in ((FAM_PREREG, "preregistradas"),
                           (FAM_CONTROL, "CONTROLES NEGATIVOS (no deben moverse)")):
            print(f"    -- {etq}")
            for fam in grupo:
                r = sub.loc[fam]
                print(f"    {fam:13} {r.f1_base_mean:6.3f}+/-{r.f1_base_std:5.3f} "
                      f"{r.f1_cascada_mean:6.3f}+/-{r.f1_cascada_std:5.3f} "
                      f"{r.delta_mean:+8.3f} [{r.ic95_inf:+.3f};{r.ic95_sup:+.3f}] "
                      f"{int(r.semillas_delta_pos):>3}/10")

    # ---- Veredicto contra el criterio de adopcion preregistrado
    print("\n" + "=" * 78)
    print("  VEREDICTO CONTRA EL CRITERIO DE ADOPCION PREREGISTRADO")
    print("  (adoptar si el macro-F1 combinado supera la base con IC 95 % del Delta "
          "pareado que EXCLUYE el cero)")
    print("=" * 78)
    veredictos = {}
    for r in [x for x in filas_res if x["variante"] in VARIANTES]:
        var = r["variante"]
        sub = df_fam[df_fam.variante == var].set_index("familia")
        controles = {f: float(sub.loc[f].delta_mean) for f in FAM_CONTROL}
        # Un control negativo solo queda VIOLADO si la regla asigno notas PROPIAS de esa
        # familia. Si su F1 se mueve sin que la regla la haya tocado, el movimiento viene de
        # falsos positivos ajenos corregidos (efecto de precision), y eso NO contradice el
        # mecanismo: la regla no invento senal donde no hay IOCs.
        controles_tocados = {f: bool(sub.loc[f].tocada_por_la_regla) for f in FAM_CONTROL}
        prereg_suben = {f: float(sub.loc[f].delta_mean) for f in FAM_PREREG
                        if float(sub.loc[f].delta_mean) > 0}
        adopta = r["ic95_f1_inf"] > 0
        control_quieto = not any(controles_tocados.values())
        if adopta and control_quieto:
            vd = ("ADOPTAR: el macro-F1 sube con IC 95 % que excluye el cero y la regla NO "
                  "toco ninguna nota de los controles negativos => el mecanismo de IOCs "
                  "queda probado")
        elif adopta and not control_quieto:
            vd = ("ADOPTAR CON RESERVA: el macro-F1 sube con IC 95 % que excluye el cero, "
                  "pero la regla asigno notas de los controles negativos => revisar si la "
                  "mejora es generica")
        elif r["delta_f1_macro"] > 0:
            vd = ("NO ADOPTAR por el criterio preregistrado: el Delta es positivo pero el "
                  "IC 95 % incluye el cero => regla de alta precision y baja cobertura que "
                  "no mueve el agregado")
        else:
            vd = "NO ADOPTAR: el Delta pareado del macro-F1 no es positivo"
        veredictos[var] = dict(
            cobertura=r["cobertura_mean"],
            acierto_donde_aplica=r["acierto_donde_aplica_mean"],
            delta_f1_macro=r["delta_f1_macro"],
            ic95=[r["ic95_f1_inf"], r["ic95_f1_sup"]],
            excluye_cero=bool(adopta), controles_negativos=controles,
            controles_tocados_por_la_regla=controles_tocados,
            controles_sin_tocar=bool(control_quieto),
            preregistradas_que_suben=prereg_suben, veredicto=vd)
        print(f"\n  [{var}] cobertura {r['cobertura_mean']:.4f} | acierto donde aplica "
              f"{r['acierto_donde_aplica_mean']:.4f} | macro-F1 {r['f1_macro_cascada']:.4f} "
              f"(Delta {r['delta_f1_macro']:+.4f}, IC95 [{r['ic95_f1_inf']:+.4f}; "
              f"{r['ic95_f1_sup']:+.4f}])")
        for f, d in controles.items():
            etq = ("TOCADA por la regla" if controles_tocados[f]
                   else "no tocada por la regla; movimiento por falsos positivos ajenos")
            print(f"    control negativo {f}: Delta {d:+.4f} ({etq})")
        print(f"    => {vd}")

    # ---- Manifiesto
    (OUT / "manifiesto_cascada.json").write_text(json.dumps(dict(
        fecha=str(date.today()), experimento="M.1 cascada IOC->texto",
        corpus=str(CORPUS_DIR), n_notas=n, n_familias=int(len(familias)),
        n_plantillas=int(len(set(grupos))), pares_neardup=len(pares),
        umbral_neardup=UMBRAL_NEARDUP, n_folds=N_FOLDS, n_semillas=N_SEMILLAS,
        protocolo="grupos (P2), StratifiedGroupKFold",
        modelo_respaldo="LinearSVC(C=1.0, class_weight=balanced) sobre vista combinada",
        patrones_ioc={e: rx.pattern for e, rx in PATRONES},
        alias_circularidad=ALIAS, variantes_preregistradas=list(VARIANTES),
        variantes_posthoc=dict(
            lista=list(VARIANTES_POSTHOC),
            aclaracion=("decididas DESPUES de ver los resultados; NO se les aplica el "
                        "criterio de adopcion y NO son candidatas a adoptarse. Solo "
                        "cuantifican cuanta cobertura bloquean las URLs de Tor")),
        iocs_por_tipo=dict(tot_tipo), notas_sin_ioc=sin_ioc,
        regla=("IOCs extraidos SOLO del pliegue de entrenamiento; en prueba se asigna la "
               "familia si todas las coincidencias apuntan a UNA sola; conflicto o ausencia "
               "de coincidencia caen al clasificador de texto del mismo pliegue"),
        puerta_de_entrada=dict(
            base_declarada=base, base_std=base_std,
            texto_solo_recomputado=round(float(f1_texto.mean()), 4),
            dif=round(float(dif_base), 4), tol=tol, ok=True,
            fuente=base_fuente),
        familias_preregistradas=FAM_PREREG, controles_negativos=FAM_CONTROL,
        criterio_adopcion=("macro-F1 combinado supera la base con Delta pareado cuyo "
                           "IC 95 % excluye el cero"),
        veredictos=veredictos,
        sklearn=sklearn.__version__, scipy=scipy.__version__,
        python=sys.version.split()[0],
    ), indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(f"\nSalidas en: {OUT}")


if __name__ == "__main__":
    main()
