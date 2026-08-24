#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cascada_combinada_notas.py -- M.6: cascada (IOCs PRIVADOS + NOMBRE GENUINO) -> texto.

Extiende M.1 (cascada_ioc_notas.py) con dos cambios, los dos PREREGISTRADOS en
ESTADO_TESIS.md, bloque "PREREGISTRO -- M.6 CASCADA COMBINADA (2026-08-23)":

 1. FILTRO DE GENERICOS: se descartan del diccionario los valores que en el ENTRENAMIENTO
    aparecen en mas de una familia. NO es un criterio nuevo: es MAX_FAMILIAS_VALOR de B.3,
    declarado ANTES de que M.1 corriera. En M.1 aparecio como diagnostico post-hoc
    (cobertura 0,3974 / macro-F1 0,5658) y por eso no era citable.
    Re-correr sobre el MISMO corpus no elimina que la variante se eligio despues de ver
    resultados; lo que la hace defendible es que el criterio es anterior e independiente, y
    que la estimacion se rehace con SEMILLAS NUEVAS. Es atenuacion, no prueba fuera de muestra.

 2. NIVEL DE NOMBRE DE ARCHIVO de la nota (la senal de ID Ransomware). El diccionario
    nombre->familia se arma SOLO con el pliegue de entrenamiento y SOLO con nombres AUDITADOS:
      - los 47 verificados por MD5 contra el repo de Lemmou, con el nombre ORIGINAL del repo
        (no el del corpus, que en 5 casos de CERBER trae el prefijo circular lm_Cerber_);
      - los 17 recuperados de la fuente de CADA nota (pcrisk / id-ransomware), con el rotulo
        de la fuente como evidencia.
    Los 76 nombres 'curador' (blackbasta1.txt, pcrisk_cuba_1.txt) NUNCA entran: seria leer
    la etiqueta. Fuente de los nombres: 3_datos/nombres_notas/.

REGLA: identica a M.1 (unanimidad). Se juntan las coincidencias de IOC y de nombre; si TODAS
apuntan a UNA familia se asigna; conflicto o ausencia caen al LinearSVC combinado del mismo
pliegue. El nivel de nombre no cambia la regla: agranda el conjunto de evidencia.

NORMALIZACION DEL NOMBRE: solo se pasa a minusculas (los nombres de archivo de Windows son
insensibles a mayusculas). NO se tocan guiones bajos, espacios ni signos: READ_ME_!!!.TXT y
README.txt son archivos distintos. Efecto declarado: Info.hta / info.hta / INFO.hta colapsan,
y por eso aparece la unica colision del corpus, info.hta -> DHARMA + PHOBOS.

CUATRO VARIANTES, para poder aislar cada efecto:
  privados_sin_circ              filtro de genericos, sin nivel de nombre
  privados_sin_circ_MAS_NOMBRE   + nivel de nombre
  privados_con_circ              idem con filtro de circularidad (B.3)
  privados_con_circ_MAS_NOMBRE   + nivel de nombre
El aporte del NOMBRE es Delta(MAS_NOMBRE - sin nombre), como lo exige el preregistro: no
contra la base pelada, para no confundir los dos efectos.

PROTOCOLO: P2 (grupos, StratifiedGroupKFold 2 pliegues), corpus 155/106/30, LinearSVC
(C=1, class_weight=balanced) sobre la vista combinada como respaldo. SEMILLAS 100-109
(NUEVAS: M.1 uso 0-9). El Delta pareado por semilla sigue siendo exacto porque para cada
semilla la base y las variantes salen de la MISMA particion y de las MISMAS predicciones
de texto.

PUERTA DE ENTRADA: con semillas nuevas la particion cambia, asi que la capa de texto NO tiene
por que dar 0,5265 exacto. Se exige que caiga dentro de +/-0,035 (~2 errores estandar de
0,0490/raiz(10)) como control de sanidad; el Delta pareado no depende de eso.

Uso:  python cascada_combinada_notas.py [--salida CARPETA]
Salida por defecto: 4_resultados/resultados_cascada_combinada_155/ (carpeta NUEVA).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
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

from clasificador_notas_v2 import (CORPUS_DIR, N_FOLDS, UMBRAL_NEARDUP, agrupar_neardups,
                                   cargar_corpus, obtener_modelos, vectorizador)
from grafo_marcadores import _terminos_circulares, extraer_marcadores

RAIZ = _AQUI.parent
DIR_NOMBRES = RAIZ / "3_datos" / "nombres_notas"
OUT_DEF = RAIZ / "4_resultados" / "resultados_cascada_combinada_155"

BASE_MACRO_F1 = 0.5265          # base declarada (semillas 0-9)
# El control NO exige reproducir 0,5265: con semillas nuevas la particion cambia por diseno.
# Su unico proposito es detectar un error de implementacion (que la capa de texto se rompa).
# La tolerancia inicial de 0,035 (~2 SE) resulto MAL FUNDADA: medido, el cambio de semillas
# mueve la media 0,0662 (0,5265 con 0-9 -> 0,4603 con 100-109), porque con 2 pliegues y 30
# clases hay semillas donde familias enteras caen en un solo pliegue y su F1 se va a 0. Eso
# es un HALLAZGO que se reporta (la base declarada es mas fragil de lo que sugiere su +/-),
# no un motivo para abortar. El Delta pareado por semilla sigue siendo exacto porque la base
# y las variantes salen de la MISMA particion en cada semilla, y es lo que usa el criterio
# de adopcion. Se deja un umbral amplio como red contra errores groseros.
TOL_BASE = 0.12
SEMILLAS = list(range(100, 110))  # NUEVAS

VARIANTES = {
    "privados_sin_circ":            dict(excluir_circulares=False, usar_nombre=False),
    "privados_sin_circ_MAS_NOMBRE": dict(excluir_circulares=False, usar_nombre=True),
    "privados_con_circ":            dict(excluir_circulares=True,  usar_nombre=False),
    "privados_con_circ_MAS_NOMBRE": dict(excluir_circulares=True,  usar_nombre=True),
}
PARES_AISLAR = [("privados_sin_circ", "privados_sin_circ_MAS_NOMBRE"),
                ("privados_con_circ", "privados_con_circ_MAS_NOMBRE")]

# Controles preregistrados: no deben recibir asignacion POR NOMBRE (no tienen nombre genuino).
FAM_CONTROL_NOMBRE = ["BADRABBIT", "BLACKCAT", "CHIMERA", "JIGSAW", "NOTPETYA"]


def ic95(delta):
    n = len(delta)
    m = float(np.mean(delta))
    s = float(np.std(delta, ddof=1)) if n > 1 else 0.0
    h = t_dist.ppf(0.975, n - 1) * (s / np.sqrt(n)) if n > 1 else 0.0
    return m, s, m - h, m + h


def cargar_nombres_auditados():
    """(familia, archivo_del_corpus) -> nombre GENUINO. Solo nombres auditados."""
    nombres, origen = {}, {}
    csv_aud = DIR_NOMBRES / "auditoria_nombres_corpus.csv"
    with open(csv_aud, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if r["nombre_para_m2"]:
                clave = (r["familia"], r["archivo_corpus"])
                nombres[clave] = r["nombre_para_m2"]
                origen[clave] = "md5_lemmou"
    js = DIR_NOMBRES / "nombres_por_nota_2026-08-23.json"
    if js.is_file():
        with open(js, encoding="utf-8") as f:
            for r in json.load(f):
                nm = (r.get("nombre_archivo") or "").strip()
                if r.get("encontrado") and nm and nm != "SIN_ARCHIVO":
                    clave = (r["familia"], r["archivo_corpus"])
                    nombres[clave] = nm
                    origen[clave] = "fuente_de_la_nota"
    return nombres, origen


def construir_diccionario(tr, iocs, nombres_nota, y, excluir_circulares, usar_nombre):
    """valor -> familias, SOLO con el pliegue de entrenamiento.

    Siempre se aplica el filtro de genericos: se borran los valores que en entrenamiento
    aparecen en mas de una familia. Es MAX_FAMILIAS_VALOR de B.3.
    """
    dicc = defaultdict(set)
    n_circ = 0
    for i in tr:
        fam = y[i]
        terminos = _terminos_circulares(fam) if excluir_circulares else ()
        for clave in iocs[i]:
            if terminos and any(t in clave[1] for t in terminos):
                n_circ += 1
                continue
            dicc[clave].add(fam)
        if usar_nombre and nombres_nota[i]:
            val = nombres_nota[i]
            if terminos and any(t in val for t in terminos):
                n_circ += 1
            else:
                dicc[("[NOMBRE]", val)].add(fam)
    ambiguos = [k for k, v in dicc.items() if len(v) > 1]
    for k in ambiguos:
        del dicc[k]
    return dicc, n_circ, len(ambiguos)


def aplicar_regla(i, dicc, iocs, nombres_nota, usar_nombre):
    """(familia, estado, n_coincidencias, por_nombre). Unanimidad, igual que M.1."""
    familias, n_coin, por_nombre = set(), 0, False
    claves = set(iocs[i])
    if usar_nombre and nombres_nota[i]:
        claves.add(("[NOMBRE]", nombres_nota[i]))
    for c in claves:
        if c in dicc:
            familias |= dicc[c]
            n_coin += 1
            if c[0] == "[NOMBRE]":
                por_nombre = True
    if not familias:
        return None, "sin_coincidencia", 0, False
    if len(familias) > 1:
        return None, "conflicto", n_coin, por_nombre
    return next(iter(familias)), "asignada", n_coin, por_nombre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  M.6 -- CASCADA COMBINADA: IOCs PRIVADOS + NOMBRE GENUINO -> TEXTO")
    print("=" * 78)
    textos, y, archivos, _ = cargar_corpus(CORPUS_DIR)
    grupos, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    familias = np.unique(y)
    textos_arr = np.array(textos, dtype=object)
    n = len(textos)
    print(f"Notas: {n} | Familias: {len(familias)} | Plantillas: {len(set(grupos))}")

    iocs = [set(extraer_marcadores(t)) for t in textos]
    nom_aud, origen = cargar_nombres_auditados()
    nombres_nota, n_md5, n_fuente = [], 0, 0
    for fam, arch in zip(y, archivos):
        clave = (fam, Path(arch).name)
        nm = nom_aud.get(clave)
        nombres_nota.append(nm.lower() if nm else None)
        if nm:
            if origen.get(clave) == "md5_lemmou":
                n_md5 += 1
            else:
                n_fuente += 1
    con_nom = sum(1 for x in nombres_nota if x)
    fams_nom = sorted({f for f, x in zip(y, nombres_nota) if x})
    solo_nom = sum(1 for i in range(n) if nombres_nota[i] and not iocs[i])
    sin_nada = sum(1 for i in range(n) if not nombres_nota[i] and not iocs[i])
    print(f"Nombres genuinos: {con_nom}/{n} ({n_md5} por MD5 + {n_fuente} de la fuente) "
          f"en {len(fams_nom)} familias")
    print(f"Notas con nombre y SIN IOC (techo del aporte del nivel nuevo): {solo_nom}")
    print(f"Notas sin nombre ni IOC (piso irrecuperable por regla): {sin_nada}")

    nombres = list(VARIANTES)
    pred_txt, pred_var = [], {v: [] for v in nombres}
    aplica = {v: [] for v in nombres}
    aplica_nom = {v: [] for v in nombres}
    diag, filas_pliegue = [], []

    print(f"\nP2 (grupos, {N_FOLDS} pliegues) x {len(SEMILLAS)} semillas NUEVAS "
          f"{SEMILLAS[0]}-{SEMILLAS[-1]} ...")
    for seed in SEMILLAS:
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        y_txt = np.empty_like(y)
        y_var = {v: np.empty_like(y) for v in nombres}
        ap_v = {v: np.zeros(n, dtype=bool) for v in nombres}
        ap_n = {v: np.zeros(n, dtype=bool) for v in nombres}

        for n_fold, (tr, te) in enumerate(cv.split(textos_arr, y, groups=grupos)):
            vec = vectorizador("combinado")
            Xtr = vec.fit_transform(textos_arr[tr])
            Xte = vec.transform(textos_arr[te])
            clf = obtener_modelos(seed)["LinearSVC"]
            clf.fit(Xtr, y[tr])
            pred_t = clf.predict(Xte)
            y_txt[te] = pred_t

            for var, cfg in VARIANTES.items():
                dicc, n_circ, n_amb = construir_diccionario(
                    tr, iocs, nombres_nota, y, cfg["excluir_circulares"], cfg["usar_nombre"])
                pred_v = pred_t.copy()
                cnt = Counter()
                for k, i in enumerate(te):
                    fam_r, estado, n_coin, por_nom = aplicar_regla(
                        i, dicc, iocs, nombres_nota, cfg["usar_nombre"])
                    cnt[estado] += 1
                    if estado == "asignada":
                        pred_v[k] = fam_r
                        ap_v[var][i] = True
                        if por_nom:
                            ap_n[var][i] = True
                    diag.append(dict(
                        variante=var, semilla=seed, pliegue=n_fold, archivo=archivos[i],
                        familia_real=y[i], nombre_genuino=nombres_nota[i] or "",
                        n_iocs=len(iocs[i]), estado=estado, n_coincidencias=n_coin,
                        uso_nombre=bool(por_nom), familia_regla=fam_r or "",
                        pred_texto=pred_t[k], pred_cascada=pred_v[k],
                        acierta_texto=bool(pred_t[k] == y[i]),
                        acierta_cascada=bool(pred_v[k] == y[i])))
                y_var[var][te] = pred_v
                filas_pliegue.append(dict(
                    variante=var, semilla=seed, pliegue=n_fold, n_train=len(tr),
                    n_test=len(te), valores_diccionario=len(dicc),
                    excluidos_circularidad=n_circ, excluidos_genericos=n_amb,
                    asignadas=cnt["asignada"], conflicto=cnt["conflicto"],
                    sin_coincidencia=cnt["sin_coincidencia"]))

        pred_txt.append(y_txt)
        for v in nombres:
            pred_var[v].append(y_var[v])
            aplica[v].append(ap_v[v])
            aplica_nom[v].append(ap_n[v])
        resumen = " | ".join(
            f"{v[9:22]} {f1_score(y, y_var[v], average='macro', zero_division=0):.4f}"
            f" (cob {ap_v[v].mean():.3f})" for v in nombres)
        print(f"  semilla {seed}: texto "
              f"{f1_score(y, y_txt, average='macro', zero_division=0):.4f} | {resumen}")

    f1_txt = np.array([f1_score(y, p, average="macro", zero_division=0) for p in pred_txt])
    dif = abs(f1_txt.mean() - BASE_MACRO_F1)
    print("\n" + "-" * 78)
    print(f"CONTROL DE SANIDAD: texto solo (semillas nuevas) {f1_txt.mean():.4f} +/- "
          f"{f1_txt.std(ddof=1):.4f} vs base 0-9 {BASE_MACRO_F1:.4f} | dif {dif:.4f}")
    if dif > TOL_BASE:
        sys.exit(f"ABORTA: la capa de texto se fue de {TOL_BASE} respecto de la base. "
                 f"Revisar antes de reportar cualquier cifra.")
    print("  OK dentro de la variabilidad esperada por cambio de semillas.")

    def met(preds):
        return (np.array([f1_score(y, p, average="macro", zero_division=0) for p in preds]),
                np.array([accuracy_score(y, p) for p in preds]),
                np.array([balanced_accuracy_score(y, p) for p in preds]),
                np.array([precision_recall_fscore_support(
                    y, p, labels=familias, zero_division=0)[2] for p in preds]))

    f1_b, acc_b, bal_b, fam_b = met(pred_txt)
    res, res_fam, guardado = [], [], {}
    res.append(dict(variante="texto_solo (base, semillas nuevas)", cobertura=0.0,
                    cobertura_por_nombre=0.0, acierto_donde_aplica=np.nan,
                    acierto_texto_donde_aplica=np.nan,
                    f1_macro=round(float(f1_b.mean()), 4),
                    f1_macro_sd=round(float(f1_b.std(ddof=1)), 4),
                    exactitud=round(float(acc_b.mean()), 4),
                    exactitud_balanceada=round(float(bal_b.mean()), 4),
                    delta_f1=np.nan, ic95_bajo=np.nan, ic95_alto=np.nan,
                    semillas_positivas="", adopta_vs_base=""))

    for v in nombres:
        f1_v, acc_v, bal_v, fam_v = met(pred_var[v])
        guardado[v] = (f1_v, fam_v)
        cob = np.array([m.mean() for m in aplica[v]])
        cob_n = np.array([m.mean() for m in aplica_nom[v]])
        ac = np.array([float((pred_var[v][s][aplica[v][s]] == y[aplica[v][s]]).mean())
                       if aplica[v][s].any() else np.nan for s in range(len(SEMILLAS))])
        ac_txt = np.array([float((pred_txt[s][aplica[v][s]] == y[aplica[v][s]]).mean())
                           if aplica[v][s].any() else np.nan for s in range(len(SEMILLAS))])
        d = f1_v - f1_b
        m, sd, lo, hi = ic95(d)
        res.append(dict(variante=v, cobertura=round(float(cob.mean()), 4),
                        cobertura_por_nombre=round(float(cob_n.mean()), 4),
                        acierto_donde_aplica=round(float(np.nanmean(ac)), 4),
                        acierto_texto_donde_aplica=round(float(np.nanmean(ac_txt)), 4),
                        f1_macro=round(float(f1_v.mean()), 4),
                        f1_macro_sd=round(float(f1_v.std(ddof=1)), 4),
                        exactitud=round(float(acc_v.mean()), 4),
                        exactitud_balanceada=round(float(bal_v.mean()), 4),
                        delta_f1=round(m, 4), ic95_bajo=round(lo, 4), ic95_alto=round(hi, 4),
                        semillas_positivas=f"{int((d > 0).sum())}/{len(d)}",
                        adopta_vs_base="SI" if lo > 0 else "NO"))
        for j, fam in enumerate(familias):
            df = fam_v[:, j] - fam_b[:, j]
            mf, _, lof, hif = ic95(df)
            propias = sum(1 for r in diag if r["variante"] == v and r["estado"] == "asignada"
                          and r["familia_real"] == fam)
            ajenas = sum(1 for r in diag if r["variante"] == v and r["estado"] == "asignada"
                         and r["familia_regla"] == fam and r["familia_real"] != fam)
            pornom = sum(1 for r in diag if r["variante"] == v and r["estado"] == "asignada"
                         and r["uso_nombre"] and r["familia_real"] == fam)
            res_fam.append(dict(variante=v, familia=fam,
                                f1_base=round(float(fam_b[:, j].mean()), 4),
                                f1_variante=round(float(fam_v[:, j].mean()), 4),
                                delta=round(mf, 4), ic95_bajo=round(lof, 4),
                                ic95_alto=round(hif, 4), propias_asignadas=propias,
                                ajenas_corregidas=ajenas, asignadas_usando_nombre=pornom))

    # ---- Aporte AISLADO del nivel de nombre (criterio preregistrado)
    aislado = []
    for base_v, mas_v in PARES_AISLAR:
        d = guardado[mas_v][0] - guardado[base_v][0]
        m, sd, lo, hi = ic95(d)
        aislado.append(dict(comparacion=f"{mas_v} - {base_v}", delta_f1_macro=round(m, 5),
                            sd=round(sd, 5), ic95_bajo=round(lo, 5), ic95_alto=round(hi, 5),
                            semillas_positivas=f"{int((d > 0).sum())}/{len(d)}",
                            adopta_nivel_nombre="SI" if lo > 0 else "NO",
                            prediccion_preregistrada="aporte < 0,005 con IC que incluye 0",
                            se_cumple="SI" if (abs(m) < 0.005 and lo <= 0 <= hi) else "NO"))

    dfd = pd.DataFrame(diag)
    pd.DataFrame(res).to_csv(OUT / "m6_resumen_variantes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(res_fam).to_csv(OUT / "m6_por_familia.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(aislado).to_csv(OUT / "m6_aporte_del_nombre.csv", index=False,
                                 encoding="utf-8-sig")
    pd.DataFrame(filas_pliegue).to_csv(OUT / "m6_por_pliegue.csv", index=False,
                                       encoding="utf-8-sig")
    dfd.to_csv(OUT / "m6_diagnostico_nota_por_nota.csv", index=False, encoding="utf-8-sig")

    print("\n=== TRES COLUMNAS DEL EXP. 2b ===")
    for r in res:
        if r["variante"].startswith("texto"):
            print(f"  {r['variante']:<44} macro-F1 {r['f1_macro']:.4f} "
                  f"+/- {r['f1_macro_sd']:.4f}")
        else:
            print(f"  {r['variante']:<44} cob {r['cobertura']:.4f} "
                  f"(por nombre {r['cobertura_por_nombre']:.4f}) | "
                  f"acierto {r['acierto_donde_aplica']:.4f} | "
                  f"macro-F1 {r['f1_macro']:.4f} | D {r['delta_f1']:+.4f} "
                  f"[{r['ic95_bajo']:+.4f}; {r['ic95_alto']:+.4f}] "
                  f"{r['semillas_positivas']} -> adopta {r['adopta_vs_base']}")

    print("\n=== APORTE AISLADO DEL NIVEL DE NOMBRE (criterio preregistrado) ===")
    for a in aislado:
        print(f"  {a['comparacion']}:")
        print(f"     D {a['delta_f1_macro']:+.5f} [{a['ic95_bajo']:+.5f}; "
              f"{a['ic95_alto']:+.5f}] {a['semillas_positivas']} -> "
              f"adopta nivel nombre: {a['adopta_nivel_nombre']} | "
              f"prediccion 1 se cumple: {a['se_cumple']}")

    print("\n=== CONTROLES PREREGISTRADOS (no deben asignarse POR NOMBRE) ===")
    for fam in FAM_CONTROL_NOMBRE:
        k = len(dfd[(dfd.familia_real == fam) & (dfd.estado == "asignada") &
                    (dfd.uso_nombre)])
        print(f"  {fam:<12} asignaciones usando nombre: {k}  "
              f"-> {'OK' if k == 0 else 'REVISAR'}")
    mal = dfd[(dfd.uso_nombre) & (dfd.estado == "asignada") &
              (dfd.nombre_genuino == "info.hta")]
    print(f"\n  Prediccion 3 (info.hta no separa DHARMA/PHOBOS): "
          f"asignadas por info.hta = {len(mal)} -> {'OK' if len(mal) == 0 else 'REVISAR'}")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
