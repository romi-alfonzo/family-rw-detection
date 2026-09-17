#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
revision_logo.py -- Revision independiente del protocolo LOGO (encargo del 2026-09-17).

Analisis que hacian falta para decidir si LOGO puede ser la cifra principal del frente de
notas y que protocolo_logo.py no hace. NO modifica ningun script canonico: importa de
clasificador_notas_v2.py, grafo_marcadores.py y protocolo_logo.py (misma carga de corpus,
mismo agrupamiento, mismo evaluador) y escribe en una carpeta NUEVA.

  --parte A
    1. Reproduce LOGO (texto y M.6) guardando la prediccion de CADA nota, y verifica que dos
       semillas distintas del LinearSVC den predicciones identicas (la premisa de reutilizar
       LOGO en las 50 semillas).
    2. IC bootstrap del macro-F1 de LOGO de cuatro maneras: por NOTAS con la convencion del
       script (labels = 30 familias, zero_division = 0), por notas con labels presentes en la
       remuestra, y las dos mismas por PLANTILLA (cluster bootstrap: se remuestrean los 99
       grupos, no las notas; las notas de una plantilla estan correlacionadas). Tambien el IC
       del delta M.6 - texto, que el script deja degenerado.
    3. Metricas del checklist del tutor bajo LOGO y bajo P2 (50 semillas): macro-F1 sobre 30
       y sobre 28 evaluables, exactitud, exactitud balanceada, MCC, F1 ponderado; P/R/F1 por
       familia con desvio bootstrap por plantilla; matrices de confusion.
    4. Distribucion del coseno maximo prueba -> entrenamiento por nota (char 3-5, el mismo
       espacio del agrupamiento) y acierto de LOGO por tramo; ademas un 1-NN bajo LOGO, para
       ver cuanto de LOGO es «parecerse a otra plantilla de la familia».
    5. Circularidad: si el texto de cada nota contiene el nombre de su familia o un alias, y
       LOGO / P2 con esos terminos ENMASCARADOS en el texto y excluidos del diccionario M.6.

  --parte B
    6. Sensibilidad al umbral de agrupamiento: se re-agrupa con 0,90 (canonico), 0,85, 0,80,
       0,75 y 0,70 y se re-corre LOGO y P2 (10 semillas) en cada uno.
    7. LOGO «con colchon»: grupos canonicos, pero de cada pliegue de entrenamiento se saca
       ademas toda nota con coseno > 0,80 (o > 0,70) con la plantilla de prueba.
    8. Anatomia de P2 (50 semillas, las mismas 0..49): familias sin ninguna plantilla de
       entrenamiento por pliegue, P2 excluyendo esas familias de la metrica frente a LOGO
       sobre las mismas familias, y P2bal: dos pliegues con las plantillas de cada familia
       repartidas de forma balanceada (ningun cero estructural salvo las de 1 plantilla),
       para separar el artefacto de la estratificacion del efecto del tamano de entrenamiento.

  --parte C
    9. Contencion: lo que el coseno no ve. Al abrir las notas de las familias de 2 plantillas
       que llegan a F1 1,00 bajo LOGO, la «segunda plantilla» resulta ser la misma nota con
       un bloque agregado o quitado (el HTML de SUNCRYPT extrae 184 caracteres que son un
       subconjunto literal de la otra nota). Se mide la contencion por 3-shingles de palabras
       (fraccion de los shingles de una nota presentes en otra plantilla de su familia), el
       acierto de LOGO por tramo de contencion, y LOGO / P2 con los grupos FUSIONADOS por
       contencion >= 0,9 / 0,8 / 0,7 ademas del coseno 0,90.

Uso:  python revision_logo.py --parte A|B|C [--salida CARPETA] [--boot 2000] [--semillas-p2 50]
Salida por defecto: 4_resultados/resultados_revision_logo_149/
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_recall_fscore_support)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, N_FOLDS, TFIDF_CHAR, UMBRAL_NEARDUP,
                                   agrupar_neardups, cargar_corpus, obtener_modelos,
                                   vectorizador)
from grafo_marcadores import _terminos_circulares, extraer_marcadores
from protocolo_logo import cargar_nombres, dicc_privados, evaluar, regla

RAIZ = _AQUI.parent
OUT_DEF = RAIZ / "4_resultados" / "resultados_revision_logo_149"

# Nombres compuestos que en las notas aparecen partidos («BLACK Matter», «Dark Side»).
# _terminos_circulares solo mira la cadena pegada; esto la complementa. Se permiten hasta 3
# separadores entre las partes para no enmascarar palabras sueltas como «black» o «side».
COMPUESTOS = {
    "BLACKMATTER": [("black", "matter")], "DARKSIDE": [("dark", "side")],
    "SUNCRYPT": [("sun", "crypt")], "NETWALKER": [("net", "walker")],
    "HELLOKITTY": [("hello", "kitty")], "BLACKBASTA": [("black", "basta")],
    "BLACKCAT": [("black", "cat")], "WASTEDLOCKER": [("wasted", "locker")],
    "BADRABBIT": [("bad", "rabbit")], "CRYPTOLOCKER": [("crypto", "locker")],
    "AVOSLOCKER": [("avos", "locker")], "MEDUZALOCKER": [("meduza", "locker"), ("medusa", "locker")],
    "RANSOMEXX": [("ransom", "exx")], "TESLACRYPT": [("tesla", "crypt")],
    "WANNACRY": [("wanna", "cry")], "NOTPETYA": [("not", "petya"), ("petya",)],
    "GANDCRAB": [("gand", "crab")], "LOCKBIT": [("lock", "bit")],
    "SODINOKIBI": [("revil",), ("sodin",)],
}


# ============================================================
# Carga comun
# ============================================================
def cargar_todo(umbral=UMBRAL_NEARDUP):
    textos, y, archivos, _ = cargar_corpus(CORPUS_DIR)
    y = np.asarray(y)
    textos_arr = np.array(textos, dtype=object)
    grupos, _ = agrupar_neardups(textos, umbral)
    grupos = np.asarray(grupos)
    iocs = [set(extraer_marcadores(t)) for t in textos]
    nom = cargar_nombres()
    nombres_nota = [nom.get((f, Path(a).name)) for f, a in zip(y, archivos)]
    return textos, textos_arr, y, archivos, grupos, iocs, nombres_nota


def sim_char(textos):
    X = TfidfVectorizer(**TFIDF_CHAR).fit_transform(textos)
    return (X @ X.T).toarray()


def plantillas_por_familia(y, grupos):
    return {f: len(set(grupos[y == f])) for f in np.unique(y)}


def evaluables(y, grupos):
    return np.array([f for f, k in plantillas_por_familia(y, grupos).items() if k >= 2])


def metricas(y, p, familias, evals):
    return dict(
        f1_macro_30=f1_score(y, p, average="macro", labels=familias, zero_division=0),
        f1_macro_evaluables=f1_score(y, p, average="macro", labels=evals, zero_division=0),
        n_evaluables=len(evals),
        exactitud=accuracy_score(y, p),
        exactitud_balanceada=balanced_accuracy_score(y, p),
        mcc=matthews_corrcoef(y, p),
        f1_ponderado=f1_score(y, p, average="weighted", zero_division=0),
    )


def ic_t(v):
    from scipy.stats import t as t_dist
    v = np.asarray(v, float)
    n = len(v)
    m = v.mean()
    if n < 2:
        return m, m, m, 0.0
    s = v.std(ddof=1)
    h = t_dist.ppf(0.975, n - 1) * s / np.sqrt(n)
    return m, m - h, m + h, s


# ============================================================
# Bootstrap: por notas y por plantilla (cluster)
# ============================================================
def bootstrap(y, preds, grupos, familias, B, seed, modo):
    """preds: dict nombre -> prediccion (misma longitud que y). Devuelve
    (estadisticos[nombre] -> array (B, 2): [macro labels=30, macro labels presentes],
     por_familia[nombre] -> array (B, n_fam) con NaN donde la familia no cayo en la remuestra)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    gid = np.unique(grupos)
    miembros = {g: np.where(grupos == g)[0] for g in gid}
    est = {k: np.empty((B, 2)) for k in preds}
    pf = {k: np.full((B, len(familias)), np.nan) for k in preds}
    for b in range(B):
        if modo == "notas":
            idx = rng.integers(0, n, n)          # identico a protocolo_logo.bootstrap_macro_f1
        else:
            gs = rng.choice(gid, len(gid), replace=True)
            idx = np.concatenate([miembros[g] for g in gs])
        yt = y[idx]
        pres = np.isin(familias, yt)
        for k, p in preds.items():
            f = f1_score(yt, p[idx], average=None, labels=familias, zero_division=0)
            est[k][b, 0] = f.mean()
            est[k][b, 1] = f[pres].mean()
            pf[k][b, pres] = f[pres]
    return est, pf


def resumen_boot(v, punto):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return dict(punto=round(float(punto), 4), ic_bajo=round(float(lo), 4), ic_alto=round(float(hi), 4),
                media_boot=round(float(np.mean(v)), 4), sd_boot=round(float(np.std(v, ddof=1)), 4),
                frac_bajo_050=round(float(np.mean(v < 0.50)), 4))


# ============================================================
# Circularidad
# ============================================================
def patron_familia(fam):
    pats = [re.escape(t) for t in sorted(_terminos_circulares(fam))]
    for partes in COMPUESTOS.get(fam, []):
        pats.append(r"[\s\W_]{0,3}".join(re.escape(p) for p in partes))
    return re.compile("|".join(pats), re.I)


def dicc_privados_sin_circ(tr, iocs, nombres_nota, y, patrones):
    """Como protocolo_logo.dicc_privados, pero excluyendo del diccionario los valores que
    contienen el nombre (o alias) de la PROPIA familia. Es excluir_circulares=True de la
    cascada canonica, con los patrones compuestos de arriba."""
    d = defaultdict(set)
    for i in tr:
        rx = patrones[y[i]]
        for c in iocs[i]:
            if rx.search(c[1]):
                continue
            d[c].add(y[i])
        if nombres_nota[i] and not rx.search(nombres_nota[i]):
            d[("[NOMBRE]", nombres_nota[i])].add(y[i])
    for k in [k for k, v in d.items() if len(v) > 1]:
        del d[k]
    return d


def evaluar_con_dicc(cv_splits, textos_arr, y, iocs, nombres_nota, seed, dicc_fn):
    """Copia de protocolo_logo.evaluar con el constructor del diccionario inyectado."""
    n = len(y)
    p_txt = np.empty(n, dtype=object)
    p_m6 = np.empty(n, dtype=object)
    aplica = np.zeros(n, dtype=bool)
    for tr, te in cv_splits:
        vec = vectorizador("combinado")
        Xtr = vec.fit_transform(textos_arr[tr])
        Xte = vec.transform(textos_arr[te])
        clf = obtener_modelos(seed)["LinearSVC"]
        clf.fit(Xtr, y[tr])
        pt = clf.predict(Xte)
        p_txt[te] = pt
        d = dicc_fn(tr)
        for k, i in enumerate(te):
            r = regla(i, d, iocs, nombres_nota)
            if r is not None:
                p_m6[i] = r
                aplica[i] = True
            else:
                p_m6[i] = pt[k]
    return p_txt, p_m6, aplica


# ============================================================
# PARTE A
# ============================================================
def parte_a(OUT, B, n_sem_p2):
    textos, textos_arr, y, archivos, grupos, iocs, nombres_nota = cargar_todo()
    familias = np.unique(y)
    evals = evaluables(y, grupos)
    ppf = plantillas_por_familia(y, grupos)
    n = len(y)
    print(f"Notas {n} | familias {len(familias)} | plantillas {len(set(grupos))} | "
          f"evaluables (>=2 plantillas) {len(evals)}")
    assert n == 149 and len(familias) == 30 and len(set(grupos)) == 99, "el corpus no es el de la propuesta"

    logo_splits = list(LeaveOneGroupOut().split(textos_arr, y, groups=grupos))

    # 1. Reproduccion y determinismo
    pt0, pm0, ap0 = evaluar(logo_splits, textos_arr, y, iocs, nombres_nota, 0)
    pt1, pm1, ap1 = evaluar(logo_splits, textos_arr, y, iocs, nombres_nota, 1)
    ident = int((pt0 == pt1).all() and (pm0 == pm1).all())
    m_txt = metricas(y, pt0, familias, evals)
    m_m6 = metricas(y, pm0, familias, evals)
    print(f"\n[1] LOGO texto macro-F1 {m_txt['f1_macro_30']:.4f} | M.6 {m_m6['f1_macro_30']:.4f} | "
          f"cobertura regla {ap0.mean():.4f} | semillas 0 y 1 identicas: {bool(ident)} "
          f"(notas distintas texto {(pt0 != pt1).sum()}, M.6 {(pm0 != pm1).sum()})")

    # 4. Coseno maximo prueba -> entrenamiento (fuera del grupo) y 1-NN bajo LOGO
    S = sim_char(textos)
    misma = grupos[:, None] == grupos[None, :]
    S_out = np.where(misma, -1.0, S)
    maxcos = S_out.max(1)
    nn = S_out.argmax(1)
    nn_fam = y[nn]
    S_prop = np.where(misma | (y[:, None] != y[None, :]), -1.0, S)
    S_ajena = np.where(y[:, None] == y[None, :], -1.0, S)
    maxcos_propia = S_prop.max(1)
    maxcos_ajena = S_ajena.max(1)

    # 5. Circularidad por nota
    patrones = {f: patron_familia(f) for f in familias}
    hits = []
    for i in range(n):
        ms = sorted({m.group(0).lower() for m in patrones[y[i]].finditer(textos[i])})
        hits.append("|".join(ms))
    tiene_nombre = np.array([h != "" for h in hits])

    df = pd.DataFrame(dict(
        archivo=archivos, familia=y, grupo=grupos, n_plantillas_familia=[ppf[f] for f in y],
        pred_txt=pt0, pred_m6=pm0, regla_aplica=ap0,
        acierto_txt=(pt0 == y), acierto_m6=(pm0 == y),
        maxcos_train=np.round(maxcos, 4), vecino_mas_cercano_familia=nn_fam,
        vecino_misma_familia=(nn_fam == y),
        maxcos_propia_fuera_grupo=np.round(maxcos_propia, 4), maxcos_otra_familia=np.round(maxcos_ajena, 4),
        texto_contiene_nombre_familia=tiene_nombre, terminos_encontrados=hits,
        nombre_genuino=[x or "" for x in nombres_nota],
    ))
    df.to_csv(OUT / "a_logo_predicciones_por_nota.csv", index=False, encoding="utf-8-sig")
    print(f"    coseno maximo prueba->entrenamiento: max {maxcos.max():.4f} (umbral {UMBRAL_NEARDUP})")

    tramos = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    filas = []
    for lo, hi in tramos:
        m = (maxcos >= lo) & (maxcos < hi)
        if m.sum() == 0:
            filas.append(dict(tramo=f"[{lo:.1f}; {hi:.1f})", n=0)); continue
        filas.append(dict(tramo=f"[{lo:.1f}; {hi:.1f})", n=int(m.sum()),
                          acierto_txt=round(float((pt0[m] == y[m]).mean()), 4),
                          acierto_m6=round(float((pm0[m] == y[m]).mean()), 4),
                          acierto_1nn=round(float((nn_fam[m] == y[m]).mean()), 4),
                          vecino_misma_familia=round(float((nn_fam[m] == y[m]).mean()), 4),
                          n_familias=len(set(y[m]))))
    tr_df = pd.DataFrame(filas)
    tr_df.to_csv(OUT / "a_coseno_tramos.csv", index=False, encoding="utf-8-sig")
    print("\n[4] acierto de LOGO por tramo de coseno maximo con el entrenamiento:")
    print(tr_df.to_string(index=False))
    acc_1nn = float((nn_fam == y).mean())
    f1_1nn = f1_score(y, nn_fam, average="macro", labels=familias, zero_division=0)
    print(f"    1-NN (char 3-5) bajo LOGO: exactitud {acc_1nn:.4f}, macro-F1 {f1_1nn:.4f} "
          f"(LinearSVC: {m_txt['exactitud']:.4f} / {m_txt['f1_macro_30']:.4f})")
    n_08 = int(((maxcos >= 0.8) & (maxcos < 0.9)).sum())
    print(f"    notas con un vecino de entrenamiento en [0,80; 0,90): {n_08} de {n}; "
          f"acierto texto en ellas {float((pt0[(maxcos>=0.8)&(maxcos<0.9)]==y[(maxcos>=0.8)&(maxcos<0.9)]).mean()):.4f}")

    # 2. Bootstrap
    preds = {"LOGO_txt": pt0, "LOGO_m6": pm0}
    filas = []
    for modo, seed in (("notas", 7), ("notas", 2026), ("plantillas", 7), ("plantillas", 2026)):
        est, pf = bootstrap(y, preds, grupos, familias, B, seed, modo)
        for k in preds:
            for j, conv in enumerate(("labels=30 (convencion del script)", "labels presentes en la remuestra")):
                r = resumen_boot(est[k][:, j], f1_score(y, preds[k], average="macro", labels=familias, zero_division=0))
                filas.append(dict(remuestreo=modo, semilla_boot=seed, capa=k, convencion=conv, **r))
        d = est["LOGO_m6"][:, 0] - est["LOGO_txt"][:, 0]
        r = resumen_boot(d, m_m6["f1_macro_30"] - m_txt["f1_macro_30"])
        r["frac_bajo_050"] = round(float(np.mean(d <= 0)), 4)
        filas.append(dict(remuestreo=modo, semilla_boot=seed, capa="delta M.6 - txt",
                          convencion="labels=30 (frac = P(delta<=0))", **r))
        if modo == "plantillas" and seed == 7:
            pf_sd = {k: np.nanstd(pf[k], axis=0, ddof=1) for k in preds}
            pf_lo = {k: np.nanpercentile(pf[k], 2.5, axis=0) for k in preds}
            pf_hi = {k: np.nanpercentile(pf[k], 97.5, axis=0) for k in preds}
    bo = pd.DataFrame(filas)
    bo.to_csv(OUT / "a_bootstrap_ic.csv", index=False, encoding="utf-8-sig")
    print(f"\n[2] IC 95 % bootstrap (B={B}) del macro-F1 de LOGO:")
    print(bo.to_string(index=False))

    # 3. Metricas del checklist, LOGO y P2 (50 semillas)
    print(f"\n[3] P2 a {n_sem_p2} semillas para las mismas metricas ...")
    p2_m = {"txt": [], "m6": []}
    for s in range(n_sem_p2):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
        sp = list(cv.split(textos_arr, y, groups=grupos))
        pt, pm, ap = evaluar(sp, textos_arr, y, iocs, nombres_nota, s)
        p2_m["txt"].append(metricas(y, pt, familias, evals))
        p2_m["m6"].append(metricas(y, pm, familias, evals))
    filas = []
    for capa, m in (("texto solo", m_txt), ("M.6", m_m6)):
        filas.append(dict(protocolo="LOGO", capa=capa, **{k: round(float(v), 4) for k, v in m.items()}))
    for capa, lst in (("texto solo", p2_m["txt"]), ("M.6", p2_m["m6"])):
        dd = pd.DataFrame(lst)
        fila = dict(protocolo=f"P2 ({n_sem_p2} sem., media)", capa=capa)
        fila.update({k: round(float(dd[k].mean()), 4) for k in dd.columns})
        filas.append(fila)
        fila = dict(protocolo=f"P2 ({n_sem_p2} sem., sd)", capa=capa)
        fila.update({k: round(float(dd[k].std(ddof=1)), 4) for k in dd.columns})
        filas.append(fila)
    me = pd.DataFrame(filas)
    me.to_csv(OUT / "a_metricas_logo_y_p2.csv", index=False, encoding="utf-8-sig")
    print(me.to_string(index=False))

    # por familia
    filas = []
    for capa, p, k in (("txt", pt0, "LOGO_txt"), ("m6", pm0, "LOGO_m6")):
        pr, rc, f1, su = precision_recall_fscore_support(y, p, labels=familias, zero_division=0)
        for j, f in enumerate(familias):
            filas.append(dict(capa=capa, familia=f, n_notas=int(su[j]), n_plantillas=ppf[f],
                              precision=round(float(pr[j]), 4), recall=round(float(rc[j]), 4),
                              f1=round(float(f1[j]), 4),
                              f1_sd_boot_plantillas=round(float(pf_sd[k][j]), 4),
                              f1_ic_boot_plantillas=f"[{pf_lo[k][j]:.3f}; {pf_hi[k][j]:.3f}]",
                              notas_con_nombre_en_texto=int(tiene_nombre[y == f].sum()),
                              plantillas_con_nombre_en_texto=len(set(grupos[(y == f) & tiene_nombre]))))
    pfam = pd.DataFrame(filas)
    pfam.to_csv(OUT / "a_por_familia.csv", index=False, encoding="utf-8-sig")
    for capa, p in (("txt", pt0), ("m6", pm0)):
        cm = confusion_matrix(y, p, labels=familias)
        pd.DataFrame(cm, index=familias, columns=familias).to_csv(
            OUT / f"a_matriz_confusion_logo_{capa}.csv", encoding="utf-8-sig")

    # 5. Circularidad: tabla por familia y corrida enmascarada
    circ = pfam[pfam.capa == "txt"][["familia", "n_notas", "n_plantillas", "notas_con_nombre_en_texto",
                                     "plantillas_con_nombre_en_texto"]]
    circ.to_csv(OUT / "a_circularidad_por_familia.csv", index=False, encoding="utf-8-sig")
    print("\n[5] notas cuyo texto contiene el nombre/alias de SU familia:")
    print(f"    {int(tiene_nombre.sum())} de {n} notas; familias con >=1: "
          f"{int((circ.notas_con_nombre_en_texto > 0).sum())} de 30")
    print(circ[circ.notas_con_nombre_en_texto > 0].to_string(index=False))

    rx_todo = re.compile("|".join(p.pattern for p in patrones.values()), re.I)
    textos_masc = np.array([rx_todo.sub(" ", t) for t in textos], dtype=object)
    n_reemplazos = sum(len(rx_todo.findall(t)) for t in textos)
    dicc_fn = lambda tr: dicc_privados_sin_circ(tr, iocs, nombres_nota, y, patrones)
    ptm, pmm, apm = evaluar_con_dicc(logo_splits, textos_masc, y, iocs, nombres_nota, 0, dicc_fn)
    mm_txt = metricas(y, ptm, familias, evals)
    mm_m6 = metricas(y, pmm, familias, evals)
    p2m = {"txt": [], "m6": []}
    for s in range(10):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
        sp = list(cv.split(textos_arr, y, groups=grupos))
        a, b, _ = evaluar_con_dicc(sp, textos_masc, y, iocs, nombres_nota, s, dicc_fn)
        p2m["txt"].append(f1_score(y, a, average="macro", labels=familias, zero_division=0))
        p2m["m6"].append(f1_score(y, b, average="macro", labels=familias, zero_division=0))
    filas = [
        dict(protocolo="LOGO", capa="texto solo", variante="canonico", f1_macro_30=round(m_txt["f1_macro_30"], 4)),
        dict(protocolo="LOGO", capa="texto solo", variante="enmascarado", f1_macro_30=round(mm_txt["f1_macro_30"], 4)),
        dict(protocolo="LOGO", capa="M.6", variante="canonico", f1_macro_30=round(m_m6["f1_macro_30"], 4),
             cobertura=round(float(ap0.mean()), 4)),
        dict(protocolo="LOGO", capa="M.6", variante="enmascarado + dicc. sin circulares",
             f1_macro_30=round(mm_m6["f1_macro_30"], 4), cobertura=round(float(apm.mean()), 4)),
        dict(protocolo="P2 (10 sem.)", capa="texto solo", variante="enmascarado",
             f1_macro_30=round(float(np.mean(p2m["txt"])), 4), sd=round(float(np.std(p2m["txt"], ddof=1)), 4)),
        dict(protocolo="P2 (10 sem.)", capa="M.6", variante="enmascarado + dicc. sin circulares",
             f1_macro_30=round(float(np.mean(p2m["m6"])), 4), sd=round(float(np.std(p2m["m6"], ddof=1)), 4)),
    ]
    em = pd.DataFrame(filas)
    em.to_csv(OUT / "a_enmascarado_resumen.csv", index=False, encoding="utf-8-sig")
    print(f"\n    Enmascaramiento: {n_reemplazos} ocurrencias de nombres/alias de las 30 familias borradas del texto.")
    print(em.to_string(index=False))
    _, _, f_can, _ = precision_recall_fscore_support(y, pt0, labels=familias, zero_division=0)
    _, _, f_mas, _ = precision_recall_fscore_support(y, ptm, labels=familias, zero_division=0)
    _, _, f6_can, _ = precision_recall_fscore_support(y, pm0, labels=familias, zero_division=0)
    _, _, f6_mas, _ = precision_recall_fscore_support(y, pmm, labels=familias, zero_division=0)
    pfe = pd.DataFrame(dict(familia=familias, n_plantillas=[ppf[f] for f in familias],
                            notas_con_nombre=[int(tiene_nombre[y == f].sum()) for f in familias],
                            LOGO_txt=np.round(f_can, 4), LOGO_txt_enmascarado=np.round(f_mas, 4),
                            LOGO_m6=np.round(f6_can, 4), LOGO_m6_enmascarado=np.round(f6_mas, 4)))
    pfe["delta_txt"] = pfe.LOGO_txt_enmascarado - pfe.LOGO_txt
    pfe.to_csv(OUT / "a_enmascarado_por_familia.csv", index=False, encoding="utf-8-sig")
    print("\n    por familia, las que cambian al enmascarar (texto):")
    print(pfe[pfe.delta_txt.abs() > 1e-9].to_string(index=False))
    cinco = ["SUNCRYPT", "CUBA", "NETWALKER", "BLACKMATTER", "DARKSIDE"]
    print("\n    las cinco de 2 plantillas que llegan a 1,00:")
    print(pfe[pfe.familia.isin(cinco)].to_string(index=False))
    print("\n    detalle nota a nota de esas cinco:")
    cols = ["archivo", "grupo", "acierto_txt", "acierto_m6", "regla_aplica", "maxcos_train",
            "vecino_mas_cercano_familia", "maxcos_propia_fuera_grupo", "maxcos_otra_familia",
            "texto_contiene_nombre_familia", "terminos_encontrados", "nombre_genuino"]
    print(df[df.familia.isin(cinco)][cols].to_string(index=False))
    print(f"\nSalidas en {OUT}")


# ============================================================
# PARTE B
# ============================================================
def split_p2bal(y, grupos, familias, rng):
    """Dos pliegues por plantilla con reparto balanceado dentro de cada familia. Las familias
    chicas se asignan primero; un grupo mixto (dos familias) queda donde lo puso la primera."""
    fold_de = {}
    porfam = {f: sorted(set(grupos[y == f])) for f in familias}
    for f in sorted(familias, key=lambda f: (len(porfam[f]), f)):
        c = Counter(fold_de[g] for g in porfam[f] if g in fold_de)
        libres = [g for g in porfam[f] if g not in fold_de]
        libres = [libres[i] for i in rng.permutation(len(libres))]
        for g in libres:
            k = int(rng.integers(2)) if c[0] == c[1] else (0 if c[0] < c[1] else 1)
            fold_de[g] = k
            c[k] += 1
    folds = np.array([fold_de[g] for g in grupos])
    return [(np.where(folds != k)[0], np.where(folds == k)[0]) for k in (0, 1)]


def parte_b(OUT, n_sem_p2, solo=("umbral", "colchon", "anatomia"), umbrales=(0.90, 0.85, 0.80, 0.75, 0.70),
            colchones=(0.90, 0.85, 0.80, 0.70)):
    textos, textos_arr, y, archivos, grupos90, iocs, nombres_nota = cargar_todo()
    familias = np.unique(y)
    n = len(y)
    S = sim_char(textos)
    sufijo = "" if len(solo) == 3 else "_" + "_".join(solo)

    if "umbral" not in solo:
        umbrales = ()
    # 6. Sensibilidad al umbral
    print("[6] sensibilidad al umbral de agrupamiento (LOGO y P2 a 10 semillas en cada uno):")
    filas = []
    for u in umbrales:
        g, _ = agrupar_neardups(textos, u)
        g = np.asarray(g)
        ppf = plantillas_por_familia(y, g)
        ev = np.array([f for f, k in ppf.items() if k >= 2])
        mixtos = sum(1 for gg in set(g) if len(set(y[g == gg])) > 1)
        sp = list(LeaveOneGroupOut().split(textos_arr, y, groups=g))
        pt, pm, ap = evaluar(sp, textos_arr, y, iocs, nombres_nota, 0)
        misma = g[:, None] == g[None, :]
        mc = np.where(misma, -1.0, S).max(1)
        p2 = {"txt": [], "m6": [], "txt_ev": [], "m6_ev": []}
        for s in range(10):
            cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
            a, b, _ = evaluar(list(cv.split(textos_arr, y, groups=g)), textos_arr, y, iocs, nombres_nota, s)
            p2["txt"].append(f1_score(y, a, average="macro", labels=familias, zero_division=0))
            p2["m6"].append(f1_score(y, b, average="macro", labels=familias, zero_division=0))
            p2["txt_ev"].append(f1_score(y, a, average="macro", labels=ev, zero_division=0))
            p2["m6_ev"].append(f1_score(y, b, average="macro", labels=ev, zero_division=0))
        fila = dict(umbral=u, n_plantillas=len(set(g)), grupos_mixtos=mixtos,
                    familias_1_plantilla=int(sum(1 for k in ppf.values() if k == 1)),
                    familias_evaluables=len(ev),
                    maxcos_train_max=round(float(mc.max()), 4),
                    notas_vecino_080_090=int(((mc >= 0.80) & (mc < 0.90)).sum()),
                    notas_vecino_070_080=int(((mc >= 0.70) & (mc < 0.80)).sum()),
                    LOGO_txt_30=round(f1_score(y, pt, average="macro", labels=familias, zero_division=0), 4),
                    LOGO_m6_30=round(f1_score(y, pm, average="macro", labels=familias, zero_division=0), 4),
                    LOGO_txt_evaluables=round(f1_score(y, pt, average="macro", labels=ev, zero_division=0), 4),
                    LOGO_m6_evaluables=round(f1_score(y, pm, average="macro", labels=ev, zero_division=0), 4),
                    LOGO_exactitud_txt=round(accuracy_score(y, pt), 4),
                    LOGO_cobertura_regla=round(float(ap.mean()), 4),
                    P2_txt_30=round(float(np.mean(p2["txt"])), 4), P2_txt_sd=round(float(np.std(p2["txt"], ddof=1)), 4),
                    P2_m6_30=round(float(np.mean(p2["m6"])), 4),
                    P2_txt_evaluables=round(float(np.mean(p2["txt_ev"])), 4),
                    P2_m6_evaluables=round(float(np.mean(p2["m6_ev"])), 4))
        fila["delta_txt_30"] = round(fila["LOGO_txt_30"] - fila["P2_txt_30"], 4)
        fila["delta_m6_30"] = round(fila["LOGO_m6_30"] - fila["P2_m6_30"], 4)
        filas.append(fila)
        print(f"  umbral {u:.2f}: plantillas {fila['n_plantillas']:3d} | fam. 1 plantilla {fila['familias_1_plantilla']:2d} | "
              f"LOGO txt {fila['LOGO_txt_30']:.4f} m6 {fila['LOGO_m6_30']:.4f} | P2 txt {fila['P2_txt_30']:.4f} m6 {fila['P2_m6_30']:.4f} | "
              f"Δtxt {fila['delta_txt_30']:+.4f} | sobre evaluables LOGO txt {fila['LOGO_txt_evaluables']:.4f} m6 {fila['LOGO_m6_evaluables']:.4f}")
    if filas:
        pd.DataFrame(filas).to_csv(OUT / f"b_umbral_sensibilidad{sufijo}.csv", index=False, encoding="utf-8-sig")

    # 7. LOGO con colchon (grupos canonicos, se saca del train todo vecino > colchon)
    print("\n[7] LOGO con colchon: grupos a 0,90, pero fuera del entrenamiento toda nota con coseno > c con la plantilla de prueba")
    filas = []
    sp90 = list(LeaveOneGroupOut().split(textos_arr, y, groups=grupos90))
    ppf90 = plantillas_por_familia(y, grupos90)
    if "colchon" not in solo:
        colchones = ()
    for c in colchones:
        splits, quitadas, fam_sin_propia = [], [], 0
        for tr, te in sp90:
            cerca = (S[np.ix_(tr, te)].max(1) > c)
            tr2 = tr[~cerca]
            quitadas.append(int(cerca.sum()))
            fam = y[te[0]]
            if ppf90[fam] >= 2 and (y[tr2] == fam).sum() == 0:
                fam_sin_propia += 1
            splits.append((tr2, te))
        pt, pm, ap = evaluar(splits, textos_arr, y, iocs, nombres_nota, 0)
        ev = evaluables(y, grupos90)
        fila = dict(colchon=c, notas_quitadas_media=round(float(np.mean(quitadas)), 2),
                    notas_quitadas_max=int(max(quitadas)),
                    pliegues_evaluables_sin_plantilla_propia=fam_sin_propia,
                    LOGO_txt_30=round(f1_score(y, pt, average="macro", labels=familias, zero_division=0), 4),
                    LOGO_m6_30=round(f1_score(y, pm, average="macro", labels=familias, zero_division=0), 4),
                    LOGO_txt_28=round(f1_score(y, pt, average="macro", labels=ev, zero_division=0), 4),
                    LOGO_m6_28=round(f1_score(y, pm, average="macro", labels=ev, zero_division=0), 4),
                    exactitud_txt=round(accuracy_score(y, pt), 4), cobertura_regla=round(float(ap.mean()), 4))
        filas.append(fila)
        print(f"  colchon {c:.2f}: quita {fila['notas_quitadas_media']:.2f} notas/pliegue (max {fila['notas_quitadas_max']}), "
              f"pliegues que pierden toda plantilla propia {fam_sin_propia:3d} | LOGO txt {fila['LOGO_txt_30']:.4f} m6 {fila['LOGO_m6_30']:.4f}")
    if filas:
        pd.DataFrame(filas).to_csv(OUT / f"b_logo_colchon{sufijo}.csv", index=False, encoding="utf-8-sig")
    if "anatomia" not in solo:
        print(f"\nSalidas en {OUT}")
        return

    # 8. Anatomia de P2 y P2bal
    print(f"\n[8] anatomia de P2 ({n_sem_p2} semillas) y P2bal")
    pt_logo, pm_logo, _ = evaluar(sp90, textos_arr, y, iocs, nombres_nota, 0)
    _, _, f_logo_txt, _ = precision_recall_fscore_support(y, pt_logo, labels=familias, zero_division=0)
    _, _, f_logo_m6, _ = precision_recall_fscore_support(y, pm_logo, labels=familias, zero_division=0)
    ppf = ppf90
    fam2 = [f for f in familias if ppf[f] == 2]
    filas_s, filas_f = [], []
    estr_por_fam = defaultdict(int)
    f1_p2_fam = defaultdict(list)          # (familia, 'separadas'|'juntas') -> F1 txt
    f1_p2_fam_m6 = defaultdict(list)
    pb = {"txt": [], "m6": [], "plant_train": []}
    rng_bal = np.random.default_rng(12345)
    for s in range(n_sem_p2):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
        sp = list(cv.split(textos_arr, y, groups=grupos90))
        sin_train = []
        plant_train = []
        for tr, te in sp:
            ftr = set(y[tr])
            sin_train.append([f for f in familias if f not in ftr])
            plant_train.append(len(set(grupos90[tr])))
        estructurales = sorted(set(sin_train[0]) | set(sin_train[1]))
        for f in estructurales:
            estr_por_fam[f] += 1
        pt, pm, _ = evaluar(sp, textos_arr, y, iocs, nombres_nota, s)
        _, _, f_txt, _ = precision_recall_fscore_support(y, pt, labels=familias, zero_division=0)
        _, _, f_m6, _ = precision_recall_fscore_support(y, pm, labels=familias, zero_division=0)
        no_estr = np.array([f not in estructurales for f in familias])
        filas_s.append(dict(
            semilla=s, fam_sin_train_pliegue0=len(sin_train[0]), fam_sin_train_pliegue1=len(sin_train[1]),
            fam_estructurales=len(estructurales), estructurales=";".join(estructurales),
            plantillas_train_p0=plant_train[0], plantillas_train_p1=plant_train[1],
            P2_txt_30=round(float(f_txt.mean()), 4), P2_m6_30=round(float(f_m6.mean()), 4),
            P2_txt_sin_estructurales=round(float(f_txt[no_estr].mean()), 4),
            P2_m6_sin_estructurales=round(float(f_m6[no_estr].mean()), 4),
            LOGO_txt_mismas_familias=round(float(f_logo_txt[no_estr].mean()), 4),
            LOGO_m6_mismas_familias=round(float(f_logo_m6[no_estr].mean()), 4),
            n_familias_no_estructurales=int(no_estr.sum())))
        for j, f in enumerate(familias):
            if f in fam2:
                clave = "juntas" if f in estructurales else "separadas"
                f1_p2_fam[(f, clave)].append(float(f_txt[j]))
                f1_p2_fam_m6[(f, clave)].append(float(f_m6[j]))
        # P2bal
        spb = split_p2bal(y, grupos90, familias, rng_bal)
        ptb, pmb, _ = evaluar(spb, textos_arr, y, iocs, nombres_nota, s)
        pb["txt"].append(f1_score(y, ptb, average="macro", labels=familias, zero_division=0))
        pb["m6"].append(f1_score(y, pmb, average="macro", labels=familias, zero_division=0))
        pb["plant_train"].append(np.mean([len(set(grupos90[tr])) for tr, _ in spb]))
    ds = pd.DataFrame(filas_s)
    ds.to_csv(OUT / "b_p2_anatomia_por_semilla.csv", index=False, encoding="utf-8-sig")
    for f in familias:
        fila = dict(familia=f, n_plantillas=ppf[f], semillas_sin_train_en_algun_pliegue=estr_por_fam.get(f, 0),
                    frac_semillas=round(estr_por_fam.get(f, 0) / n_sem_p2, 3))
        if f in fam2:
            for clave in ("separadas", "juntas"):
                v = f1_p2_fam.get((f, clave), [])
                fila[f"P2_txt_{clave}_n"] = len(v)
                fila[f"P2_txt_{clave}_media"] = round(float(np.mean(v)), 4) if v else np.nan
                v6 = f1_p2_fam_m6.get((f, clave), [])
                fila[f"P2_m6_{clave}_media"] = round(float(np.mean(v6)), 4) if v6 else np.nan
            j = list(familias).index(f)
            fila["LOGO_txt"] = round(float(f_logo_txt[j]), 4)
            fila["LOGO_m6"] = round(float(f_logo_m6[j]), 4)
        filas_f.append(fila)
    dfam = pd.DataFrame(filas_f)
    dfam.to_csv(OUT / "b_p2_anatomia_por_familia.csv", index=False, encoding="utf-8-sig")

    m_p2, lo, hi, sd = ic_t(ds.P2_txt_30)
    m_p2m, lo6, hi6, sd6 = ic_t(ds.P2_m6_30)
    m_se, lo_se, hi_se, _ = ic_t(ds.P2_txt_sin_estructurales)
    m_se6, _, _, _ = ic_t(ds.P2_m6_sin_estructurales)
    m_lg, _, _, _ = ic_t(ds.LOGO_txt_mismas_familias)
    m_lg6, _, _, _ = ic_t(ds.LOGO_m6_mismas_familias)
    m_pb, lo_pb, hi_pb, sd_pb = ic_t(pb["txt"])
    m_pb6, lo_pb6, hi_pb6, sd_pb6 = ic_t(pb["m6"])
    fam_pliegue = float(np.concatenate([ds.fam_sin_train_pliegue0, ds.fam_sin_train_pliegue1]).mean())
    res = pd.DataFrame([
        dict(fila="P2 canonico, 30 familias", txt=round(m_p2, 4), txt_ic=f"[{lo:.4f}; {hi:.4f}]", txt_sd=round(sd, 4),
             m6=round(m_p2m, 4), m6_ic=f"[{lo6:.4f}; {hi6:.4f}]"),
        dict(fila="P2 sin las familias estructurales de cada semilla", txt=round(m_se, 4), txt_ic=f"[{lo_se:.4f}; {hi_se:.4f}]",
             m6=round(m_se6, 4), nota=f"familias promedio {ds.n_familias_no_estructurales.mean():.1f}"),
        dict(fila="LOGO sobre esas mismas familias (media entre semillas)", txt=round(m_lg, 4), m6=round(m_lg6, 4)),
        dict(fila="P2bal (plantillas repartidas de forma balanceada), 30 familias", txt=round(m_pb, 4),
             txt_ic=f"[{lo_pb:.4f}; {hi_pb:.4f}]", txt_sd=round(sd_pb, 4), m6=round(m_pb6, 4),
             m6_ic=f"[{lo_pb6:.4f}; {hi_pb6:.4f}]", nota=f"plantillas en train por pliegue {np.mean(pb['plant_train']):.1f}"),
        dict(fila="LOGO, 30 familias", txt=round(float(f_logo_txt.mean()), 4), m6=round(float(f_logo_m6.mean()), 4)),
        dict(fila="familias sin plantilla de entrenamiento por PLIEGUE (media P2)", txt=round(fam_pliegue, 2)),
        dict(fila="familias estructurales por SEMILLA (union de los 2 pliegues, media P2)", txt=round(float(ds.fam_estructurales.mean()), 2)),
        dict(fila="plantillas en train por pliegue (media P2)", txt=round(float(np.concatenate([ds.plantillas_train_p0, ds.plantillas_train_p1]).mean()), 1)),
    ])
    res.to_csv(OUT / "b_p2_resumen.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    print("\n  distribucion de familias sin train por pliegue:",
          dict(sorted(Counter(np.concatenate([ds.fam_sin_train_pliegue0, ds.fam_sin_train_pliegue1]).tolist()).items())))
    print("\n  por familia: semillas en que queda sin train en algun pliegue, y para las de 2 plantillas F1 segun caigan separadas o juntas:")
    print(dfam[(dfam.frac_semillas > 0) | (dfam.n_plantillas == 2)].to_string(index=False))
    print(f"\nSalidas en {OUT}")


# ============================================================
# PARTE C: contencion (lo que el coseno no ve)
# ============================================================
def shingles(t, k=3):
    w = re.findall(r"\w+", t.lower())
    if len(w) < k:
        return {tuple(w)} if w else set()
    return {tuple(w[i:i + k]) for i in range(len(w) - k + 1)}


def matriz_contencion(textos, k=3):
    """C[i, j] = fraccion de los k-shingles de palabras de i que aparecen en j (Broder).
    Asimetrica a proposito: una nota corta contenida en una larga da C[corta, larga] ~ 1."""
    S = [shingles(t, k) for t in textos]
    n = len(S)
    C = np.zeros((n, n))
    for i in range(n):
        if not S[i]:
            continue
        for j in range(n):
            if i != j:
                C[i, j] = len(S[i] & S[j]) / len(S[i])
    return C


def fusionar_por_contencion(grupos, C, umbral):
    """Une dos grupos cuando alguna nota de uno esta contenida (>= umbral) en una nota del
    otro, en cualquier direccion. No usa etiquetas: es deduplicacion, como agrupar_neardups."""
    padre = {g: g for g in set(grupos)}

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x
    n = len(grupos)
    for i in range(n):
        for j in range(n):
            if grupos[i] != grupos[j] and C[i, j] >= umbral:
                ri, rj = raiz(grupos[i]), raiz(grupos[j])
                if ri != rj:
                    padre[rj] = ri
    return np.array([raiz(g) for g in grupos])


def parte_c(OUT):
    textos, textos_arr, y, archivos, grupos, iocs, nombres_nota = cargar_todo()
    familias = np.unique(y)
    n = len(y)
    ppf = plantillas_por_familia(y, grupos)
    S = sim_char(textos)
    C = matriz_contencion(textos)
    misma = grupos[:, None] == grupos[None, :]
    propia = (y[:, None] == y[None, :]) & ~misma
    ajena = y[:, None] != y[None, :]
    cont_propia = np.where(propia, C, -1.0).max(1)
    j_propia = np.where(propia, C, -1.0).argmax(1)
    cont_ajena = np.where(ajena, C, -1.0).max(1)
    cos_propia = np.where(propia, S, -1.0).max(1)

    sp = list(LeaveOneGroupOut().split(textos_arr, y, groups=grupos))
    pt, pm, ap = evaluar(sp, textos_arr, y, iocs, nombres_nota, 0)

    df = pd.DataFrame(dict(archivo=archivos, familia=y, grupo=grupos, n_plantillas_familia=[ppf[f] for f in y],
                           n_palabras=[len(re.findall(r"\w+", t)) for t in textos],
                           contencion_max_en_otra_plantilla_propia=np.round(cont_propia, 3),
                           nota_que_la_contiene=[archivos[j] if cont_propia[i] >= 0 else "" for i, j in enumerate(j_propia)],
                           coseno_max_otra_plantilla_propia=np.round(cos_propia, 4),
                           contencion_max_otra_familia=np.round(cont_ajena, 3),
                           acierto_logo_txt=(pt == y), acierto_logo_m6=(pm == y)))
    df.to_csv(OUT / "c_contencion_por_nota.csv", index=False, encoding="utf-8-sig")

    print("[C1] contencion (fraccion de 3-shingles de palabras de la nota que aparecen en OTRA plantilla de su familia):")
    tramos = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    filas = []
    ev = (cont_propia >= 0)     # notas con al menos otra plantilla propia
    for lo, hi in tramos:
        m = ev & (cont_propia >= lo) & (cont_propia < hi)
        filas.append(dict(tramo=f"[{lo:.1f}; {hi:.1f})", n_notas=int(m.sum()),
                          n_plantillas=len(set(grupos[m])),
                          acierto_txt=round(float((pt[m] == y[m]).mean()), 4) if m.any() else np.nan,
                          acierto_m6=round(float((pm[m] == y[m]).mean()), 4) if m.any() else np.nan,
                          coseno_medio=round(float(cos_propia[m].mean()), 4) if m.any() else np.nan))
    m = ~ev
    filas.append(dict(tramo="sin otra plantilla propia (1 plantilla)", n_notas=int(m.sum()), n_plantillas=len(set(grupos[m])),
                      acierto_txt=round(float((pt[m] == y[m]).mean()), 4), acierto_m6=round(float((pm[m] == y[m]).mean()), 4)))
    t1 = pd.DataFrame(filas)
    t1.to_csv(OUT / "c_contencion_tramos.csv", index=False, encoding="utf-8-sig")
    print(t1.to_string(index=False))

    print("\n[C2] por familia: contencion maxima entre sus plantillas (cualquier direccion) y coseno maximo entre plantillas:")
    filas = []
    for f in familias:
        idx = np.where(y == f)[0]
        if ppf[f] < 2:
            filas.append(dict(familia=f, n_plantillas=1)); continue
        sub = C[np.ix_(idx, idx)].copy()
        subm = misma[np.ix_(idx, idx)]
        sub[subm] = -1
        k = np.unravel_index(sub.argmax(), sub.shape)
        cs = S[np.ix_(idx, idx)].copy(); cs[subm] = -1
        _, _, f1s, _ = precision_recall_fscore_support(y, pt, labels=[f], zero_division=0)
        filas.append(dict(familia=f, n_plantillas=ppf[f], contencion_max=round(float(sub.max()), 3),
                          par=f"{Path(archivos[idx[k[0]]]).name} -> {Path(archivos[idx[k[1]]]).name}",
                          coseno_max_entre_plantillas=round(float(cs.max()), 4),
                          LOGO_txt_f1=round(float(f1s[0]), 4)))
    t2 = pd.DataFrame(filas)
    t2.to_csv(OUT / "c_contencion_por_familia.csv", index=False, encoding="utf-8-sig")
    print(t2.sort_values("contencion_max", ascending=False).to_string(index=False))

    print("\n[C3] LOGO y P2 (10 semillas) con grupos FUSIONADOS por contencion (ademas del coseno 0,90):")
    filas = []
    for u in (None, 0.9, 0.8, 0.7):
        g = grupos if u is None else fusionar_por_contencion(grupos, C, u)
        ppf_u = plantillas_por_familia(y, g)
        ev_u = np.array([f for f, k in ppf_u.items() if k >= 2])
        mixtos = sum(1 for gg in set(g) if len(set(y[g == gg])) > 1)
        spu = list(LeaveOneGroupOut().split(textos_arr, y, groups=g))
        a, b, apu = evaluar(spu, textos_arr, y, iocs, nombres_nota, 0)
        p2 = {"txt": [], "m6": [], "txt_ev": [], "m6_ev": []}
        for s in range(10):
            cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
            c1, c2, _ = evaluar(list(cv.split(textos_arr, y, groups=g)), textos_arr, y, iocs, nombres_nota, s)
            p2["txt"].append(f1_score(y, c1, average="macro", labels=familias, zero_division=0))
            p2["m6"].append(f1_score(y, c2, average="macro", labels=familias, zero_division=0))
            p2["txt_ev"].append(f1_score(y, c1, average="macro", labels=ev_u, zero_division=0))
            p2["m6_ev"].append(f1_score(y, c2, average="macro", labels=ev_u, zero_division=0))
        _, _, f1u, _ = precision_recall_fscore_support(y, a, labels=familias, zero_division=0)
        cinco = {f: round(float(f1u[list(familias).index(f)]), 3) for f in ("SUNCRYPT", "CUBA", "NETWALKER", "BLACKMATTER", "DARKSIDE")}
        fila = dict(fusion_contencion="ninguna (canonico)" if u is None else f">= {u}", n_plantillas=len(set(g)),
                    grupos_mixtos=mixtos, familias_1_plantilla=int(sum(1 for k in ppf_u.values() if k == 1)),
                    familias_evaluables=len(ev_u),
                    LOGO_txt_30=round(f1_score(y, a, average="macro", labels=familias, zero_division=0), 4),
                    LOGO_m6_30=round(f1_score(y, b, average="macro", labels=familias, zero_division=0), 4),
                    LOGO_txt_evaluables=round(f1_score(y, a, average="macro", labels=ev_u, zero_division=0), 4),
                    LOGO_m6_evaluables=round(f1_score(y, b, average="macro", labels=ev_u, zero_division=0), 4),
                    LOGO_exactitud_txt=round(accuracy_score(y, a), 4), LOGO_cobertura=round(float(apu.mean()), 4),
                    P2_txt_30=round(float(np.mean(p2["txt"])), 4), P2_txt_sd=round(float(np.std(p2["txt"], ddof=1)), 4),
                    P2_m6_30=round(float(np.mean(p2["m6"])), 4),
                    P2_txt_evaluables=round(float(np.mean(p2["txt_ev"])), 4), P2_m6_evaluables=round(float(np.mean(p2["m6_ev"])), 4),
                    cinco_familias_LOGO_txt=str(cinco))
        fila["delta_txt_30"] = round(fila["LOGO_txt_30"] - fila["P2_txt_30"], 4)
        fila["delta_m6_30"] = round(fila["LOGO_m6_30"] - fila["P2_m6_30"], 4)
        filas.append(fila)
        print(f"  fusion {fila['fusion_contencion']:<20} plantillas {fila['n_plantillas']:3d} | fam. 1 plantilla {fila['familias_1_plantilla']:2d} | "
              f"LOGO txt {fila['LOGO_txt_30']:.4f} m6 {fila['LOGO_m6_30']:.4f} (evaluables {fila['LOGO_txt_evaluables']:.4f} / {fila['LOGO_m6_evaluables']:.4f}) | "
              f"P2 txt {fila['P2_txt_30']:.4f} m6 {fila['P2_m6_30']:.4f} | Δtxt {fila['delta_txt_30']:+.4f}")
        print(f"      cinco familias LOGO txt: {cinco}")
    pd.DataFrame(filas).to_csv(OUT / "c_fusion_contencion.csv", index=False, encoding="utf-8-sig")
    print(f"\nSalidas en {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parte", choices=["A", "B", "C"], required=True)
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--semillas-p2", type=int, default=50)
    ap.add_argument("--solo", nargs="+", choices=["umbral", "colchon", "anatomia"],
                    default=["umbral", "colchon", "anatomia"],
                    help="parte B: correr solo estos bloques (para repartirlos en procesos)")
    ap.add_argument("--umbrales", nargs="+", type=float, default=[0.90, 0.85, 0.80, 0.75, 0.70])
    ap.add_argument("--colchones", nargs="+", type=float, default=[0.90, 0.85, 0.80, 0.70])
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"  REVISION INDEPENDIENTE DE LOGO -- parte {args.parte}")
    print("=" * 78)
    if args.parte == "A":
        parte_a(args.salida, args.boot, args.semillas_p2)
    elif args.parte == "B":
        parte_b(args.salida, args.semillas_p2, tuple(args.solo), tuple(args.umbrales), tuple(args.colchones))
    else:
        parte_c(args.salida)


if __name__ == "__main__":
    main()
