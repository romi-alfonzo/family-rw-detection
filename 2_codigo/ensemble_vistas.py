#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ensemble_vistas.py -- ¿combinar las vistas como clasificadores SEPARADOS mejora sobre CONCATENARLAS?

DE DONDE SALE. La ultima capa de la cascada del frente de notas es un LinearSVC sobre la vista
"combinado" de vectorizador() (clasificador_notas_v2.py). "combinado" es un FeatureUnion: pega
en un unico espacio el TF-IDF de palabras (word 1-2, max 5000) y el de caracteres (char_wb 3-5,
max 5000). Es decir, las dos vistas nunca fueron dos clasificadores: son un solo vector.

EL PROBLEMA CONOCIDO DE CONCATENAR. En un espacio concatenado el peso relativo de cada bloque
no se elige: lo fija cuantas dimensiones activas aporta cada uno al producto interno. La vista
que mas caracteristicas enciende por documento domina de hecho. Nadie decidio ese peso; salio
de pegar. La alternativa clasica es entrenar UN clasificador POR VISTA y combinar sus
decisiones de forma explicita, con una regla escrita y auditable.

POR QUE ES UNA PREGUNTA DISTINTA DE LAS QUE YA SE DESCARTARON. El proyecto ya midio en negativo
la busqueda de hiperparametros (840 configuraciones) y los embeddings multilingues
(paraphrase-multilingual-MiniLM-L12-v2: solo empeora -0,0733; concatenado +0,0092 no adoptado).
Las dos cambiaban la REPRESENTACION. Esto no toca la representacion: usa exactamente las mismas
dos vistas TF-IDF y cambia COMO SE AGREGAN LAS DECISIONES. Es una pregunta de agregacion, no de
rasgos, y es la unica de esa familia que nunca se probo.

EL PRIOR, DICHO SIN MAQUILLAJE. Tres intentos de mejorar la capa de texto dieron negativo
(hiperparametros, abstraccion de marcadores, embeddings) y un cuarto -- la cascada jerarquica
por linaje, 2026-09-28 -- dio -0,0011 de exactitud con IC que cruza el cero. Lo mas probable es
que este tambien de nulo. El preregistro esta escrito para reflejar ESE prior: la prediccion
principal es que NO mejora, y lo que se pone a prueba es esa prediccion. Si el ensemble gana,
la prediccion falla y hay que decirlo; si no gana, se suma un negativo bien medido a un
argumento que ya converge.

LO QUE SE PRUEBA (todo entrenado SOLO con el pliegue de entrenamiento):
  Vistas base: "palabras", "caracteres" y "combinado", cada una con su propia LinearSVC.
  Normalizacion por vista: la matriz decision_function se lleva a z con la media y el desvio
  calculados sobre el decision_function del PROPIO PLIEGUE DE ENTRENAMIENTO. (Restar una
  constante y dividir por un escalar positivo no cambia el argmax de una vista aislada: lo
  unico que cambia es la ESCALA RELATIVA ENTRE VISTAS, que es justo lo que hay que igualar.
  Normalizar con estadisticos del pliegue de prueba seria transductivo y no se hace.)

  6 formas de combinar -- 3 reglas x 2 conjuntos de vistas:
    suma_2v / suma_3v   suma de decision_function normalizada, pesos iguales.
    voto_2v / voto_3v   voto por mayoria del argmax de cada vista; los empates -- con 2 vistas
                        el empate es SIEMPRE que discrepan -- se resuelven por el margen mayor
                        (top1 - top2 de la fila normalizada de esa vista).
    pond_2v / pond_3v   suma ponderada, pesos elegidos por VALIDACION INTERNA dentro del
                        pliegue de entrenamiento (ver abajo).
  "2v" = {palabras, caracteres}: es la alternativa PURA a concatenar, sin usar el concatenado.
  "3v" = {palabras, caracteres, combinado}: el ensemble practico, y el unico que permite que el
         voto por mayoria tenga tres votantes de verdad.

COMO SE ELIGEN LOS PESOS (el punto donde es facil hacer trampa). Dentro de CADA pliegue de
entrenamiento, y sin mirar nunca el pliegue de prueba:
  1. Se parte el pliegue de entrenamiento en 2 pliegues internos con el MISMO split_p2bal
     (corte por plantilla), con un rng propio (90_000 + 100*semilla + pliegue) para no tocar la
     particion externa.
  2. Se entrenan las 3 vistas en cada mitad interna y se predice la otra: cada nota del pliegue
     de entrenamiento recibe una prediccion interna por cada juego de pesos de la rejilla.
  3. Rejilla: paso 0,1 sobre el simplejo -- 11 combinaciones para 2 vistas, 66 para 3.
  4. Gana el juego de pesos con mayor macro-F1 de validacion interna (etiquetas presentes en el
     pliegue de entrenamiento). Empate -> el mas cercano al uniforme; si persiste -> el primero
     de la rejilla. Es determinista.
  5. Recien entonces se reentrenan las vistas con TODO el pliegue de entrenamiento y se aplican
     esos pesos al pliegue de prueba.
Los pesos elegidos se guardan y se reportan promediados: si salieran siempre uniformes, el
ponderado colapsa al de suma y hay que decirlo.

=============================================================================================
PREREGISTRO -- escrito y COMMITEADO ANTES de correr (2026-09-28).

H1. PUERTA DE ENTRADA. La vista "combinado" sola reproduce las dos cifras de cabecera:
    capa de texto macro-F1 0,6551 y exactitud 0,7191; cascada macro-F1 0,7417 y exactitud
    0,8123 (tolerancia 0,01, fuente 4_resultados/_log_p2bal_149.txt). Si no, ABORTA y no se
    reporta nada.

H2. LA PREDICCION PRINCIPAL, Y ES NEGATIVA. NINGUNA de las 6 formas de combinar supera a la
    cascada con "combinado" (exactitud 0,8123) con Delta pareado por semilla cuyo IC 95 %
    CORREGIDO POR BONFERRONI (alfa 0,05/6 = 0,00833) excluya el cero. Se falsea si alguna lo
    logra. La cascada es la que decide, porque el sistema es la cascada.

H3. CUANTITATIVA Y FALSABLE. El Delta de exactitud de la cascada de las 6 variantes cae dentro
    de [-0,020; +0,020]. Se falsea si alguna variante se sale de esa banda en cualquier
    direccion. Es la prediccion de "la concatenacion no esta groseramente mal calibrada".

H4. Sobre la capa de TEXTO SOLA, el mejor ensemble no supera 0,6551 + 0,020 = 0,6751 de
    macro-F1. Se mide aparte de la cascada porque puede pasar que el ensemble mejore el texto y
    la cascada se lo coma (las reglas ya resuelven lo que resuelven, y ahi el texto no decide).

H5. LA PRUEBA DIRECTA DE LA PREGUNTA. El ensemble de 2 vistas -- suma_2v, voto_2v, pond_2v, que
    NO usan el concatenado -- NO supera a "combinado" en macro-F1 de texto. Si la concatenacion
    estuviera mal ponderada, es exactamente aca donde tendria que verse: mismas dos vistas,
    mismo clasificador, unica diferencia el modo de agregar. Si H5 falla, la concatenacion SI
    era el problema.

H6. DIAGNOSTICO DE QUIEN DOMINA. La vista "caracteres" sola da mas macro-F1 de texto que
    "palabras" sola, y el peso medio que la validacion interna le asigna a "caracteres" en
    pond_2v es >= 0,5. Es la lectura de a cual de las dos favorece de hecho la concatenacion.

H7. INTERPRETACION FIJADA DE ANTEMANO, para no racionalizarla despues:
    - Si H2 falla (alguna variante gana con Bonferroni): la concatenacion SI estaba mal
      calibrada; se adopta esa variante y se documenta el peso.
    - Si H2 y H5 se cumplen: agregar las decisiones explicitamente no cambia nada. Para ESTE
      corpus la concatenacion es una forma razonable de combinar las dos vistas, y el margen
      que quedaba por ganar ahi era nulo. Cuarto negativo convergente del frente de notas.
    - Si H2 se cumple pero H4 falla (mejora el texto y no la cascada): la ganancia del ensemble
      cae sobre notas que las reglas de la cascada ya resolvian. Se reporta como tal y NO se
      adopta, porque el sistema es la cascada.

NOTA DE HONESTIDAD. Se prueban 6 variantes; para ADOPTAR alguna se exige Bonferroni sobre 6
(alfa efectivo 0,00833 a dos colas). El barrido completo se reporta igual, con el IC 95 % sin
corregir al lado del corregido, para que se vea que la correccion se aplico y no se eligio la
mejor de 6 a posteriori. Las vistas NO se eligieron mirando resultados: son las tres que
vectorizador() ya ofrecia antes de este experimento.
=============================================================================================

Uso:  python ensemble_vistas.py [--n-semillas 50] [--salida CARPETA]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
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
OUT_DEF = RAIZ / "4_resultados" / "resultados_ensemble_vistas_149"

# cifras de cabecera P2bal, 149 notas, 50 semillas (fuente: 4_resultados/_log_p2bal_149.txt)
CANON_TXT_F1, CANON_TXT_ACC = 0.6551, 0.7191
CANON_CAS_F1, CANON_CAS_ACC = 0.7417, 0.8123
TOL = 0.01

VISTAS = ["palabras", "caracteres", "combinado"]
V2 = ["palabras", "caracteres"]
V3 = ["palabras", "caracteres", "combinado"]
ENSEMBLES = ["suma_2v", "voto_2v", "pond_2v", "suma_3v", "voto_3v", "pond_3v"]
N_COMPARACIONES = len(ENSEMBLES)          # Bonferroni: alfa 0,05 / 6
ALFA = 0.05
PASO_REJILLA = 0.1


# ------------------------------------------------------------------ utilidades
def ic(v, alfa=ALFA):
    """Media e IC de t-Student al nivel 1-alfa (bilateral)."""
    v = np.asarray(v, float)
    m, nn = float(np.mean(v)), len(v)
    s = float(np.std(v, ddof=1)) if nn > 1 else 0.0
    h = t_dist.ppf(1 - alfa / 2, nn - 1) * (s / np.sqrt(nn)) if nn > 1 else 0.0
    return m, m - h, m + h


def a2d(D):
    """decision_function devuelve (n,) si hay 2 clases; se lleva siempre a (n, n_clases)."""
    D = np.asarray(D, float)
    return np.column_stack([-D, D]) if D.ndim == 1 else D


def normaliza(D_tr, D_te):
    """z por vista con estadisticos del PLIEGUE DE ENTRENAMIENTO (nunca del de prueba)."""
    mu, sd = float(D_tr.mean()), float(D_tr.std())
    return (D_te - mu) / (sd if sd > 0 else 1.0)


def margen(D):
    """top1 - top2 de cada fila: cuanta ventaja le saca la clase ganadora a la segunda."""
    P = np.partition(D, -2, axis=1)
    return P[:, -1] - P[:, -2]


def rejilla(n_vistas, paso=PASO_REJILLA):
    """Pesos no negativos que suman 1, en pasos de `paso`. 11 combinaciones con 2 vistas, 66 con 3."""
    k = int(round(1 / paso))
    if n_vistas == 2:
        return [(i / k, (k - i) / k) for i in range(k + 1)]
    out = []
    for i in range(k + 1):
        for j in range(k + 1 - i):
            out.append((i / k, j / k, (k - i - j) / k))
    return out


def dist_uniforme(w):
    u = 1.0 / len(w)
    return float(sum((x - u) ** 2 for x in w))


# ------------------------------------------------------- entrenamiento y reglas
def entrena_vistas(tr, te, textos_arr, y, s):
    """Una LinearSVC por vista, entrenada SOLO con tr. Devuelve {vista: D_te normalizada}, clases."""
    Ds, clases = {}, None
    for v in VISTAS:
        vec = vectorizador(v)
        Xtr = vec.fit_transform(textos_arr[tr])
        Xte = vec.transform(textos_arr[te])
        clf = obtener_modelos(s)["LinearSVC"]
        clf.fit(Xtr, y[tr])
        Ds[v] = normaliza(a2d(clf.decision_function(Xtr)), a2d(clf.decision_function(Xte)))
        if clases is None:
            clases = clf.classes_
        elif not np.array_equal(clases, clf.classes_):
            sys.exit("ERROR: las vistas no comparten el orden de clases; no se pueden sumar.")
    return Ds, clases


def pred_suma(Ds, vistas, pesos, clases):
    S = np.zeros_like(Ds[vistas[0]])
    for v, w in zip(vistas, pesos):
        S = S + w * Ds[v]
    return clases[S.argmax(axis=1)]


def pred_voto(Ds, vistas, clases):
    """Mayoria del argmax de cada vista; empates por el margen mayor entre las vistas que votaron
    esa clase. Con 2 vistas TODA discrepancia es empate, asi que la regla la decide el margen."""
    votos = np.array([Ds[v].argmax(axis=1) for v in vistas])
    marg = np.array([margen(Ds[v]) for v in vistas])
    n = votos.shape[1]
    out = np.empty(n, dtype=object)
    for i in range(n):
        cnt = Counter(votos[:, i])
        top = max(cnt.values())
        cands = [c for c, k in cnt.items() if k == top]
        if len(cands) == 1:
            out[i] = clases[cands[0]]
            continue
        mejor, mejor_m = cands[0], -np.inf
        for c in cands:
            m = max(marg[j, i] for j in range(len(vistas)) if votos[j, i] == c)
            if m > mejor_m:
                mejor, mejor_m = c, m
        out[i] = clases[mejor]
    return out


def elige_pesos(tr, textos_arr, y, grupos, familias, s, rng):
    """Pesos por VALIDACION INTERNA dentro del pliegue de entrenamiento. No ve el de prueba.

    Devuelve (pesos_2v, pesos_3v, f1_interno_2v, f1_interno_3v).
    """
    n_tr = len(tr)
    rej2, rej3 = rejilla(2), rejilla(3)
    pred2 = [np.empty(n_tr, dtype=object) for _ in rej2]
    pred3 = [np.empty(n_tr, dtype=object) for _ in rej3]
    for itr, ite in split_p2bal(y[tr], grupos[tr], familias, rng):
        Ds, clases = entrena_vistas(tr[itr], tr[ite], textos_arr, y, s)
        for j, w in enumerate(rej2):
            pred2[j][ite] = pred_suma(Ds, V2, w, clases)
        for j, w in enumerate(rej3):
            pred3[j][ite] = pred_suma(Ds, V3, w, clases)
    lab = np.unique(y[tr])

    def mejor(rej, preds):
        punt = [f1_score(y[tr], p, average="macro", labels=lab, zero_division=0) for p in preds]
        top = max(punt)
        cands = [j for j, x in enumerate(punt) if x == top]
        j = min(cands, key=lambda j: (dist_uniforme(rej[j]), j))
        return rej[j], top

    w2, f2 = mejor(rej2, pred2)
    w3, f3 = mejor(rej3, pred3)
    return w2, w3, f2, f3


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=50)
    ap.add_argument("--sin-puerta", action="store_true",
                    help="NO usar salvo depuracion: saltea la puerta de entrada H1.")
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)
    S = args.n_semillas

    print("=" * 78)
    print("  ENSEMBLE DE VISTAS: ¿agregar decisiones mejora sobre CONCATENAR caracteristicas?")
    print("=" * 78)
    textos, textos_arr, y, archivos, grupos, iocs, nombres_nota = cargar_todo()
    familias = np.unique(y)
    n = len(y)
    print(f"Notas: {n} | Familias: {len(familias)} | Plantillas: {len(set(grupos))}")
    print(f"Vistas: {VISTAS}")
    print(f"Formas de combinar: {ENSEMBLES}  (Bonferroni sobre {N_COMPARACIONES})")
    print(f"Rejilla de pesos: {len(rejilla(2))} combinaciones (2 vistas), "
          f"{len(rejilla(3))} (3 vistas), paso {PASO_REJILLA}")
    print(f"Semillas: {S}\n")

    sistemas = VISTAS + ENSEMBLES
    p_txt = {k: np.empty((S, n), dtype=object) for k in sistemas}
    p_cas = {k: np.empty((S, n), dtype=object) for k in sistemas}
    pesos_log = []

    print("Evaluando ...")
    for s in range(S):
        rng = np.random.default_rng(20_000 + s)
        for kf, (tr, te) in enumerate(split_p2bal(y, grupos, familias, rng)):
            # ---- pesos por validacion interna: SOLO con el pliegue de entrenamiento ----
            rng_int = np.random.default_rng(90_000 + 100 * s + kf)
            w2, w3, f2, f3 = elige_pesos(tr, textos_arr, y, grupos, familias, s, rng_int)
            pesos_log.append(dict(semilla=s, pliegue=kf,
                                  w2_palabras=w2[0], w2_caracteres=w2[1], f1_interno_2v=round(f2, 4),
                                  w3_palabras=w3[0], w3_caracteres=w3[1], w3_combinado=w3[2],
                                  f1_interno_3v=round(f3, 4)))

            # ---- las 3 vistas, reentrenadas con TODO el pliegue de entrenamiento ----
            Ds, clases = entrena_vistas(tr, te, textos_arr, y, s)

            pred = {v: clases[Ds[v].argmax(axis=1)] for v in VISTAS}
            u2, u3 = (0.5, 0.5), (1 / 3, 1 / 3, 1 / 3)
            pred["suma_2v"] = pred_suma(Ds, V2, u2, clases)
            pred["suma_3v"] = pred_suma(Ds, V3, u3, clases)
            pred["voto_2v"] = pred_voto(Ds, V2, clases)
            pred["voto_3v"] = pred_voto(Ds, V3, clases)
            pred["pond_2v"] = pred_suma(Ds, V2, w2, clases)
            pred["pond_3v"] = pred_suma(Ds, V3, w3, clases)

            # ---- capa de reglas de la cascada: identica para todos los sistemas ----
            d = dicc_privados(tr, iocs, nombres_nota, y)
            reglas = [regla(i, d, iocs, nombres_nota) for i in te]
            for k in sistemas:
                for j, i in enumerate(te):
                    p_txt[k][s, i] = pred[k][j]
                    p_cas[k][s, i] = pred[k][j] if reglas[j] is None else reglas[j]
        if (s + 1) % 10 == 0:
            print(f"  {s+1}/{S} semillas")

    # ------------------------------------------------ metricas por semilla
    def serie(P, fn):
        return np.array([fn(y, P[s]) for s in range(S)])

    f1m = lambda yy, pp: f1_score(yy, pp, average="macro", labels=familias, zero_division=0)
    M = {}
    for k in sistemas:
        M[k] = dict(txt_f1=serie(p_txt[k], f1m), txt_acc=serie(p_txt[k], accuracy_score),
                    cas_f1=serie(p_cas[k], f1m), cas_acc=serie(p_cas[k], accuracy_score),
                    cas_bal=serie(p_cas[k], balanced_accuracy_score),
                    cas_mcc=serie(p_cas[k], matthews_corrcoef))

    # ------------------------------------------------ H1: puerta de entrada
    b = M["combinado"]
    print("\n" + "-" * 78)
    print("  H1 -- PUERTA DE ENTRADA (la vista 'combinado' sola tiene que dar la cabecera)")
    print("-" * 78)
    print(f"  texto  : macro-F1 {b['txt_f1'].mean():.4f} vs {CANON_TXT_F1} | "
          f"exactitud {b['txt_acc'].mean():.4f} vs {CANON_TXT_ACC}")
    print(f"  cascada: macro-F1 {b['cas_f1'].mean():.4f} vs {CANON_CAS_F1} | "
          f"exactitud {b['cas_acc'].mean():.4f} vs {CANON_CAS_ACC}")
    ok = (abs(b["txt_f1"].mean() - CANON_TXT_F1) <= TOL and
          abs(b["txt_acc"].mean() - CANON_TXT_ACC) <= TOL and
          abs(b["cas_f1"].mean() - CANON_CAS_F1) <= TOL and
          abs(b["cas_acc"].mean() - CANON_CAS_ACC) <= TOL)
    if not ok and not args.sin_puerta:
        sys.exit("ABORTADO (H1): 'combinado' no reproduce la cabecera. No se reporta nada.")
    print("  OK\n" if ok else "  FUERA DE TOLERANCIA (se sigue por --sin-puerta)\n")

    # ------------------------------------------------ tabla de niveles
    filas = [dict(sistema=k,
                  tipo=("vista sola" if k in VISTAS else "ensemble"),
                  txt_macro_f1=round(float(M[k]["txt_f1"].mean()), 4),
                  txt_exactitud=round(float(M[k]["txt_acc"].mean()), 4),
                  cas_macro_f1=round(float(M[k]["cas_f1"].mean()), 4),
                  cas_exactitud=round(float(M[k]["cas_acc"].mean()), 4),
                  cas_exact_balanceada=round(float(M[k]["cas_bal"].mean()), 4),
                  cas_mcc=round(float(M[k]["cas_mcc"].mean()), 4)) for k in sistemas]
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "ensemble_resumen.csv", index=False, encoding="utf-8-sig")
    print("=== NIVELES (149 notas, P2bal, 50 semillas). Referencia = vista 'combinado' ===")
    print(df.to_string(index=False))

    # ------------------------------------------------ deltas pareados vs 'combinado'
    alfa_b = ALFA / N_COMPARACIONES
    dfilas = []
    for k in sistemas:
        if k == "combinado":
            continue
        for capa, met in (("texto", "txt"), ("cascada", "cas")):
            for nom_m, suf in (("macro_f1", "f1"), ("exactitud", "acc")):
                d = M[k][f"{met}_{suf}"] - b[f"{met}_{suf}"]
                m, lo, hi = ic(d)
                _, lob, hib = ic(d, alfa_b)
                dfilas.append(dict(
                    sistema=k, capa=capa, metrica=nom_m, delta=round(float(m), 4),
                    ic95_bajo=round(float(lo), 4), ic95_alto=round(float(hi), 4),
                    bonf_bajo=round(float(lob), 4), bonf_alto=round(float(hib), 4),
                    semillas_positivas=f"{int((d > 0).sum())}/{S}",
                    sig_95="SI" if lo > 0 or hi < 0 else "no",
                    sig_bonferroni="SI" if lob > 0 or hib < 0 else "no"))
    dd = pd.DataFrame(dfilas)
    dd.to_csv(OUT / "ensemble_deltas.csv", index=False, encoding="utf-8-sig")

    for capa in ("texto", "cascada"):
        base_f1 = CANON_TXT_F1 if capa == "texto" else CANON_CAS_F1
        base_ac = CANON_TXT_ACC if capa == "texto" else CANON_CAS_ACC
        print(f"\n=== DELTA PAREADO POR SEMILLA vs 'combinado' -- CAPA {capa.upper()} "
              f"(base {base_f1} macro-F1 / {base_ac} exactitud) ===")
        print(f"{'sistema':<12} {'metrica':<10} {'delta':>8}  {'IC 95 %':>22}  "
              f"{'IC Bonferroni':>22}  {'sem+':>6}  adopta")
        for r in dfilas:
            if r["capa"] != capa:
                continue
            print(f"{r['sistema']:<12} {r['metrica']:<10} {r['delta']:>+8.4f}  "
                  f"[{r['ic95_bajo']:+.4f}; {r['ic95_alto']:+.4f}]  "
                  f"[{r['bonf_bajo']:+.4f}; {r['bonf_alto']:+.4f}]  "
                  f"{r['semillas_positivas']:>6}  "
                  f"{'SI' if (r['sig_bonferroni'] == 'SI' and r['delta'] > 0) else 'no'}")

    # ------------------------------------------------ pesos elegidos
    dp = pd.DataFrame(pesos_log)
    dp.to_csv(OUT / "ensemble_pesos.csv", index=False, encoding="utf-8-sig")
    print("\n=== PESOS ELEGIDOS POR VALIDACION INTERNA (media sobre 50 semillas x 2 pliegues) ===")
    print(f"  pond_2v: palabras {dp.w2_palabras.mean():.3f} · caracteres {dp.w2_caracteres.mean():.3f}"
          f"   (macro-F1 interno medio {dp.f1_interno_2v.mean():.4f})")
    print(f"  pond_3v: palabras {dp.w3_palabras.mean():.3f} · caracteres {dp.w3_caracteres.mean():.3f}"
          f" · combinado {dp.w3_combinado.mean():.3f}"
          f"   (macro-F1 interno medio {dp.f1_interno_3v.mean():.4f})")
    unif2 = float((np.abs(dp.w2_palabras - 0.5) < 1e-9).mean())
    print(f"  pond_2v exactamente uniforme en {unif2 * 100:.1f} % de los pliegues "
          f"(si fuera 100 %, el ponderado colapsa a suma_2v)")
    print("  Reparto de w(caracteres) en pond_2v:")
    for w, c in sorted(Counter(np.round(dp.w2_caracteres, 3)).items()):
        print(f"    {w:.1f} -> {c:>3} pliegues de {len(dp)}")

    # ------------------------------------------------ por familia (mejor ensemble en cascada)
    mejor_ens = max(ENSEMBLES, key=lambda k: M[k]["cas_acc"].mean())
    ffilas = []
    for f in familias:
        idx = np.where(y == f)[0]
        ffilas.append(dict(
            familia=f, n_notas=len(idx),
            combinado=round(float(np.mean([(p_cas["combinado"][s, idx] == f).mean()
                                           for s in range(S)])), 4),
            ensemble=round(float(np.mean([(p_cas[mejor_ens][s, idx] == f).mean()
                                          for s in range(S)])), 4)))
    dff = pd.DataFrame(ffilas)
    dff["delta"] = (dff.ensemble - dff.combinado).round(4)
    dff = dff.sort_values("delta", ascending=False)
    dff.to_csv(OUT / "ensemble_por_familia.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== POR FAMILIA (cascada): '{mejor_ens}' contra 'combinado' -- "
          f"las 5 que mas suben y las 5 que mas bajan ===")
    print(pd.concat([dff.head(5), dff.tail(5)]).to_string(index=False))

    # ------------------------------------------------ veredicto del preregistro
    def get(k, capa, met):
        return next(r for r in dfilas if r["sistema"] == k and r["capa"] == capa
                    and r["metrica"] == met)

    gana_bonf = [k for k in ENSEMBLES
                 if get(k, "cascada", "exactitud")["sig_bonferroni"] == "SI"
                 and get(k, "cascada", "exactitud")["delta"] > 0]
    h2 = not gana_bonf
    d_cas = {k: get(k, "cascada", "exactitud")["delta"] for k in ENSEMBLES}
    fuera = [k for k, v in d_cas.items() if abs(v) > 0.020]
    h3 = not fuera
    mejor_txt = max(ENSEMBLES, key=lambda k: M[k]["txt_f1"].mean())
    h4 = M[mejor_txt]["txt_f1"].mean() <= CANON_TXT_F1 + 0.020
    ens2 = [k for k in ENSEMBLES if k.endswith("_2v")]
    gana2 = [k for k in ens2 if get(k, "texto", "macro_f1")["sig_95"] == "SI"
             and get(k, "texto", "macro_f1")["delta"] > 0]
    h5 = not gana2
    car_gt_pal = M["caracteres"]["txt_f1"].mean() > M["palabras"]["txt_f1"].mean()
    h6 = car_gt_pal and dp.w2_caracteres.mean() >= 0.5

    print("\n" + "=" * 78)
    print("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    print("=" * 78)
    chk = [
        ("H1 puerta de entrada ('combinado' = cabecera)", ok,
         f"txt {b['txt_f1'].mean():.4f}/{b['txt_acc'].mean():.4f} · "
         f"cas {b['cas_f1'].mean():.4f}/{b['cas_acc'].mean():.4f}"),
        ("H2 NINGUN ensemble gana la cascada con Bonferroni", h2,
         "ninguno gana" if h2 else f"ganan: {', '.join(gana_bonf)}"),
        ("H3 todos los Delta de cascada en [-0,020; +0,020]", h3,
         (f"peor |D| {max(abs(v) for v in d_cas.values()):.4f}" if h3
          else f"se salen: {', '.join(fuera)}")),
        ("H4 el mejor ensemble de TEXTO no pasa 0,6751", h4,
         f"{mejor_txt} {M[mejor_txt]['txt_f1'].mean():.4f}"),
        ("H5 el ensemble de 2 vistas no supera a 'combinado' (texto)", h5,
         "ninguno supera" if h5 else f"superan: {', '.join(gana2)}"),
        ("H6 caracteres > palabras y w(caracteres) >= 0,5", h6,
         f"car {M['caracteres']['txt_f1'].mean():.4f} vs pal "
         f"{M['palabras']['txt_f1'].mean():.4f} · w_car {dp.w2_caracteres.mean():.3f}"),
    ]
    for nombre, cumple, det in chk:
        print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<52} {det}")

    print("\n  LECTURA SEGUN H7 (fijada antes de correr):")
    if not h2:
        print("    ALGUN ENSEMBLE GANA con Bonferroni: la concatenacion SI estaba mal calibrada.")
        print(f"    Variantes adoptables: {', '.join(gana_bonf)}. Documentar el peso y adoptar.")
    elif h5 and h4:
        print("    NADA CAMBIA. Agregar las decisiones de forma explicita no mejora ni la capa de")
        print("    texto ni la cascada. Para este corpus la CONCATENACION ya es una forma")
        print("    razonable de combinar palabras y caracteres: el peso implicito que le sale a")
        print("    la vista dominante no estaba costando nada medible. Cuarto negativo")
        print("    convergente del frente de notas -- resultado, no fracaso. NO ADOPTAR.")
    elif not h4:
        print("    El ensemble MEJORA LA CAPA DE TEXTO pero la cascada no lo recoge: la ganancia")
        print("    cae sobre notas que las reglas (IOCs privados + nombre) ya resolvian. Se")
        print("    reporta el efecto sobre el texto y NO se adopta: el sistema es la cascada.")
    else:
        print("    Resultado mixto: revisar fila por fila antes de concluir nada.")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
