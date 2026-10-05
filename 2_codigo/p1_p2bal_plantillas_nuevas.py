#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p1_p2bal_plantillas_nuevas.py -- la cascada con PLANTILLAS NUEVAS de tria.ge en el entrenamiento.

QUÉ SE QUIERE SABER. Si agregar al ENTRENAMIENTO plantillas que el corpus no tiene, de las mismas
familias y tomadas de tria.ge, mejora la cascada (reglas -> texto). La PRUEBA son siempre las 149
notas del corpus canónico (30 familias), así que la comparación contra V0 es pareada, semilla a
semilla, y sobre la misma base.
Motivo, con fuente:
  - bajo P2bal, la curva de aprendizaje del texto solo sube de macro-F1 0,5759 a 0,6599 al pasar de
    1 a 2 plantillas por familia (4_resultados/_log_curva_p2bal_149.txt, tope por plantillas);
  - bajo P1, el 84,5 % del error de la cascada (14,28 de 16,90 errores por semilla) está en las 76
    notas que son la única de su plantilla (4_resultados/_log_diagnostico_ampliacion_p1.txt).
Las copias de plantillas conocidas (p1_copias_sin_apuntar.py) no llegan a esas notas: este
experimento prueba con plantillas nuevas.

DATOS. No se imprime el texto de ninguna nota y nada de 3_datos ni de 4_resultados se commitea.
  - Fuente: 3_datos/fuentes_notas/triage_sin_apuntar_2026-10/<FAMILIA>/, la recolección sin apuntar
    de copias_triage.py --todas. El rótulo es la etiqueta de familia de tria.ge (la carpeta).
  - Elegibles: filas de 4_resultados/resultados_copias_sin_apuntar/inventario.csv con fuente ==
    "triage_sin_apuntar" y estado == "TEXTO_NUEVO". Por definición ninguna nota del corpus supera
    coseno char 0,90 con ellas: no son copias de plantillas de prueba. El texto sale de
    extractor_notas.extraer_texto, como en el corpus.
  - Exclusión 1, informe con varias etiquetas: el id del informe es el prefijo del nombre antes de
    "__" (\\d{6}-[a-z0-9]{10}). Si el informe tiene archivos en más de una carpeta de familia de la
    recolección (se miran TODOS sus archivos, no solo las elegibles), sus notas se excluyen: el
    rótulo sería ambiguo.
  - Plantillas nuevas: las elegibles que quedan se agrupan ENTRE SÍ con la regla del corpus
    (agrupar_neardups de clasificador_notas_v2: TFIDF_CHAR, coseno > UMBRAL_NEARDUP = 0,90), en
    orden de ruta.
  - Exclusión 2, plantilla mixta. Decisión de diseño tomada al preparar, ANTES de medir: una
    plantilla nueva con notas de más de una carpeta de familia se excluye entera, por la misma razón
    que la exclusión 1. Sin esta regla, «la primera por ruta ordenada» rotularía la plantilla por el
    orden alfabético de las carpetas: la nota de Cerber «_R_E_A_D___T_H_I_S___» (16 notas en CERBER,
    1 en BADRABBIT y 1 en NETWALKER) quedaría como BADRABBIT, y un mismo DECRYPT_YOUR_FILES.HTML
    que aparece en 7 carpetas quedaría como CERBER.
  - N-rep (variante PRIMARIA): una representante por plantilla nueva admitida, la primera por ruta
    ordenada. N-all (secundaria): todas las notas de las plantillas nuevas admitidas.
  Conteo al preparar (2026-10-05): 275 elegibles; 24 excluidas por informe con varias etiquetas
  (11 informes; en toda la recolección hay 12 así); 251 notas en 64 plantillas nuevas; 4 plantillas
  mixtas (45 notas) excluidas. Quedan 60 plantillas nuevas (N-rep) y 206 notas (N-all), en 23
  familias; CERBER, CLOP, CRYPTOLOCKER, GANDCRAB, LORENZ, NOTPETYA y WANNACRY no reciben ninguna.
  LIMITACIÓN que no se filtra: el rótulo es el de tria.ge. Hay plantillas admitidas cuya nota más
  parecida del corpus es de OTRA familia, algunas con coseno cercano a 0,90 (rótulo posiblemente
  equivocado), y archivos que por el nombre podrían no ser notas. No se filtran: elegir a mano qué
  entra sería apuntar. El script las cuenta y las deja en el CSV de notas nuevas.

MEDICIÓN. Las extras entran SOLO al entrenamiento, igual que las copias de p1_copias_controladas.py:
al ajuste del TF-IDF, al LinearSVC y al diccionario de marcadores privados (extraer_marcadores), con
nombre de archivo None (no está auditado, así que la regla por nombre no cambia). Nunca se prueban.
  P1     StratifiedKFold de 2 pliegues con mezcla, random_state = s, y la cascada de
         p1_copias_controladas.py. Variantes V0 (sin extras), N-rep y N-all.
  P2bal  split_p2bal de protocolo_p2bal.py con rng = default_rng(20000 + s) y la cascada de
         protocolo_logo.evaluar, la de la cifra canónica de _log_p2bal_149.txt. Las extras entran a
         todos los pliegues de entrenamiento. Se informa también el texto solo.
  50 semillas (--semillas N). Exactitud y macro-F1 con labels = las 30 familias. Por variante:
  media ± desvío; delta pareado contra V0 con IC 95 % t sobre semillas; semillas en que mejora y
  empeora; cobertura de la capa de reglas; en P2bal, F1 por familia.

PREREGISTRO (commiteado y pusheado ANTES de la corrida completa, 2026-10-05). Se reporta lo que dé.
  PN-0  PUERTAS. Con 50 semillas, V0 reproduce las cifras canónicas (tolerancia 0,0005) o el script
        ABORTA sin reportar nada:
          P1 cascada 0,8866 / 0,8592 (exactitud / macro-F1; _log_m3_149_P1.txt, umbral 0, y la
          puerta de p1_copias_controladas.py);
          P2bal cascada 0,8123 / 0,7417 y texto solo 0,7191 / 0,6551 (_log_p2bal_149.txt).
  PN-1  P2bal, N-rep: el macro-F1 de la cascada sube contra V0, con el IC 95 % de la diferencia
        pareada por encima de cero.
  PN-2  P1, N-rep: la exactitud de la cascada sube contra V0, con el IC 95 % por encima de cero.
  PN-3  P1, N-rep: las notas que son la única de su plantilla en el corpus (76) ganan más exactitud
        media que las demás (73). Ganancia = acierto N-rep - acierto V0, promediado sobre notas y
        semillas. El criterio es la comparación de las dos medias; el IC 95 % t de la diferencia
        (sobre semillas) se informa, pero no forma parte del criterio.
  PN-4  Control de riesgo, P2bal, N-rep: ninguna familia pierde más de 0,10 de F1 de la cascada
        (media sobre semillas) contra V0. Vigila el sesgo hacia las familias con muchas notas
        nuevas, como MAZE.
N-all se informa con las mismas cifras, sin predicción propia.

Uso:  python p1_p2bal_plantillas_nuevas.py [--semillas 50]
Salida: 4_resultados/resultados_plantillas_nuevas/. Con --semillas distinto de 50 la salida va a la
subcarpeta prueba_<N>_semillas: las puertas no se comparan y el veredicto no vale (prueba de humo).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from clasificador_notas_v2 import (CORPUS_DIR, MIN_CHARS_NOTA, N_FOLDS, UMBRAL_NEARDUP,  # noqa: E402
                                   agrupar_neardups, cargar_corpus, obtener_modelos, vectorizador)
from extractor_notas import extraer_texto  # noqa: E402
from grafo_marcadores import extraer_marcadores  # noqa: E402
# Las mismas funciones que usa la cifra canónica de P2bal (protocolo_logo.evaluar).
# abstencion_notas.py, que usa p1_copias_controladas.py, tiene copias idénticas de las tres.
from protocolo_logo import cargar_nombres, dicc_privados, regla  # noqa: E402
from protocolo_p2bal import split_p2bal  # noqa: E402

RAIZ = AQUI.parent
DATOS = RAIZ / "3_datos"
FUENTE = DATOS / "fuentes_notas" / "triage_sin_apuntar_2026-10"
INVENTARIO = RAIZ / "4_resultados" / "resultados_copias_sin_apuntar" / "inventario.csv"
SALIDA = RAIZ / "4_resultados" / "resultados_plantillas_nuevas"
ID_INFORME = re.compile(r"^(\d{6}-[a-z0-9]{10})__")
VARIANTES = ("V0", "N-rep", "N-all")
PROTOCOLOS = ("P1", "P2bal")
CAPAS = ("cascada", "texto")
N_CANON = 50
TOL = 0.0005
# PN-0: (exactitud, macro-F1) de V0 con 50 semillas. P1: _log_m3_149_P1.txt (umbral 0) y puerta de
# p1_copias_controladas.py. P2bal: _log_p2bal_149.txt (protocolo_p2bal.py).
PUERTAS = {("P1", "cascada"): (0.8866, 0.8592),
           ("P2bal", "cascada"): (0.8123, 0.7417),
           ("P2bal", "texto"): (0.7191, 0.6551)}
VECINA_ALTA = 0.80   # solo para el diagnóstico de rótulo dudoso; no filtra nada


def ic(d):
    """Media e IC 95 % t sobre semillas (sin IC con una sola semilla)."""
    d = np.asarray(d, float)
    if len(d) < 2:
        return float(d.mean()), float("nan"), float("nan")
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - h), float(d.mean() + h)


def sd(d):
    return float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")


def cargar_plantillas_nuevas():
    """Elegibles, las dos exclusiones y las plantillas nuevas. No imprime texto de notas.

    Devuelve (filas, quedan, textos): filas = todas las elegibles con su motivo de exclusión;
    quedan = las que pasan la exclusión 1, en orden de ruta, con su plantilla nueva y si son
    representantes; textos = el texto de cada una de quedan."""
    carpetas = defaultdict(set)              # informe -> carpetas de familia donde tiene archivos
    for p in FUENTE.rglob("*"):
        rel = p.relative_to(FUENTE)
        if p.is_file() and len(rel.parts) >= 2:
            m = ID_INFORME.match(p.name)
            if m:
                carpetas[m.group(1)].add(rel.parts[0])
    multi = {inf for inf, c in carpetas.items() if len(c) > 1}

    with open(INVENTARIO, encoding="utf-8") as fh:
        inv = [r for r in csv.DictReader(fh)
               if r["fuente"] == "triage_sin_apuntar" and r["estado"] == "TEXTO_NUEVO"]
    filas = []
    for r in inv:
        ruta = Path(r["ruta"].replace("\\", "/"))
        m = ID_INFORME.match(ruta.name)
        if m is None:
            sys.exit(f"ABORTADO: elegible sin id de informe en el nombre: {ruta.as_posix()}")
        filas.append(dict(ruta=ruta.as_posix(), familia=r["familia"], informe=m.group(1),
                          motivo="informe con varias etiquetas" if m.group(1) in multi else "",
                          plantilla_nueva="", representante=0, coseno_max_corpus=r["coseno_max"],
                          familia_vecina_corpus=r["familia_vecina"]))
    filas.sort(key=lambda f: f["ruta"])

    quedan = [f for f in filas if not f["motivo"]]
    textos = []
    for f in quedan:
        t, metodo = extraer_texto(DATOS / f["ruta"])
        if metodo.startswith("error") or len(t.strip()) < MIN_CHARS_NOTA:
            sys.exit(f"ABORTADO: elegible ilegible ({metodo}): {f['ruta']}")
        textos.append(t)
    g, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    fam = np.array([f["familia"] for f in quedan])
    mixtas = {k for k in set(g) if len(set(fam[g == k])) > 1}
    vistas = set()
    for f, k in zip(quedan, g):
        f["plantilla_nueva"] = int(k)
        if k in mixtas:
            f["motivo"] = "plantilla mixta"
        elif k not in vistas:
            f["representante"] = 1
            vistas.add(k)
    print(f"Elegibles (TEXTO_NUEVO de tria.ge sin apuntar): {len(filas)}")
    print(f"  excluidas por informe con varias etiquetas: "
          f"{sum(f['motivo'] == 'informe con varias etiquetas' for f in filas)} "
          f"({len({f['informe'] for f in filas if f['informe'] in multi})} informes)")
    print(f"  quedan {len(quedan)} notas en {len(set(g))} plantillas nuevas; "
          f"{len(mixtas)} plantillas mixtas excluidas "
          f"({sum(f['motivo'] == 'plantilla mixta' for f in quedan)} notas)")
    return filas, quedan, textos


def main():
    ap = argparse.ArgumentParser(description="Cascada con plantillas nuevas de tria.ge en el "
                                             "entrenamiento, bajo P1 y P2bal.")
    ap.add_argument("--semillas", type=int, default=N_CANON,
                    help=f"semillas (por omisión {N_CANON}; con otro valor las puertas no se comparan)")
    args = ap.parse_args()
    N = args.semillas
    if N < 1:
        sys.exit("--semillas tiene que ser 1 o más")
    canon = N == N_CANON
    salida = SALIDA if canon else SALIDA / f"prueba_{N}_semillas"
    t0 = time.time()

    print("=" * 78 + "\n  CASCADA CON PLANTILLAS NUEVAS DE TRIA.GE EN EL ENTRENAMIENTO\n" + "=" * 78)
    # --- corpus = prueba, igual que p1_copias_controladas.py y protocolo_p2bal.py ---------------
    textos, y, archivos, _ = cargar_corpus(CORPUS_DIR)
    grupos, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    textos_arr = np.array(textos, dtype=object)
    y, grupos = np.asarray(y), np.asarray(grupos)
    familias = np.unique(y)
    n = len(y)
    iocs = [set(extraer_marcadores(t)) for t in textos]
    nom = cargar_nombres()
    nombres_nota = [nom.get((f, Path(a).name)) for f, a in zip(y, archivos)]
    tam = Counter(grupos)
    unica = np.array([tam[g] == 1 for g in grupos])
    ppf = {f: len(set(grupos[y == f])) for f in familias}
    print(f"Corpus (prueba): {n} notas, {len(familias)} familias, {len(tam)} plantillas; "
          f"{int(unica.sum())} notas son la única de su plantilla. Semillas: {N}\n")

    # --- plantillas nuevas (solo entrenamiento) ---------------------------------------------------
    filas, quedan, textos_e = cargar_plantillas_nuevas()
    fam_e = np.array([f["familia"] for f in quedan])
    if not set(fam_e) <= set(familias):
        sys.exit(f"ABORTADO: familias fuera del corpus: {sorted(set(fam_e) - set(familias))}")
    iocs_e = [set(extraer_marcadores(t)) for t in textos_e]
    sel = {"N-rep": [j for j, f in enumerate(quedan) if f["representante"]],
           "N-all": [j for j, f in enumerate(quedan) if not f["motivo"]]}
    extra = {"V0": None}
    for v, idx in sel.items():
        y_x = np.array([fam_e[j] for j in idx])
        extra[v] = dict(t=np.array([textos_e[j] for j in idx], dtype=object), y=y_x,
                        iocs=iocs + [iocs_e[j] for j in idx],
                        nombres=nombres_nota + [None] * len(idx),
                        y_ext=np.concatenate([y, y_x]), idx=list(range(n, n + len(idx))))
    rep_fam = Counter(fam_e[sel["N-rep"]])
    all_fam = Counter(fam_e[sel["N-all"]])
    print(f"\nN-rep: {len(sel['N-rep'])} plantillas nuevas | N-all: {len(sel['N-all'])} notas | "
          f"{len(rep_fam)} familias reciben alguna")
    print(f"  {'familia':<14}{'plantillas corpus':>18}{'nuevas N-rep':>14}{'notas N-all':>13}")
    for f in familias:
        print(f"  {f:<14}{ppf[f]:>18}{rep_fam.get(f, 0):>14}{all_fam.get(f, 0):>13}")
    dudosas = [quedan[j] for j in sel["N-rep"]
               if quedan[j]["familia_vecina_corpus"] != quedan[j]["familia"]
               and float(quedan[j]["coseno_max_corpus"]) >= VECINA_ALTA]
    print(f"  Diagnóstico (no filtra): {len(dudosas)} plantillas nuevas cuya nota más parecida del "
          f"corpus es de OTRA familia con coseno >= {VECINA_ALTA}: "
          + ", ".join(f"{d['familia']}~{d['familia_vecina_corpus']} {float(d['coseno_max_corpus']):.2f}"
                      for d in dudosas))
    salida.mkdir(parents=True, exist_ok=True)
    with open(salida / "plantillas_nuevas_notas.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    # --- la cascada, una pasada por semilla y variante ---------------------------------------------
    def correr(splits, s, ex):
        """Cascada canónica (reglas -> texto) en todos los pliegues; ex = extras de entrenamiento.
        Con ex = None es exactamente protocolo_logo.evaluar / la V0 de p1_copias_controladas.py."""
        p_txt = np.empty(n, dtype=object)
        p_cas = np.empty(n, dtype=object)
        aplica = np.zeros(n, dtype=bool)
        for tr, te in splits:
            if ex is None:
                t_tr, y_tr = textos_arr[tr], y[tr]
                tr_d, iocs_d, nom_d, y_d = tr, iocs, nombres_nota, y
            else:
                t_tr = np.concatenate([textos_arr[tr], ex["t"]])
                y_tr = np.concatenate([y[tr], ex["y"]])
                tr_d, iocs_d, nom_d, y_d = list(tr) + ex["idx"], ex["iocs"], ex["nombres"], ex["y_ext"]
            vec = vectorizador("combinado")
            Xtr = vec.fit_transform(t_tr)
            Xte = vec.transform(textos_arr[te])
            clf = obtener_modelos(s)["LinearSVC"]
            clf.fit(Xtr, y_tr)
            pt = clf.predict(Xte)
            p_txt[te] = pt
            d = dicc_privados(tr_d, iocs_d, nom_d, y_d)
            for k, i in enumerate(te):
                r = regla(i, d, iocs, nombres_nota)
                aplica[i] = r is not None
                p_cas[i] = r if r is not None else pt[k]
        return p_txt, p_cas, aplica

    pred = {(p, c, v): np.empty((N, n), dtype=object)
            for p in PROTOCOLOS for c in CAPAS for v in VARIANTES}
    aplica = {(p, v): np.zeros((N, n), dtype=bool) for p in PROTOCOLOS for v in VARIANTES}
    print(f"\nPreparación lista en {time.time() - t0:.0f} s. Evaluando ({N} semillas x 2 protocolos x "
          f"3 variantes x {N_FOLDS} pliegues) ...", flush=True)
    for s in range(N):
        cortes = {"P1": list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                             random_state=s).split(textos_arr, y)),
                  "P2bal": split_p2bal(y, grupos, familias, np.random.default_rng(20_000 + s))}
        for p in PROTOCOLOS:
            for v in VARIANTES:
                pt, pc, ap_ = correr(cortes[p], s, extra[v])
                pred[(p, "texto", v)][s], pred[(p, "cascada", v)][s], aplica[(p, v)][s] = pt, pc, ap_
        if (s + 1) % 10 == 0 or s + 1 == N:
            print(f"  {s + 1}/{N} semillas ({time.time() - t0:.0f} s)", flush=True)

    met = {}
    for key, P in pred.items():
        met[key] = (np.array([accuracy_score(y, P[s]) for s in range(N)]),
                    np.array([f1_score(y, P[s], average="macro", labels=familias, zero_division=0)
                              for s in range(N)]))

    # --- PN-0: puertas ------------------------------------------------------------------------------
    print("\n" + "-" * 78 + "\n  PN-0  PUERTAS: V0 = cifras canónicas (exactitud / macro-F1)\n" + "-" * 78)
    ok = True
    for (p, c), (ref_ac, ref_f1) in PUERTAS.items():
        ac, f1 = met[(p, c, "V0")]
        bien = abs(ac.mean() - ref_ac) <= TOL and abs(f1.mean() - ref_f1) <= TOL
        ok = ok and bien
        estado = ("OK" if bien else "NO REPRODUCE") if canon else f"no se compara ({N} semillas)"
        print(f"  {p:<6}{c:<8} V0 exactitud {ac.mean():.4f} ({ref_ac}) | macro-F1 {f1.mean():.4f} "
              f"({ref_f1})  {estado}")
    if canon and not ok:
        sys.exit("ABORTADO (PN-0): V0 no reproduce las cifras canónicas. No se reporta nada.")

    # --- resumen por protocolo, capa y variante -----------------------------------------------------
    filas_res = []
    for p in PROTOCOLOS:
        print("\n" + "=" * 78 + f"\n  {p}: prueba = las {n} notas del corpus, {len(familias)} familias, "
              f"{N} semillas\n" + "=" * 78)
        for c in CAPAS:
            ac0, f10 = met[(p, c, "V0")]
            for v in VARIANTES:
                ac, f1 = met[(p, c, v)]
                fila = dict(protocolo=p, capa=c, variante=v, exactitud=round(ac.mean(), 4),
                            exactitud_sd=round(sd(ac), 4), macroF1=round(f1.mean(), 4),
                            macroF1_sd=round(sd(f1), 4),
                            cobertura_regla=round(float(aplica[(p, v)].mean()), 4) if c == "cascada" else "")
                linea = (f"  {c:<8}{v:<6} exactitud {ac.mean():.4f} ± {sd(ac):.4f} | "
                         f"macro-F1 {f1.mean():.4f} ± {sd(f1):.4f}")
                if c == "cascada":
                    linea += f" | cobertura de la regla {aplica[(p, v)].mean():.4f}"
                print(linea)
                if v != "V0":
                    for nombre, d in (("exactitud", ac - ac0), ("macroF1", f1 - f10)):
                        m_, lo, hi = ic(d)
                        fila.update({f"delta_{nombre}": round(m_, 4), f"ic95_{nombre}": f"[{lo:+.4f}; {hi:+.4f}]",
                                     f"semillas_mejor_{nombre}": int((d > 0).sum()),
                                     f"semillas_peor_{nombre}": int((d < 0).sum())})
                    print(f"          {v} - V0: exactitud {fila['delta_exactitud']:+.4f} {fila['ic95_exactitud']}"
                          f" (mejor en {fila['semillas_mejor_exactitud']}, peor en {fila['semillas_peor_exactitud']})"
                          f" | macro-F1 {fila['delta_macroF1']:+.4f} {fila['ic95_macroF1']}"
                          f" (mejor en {fila['semillas_mejor_macroF1']}, peor en {fila['semillas_peor_macroF1']})")
                filas_res.append(fila)

    # --- PN-3: P1, ganancia de las notas únicas frente a las demás ----------------------------------
    hit = {v: (pred[("P1", "cascada", v)] == y).astype(bool) for v in VARIANTES}
    print("\n  P1, cascada: ganancia por tamaño de plantilla en el corpus (acierto medio por nota)")
    ganan = {}
    for v in ("N-rep", "N-all"):
        g_u = hit[v][:, unica].mean(1) - hit["V0"][:, unica].mean(1)
        g_o = hit[v][:, ~unica].mean(1) - hit["V0"][:, ~unica].mean(1)
        ganan[v] = (g_u, g_o)
        m_, lo, hi = ic(g_u - g_o)
        print(f"    {v:<6} únicas ({int(unica.sum())}): {hit['V0'][:, unica].mean():.4f} -> "
              f"{hit[v][:, unica].mean():.4f} ({g_u.mean():+.4f}) | demás ({int((~unica).sum())}): "
              f"{hit['V0'][:, ~unica].mean():.4f} -> {hit[v][:, ~unica].mean():.4f} ({g_o.mean():+.4f})"
              f" | diferencia {m_:+.4f} [{lo:+.4f}; {hi:+.4f}]")

    # --- P2bal: F1 por familia (PN-4) ---------------------------------------------------------------
    f1fam = {(c, v): np.mean([precision_recall_fscore_support(y, pred[("P2bal", c, v)][s], labels=familias,
                                                              zero_division=0)[2] for s in range(N)], axis=0)
             for c in CAPAS for v in VARIANTES}
    d_rep = f1fam[("cascada", "N-rep")] - f1fam[("cascada", "V0")]
    print("\n  P2bal, F1 por familia de la cascada (media sobre semillas)")
    print(f"    {'familia':<14}{'nuevas':>7}{'V0':>8}{'N-rep':>8}{'delta':>8}{'N-all':>8}{'delta':>8}")
    filas_fam = []
    for j, f in enumerate(familias):
        a0, ar, aa = (f1fam[("cascada", v)][j] for v in VARIANTES)
        print(f"    {f:<14}{rep_fam.get(f, 0):>7}{a0:>8.3f}{ar:>8.3f}{ar - a0:>+8.3f}{aa:>8.3f}{aa - a0:>+8.3f}")
        fila = dict(familia=f, plantillas_corpus=ppf[f], plantillas_nuevas_rep=rep_fam.get(f, 0),
                    notas_nuevas_all=all_fam.get(f, 0))
        for c in CAPAS:
            for v in VARIANTES:
                fila[f"f1_{c}_{v}"] = round(float(f1fam[(c, v)][j]), 4)
        filas_fam.append(fila)

    # --- salidas ------------------------------------------------------------------------------------
    def escribir(nombre, filas_csv):
        campos = list(dict.fromkeys(k for fl in filas_csv for k in fl))
        with open(salida / nombre, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            w.writerows(filas_csv)

    escribir("resumen.csv", filas_res)
    escribir("por_semilla.csv", [dict(protocolo=p, capa=c, variante=v, semilla=s,
                                      exactitud=round(float(met[(p, c, v)][0][s]), 6),
                                      macroF1=round(float(met[(p, c, v)][1][s]), 6))
                                 for p in PROTOCOLOS for c in CAPAS for v in VARIANTES for s in range(N)])
    escribir("p1_por_nota.csv", [dict(familia=y[i], archivo=archivos[i], tam_plantilla=tam[grupos[i]],
                                      unica=int(unica[i]),
                                      **{f"acierto_cascada_{v}": round(float(hit[v][:, i].mean()), 4)
                                         for v in VARIANTES})
                                 for i in range(n)])
    escribir("p2bal_por_familia.csv", filas_fam)

    # --- veredicto ----------------------------------------------------------------------------------
    def r(p, c, v):
        return next(fl for fl in filas_res if (fl["protocolo"], fl["capa"], fl["variante"]) == (p, c, v))

    pn1, pn2 = r("P2bal", "cascada", "N-rep"), r("P1", "cascada", "N-rep")
    lo1 = ic(met[("P2bal", "cascada", "N-rep")][1] - met[("P2bal", "cascada", "V0")][1])[1]
    lo2 = ic(met[("P1", "cascada", "N-rep")][0] - met[("P1", "cascada", "V0")][0])[1]
    g_u, g_o = ganan["N-rep"]
    peores = [(familias[j], d_rep[j]) for j in np.argsort(d_rep)[:3]]
    print("\n" + "=" * 78)
    print("  VEREDICTO DEL PREREGISTRO (se reporta lo que dé)" if canon else
          f"  VEREDICTO SIN VALIDEZ: {N} semillas, prueba de humo (las puertas no se compararon)")
    print("=" * 78)
    chk = [("PN-0 puertas: V0 reproduce las cifras canónicas", ok if canon else None, ""),
           ("PN-1 P2bal N-rep: macro-F1 cascada sube, IC sobre cero", lo1 > 0,
            f"{pn1['delta_macroF1']:+.4f} {pn1['ic95_macroF1']}"),
           ("PN-2 P1 N-rep: exactitud cascada sube, IC sobre cero", lo2 > 0,
            f"{pn2['delta_exactitud']:+.4f} {pn2['ic95_exactitud']}"),
           ("PN-3 P1 N-rep: las únicas ganan más que las demás", g_u.mean() > g_o.mean(),
            f"únicas {g_u.mean():+.4f} | demás {g_o.mean():+.4f}"),
           ("PN-4 P2bal N-rep: ninguna familia pierde más de 0,10", bool(d_rep.min() >= -0.10),
            "peores: " + ", ".join(f"{f} {d:+.3f}" for f, d in peores))]
    for nombre, cumple, det in chk:
        etq = "NO EVALUADA" if cumple is None else ("CUMPLE" if cumple else "FALLA ")
        print(f"  [{etq}] {nombre:<56} {det}")
    print("\n  RECORDAR AL CITAR: base = las 149 notas / 30 familias del corpus; las plantillas nuevas solo")
    print("  entrenan. Bajo P2bal, «plantilla no vista» es según coseno char 0,90 (no detecta contención).")
    print(f"\nSalidas en {salida}  |  tiempo total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
