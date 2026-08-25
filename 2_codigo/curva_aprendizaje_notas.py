#!/usr/bin/env python3
"""
curva_aprendizaje_notas.py -- B.1: curva de aprendizaje del frente de notas.

PREGUNTA QUE CONTESTA (pedido textual del tutor, reunion 2026-08-12, punto 2):
    "ver el porcentaje de error para saber la cantidad de notas a necesitar"

QUE HACE
Entrena a proposito con MENOS de lo que hay --tope de k unidades por familia-- y mide
cuanto empeora. Si el rendimiento sigue subiendo en el ultimo punto medido, conviene
salir a recolectar; si esta plano, el corpus llego a su techo y corresponde documentarlo
y pasar a few-shot (bifurcacion del Sprint C de PLAN_MEJORAS.md).

DOS UNIDADES, Y LA DIFERENCIA ENTRE ELLAS ES EL RESULTADO
  - unidad "plantillas": las notas que se agregan son TEXTOS DISTINTOS  -> diversidad (caro)
  - unidad "notas":      notas al azar, que muchas veces repiten un texto -> volumen (barato)
Las dos curvas se miran sobre el mismo eje de notas de entrenamiento por familia. La
separacion entre ambas al mismo numero de notas es el valor de la diversidad, medido.

TRES CONJUNTOS DE FAMILIAS (la trampa central de este experimento)
Si se dejan las 30 familias y se sube k, las familias con 2 plantillas se agotan y la
curva satura sola; si se dejan caer, el macro-F1 sube porque hay menos clases que
adivinar, no porque el modelo aprenda. Las dos versiones enganan en direcciones
opuestas, asi que se corren y se reportan las dos, por separado:
  - "30fam": las 30 familias, tope k. Se anota cuantas familias quedan efectivamente
             por debajo del tope, que es la fraccion del punto que es informacion nueva.
  - "11fam": las familias con >= 4 plantillas. Presentes en TODOS los puntos de la
             curva, de punta a punta. Es la unica extrapolable.
  - "5fam":  las que tienen >= 5 plantillas. Un punto mas de alcance, mucho mas fragil.
Los nombres "11fam" y "5fam" son HISTORICOS, no un conteo: sobre 144 notas esos subconjuntos
tenian 11 y 5 familias; sobre 149 tienen 15 y 3. Las tres curvas NO son comparables entre si,
porque el azar es 1/n_familias del subconjunto y ese n cambia con el corpus: 30 familias dan
~0,033, pero el de los subconjuntos hay que leerlo de manifiesto_b1.json -> azar_macro_f1
(sobre 149: 0,067 y 0,333). Declarar cuantas familias y que azar en toda tabla.

PROTOCOLOS
  P1     StratifiedKFold por nota (plantilla conocida). Canonico: 2 pliegues, 10 semillas.
  P2     StratifiedGroupKFold por plantilla (variante nunca vista). Idem canonico.
  P2ret  Retencion de UNA plantilla por familia al test, repetida R veces. Bajo P2 de 2
         pliegues una familia de 2 plantillas aporta 1 sola al train, asi que el tope k
         casi no muerde y la curva no tiene alcance. La retencion maximiza el lado de
         entrenamiento (hasta n-1 plantillas) y es exactamente la pregunta de despliegue.

ANIDAMIENTO (importante para la estadistica)
El orden de plantillas/notas de cada familia se sortea UNA vez por (repeticion, pliegue)
y k toma el prefijo, de modo que el train de k=1 esta contenido en el de k=2. Los puntos
quedan PAREADOS: la diferencia entre k y k+1 se calcula por repeticion y despues se
promedia, en vez de restar dos medias independientes. Con +/- 0,057 de desvio en el
macro-F1 de P2, restar medias sueltas no distingue nada.

CONTROL DE CORRECCION
El punto k="todo" bajo el protocolo canonico tiene que dar lo MISMO que evaluar() de
clasificador_notas_v2.py sobre el mismo corpus. Se verifica con --validar y no se sigue
si no coincide.

BASE
Corpus en disco al 2026-08-18: 144 notas / 95 plantillas / 30 familias. El numero oficial
de la tesis (macro-F1 0,435 +/- 0,057 en P2) se midio sobre 146 notas; las 2 que faltan
(DHARMA/Info__3.hta, Info__13.hta) estan en cuarentena de Defender y ambas caen dentro de
la misma plantilla, de modo que el eje de plantillas es identico. El punto "todo" de este
script mide, de paso, cuanto cuestan esas 2 notas.

Uso:
    python curva_aprendizaje_notas.py --validar        # solo el control de correccion
    python curva_aprendizaje_notas.py --rapido         # R=15, para probar el cableado
    python curva_aprendizaje_notas.py                  # corrida completa (R=100)

Salidas en 4_resultados/resultados_curva_notas/ (una fila por repeticion y por punto;
los promedios los calcula resumen_para_capitulo4.py, no este script).
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
import sklearn
from joblib import Parallel, delayed
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, N_FOLDS, N_SEMILLAS, UMBRAL_NEARDUP,
                                   agrupar_neardups, cargar_corpus, evaluar,
                                   obtener_modelos, vectorizador)

OUT_DIR = (_AQUI.parent / "4_resultados" / "resultados_curva_notas"
           if (_AQUI.parent / "4_resultados").is_dir()
           else _AQUI / "resultados_curva_notas")

# Configuracion ganadora de la corrida canonica, por protocolo. Verificado el 2026-08-18
# en 4_resultados/resultados_canonicos/corrida_canonica_resumen.csv:
#   estratificado -> caracteres + LinearSVC : macro-F1 0,7597 +/- 0,0289
#   grupos        -> combinado  + LinearSVC : macro-F1 0,4353 +/- 0,0565
# Se fija la configuracion y NO se vuelve a seleccionar modelo dentro de la curva: lo que
# se mide es el efecto de la cantidad de datos, no una nueva busqueda de hiperparametros.
CONFIG = {
    "P1":    ("caracteres", "LinearSVC"),
    "P2":    ("combinado",  "LinearSVC"),
    "P2ret": ("combinado",  "LinearSVC"),
}

R_DEFECTO = 100        # repeticiones de la retencion
R_RAPIDO = 15

# Un ajuste cuesta 1-2,5 s (el TF-IDF de char_wb 3-5 sobre notas largas es lo caro), y la
# corrida son unos 3.600 ajustes. Se paraleliza POR REPETICION, que es la unica dimension
# donde no hay dependencia entre tareas. LinearSVC es de un solo hilo, asi que no hay
# sobresuscripcion. En un cluster, respetar los nucleos del job.
import os as _os  # noqa: E402  (local al bloque de configuracion)
N_JOBS_REPS = int(_os.environ.get("SLURM_CPUS_PER_TASK", 0)) or max(1, (_os.cpu_count() or 2) - 2)


# ============================================================
# SELECCION DEL SUBCONJUNTO DE ENTRENAMIENTO
# ============================================================
def _por_familia_grupo(y, grupos, idx):
    """{familia: {plantilla: [indices]}} restringido a idx.

    La clave es el par (familia, plantilla) y NO la plantilla sola: dos plantillas del
    corpus contienen notas de dos familias distintas (grupo 6 = BLACKBASTA + CONTI;
    grupo 55 = DHARMA + PHOBOS). Contando por par, el tope k queda exacto por familia y
    elegir esa plantilla para una familia no le regala notas a la otra.
    """
    d = defaultdict(lambda: defaultdict(list))
    for i in idx:
        d[y[i]][grupos[i]].append(int(i))
    return d


def _ordenes(y, grupos, idx_train, rng):
    """Sortea UNA vez, por familia, el orden de sus plantillas y el de sus notas.

    Devolver el orden (y no la seleccion) es lo que hace la curva anidada: k toma el
    prefijo, asi que el train de k esta contenido en el de k+1.
    """
    porfam = _por_familia_grupo(y, grupos, idx_train)
    ord_plantillas, ord_notas = {}, {}
    for fam, gs in porfam.items():
        claves = sorted(gs)                                # determinista antes de mezclar
        claves = [claves[i] for i in rng.permutation(len(claves))]
        ord_plantillas[fam] = [gs[g] for g in claves]
        notas = sorted(i for g in claves for i in gs[g])
        ord_notas[fam] = [notas[i] for i in rng.permutation(len(notas))]
    return ord_plantillas, ord_notas


def _prefijo(ord_plantillas, ord_notas, k, unidad):
    """Indices de entrenamiento con tope k por familia. k=None => sin tope."""
    if unidad == "plantillas":
        if k is None:
            return np.array(sorted(i for ls in ord_plantillas.values()
                                   for l in ls for i in l), int)
        return np.array(sorted(i for ls in ord_plantillas.values()
                               for l in ls[:k] for i in l), int)
    if unidad == "notas":
        if k is None:
            return np.array(sorted(i for ns in ord_notas.values() for i in ns), int)
        return np.array(sorted(i for ns in ord_notas.values() for i in ns[:k]), int)
    raise ValueError(unidad)


def _contabilidad(y, grupos, idx_train, idx_k, k, unidad, familias):
    """Lo que hay que reportar al lado de cada punto para que sea interpretable."""
    disp = _por_familia_grupo(y, grupos, idx_train)
    usa = _por_familia_grupo(y, grupos, idx_k)
    if unidad == "plantillas":
        disponibles = {f: len(g) for f, g in disp.items()}
    else:
        disponibles = {f: sum(len(v) for v in g.values()) for f, g in disp.items()}
    tope = 10 ** 9 if k is None else k
    return dict(
        n_notas_train=int(len(idx_k)),
        n_plantillas_train=int(sum(len(g) for g in usa.values())),
        # Familias a las que el tope les corto algo: son las que aportarian mas si se
        # subiera k. Si son pocas, el punto siguiente casi no trae informacion nueva.
        n_fam_bajo_tope=int(sum(1 for f in familias if disponibles.get(f, 0) > tope)),
        n_fam_sin_train=int(sum(1 for f in familias if f not in usa)),
    )


# ============================================================
# PARTICIONES
# ============================================================
def _splits(protocolo, T, Y, G, rep, n_folds, rng):
    if protocolo == "P1":
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rep)
        return list(cv.split(T, Y))
    if protocolo == "P2":
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=rep)
        return list(cv.split(T, Y, groups=G))
    if protocolo == "P2ret":
        # Una plantilla por familia al test. Si la plantilla elegida contiene notas de
        # otra familia (grupos 6 y 55), TODAS sus notas van al test: de otro modo
        # quedaria en entrenamiento contenido casi identico al de prueba, que es
        # exactamente la fuga que P2 evita.
        porfam = _por_familia_grupo(Y, G, np.arange(len(Y)))
        elegidos = set()
        for fam in sorted(porfam):
            claves = sorted(porfam[fam])
            elegidos.add(claves[int(rng.integers(len(claves)))])
        te = np.array([i for i in range(len(Y)) if G[i] in elegidos], int)
        tr = np.array([i for i in range(len(Y)) if G[i] not in elegidos], int)
        return [(tr, te)]
    raise ValueError(protocolo)


# ============================================================
# UNA CURVA
# ============================================================
def _una_repeticion(T, Y, G, familias, *, protocolo, unidad, ks, etiqueta,
                    rep, n_folds, vista, modelo):
    """Una repeticion completa. Aislada para poder paralelizar por repeticion.

    Paralelizar NO cambia ninguna cifra: cada repeticion lleva su propio generador
    (semilla 10.000 + rep) y su propio random_state de modelo, y LinearSVC es de un
    solo hilo, asi que no hay sobresuscripcion.
    """
    rng = np.random.default_rng(10_000 + rep)
    splits = _splits(protocolo, T, Y, G, rep, n_folds, rng)
    pred = {k: np.array([""] * len(Y), dtype=object) for k in ks}
    cub = {k: np.zeros(len(Y), bool) for k in ks}
    cont = {k: [] for k in ks}

    for tr, te in splits:
        ordp, ordn = _ordenes(Y, G, tr, rng)
        for k in ks:
            idx = _prefijo(ordp, ordn, k, unidad)
            cont[k].append(_contabilidad(Y, G, tr, idx, k, unidad, familias))
            if len(idx) == 0 or len(np.unique(Y[idx])) < 2:
                continue
            pipe = Pipeline([("tfidf", vectorizador(vista)),
                             ("clf", obtener_modelos(rep)[modelo])])
            pipe.fit(T[idx], Y[idx])
            pred[k][te] = pipe.predict(T[te])
            cub[k][te] = True

    filas, porfam_filas = [], []
    for k in ks:
        m = cub[k]
        if not m.any():
            continue
        yv, yp = Y[m], pred[k][m].astype(str)
        c = {kk: float(np.mean([d[kk] for d in cont[k]])) for kk in cont[k][0]}
        filas.append(dict(
            curva=etiqueta, protocolo=protocolo, unidad=unidad,
            k=("todo" if k is None else k), repeticion=rep,
            n_familias=len(familias), n_notas_evaluadas=int(m.sum()),
            exactitud=accuracy_score(yv, yp),
            exactitud_balanceada=balanced_accuracy_score(yv, yp),
            f1_macro=f1_score(yv, yp, average="macro", zero_division=0),
            f1_weighted=f1_score(yv, yp, average="weighted", zero_division=0),
            **c))
        _, _, f1f, _ = precision_recall_fscore_support(
            yv, yp, labels=familias, zero_division=0)
        for fam, v in zip(familias, f1f):
            porfam_filas.append(dict(curva=etiqueta, protocolo=protocolo,
                                     unidad=unidad, k=("todo" if k is None else k),
                                     repeticion=rep, familia=fam, f1=v))
    return filas, porfam_filas


def curva(textos, y, grupos, *, protocolo, unidad, ks, familias_incluidas,
          etiqueta, n_reps, n_folds):
    mask = np.isin(y, list(familias_incluidas))
    T, Y, G = np.array(textos, dtype=object)[mask], y[mask], grupos[mask]
    familias = np.unique(Y)
    vista, modelo = CONFIG[protocolo]

    print(f"\n[{etiqueta} | {protocolo} | tope por {unidad} | {len(familias)} familias | "
          f"{len(Y)} notas | R={n_reps}]", flush=True)

    salidas = Parallel(n_jobs=N_JOBS_REPS, verbose=0)(
        delayed(_una_repeticion)(T, Y, G, familias, protocolo=protocolo, unidad=unidad,
                                 ks=ks, etiqueta=etiqueta, rep=rep, n_folds=n_folds,
                                 vista=vista, modelo=modelo)
        for rep in range(n_reps))
    filas = [f for s in salidas for f in s[0]]
    porfam_filas = [f for s in salidas for f in s[1]]

    df = pd.DataFrame(filas)
    for k in ks:
        kk = "todo" if k is None else k
        s = df[df.k == kk]
        if len(s):
            print(f"    k={str(kk):<5} macro-F1 {s.f1_macro.mean():.4f} "
                  f"+/- {s.f1_macro.std():.4f} | exactitud {s.exactitud.mean():.4f} | "
                  f"notas train/fam {s.n_notas_train.mean() / len(familias):.2f} | "
                  f"plantillas train/fam {s.n_plantillas_train.mean() / len(familias):.2f} | "
                  f"fam. bajo tope {s.n_fam_bajo_tope.mean():.1f} | "
                  f"fam. sin train {s.n_fam_sin_train.mean():.1f}")
    return df, pd.DataFrame(porfam_filas)


# ============================================================
# CONTROL DE CORRECCION
# ============================================================
def validar(textos, y, grupos, familias):
    """El punto k="todo" tiene que dar lo mismo que evaluar() del script canonico."""
    ok = True
    for protocolo, prot_canon in (("P1", "estratificado"), ("P2", "grupos")):
        vista, modelo = CONFIG[protocolo]
        ref, _, _ = evaluar(textos, y, grupos, vista, modelo, prot_canon, familias)
        df, _ = curva(textos, y, grupos, protocolo=protocolo, unidad="plantillas",
                      ks=[None], familias_incluidas=familias,
                      etiqueta="validacion", n_reps=N_SEMILLAS, n_folds=N_FOLDS)
        mio = df.f1_macro.mean()
        d = abs(mio - ref["f1_macro_mean"])
        print(f"  {protocolo}: canonico {ref['f1_macro_mean']:.6f} | curva {mio:.6f} | "
              f"diferencia {d:.2e}  {'OK' if d < 1e-9 else 'NO COINCIDE'}")
        ok = ok and d < 1e-9
    return ok


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--validar", action="store_true",
                    help="solo el control de correccion, sin correr las curvas")
    ap.add_argument("--rapido", action="store_true",
                    help=f"R={R_RAPIDO} en vez de {R_DEFECTO} (prueba de cableado)")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--salida", type=Path, default=None,
                    help="carpeta de salida (por defecto, 4_resultados/resultados_curva_notas). "
                         "Usar una carpeta NUEVA para no pisar la corrida canonica.")
    args = ap.parse_args()
    reps = args.reps or (R_RAPIDO if args.rapido else R_DEFECTO)
    if args.salida is not None:
        OUT_DIR = args.salida

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("  B.1 -- CURVA DE APRENDIZAJE DEL FRENTE DE NOTAS")
    print("=" * 78)
    textos, y, archivos, metodos = cargar_corpus(CORPUS_DIR)
    grupos, pares = agrupar_neardups(textos, UMBRAL_NEARDUP)
    familias = np.unique(y)
    plantillas_por_fam = {f: len({g for g, ff in zip(grupos, y) if ff == f})
                          for f in familias}
    fam11 = sorted(f for f, n in plantillas_por_fam.items() if n >= 4)
    fam5 = sorted(f for f, n in plantillas_por_fam.items() if n >= 5)
    print(f"Corpus: {CORPUS_DIR}")
    print(f"Notas: {len(textos)} | Familias: {len(familias)} | "
          f"Plantillas (componentes): {len(set(grupos))}")
    print(f"Familias con >=4 plantillas ({len(fam11)}): {', '.join(fam11)}")
    print(f"Familias con >=5 plantillas ({len(fam5)}): {', '.join(fam5)}")

    print("\n" + "-" * 78)
    print("  CONTROL DE CORRECCION -- k=todo contra el evaluador canonico")
    print("-" * 78)
    if not validar(textos, y, grupos, familias):
        sys.exit("ABORTADO: el punto k=todo no reproduce el evaluador canonico.")
    if args.validar:
        return

    trabajos = [
        # etiqueta, protocolo, unidad, ks, familias, reps, pliegues
        ("30fam", "P2ret", "plantillas", [1, 2, 3, 4, 5, 6, 7, None], familias, reps, 1),
        ("30fam", "P2ret", "notas", [1, 2, 3, 4, 6, 8, 12, None], familias, reps, 1),
        ("11fam", "P2ret", "plantillas", [1, 2, 3, None], fam11, reps, 1),
        ("11fam", "P2ret", "notas", [1, 2, 3, 4, 6, None], fam11, reps, 1),
        ("5fam", "P2ret", "plantillas", [1, 2, 3, 4, None], fam5, reps, 1),
        ("30fam", "P2", "plantillas", [1, 2, 3, 4, None], familias, N_SEMILLAS, N_FOLDS),
        ("30fam", "P2", "notas", [1, 2, 3, 4, 6, 8, None], familias, N_SEMILLAS, N_FOLDS),
        ("30fam", "P1", "plantillas", [1, 2, 3, 4, None], familias, N_SEMILLAS, N_FOLDS),
        ("30fam", "P1", "notas", [1, 2, 3, 4, 6, 8, None], familias, N_SEMILLAS, N_FOLDS),
    ]
    todas, todas_fam = [], []
    for etiqueta, prot, unidad, ks, fams, r, nf in trabajos:
        df, dff = curva(textos, y, grupos, protocolo=prot, unidad=unidad, ks=ks,
                        familias_incluidas=fams, etiqueta=etiqueta, n_reps=r, n_folds=nf)
        todas.append(df)
        todas_fam.append(dff)

    pd.concat(todas, ignore_index=True).to_csv(
        OUT_DIR / "b1_curva_por_repeticion.csv", index=False)
    pd.concat(todas_fam, ignore_index=True).to_csv(
        OUT_DIR / "b1_curva_por_familia.csv", index=False)

    (OUT_DIR / "manifiesto_b1.json").write_text(json.dumps(dict(
        fecha=str(date.today()), corpus=str(CORPUS_DIR), n_notas=len(textos),
        n_familias=int(len(familias)), n_plantillas=int(len(set(grupos))),
        pares_neardup=len(pares), umbral_neardup=UMBRAL_NEARDUP,
        plantillas_por_familia={k: int(v) for k, v in plantillas_por_fam.items()},
        notas_por_familia={str(k): int(v) for k, v in Counter(y).items()},
        familias_4mas=fam11, familias_5mas=fam5,
        config_por_protocolo={k: list(v) for k, v in CONFIG.items()},
        repeticiones_retencion=reps, n_folds_canonico=N_FOLDS,
        n_semillas_canonico=N_SEMILLAS,
        azar_macro_f1=dict(fam30=round(1 / 30, 4), fam11=round(1 / len(fam11), 4),
                           fam5=round(1 / len(fam5), 4)),
        sklearn=sklearn.__version__, python=sys.version.split()[0],
    ), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSalidas en: {OUT_DIR}")
    print("Los promedios y la extrapolacion: python resumen_para_capitulo4.py --b1")


if __name__ == "__main__":
    main()
