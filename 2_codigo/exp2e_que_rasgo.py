#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""Exp. 2e-b -- ¿QUÉ rasgo estructural hace el trabajo? Diagnóstico del job 4058.

POR QUÉ HACE FALTA
------------------
El Exp. 2e mejoró el frente de archivos de 0,9114 a 0,9359 de macro-F1, y la mejora cayó
entera en las seis familias difíciles (+0,1186 contra +0,0006 en las otras 24). Pero la
predicción sobre el MECANISMO falló, y eso deja el resultado sin explicación.

Lo predicho: que subieran SUNCRYPT (entropía de cola 4,78) y NOTPETYA (6,58), las dos que sí
dejan un pie poco aleatorio, porque `largo_cola_no_aleatoria` se diseñó para ellas.

Lo que pasó: las que más subieron fueron WASTEDLOCKER (+0,1920) y DARKSIDE (+0,1345), que el
capítulo describe como «exactamente en el techo» de entropía --7,591 y 7,585 en WASTEDLOCKER,
7,593 en la cabecera de DARKSIDE--, es decir indistinguibles de datos aleatorios en AMBOS
extremos. Si no hay estructura de entropía que medir, la señal tiene que venir de otro lado.

La hipótesis que queda: los rasgos de TAMAÑO. Todas las familias de NapierOne cifraron el
mismo conjunto base de documentos, de modo que las diferencias de tamaño entre familias son
diferencias en CUÁNTO AGREGA CADA UNA --relleno a bloque, pie de longitud fija, cabecera
propia--. Eso es una propiedad del código de la familia, y es invisible para una
representación que mira valores de byte en posiciones fijas.

Un jurado va a preguntar qué rasgo hace el trabajo. Esto lo contesta con medición.

QUÉ MIDE, en dos partes
-----------------------
(A) IMPORTANCIAS del bosque sobre «bytes + estructura», la columna adoptada. Reparte la
    importancia total entre los 1.024 bytes y los 44 rasgos, y ordena los 44. Es un ajuste
    único sobre todos los datos: es diagnóstico, no una métrica reportable, y así se declara.

(B) ABLACIÓN POR GRUPO sobre «solo estructura», con validación cruzada de verdad. Se quita un
    grupo de rasgos por vez y se mide cuánto cae. La ablación es más informativa que las
    importancias cuando los rasgos están correlacionados ---y acá lo están: ocho entropías de
    cola a ocho profundidades miden casi lo mismo--- porque las importancias reparten el
    crédito entre rasgos redundantes y hacen parecer débil a cada uno. La caída al quitar el
    grupo entero no tiene ese problema.

    Los siete grupos: tamaño (5) · entropía de cabecera (8) · entropía de cola (8) · entropía
    del medio (12, incluidos los cuatro estadísticos) · distribución de bytes (9) · pie no
    aleatorio (1) · salto cabecera-cola (1).

    Se reporta la caída global y la caída en las SEIS DIFÍCILES por separado, porque es ahí
    donde está toda la mejora y puede ser un grupo distinto del que sostiene el agregado.

Reutiliza la carga y los rasgos de `exp2e_estructura_bytes.py`: si ese cambia, este cambia.
"""

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))
from exp2e_estructura_bytes import (  # noqa: E402
    HIPER, NOMBRES_ESTRUCTURA, N_JOBS, cargar)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIFICILES = ["SUNCRYPT", "WASTEDLOCKER", "CRYPTOLOCKER", "DARKSIDE", "JIGSAW", "NOTPETYA"]

GRUPOS = {
    "tamaño": ["tam", "log_tam", "tam_mod16", "tam_mod512", "tam_mod4096"],
    "entropia_cabecera": [f"H_cab_{n}" for n in (16, 32, 64, 128, 256, 512, 1024, 4096)],
    "entropia_cola": [f"H_cola_{n}" for n in (16, 32, 64, 128, 256, 512, 1024, 4096)],
    "entropia_medio": [f"H_medio_{i}" for i in range(8)]
                      + ["H_medio_media", "H_medio_desvio", "H_medio_min", "H_medio_max"],
    "distribucion": ["chi2_cab", "chi2_cola", "distintos_cab", "distintos_cola",
                     "maxfrec_cab", "maxfrec_cola", "ceros_cab", "ceros_cola", "ascii_cola"],
    "pie_no_aleatorio": ["largo_cola_no_aleatoria"],
    "salto_cab_cola": ["salto_cab_cola"],
}


def f1_de(y, yp, familias):
    rep = classification_report(y, yp, zero_division=0, output_dict=True)
    return {f: rep[f]["f1-score"] for f in familias if f in rep}


def media_dificiles(porfam):
    """Media de F1 sobre las seis difíciles, o NaN si ninguna está en el conjunto.

    El `nan` aparece solo con corpus de prueba que no contienen esas familias; con
    NapierOne están las seis. Se maneja para que el humo no ensucie el log con avisos."""
    v = [porfam[f] for f in DIFICILES if f in porfam]
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz")
    ap.add_argument("--por-familia", type=int, default=500)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--prueba", action="store_true")
    args = ap.parse_args()
    if args.prueba:
        args.por_familia, args.folds = 30, 2

    base = _AQUI.parent / "4_resultados" if (_AQUI.parent / "4_resultados").is_dir() else _AQUI
    out = args.salida or (base / ("resultados_exp2e_rasgo_job" +
                                  os.environ.get("SLURM_JOB_ID", "local")))
    if out.exists() and any(out.iterdir()):
        sys.exit(f"ABORTA: {out} ya tiene resultados. Borrarla o pasar --salida.")
    out.mkdir(parents=True, exist_ok=True)
    reg = open(out / "log.txt", "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        reg.write(m + "\n")
        reg.flush()

    # comprobación de que los grupos cubren los 44 rasgos exactamente una vez
    cubiertos = [r for g in GRUPOS.values() for r in g]
    faltan = set(NOMBRES_ESTRUCTURA) - set(cubiertos)
    sobran = set(cubiertos) - set(NOMBRES_ESTRUCTURA)
    if faltan or sobran or len(cubiertos) != len(NOMBRES_ESTRUCTURA):
        sys.exit(f"ABORTA: los grupos no parten los rasgos. Faltan {faltan}, sobran {sobran}.")

    log("=" * 78)
    log("  EXP. 2e-b -- ¿QUÉ RASGO ESTRUCTURAL HACE EL TRABAJO?")
    log("=" * 78)
    log(f"  datos: {args.raiz} · semilla {args.semilla} · {args.folds} folds")
    log(f"  {len(NOMBRES_ESTRUCTURA)} rasgos en {len(GRUPOS)} grupos · núcleos {N_JOBS}")

    t0 = time.time()
    Xb, Xe, y, familias, _ = cargar(args.raiz, args.por_familia, args.semilla, log)
    log(f"\n  {len(y)} archivos · {len(familias)} familias  ({round(time.time() - t0)} s)")
    idx = {r: i for i, r in enumerate(NOMBRES_ESTRUCTURA)}

    # ---------------------------------------------------------------- (A)
    log("\n" + "=" * 78)
    log("  (A) IMPORTANCIAS sobre «bytes + estructura» (ajuste único, DIAGNÓSTICO)")
    log("=" * 78)
    X2 = np.hstack([Xb, Xe])
    clf = RandomForestClassifier(random_state=args.semilla, n_jobs=N_JOBS,
                                 class_weight="balanced", **HIPER).fit(X2, y)
    imp = clf.feature_importances_
    imp_bytes, imp_est = imp[:Xb.shape[1]].sum(), imp[Xb.shape[1]:].sum()
    log(f"\n  1.024 bytes posicionales : {imp_bytes:.4f}")
    log(f"  44 rasgos estructurales  : {imp_est:.4f}")
    log(f"  => cada rasgo estructural pesa {imp_est / 44 / (imp_bytes / 1024):.1f} veces "
        f"lo que un byte")
    fi = pd.DataFrame({"rasgo": NOMBRES_ESTRUCTURA, "importancia": imp[Xb.shape[1]:]})
    fi["grupo"] = fi.rasgo.map({r: g for g, rs in GRUPOS.items() for r in rs})
    fi = fi.sort_values("importancia", ascending=False)
    fi.to_csv(out / "importancias_rasgos.csv", index=False)
    log("\n  Los 12 rasgos estructurales más importantes:")
    log(fi.head(12).to_string(index=False))
    log("\n  Importancia acumulada por grupo:")
    log(fi.groupby("grupo").importancia.sum().sort_values(ascending=False).to_string())

    # ---------------------------------------------------------------- (B)
    log("\n" + "=" * 78)
    log("  (B) ABLACIÓN POR GRUPO sobre «solo estructura» (validación cruzada)")
    log("=" * 78)
    cv = StratifiedKFold(args.folds, shuffle=True, random_state=args.semilla)

    def evaluar(X):
        m = RandomForestClassifier(random_state=args.semilla, n_jobs=1,
                                   class_weight="balanced", **HIPER)
        yp = cross_val_predict(m, X, y, cv=cv, n_jobs=N_JOBS)
        return f1_score(y, yp, average="macro", zero_division=0), f1_de(y, yp, familias)

    f_base, porfam_base = evaluar(Xe)
    dif_base = media_dificiles(porfam_base)
    log(f"\n  completo (44 rasgos):  macro-F1 {f_base:.4f}  ·  "
        f"seis difíciles {dif_base:.4f}")
    log("\n  Al QUITAR cada grupo:")
    log(f"  {'grupo quitado':<20} {'n':>3} {'macro-F1':>9} {'caída':>9} "
        f"{'seis dif.':>10} {'caída dif.':>11}")
    filas = []
    for g, rs in GRUPOS.items():
        quedan = [idx[r] for r in NOMBRES_ESTRUCTURA if r not in rs]
        f, porfam = evaluar(Xe[:, quedan])
        d = media_dificiles(porfam)
        log(f"  {g:<20} {len(rs):>3} {f:>9.4f} {f - f_base:>+9.4f} "
            f"{d:>10.4f} {d - dif_base:>+11.4f}")
        filas.append(dict(grupo=g, n_rasgos=len(rs), f1_macro=round(f, 4),
                          caida=round(f - f_base, 4), f1_dificiles=round(d, 4),
                          caida_dificiles=round(d - dif_base, 4)))
    # y el complemento: SOLO el grupo de tamaño, que es la hipótesis
    solo_tam = [idx[r] for r in GRUPOS["tamaño"]]
    f_tam, porfam_tam = evaluar(Xe[:, solo_tam])
    d_tam = media_dificiles(porfam_tam)
    log(f"\n  SOLO los 5 rasgos de tamaño: macro-F1 {f_tam:.4f} · "
        f"seis difíciles {d_tam:.4f}")
    filas.append(dict(grupo="SOLO_tamaño", n_rasgos=5, f1_macro=round(f_tam, 4),
                      caida=None, f1_dificiles=round(d_tam, 4), caida_dificiles=None))
    pd.DataFrame(filas).to_csv(out / "ablacion_por_grupo.csv", index=False)
    pd.DataFrame([porfam_base]).T.rename(columns={0: "f1"}).to_csv(
        out / "f1_por_familia_solo_estructura.csv")

    log("\n" + "-" * 78)
    log("  CÓMO LEERLO")
    log("-" * 78)
    log("  La caída al quitar un grupo es lo que ese grupo aporta y ningún otro cubre.")
    log("  Grupos redundantes entre sí dan caídas chicas los dos: la entropía de cola a")
    log("  ocho profundidades mide casi lo mismo ocho veces. Por eso se mira también")
    log("  «SOLO tamaño», que no tiene ese problema.")
    log("  La columna de las seis difíciles es la que decide: ahí está toda la mejora.")

    (out / "manifiesto.json").write_text(json.dumps(dict(
        fecha=str(date.today()), raiz=str(args.raiz), semilla=args.semilla,
        por_familia=args.por_familia, folds=args.folds, hiperparametros=HIPER,
        grupos={g: len(r) for g, r in GRUPOS.items()},
        importancia_bytes=float(imp_bytes), importancia_estructura=float(imp_est),
        nota_A="ajuste unico sobre todos los datos: diagnostico, no metrica reportable",
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n  Salidas en: {out}")
    reg.close()


if __name__ == "__main__":
    main()
