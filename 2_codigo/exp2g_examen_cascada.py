# -*- coding: utf-8 -*-
"""Cascada del Exp. 2g: que decida el contenido y que la extensión solo desempate cuando el contenido duda.

Pedido de Romina (2026-11-01), después del examen de renombrado (job 5823). Allí el sistema completo cayó muy por
debajo del modelo de contenido cuando la extensión cambia: 0,4936 de macro-F1 con extensión aleatoria y 0,3446 con
la de otra familia, frente a 0,9361 del contenido. Se prueba una regla, sin reentrenar nada: si el modelo de contenido
le da a su familia más probable una probabilidad >= U, decide el contenido; si no, decide el sistema completo, que
usa también la forma de la extensión.

Mismos modelos (job 5818), mismos 14.948 archivos no vistos y mismos cuatro nombres que el job 5823: se repiten la
selección y las perturbaciones con las mismas semillas, y se controla que contenido y completo den lo mismo que allí.

UMBRALES fijados antes de correr: U = 0,5, 0,7 y 0,9. Se informan los tres; no se elige uno mirando este examen.

PREDICCIONES, registradas antes de correr (01-11):
  C1. Control: contenido y completo reproducen el job 5823 (macro-F1 0,9361 el contenido; el completo 0,9999 /
      0,0941 / 0,4936 / 0,3446 con nombre original / sin la extensión agregada / aleatoria / de otra familia).
  C2. Con el nombre original, la cascada con U = 0,7 llega a 0,98 de macro-F1 o más.
  C3. Con la extensión cambiada o quitada, la cascada con U = 0,7 queda por debajo del contenido (0,9361), pero a
      menos de 0,05 de macro-F1.
  C4. A mayor umbral, más archivos los decide el sistema completo: la cascada sube con el nombre original y baja con
      la extensión cambiada o quitada.

Uso en el clúster: python3.11 exp2g_examen_cascada.py resultados_exp2g_modelo_job5818/modelos_exp2g.joblib
Prueba local:      python exp2g_examen_cascada.py <joblib de --prueba> --raiz <carpeta sintética> --salida <carpeta>
"""
import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))
from exp2e_estructura_bytes import N_HEAD, N_TAIL, leer, rasgos_estructura  # noqa: E402
from exp2g_examen_renombrado import nombres_perturbados, particion  # noqa: E402
from exp2g_nombre_robusto import forma_de_la_extension  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

UMBRALES = (0.5, 0.7, 0.9)
CAMBIADAS = ("sin_agregada", "aleatoria", "otra_familia")
REF_5823 = {("original", "contenido"): 0.9361, ("original", "completo"): 0.9999,
            ("sin_agregada", "completo"): 0.0941, ("aleatoria", "completo"): 0.4936,
            ("otra_familia", "completo"): 0.3446}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modelos", type=Path)
    ap.add_argument("--raiz", default=None, help="por defecto, la del entrenamiento")
    ap.add_argument("--max-por-familia", type=int, default=500)
    ap.add_argument("--salida", type=Path, default=None)
    args = ap.parse_args()
    paq = joblib.load(args.modelos)
    raiz = args.raiz or paq["raiz"]
    out = args.salida or _AQUI / ("resultados_exp2g_cascada_job" + os.environ.get("SLURM_JOB_ID", "local"))
    out.mkdir(parents=True, exist_ok=True)
    reg = open(out / "log.txt", "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        reg.write(m + "\n")
        reg.flush()

    # La misma selección que exp2g_examen_renombrado.py (job 5823): mismas semillas y mismo orden.
    entrenados, retenidos = particion(raiz, paq["por_familia"], paq["semilla"])
    assert len(entrenados) == paq["n_entrenamiento"], (len(entrenados), paq["n_entrenamiento"])
    rng = np.random.default_rng(1)
    elegidos = []
    for fam in sorted({f for f, _ in retenidos}):
        de_fam = [x for x in retenidos if x[0] == fam]
        elegidos += [de_fam[i] for i in sorted(rng.permutation(len(de_fam))[:args.max_por_familia])]
    log(f"  EXP. 2g -- CASCADA (decide el contenido si su confianza es >= U) · {raiz}")
    log(f"  modelo: {paq['n_entrenamiento']} archivos de entrenamiento (verificado) · examen: {len(elegidos)} archivos "
        f"no vistos de {len({f for f, _ in elegidos})} familias · umbrales {UMBRALES}")

    Xb, Xe, y, nombres = [], [], [], []
    for fam, p in elegidos:
        head, cola, cabp, colap, medios, tam = leer(p)
        Xb.append(head + cola)
        Xe.append(rasgos_estructura(cabp, colap, medios, tam))
        y.append(fam)
        nombres.append(p.name)
    Xb = np.frombuffer(b"".join(Xb), dtype=np.uint8).reshape(len(Xb), N_HEAD + N_TAIL).astype(np.float32)
    Xe = np.nan_to_num(np.asarray(Xe, dtype=np.float32))
    y = np.array(y)

    m_cont = paq["modelos"]["contenido"]
    proba = m_cont.predict_proba(np.hstack([Xb, Xe]))
    pred_c = m_cont.classes_[proba.argmax(1)]
    conf_c = proba.max(1)

    log("\n  quién decide (depende solo del contenido, igual en las cuatro condiciones):")
    for u in UMBRALES:
        dec = conf_c >= u
        log(f"    U={u}: el contenido decide {dec.mean():.4f} de los archivos (acierta {(pred_c[dec] == y[dec]).mean():.4f} "
            f"ahí); el completo decide {(~dec).mean():.4f} (donde el contenido solo acertaba "
            f"{(pred_c[~dec] == y[~dec]).mean() if (~dec).any() else float('nan'):.4f})")

    filas, porfam = [], []
    for cond, ns in nombres_perturbados(nombres, y, np.random.default_rng(2)).items():
        comp = paq["modelos"]["completo"].predict(np.hstack([Xb, Xe, forma_de_la_extension(ns)]))
        preds = {"contenido": pred_c, "completo": comp}
        for u in UMBRALES:
            preds[f"cascada_{u}"] = np.where(conf_c >= u, pred_c, comp)
        for modelo, yp in preds.items():
            filas.append(dict(condicion=cond, modelo=modelo, accuracy=round(accuracy_score(y, yp), 4),
                              f1_macro=round(f1_score(y, yp, average="macro", zero_division=0), 4)))
            for fam in sorted(set(y)):
                m = y == fam
                porfam.append(dict(condicion=cond, modelo=modelo, familia=fam, archivos=int(m.sum()),
                                   acierto=round(float((yp[m] == fam).mean()), 4)))
    res = pd.DataFrame(filas)
    pf = pd.DataFrame(porfam)
    res.to_csv(out / "cascada_resumen.csv", index=False)
    pf.to_csv(out / "cascada_por_familia.csv", index=False)
    tabla = res.pivot(index="modelo", columns="condicion", values="f1_macro")
    orden = ["contenido", "completo"] + [f"cascada_{u}" for u in UMBRALES]
    log("\n  macro-F1 por modelo y condición:")
    log(tabla.loc[orden, ["original", "sin_agregada", "aleatoria", "otra_familia"]].to_string())
    log("\n  exactitud por modelo y condición:")
    log(res.pivot(index="modelo", columns="condicion", values="accuracy")
        .loc[orden, ["original", "sin_agregada", "aleatoria", "otra_familia"]].to_string())
    t07 = pf[pf.modelo == "cascada_0.7"].pivot(index="familia", columns="condicion", values="acierto")
    t07["contenido"] = pf[(pf.modelo == "contenido") & (pf.condicion == "original")].set_index("familia").acierto
    log("\n  acierto por familia, cascada con U = 0,7 (y contenido como referencia):")
    log(t07[["original", "sin_agregada", "aleatoria", "otra_familia", "contenido"]].to_string())

    r = res.set_index(["condicion", "modelo"]).f1_macro
    log("\n  PREDICCIONES")
    dif = {k: round(r[k] - v, 4) for k, v in REF_5823.items()}
    log(f"  C1 reproduce el 5823: {'SÍ' if all(abs(d) <= 0.0001 for d in dif.values()) else 'NO'} "
        f"(diferencias {', '.join(f'{c}/{m} {d:+.4f}' for (c, m), d in dif.items())})")
    log(f"  C2 cascada 0,7 con nombre original >= 0,98: {r['original', 'cascada_0.7']} -> "
        f"{'SÍ' if r['original', 'cascada_0.7'] >= 0.98 else 'NO'}")
    cont = r["original", "contenido"]
    for c in CAMBIADAS:
        v = r[c, "cascada_0.7"]
        log(f"  C3 {c}: cascada 0,7 = {v} frente a contenido {cont} -> "
            f"{'SÍ' if cont - 0.05 < v < cont else 'NO'} (diferencia {v - cont:+.4f})")
    sube = r["original", "cascada_0.5"] <= r["original", "cascada_0.7"] <= r["original", "cascada_0.9"]
    baja = all(r[c, "cascada_0.5"] >= r[c, "cascada_0.7"] >= r[c, "cascada_0.9"] for c in CAMBIADAS)
    log(f"  C4 a mayor umbral sube con el original y baja con la extensión cambiada: {'SÍ' if sube and baja else 'NO'} "
        f"(sube con el original: {'sí' if sube else 'no'}; baja en las tres cambiadas: {'sí' if baja else 'no'})")


if __name__ == "__main__":
    main()
