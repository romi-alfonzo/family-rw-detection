# -*- coding: utf-8 -*-
"""Filtro del Exp. 2g: la extensión solo puede elegir entre las familias que propone el contenido.

Idea de Romina (2026-11-01), diseñada DESPUÉS de ver la cascada (job 5828), y declarada así. En la cascada, cuando el
contenido duda decide el sistema completo, y si la extensión miente se pierden archivos que el contenido acertaba sin
estar seguro (JIGSAW: el contenido acierta el 94,5 %; la cascada con la extensión quitada, 0). Regla del filtro: el
modelo de contenido propone sus k familias más probables. Si la respuesta del sistema completo, que usa también la
forma de la extensión, está entre ellas, se acepta; si apunta a otra familia, se ignora y queda la del contenido.

Mismos modelos (job 5818), mismos 14.948 archivos no vistos y mismos cuatro nombres que los jobs 5823 y 5828: se
repiten la selección y las perturbaciones con las mismas semillas, y se controla que den lo mismo que allí.

k fijado antes de correr: 2 (la regla propuesta) y 3 (para ver el compromiso). Se informan los dos.

PREDICCIONES, registradas antes de correr (01-11):
  F1. Control: contenido, completo y cascada U=0,5 reproducen los jobs 5823 y 5828.
  F2. Con el nombre original, el filtro k=2 llega a 0,98 de macro-F1 o más.
  F3. Con la extensión cambiada o quitada, el filtro k=2 queda a menos de 0,03 de macro-F1 por debajo del contenido
      (0,9361), o por encima.
  F4. Con la extensión cambiada o quitada, el filtro k=2 supera a la cascada U=0,5 (0,9005 / 0,8761 / 0,8627).
  F5. k=3 frente a k=2: más alto con el nombre original y más bajo con la extensión cambiada o quitada.

Agregado el 11-11 (pedido del tutor: «debemos mostrar matriz de confusión»): con el nombre original, guarda la matriz
de confusión de cada modelo (filas: familia verdadera; columnas: familia predicha) y las predicciones por archivo. No
cambia ninguna cifra; el control es que la diagonal de la matriz del filtro k=2 dé su exactitud (0,9786).

Uso en el clúster: python3.11 exp2g_examen_filtro.py resultados_exp2g_modelo_job5818/modelos_exp2g.joblib
Prueba local:      python exp2g_examen_filtro.py <joblib de --prueba> --raiz <carpeta sintética> --salida <carpeta>
"""
import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))
from exp2e_estructura_bytes import N_HEAD, N_TAIL, leer, rasgos_estructura  # noqa: E402
from exp2g_examen_renombrado import nombres_perturbados, particion  # noqa: E402
from exp2g_nombre_robusto import forma_de_la_extension  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

KS = (2, 3)
CONDICIONES = ["original", "sin_agregada", "aleatoria", "otra_familia"]
CAMBIADAS = ("sin_agregada", "aleatoria", "otra_familia")
REF = {("original", "contenido"): 0.9361, ("original", "completo"): 0.9999,
       ("sin_agregada", "completo"): 0.0941, ("aleatoria", "completo"): 0.4936,
       ("otra_familia", "completo"): 0.3446, ("original", "cascada_0.5"): 0.9940,
       ("sin_agregada", "cascada_0.5"): 0.9005, ("aleatoria", "cascada_0.5"): 0.8761,
       ("otra_familia", "cascada_0.5"): 0.8627}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modelos", type=Path)
    ap.add_argument("--raiz", default=None, help="por defecto, la del entrenamiento")
    ap.add_argument("--max-por-familia", type=int, default=500)
    ap.add_argument("--salida", type=Path, default=None)
    args = ap.parse_args()
    paq = joblib.load(args.modelos)
    raiz = args.raiz or paq["raiz"]
    out = args.salida or _AQUI / ("resultados_exp2g_filtro_job" + os.environ.get("SLURM_JOB_ID", "local"))
    out.mkdir(parents=True, exist_ok=True)
    reg = open(out / "log.txt", "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        reg.write(m + "\n")
        reg.flush()

    # La misma selección que los jobs 5823 y 5828: mismas semillas y mismo orden.
    entrenados, retenidos = particion(raiz, paq["por_familia"], paq["semilla"])
    assert len(entrenados) == paq["n_entrenamiento"], (len(entrenados), paq["n_entrenamiento"])
    rng = np.random.default_rng(1)
    elegidos = []
    for fam in sorted({f for f, _ in retenidos}):
        de_fam = [x for x in retenidos if x[0] == fam]
        elegidos += [de_fam[i] for i in sorted(rng.permutation(len(de_fam))[:args.max_por_familia])]
    log(f"  EXP. 2g -- FILTRO (la extensión solo elige entre las k familias del contenido) · {raiz}")
    log(f"  modelo: {paq['n_entrenamiento']} archivos de entrenamiento (verificado) · examen: {len(elegidos)} archivos "
        f"no vistos de {len({f for f, _ in elegidos})} familias · k {KS}")

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
    orden = np.argsort(-proba, axis=1, kind="stable")
    pred_c = m_cont.classes_[orden[:, 0]]
    conf_c = proba.max(1)
    top = {k: m_cont.classes_[orden[:, :k]] for k in KS}
    log("\n  lo que propone el contenido (igual en las cuatro condiciones):")
    log(f"    la primera familia acierta {(pred_c == y).mean():.4f}")
    for k in KS:
        log(f"    la familia verdadera está entre las {k} primeras en {(top[k] == y[:, None]).any(1).mean():.4f} de los archivos")

    filas, porfam, acept = [], [], []
    for cond, ns in nombres_perturbados(nombres, y, np.random.default_rng(2)).items():
        comp = paq["modelos"]["completo"].predict(np.hstack([Xb, Xe, forma_de_la_extension(ns)]))
        preds = {"contenido": pred_c, "completo": comp, "cascada_0.5": np.where(conf_c >= 0.5, pred_c, comp)}
        for k in KS:
            entra = (top[k] == comp[:, None]).any(1)
            preds[f"filtro_k{k}"] = np.where(entra, comp, pred_c)
            cambia = entra & (comp != pred_c)
            acept.append(dict(condicion=cond, k=k, se_acepta=round(float(entra.mean()), 4),
                              cambia_la_respuesta=round(float(cambia.mean()), 4),
                              acierto_donde_cambia=round(float((comp[cambia] == y[cambia]).mean()), 4)
                              if cambia.any() else float("nan"),
                              se_ignora=round(float((~entra).mean()), 4)))
        if cond == "original":
            fams = sorted(set(y))
            for modelo, yp in preds.items():
                cm = pd.DataFrame(confusion_matrix(y, yp, labels=fams), index=fams, columns=fams)
                cm.to_csv(out / f"matriz_confusion_{modelo}_original.csv", encoding="utf-8")
                if modelo == "filtro_k2":
                    log(f"\n  matriz de confusión del filtro k=2 con el nombre original: {int(cm.values.sum())} "
                        f"archivos, diagonal / total = {np.trace(cm.values) / cm.values.sum():.4f}")
            pd.DataFrame(dict(familia=y, archivo=nombres, **{f"pred_{m}": v for m, v in preds.items()})).to_csv(
                out / "predicciones_por_archivo_original.csv", index=False, encoding="utf-8")
        for modelo, yp in preds.items():
            filas.append(dict(condicion=cond, modelo=modelo, accuracy=round(accuracy_score(y, yp), 4),
                              f1_macro=round(f1_score(y, yp, average="macro", zero_division=0), 4)))
            for fam in sorted(set(y)):
                m = y == fam
                porfam.append(dict(condicion=cond, modelo=modelo, familia=fam, archivos=int(m.sum()),
                                   acierto=round(float((yp[m] == fam).mean()), 4)))
    res = pd.DataFrame(filas)
    pf = pd.DataFrame(porfam)
    ac = pd.DataFrame(acept)
    res.to_csv(out / "filtro_resumen.csv", index=False)
    pf.to_csv(out / "filtro_por_familia.csv", index=False)
    ac.to_csv(out / "filtro_aceptacion.csv", index=False)
    modelos = ["contenido", "completo", "cascada_0.5"] + [f"filtro_k{k}" for k in KS]
    log("\n  qué hace la extensión en el filtro (fracción de archivos):")
    log(ac.to_string(index=False))
    log("\n  macro-F1 por modelo y condición:")
    log(res.pivot(index="modelo", columns="condicion", values="f1_macro").loc[modelos, CONDICIONES].to_string())
    log("\n  exactitud por modelo y condición:")
    log(res.pivot(index="modelo", columns="condicion", values="accuracy").loc[modelos, CONDICIONES].to_string())
    t2 = pf[pf.modelo == "filtro_k2"].pivot(index="familia", columns="condicion", values="acierto")
    t2["contenido"] = pf[(pf.modelo == "contenido") & (pf.condicion == "original")].set_index("familia").acierto
    log("\n  acierto por familia, filtro k=2 (y contenido como referencia):")
    log(t2[CONDICIONES + ["contenido"]].to_string())

    r = res.set_index(["condicion", "modelo"]).f1_macro
    log("\n  PREDICCIONES")
    dif = {k: round(r[k] - v, 4) for k, v in REF.items()}
    log(f"  F1 reproduce los jobs 5823 y 5828: {'SÍ' if all(abs(d) <= 0.0001 for d in dif.values()) else 'NO'} "
        f"(mayor diferencia {max(abs(d) for d in dif.values()):.4f})")
    log(f"  F2 filtro k=2 con nombre original >= 0,98: {r['original', 'filtro_k2']} -> "
        f"{'SÍ' if r['original', 'filtro_k2'] >= 0.98 else 'NO'}")
    cont = r["original", "contenido"]
    for c in CAMBIADAS:
        v = r[c, "filtro_k2"]
        log(f"  F3 {c}: filtro k=2 = {v} frente a contenido {cont} -> {'SÍ' if v > cont - 0.03 else 'NO'} "
            f"(diferencia {v - cont:+.4f})")
    for c in CAMBIADAS:
        v, w = r[c, "filtro_k2"], r[c, "cascada_0.5"]
        log(f"  F4 {c}: filtro k=2 = {v} frente a cascada 0,5 = {w} -> {'SÍ' if v > w else 'NO'}")
    sube = r["original", "filtro_k3"] >= r["original", "filtro_k2"]
    baja = all(r[c, "filtro_k3"] <= r[c, "filtro_k2"] for c in CAMBIADAS)
    log(f"  F5 k=3 sube con el original y baja con la extensión cambiada: {'SÍ' if sube and baja else 'NO'} "
        f"(sube con el original: {'sí' if sube else 'no'}; baja en las tres cambiadas: {'sí' if baja else 'no'})")


if __name__ == "__main__":
    main()
