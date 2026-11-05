# -*- coding: utf-8 -*-
"""Exp. 2i: la extensión se usa solo si el archivo trae una (regla de ausencia).

Pregunta que motiva la regla (Romina, 2026-11-05): «¿por qué no ignorar la extensión cuando el archivo no la trae?».
Diseñada DESPUÉS de ver el filtro (job 5830), y declarada así. Parte del Experimento 2i, no un experimento nuevo.

Regla: si el nombre del archivo no termina en una extensión agregada (no tiene extensión, o la última es la de un tipo
de documento conocido, EXT_DOCUMENTO del Exp. 2g), decide solo el contenido. Si trae una extensión agregada:
  - variante «regla + filtro»: decide el filtro k=2 (la respuesta del sistema completo se acepta solo si está entre
    las dos familias más probables del contenido; si no, queda la del contenido);
  - variante «regla + completo»: decide el sistema completo.
Límite que la regla no puede evitar: NOTPETYA y BADRABBIT no agregan extensión en NapierOne, así que con el nombre
original sus archivos también van al contenido, que confunde NOTPETYA con SUNCRYPT.

Mismos modelos (job 5818), mismos 14.948 archivos no vistos y mismos cuatro nombres que los jobs 5823, 5828 y 5830: se
repiten la selección y las perturbaciones con las mismas semillas, y se controla que den lo mismo que allí.

PREDICCIONES, registradas antes de correr (05-11), macro-F1:
  R1. Control: contenido, completo y filtro k=2 reproducen el job 5830 (diferencia <= 0,0001).
  R2. Sin la extensión agregada, la regla + filtro queda a menos de 0,01 del contenido (0,9361).
  R3. Con extensión aleatoria y con la de otra familia, la regla + filtro queda a menos de 0,01 del filtro k=2
      (0,9287 y 0,9164).
  R4. Con el nombre original, la regla + filtro queda entre 0,95 y 0,9790 (por debajo del filtro, por NOTPETYA).
  R5. Peor caso de las cuatro condiciones: la regla + filtro supera 0,91, más que el filtro k=2 (0,8915) y que la
      cascada 0,5 (0,8627).

Uso en el clúster: python3.11 exp2g_examen_regla.py resultados_exp2g_modelo_job5818/modelos_exp2g.joblib
Prueba local:      python exp2g_examen_regla.py <joblib de --prueba> --raiz <carpeta sintética> --salida <carpeta>
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
from exp2g_nombre_robusto import EXT_DOCUMENTO, forma_de_la_extension  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CONDICIONES = ["original", "sin_agregada", "aleatoria", "otra_familia"]
MODELOS = ["contenido", "completo", "cascada_0.5", "filtro_k2", "regla_filtro", "regla_completo"]
REF = {("original", "contenido"): 0.9361, ("sin_agregada", "contenido"): 0.9361,
       ("original", "completo"): 0.9999, ("sin_agregada", "completo"): 0.0941,
       ("aleatoria", "completo"): 0.4936, ("otra_familia", "completo"): 0.3446,
       ("original", "filtro_k2"): 0.9790, ("sin_agregada", "filtro_k2"): 0.8915,
       ("aleatoria", "filtro_k2"): 0.9287, ("otra_familia", "filtro_k2"): 0.9164}
VIGILADAS = ["NOTPETYA", "BADRABBIT", "SUNCRYPT", "JIGSAW"]


def trae_extension(nombre):
    """True si la última extensión del nombre existe y no es la de un tipo de documento conocido."""
    base, _, ext = nombre.rpartition(".")
    return bool(base) and bool(ext) and ext.lower() not in EXT_DOCUMENTO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modelos", type=Path)
    ap.add_argument("--raiz", default=None, help="por defecto, la del entrenamiento")
    ap.add_argument("--max-por-familia", type=int, default=500)
    ap.add_argument("--salida", type=Path, default=None)
    args = ap.parse_args()
    paq = joblib.load(args.modelos)
    raiz = args.raiz or paq["raiz"]
    out = args.salida or _AQUI / ("resultados_exp2g_regla_job" + os.environ.get("SLURM_JOB_ID", "local"))
    out.mkdir(parents=True, exist_ok=True)
    reg = open(out / "log.txt", "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        reg.write(m + "\n")
        reg.flush()

    # La misma selección que los jobs 5823, 5828 y 5830: mismas semillas y mismo orden.
    entrenados, retenidos = particion(raiz, paq["por_familia"], paq["semilla"])
    assert len(entrenados) == paq["n_entrenamiento"], (len(entrenados), paq["n_entrenamiento"])
    rng = np.random.default_rng(1)
    elegidos = []
    for fam in sorted({f for f, _ in retenidos}):
        de_fam = [x for x in retenidos if x[0] == fam]
        elegidos += [de_fam[i] for i in sorted(rng.permutation(len(de_fam))[:args.max_por_familia])]
    log(f"  EXP. 2i -- REGLA DE AUSENCIA (la extensión solo se usa si el archivo trae una) · {raiz}")
    log(f"  modelo: {paq['n_entrenamiento']} archivos de entrenamiento (verificado) · examen: {len(elegidos)} archivos "
        f"no vistos de {len({f for f, _ in elegidos})} familias")

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
    top2 = m_cont.classes_[orden[:, :2]]

    filas, porfam, ruta = [], [], []
    for cond, ns in nombres_perturbados(nombres, y, np.random.default_rng(2)).items():
        comp = paq["modelos"]["completo"].predict(np.hstack([Xb, Xe, forma_de_la_extension(ns)]))
        filtro = np.where((top2 == comp[:, None]).any(1), comp, pred_c)
        trae = np.array([trae_extension(n) for n in ns])
        preds = {"contenido": pred_c, "completo": comp, "cascada_0.5": np.where(conf_c >= 0.5, pred_c, comp),
                 "filtro_k2": filtro, "regla_filtro": np.where(trae, filtro, pred_c),
                 "regla_completo": np.where(trae, comp, pred_c)}
        ruta.append(dict(condicion=cond, trae_extension=round(float(trae.mean()), 4),
                         **{f"trae_{f}": round(float(trae[y == f].mean()), 4) for f in VIGILADAS if (y == f).any()}))
        for modelo, yp in preds.items():
            filas.append(dict(condicion=cond, modelo=modelo, accuracy=round(accuracy_score(y, yp), 4),
                              f1_macro=round(f1_score(y, yp, average="macro", zero_division=0), 4)))
            f1s = f1_score(y, yp, labels=sorted(set(y)), average=None, zero_division=0)
            for fam, f1f in zip(sorted(set(y)), f1s):
                m = y == fam
                porfam.append(dict(condicion=cond, modelo=modelo, familia=fam, archivos=int(m.sum()),
                                   acierto=round(float((yp[m] == fam).mean()), 4), f1=round(float(f1f), 4)))
    res = pd.DataFrame(filas)
    pf = pd.DataFrame(porfam)
    rt = pd.DataFrame(ruta)
    res.to_csv(out / "regla_resumen.csv", index=False)
    pf.to_csv(out / "regla_por_familia.csv", index=False)
    rt.to_csv(out / "regla_ruta.csv", index=False)

    log("\n  fracción de archivos que traen una extensión agregada (los demás van al contenido):")
    log(rt.to_string(index=False))
    tabla = res.pivot(index="modelo", columns="condicion", values="f1_macro").loc[MODELOS, CONDICIONES]
    tabla["peor_caso"] = tabla.min(axis=1)
    log("\n  macro-F1 por modelo y condición (con el peor caso):")
    log(tabla.to_string())
    log("\n  exactitud por modelo y condición:")
    log(res.pivot(index="modelo", columns="condicion", values="accuracy").loc[MODELOS, CONDICIONES].to_string())
    for modelo in ("filtro_k2", "regla_filtro"):
        t = pf[(pf.modelo == modelo) & pf.familia.isin(VIGILADAS)].pivot(index="familia", columns="condicion",
                                                                            values="f1")[CONDICIONES]
        log(f"\n  F1 de la familia, {modelo}, familias vigiladas:")
        log(t.to_string())

    r = res.set_index(["condicion", "modelo"]).f1_macro
    log("\n  PREDICCIONES")
    dif = {k: round(r[k] - v, 4) for k, v in REF.items()}
    log(f"  R1 reproduce el job 5830: {'SÍ' if all(abs(d) <= 0.0001 for d in dif.values()) else 'NO'} "
        f"(mayor diferencia {max(abs(d) for d in dif.values()):.4f})")
    v = r["sin_agregada", "regla_filtro"]
    log(f"  R2 sin_agregada: regla + filtro = {v} frente a contenido 0.9361 -> "
        f"{'SÍ' if abs(v - 0.9361) < 0.01 else 'NO'} (diferencia {v - 0.9361:+.4f})")
    for c in ("aleatoria", "otra_familia"):
        v, w = r[c, "regla_filtro"], r[c, "filtro_k2"]
        log(f"  R3 {c}: regla + filtro = {v} frente a filtro k=2 = {w} -> {'SÍ' if abs(v - w) < 0.01 else 'NO'} "
            f"(diferencia {v - w:+.4f})")
    v = r["original", "regla_filtro"]
    log(f"  R4 original: regla + filtro = {v} -> {'SÍ' if 0.95 <= v <= 0.9790 else 'NO'} (esperado entre 0,95 y 0,9790)")
    pc = tabla["peor_caso"]
    log(f"  R5 peor caso: regla + filtro = {pc['regla_filtro']}, filtro k=2 = {pc['filtro_k2']}, cascada 0,5 = "
        f"{pc['cascada_0.5']} -> {'SÍ' if pc['regla_filtro'] > 0.91 and pc['regla_filtro'] > max(pc['filtro_k2'], pc['cascada_0.5']) else 'NO'}")


if __name__ == "__main__":
    main()
