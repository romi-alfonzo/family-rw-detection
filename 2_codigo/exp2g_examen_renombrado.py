# -*- coding: utf-8 -*-
"""Examen de renombrado del Exp. 2g: ¿cuánto depende el sistema completo de que la extensión no cambie?

Pedido de Romina (2026-11-01). No hay archivos cifrados públicos fuera de NapierOne, así que se mide la otra mitad
de la pregunta con datos reales: se toma el modelo guardado por exp2g_guardar_modelo.py (job 5818, 500 archivos por
familia, semilla 0) y se lo evalúa sobre los archivos de NapierOne-small que NO usó para entrenar (la misma
selección aleatoria de cargar(), tomada desde la posición 500 de cada permutación). Esos archivos se presentan con
cuatro nombres:

  - original:      tal como están (generalización a archivos no vistos de la misma ejecución);
  - sin_agregada:  sin la extensión que agregó el ransomware (si la última no es de documento, se quita);
  - aleatoria:     la extensión final cambiada por cinco caracteres alfanuméricos al azar;
  - otra_familia:  la extensión final cambiada por la de un archivo de otra familia, elegido al azar.

Los bytes no cambian: el modelo de contenido (bytes + estructura) da lo mismo en las cuatro, y sirve de control.

PREDICCIONES, registradas antes de correr (01-11):
  P1. Contenido: igual en las cuatro condiciones (control; si no, hay un error).
  P2. Completo, original: >= 0,99 de macro-F1.
  P3. Completo, aleatoria: cae más de 0,02 de macro-F1 respecto de original.
  P4. Completo, otra_familia: queda por debajo del modelo de contenido (la forma de otra familia lo engaña).
  P5. La caída de sin_agregada y aleatoria se concentra en las seis familias difíciles (NOTPETYA, JIGSAW,
      CRYPTOLOCKER, DARKSIDE, WASTEDLOCKER, SUNCRYPT).

Uso en el clúster: python3.11 exp2g_examen_renombrado.py resultados_exp2g_modelo_job5818/modelos_exp2g.joblib
Prueba local:      python exp2g_examen_renombrado.py <joblib de --prueba> --raiz <carpeta sintética> --salida <carpeta>
"""
import argparse
import os
import string
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))
from exp2e_estructura_bytes import N_HEAD, N_TAIL, es_documentacion, leer, rasgos_estructura  # noqa: E402
from exp2g_nombre_robusto import DIFICILES, EXT_DOCUMENTO, forma_de_la_extension  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def particion(raiz, por_familia, seed):
    """Repite exactamente la selección de cargar() (misma semilla, mismo orden) y separa lo no usado."""
    rng = np.random.default_rng(seed)
    entrenados, retenidos = [], []
    for d in sorted(p for p in Path(raiz).iterdir() if p.is_dir()):
        fam = d.name.upper()
        for suf in ("-SMALL", "_SMALL", "-TINY", "_TINY"):
            fam = fam.removesuffix(suf)
        arch = sorted(p for p in d.iterdir() if p.is_file() and not es_documentacion(p, fam))
        if len(arch) < 6:
            continue
        perm = rng.permutation(len(arch))
        entrenados += [(fam, arch[i]) for i in perm[:por_familia]]
        retenidos += [(fam, arch[i]) for i in perm[por_familia:]]
    return entrenados, retenidos


def nombres_perturbados(nombres, familias, rng):
    finales = [n.rpartition(".")[2] if "." in n else "" for n in nombres]
    sin, alea, otra = [], [], []
    for i, n in enumerate(nombres):
        base, _, ext = n.rpartition(".")
        if not base:
            base, ext = n, ""
        sin.append(base if ext and ext.lower() not in EXT_DOCUMENTO else n)
        alea.append(base + "." + "".join(rng.choice(list(string.ascii_lowercase + string.digits), 5)))
        ajenas = [j for j in rng.integers(0, len(nombres), 50) if familias[j] != familias[i]]
        otra.append(base + "." + finales[ajenas[0]] if ajenas else n)
    return {"original": nombres, "sin_agregada": sin, "aleatoria": alea, "otra_familia": otra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modelos", type=Path)
    ap.add_argument("--raiz", default=None, help="por defecto, la del entrenamiento")
    ap.add_argument("--max-por-familia", type=int, default=500)
    ap.add_argument("--salida", type=Path, default=None)
    args = ap.parse_args()
    paq = joblib.load(args.modelos)
    raiz = args.raiz or paq["raiz"]
    out = args.salida or _AQUI / ("resultados_exp2g_renombrado_job" + os.environ.get("SLURM_JOB_ID", "local"))
    out.mkdir(parents=True, exist_ok=True)
    reg = open(out / "log.txt", "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        reg.write(m + "\n")
        reg.flush()

    entrenados, retenidos = particion(raiz, paq["por_familia"], paq["semilla"])
    assert len(entrenados) == paq["n_entrenamiento"], (len(entrenados), paq["n_entrenamiento"])
    assert not {p for _, p in entrenados} & {p for _, p in retenidos}
    rng = np.random.default_rng(1)
    elegidos = []
    for fam in sorted({f for f, _ in retenidos}):
        de_fam = [x for x in retenidos if x[0] == fam]
        elegidos += [de_fam[i] for i in sorted(rng.permutation(len(de_fam))[:args.max_por_familia])]
    log(f"  EXP. 2g -- EXAMEN DE RENOMBRADO · {raiz}")
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
    cont = paq["modelos"]["contenido"].predict(np.hstack([Xb, Xe]))

    filas, porfam = [], []
    for cond, ns in nombres_perturbados(nombres, y, np.random.default_rng(2)).items():
        comp = paq["modelos"]["completo"].predict(np.hstack([Xb, Xe, forma_de_la_extension(ns)]))
        for modelo, yp in (("contenido", cont), ("completo", comp)):
            filas.append(dict(condicion=cond, modelo=modelo, accuracy=round(accuracy_score(y, yp), 4),
                              f1_macro=round(f1_score(y, yp, average="macro", zero_division=0), 4)))
            for fam in sorted(set(y)):
                m = y == fam
                porfam.append(dict(condicion=cond, modelo=modelo, familia=fam, archivos=int(m.sum()),
                                   acierto=round(float((yp[m] == fam).mean()), 4)))
        if cond != "original":
            log(f"  ejemplo {cond}: {nombres[0]} -> {ns[0]}")
    res = pd.DataFrame(filas)
    pf = pd.DataFrame(porfam)
    res.to_csv(out / "renombrado_resumen.csv", index=False)
    pf.to_csv(out / "renombrado_por_familia.csv", index=False)
    log("\n" + res.to_string(index=False))
    tabla = pf[pf.modelo == "completo"].pivot(index="familia", columns="condicion", values="acierto")
    tabla["contenido"] = pf[(pf.modelo == "contenido") & (pf.condicion == "original")].set_index("familia").acierto
    log("\n  acierto por familia, modelo completo (y contenido como control):")
    log(tabla[["original", "sin_agregada", "aleatoria", "otra_familia", "contenido"]].to_string())

    r = res.set_index(["condicion", "modelo"]).f1_macro
    log("\n  PREDICCIONES")
    log(f"  P1 contenido igual en las cuatro: {'SÍ' if res[res.modelo == 'contenido'].f1_macro.nunique() == 1 else 'NO'}")
    log(f"  P2 completo original >= 0,99: {r['original', 'completo']} -> {'SÍ' if r['original', 'completo'] >= 0.99 else 'NO'}")
    d = r['original', 'completo'] - r['aleatoria', 'completo']
    log(f"  P3 aleatoria cae > 0,02: {d:+.4f} -> {'SÍ' if d > 0.02 else 'NO'}")
    log(f"  P4 otra_familia < contenido: {r['otra_familia', 'completo']} vs {r['original', 'contenido']} -> "
        f"{'SÍ' if r['otra_familia', 'completo'] < r['original', 'contenido'] else 'NO'}")
    caida = (tabla["original"] - tabla[["sin_agregada", "aleatoria"]].min(axis=1)).sort_values(ascending=False)
    top = list(caida.index[:6])
    log(f"  P5 las seis con mayor caída: {', '.join(f'{f} {caida[f]:+.2f}' for f in top)}; "
        f"de las difíciles: {len(set(top) & set(DIFICILES))} de 6")


if __name__ == "__main__":
    main()
