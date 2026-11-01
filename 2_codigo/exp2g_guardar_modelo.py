# -*- coding: utf-8 -*-
"""Entrena y guarda los dos modelos del Experimento 2g para el examen con archivos de otra procedencia.

Pedido de Romina (2026-11-01): «medir la robustez» del frente de archivos con archivos cifrados reales que no
salgan de NapierOne. Para eso hace falta el modelo entrenado fuera del clúster. Este guion entrena, con los mismos
rasgos e hiperparámetros del Exp. 2g (job 4091), dos Random Forest sobre TODOS los archivos que se cargan (no hay
validación cruzada acá: la cifra de validación ya está medida) y los guarda con joblib:

  - contenido:  bytes de cabecera y cola (512 + 512) + 44 rasgos estructurales;
  - completo:   lo anterior + los 14 rasgos de forma de la extensión final.

Salida: resultados_exp2g_modelo_job<ID>/modelos_exp2g.joblib (un diccionario con los dos modelos, la lista de
familias y la descripción de los rasgos) y log.txt. El examen en local lo hace examen_archivos.py.

Uso en el clúster:  python3.11 exp2g_guardar_modelo.py /scratch/ralfonzo/Napierone-small --por-familia 500
Prueba local:       python exp2g_guardar_modelo.py <carpeta sintética> --prueba
"""
import argparse
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))
from exp2e_estructura_bytes import N_HEAD, N_JOBS, N_TAIL  # noqa: E402
from exp2g_nombre_robusto import HIPER, NOMBRES_EXT, cargar  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz")
    ap.add_argument("--por-familia", type=int, default=500)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--prueba", action="store_true", help="pocos archivos y árboles, para probar en local")
    args = ap.parse_args()
    if args.prueba:
        args.por_familia = 30
    hiper = dict(HIPER)
    if args.prueba:
        hiper["n_estimators"] = 10

    base = _AQUI.parent / "4_resultados" if (_AQUI.parent / "4_resultados").is_dir() else _AQUI
    out = args.salida or base / ("resultados_exp2g_modelo_job" + os.environ.get("SLURM_JOB_ID", "local"))
    out.mkdir(parents=True, exist_ok=True)
    reg = open(out / "log.txt", "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        reg.write(m + "\n")
        reg.flush()

    log("  EXP. 2g -- GUARDAR LOS MODELOS (contenido y completo) PARA EL EXAMEN EXTERNO")
    log(f"  {args.raiz} · {args.por_familia}/familia · semilla {args.semilla} · hiper {hiper}")
    t0 = time.time()
    Xb, Xe, _Xf, Xr, y, tipos, nombres = cargar(args.raiz, args.por_familia, args.semilla, log)
    log(f"  cargados {len(y)} archivos de {len(set(y))} familias en {time.time() - t0:.0f} s")
    modelos = {}
    for nombre, X in (("contenido", np.hstack([Xb, Xe])), ("completo", np.hstack([Xb, Xe, Xr]))):
        t0 = time.time()
        clf = RandomForestClassifier(random_state=args.semilla, n_jobs=N_JOBS, class_weight="balanced", **hiper)
        clf.fit(X, y)
        modelos[nombre] = clf
        log(f"  {nombre}: {X.shape[1]} rasgos, entrenado en {time.time() - t0:.0f} s, "
            f"exactitud en entrenamiento {clf.score(X, y):.4f}")
    paquete = dict(modelos=modelos, familias=sorted(set(y)), n_head=N_HEAD, n_tail=N_TAIL,
                   n_estructura=Xe.shape[1], nombres_ext=NOMBRES_EXT, por_familia=args.por_familia,
                   semilla=args.semilla, hiper=hiper, n_entrenamiento=len(y), raiz=str(args.raiz))
    ruta = out / "modelos_exp2g.joblib"
    joblib.dump(paquete, ruta, compress=3)
    log(f"  guardado {ruta} ({ruta.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
