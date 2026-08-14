#!/usr/bin/env python3.11
"""Ablación de ventana extendida y bloque del medio (pedidos del tutor, 2026-08-12).

Responde tres preguntas planteadas en la reunión de revisión de resultados:

(a) ¿Dónde se corta el crecimiento de la curva?
    La ablación anterior llegaba hasta 512+512 con exactitud 0,908 y **seguía subiendo**,
    de modo que el gráfico no muestra saturación. Acá se extiende a 1024, 2048 y 4096
    bytes por extremo hasta que la curva se aplane o baje.

(b) ¿La curva sube por más señal o por un artefacto?
    Al agrandar la ventana, cada vez más archivos son *más cortos* que la ventana y se
    rellenan con ceros. Ese relleno codifica el tamaño del archivo, que es una pista
    ajena al contenido. Por eso todo se mide dos veces: sobre el corpus completo (que es
    lo comparable con los resultados previos) y sobre el subconjunto de archivos lo
    bastante grandes como para que no haya ni relleno ni solapamiento entre cabecera y
    cola. Si la curva del subconjunto limpio también sube, la mejora es real.

(c) ¿Por qué no se revisan los bytes del medio?
    Se agrega un bloque tomado del centro del archivo y se lo evalúa solo y combinado.

Uso:
    python3.11 -u ablacion_ventana_extendida.py /scratch/ralfonzo/Napierone-small \
        --por-familia 500

Los hiperparámetros del bosque son los mismos del Experimento 2c, hallados con una
ventana de 512+512. Se mantienen sin tocar para que las cifras sean comparables; que
estén ajustados a la ventana chica es una limitación a declarar, no un error.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or -1
_AQUI = Path(__file__).resolve().parent
OUT_DIR = (_AQUI.parent / "4_resultados" / "resultados_ablacion_extendida"
           if (_AQUI.parent / "4_resultados").is_dir()
           else _AQUI / "resultados_ablacion_extendida")

MAX_W = 4096          # bytes que se leen de cada extremo
N_MEDIO = 1024        # bytes que se leen del centro del archivo
VENTANAS = [64, 128, 256, 512, 1024, 2048, 4096]

RF_PARAMS = dict(n_estimators=300, max_depth=20, min_samples_leaf=2,
                 max_features=0.3, random_state=42, n_jobs=1,
                 class_weight="balanced")


def tipo_documento(nombre):
    m = re.match(r"^\d+-([a-z0-9]+)", nombre.lower())
    return m.group(1) if m else "desconocido"


def leer(path):
    """Devuelve (cabecera, cola, medio, tamaño real). Los extremos se rellenan con
    ceros si el archivo es más corto que la ventana; `tam` permite saber después
    cuáles quedaron rellenados."""
    with open(path, "rb") as f:
        head = f.read(MAX_W)
        f.seek(0, 2)
        n = f.tell()
        f.seek(max(0, n - MAX_W))
        tail = f.read(MAX_W)
        f.seek(max(0, (n - N_MEDIO) // 2))
        medio = f.read(N_MEDIO)
    return (head.ljust(MAX_W, b"\x00"), tail.ljust(MAX_W, b"\x00"),
            medio.ljust(N_MEDIO, b"\x00"), n)


def cargar(raiz, por_familia, seed=42):
    rng = np.random.default_rng(seed)
    H, T, M, y, tipos, tam = [], [], [], [], [], []
    for d in sorted(p for p in Path(raiz).iterdir() if p.is_dir()):
        fam = d.name.upper()
        for suf in ("-SMALL", "_SMALL", "-TINY", "_TINY"):
            fam = fam.removesuffix(suf)
        archivos = [p for p in sorted(d.iterdir())
                    if p.is_file() and p.suffix.lower() != ".pdf"]
        if len(archivos) < 6:
            continue
        sel = [archivos[i] for i in rng.permutation(len(archivos))[:por_familia]]
        for p in sel:
            h, t, m, n = leer(p)
            H.append(h); T.append(t); M.append(m)
            y.append(fam); tipos.append(tipo_documento(p.name)); tam.append(n)
    k = len(y)
    H = np.frombuffer(b"".join(H), dtype=np.uint8).reshape(k, MAX_W)
    T = np.frombuffer(b"".join(T), dtype=np.uint8).reshape(k, MAX_W)
    M = np.frombuffer(b"".join(M), dtype=np.uint8).reshape(k, N_MEDIO)
    return H, T, M, np.array(y), np.array(tipos), np.array(tam)


def matriz(partes):
    return np.hstack([p for p in partes if p is not None and p.shape[1]]).astype(np.float32)


def evaluar(X, y, semilla=42):
    """Validación cruzada de 3 pliegues. Con muchas columnas se paraleliza *dentro*
    del bosque (hilos que comparten X) en vez de entre pliegues (procesos que copian
    X), porque a 4096+4096 cada copia de X pesa cientos de megabytes."""
    cv = StratifiedKFold(3, shuffle=True, random_state=semilla)
    params = dict(RF_PARAMS)
    if X.shape[1] > 2048:
        params["n_jobs"] = N_JOBS
        cv_jobs = 1
    else:
        cv_jobs = N_JOBS
    yp = cross_val_predict(RandomForestClassifier(**params), X, y, cv=cv, n_jobs=cv_jobs)
    return (accuracy_score(y, yp),
            f1_score(y, yp, average="macro", zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz")
    ap.add_argument("--por-familia", type=int, default=500)
    ap.add_argument("--ventanas", type=int, nargs="+", default=VENTANAS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.por_familia = 40
        args.ventanas = [64, 512, 1024]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)
    resultados = {}
    t0 = time.time()

    log("=" * 74)
    log("  ABLACIÓN DE VENTANA EXTENDIDA + BLOQUE DEL MEDIO")
    log("=" * 74)
    H, T, M, y, tipos, tam = cargar(args.raiz, args.por_familia)
    familias = np.unique(y)
    log(f"Cargados {len(y)} archivos | {len(familias)} familias | "
        f"núcleos {N_JOBS} | {time.time()-t0:.0f}s")
    log(f"Tipos de documento: {dict(Counter(tipos))}\n")

    # ------------------------------------------------------------------ (0)
    # Cuántos archivos son más cortos que cada ventana. Es lo que decide si la
    # curva se puede leer como «más señal» o hay que sospechar del relleno.
    log("=" * 74)
    log("  (0) TAMAÑO DE LOS ARCHIVOS Y RIESGO DE RELLENO")
    log("=" * 74)
    log(f"  Tamaño: mínimo {tam.min()} | mediana {int(np.median(tam))} | "
        f"máximo {tam.max()} bytes")
    filas_0 = []
    for w in args.ventanas:
        cortos = int((tam < 2 * w).sum())
        filas_0.append(dict(ventana=w, n_bytes=2 * w, archivos_mas_cortos=cortos,
                            pct=round(100 * cortos / len(tam), 1)))
        log(f"  ventana {w:>5}+{w:<5} → {cortos:>6} archivos "
            f"({100*cortos/len(tam):>5.1f} %) son más cortos que {2*w} bytes")
    pd.DataFrame(filas_0).to_csv(OUT_DIR / "0_tamanos.csv", index=False)
    resultados["0_tamanos"] = filas_0

    w_max = max(args.ventanas)
    limpio = tam >= 2 * w_max
    fam_limpio = Counter(y[limpio])
    usables = [f for f in familias if fam_limpio.get(f, 0) >= 30]
    log(f"\n  Subconjunto sin relleno ni solapamiento (tamaño ≥ {2*w_max} bytes): "
        f"{int(limpio.sum())} archivos, {len(usables)} familias con ≥ 30 ejemplares")
    if len(usables) < 5:
        log("  ⚠ Quedan muy pocas familias; el control limpio se omite.")
        limpio = None
    else:
        faltan = [f for f in familias if f not in usables]
        if faltan:
            log(f"  Familias excluidas del control por falta de archivos grandes: {faltan}")
        limpio = limpio & np.isin(y, usables)

    # ------------------------------------------------------------------ (a)
    log("\n" + "=" * 74)
    log("  (a) CURVA DE ABLACIÓN — ¿dónde deja de crecer?")
    log("=" * 74)
    filas_a = []
    for w in args.ventanas:
        Xc = matriz([H[:, :w], T[:, -w:]])
        acc, f1 = evaluar(Xc, y)
        fila = dict(ventana=f"{w}+{w}", n_bytes=2 * w, subconjunto="todos",
                    n_archivos=len(y), accuracy=round(acc, 4), f1_macro=round(f1, 4))
        log(f"  {w:>5}+{w:<5} ({2*w:>5} bytes)  todos    "
            f"exactitud {acc:.3f}  macro-F1 {f1:.3f}   [{(time.time()-t0)/60:.0f} min]")
        filas_a.append(fila)
        del Xc

        if limpio is not None:
            Xl = matriz([H[limpio][:, :w], T[limpio][:, -w:]])
            acc_l, f1_l = evaluar(Xl, y[limpio])
            filas_a.append(dict(ventana=f"{w}+{w}", n_bytes=2 * w,
                                subconjunto="sin_relleno", n_archivos=int(limpio.sum()),
                                accuracy=round(acc_l, 4), f1_macro=round(f1_l, 4)))
            log(f"  {w:>5}+{w:<5} ({2*w:>5} bytes)  limpio   "
                f"exactitud {acc_l:.3f}  macro-F1 {f1_l:.3f}")
            del Xl
    pd.DataFrame(filas_a).to_csv(OUT_DIR / "a_curva_ablacion.csv", index=False)
    resultados["a_curva"] = filas_a

    # ------------------------------------------------------------------ (b)
    log("\n" + "=" * 74)
    log("  (b) BLOQUE DEL MEDIO — ¿hay información fuera de los extremos?")
    log("=" * 74)
    w = 512 if 512 in args.ventanas else args.ventanas[0]
    configs = [
        (f"solo medio {N_MEDIO}", [M]),
        (f"solo cabecera {w}", [H[:, :w]]),
        (f"solo cola {w}", [T[:, -w:]]),
        (f"cabecera+cola {w}", [H[:, :w], T[:, -w:]]),
        (f"cabecera+cola+medio {w}", [H[:, :w], T[:, -w:], M]),
    ]
    filas_b = []
    for nombre, partes in configs:
        Xc = matriz(partes)
        acc, f1 = evaluar(Xc, y)
        filas_b.append(dict(config=nombre, n_bytes=Xc.shape[1],
                            accuracy=round(acc, 4), f1_macro=round(f1, 4)))
        log(f"  {nombre:<26} {Xc.shape[1]:>5} bytes   "
            f"exactitud {acc:.3f}   macro-F1 {f1:.3f}")
        del Xc
    pd.DataFrame(filas_b).to_csv(OUT_DIR / "b_bloque_medio.csv", index=False)
    resultados["b_medio"] = filas_b

    # ------------------------------------------------------------------ figura
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        df = pd.DataFrame(filas_a)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for sub, color, marca in (("todos", "#2c5f8a", "o"),
                                  ("sin_relleno", "#c8763c", "s")):
            d = df[df.subconjunto == sub]
            if not len(d):
                continue
            etiqueta = ("corpus completo" if sub == "todos"
                        else "solo archivos sin relleno")
            ax.plot(d.n_bytes, d.f1_macro, marca + "-", color=color, label=etiqueta)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("bytes leídos (cabecera + cola)")
        ax.set_ylabel("macro-F1")
        ax.set_title("Cuántos bytes hacen falta para identificar la familia")
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig_ablacion_extendida.png", dpi=160)
        log(f"\n  Figura: {OUT_DIR / 'fig_ablacion_extendida.png'}")
    except Exception as e:
        log(f"  (figura omitida: {e})")

    (OUT_DIR / "manifiesto.json").write_text(json.dumps(dict(
        fecha=str(date.today()), raiz=str(args.raiz), n_archivos=int(len(y)),
        n_familias=int(len(familias)), por_familia=args.por_familia,
        ventanas=args.ventanas, max_w=MAX_W, n_medio=N_MEDIO,
        rf_params={k: str(v) for k, v in RF_PARAMS.items()},
        resultados=resultados, sklearn=sklearn.__version__,
        python=sys.version.split()[0],
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\nSalidas en: {OUT_DIR}")
    log(f"Tiempo total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
