#!/usr/bin/env python3
"""
analisis_bytes.py -- Análisis de robustez del clasificador de bytes (Experimento 2c).

No busca mejorar el 0,910: busca ENTENDERLO y comprobar que no descansa sobre un
confundido. Ejecuta cuatro análisis sobre los mismos datos, en una sola pasada.

(a) GENERALIZACIÓN A TIPOS DE ARCHIVO NO VISTOS  ← el crítico
    En NapierOne todas las familias cifraron el MISMO conjunto base de documentos
    (0001-doc, 0001-pdf, 0001-jpg...). Si alguna familia aplica cifrado parcial y
    conserva parte de la cabecera original, el clasificador podría estar aprendiendo
    rasgos del DOCUMENTO y no del ransomware. Se evalúa con validación
    "dejar-un-tipo-fuera": se entrena con todos los tipos salvo uno y se evalúa sobre
    ese tipo nunca visto. Si la exactitud se sostiene, la marca es del ransomware.

(b) IMPORTANCIA POR POSICIÓN DE BYTE
    Dónde, dentro de la ventana de 1.024 bytes, reside la información. Produce una
    figura que debería mostrar picos en los desplazamientos donde el Experimento 2b
    localizó las firmas.

(c) ABLACIÓN DE VENTANA
    Cuántos bytes hacen falta realmente: 64, 128, 256, 512; y cabecera sola frente a
    cola sola. Responde una pregunta práctica para quien quiera implementarlo.

(d) DIAGNÓSTICO DE LAS FAMILIAS DIFÍCILES
    Confusión entre las seis familias de bajo rendimiento y comparación de la entropía
    de sus cabeceras y colas frente al resto: ¿fallan porque cifran sin dejar estructura?

Uso:
    python3.11 analisis_bytes.py /ruta/Napierone-small
    python3.11 analisis_bytes.py /ruta --por-familia 500 --smoke
Salidas en 4_resultados/resultados_analisis_bytes/
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict

# La consola de Windows usa cp1252; forzar UTF-8 evita fallos de codificacion
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or -1
_AQUI = Path(__file__).resolve().parent
OUT_DIR = (_AQUI.parent / "4_resultados" / "resultados_analisis_bytes"
           if (_AQUI.parent / "4_resultados").is_dir()
           else _AQUI / "resultados_analisis_bytes")

N_HEAD = N_TAIL = 512
# Hiperparámetros hallados por la búsqueda anidada del Experimento 2c
RF_PARAMS = dict(n_estimators=300, max_depth=20, min_samples_leaf=2,
                 max_features=0.3, random_state=42, n_jobs=1,
                 class_weight="balanced")
DIFICILES = ["SUNCRYPT", "WASTEDLOCKER", "CRYPTOLOCKER", "DARKSIDE", "JIGSAW", "NOTPETYA"]


def tipo_documento(nombre):
    """De '0001-jpg-fromweb.jpg.avos2' devuelve 'jpg'; de '0001-doc.doc.x' devuelve 'doc'."""
    m = re.match(r"^\d+-([a-z0-9]+)", nombre.lower())
    return m.group(1) if m else "desconocido"


def leer(path):
    with open(path, "rb") as f:
        head = f.read(N_HEAD)
        f.seek(0, 2)
        n = f.tell()
        f.seek(max(0, n - N_TAIL))
        tail = f.read(N_TAIL)
    return head.ljust(N_HEAD, b"\x00"), tail.ljust(N_TAIL, b"\x00")


def cargar(raiz, por_familia, seed=42):
    rng = np.random.default_rng(seed)
    H, T, y, tipos = [], [], [], []
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
            h, t = leer(p)
            H.append(h); T.append(t); y.append(fam); tipos.append(tipo_documento(p.name))
    H = np.frombuffer(b"".join(H), dtype=np.uint8).reshape(len(y), N_HEAD)
    T = np.frombuffer(b"".join(T), dtype=np.uint8).reshape(len(y), N_TAIL)
    return H, T, np.array(y), np.array(tipos)


def matriz(H, T, n_head=N_HEAD, n_tail=N_TAIL):
    partes = []
    if n_head:
        partes.append(H[:, :n_head])
    if n_tail:
        partes.append(T[:, -n_tail:])
    return np.hstack(partes).astype(np.float32)


def entropia(fila):
    c = np.bincount(fila.astype(np.uint8), minlength=256)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz")
    ap.add_argument("--por-familia", type=int, default=500)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.por_familia = 40

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)
    resultados = {}

    log("=" * 74)
    log("  ANÁLISIS DE ROBUSTEZ DEL CLASIFICADOR DE BYTES")
    log("=" * 74)
    t0 = time.time()
    H, T, y, tipos = cargar(args.raiz, args.por_familia)
    familias = np.unique(y)
    log(f"Cargados {len(y)} archivos | {len(familias)} familias | "
        f"núcleos {N_JOBS} | {time.time()-t0:.0f}s")
    log(f"Tipos de documento: {dict(Counter(tipos))}\n")
    X = matriz(H, T)

    # ---------------------------------------------------------------- (a)
    log("=" * 74)
    log("  (a) GENERALIZACIÓN A TIPOS DE ARCHIVO NUNCA VISTOS")
    log("=" * 74)
    log("  Se entrena con todos los tipos salvo uno y se evalúa sobre ese tipo.")
    # Algunas familias renombran por completo el archivo, de modo que no conservan el
    # tipo del documento original ('desconocido' o cadenas aleatorias). Esas familias
    # participan del entrenamiento pero no aparecen en los conjuntos de prueba por tipo;
    # no se exige, por tanto, que cada tipo cubra las 29 familias. Solo se usan como
    # tipo excluido los formatos con volumen suficiente.
    cuenta_tipos = Counter(tipos)
    candidatos = sorted(t for t, c in cuenta_tipos.items()
                        if c >= 200 and t != "desconocido")
    sin_tipo = sorted({f for f, t in zip(y, tipos)
                       if t == "desconocido" or cuenta_tipos[t] < 200})
    log(f"  Tipos evaluables: {candidatos}")
    if sin_tipo:
        log(f"  Familias que renombran el archivo por completo (participan solo del "
            f"entrenamiento): {sin_tipo}\n")
    else:
        log("")

    filas_a = []
    for tipo in candidatos:
        te = np.flatnonzero(tipos == tipo)
        tr = np.flatnonzero(tipos != tipo)
        fams_te = np.unique(y[te])
        if len(fams_te) < 10:
            log(f"  {tipo:<12} omitido (solo {len(fams_te)} familias en la prueba)")
            continue
        m = RandomForestClassifier(**RF_PARAMS)
        m.n_jobs = N_JOBS
        m.fit(X[tr], y[tr])
        yp = m.predict(X[te])
        acc = accuracy_score(y[te], yp)
        f1 = f1_score(y[te], yp, average="macro", labels=fams_te, zero_division=0)
        filas_a.append(dict(tipo_excluido=tipo, n_prueba=len(te),
                            n_familias=len(fams_te),
                            accuracy=round(acc, 4), f1_macro=round(f1, 4)))
        log(f"  {tipo:<12} n={len(te):>5}  familias={len(fams_te):>3}  "
            f"exactitud {acc:.3f}  macro-F1 {f1:.3f}")

    if filas_a:
        acc_m = float(np.mean([r["accuracy"] for r in filas_a]))
        f1_m = float(np.mean([r["f1_macro"] for r in filas_a]))
        log(f"\n  PROMEDIO: exactitud {acc_m:.3f} | macro-F1 {f1_m:.3f}")
        log(f"  Referencia del Exp. 2c (mismo tipo en train y test): 0,910")
        dif = acc_m - 0.910
        log(f"  Diferencia: {dif:+.3f}")
        if dif > -0.10:
            log("  => El rendimiento SE SOSTIENE: la marca aprendida es del ransomware,")
            log("     no del documento original. El resultado principal queda confirmado.")
        else:
            log("  => CAÍDA IMPORTANTE: parte del rendimiento provendría del tipo de")
            log("     documento y no de la familia. Debe matizarse el resultado principal.")
        resultados["a_generalizacion_tipos"] = dict(
            por_tipo=filas_a, accuracy_media=round(acc_m, 4),
            f1_macro_media=round(f1_m, 4), referencia_2c=0.910,
            diferencia=round(dif, 4))
    pd.DataFrame(filas_a).to_csv(OUT_DIR / "a_generalizacion_tipos.csv", index=False)

    # ---------------------------------------------------------------- (b)
    log("\n" + "=" * 74)
    log("  (b) IMPORTANCIA POR POSICIÓN DE BYTE")
    log("=" * 74)
    m = RandomForestClassifier(**RF_PARAMS)
    m.n_jobs = N_JOBS
    m.fit(X, y)
    imp = m.feature_importances_
    df_imp = pd.DataFrame(dict(
        posicion=np.arange(len(imp)),
        region=["cabecera"] * N_HEAD + ["cola"] * N_TAIL,
        offset=list(range(N_HEAD)) + list(range(-N_TAIL, 0)),
        importancia=imp))
    df_imp.to_csv(OUT_DIR / "b_importancia_por_posicion.csv", index=False)
    top = df_imp.nlargest(12, "importancia")
    log("  Posiciones más informativas:")
    for _, r in top.iterrows():
        log(f"    {r.region:<9} offset {int(r.offset):>5}   importancia {r.importancia:.4f}")
    ih, ic = imp[:N_HEAD].sum(), imp[N_HEAD:].sum()
    log(f"\n  Importancia acumulada -- cabecera: {ih:.3f} | cola: {ic:.3f}")
    resultados["b_importancia"] = dict(
        cabecera=round(float(ih), 4), cola=round(float(ic), 4),
        top=[dict(region=r.region, offset=int(r.offset),
                  importancia=round(float(r.importancia), 5))
             for _, r in top.iterrows()])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 5.4), sharey=True)
        ax[0].fill_between(range(N_HEAD), imp[:N_HEAD], color="#2c5f8a", lw=0)
        ax[0].set_title("Importancia por posición -- cabecera del archivo (bytes 0 a 511)",
                        fontsize=10)
        ax[1].fill_between(range(-N_TAIL, 0), imp[N_HEAD:], color="#c8763c", lw=0)
        ax[1].set_title("Importancia por posición -- cola del archivo (últimos 512 bytes)",
                        fontsize=10)
        for a in ax:
            a.spines[["top", "right"]].set_visible(False)
            a.grid(alpha=0.2, lw=0.5)
            a.set_ylabel("importancia")
        ax[1].set_xlabel("desplazamiento en bytes")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig_importancia_por_posicion.png", dpi=200)
        log(f"  Figura: {OUT_DIR / 'fig_importancia_por_posicion.png'}")
    except Exception as e:
        log(f"  (figura omitida: {e})")

    # ---------------------------------------------------------------- (c)
    log("\n" + "=" * 74)
    log("  (c) ABLACIÓN DE VENTANA -- ¿cuántos bytes hacen falta?")
    log("=" * 74)
    cv3 = StratifiedKFold(3, shuffle=True, random_state=42)
    configs = [("64+64", 64, 64), ("128+128", 128, 128), ("256+256", 256, 256),
               ("512+512", 512, 512), ("solo cabecera 512", 512, 0),
               ("solo cola 512", 0, 512)]
    filas_c = []
    for nombre, nh, nt in configs:
        Xc = matriz(H, T, nh, nt)
        mdl = RandomForestClassifier(**RF_PARAMS)
        mdl.n_jobs = 1
        yp = cross_val_predict(mdl, Xc, y, cv=cv3, n_jobs=N_JOBS)
        acc = accuracy_score(y, yp)
        f1 = f1_score(y, yp, average="macro", zero_division=0)
        filas_c.append(dict(ventana=nombre, n_bytes=nh + nt,
                            accuracy=round(acc, 4), f1_macro=round(f1, 4)))
        log(f"  {nombre:<20} {nh+nt:>5} bytes   exactitud {acc:.3f}   macro-F1 {f1:.3f}")
    pd.DataFrame(filas_c).to_csv(OUT_DIR / "c_ablacion_ventana.csv", index=False)
    resultados["c_ablacion_ventana"] = filas_c

    # ---------------------------------------------------------------- (d)
    log("\n" + "=" * 74)
    log("  (d) DIAGNÓSTICO DE LAS FAMILIAS DIFÍCILES")
    log("=" * 74)
    mdl = RandomForestClassifier(**RF_PARAMS)
    mdl.n_jobs = 1
    yp = cross_val_predict(mdl, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=42),
                           n_jobs=N_JOBS)
    dif_presentes = [f for f in DIFICILES if f in familias]
    cm = confusion_matrix(y, yp, labels=list(familias))
    idx = {f: i for i, f in enumerate(familias)}
    log("  ¿Se confunden entre ellas? (porcentaje de sus predicciones que cae en el grupo)")
    filas_d = []
    for f in dif_presentes:
        fila = cm[idx[f]]
        total = fila.sum()
        dentro = sum(fila[idx[g]] for g in dif_presentes)
        eh = float(np.mean([entropia(r) for r in H[y == f]]))
        et = float(np.mean([entropia(r) for r in T[y == f]]))
        filas_d.append(dict(familia=f, pct_confusion_interna=round(100 * dentro / total, 1),
                            entropia_cabecera=round(eh, 3), entropia_cola=round(et, 3)))
        log(f"    {f:<14} {100*dentro/total:>5.1f} %   "
            f"entropía cabecera {eh:.2f}  cola {et:.2f}")
    resto = [f for f in familias if f not in dif_presentes]
    eh_r = float(np.mean([entropia(r) for r in H[np.isin(y, resto)]]))
    et_r = float(np.mean([entropia(r) for r in T[np.isin(y, resto)]]))
    log(f"\n  Resto de familias:  entropía cabecera {eh_r:.2f}  cola {et_r:.2f}")
    log("  (una entropía alta y uniforme indica ausencia de estructura añadida:")
    log("   el cifrado ocupa todo el archivo y no hay marca que aprender)")
    pd.DataFrame(filas_d).to_csv(OUT_DIR / "d_familias_dificiles.csv", index=False)
    resultados["d_dificiles"] = dict(familias=filas_d,
                                     entropia_resto_cabecera=round(eh_r, 3),
                                     entropia_resto_cola=round(et_r, 3))

    (OUT_DIR / "analisis_manifiesto.json").write_text(json.dumps(dict(
        fecha=str(date.today()), raiz=str(args.raiz), n_archivos=int(len(y)),
        n_familias=int(len(familias)), por_familia=args.por_familia,
        n_head=N_HEAD, n_tail=N_TAIL, rf_params={k: str(v) for k, v in RF_PARAMS.items()},
        resultados=resultados, sklearn=sklearn.__version__,
        python=sys.version.split()[0],
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\nSalidas en: {OUT_DIR}")
    log(f"Tiempo total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
