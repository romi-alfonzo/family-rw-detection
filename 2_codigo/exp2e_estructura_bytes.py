#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""Exp. 2e -- MEJORAR el frente de archivos con rasgos ESTRUCTURALES, sin mirar metadatos.

QUÉ PROBLEMA ATACA
------------------
El Exp. 2c deja un residuo de seis familias por debajo de 0,75 de F1 que se confunden entre
ellas en el 97-99 % de sus errores. El diagnóstico ya medido (job 3630) dice dos cosas que
juntas señalan la salida:

  1. La señal dominante está en la COLA: 79,6 % de la importancia contra 20,4 % de cabecera,
     y los doce desplazamientos más importantes son todos de cola.
  2. Dos de las seis difíciles SÍ dejan estructura al final --SUNCRYPT tiene entropía de cola
     4,78 y NOTPETYA 6,58, contra 7,44 del resto-- y aun así no se identifican.

La explicación escrita en el capítulo es que ese bloque «varía en cada archivo: una clave, un
identificador de víctima o un contador». Y ahí está el punto: la representación posicional
aprende VALORES de byte en posiciones fijas, de modo que un bloque cuyo contenido cambia en
cada archivo es invisible para ella, por más que su PRESENCIA, su TAMAÑO y su ALEATORIEDAD
sean constantes dentro de la familia.

Este experimento agrega rasgos que describen la FORMA del archivo en vez de su contenido:
entropía a distintas profundidades, largo del bloque no aleatorio final, tamaño y sus restos
módulo los tamaños de bloque habituales, y estadísticos de la distribución de bytes. Ninguno
depende de qué byte concreto hay en qué posición.

  «hay 200 bytes poco aleatorios al final y el tamaño es múltiplo de 16» es un rasgo de
  familia aunque esos 200 bytes sean distintos en cada archivo.

NO USA NOMBRE NI EXTENSIÓN. Esa es la diferencia con el Exp. 2d: acá la mejora, si aparece,
es del contenido, y por lo tanto no arrastra la limitación de campaña del nombre.

COLUMNAS (las tres sobre la MISMA partición, para que los delta sean pareados)
  (1) bytes canónico ....... 512 cabecera + 512 cola, posicional. La referencia del Exp. 2c.
  (2) bytes + estructura ... (1) más los rasgos estructurales. Es la propuesta.
  (3) solo estructura ...... control: ¿los rasgos solos alcanzan, o son complementarios?

CORPUS: se corre sobre el corpus CORREGIDO (ver `es_documentacion`), que devuelve 143
archivos cifrados a BADRABBIT y 167 a NOTPETYA que el filtro por extensión descartaba. Por
eso la columna (1) NO tiene por qué reproducir el 0,912 publicado: la base es otra, y medir
cuánto se movió es parte del resultado. Se compara e informa, no se aborta.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

N_HEAD = 512
N_TAIL = 512
# Profundidad que se lee de cada extremo para los rasgos estructurales. No entra al modelo
# como bytes: solo se usa para calcular entropías y el largo del bloque final no aleatorio.
N_PERFIL = 4096
N_BLOQUES_MEDIO = 8          # bloques de 512 repartidos por el archivo, solo para entropía

# Los mismos del Exp. 2c (HIPER_2C en clasificador_bytes.py). No se vuelve a buscar: lo que
# se mide es el aporte de los rasgos, no una nueva selección de modelo.
HIPER = dict(n_estimators=300, max_depth=20, min_samples_leaf=2, max_features=0.3)
REF_2C = 0.9120              # exactitud publicada (job 3648, 10 semillas, corpus SIN corregir)

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or max(1, (os.cpu_count() or 2) - 1)

MAGIAS_EN_CLARO = {
    b"\xff\xd8\xff": "JPEG", b"%PDF": "PDF", b"PK\x03\x04": "ZIP/OOXML",
    b"\xd0\xcf\x11\xe0": "OLE/Office", b"\x89PNG": "PNG", b"GIF8": "GIF",
    b"{\\rtf": "RTF", b"\x1f\x8b": "GZIP",
}


def _magia(head):
    for firma, nombre in MAGIAS_EN_CLARO.items():
        if head.startswith(firma):
            return nombre
    return None


def es_documentacion(p, familia):
    """¿Es el PDF descriptivo de NapierOne (`<FAMILIA>.pdf`) y no una muestra cifrada?

    NO se filtra por extensión: BADRABBIT y NOTPETYA no cambian la extensión de lo que
    cifran, así que sus PDF cifrados siguen llamándose `.pdf`. Ver el commit del 2026-09-22.
    """
    return p.suffix.lower() == ".pdf" and p.stem.upper() == familia


# ============================================================
# RASGOS ESTRUCTURALES
# ============================================================
def _entropia(b):
    if not b:
        return 0.0
    c = Counter(b)
    n = len(b)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def _chi2_uniforme(b):
    """Distancia a la distribución uniforme. Alto = estructurado, bajo = aleatorio."""
    if not b:
        return 0.0
    esperado = len(b) / 256.0
    c = Counter(b)
    return sum((c.get(i, 0) - esperado) ** 2 for i in range(256)) / esperado


def _largo_cola_no_aleatoria(cola, bloque=32, umbral=4.4):
    """Cuántos bytes del final forman un bloque poco aleatorio.

    Recorre la cola de atrás hacia adelante en bloques de 32 bytes y cuenta cuántos
    consecutivos tienen entropía por debajo del umbral. Es el rasgo que busca capturar el
    pie de página que SUNCRYPT (entropía de cola 4,78) y NOTPETYA (6,58) dejan: su CONTENIDO
    cambia entre archivos, pero su PRESENCIA y su TAMAÑO son de la familia.

    El umbral 4,4 sale del techo de entropía de una muestra de 32 bytes: datos uniformemente
    aleatorios de 32 bytes dan ~4,9 bits/byte (no 8: la muestra no alcanza a visitar los 256
    valores), así que 4,4 está claramente por debajo del ruido y no lo marca.
    """
    n = 0
    for i in range(len(cola) - bloque, -1, -bloque):
        if _entropia(cola[i:i + bloque]) < umbral:
            n += bloque
        else:
            break
    return n


NOMBRES_ESTRUCTURA = (
    ["tam", "log_tam", "tam_mod16", "tam_mod512", "tam_mod4096"]
    + [f"H_cab_{n}" for n in (16, 32, 64, 128, 256, 512, 1024, 4096)]
    + [f"H_cola_{n}" for n in (16, 32, 64, 128, 256, 512, 1024, 4096)]
    + [f"H_medio_{i}" for i in range(N_BLOQUES_MEDIO)]
    + ["H_medio_media", "H_medio_desvio", "H_medio_min", "H_medio_max",
       "chi2_cab", "chi2_cola", "distintos_cab", "distintos_cola",
       "maxfrec_cab", "maxfrec_cola", "ceros_cab", "ceros_cola",
       "ascii_cola", "largo_cola_no_aleatoria", "salto_cab_cola"]
)


def rasgos_estructura(head, cola, medios, tam):
    """Descripción de la FORMA del archivo. Ningún valor de byte en ninguna posición."""
    f = [float(tam), math.log1p(tam), tam % 16, tam % 512, tam % 4096]
    for n in (16, 32, 64, 128, 256, 512, 1024, 4096):
        f.append(_entropia(head[:n]))
    for n in (16, 32, 64, 128, 256, 512, 1024, 4096):
        f.append(_entropia(cola[-n:]))
    hm = [_entropia(b) for b in medios]
    hm = (hm + [0.0] * N_BLOQUES_MEDIO)[:N_BLOQUES_MEDIO]
    f.extend(hm)
    arr = np.asarray(hm, dtype=np.float64)
    f.extend([float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())])
    c512, t512 = head[:512], cola[-512:]
    f.append(_chi2_uniforme(c512))
    f.append(_chi2_uniforme(t512))
    f.append(float(len(set(c512))))
    f.append(float(len(set(t512))))
    f.append(float(max(Counter(c512).values())) if c512 else 0.0)
    f.append(float(max(Counter(t512).values())) if t512 else 0.0)
    f.append(float(c512.count(0)))
    f.append(float(t512.count(0)))
    f.append(sum(1 for b in t512 if 32 <= b < 127) / max(1, len(t512)))
    f.append(float(_largo_cola_no_aleatoria(cola)))
    f.append(_entropia(c512) - _entropia(t512))
    return f


# ============================================================
# CARGA
# ============================================================
def leer(path):
    """Devuelve (cabecera 512, cola 512, perfil de cabeza, perfil de cola, medios, tamaño)."""
    with open(path, "rb") as fh:
        cab_perfil = fh.read(N_PERFIL)
        fh.seek(0, 2)
        tam = fh.tell()
        fh.seek(max(0, tam - N_PERFIL))
        cola_perfil = fh.read(N_PERFIL)
        medios = []
        if tam > 2 * N_PERFIL:
            paso = tam // (N_BLOQUES_MEDIO + 1)
            for i in range(1, N_BLOQUES_MEDIO + 1):
                fh.seek(min(i * paso, max(0, tam - 512)))
                medios.append(fh.read(512))
    head = cab_perfil[:N_HEAD].ljust(N_HEAD, b"\x00")
    cola = cola_perfil[-N_TAIL:].rjust(N_TAIL, b"\x00") if cola_perfil else b"\x00" * N_TAIL
    return head, cola, cab_perfil, cola_perfil, medios, tam


def cargar(raiz, por_familia, seed, log=print):
    rng = np.random.default_rng(seed)
    Xb, Xe, y, familias = [], [], [], []
    sospechosos = defaultdict(list)
    for d in sorted(p for p in Path(raiz).iterdir() if p.is_dir()):
        fam = d.name.upper()
        for suf in ("-SMALL", "_SMALL", "-TINY", "_TINY"):
            fam = fam.removesuffix(suf)
        arch = sorted(p for p in d.iterdir()
                      if p.is_file() and not es_documentacion(p, fam))
        if len(arch) < 6:
            log(f"  ADVERTENCIA: {fam} tiene {len(arch)} archivos, omitida")
            continue
        sel = [arch[i] for i in rng.permutation(len(arch))[:por_familia]]
        for p in sel:
            head, cola, cabp, colap, medios, tam = leer(p)
            m = _magia(head)
            if m:
                sospechosos[fam].append((p.name, m))
            Xb.append(head + cola)
            Xe.append(rasgos_estructura(cabp, colap, medios, tam))
            y.append(fam)
        familias.append(fam)
        log(f"  {fam:<15} {len(sel):>4} de {len(arch):>4} disponibles")

    if sospechosos:
        log("\n  ⚠ ARCHIVOS QUE PARECEN ESTAR EN CLARO (magia de tipo conocido):")
        for fam, lista in sorted(sospechosos.items()):
            ej = ", ".join(f"{n} [{m}]" for n, m in lista[:3])
            log(f"     {fam:<15} {len(lista):>4} archivo(s)   ej.: {ej}")
    else:
        log("\n  Control de integridad: ningún archivo con magia de tipo conocido. OK.")

    Xb = np.frombuffer(b"".join(Xb), dtype=np.uint8).reshape(
        len(Xb), N_HEAD + N_TAIL).astype(np.float32)
    Xe = np.asarray(Xe, dtype=np.float32)
    Xe = np.nan_to_num(Xe, nan=0.0, posinf=0.0, neginf=0.0)
    return Xb, Xe, np.array(y), sorted(set(familias)), {f: len(v) for f, v in sospechosos.items()}


# ============================================================
# EVALUACIÓN
# ============================================================
def evaluar(X, y, semilla, folds):
    clf = RandomForestClassifier(random_state=semilla, n_jobs=1,
                                 class_weight="balanced", **HIPER)
    cv = StratifiedKFold(folds, shuffle=True, random_state=semilla)
    yp = cross_val_predict(clf, X, y, cv=cv, n_jobs=N_JOBS)
    return dict(accuracy=round(accuracy_score(y, yp), 4),
                balanced_accuracy=round(balanced_accuracy_score(y, yp), 4),
                f1_macro=round(f1_score(y, yp, average="macro", zero_division=0), 4)), yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz")
    ap.add_argument("--por-familia", type=int, default=500)
    ap.add_argument("--semillas", default="0,1,2,3,4")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--prueba", action="store_true", help="corrida mínima de humo")
    args = ap.parse_args()
    if args.prueba:
        args.por_familia, args.semillas, args.folds = 30, "0", 2
    semillas = [int(s) for s in args.semillas.split(",") if s.strip()]

    aqui = Path(__file__).resolve().parent
    base = aqui.parent / "4_resultados" if (aqui.parent / "4_resultados").is_dir() else aqui
    out = args.salida or (base / ("resultados_exp2e_job" +
                                  os.environ.get("SLURM_JOB_ID", "local")))
    if out.exists() and any(out.iterdir()):
        sys.exit(f"ABORTA: {out} ya tiene resultados. Borrarla o pasar --salida.")
    out.mkdir(parents=True, exist_ok=True)
    reg = open(out / "log.txt", "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        reg.write(m + "\n")
        reg.flush()

    log("=" * 78)
    log("  EXPERIMENTO 2e -- RASGOS ESTRUCTURALES SOBRE LOS BYTES (sin nombre ni extensión)")
    log("=" * 78)
    log(f"  datos: {args.raiz}")
    log(f"  {args.por_familia} archivos/familia · semillas {semillas} · {args.folds} folds")
    log(f"  {len(NOMBRES_ESTRUCTURA)} rasgos estructurales · núcleos {N_JOBS}")
    log("  CORPUS CORREGIDO: incluye los .pdf cifrados de BADRABBIT y NOTPETYA que el filtro")
    log("  por extensión descartaba. La columna (1) NO tiene por qué dar el 0,912 publicado.")

    filas, en_claro, disponibles = [], {}, {}
    for semilla in semillas:
        log(f"\n--- semilla {semilla} ---")
        t0 = time.time()
        Xb, Xe, y, familias, sosp = cargar(args.raiz, args.por_familia, semilla, log)
        if sosp:
            en_claro = sosp
        columnas = {
            "1_bytes_canonico": Xb,
            "2_bytes_mas_estructura": np.hstack([Xb, Xe]),
            "3_solo_estructura": Xe,
        }
        log(f"\n  {len(y)} archivos · {len(familias)} familias")
        for nombre, X in columnas.items():
            m, yp = evaluar(X, y, semilla, args.folds)
            m.update(columna=nombre, semilla=semilla, n_caracteristicas=X.shape[1],
                     n_archivos=len(y), n_familias=len(familias))
            filas.append(m)
            log(f"    {nombre:<24} exactitud {m['accuracy']:.4f} | "
                f"bal {m['balanced_accuracy']:.4f} | macro-F1 {m['f1_macro']:.4f}")
            if semilla == semillas[0]:
                rep = classification_report(y, yp, zero_division=0, output_dict=True)
                pd.DataFrame(rep).T.to_csv(out / f"por_familia_{nombre}.csv")
        log(f"  ({round(time.time() - t0)} s)")
        pd.DataFrame(filas).to_csv(out / "exp2e_por_semilla.csv", index=False)

    df = pd.DataFrame(filas)
    log("\n" + "=" * 78)
    log("  RESUMEN (media ± desvío sobre las semillas)")
    log("=" * 78)
    res = df.groupby("columna")[["accuracy", "balanced_accuracy", "f1_macro"]].agg(
        ["mean", "std"]).round(4)
    log(res.to_string())
    res.to_csv(out / "exp2e_resumen.csv")

    m1 = float(res.loc["1_bytes_canonico", ("accuracy", "mean")])
    log(f"\n  Efecto del CORPUS CORREGIDO sobre la columna canónica:")
    log(f"     {m1:.4f} con los 310 archivos devueltos  vs  {REF_2C:.4f} publicado "
        f"({m1 - REF_2C:+.4f})")

    from scipy import stats
    base_f = df[df.columna == "1_bytes_canonico"].set_index("semilla")
    log("\n  Delta PAREADO por semilla contra «bytes canónico»:")
    filas_d = []
    for col in ("2_bytes_mas_estructura", "3_solo_estructura"):
        v = df[df.columna == col].set_index("semilla")
        for met in ("accuracy", "f1_macro"):
            d = (v[met] - base_f[met]).dropna()
            if len(d) < 2:
                log(f"    {col:<24} {met:<10} {d.mean():+.4f}  (n={len(d)}, sin IC)")
                filas_d.append(dict(columna=col, metrica=met, delta=round(d.mean(), 4)))
                continue
            ee = d.std(ddof=1) / np.sqrt(len(d))
            t = stats.t.ppf(0.975, len(d) - 1)
            log(f"    {col:<24} {met:<10} {d.mean():+.4f} "
                f"[{d.mean() - t * ee:+.4f}; {d.mean() + t * ee:+.4f}]  "
                f"{int((d > 0).sum())}/{len(d)} semillas")
            filas_d.append(dict(columna=col, metrica=met, delta=round(d.mean(), 4),
                                ic95_inf=round(d.mean() - t * ee, 4),
                                ic95_sup=round(d.mean() + t * ee, 4),
                                semillas_pos=int((d > 0).sum()), n=len(d)))
    pd.DataFrame(filas_d).to_csv(out / "exp2e_deltas.csv", index=False)

    # Comparación por familia entre (1) y (2), que es lo que decide si la mejora cae donde
    # tiene que caer: las seis difíciles del Exp. 2c.
    try:
        a = pd.read_csv(out / "por_familia_1_bytes_canonico.csv", index_col=0)
        b = pd.read_csv(out / "por_familia_2_bytes_mas_estructura.csv", index_col=0)
        fam = [i for i in a.index if i.isupper() and i in b.index]
        comp = pd.DataFrame({"f1_bytes": a.loc[fam, "f1-score"],
                             "f1_con_estructura": b.loc[fam, "f1-score"]})
        comp["delta"] = (comp.f1_con_estructura - comp.f1_bytes).round(4)
        comp = comp.sort_values("delta", ascending=False).round(4)
        comp.to_csv(out / "exp2e_por_familia_delta.csv")
        log("\n" + "-" * 78)
        log("  DÓNDE CAE LA MEJORA (semilla %d, F1 por familia)" % semillas[0])
        log("-" * 78)
        log(comp.to_string())
        dificiles = ["SUNCRYPT", "WASTEDLOCKER", "CRYPTOLOCKER", "DARKSIDE", "JIGSAW", "NOTPETYA"]
        hay = [f for f in dificiles if f in comp.index]
        if hay:
            log(f"\n  Las seis difíciles del Exp. 2c: delta medio "
                f"{comp.loc[hay, 'delta'].mean():+.4f}")
            log(f"  Las demás:                      delta medio "
                f"{comp.drop(index=hay)['delta'].mean():+.4f}")
    except Exception as e:
        log(f"\n  (no se pudo armar la comparación por familia: {e})")

    (out / "manifiesto_exp2e.json").write_text(json.dumps(dict(
        fecha=str(date.today()), raiz=str(args.raiz), por_familia=args.por_familia,
        semillas=semillas, folds=args.folds, hiperparametros=HIPER,
        n_head=N_HEAD, n_tail=N_TAIL, n_perfil=N_PERFIL,
        rasgos_estructura=NOMBRES_ESTRUCTURA,
        usa_nombre_o_extension=False,
        corpus="corregido: es_documentacion() en vez de filtro por extension .pdf",
        referencia_2c_publicada=REF_2C,
        archivos_con_magia_en_claro=en_claro,
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n  Salidas en: {out}")
    reg.close()


if __name__ == "__main__":
    main()
