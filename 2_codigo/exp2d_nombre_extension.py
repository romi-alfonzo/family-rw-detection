#!/usr/bin/env python3
"""
exp2d_nombre_extension.py -- Experimento 2d: ¿cuánto agrega el NOMBRE del archivo
sobre los bytes, y cuánto de eso es legítimo?

EL PEDIDO. En la reunión del 12-08-2026 el tutor pidió cuantificar el aporte del nombre y
de la extensión del archivo, que el Exp. 2c excluye deliberadamente (`clasificador_bytes.py`
declara `usa_nombre_o_extension=False`). El Exp. 2b ya había medido que la extensión
aportaba el 82,8 % de las marcas estructurales, y la razón de excluirla era que en
NapierOne la extensión es un identificador de CAMPAÑA, no de familia.

TRES COLUMNAS, Y LA TERCERA NO ES UNA PROPUESTA
  (1) Solo bytes .................. la configuración canónica del Exp. 2c. Es la referencia.
  (2) Bytes + FORMA del nombre .... agrega rasgos de la forma del nombre (longitud,
      composición de caracteres, entropía, aspecto del identificador), SIN usar la
      extensión literal. Es lo defendible: en un incidente real el nombre está a la vista,
      y su forma puede diferir entre familias aunque la extensión concreta sea nueva.
  (3) Bytes + EXTENSIÓN LITERAL ... agrega la extensión como variable categórica. Es una
      COTA SUPERIOR DECLARADA, no un método propuesto: mide cuánto se podría alcanzar si
      la extensión de la campaña evaluada ya se conociera.

POR QUÉ LA TERCERA ES CIRCULAR, Y CÓMO SE DEMUESTRA EN LUGAR DE AFIRMARLO
NapierOne representa cada familia con una sola campaña, de modo que es previsible que cada
familia tenga UNA extensión y que la extensión determine la etiqueta. El script MIDE eso
antes de entrenar: cuenta extensiones distintas por familia y familias por extensión, y
calcula qué exactitud alcanza una tabla de consulta que solo mira la extensión. Si esa
tabla ya acierta casi todo, la columna (3) no mide capacidad de identificar familias sino
la memorización de un diccionario, y así hay que reportarla.

PROTOCOLO. Idéntico al del Exp. 2c para que las columnas sean comparables: mismos archivos,
mismos pliegues, misma semilla, mismo RandomForest. La única diferencia entre columnas es
el conjunto de características. La comparación es PAREADA: las tres columnas se evalúan
sobre la misma partición en cada semilla.

ADEMÁS, A.3: LA CURVA DE APRENDIZAJE
El tutor pidió la curva en los dos frentes. Va en este mismo job porque comparte la carga
de datos, que es lo caro. Mide exactitud y macro-F1 contra archivos por familia, sobre
subconjuntos ANIDADOS y solo con bytes, y reporta el delta pareado entre tamaños
consecutivos con IC 95 % — el mismo criterio con el que se leyó B.1 en el frente de notas.

Uso:
    python3.11 exp2d_nombre_extension.py /ruta/Napierone-small
    python3.11 exp2d_nombre_extension.py /ruta --por-familia 500 --semillas 0,1,2 --folds 5
    python3.11 exp2d_nombre_extension.py /ruta --tamanos ""      # 2d sin la curva
    python3.11 exp2d_nombre_extension.py /ruta --smoke
"""
from __future__ import annotations

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

N_HEAD = 512
N_TAIL = 512
# Hiperparámetros ya elegidos por la búsqueda anidada del Exp. 2c: NO se vuelve a buscar.
# Lo que se mide es el aporte de las características, no una nueva selección de modelo.
# Tienen que ser EXACTAMENTE los de `clasificador_bytes.py::HIPER_2C`, porque si no la
# columna (1) deja de reproducir la referencia publicada (0,912 ± 0,002 de exactitud) y las
# tres columnas dejan de ser comparables contra el Exp. 2c. Verificado contra tres fuentes:
# `clasificador_bytes.py` línea 88, `resultados_bytes/bytes_manifiesto.json` (job 3639) y
# el log `slurm-bytesms-3648.out`.
HIPER = dict(n_estimators=300, max_depth=20, min_samples_leaf=2, max_features=0.3)

# Referencia publicada del Exp. 2c (job 3648, 30 familias, 10 semillas): exactitud
# 0,9120 +- 0,0016. La columna (1) de este experimento usa la MISMA carga de archivos, los
# mismos hiperparametros y el mismo esquema de validacion, asi que tiene que reproducirla.
# Si no la reproduce hay una diferencia no declarada y los incrementos de (2) y (3) no se
# pueden sumar a la cifra publicada. La tolerancia son ~3 desvios de la referencia.
REF_2C, TOL_2C = 0.9120, 0.0050
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or max(1, (os.cpu_count() or 2) - 1)


# ============================================================
# CARGA: igual que el Exp. 2c pero conservando el NOMBRE del archivo
# ============================================================
def leer_bytes(path, n_head=N_HEAD, n_tail=N_TAIL):
    with open(path, "rb") as f:
        head = f.read(n_head)
        f.seek(0, 2)
        n = f.tell()
        f.seek(max(0, n - n_tail))
        tail = f.read(n_tail)
    return head.ljust(n_head, b"\x00"), tail.ljust(n_tail, b"\x00")


# Magias de archivos EN CLARO. Este experimento usa el nombre, así que un archivo sin
# cifrar que conserve su nombre original es veneno: el modelo aprendería «se llama .jpg y
# empieza con FFD8FF => CERBER». Es exactamente el caso de los 12 JPEG hallados en
# CERBER-small el 2026-08-16 (movidos a CERBER-small/_sin_cifrar/ desde el job 3639).
# `p.is_file()` ya deja fuera ese subdirectorio; esto lo verifica y lo deja en el log.
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


def cargar(raiz, por_familia, seed, log=print):
    rng = np.random.default_rng(seed)
    Xb, y, nombres, familias = [], [], [], []
    sospechosos = defaultdict(list)
    raiz = Path(raiz)
    for d in sorted(p for p in raiz.iterdir() if p.is_dir()):
        fam = d.name.upper()
        for suf in ("-SMALL", "_SMALL", "-TINY", "_TINY"):
            fam = fam.removesuffix(suf)
        arch = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() != ".pdf")
        if len(arch) < 6:
            log(f"  ADVERTENCIA: {fam} tiene {len(arch)} archivos, omitida")
            continue
        sel = [arch[i] for i in rng.permutation(len(arch))[:por_familia]]
        for p in sel:
            h, t = leer_bytes(p)
            m = _magia(h)
            if m:
                sospechosos[fam].append((p.name, m))
            Xb.append(h + t)
            y.append(fam)
            nombres.append(p.name)
        familias.append(fam)
        log(f"  {fam:<15} {len(sel):>4} archivos")

    # Control de integridad: sin esto la columna del nombre no es interpretable.
    if sospechosos:
        log("\n  ⚠ ARCHIVOS QUE PARECEN ESTAR EN CLARO (magia de tipo conocido):")
        for fam, lista in sorted(sospechosos.items()):
            ej = ", ".join(f"{n} [{m}]" for n, m in lista[:3])
            log(f"     {fam:<15} {len(lista):>4} archivo(s)   ej.: {ej}")
        log("     Revisar antes de leer los resultados: un archivo sin cifrar que conserva")
        log("     su nombre original hace trivial la columna del nombre.")
    else:
        log("\n  Control de integridad: ningún archivo con magia de tipo conocido. OK.")
    return (Xb, np.array(y), np.array(nombres, dtype=object), sorted(set(familias)),
            {f: len(v) for f, v in sospechosos.items()})


# ============================================================
# CARACTERÍSTICAS
# ============================================================
def bytes_posicional(Xb):
    return np.frombuffer(b"".join(Xb), dtype=np.uint8).reshape(
        len(Xb), N_HEAD + N_TAIL).astype(np.float32)


def _entropia(s):
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def forma_del_nombre(nombres):
    """Rasgos de la FORMA del nombre. No incluye la extensión literal ni el nombre literal.

    La distinción es la que hace defendible esta columna: se usa que el nombre tenga, por
    ejemplo, 32 caracteres hexadecimales o una extensión de 6 letras, no CUÁL es esa
    extensión. Un atacante que cambie la cadena concreta manteniendo el formato seguiría
    siendo reconocible; uno que cambie el formato, no.
    """
    filas = []
    for n in nombres:
        base, punto, ext = n.rpartition(".")
        if not punto:
            base, ext = n, ""
        d = sum(ch.isdigit() for ch in base)
        al = sum(ch.isalpha() for ch in base)
        hexa = sum(ch in "0123456789abcdefABCDEF" for ch in base)
        sep = sum(ch in "-_. " for ch in base)
        may = sum(ch.isupper() for ch in base)
        L = max(1, len(base))
        filas.append([
            len(n), len(base), len(ext),
            d, al, sep, may,
            d / L, al / L, hexa / L, sep / L, may / L,
            _entropia(base),
            1.0 if base and all(c in "0123456789abcdefABCDEF" for c in base) else 0.0,
            1.0 if re.fullmatch(r"\d+", base or "x") else 0.0,
            1.0 if ext and ext.isalpha() else 0.0,
            1.0 if ext and any(c.isdigit() for c in ext) else 0.0,
            n.count("."),
        ])
    return np.asarray(filas, dtype=np.float32)


NOMBRES_FORMA = ["len_total", "len_base", "len_ext", "n_digitos", "n_letras", "n_sep",
                 "n_mayus", "prop_digitos", "prop_letras", "prop_hex", "prop_sep",
                 "prop_mayus", "entropia_base", "base_toda_hex", "base_solo_digitos",
                 "ext_solo_letras", "ext_con_digitos", "n_puntos"]


def extension_literal(nombres, vocabulario=None):
    """La extensión como variable categórica (one-hot). Es la COTA SUPERIOR."""
    ext = [n.rpartition(".")[2].lower() if "." in n else "" for n in nombres]
    if vocabulario is None:
        vocabulario = sorted(set(ext))
    idx = {e: i for i, e in enumerate(vocabulario)}
    M = np.zeros((len(ext), len(vocabulario) + 1), dtype=np.float32)
    for i, e in enumerate(ext):
        M[i, idx.get(e, len(vocabulario))] = 1.0
    return M, vocabulario, ext


# ============================================================
# DIAGNÓSTICO: ¿la extensión ES la etiqueta?
# ============================================================
def diagnostico_extension(nombres, y, log):
    ext = [n.rpartition(".")[2].lower() if "." in n else "" for n in nombres]
    por_fam = defaultdict(set)
    por_ext = defaultdict(set)
    for e, f in zip(ext, y):
        por_fam[f].add(e)
        por_ext[e].add(f)
    log("\n" + "-" * 78)
    log("  DIAGNÓSTICO PREVIO: ¿la extensión identifica la campaña o la familia?")
    log("-" * 78)
    una = sum(1 for f, s in por_fam.items() if len(s) == 1)
    log(f"  familias con UNA sola extensión: {una} de {len(por_fam)}")
    amb = {e: s for e, s in por_ext.items() if len(s) > 1}
    log(f"  extensiones compartidas por más de una familia: {len(amb)} de {len(por_ext)}")
    for e, s in sorted(amb.items())[:6]:
        log(f"     .{e or '(sin extensión)'} -> {sorted(s)}")
    # tabla de consulta: predecir la familia más frecuente de cada extensión
    mayoria = {e: Counter(f for ee, f in zip(ext, y) if ee == e).most_common(1)[0][0]
               for e in por_ext}
    pred = np.array([mayoria[e] for e in ext])
    acc = accuracy_score(y, pred)
    log(f"  exactitud de una TABLA DE CONSULTA que solo mira la extensión: {acc:.4f}")
    if acc > 0.95:
        log("  => la extensión determina la etiqueta en este conjunto. La columna (3) mide")
        log("     memorización de un diccionario, NO capacidad de identificar familias.")
    return dict(familias_una_extension=una, n_familias=len(por_fam),
                extensiones_ambiguas=len(amb), n_extensiones=len(por_ext),
                exactitud_tabla_consulta=round(float(acc), 4))


# ============================================================
# A.3 -- CURVA DE APRENDIZAJE: ¿cuántos archivos por familia hacen falta?
# ============================================================
# El tutor pidió la curva en los DOS frentes (reunión del 12-08-2026). En notas ya está
# medida (B.1, la moneda son las plantillas); acá la moneda son los archivos por familia.
# Va en este job y no en otro porque comparte la carga de datos, que es lo caro.
#
# Los subconjuntos son ANIDADOS: para cada semilla se permutan los índices de cada familia
# una vez y cada tamaño toma un prefijo. Así la curva mide el efecto de agregar datos y no
# el ruido de volver a sortear, que es el mismo criterio con el que se midió B.1 en notas.
def curva_aprendizaje(X, y, tamanos, semillas, folds, log):
    familias = sorted(set(y.tolist()))
    idx_por_fam = {f: np.flatnonzero(y == f) for f in familias}
    filas = []
    for semilla in semillas:
        rng = np.random.default_rng(1000 + semilla)
        orden = {f: idx[rng.permutation(len(idx))] for f, idx in idx_por_fam.items()}
        for k in tamanos:
            if any(len(v) < k for v in orden.values()):
                log(f"    tamaño {k}: alguna familia no llega, se omite")
                continue
            sel = np.concatenate([v[:k] for v in orden.values()])
            m, _ = evaluar(X[sel], y[sel], semilla, folds)
            m.update(por_familia=k, semilla=semilla, n_archivos=len(sel))
            filas.append(m)
            log(f"    {k:>4} arch./familia ({len(sel):>6} en total)  "
                f"exactitud {m['accuracy']:.4f} | macro-F1 {m['f1_macro']:.4f}")
    return filas


def pasos_de_la_curva(df, log):
    """Delta pareado entre tamaños consecutivos, con IC 95 %. Mismo criterio de adopción
    que en el frente de notas: el último paso que aporta es aquel cuyo IC excluye el 0."""
    tamanos = sorted(df.por_familia.unique())
    filas = []
    log("\n  Paso a paso (delta pareado por semilla, macro-F1):")
    for a, b in zip(tamanos, tamanos[1:]):
        va = df[df.por_familia == a].set_index("semilla").f1_macro
        vb = df[df.por_familia == b].set_index("semilla").f1_macro
        d = (vb - va).dropna()
        fila = dict(desde=a, hasta=b, delta=round(float(d.mean()), 4), n=len(d))
        if len(d) >= 2:
            from scipy import stats
            ee = d.std(ddof=1) / np.sqrt(len(d))
            t = stats.t.ppf(0.975, len(d) - 1)
            lo, hi = d.mean() - t * ee, d.mean() + t * ee
            fila.update(ic95_inf=round(float(lo), 4), ic95_sup=round(float(hi), 4),
                        significativo=bool(lo > 0 or hi < 0))
            log(f"    {a:>4} -> {b:<4} {d.mean():+.4f}  [{lo:+.4f}; {hi:+.4f}]"
                f"{'  <- aporta' if lo > 0 else ''}")
        else:
            log(f"    {a:>4} -> {b:<4} {d.mean():+.4f}  (n={len(d)}, sin IC)")
        filas.append(fila)
    utiles = [f for f in filas if f.get("significativo") and f["delta"] > 0]
    if utiles:
        log(f"\n  El último paso que aporta de forma medible es "
            f"{utiles[-1]['desde']} -> {utiles[-1]['hasta']} archivos/familia.")
    elif any("ic95_inf" in f for f in filas):
        log("\n  Ningún paso tiene un IC 95 % que excluya el cero: la curva ya está plana "
            "en el rango medido.")
    else:
        log("\n  Con una sola semilla no hay IC: para leer la curva hacen falta al menos "
            "dos semillas de submuestreo (--semillas-curva).")
    return filas


# ============================================================
# EVALUACIÓN
# ============================================================
def metricas(y, yp):
    return dict(accuracy=round(accuracy_score(y, yp), 4),
                balanced_accuracy=round(balanced_accuracy_score(y, yp), 4),
                f1_macro=round(f1_score(y, yp, average="macro", zero_division=0), 4))


def evaluar(X, y, semilla, folds):
    clf = RandomForestClassifier(random_state=semilla, n_jobs=1,
                                 class_weight="balanced", **HIPER)
    cv = StratifiedKFold(folds, shuffle=True, random_state=semilla)
    yp = cross_val_predict(clf, X, y, cv=cv, n_jobs=N_JOBS)
    return metricas(y, yp), yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz", help="carpeta con subcarpetas <FAMILIA>")
    ap.add_argument("--por-familia", type=int, default=500)
    ap.add_argument("--semillas", default="0,1,2,3,4",
                    help="semillas separadas por coma (default 0-4)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--tamanos", default="10,25,50,100,200,350,500",
                    help="A.3: archivos/familia de la curva de aprendizaje. "
                         "Cadena vacía para saltear la curva.")
    ap.add_argument("--semillas-curva", default="0,1,2",
                    help="A.3: semillas de submuestreo de la curva (default 0,1,2)")
    ap.add_argument("--smoke", action="store_true",
                    help="corrida mínima de cableado: 40 archivos/familia, 1 semilla, 2 folds")
    args = ap.parse_args()

    if args.smoke:
        args.por_familia, args.semillas, args.folds = 40, "0", 2
        args.tamanos, args.semillas_curva = "10,20,40", "0"
    semillas = [int(s) for s in args.semillas.split(",") if s.strip()]
    tamanos = [int(t) for t in args.tamanos.split(",") if t.strip()]
    semillas_curva = [int(s) for s in args.semillas_curva.split(",") if s.strip()]
    tamanos = [t for t in tamanos if t <= args.por_familia]

    # En el cluster los scripts viven planos en /scratch/ralfonzo/tesis y NO existe
    # 4_resultados/: en ese caso la salida va junto al script. Misma convencion que
    # clasificador_bytes.py, y el nombre lleva el job de SLURM para no pisar corridas.
    base_salida = (_AQUI.parent / "4_resultados"
                   if (_AQUI.parent / "4_resultados").is_dir() else _AQUI)
    out = args.salida or (base_salida /
                          ("resultados_exp2d_job" + os.environ.get("SLURM_JOB_ID", "local")))
    if out.exists() and any(out.iterdir()):
        sys.exit(f"ABORTA: {out} ya tiene resultados. Usar --salida con otro nombre "
                 f"(el job 3639 borro los CSV del 3633 por sobrescribir).")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.txt"
    _f = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        _f.write(str(msg) + "\n")
        _f.flush()

    log("=" * 78)
    log("  EXPERIMENTO 2d -- BYTES, FORMA DEL NOMBRE Y EXTENSIÓN LITERAL")
    log("=" * 78)
    log(f"  datos: {args.raiz}")
    log(f"  {args.por_familia} archivos/familia · semillas {semillas} · {args.folds} folds")
    log(f"  núcleos: {N_JOBS}")

    filas, diag, en_claro = [], None, {}
    guardado = None            # bytes de la última semilla, para la curva A.3
    for semilla in semillas:
        log(f"\n--- semilla {semilla} ---")
        t0 = time.time()
        Xb, y, nombres, familias, sosp = cargar(args.raiz, args.por_familia, semilla, log)
        if sosp:
            en_claro = sosp
        if diag is None:
            diag = diagnostico_extension(nombres, y, log)

        Xb_pos = bytes_posicional(Xb)
        Xforma = forma_del_nombre(nombres)
        Xext, vocab, _ = extension_literal(nombres)

        columnas = {
            "1_solo_bytes": Xb_pos,
            "2_bytes_mas_forma_del_nombre": np.hstack([Xb_pos, Xforma]),
            "3_bytes_mas_extension_literal": np.hstack([Xb_pos, Xext]),
            # Controles SIN bytes. Hacen falta para poder LEER las columnas (2) y (3).
            # El diagnostico previo mide la circularidad de la extension LITERAL, pero no
            # la de la FORMA del nombre, y la forma es una huella de la campana igual que
            # la extension: en NapierOne cada familia es una sola campana, de modo que el
            # esquema de renombrado (largo de la extension, largo y composicion de la base)
            # puede identificar la familia sin mirar un solo byte del contenido. Si (0a)
            # sola ya alcanza casi todo, el incremento de la columna (2) no es <<los bytes
            # ayudados por el nombre>>: es el nombre, con los bytes de acompanantes.
            "0a_solo_forma_del_nombre": Xforma,
            "0b_solo_extension_literal": Xext,
        }
        log(f"\n  {len(y)} archivos · {len(familias)} familias · "
            f"{len(vocab)} extensiones distintas")
        for nombre, X in columnas.items():
            m, _yp = evaluar(X, y, semilla, args.folds)
            m.update(columna=nombre, semilla=semilla, n_caracteristicas=X.shape[1],
                     n_archivos=len(y), n_familias=len(familias))
            filas.append(m)
            log(f"    {nombre:<32} exactitud {m['accuracy']:.4f} | "
                f"bal {m['balanced_accuracy']:.4f} | macro-F1 {m['f1_macro']:.4f}")
        log(f"  ({round(time.time() - t0)} s)")
        guardado = (Xb_pos, y)
        # Guardado incremental: el job 3771 se cancelo en la semilla 4 y perdio las cuatro
        # semillas ya evaluadas porque todos los CSV se escribian recien al final. Una
        # corrida interrumpida ahora deja en disco lo que alcanzo a medir.
        pd.DataFrame(filas).to_csv(out / "exp2d_por_semilla.csv", index=False)

    df = pd.DataFrame(filas)
    df.to_csv(out / "exp2d_por_semilla.csv", index=False)

    log("\n" + "=" * 78)
    log("  RESUMEN (media ± desvío sobre las semillas)")
    log("=" * 78)
    res = df.groupby("columna")[["accuracy", "balanced_accuracy", "f1_macro"]].agg(
        ["mean", "std"]).round(4)
    log(res.to_string())
    res.to_csv(out / "exp2d_resumen.csv")

    # Puerta de entrada: la columna (1) tiene que reproducir el Exp. 2c publicado.
    try:
        m1 = float(res.loc["1_solo_bytes", ("accuracy", "mean")])
    except Exception:
        m1 = float("nan")
    if not (abs(m1 - REF_2C) <= TOL_2C):
        log("\n  ATENCION -- LA COLUMNA (1) NO REPRODUCE LA REFERENCIA DEL EXP. 2c:")
        log(f"     medido {m1:.4f} vs publicado {REF_2C:.4f} +- {TOL_2C:.4f} "
            f"(diferencia {m1 - REF_2C:+.4f})")
        log("     Los incrementos de (2) y (3) siguen siendo validos ENTRE SI, porque las")
        log("     tres columnas comparten la particion, pero NO se pueden sumar al 0,912")
        log("     publicado. Declarar la base de este experimento por separado antes de")
        log("     citar cualquier numero, y averiguar la diferencia: misma carga, mismos")
        log("     hiperparametros y misma validacion deberian dar la misma cifra.")
    else:
        log(f"\n  Puerta de entrada: la columna (1) reproduce el Exp. 2c "
            f"({m1:.4f} vs {REF_2C:.4f}). OK.")

    # deltas pareados contra la columna 1
    base = df[df.columna == "1_solo_bytes"].set_index("semilla")
    log("\n  Delta PAREADO por semilla contra «solo bytes»:")
    filas_d = []
    for col in ("2_bytes_mas_forma_del_nombre", "3_bytes_mas_extension_literal",
                "0a_solo_forma_del_nombre", "0b_solo_extension_literal"):
        v = df[df.columna == col].set_index("semilla")
        for met in ("accuracy", "f1_macro"):
            d = (v[met] - base[met]).dropna()
            if len(d) < 2:
                log(f"    {col:<32} {met:<10} {d.mean():+.4f}  (n={len(d)}, sin IC)")
                filas_d.append(dict(columna=col, metrica=met, delta=round(d.mean(), 4)))
                continue
            from scipy import stats
            ee = d.std(ddof=1) / np.sqrt(len(d))
            t = stats.t.ppf(0.975, len(d) - 1)
            log(f"    {col:<32} {met:<10} {d.mean():+.4f} "
                f"[{d.mean() - t * ee:+.4f}; {d.mean() + t * ee:+.4f}]  "
                f"{int((d > 0).sum())}/{len(d)} semillas")
            filas_d.append(dict(columna=col, metrica=met, delta=round(d.mean(), 4),
                                ic95_inf=round(d.mean() - t * ee, 4),
                                ic95_sup=round(d.mean() + t * ee, 4),
                                semillas_pos=int((d > 0).sum()), n=len(d)))
    pd.DataFrame(filas_d).to_csv(out / "exp2d_deltas.csv", index=False)

    # ------------------------------------------------------------------ A.3
    curva, pasos = [], []
    if tamanos and guardado is not None:
        Xb_pos, y = guardado
        log("\n" + "=" * 78)
        log("  A.3 -- CURVA DE APRENDIZAJE (solo bytes, la representación canónica)")
        log("=" * 78)
        log(f"  tamaños {tamanos} · semillas de submuestreo {semillas_curva} · "
            f"subconjuntos anidados")
        t0 = time.time()
        curva = curva_aprendizaje(Xb_pos, y, tamanos, semillas_curva, args.folds, log)
        dfc = pd.DataFrame(curva)
        dfc.to_csv(out / "a3_curva_por_semilla.csv", index=False)
        resc = dfc.groupby("por_familia")[["accuracy", "f1_macro"]].agg(
            ["mean", "std"]).round(4)
        log("\n  Resumen de la curva (media ± desvío sobre las semillas):")
        log(resc.to_string())
        resc.to_csv(out / "a3_curva_resumen.csv")
        pasos = pasos_de_la_curva(dfc, log)
        pd.DataFrame(pasos).to_csv(out / "a3_curva_pasos.csv", index=False)
        log(f"  ({round(time.time() - t0)} s)")
    elif tamanos:
        log("\n  A.3: no hay datos cargados para la curva, se saltea.")

    log("\n" + "-" * 78)
    log("  CÓMO REPORTAR ESTAS TRES COLUMNAS")
    log("-" * 78)
    log("  (1) y (2) son comparables y ambas son defendibles.")
    log("  (3) es una COTA SUPERIOR DECLARADA. En este conjunto cada familia corresponde")
    log("      a una sola campaña, así que la extensión funciona como identificador de la")
    log(f"      campaña: una tabla de consulta sobre la extensión ya acierta "
        f"{diag['exactitud_tabla_consulta']:.4f}.")
    log("      Presentarla como resultado del método sería reportar la memorización de un")
    log("      diccionario. Se incluye para acotar cuánto queda por encima de los bytes.")

    (out / "manifiesto_exp2d.json").write_text(json.dumps(dict(
        fecha=str(date.today()), experimento="2d + A.3", datos=str(args.raiz),
        por_familia=args.por_familia, semillas=semillas, folds=args.folds,
        hiperparametros=HIPER,
        origen_hiperparametros=("HIPER_2C de clasificador_bytes.py -- los mismos de la "
                                "búsqueda anidada del Exp. 2c, sin volver a buscar"),
        n_head=N_HEAD, n_tail=N_TAIL,
        caracteristicas_forma=NOMBRES_FORMA,
        curva_tamanos=tamanos, curva_semillas=semillas_curva,
        archivos_con_magia_en_claro=en_claro,
        diagnostico_extension=diag,
        aclaracion_columna3=("cota superior declarada: la extensión identifica la campaña, "
                             "no la familia; no es un método propuesto"),
        sklearn=__import__("sklearn").__version__, python=sys.version.split()[0],
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n  Salidas en: {out}")
    _f.close()


if __name__ == "__main__":
    main()
