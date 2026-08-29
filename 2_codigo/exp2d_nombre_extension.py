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

Uso:
    python3.11 exp2d_nombre_extension.py /ruta/Napierone-small
    python3.11 exp2d_nombre_extension.py /ruta --por-familia 500 --semillas 0,1,2 --folds 5
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
HIPER = dict(n_estimators=300, max_depth=None, min_samples_leaf=1, max_features="sqrt")
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


def cargar(raiz, por_familia, seed):
    rng = np.random.default_rng(seed)
    Xb, y, nombres, familias = [], [], [], []
    raiz = Path(raiz)
    for d in sorted(p for p in raiz.iterdir() if p.is_dir()):
        fam = d.name.upper()
        for suf in ("-SMALL", "_SMALL", "-TINY", "_TINY"):
            fam = fam.removesuffix(suf)
        arch = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() != ".pdf")
        if len(arch) < 6:
            print(f"  ADVERTENCIA: {fam} tiene {len(arch)} archivos, omitida", flush=True)
            continue
        sel = [arch[i] for i in rng.permutation(len(arch))[:por_familia]]
        for p in sel:
            h, t = leer_bytes(p)
            Xb.append(h + t)
            y.append(fam)
            nombres.append(p.name)
        familias.append(fam)
        print(f"  {fam:<15} {len(sel):>4} archivos", flush=True)
    return Xb, np.array(y), np.array(nombres, dtype=object), sorted(set(familias))


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
    ap.add_argument("--smoke", action="store_true",
                    help="corrida mínima de cableado: 40 archivos/familia, 1 semilla, 2 folds")
    args = ap.parse_args()

    if args.smoke:
        args.por_familia, args.semillas, args.folds = 40, "0", 2
    semillas = [int(s) for s in args.semillas.split(",") if s.strip()]

    out = args.salida or (_AQUI.parent / "4_resultados" /
                          ("resultados_exp2d_job" + os.environ.get("SLURM_JOB_ID", "local")))
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

    filas, diag = [], None
    for semilla in semillas:
        log(f"\n--- semilla {semilla} ---")
        t0 = time.time()
        Xb, y, nombres, familias = cargar(args.raiz, args.por_familia, semilla)
        if diag is None:
            diag = diagnostico_extension(nombres, y, log)

        Xb_pos = bytes_posicional(Xb)
        Xforma = forma_del_nombre(nombres)
        Xext, vocab, _ = extension_literal(nombres)

        columnas = {
            "1_solo_bytes": Xb_pos,
            "2_bytes_mas_forma_del_nombre": np.hstack([Xb_pos, Xforma]),
            "3_bytes_mas_extension_literal": np.hstack([Xb_pos, Xext]),
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

    df = pd.DataFrame(filas)
    df.to_csv(out / "exp2d_por_semilla.csv", index=False)

    log("\n" + "=" * 78)
    log("  RESUMEN (media ± desvío sobre las semillas)")
    log("=" * 78)
    res = df.groupby("columna")[["accuracy", "balanced_accuracy", "f1_macro"]].agg(
        ["mean", "std"]).round(4)
    log(res.to_string())
    res.to_csv(out / "exp2d_resumen.csv")

    # deltas pareados contra la columna 1
    base = df[df.columna == "1_solo_bytes"].set_index("semilla")
    log("\n  Delta PAREADO por semilla contra «solo bytes»:")
    filas_d = []
    for col in ("2_bytes_mas_forma_del_nombre", "3_bytes_mas_extension_literal"):
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
        fecha=str(date.today()), experimento="2d", datos=str(args.raiz),
        por_familia=args.por_familia, semillas=semillas, folds=args.folds,
        hiperparametros=HIPER, n_head=N_HEAD, n_tail=N_TAIL,
        caracteristicas_forma=NOMBRES_FORMA,
        diagnostico_extension=diag,
        aclaracion_columna3=("cota superior declarada: la extensión identifica la campaña, "
                             "no la familia; no es un método propuesto"),
        sklearn=__import__("sklearn").__version__, python=sys.version.split()[0],
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n  Salidas en: {out}")
    _f.close()


if __name__ == "__main__":
    main()
