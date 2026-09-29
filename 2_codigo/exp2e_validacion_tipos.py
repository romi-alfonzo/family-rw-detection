#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""Exp. 2e-c -- ¿La configuración canónica nueva (bytes + estructura) generaliza a tipos de
documento nunca vistos, igual que lo hizo la de solo bytes?

POR QUÉ HACE FALTA
------------------
Romina decidió (2026-09-28) que el 0,936 del Exp. 2e sea la cifra canónica del frente de
archivos, en lugar del 0,912 de solo bytes. Eso deja un hueco: el 0,912 pasó por la
validación «dejar-un-tipo-fuera» (§subsec:exp2c_tipos: se entrena sin un tipo de documento y
se evalúa sobre ese tipo, promedio 0,879 de exactitud sobre siete pliegues, caída de 0,031),
que es lo que descarta la objeción «el clasificador aprende el documento de origen y no el
ransomware». El 0,936 NO pasó por ella. Un jurado puede preguntarlo.

Y hay un riesgo concreto, no retórico: entre los 44 rasgos estructurales están el TAMAÑO del
archivo y sus restos, y el tamaño correlaciona con el tipo de documento. Si la representación
estructural se apoya en «los PDF miden tanto», al dejar los PDF fuera del entrenamiento
debería caer MÁS que los bytes. Si se apoya en «cuánto agrega la familia» (que es la
interpretación del Exp. 2e-b, con tam_mod16 como rasgo principal), debería resistir igual.
Este experimento distingue las dos cosas.

DISEÑO
------
Réplica exacta de la validación del 2c (analisis_bytes.py, sección (a)): el tipo de documento
sale del nombre (`0143-pdf.pdf` -> pdf); son tipos evaluables los que tienen >= 200 archivos;
para cada tipo se entrena con TODOS los demás y se evalúa sobre ese tipo; se omite el pliegue
si en la prueba hay menos de 10 familias; el macro-F1 se calcula sobre las familias presentes
en la prueba. Un solo ajuste por pliegue, mismos hiperparámetros del 2c y misma semilla 42.

La diferencia con el 2c es que cada pliegue se evalúa con DOS representaciones sobre la misma
partición, para que el delta sea pareado:
  (1) solo bytes 512+512 .......... la referencia, re-medida sobre la base actual
  (2) bytes + 44 rasgos ........... la configuración canónica nueva

Base: 30 familias y corpus corregido (con los .pdf cifrados de BADRABBIT y NOTPETYA), así que
la columna (1) NO tiene por qué reproducir el 0,879 publicado, que era sobre 29 familias sin
esos archivos. Se compara e informa.

PREREGISTRO -- escrito y commiteado ANTES de correr (2026-09-28)
----------------------------------------------------------------
P1. La columna (1) promedia entre 0,86 y 0,90 de exactitud: cerca del 0,879 publicado, con
    margen por el cambio de base (30 familias, 310 archivos más).
P2. El delta (2)-(1) promedio en macro-F1 es POSITIVO pero MENOR que el +0,0246 medido bajo
    validación cruzada aleatoria: entre 0,000 y +0,025. Razón: parte de los rasgos de tamaño
    dependen del tipo y se pierden al dejarlo fuera, pero tam_mod16 y las entropías no.
P3. Los pliegues con MENOR delta son «pdf» y «jpg», que son los tipos con distribución de
    tamaño más distinta del resto (el 2c ya mostró que jpg es el pliegue más difícil).
P4. En ningún pliegue la columna (2) queda por debajo de la (1) en más de 0,02 de macro-F1.
    Si esto falla, la representación estructural SÍ aprende el tipo de documento en alguna
    medida, y la tesis tiene que declararlo junto al 0,936.

Cómo se lee cada desenlace:
  - P2 y P4 cumplen -> el 0,936 pasa la misma prueba que el 0,912. Se escribe como tal.
  - P4 falla en uno o dos pliegues -> el 0,936 se sostiene pero con la salvedad del tipo.
  - El delta promedio es negativo -> la mejora del 2e es en parte «aprender el documento»;
    el canónico vuelve a ser el 0,912 y el 2e queda como mejora con limitación declarada.

Reutiliza la lectura y los rasgos de `exp2e_estructura_bytes.py`; no toca ningún canónico.
"""

import argparse
import json
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
from sklearn.metrics import accuracy_score, classification_report, f1_score

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))
from exp2e_estructura_bytes import (  # noqa: E402
    N_HEAD, N_JOBS, N_TAIL, _magia, es_documentacion, leer, rasgos_estructura)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Hiperparámetros del Exp. 2c, pero SIN class_weight. ⚠ Este comentario decía «sin class_weight,
# como allí», y era FALSO: analisis_bytes.py (la validación por tipos publicada) usa
# class_weight="balanced". Detectado el 2026-09-28. Se deja el modelo como estaba para que la
# corrida del job 4079 sea reproducible; la medición con ponderación es validacion_tipos_ponderada.py.
RF_PARAMS = dict(n_estimators=300, max_depth=20, min_samples_leaf=2, max_features=0.3,
                 random_state=42, n_jobs=N_JOBS)
MIN_ARCHIVOS_TIPO = 200
MIN_FAMILIAS_PRUEBA = 10
REF_2C = dict(accuracy=0.879, f1_macro=0.861)   # promedio publicado, 29 familias, 7 pliegues
DIFICILES = ["SUNCRYPT", "WASTEDLOCKER", "CRYPTOLOCKER", "DARKSIDE", "JIGSAW", "NOTPETYA"]


def tipo_documento(nombre):
    """De '0001-jpg-fromweb.jpg.avos2' devuelve 'jpg'; de '0143-pdf.pdf' devuelve 'pdf'.
    Idéntica a analisis_bytes.py::tipo_documento."""
    m = re.match(r"^\d+-([a-z0-9]+)", nombre.lower())
    return m.group(1) if m else "desconocido"


def cargar(raiz, por_familia, seed, log=print):
    """Igual que exp2e_estructura_bytes.cargar, pero conserva el TIPO de documento."""
    rng = np.random.default_rng(seed)
    Xb, Xe, y, tipos, familias = [], [], [], [], []
    sospechosos = defaultdict(list)
    for d in sorted(p for p in Path(raiz).iterdir() if p.is_dir()):
        fam = d.name.upper()
        for suf in ("-SMALL", "_SMALL", "-TINY", "_TINY"):
            fam = fam.removesuffix(suf)
        arch = sorted(p for p in d.iterdir() if p.is_file() and not es_documentacion(p, fam))
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
            tipos.append(tipo_documento(p.name))
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
    Xe = np.nan_to_num(np.asarray(Xe, dtype=np.float32))
    return Xb, Xe, np.array(y), np.array(tipos), sorted(set(familias))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz")
    ap.add_argument("--por-familia", type=int, default=500)
    ap.add_argument("--semilla", type=int, default=0, help="semilla del MUESTREO de archivos")
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--prueba", action="store_true")
    args = ap.parse_args()
    if args.prueba:
        # humo: umbrales pensados para NapierOne, bajados solo aca y solo para probar
        args.por_familia = 60
        global MIN_ARCHIVOS_TIPO, MIN_FAMILIAS_PRUEBA
        MIN_ARCHIVOS_TIPO, MIN_FAMILIAS_PRUEBA = 20, 2

    base = _AQUI.parent / "4_resultados" if (_AQUI.parent / "4_resultados").is_dir() else _AQUI
    out = args.salida or (base / ("resultados_exp2e_tipos_job" +
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
    log("  EXP. 2e-c -- DEJAR-UN-TIPO-FUERA: ¿bytes + estructura generaliza como solo bytes?")
    log("=" * 78)
    log(f"  datos: {args.raiz} · {args.por_familia} archivos/familia · semilla de muestreo "
        f"{args.semilla} · núcleos {N_JOBS}")
    log(f"  modelo: RandomForest {RF_PARAMS['n_estimators']}/{RF_PARAMS['max_depth']}/"
        f"{RF_PARAMS['min_samples_leaf']}/{RF_PARAMS['max_features']}, semilla 42, un ajuste "
        f"por pliegue (idéntico al 2c)")

    t0 = time.time()
    Xb, Xe, y, tipos, familias = cargar(args.raiz, args.por_familia, args.semilla, log)
    X2 = np.hstack([Xb, Xe])
    log(f"\n  {len(y)} archivos · {len(familias)} familias  ({round(time.time() - t0)} s)")
    cuenta = Counter(tipos)
    log(f"  Tipos de documento: {dict(sorted(cuenta.items()))}")
    candidatos = sorted(t for t, c in cuenta.items()
                        if c >= MIN_ARCHIVOS_TIPO and t != "desconocido")
    sin_tipo = sorted({f for f, t in zip(y, tipos)
                       if t == "desconocido" or cuenta[t] < MIN_ARCHIVOS_TIPO})
    nunca_en_prueba = sorted(set(y) - set(y[np.isin(tipos, candidatos)]))
    log(f"  Tipos evaluables: {candidatos}")
    if sin_tipo:
        # Tener ALGÚN archivo sin tipo no saca a la familia de la prueba (BLACKMATTER renombra
        # 13 de ~1.000): lo que solo entrena son esos archivos. Las familias que no entran a
        # ninguna prueba se calculan aparte.
        log(f"  Familias con archivos sin tipo reconocible (esos archivos solo entrenan): {sin_tipo}")
    log(f"  Familias que no entran a ningún pliegue de prueba: {nunca_en_prueba or 'ninguna'}")

    columnas = {"1_solo_bytes": Xb, "2_bytes_mas_estructura": X2}
    filas, porfam = [], []
    log("\n  " + f"{'tipo':<8} {'n':>5} {'fam':>4}   {'bytes acc':>9} {'bytes F1':>9}   "
        f"{'b+e acc':>9} {'b+e F1':>9}   {'Δ acc':>7} {'Δ F1':>7}")
    for tipo in candidatos:
        te = np.flatnonzero(tipos == tipo)
        tr = np.flatnonzero(tipos != tipo)
        fams_te = np.unique(y[te])
        if len(fams_te) < MIN_FAMILIAS_PRUEBA:
            log(f"  {tipo:<8} omitido (solo {len(fams_te)} familias en la prueba)")
            continue
        r = dict(tipo_excluido=tipo, n_prueba=int(len(te)), n_familias=int(len(fams_te)))
        for nombre, X in columnas.items():
            m = RandomForestClassifier(**RF_PARAMS).fit(X[tr], y[tr])
            yp = m.predict(X[te])
            r[f"acc_{nombre}"] = round(accuracy_score(y[te], yp), 4)
            r[f"f1_{nombre}"] = round(f1_score(y[te], yp, average="macro",
                                               labels=fams_te, zero_division=0), 4)
            rep = classification_report(y[te], yp, labels=fams_te, zero_division=0,
                                        output_dict=True)
            for f in fams_te:
                porfam.append(dict(tipo_excluido=tipo, columna=nombre, familia=f,
                                   f1=round(rep[f]["f1-score"], 4)))
        r["delta_acc"] = round(r["acc_2_bytes_mas_estructura"] - r["acc_1_solo_bytes"], 4)
        r["delta_f1"] = round(r["f1_2_bytes_mas_estructura"] - r["f1_1_solo_bytes"], 4)
        filas.append(r)
        log(f"  {tipo:<8} {len(te):>5} {len(fams_te):>4}   "
            f"{r['acc_1_solo_bytes']:>9.4f} {r['f1_1_solo_bytes']:>9.4f}   "
            f"{r['acc_2_bytes_mas_estructura']:>9.4f} {r['f1_2_bytes_mas_estructura']:>9.4f}   "
            f"{r['delta_acc']:>+7.4f} {r['delta_f1']:>+7.4f}")
        pd.DataFrame(filas).to_csv(out / "tipos_por_pliegue.csv", index=False)

    if not filas:
        log("\n  Ningún pliegue evaluable.")
        return
    df = pd.DataFrame(filas)
    pd.DataFrame(porfam).to_csv(out / "tipos_por_familia.csv", index=False)

    log("\n" + "=" * 78)
    log("  PROMEDIO SOBRE LOS PLIEGUES")
    log("=" * 78)
    for nombre in columnas:
        log(f"  {nombre:<24} exactitud {df[f'acc_{nombre}'].mean():.4f} | "
            f"macro-F1 {df[f'f1_{nombre}'].mean():.4f}")
    d_acc, d_f1 = df.delta_acc, df.delta_f1
    from scipy import stats
    n = len(df)
    t = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")

    def ic(s):
        ee = s.std(ddof=1) / np.sqrt(len(s))
        return f"{s.mean():+.4f} [{s.mean() - t * ee:+.4f}; {s.mean() + t * ee:+.4f}]"

    log(f"\n  Δ (2)-(1) pareado por pliegue, n={n}:")
    log(f"     exactitud {ic(d_acc)}   {int((d_acc > 0).sum())}/{n} pliegues a favor")
    log(f"     macro-F1  {ic(d_f1)}   {int((d_f1 > 0).sum())}/{n} pliegues a favor")

    # las seis dificiles, promediadas sobre los pliegues donde aparecen
    pf = pd.DataFrame(porfam)
    dif = pf[pf.familia.isin(DIFICILES)].pivot_table(index="familia", columns="columna",
                                                     values="f1", aggfunc="mean")
    if not dif.empty and set(columnas) <= set(dif.columns):
        dif["delta"] = dif["2_bytes_mas_estructura"] - dif["1_solo_bytes"]
        log("\n  Las seis difíciles bajo dejar-un-tipo-fuera (F1 medio sobre los pliegues):")
        log(dif.round(4).to_string())

    log("\n" + "-" * 78)
    log("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    log("-" * 78)
    a1 = df.acc_1_solo_bytes.mean()
    p1 = 0.86 <= a1 <= 0.90
    log(f"  [{'CUMPLE' if p1 else 'FALLA '}] P1 bytes promedia 0,86-0,90 de exactitud      "
        f"{a1:.4f} (publicado 0,879 sobre 29 familias)")
    p2 = 0.0 <= d_f1.mean() <= 0.025
    log(f"  [{'CUMPLE' if p2 else 'FALLA '}] P2 Δ macro-F1 promedio en [0,000; +0,025]      "
        f"{d_f1.mean():+.4f} (bajo VC aleatoria era +0,0246)")
    peores = df.nsmallest(2, "delta_f1").tipo_excluido.tolist()
    p3 = set(peores) == {"pdf", "jpg"}
    log(f"  [{'CUMPLE' if p3 else 'FALLA '}] P3 los dos pliegues con menor Δ son pdf y jpg  "
        f"{peores}")
    peor = df.delta_f1.min()
    p4 = peor >= -0.02
    log(f"  [{'CUMPLE' if p4 else 'FALLA '}] P4 ningún pliegue con Δ macro-F1 < -0,02       "
        f"mínimo {peor:+.4f} ({df.loc[df.delta_f1.idxmin(), 'tipo_excluido']})")

    log("\n  LECTURA:")
    if p2 and p4:
        log("    El 0,936 pasa la misma prueba que el 0,912: los rasgos estructurales no")
        log("    aprenden el tipo de documento. Se escribe como validación del canónico.")
    elif d_f1.mean() < 0:
        log("    La mejora del 2e es en parte «aprender el documento»: bajo tipo no visto")
        log("    los rasgos estructurales PERJUDICAN. El canónico vuelve a ser el 0,912.")
    else:
        log("    La mejora se sostiene en promedio pero algún tipo la rompe: el 0,936 va con")
        log("    la salvedad del tipo de documento declarada al lado.")

    (out / "manifiesto.json").write_text(json.dumps(dict(
        fecha=str(date.today()), raiz=str(args.raiz), por_familia=args.por_familia,
        semilla_muestreo=args.semilla, rf_params=RF_PARAMS, tipos_evaluables=candidatos,
        familias_sin_tipo=sin_tipo, referencia_2c_publicada=REF_2C,
        preregistro=dict(P1=bool(p1), P2=bool(p2), P3=bool(p3), P4=bool(p4)),
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n  Salidas en: {out}")
    reg.close()


if __name__ == "__main__":
    main()
