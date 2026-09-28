#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""Diagnóstico de las familias que el frente de archivos no resuelve. SIN aprendizaje: mira los
bytes directamente, para diseñar el próximo rasgo a partir de los datos y no de una hipótesis.

POR QUÉ ASÍ
-----------
Con la configuración canónica (bytes + 44 rasgos estructurales, macro-F1 0,936) quedan cuatro
familias por debajo de 0,75 de F1 en la corrida de referencia: NOTPETYA, JIGSAW, CRYPTOLOCKER y
DARKSIDE. En los tres experimentos anteriores (2e, 2e-b, 2e-c) el resultado se acertó y el
MECANISMO se erró las tres veces, por razonar desde la entropía en vez de mirar los archivos.
Este script mira, y no entrena nada.

QUÉ BUSCA, familia por familia
------------------------------
(a) PREFIJOS REPETIDOS. ¿Cuántos primeros-16-bytes distintos hay, en total y por tipo de
    documento? Un cifrado con clave fija y sin vector de inicialización por archivo produce el
    MISMO prefijo cifrado para dos archivos con el mismo comienzo en claro, y todos los PDF
    empiezan igual (`%PDF-1.`). Si una familia tiene un prefijo por tipo, hay una firma que el
    Exp. 2b no pudo ver: exigía un prefijo común a TODA la familia, no por tipo. Indicio previo:
    dos .pdf cifrados de NOTPETYA compartían sus 8 primeros bytes (visto el 2026-09-22).
(b) SUFIJOS REPETIDOS. Lo mismo al final: pie de longitud y contenido fijos.
(c) COLISIONES ENTRE FAMILIAS. Si el prefijo de una familia aparece también en otra, esa firma
    no discrimina; puede explicar la confusión mutua del grupo de seis (97-99 % interna).
(d) RESTO DEL TAMAÑO MÓDULO 16. El rasgo individual más importante del 2e-b. ¿Qué distribución
    tiene cada difícil? Si dos familias tienen la misma, por ahí no se separan.
(e) PERFIL DE ENTROPÍA DENSO, solo en las seis difíciles: hasta 32 bloques de 512 bytes
    repartidos por el archivo. El cifrado intermitente (arXiv 2510.15133 lista a DARKSIDE) deja
    bloques en claro en el medio, que los 8 bloques del 2e pueden no ver.
(f) CONTROL DE LOS 310 PDF DEVUELTOS. Se afirmó que los .pdf no-documentación de BADRABBIT y
    NOTPETYA son muestras CIFRADAS, sobre la base de dos por familia más el control de
    integridad del 2e. Acá se cuentan todos: cuántos empiezan con firma PDF en claro.

Salida: tablas en el log y CSV por familia y por archivo. No toca ningún canónico.
"""
import argparse
import collections
import json
import math
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIFICILES = ["NOTPETYA", "JIGSAW", "CRYPTOLOCKER", "DARKSIDE", "WASTEDLOCKER", "SUNCRYPT"]
N_BLOQUES, T_BLOQUE = 32, 512
UMBRAL_BAJA = 6.5   # 512 bytes aleatorios dan ~7,59 bits/byte: 6,5 está claramente por debajo


def entropia(b):
    if not b:
        return 0.0
    c = collections.Counter(b)
    n = len(b)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def tipo_documento(nombre):
    m = re.match(r"^\d+-([a-z0-9]+)", nombre.lower())
    return m.group(1) if m else "desconocido"


def es_documentacion(p, familia):
    return p.suffix.lower() == ".pdf" and p.stem.upper() == familia


def perfil(path, tam):
    """Entropía de hasta N_BLOQUES bloques de T_BLOQUE bytes repartidos parejo por el archivo."""
    with open(path, "rb") as f:
        if tam <= N_BLOQUES * T_BLOQUE:
            datos = f.read()
            bloques = [datos[i:i + T_BLOQUE] for i in range(0, len(datos), T_BLOQUE)]
        else:
            paso = (tam - T_BLOQUE) / (N_BLOQUES - 1)
            bloques = []
            for i in range(N_BLOQUES):
                f.seek(int(i * paso))
                bloques.append(f.read(T_BLOQUE))
    bloques = [b for b in bloques if len(b) == T_BLOQUE]   # el último, parcial, se descarta
    return [entropia(b) for b in bloques]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz")
    ap.add_argument("--salida", type=Path, default=None)
    args = ap.parse_args()
    aqui = Path(__file__).resolve().parent
    base = aqui.parent / "4_resultados" if (aqui.parent / "4_resultados").is_dir() else aqui
    out = args.salida or base / ("resultados_diagnostico_dificiles_job"
                                 + os.environ.get("SLURM_JOB_ID", "local"))
    out.mkdir(parents=True, exist_ok=True)
    reg = open(out / "log.txt", "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        reg.write(m + "\n")
        reg.flush()

    log("=" * 78)
    log("  DIAGNÓSTICO DE LAS FAMILIAS DIFÍCILES -- sin aprendizaje, mirando los bytes")
    log("=" * 78)
    t0 = time.time()
    filas = []
    for d in sorted(p for p in Path(args.raiz).iterdir() if p.is_dir()):
        fam = d.name.upper()
        for suf in ("-SMALL", "_SMALL", "-TINY", "_TINY"):
            fam = fam.removesuffix(suf)
        for p in sorted(x for x in d.iterdir() if x.is_file() and not es_documentacion(x, fam)):
            tam = p.stat().st_size
            with open(p, "rb") as f:
                pre = f.read(16)
                f.seek(max(0, tam - 16))
                suf16 = f.read(16)
            fila = dict(familia=fam, archivo=p.name, tipo=tipo_documento(p.name),
                        extension=p.suffix.lower(), tam=tam, mod16=tam % 16,
                        pre16=pre.hex(), suf16=suf16.hex(), pdf_en_claro=pre.startswith(b"%PDF"))
            if fam in DIFICILES:
                h = perfil(p, tam)
                fila.update(n_bloques=len(h),
                            h_media=float(np.mean(h)) if h else np.nan,
                            frac_bajos=float(np.mean([x < UMBRAL_BAJA for x in h])) if h else np.nan,
                            h_min=float(min(h)) if h else np.nan)
            filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(out / "por_archivo.csv", index=False)
    log(f"  {len(df)} archivos · {df.familia.nunique()} familias  ({round(time.time() - t0)} s)\n")

    fam_de_pre = df.groupby("pre16").familia.nunique()
    fam_de_suf = df.groupby("suf16").familia.nunique()
    res = []
    for fam, g in df.groupby("familia"):
        vp, vs, vm = g.pre16.value_counts(), g.suf16.value_counts(), g.mod16.value_counts()
        res.append(dict(
            familia=fam, n=len(g), dificil=fam in DIFICILES,
            pre_distintos=int(g.pre16.nunique()), pre_top=round(vp.iloc[0] / len(g), 3),
            suf_distintos=int(g.suf16.nunique()), suf_top=round(vs.iloc[0] / len(g), 3),
            pre_en_otra_fam=round(float((g.pre16.map(fam_de_pre) > 1).mean()), 3),
            suf_en_otra_fam=round(float((g.suf16.map(fam_de_suf) > 1).mean()), 3),
            mod16_moda=int(vm.index[0]), mod16_moda_frac=round(vm.iloc[0] / len(g), 3),
            mod16_cero=round(float((g.mod16 == 0).mean()), 3)))
    r = pd.DataFrame(res).sort_values(["dificil", "familia"], ascending=[False, True])
    r.to_csv(out / "por_familia.csv", index=False)

    log("-" * 78)
    log("  (a)(b)(c) PREFIJOS Y SUFIJOS DE 16 BYTES, Y COLISIONES CON OTRAS FAMILIAS")
    log("-" * 78)
    log("  *_top = fracción de la familia con el prefijo/sufijo más frecuente.")
    log("  *_en_otra_fam = fracción de archivos cuyo prefijo/sufijo aparece también en OTRA familia.\n")
    log(r[["familia", "n", "pre_distintos", "pre_top", "pre_en_otra_fam",
           "suf_distintos", "suf_top", "suf_en_otra_fam"]].to_string(index=False))

    log("\n" + "-" * 78)
    log("  (a') PREFIJOS DISTINTOS POR TIPO DE DOCUMENTO, en las seis difíciles")
    log("-" * 78)
    log("  archivos/prefijos: si un tipo tiene 1 prefijo sobre muchos archivos, el cifrado es")
    log("  determinista sin vector de inicialización por archivo.\n")
    for fam in DIFICILES:
        g = df[df.familia == fam]
        if g.empty:
            continue
        partes = []
        for t, gt in g.groupby("tipo"):
            partes.append(f"{t}={len(gt)}/{gt.pre16.nunique()} (top {gt.pre16.value_counts().iloc[0] / len(gt):.2f})")
        log(f"  {fam:<13} " + "  ".join(partes))

    log("\n" + "-" * 78)
    log("  (d) RESTO DEL TAMAÑO MÓDULO 16")
    log("-" * 78)
    log(r[["familia", "mod16_moda", "mod16_moda_frac", "mod16_cero"]].to_string(index=False))

    log("\n" + "-" * 78)
    log("  (e) PERFIL DE ENTROPÍA DENSO (hasta 32 bloques de 512 B), seis difíciles")
    log("-" * 78)
    log(f"  frac_bajos = fracción media de bloques con entropía < {UMBRAL_BAJA} (aleatorio ~ 7,59).\n")
    e = df[df.familia.isin(DIFICILES)].groupby("familia").agg(
        archivos=("tam", "size"), bloques_medio=("n_bloques", "mean"),
        h_media=("h_media", "mean"), h_min_medio=("h_min", "mean"),
        frac_bajos=("frac_bajos", "mean"),
        con_algun_bloque_bajo=("frac_bajos", lambda s: float((s > 0).mean())))
    log(e.round(3).to_string())

    log("\n" + "-" * 78)
    log("  (f) CONTROL DE LOS .pdf NO-DOCUMENTACIÓN (los 310 devueltos al corpus)")
    log("-" * 78)
    pdfs = df[df.extension == ".pdf"]
    for fam, g in pdfs.groupby("familia"):
        log(f"  {fam:<13} {len(g):>4} .pdf · {int(g.pdf_en_claro.sum()):>3} empiezan con %PDF en claro")

    (out / "manifiesto.json").write_text(json.dumps(dict(
        fecha=str(date.today()), raiz=str(args.raiz), dificiles=DIFICILES,
        n_bloques=N_BLOQUES, t_bloque=T_BLOQUE, umbral_bajo=UMBRAL_BAJA),
        indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n  Salidas en: {out}")
    reg.close()


if __name__ == "__main__":
    main()
