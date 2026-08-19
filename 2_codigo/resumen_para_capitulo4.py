#!/usr/bin/env python3.11
"""Agrega los resultados multisemilla y emite las tablas del capítulo 4.

Los experimentos escriben UN archivo por semilla. Este script hace el paso que
faltaba: promediar entre semillas y contar familias. Antes se hacía a mano en el
chat, que es exactamente donde no debe estar — una cifra de la tesis tiene que
poder regenerarse con un comando.

Uso:
    python resumen_para_capitulo4.py                    # usa el job más reciente
    python resumen_para_capitulo4.py --job-estructural 3651

Salidas en 4_resultados/resumen_capitulo4/:
    2b_ablacion.csv        exactitud/cobertura por modalidad, media ± desvío
    2b_tipos_de_marca.csv  cuántas familias tienen firma, extensión, ambas o nada
    2c_por_familia.csv     F1 por familia, media ± desvío
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
RES = _AQUI.parent / "4_resultados"
OUT = RES / "resumen_capitulo4"
MIN_MARCA = 4


def carpetas_estructural(job=None):
    """Carpetas de una MISMA corrida. Filtrar por job es imprescindible: el
    directorio mezcla corridas sueltas (semilla 42, pruebas previas) con la
    tanda buena, y promediarlas juntas falsea el desvío."""
    todas = sorted(RES.glob("**/resultados_estructural_s*_job*"))
    if not todas:
        return [], None
    jobs = {int(re.search(r"_job(\d+)$", str(d)).group(1)) for d in todas}
    job = job or max(jobs)
    return [d for d in todas if d.name.endswith(f"_job{job}")], job


def resumen_2b(job=None):
    dirs, job = carpetas_estructural(job)
    if not dirs:
        print("  (sin carpetas del detector estructural)")
        return
    filas_abl, tipos, semillas = [], [], []
    for d in sorted(dirs):
        man = json.loads((d / "manifiesto.json").read_text(encoding="utf-8"))
        semillas.append(man["semilla"])
        for etiqueta, crit in man["criterios"].items():
            for modo, v in crit["ablacion"].items():
                cubiertos = v["total"] - v["sin_marca"]
                filas_abl.append(dict(
                    criterio=etiqueta, modo=modo, semilla=man["semilla"],
                    exactitud=v["exactitud"], cobertura=v["cobertura"],
                    donde_aplica=round(v["aciertos"] / cubiertos, 4) if cubiertos else 0.0,
                    familias_con_marca=crit["familias_con_marca"]))
        c = Counter()
        for r in csv.DictReader(open(d / "marcas_por_familia_umbral_90.csv", encoding="utf-8")):
            firma = max(int(r["prefijo_len"]), int(r["sufijo_len"])) >= MIN_MARCA
            ext = bool(r["extension"].strip())
            c[("firma y extensión" if firma and ext else "solo firma" if firma
               else "solo extensión" if ext else "sin marca")] += 1
        tipos.append((man["semilla"], c))

    print(f"  job {job} | semillas {sorted(semillas)} | n = {len(semillas)}\n")
    agg = defaultdict(lambda: defaultdict(list))
    for f in filas_abl:
        for m in ("exactitud", "cobertura", "donde_aplica"):
            agg[(f["criterio"], f["modo"])][m].append(f[m])

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "2b_ablacion.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["criterio", "modo", "exactitud_media", "exactitud_desvio",
                    "cobertura_media", "cobertura_desvio",
                    "donde_aplica_media", "donde_aplica_desvio", "n_semillas"])
        for (crit, modo), d in sorted(agg.items()):
            vals = [x for m in ("exactitud", "cobertura", "donde_aplica")
                    for x in (st.mean(d[m]), st.stdev(d[m]) if len(d[m]) > 1 else 0.0)]
            w.writerow([crit, modo] + [round(v, 4) for v in vals] + [len(d["exactitud"])])
            print(f"  {crit:<12} {modo:<22} exactitud {vals[0]:.4f} ± {vals[1]:.4f} | "
                  f"cobertura {vals[2]:.4f} ± {vals[3]:.4f} | "
                  f"donde aplica {vals[4]:.4f} ± {vals[5]:.4f}")

    print("\n  Tipos de marca (familias):")
    claves = ["firma y extensión", "solo firma", "solo extensión", "sin marca"]
    with open(OUT / "2b_tipos_de_marca.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["tipo", "media", "minimo", "maximo"])
        for k in claves:
            v = [c[k] for _, c in tipos]
            w.writerow([k, round(st.mean(v), 2), min(v), max(v)])
            marca = "" if min(v) == max(v) else "   ← varía entre semillas"
            print(f"    {k:<20} {st.mean(v):>5.1f}  (mín {min(v)}, máx {max(v)}){marca}")


def resumen_2c(umbral=0.98):
    f = next(RES.glob("**/bytes_multisemilla_por_familia.csv"), None)
    if f is None:
        print("  (falta bytes_multisemilla_por_familia.csv)")
        return
    d = defaultdict(list)
    for r in csv.DictReader(open(f, encoding="utf-8")):
        d[r["familia"]].append(float(r["f1"]))
    res = {fam: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v))
           for fam, v in d.items()}
    n_sem = max(v[2] for v in res.values())
    altas = [f for f, (m, _, _) in res.items() if m >= umbral]
    print(f"  {len(res)} familias | {n_sem} semillas")
    print(f"  F1 ≥ {umbral}: {len(altas)} de {len(res)}\n")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "2c_por_familia.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["familia", "f1_media", "f1_desvio", "n_semillas"])
        for fam, (m, s, n) in sorted(res.items(), key=lambda x: x[1][0]):
            w.writerow([fam, round(m, 4), round(s, 4), n])
    print("  Familias por debajo del umbral:")
    for fam, (m, s, _) in sorted(res.items(), key=lambda x: x[1][0]):
        if m < umbral:
            print(f"    {fam:<15} {m:.3f} ± {s:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-estructural", type=int, default=None,
                    help="job SLURM de la tanda del detector (por defecto, el mayor)")
    ap.add_argument("--umbral-f1", type=float, default=0.98)
    args = ap.parse_args()
    print("=" * 74)
    print("  EXPERIMENTO 2b — ablación del detector estructural")
    print("=" * 74)
    resumen_2b(args.job_estructural)
    print("\n" + "=" * 74)
    print("  EXPERIMENTO 2c — clasificador de bytes, por familia")
    print("=" * 74)
    resumen_2c(args.umbral_f1)
    print(f"\nSalidas en: {OUT}")


if __name__ == "__main__":
    main()
