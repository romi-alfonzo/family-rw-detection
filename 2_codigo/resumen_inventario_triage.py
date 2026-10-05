#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resumen_inventario_triage.py -- cuánto se recolectó de tria.ge y qué hizo con eso la regla de admisión.

Suma los totales de los logs de recolección (copias_triage.py) y cuenta los estados del inventario
(inventario_copias_controladas.py) para una fuente, para que las cifras que cita el capítulo queden
en un log verificable.

Uso:  python resumen_inventario_triage.py [--fuente triage_sin_apuntar]
          [--salida resultados_copias_sin_apuntar] [--logs "_log_triage_sin_apuntar_*.txt"]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RES = Path(__file__).resolve().parent.parent / "4_resultados"
ap = argparse.ArgumentParser()
ap.add_argument("--fuente", default="triage_sin_apuntar")
ap.add_argument("--salida", default="resultados_copias_sin_apuntar")
ap.add_argument("--logs", default="_log_triage_sin_apuntar_*.txt")
a = ap.parse_args()

notas = informes = fallidas = 0
for p in sorted(RES.glob(a.logs)):
    t = p.read_text(encoding="utf-8", errors="replace")
    for n, i in re.findall(r"Total: (\d+) notas distintas de (\d+) informes", t):
        notas += int(n)
        informes += int(i)
    fallidas += t.count("no se pudo leer")
print(f"Recolección ({a.logs}): {informes} informes leídos con la etiqueta esperada | "
      f"{notas} notas distintas | {fallidas} lecturas fallidas")

with open(RES / a.salida / "inventario.csv", encoding="utf-8") as fh:
    filas = [r for r in csv.DictReader(fh) if r["fuente"] == a.fuente]
print(f"Inventario, fuente {a.fuente}: {len(filas)} notas")
for e, k in Counter(r["estado"] for r in filas).most_common():
    print(f"  {e}: {k}")
copias = [r for r in filas if r["estado"] == "COPIA"]
admitidas = [r for r in copias if r["marcadores_iguales"] != "si"]
print(f"  COPIA con los mismos marcadores que su plantilla, no admitidas: {len(copias) - len(admitidas)}")
print(f"  COPIA admitidas: {len(admitidas)}, en {len(set(r['familia'] for r in admitidas))} familias")
