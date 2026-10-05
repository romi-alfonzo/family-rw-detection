# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
diagnostico_ampliacion_p1.py -- dónde están los errores de P1 que la ampliación sin apuntar no arregla.

Pregunta de Romina (05-10): «la mejora es chica, explicame por qué no sube más». Solo lee resultados de
p1_copias_sin_apuntar.py (p1_copias_por_nota.csv, inventario.csv) y los logs de la recolección; no
entrena nada. Errores por semilla = suma sobre las notas de (1 - acierto en las 50 semillas).

Uso:  python diagnostico_ampliacion_p1.py > ../4_resultados/_log_diagnostico_ampliacion_p1.txt
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
R = Path(r"C:\Users\Romina\Tesis\4_resultados")
S = R / "resultados_copias_sin_apuntar"
with open(S / "p1_copias_por_nota.csv", encoding="utf-8") as fh:
    notas = list(csv.DictReader(fh))
with open(S / "inventario.csv", encoding="utf-8") as fh:
    inv = [r for r in csv.DictReader(fh) if r["fuente"] == "triage_sin_apuntar"]
leidos = {}
for p in R.glob("_log_triage_sin_apuntar_*.txt"):
    t = p.read_text(encoding="utf-8", errors="replace")
    for fam, m in re.findall(r"^(\w+): \d+ informes en la búsqueda pública, se leen (\d+)", t, re.M):
        leidos[fam] = int(m)

e = lambda rs, k: sum(1 - float(r[k]) for r in rs)
print(f"notas: {len(notas)} | errores por semilla (promedio): V0 {e(notas, 'acierto_V0'):.2f} -> C1 {e(notas, 'acierto_C1'):.2f}")
for nombre, g in (("plantilla CON copia", [r for r in notas if r["plantilla_con_copia"] == "1"]),
                  ("plantilla SIN copia", [r for r in notas if r["plantilla_con_copia"] == "0"])):
    print(f"  {nombre}: {len(g)} notas | errores V0 {e(g, 'acierto_V0'):.2f} -> C1 {e(g, 'acierto_C1'):.2f}")
for nombre, g in (("única de su plantilla", [r for r in notas if r["tam_plantilla"] == "1"]),
                  ("plantilla de 2 o más", [r for r in notas if r["tam_plantilla"] != "1"])):
    con = sum(r["plantilla_con_copia"] == "1" for r in g)
    print(f"  {nombre}: {len(g)} notas ({con} con copia) | errores V0 {e(g, 'acierto_V0'):.2f} -> C1 {e(g, 'acierto_C1'):.2f}")

est = defaultdict(Counter)
for r in inv:
    est[r["familia"]][r["estado"]] += 1
fam = defaultdict(list)
for r in notas:
    fam[r["familia"]].append(r)
print("\nfamilias con error en C1 | notas | errores C1/semilla | notas con copia | informes leídos | notas de tria.ge por estado")
for f, rs in sorted(fam.items(), key=lambda x: -e(x[1], "acierto_C1")):
    if e(rs, "acierto_C1") < 0.05:
        continue
    con = sum(r["plantilla_con_copia"] == "1" for r in rs)
    print(f"  {f:<13}{len(rs):>3}{e(rs, 'acierto_C1'):>7.2f}{con:>4}{leidos.get(f, 0):>5}   {dict(est[f])}")
