#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p1_copias_sin_apuntar.py -- P1 con el catálogo ampliado SIN apuntar a las notas que fallan.
Esta corrida da la cifra CANÓNICA de P1 (decisión de Romina, después de ver la ampliación apuntada).
La recolección es la de copias_triage.py --todas, con su lectura segura (ver SEGURIDAD en ese script).

POR QUÉ NO SIRVE LA CORRIDA ANTERIOR. p1_copias_triage.py amplió el catálogo buscando en tria.ge
copias de las 12 notas que la cascada fallaba, con la frase de apertura de cada una: la prueba
guió qué datos se agregaban, y su 0,9148 sobreestima lo que daría un catálogo ampliado sin ese
conocimiento. Sirve como evidencia de que el límite son los datos, no como cifra de rendimiento.

PROCEDIMIENTO, fijado ANTES de recolectar:
  - las 30 familias, sin excepción;
  - para cada una, la PRIMERA página de la búsqueda pública de tria.ge por cada etiqueta de su
    familia (copias_triage.py --todas: el nombre de la familia más los alias del inventario), hasta
    50 informes, sea cual sea su contenido y sin filtrar por frases;
  - solo informes con la etiqueta de familia esperada; todas sus notas, con su extensión original;
  - carpeta propia, 3_datos/fuentes_notas/triage_sin_apuntar_2026-10: las 36 copias apuntadas de
    triage_2026-10 NO entran;
  - admisión: la regla de inventario_copias_controladas.py, sin cambios. El catálogo incluye también
    las 6 copias de las fuentes que el corpus ya cita, que la revisión tomó todas, sin apuntar;
  - medición: p1_copias_controladas.py tal cual (P1, 50 semillas, prueba = las 149 notas, copias
    solo en entrenamiento, puerta V0 = 0,8866 / 0,8592).

PREREGISTRO (commiteado ANTES de recolectar, 2026-10-05). Se reporta lo que dé.
  PS-1  La cascada con el catálogo ampliado supera a V0 en exactitud, con el IC 95 % de la
        diferencia pareada por encima de cero.
  PS-2  La ganancia es MENOR que la de la ampliación apuntada (+0,0282): las copias no se concentran
        en las notas que fallan.
  PS-3  En las notas cuya plantilla NO recibe copia, la exactitud cambia menos de 0,005 en valor
        absoluto.
  PS-4  Toda nota objetivo de objetivos_p1.py (única y fallada en más de la mitad de las semillas)
        cuya plantilla reciba alguna copia pasa a acertarse en el 90 % o más de las semillas.

Uso:  python p1_copias_sin_apuntar.py      (después de recolectar con copias_triage.py --todas)
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
FUENTE = RAIZ / "3_datos" / "fuentes_notas" / "triage_sin_apuntar_2026-10"
SALIDA = RAIZ / "4_resultados" / "resultados_copias_sin_apuntar"
OBJETIVOS = RAIZ / "4_resultados" / "resultados_objetivos_p1" / "objetivos_p1.csv"
APUNTADA = 0.0282

if not FUENTE.is_dir():
    sys.exit(f"Falta {FUENTE}: correr antes copias_triage.py --todas")
env = dict(os.environ, SALIDA_COPIAS=str(SALIDA), FUENTES_EXTRA=f"{FUENTE}::triage_sin_apuntar",
           PYTHONIOENCODING="utf-8")
for script in ("inventario_copias_controladas.py", "p1_copias_controladas.py"):
    print(f"\n######## {script} ########", flush=True)
    r = subprocess.run([sys.executable, "-u", str(AQUI / script)], env=env, cwd=AQUI)
    if r.returncode != 0:
        sys.exit(f"ABORTADO: {script} terminó con código {r.returncode}")

with open(SALIDA / "inventario.csv", encoding="utf-8") as fh:
    inv = list(csv.DictReader(fh))
nuevas = [r for r in inv if r["fuente"] == "triage_sin_apuntar" and r["estado"] == "COPIA"
          and r["marcadores_iguales"] != "si"]
with open(SALIDA / "p1_copias_por_semilla.csv", encoding="utf-8") as fh:
    sem = list(csv.DictReader(fh))
d_ac = np.array([float(r["exactitud_C1"]) - float(r["exactitud_V0"]) for r in sem])
d_f1 = np.array([float(r["macroF1_C1"]) - float(r["macroF1_V0"]) for r in sem])
h = stats.t.ppf(0.975, len(d_ac) - 1) * d_ac.std(ddof=1) / np.sqrt(len(d_ac))
with open(SALIDA / "p1_copias_por_nota.csv", encoding="utf-8") as fh:
    notas = list(csv.DictReader(fh))
for r in notas:
    r["corto"] = Path(r["archivo"].replace("\\", "/")).name
with open(OBJETIVOS, encoding="utf-8") as fh:
    objetivos = {(r["familia"], r["archivo"]) for r in csv.DictReader(fh)
                 if r["tam_plantilla"] == "1" and float(r["acierto"]) < 0.5}
con = [r for r in notas if r["plantilla_con_copia"] == "1"]
sin = [r for r in notas if r["plantilla_con_copia"] == "0"]
obj_con = [r for r in con if (r["familia"], r["corto"]) in objetivos]
d_sin = np.mean([float(r["acierto_C1"]) - float(r["acierto_V0"]) for r in sin]) if sin else 0.0

print("\n" + "=" * 78 + "\n  AMPLIACIÓN SIN APUNTAR: VEREDICTO DEL PREREGISTRO PS\n" + "=" * 78)
print(f"  Copias admitidas de tria.ge sin apuntar: {len(nuevas)}, en {len(set(r['familia'] for r in nuevas))} familias")
for fam, k in sorted(Counter(r["familia"] for r in nuevas).items()):
    print(f"    {fam:<14}{k}")
print(f"  Exactitud: V0 {np.mean([float(r['exactitud_V0']) for r in sem]):.4f} -> "
      f"C1 {np.mean([float(r['exactitud_C1']) for r in sem]):.4f}  "
      f"(delta {d_ac.mean():+.4f} [{d_ac.mean() - h:+.4f}; {d_ac.mean() + h:+.4f}])")
print(f"  Macro-F1:  V0 {np.mean([float(r['macroF1_V0']) for r in sem]):.4f} -> "
      f"C1 {np.mean([float(r['macroF1_C1']) for r in sem]):.4f}  (delta {d_f1.mean():+.4f})")
print(f"  Notas objetivo que reciben copia: {len(obj_con)} de {len(objetivos)}")
for r in obj_con:
    print(f"    {r['familia']:<13}{r['corto'][:45]:<47}V0 {float(r['acierto_V0']):.2f} -> C1 {float(r['acierto_C1']):.2f}")
print(f"  [{'CUMPLE' if d_ac.mean() - h > 0 else 'FALLA '}] PS-1 C1 supera a V0 con IC sobre cero")
print(f"  [{'CUMPLE' if d_ac.mean() < APUNTADA else 'FALLA '}] PS-2 ganancia menor que la apuntada (+0,0282)   {d_ac.mean():+.4f}")
print(f"  [{'CUMPLE' if abs(d_sin) < 0.005 else 'FALLA '}] PS-3 sin copia, |delta| < 0,005   {d_sin:+.4f}")
pt4 = all(float(r["acierto_C1"]) >= 0.90 for r in obj_con) if obj_con else None
print(f"  [{'CUMPLE' if pt4 else ('SIN DATOS' if pt4 is None else 'FALLA ')}] PS-4 toda nota objetivo con copia llega a 0,90")
