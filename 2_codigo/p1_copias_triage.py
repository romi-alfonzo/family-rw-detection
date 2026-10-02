#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p1_copias_triage.py -- la cascada bajo P1 con el catálogo ampliado por copias de tria.ge.

POR QUÉ. Romina (2026-10-02, «dale nomás» a buscar datos en fuentes nuevas). Con las fuentes que el
corpus ya cita, la ampliación controlada dio 6 copias (p1_copias_controladas.py, preregistro
be55736: +0,0054 de exactitud). objetivos_p1.py identificó 12 notas ÚNICAS que la cascada falla en
más de la mitad de las semillas bajo P1. Este experimento agrega como fuente nueva las notas de
informes públicos de tria.ge (copias_triage.py) y mide cuánto mueve P1.

MÉTODO. Sin código nuevo de medición: corre inventario_copias_controladas.py y
p1_copias_controladas.py tal cual, con dos variables de entorno: FUENTES_EXTRA agrega la carpeta
3_datos/fuentes_notas/triage_2026-10 como fuente «triage», y SALIDA_COPIAS manda todo a
4_resultados/resultados_copias_triage, sin pisar la corrida de la revisión. La regla de admisión
es la del inventario, sin cambios. El grupo C1 incluye las 6 copias locales MÁS las de tria.ge:
para aislar el aporte de tria.ge se compara también contra el C1 de la revisión (0,8919 / 0,8645).
Las líneas CC-0 a CC-3 que imprime p1_copias_controladas.py son las predicciones de la revisión para
SU grupo de 6 copias; acá no se evalúan. Se evalúan estas:

PREREGISTRO (commiteado ANTES de recolectar, 2026-10-02). Se reporta lo que dé.
  PT-0  Descriptivo, sin umbral: cuántas de las 12 notas objetivo reciben al menos una copia.
  PT-1  Cada nota objetivo cuya plantilla recibe al menos una copia admitida pasa a acertarse en
        el 90 % o más de las semillas. Fundamento: con una hermana en entrenamiento la cascada
        acierta 0,9740 (P1cat) y todo su error estaba en las dos plantillas mixtas, que no son
        objetivos.
  PT-2  En las notas cuya plantilla NO recibe copia, la exactitud media cambia menos de 0,003 en
        valor absoluto (solo cambian por el reajuste del TF-IDF, del SVM y del diccionario).
  PT-3  Ninguna nota cuya plantilla recibe copia baja su acierto en más de 0,05.
PUERTA. La de p1_copias_controladas.py: V0 tiene que reproducir 0,8866 / 0,8592, o se aborta.

Uso:  python p1_copias_triage.py      (después de correr copias_triage.py)
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
TRIAGE = RAIZ / "3_datos" / "fuentes_notas" / "triage_2026-10"
SALIDA = RAIZ / "4_resultados" / "resultados_copias_triage"
OBJETIVOS = RAIZ / "4_resultados" / "resultados_objetivos_p1" / "objetivos_p1.csv"
C1_REVISION = (0.8919, 0.8645)

if not TRIAGE.is_dir():
    sys.exit(f"Falta {TRIAGE}: correr antes copias_triage.py")
env = dict(os.environ, SALIDA_COPIAS=str(SALIDA), FUENTES_EXTRA=f"{TRIAGE}::triage",
           PYTHONIOENCODING="utf-8")
for script in ("inventario_copias_controladas.py", "p1_copias_controladas.py"):
    print(f"\n######## {script} ########", flush=True)
    r = subprocess.run([sys.executable, "-u", str(AQUI / script)], env=env, cwd=AQUI)
    if r.returncode != 0:
        sys.exit(f"ABORTADO: {script} terminó con código {r.returncode}")

with open(SALIDA / "inventario.csv", encoding="utf-8") as fh:
    inv = list(csv.DictReader(fh))
nuevas = [r for r in inv if r["fuente"] == "triage" and r["estado"] == "COPIA" and r["marcadores_iguales"] != "si"]
with open(OBJETIVOS, encoding="utf-8") as fh:
    objetivos = {(r["familia"], r["archivo"]) for r in csv.DictReader(fh)
                 if r["tam_plantilla"] == "1" and float(r["acierto"]) < 0.5}
with open(SALIDA / "p1_copias_por_nota.csv", encoding="utf-8") as fh:
    notas = list(csv.DictReader(fh))
for r in notas:
    r["archivo_corto"] = Path(r["archivo"].replace("\\", "/")).name
con = [r for r in notas if r["plantilla_con_copia"] == "1"]
sin = [r for r in notas if r["plantilla_con_copia"] == "0"]
obj_con = [r for r in con if (r["familia"], r["archivo_corto"]) in objetivos]

print("\n" + "=" * 78 + "\n  COPIAS DE TRIA.GE Y VEREDICTO DEL PREREGISTRO PT\n" + "=" * 78)
print(f"  Copias admitidas de tria.ge: {len(nuevas)}")
from collections import Counter  # noqa: E402
for fam, k in sorted(Counter(r["familia"] for r in nuevas).items()):
    print(f"    {fam:<14}{k}")
print(f"\n  PT-0  notas objetivo que reciben copia: {len(obj_con)} de {len(objetivos)}")
for r in obj_con:
    print(f"        {r['familia']:<13}{r['archivo_corto'][:45]:<47}V0 {float(r['acierto_V0']):.2f} -> "
          f"C1 {float(r['acierto_C1']):.2f}")
pt1 = all(float(r["acierto_C1"]) >= 0.90 for r in obj_con) if obj_con else None
d_sin = (sum(float(r["acierto_C1"]) for r in sin) - sum(float(r["acierto_V0"]) for r in sin)) / max(len(sin), 1)
caidas = [r for r in con if float(r["acierto_C1"]) < float(r["acierto_V0"]) - 0.05]
print(f"  [{'CUMPLE' if pt1 else ('SIN DATOS' if pt1 is None else 'FALLA ')}] PT-1 toda nota objetivo con copia llega a 0,90 o más")
print(f"  [{'CUMPLE' if abs(d_sin) < 0.003 else 'FALLA '}] PT-2 sin copia, |delta| < 0,003   {d_sin:+.4f}")
print(f"  [{'CUMPLE' if not caidas else 'FALLA '}] PT-3 ninguna nota con copia baja más de 0,05   "
      f"{len(caidas)} bajan" + (": " + ", ".join(r["archivo_corto"] for r in caidas) if caidas else ""))
print(f"\n  Para aislar tria.ge, comparar C1 de esta corrida contra el C1 de la revisión: "
      f"{C1_REVISION[0]} / {C1_REVISION[1]}")
