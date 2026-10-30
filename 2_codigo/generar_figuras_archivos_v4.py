#!/usr/bin/env python3
"""
generar_figuras_archivos_v4.py — Curva de aprendizaje del frente de archivos, LEÍDA de los resultados, con el mismo
estilo que la del frente de notas (generar_figuras_notas_v4.py).

  fig_curva_archivos.png   Macro-F1 medio con intervalo del 95 % según los archivos por familia (10 a 500), con
                           subconjuntos anidados, solo bytes, 30 familias y tres semillas (Exp. 2c, Sección 5.1.8).
                           Fuente: 4_resultados/resultados_exp2d/resultados_exp2d_job3937/a3_curva_por_semilla.csv.

Pedida por Romina el 2026-10-30 para la guía de defensa y anotada para la tesis (observaciones de la v4). El intervalo
se calcula igual que en la figura de notas: 1,96 · desvío / √n sobre las semillas. Salida:
1_documento/tesis_v4_en_preparacion/images/
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "4_resultados"
IMG = RAIZ / "1_documento" / "tesis_v4_en_preparacion" / "images"
IMG.mkdir(parents=True, exist_ok=True)

AZUL = "#2a6fb0"
GRIS = "#6b6b6b"


def estilo(eje):
    eje.spines[["top", "right"]].set_visible(False)
    eje.spines[["left", "bottom"]].set_color(GRIS)
    eje.tick_params(colors="#333333", labelsize=8)
    eje.grid(axis="y", color="#e3e3e3", linewidth=0.6)
    eje.set_axisbelow(True)


def media_ic(valores):
    n = len(valores)
    m = sum(valores) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in valores) / (n - 1)) if n > 1 else 0.0
    return m, 1.96 * sd / math.sqrt(n)


def figura_curva_archivos():
    datos = defaultdict(list)
    fuente = RES / "resultados_exp2d" / "resultados_exp2d_job3937" / "a3_curva_por_semilla.csv"
    for f in csv.DictReader(open(fuente, encoding="utf-8")):
        datos[int(f["por_familia"])].append(float(f["f1_macro"]))
    xs = sorted(datos)
    puntos = [media_ic(datos[x]) for x in xs]
    ms = [m for m, _ in puntos]
    ic = [c for _, c in puntos]

    # control contra el texto de la tesis (experimentos.tex, subsec:exp2c_curva y tab:verificaciones_archivos)
    medias = dict(zip(xs, ms))
    assert (round(medias[10], 4), round(medias[200], 4), round(medias[500], 4)) == (0.7809, 0.9045, 0.9117), medias
    assert all(len(v) == 3 for v in datos.values()), {x: len(v) for x, v in datos.items()}

    fig, eje = plt.subplots(figsize=(6.4, 3.4))
    eje.fill_between(xs, [m - c for m, c in zip(ms, ic)], [m + c for m, c in zip(ms, ic)],
                     color=AZUL, alpha=0.15, linewidth=0)
    eje.plot(xs, ms, color=AZUL, linewidth=2, marker="o", markersize=5)
    eje.axvline(200, color=GRIS, linewidth=1, linestyle=(0, (3, 3)))
    eje.annotate("desde 200, ningún paso\nmejora de forma medible", xy=(200, ms[xs.index(200)]),
                 xytext=(215, 0.80), fontsize=8, color="#333333", va="center")
    eje.set_xscale("log")
    eje.xaxis.set_major_locator(FixedLocator(xs))
    eje.xaxis.set_minor_locator(NullLocator())
    eje.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))
    eje.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    eje.set_xlabel("Archivos por familia (escala logarítmica)", fontsize=9)
    eje.set_ylabel("Macro-F1 (media e IC 95 %)", fontsize=9)
    estilo(eje)
    fig.tight_layout()
    salida = IMG / "fig_curva_archivos.png"
    fig.savefig(salida, dpi=300)
    plt.close(fig)
    print("figura:", salida)


if __name__ == "__main__":
    figura_curva_archivos()
