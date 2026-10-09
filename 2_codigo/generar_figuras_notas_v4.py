#!/usr/bin/env python3
"""
generar_figuras_notas_v4.py — Dos figuras del frente de notas para el capítulo 4 de la v4, LEÍDAS de los resultados.

  fig_f1_familias_cascada.png   F1 por familia bajo P2bal (149 notas, 30 familias, 50 semillas), texto solo frente a
                                la cascada. Fuente: 4_resultados/resultados_protocolo_p2bal_149/p2bal_por_familia.csv
                                (columnas P2bal_txt y P2bal_m6). Son los mismos valores de la tabla tab:p2bal_familias.
  fig_curva_plantillas.png      Curva de aprendizaje (149 notas, 30 familias): macro-F1 medio con intervalo del 95 %
                                según el tope de material por familia. (a) tope por plantillas, P1 y P2 con retención;
                                (b) P2 con retención, tope por plantillas frente a tope por notas.
                                Fuente: 4_resultados/resultados_curva_149/b1_curva_por_repeticion.csv (curva 30fam).

Colores validados con el validador de paleta (azul #2a6fb0, naranja #c8763c: contraste, croma y separación para
daltonismo). Salida: 1_documento/tesis_v4_en_preparacion/images/
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "4_resultados"
IMG = RAIZ / "1_documento" / "tesis_v4_en_preparacion" / "images"
IMG.mkdir(parents=True, exist_ok=True)

AZUL = "#2a6fb0"
NARANJA = "#c8763c"
GRIS = "#6b6b6b"
COMA = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))


def estilo(eje):
    eje.spines[["top", "right"]].set_visible(False)
    eje.spines[["left", "bottom"]].set_color(GRIS)
    eje.tick_params(colors="#333333", labelsize=8)
    eje.grid(axis="x", color="#e3e3e3", linewidth=0.6)
    eje.set_axisbelow(True)


def figura_familias():
    filas = list(csv.DictReader(open(RES / "resultados_protocolo_p2bal_149" / "p2bal_por_familia.csv", encoding="utf-8-sig")))
    filas.sort(key=lambda f: (float(f["P2bal_m6"]), float(f["P2bal_txt"])))
    fams = [f["familia"] for f in filas]
    txt = [float(f["P2bal_txt"]) for f in filas]
    cas = [float(f["P2bal_m6"]) for f in filas]
    for control, esperado in (("CHIMERA", (0.2947, 0.8920)), ("BLACKBASTA", (0.2020, 0.7456)), ("AVOSLOCKER", (0.9943, 0.9611))):
        i = fams.index(control)
        assert (round(txt[i], 4), round(cas[i], 4)) == esperado, (control, txt[i], cas[i])
    fig, eje = plt.subplots(figsize=(6.6, 7.4))
    y = range(len(fams))
    for yi, a, b in zip(y, txt, cas):
        eje.plot([a, b], [yi, yi], color="#bfbfbf", linewidth=2, zorder=1)
    eje.scatter(txt, y, s=36, color=NARANJA, label="Texto solo", zorder=2, edgecolors="white", linewidths=0.8)
    eje.scatter(cas, y, s=36, color=AZUL, label="Cascada", zorder=3, edgecolors="white", linewidths=0.8)
    eje.set_yticks(list(y))
    eje.set_yticklabels(fams, fontsize=7.5)
    eje.set_xlim(-0.02, 1.02)
    eje.xaxis.set_major_formatter(COMA)
    eje.set_xlabel("F1 por familia (P2bal, 50 semillas)", fontsize=9)
    estilo(eje)
    eje.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    salida = IMG / "fig_f1_familias_cascada.png"
    fig.savefig(salida, dpi=300)
    plt.close(fig)
    print("figura:", salida, "| familias:", len(fams))


def media_ic(valores):
    n = len(valores)
    m = sum(valores) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in valores) / (n - 1)) if n > 1 else 0.0
    return m, 1.96 * sd / math.sqrt(n)


def figura_curva():
    datos = defaultdict(list)
    for f in csv.DictReader(open(RES / "resultados_curva_149" / "b1_curva_por_repeticion.csv", encoding="utf-8")):
        if f["curva"] == "30fam":
            datos[(f["protocolo"], f["unidad"], f["k"])].append(float(f["f1_macro"]))

    def serie(protocolo, unidad):
        ks = sorted({k for (p, u, k) in datos if p == protocolo and u == unidad and k != "todo"}, key=int)
        puntos = [(int(k), *media_ic(datos[(protocolo, unidad, k)])) for k in ks]
        return puntos, media_ic(datos[(protocolo, unidad, "todo")])

    # control contra la tabla del texto (tab:curva_pasos): paso 2->3 bajo P2 con retención, tope por plantillas
    p2r, _ = serie("P2ret", "plantillas")
    paso = dict((k, m) for k, m, _ in p2r)
    assert round(paso[3] - paso[2], 3) == 0.015, paso

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 3.6), sharey=True)
    for eje, curvas, titulo in (
        (a, [("P1", "plantillas", "P1", AZUL), ("P2ret", "plantillas", "P2 con retención", NARANJA)], "(a) Tope por plantillas distintas"),
        (b, [("P2ret", "plantillas", "Tope por plantillas", NARANJA), ("P2ret", "notas", "Tope por notas", AZUL)], "(b) P2 con retención: plantillas o notas"),
    ):
        for protocolo, unidad, nombre, color in curvas:
            puntos, (mt, ict) = serie(protocolo, unidad)
            xs = [k for k, _, _ in puntos]
            ms = [m for _, m, _ in puntos]
            ic = [c for _, _, c in puntos]
            eje.fill_between(xs, [m - c for m, c in zip(ms, ic)], [m + c for m, c in zip(ms, ic)], color=color, alpha=0.15, linewidth=0)
            eje.plot(xs, ms, color=color, linewidth=2, marker="o", markersize=5, label=nombre)
            eje.axhline(mt, color=color, linewidth=1, linestyle=(0, (3, 3)), alpha=0.8)
        eje.set_title(titulo, fontsize=9.5)
        eje.set_xlabel("Tope de material por familia", fontsize=9)
        eje.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
        estilo(eje)
        eje.grid(axis="y", color="#e3e3e3", linewidth=0.6)
        eje.legend(fontsize=8, frameon=False, loc="lower right")
    a.set_ylabel("Macro-F1 (media e IC 95 %)", fontsize=9)
    fig.tight_layout()
    salida = IMG / "fig_curva_plantillas.png"
    fig.savefig(salida, dpi=300)
    plt.close(fig)
    print("figura:", salida)


if __name__ == "__main__":
    figura_familias()
    figura_curva()
