# -*- coding: utf-8 -*-
"""Figura de una matriz de confusión normalizada por fila (comentarios de Cappo, 10 y 11-11-2026).

Se hizo para la cascada del frente de notas (comentario 32): lee la matriz que escribió el análisis de linaje (149
notas, P2bal, 50 semillas; 7.450 decisiones, cuya diagonal da la exactitud de 0,8123 del sistema). Desde el 12-11 sirve
también para el frente de archivos: la matriz del filtro k=2 con el nombre original que escribe exp2g_examen_filtro.py
(job 6139: 14.948 archivos no vistos, diagonal 0,9786).

La matriz se dibuja normalizada por fila: cada fila suma 1 y muestra a qué familias se asignaron los casos de una
familia verdadera. Escala secuencial de un solo tono, de blanco al azul de las demás figuras de la tesis. Se rotulan
las celdas fuera de la diagonal que reúnen al menos --umbral de su fila (0,20 en las notas) y, si se pide, las de la
diagonal por debajo de --diagonal-menor.

Uso: python generar_figura_confusion_notas.py <matriz.csv> <salida.png> [--unidad archivos] [--umbral 0.03]
                                              [--diagonal-menor 0.95]
Sin las opciones, la figura de las notas sale igual que antes.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ap = argparse.ArgumentParser(description="Figura de una matriz de confusión normalizada por fila")
ap.add_argument("entrada", help="CSV con filas = familia verdadera y columnas = familia predicha")
ap.add_argument("salida", help="PNG de salida")
ap.add_argument("--unidad", default="notas", help="qué se cuenta en cada fila, para el rótulo de la barra")
ap.add_argument("--umbral", type=float, default=0.20,
                help="se rotulan las celdas fuera de la diagonal con al menos esta proporción de su fila")
ap.add_argument("--diagonal-menor", type=float, default=None,
                help="se rotulan también las celdas de la diagonal por debajo de esta proporción")
a = ap.parse_args()

m = pd.read_csv(a.entrada, index_col=0, encoding="utf-8-sig")
assert list(m.index) == list(m.columns), "filas y columnas deben ser las mismas familias"
total = int(m.values.sum())
acierto = np.trace(m.values) / total
print(f"{len(m)} familias, {total} decisiones, exactitud {acierto:.4f}")

p = m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).values
azul = LinearSegmentedColormap.from_list("azul", ["#ffffff", "#2c5f8a"])

fig, ax = plt.subplots(figsize=(7.2, 6.6), dpi=300)
im = ax.imshow(p, cmap=azul, vmin=0, vmax=1, interpolation="nearest")
n = len(m)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(m.columns, rotation=90, fontsize=6)
ax.set_yticklabels(m.index, fontsize=6)
ax.set_xlabel("Familia predicha", fontsize=8)
ax.set_ylabel("Familia verdadera", fontsize=8)
ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
ax.grid(which="minor", color="#e6e6e6", linewidth=0.3)
ax.tick_params(which="minor", length=0)
for i in range(n):
    for j in range(n):
        fuera = i != j and p[i, j] >= a.umbral
        diagonal = i == j and a.diagonal_menor is not None and p[i, j] < a.diagonal_menor
        if fuera or diagonal:
            ax.text(j, i, f"{p[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=4.5,
                    color="white" if p[i, j] > 0.55 else "#1a1a1a")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.set_label(f"Proporción de las {a.unidad} de la familia verdadera", fontsize=7)
cb.ax.tick_params(labelsize=6)
cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
fig.tight_layout()
fig.savefig(a.salida, bbox_inches="tight")
print("guardada", a.salida)
