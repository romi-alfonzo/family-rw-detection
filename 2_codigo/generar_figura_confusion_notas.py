# -*- coding: utf-8 -*-
"""Figura de la matriz de confusión de la cascada del frente de notas (comentario 32 de Cappo, 10-11-2026).

Lee la matriz que escribió el análisis de linaje (149 notas, P2bal, 50 semillas; 7.450 decisiones, cuya diagonal da
la exactitud de 0,8123 del sistema) y la dibuja normalizada por fila: cada fila suma 1 y muestra a qué familias se
asignaron las notas de una familia verdadera. Escala secuencial de un solo tono, de blanco al azul de las demás
figuras de la tesis. Se rotulan solo las celdas fuera de la diagonal que reúnen al menos el 20 % de su fila.

Uso: python generar_figura_confusion_notas.py <matriz_confusion_cascada.csv> <salida.png>
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

entrada, salida = sys.argv[1], sys.argv[2]
m = pd.read_csv(entrada, index_col=0, encoding="utf-8-sig")
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
        if i != j and p[i, j] >= 0.20:
            ax.text(j, i, f"{p[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=4.5,
                    color="white" if p[i, j] > 0.55 else "#1a1a1a")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.set_label("Proporción de las notas de la familia verdadera", fontsize=7)
cb.ax.tick_params(labelsize=6)
cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
fig.tight_layout()
fig.savefig(salida, bbox_inches="tight")
print("guardada", salida)
