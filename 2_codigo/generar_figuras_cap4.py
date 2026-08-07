#!/usr/bin/env python3
"""
generar_figuras_cap4.py — Figuras del capítulo 4 a partir de los resultados reales.
Las cifras están tomadas de:
  4_resultados/resultados_estructural/  (Exp. 2b)
  4_resultados/resultados_bytes/        (Exp. 2c)
  4_resultados/resultados_canonicos/    (Exp. 3)
  las salidas SLURM del cluster NIDTEC  (Exp. 2 con 275 características)
Salida: 1_documento/Plantilla_de_Tesis___Romina_Carlos/images/
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
IMG = RAIZ / "1_documento" / "Plantilla_de_Tesis___Romina_Carlos" / "images"
IMG.mkdir(parents=True, exist_ok=True)

GRIS = "#4d4d4d"
AZUL = "#2c5f8a"
NARANJA = "#c8763c"
VERDE = "#4a7c59"
ROJO = "#a83e3e"


# ---------------------------------------------------------------- Figura 1
def fig_progresion():
    """Progresión de enfoques sobre los archivos cifrados."""
    etiquetas = ["Entropía global\n+ tamaño\n(2 caract.)",
                 "Estadísticas\nregionales\n(19 caract.)",
                 "Firmas binarias\nexactas\n(Exp. 2b)",
                 "Extensión\ndel archivo\n(Exp. 2b)",
                 "Aprendizaje\nsobre bytes\n(Exp. 2c)"]
    valores = [0.166, 0.603, 0.517, 0.828, 0.910]
    cobertura = [1.00, 1.00, 0.533, 0.828, 1.00]
    colores = [GRIS, AZUL, NARANJA, "#b0b0b0", VERDE]

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(len(valores))
    barras = ax.bar(x, valores, color=colores, width=0.62, edgecolor="white")

    ax.axhline(0.0345, color=ROJO, ls="--", lw=1.2)
    ax.text(len(valores) - 0.42, 0.055, "azar (1/29 = 0,034)", color=ROJO,
            fontsize=8.5, ha="right")

    for i, (b, v, c) in enumerate(zip(barras, valores, cobertura)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.3f}".replace(".", ","),
                ha="center", fontsize=10, fontweight="bold")
        if c < 1.0:
            ax.text(b.get_x() + b.get_width() / 2, v / 2,
                    f"cobertura\n{c:.0%}".replace("%", "\\%" if False else "%"),
                    ha="center", va="center", fontsize=8, color="white")

    # marcar cuál usa metadatos del nombre
    ax.text(3, -0.075, "usa el nombre\ndel archivo", ha="center", fontsize=8,
            color=ROJO, style="italic")

    ax.set_xticks(x, etiquetas, fontsize=8.6)
    ax.set_ylabel("Exactitud multiclase (29 familias)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Identificación de familia a partir de archivos cifrados:\n"
                 "la información no está en la aleatoriedad del cifrado sino en la estructura",
                 fontsize=10.5, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(IMG / "fig_progresion_archivos.png", dpi=220)
    print("  fig_progresion_archivos.png")


# ---------------------------------------------------------------- Figura 2
def fig_por_familia():
    """F1 por familia en el Exp. 2c, coloreado según qué marca tenía en el Exp. 2b."""
    f1 = {
        "AVOSLOCKER": 1.00, "BADRABBIT": 0.98, "BLACKCAT": 1.00, "BLACKMATTER": 0.99,
        "CERBER": 1.00, "CHIMERA": 1.00, "CLOP": 1.00, "CONTI": 1.00,
        "CRYPTOLOCKER": 0.61, "CUBA": 1.00, "DARKSIDE": 0.60, "DHARMA": 1.00,
        "GANDCRAB": 1.00, "HELLOKITTY": 0.99, "JIGSAW": 0.44, "LOCKBIT": 1.00,
        "LORENZ": 0.99, "MAZE": 1.00, "MEDUZALOCKER": 1.00, "NETWALKER": 1.00,
        "NOTPETYA": 0.38, "PHOBOS": 1.00, "RANSOMEXX": 1.00, "RYUK": 0.99,
        "SODINOKIBI": 1.00, "SUNCRYPT": 0.75, "TESLACRYPT": 1.00, "WANNACRY": 1.00,
        "WASTEDLOCKER": 0.64,
    }
    con_firma = {"BLACKCAT", "CERBER", "CONTI", "CUBA", "GANDCRAB", "HELLOKITTY",
                 "LOCKBIT", "LORENZ", "MAZE", "MEDUZALOCKER", "NETWALKER", "PHOBOS",
                 "RANSOMEXX", "TESLACRYPT", "WANNACRY"}
    sin_marca = {"BADRABBIT", "JIGSAW", "NOTPETYA", "SUNCRYPT"}

    orden = sorted(f1, key=f1.get)
    vals = [f1[k] for k in orden]

    def color(fam):
        if fam in con_firma:
            return VERDE
        if fam in sin_marca:
            return ROJO
        return NARANJA

    fig, ax = plt.subplots(figsize=(7.6, 8.2))
    ax.barh(range(len(orden)), vals,
            color=[color(k) for k in orden], edgecolor="white", height=0.72)
    ax.set_yticks(range(len(orden)), orden, fontsize=8.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.012, i, f"{v:.2f}".replace(".", ","), va="center", fontsize=8)

    ax.axvline(0.98, color=GRIS, ls=":", lw=1)
    ax.set_xlim(0, 1.09)
    ax.set_xlabel("F1 por familia (Experimento 2c: aprendizaje sobre bytes)")
    ax.set_title("Las familias que dejan estructura en el archivo se identifican\n"
                 "casi perfectamente; las que no, forman un grupo de confusión mutua",
                 fontsize=10.5, pad=12)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=VERDE, label="Con firma binaria en el Exp. 2b (15)"),
        Patch(facecolor=NARANJA, label="Solo extensión propia en el Exp. 2b (10)"),
        Patch(facecolor=ROJO, label="Sin marca detectable en el Exp. 2b (4)"),
    ], loc="lower right", fontsize=8.4, framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(IMG / "fig_f1_por_familia_bytes.png", dpi=220)
    print("  fig_f1_por_familia_bytes.png")


# ---------------------------------------------------------------- Figura 3
def fig_simetria():
    """Señal fácil (campaña/plantilla) vs señal robusta (contenido), en los dos frentes."""
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    grupos = ["Notas de rescate\n(macro-F1)", "Archivos cifrados\n(exactitud)"]
    facil = [0.760, 0.828]
    robusta = [0.435, 0.910]
    etiq_facil = ["P1: plantilla conocida", "Extensión del archivo"]
    etiq_robusta = ["P2: variante nunca vista", "Bytes del contenido"]

    x = np.arange(2)
    ancho = 0.34
    b1 = ax.bar(x - ancho / 2, facil, ancho, color="#b0b0b0",
                label="Señal ligada a la campaña/plantilla concreta", edgecolor="white")
    b2 = ax.bar(x + ancho / 2, robusta, ancho, color=AZUL,
                label="Señal del contenido", edgecolor="white")

    for b, v, e in zip(b1, facil, etiq_facil):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}".replace(".", ","),
                ha="center", fontsize=9.5, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 0.03, e, ha="center", fontsize=7.6,
                rotation=90, va="bottom", color="#333333")
    for b, v, e in zip(b2, robusta, etiq_robusta):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}".replace(".", ","),
                ha="center", fontsize=9.5, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 0.03, e, ha="center", fontsize=7.6,
                rotation=90, va="bottom", color="white")

    ax.set_xticks(x, grupos, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rendimiento")
    ax.set_title("En ambos artefactos, la señal ligada a la campaña concreta y la señal\n"
                 "del contenido difieren: en las notas la penaliza, en los archivos la supera",
                 fontsize=10, pad=12)
    ax.legend(fontsize=8.6, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(IMG / "fig_simetria_frentes.png", dpi=220)
    print("  fig_simetria_frentes.png")


if __name__ == "__main__":
    print(f"Generando figuras en {IMG}")
    fig_progresion()
    fig_por_familia()
    fig_simetria()
    print("Listo.")
