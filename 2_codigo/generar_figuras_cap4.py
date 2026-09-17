#!/usr/bin/env python3
"""
generar_figuras_cap4.py — Figuras del capítulo 4, LEÍDAS de los archivos de resultados.

Reescrito el 2026-08-17 (auditoría de la descarga): la versión anterior tenía las cifras
hardcodeadas de corridas superadas (29 familias, Exp. 2b pre-corrección). Esta versión lee:

  4_resultados/resultados_bytes/bytes_por_familia.txt        Exp. 2c por familia (job 3639)
  4_resultados/resultados_bytes/bytes_resumen.csv            Exp. 2c resumen (job 3639)
  4_resultados/resultados_estructural/manifiesto.json        Exp. 2b corregido (job 3638)
  4_resultados/resultados_estructural/marcas_por_familia_umbral_90.csv
  4_resultados/resultados_ablacion_extendida/a_curva_ablacion.csv   (job 3630)
  4_resultados/resultados_ablacion_extendida/b_bloque_medio.csv     (job 3630)
  4_resultados/resultados_canonicos/corrida_canonica_resumen.csv    Exp. 3 (notas)
  4_resultados/resultados_gridsearch_estadisticas/gridsearch_estadisticas_manifiesto.json

Únicas cifras no disponibles en CSV local (provienen de la salida SLURM del job 3547,
transcritas en la tabla §4.3.1 del documento): la exactitud con 2 características (0,166).
La referencia de 19 características (0,603) se lee del manifiesto del gridsearch.

Salida: 1_documento/Plantilla_de_Tesis___Romina_Carlos/images/
"""
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "4_resultados"
IMG = RAIZ / "1_documento" / "Plantilla_de_Tesis___Romina_Carlos" / "images"
IMG.mkdir(parents=True, exist_ok=True)

GRIS = "#4d4d4d"
AZUL = "#2c5f8a"
NARANJA = "#c8763c"
VERDE = "#4a7c59"
ROJO = "#a83e3e"

# Única cifra sin CSV local: baseline de 2 características (SLURM job 3547; ver §4.3.1).
EXACTITUD_2FEAT_29FAM = 0.166


# ---------------------------------------------------------------- lectura de fuentes
def leer_por_familia_2c():
    """Parsea el classification_report del Exp. 2c (job 3639, 30 familias)."""
    txt = (RES / "resultados_bytes" / "bytes_por_familia.txt").read_text(encoding="utf-8")
    f1 = {}
    for linea in txt.splitlines():
        m = re.match(r"\s*([A-Z]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", linea)
        if m and m.group(1) not in ("accuracy",):
            f1[m.group(1)] = float(m.group(4))
    assert len(f1) == 30, f"se esperaban 30 familias, hay {len(f1)}"
    return f1


def leer_resumen_2c():
    """Fila final del Exp. 2c (job 3639)."""
    with open(RES / "resultados_bytes" / "bytes_resumen.csv", encoding="utf-8") as fh:
        filas = [f for f in csv.DictReader(fh) if f["etapa"] == "final"]
    assert len(filas) == 1
    return float(filas[0]["accuracy"]), float(filas[0]["f1_macro"])


def leer_marcas_2b():
    """Clasifica cada familia según su marca en el Exp. 2b corregido (umbral 0,90).
    Firma binaria = prefijo o sufijo >= 4 bytes (min_marca del manifiesto)."""
    man = json.loads((RES / "resultados_estructural" / "manifiesto.json")
                     .read_text(encoding="utf-8"))
    min_marca = man["min_marca"]
    con_firma, solo_ext, sin_marca = set(), set(), set()
    with open(RES / "resultados_estructural" / "marcas_por_familia_umbral_90.csv",
              encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            fam = fila["familia"]
            tiene_firma = (int(fila["prefijo_len"]) >= min_marca
                           or int(fila["sufijo_len"]) >= min_marca)
            if tiene_firma:
                con_firma.add(fam)
            elif fila["marca_detectable"] == "True":
                solo_ext.add(fam)
            else:
                sin_marca.add(fam)
    abl = man["criterios"]["umbral_90"]["ablacion"]
    return con_firma, solo_ext, sin_marca, abl


def leer_notas():
    """Mejor macro-F1 por protocolo de la corrida canónica de notas."""
    mejor = {"grupos": 0.0, "estratificado": 0.0}
    with open(RES / "resultados_canonicos" / "corrida_canonica_resumen.csv",
              encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            p = fila["protocolo"]
            mejor[p] = max(mejor[p], float(fila["f1_macro_mean"]))
    return mejor["estratificado"], mejor["grupos"]  # P1, P2


def leer_referencia_19feat():
    man = json.loads((RES / "resultados_gridsearch_estadisticas" /
                      "gridsearch_estadisticas_manifiesto.json").read_text(encoding="utf-8"))
    return float(man["referencia_sin_ajustar"])


# ---------------------------------------------------------------- Figura 1
def fig_progresion(acc_2c, abl):
    """Progresión de enfoques sobre los archivos cifrados."""
    v_19 = leer_referencia_19feat()
    v_firmas = abl["solo_firmas_binarias"]["exactitud"]
    c_firmas = abl["solo_firmas_binarias"]["cobertura"]
    v_ext = abl["solo_extension"]["exactitud"]
    c_ext = abl["solo_extension"]["cobertura"]

    etiquetas = ["Entropía global\n+ tamaño\n(2 caract., 29 fam.)",
                 "Estadísticas\nregionales\n(19 caract., 29 fam.)",
                 "Firmas binarias\nexactas\n(Exp. 2b)",
                 "Extensión\ndel archivo\n(Exp. 2b)",
                 "Aprendizaje\nsobre bytes\n(Exp. 2c)"]
    valores = [EXACTITUD_2FEAT_29FAM, v_19, v_firmas, v_ext, acc_2c]
    cobertura = [1.00, 1.00, c_firmas, c_ext, 1.00]
    colores = [GRIS, AZUL, NARANJA, "#b0b0b0", VERDE]

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(len(valores))
    barras = ax.bar(x, valores, color=colores, width=0.62, edgecolor="white")

    ax.axhline(1 / 30, color=ROJO, ls="--", lw=1.2)
    ax.text(len(valores) - 0.42, 0.055, "azar (1/30 = 0,033)", color=ROJO,
            fontsize=8.5, ha="right")

    for b, v, c in zip(barras, valores, cobertura):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.3f}".replace(".", ","),
                ha="center", fontsize=10, fontweight="bold")
        if c < 1.0:
            ax.text(b.get_x() + b.get_width() / 2, v / 2,
                    f"cobertura\n{c:.0%}", ha="center", va="center",
                    fontsize=8, color="white")

    ax.text(3, -0.075, "usa el nombre\ndel archivo", ha="center", fontsize=8,
            color=ROJO, style="italic")

    ax.set_xticks(x, etiquetas, fontsize=8.6)
    ax.set_ylabel("Exactitud multiclase")
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
def fig_por_familia(f1, con_firma, solo_ext, sin_marca):
    """F1 por familia en el Exp. 2c, coloreado según la marca del Exp. 2b corregido."""
    orden = sorted(f1, key=f1.get)
    vals = [f1[k] for k in orden]

    def color(fam):
        if fam in con_firma:
            return VERDE
        if fam in sin_marca:
            return ROJO
        return NARANJA

    fig, ax = plt.subplots(figsize=(7.6, 8.4))
    ax.barh(range(len(orden)), vals,
            color=[color(k) for k in orden], edgecolor="white", height=0.72)
    ax.set_yticks(range(len(orden)), orden, fontsize=8.4)
    for i, v in enumerate(vals):
        ax.text(v + 0.012, i, f"{v:.2f}".replace(".", ","), va="center", fontsize=7.8)

    ax.axvline(0.98, color=GRIS, ls=":", lw=1)
    ax.set_xlim(0, 1.09)
    ax.set_xlabel("F1 por familia (Experimento 2c: aprendizaje sobre bytes, 30 familias)")
    ax.set_title("Las familias que dejan estructura en el archivo se identifican\n"
                 "casi perfectamente; las que no, forman un grupo de confusión mutua",
                 fontsize=10.5, pad=12)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=VERDE, label=f"Con firma binaria en el Exp. 2b ({len(con_firma)})"),
        Patch(facecolor=NARANJA, label=f"Solo extensión propia en el Exp. 2b ({len(solo_ext)})"),
        Patch(facecolor=ROJO, label=f"Sin marca detectable en el Exp. 2b ({len(sin_marca)})"),
    ], loc="lower right", fontsize=8.4, framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(IMG / "fig_f1_por_familia_bytes.png", dpi=220)
    print("  fig_f1_por_familia_bytes.png")


# ---------------------------------------------------------------- Figura 3
def fig_simetria(p1, p2, abl, acc_2c):
    """Señal fácil (campaña/plantilla) vs señal robusta (contenido), en los dos frentes."""
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    grupos = ["Notas de rescate\n(macro-F1)", "Archivos cifrados\n(exactitud)"]
    facil = [p1, abl["solo_extension"]["exactitud"]]
    robusta = [p2, acc_2c]
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


# ---------------------------------------------------------------- Figura 4
def fig_ablacion_extendida():
    """Curva de ventana (con control sin relleno) + bloque del medio. Exactitud."""
    curva = {"todos": [], "sin_relleno": []}
    with open(RES / "resultados_ablacion_extendida" / "a_curva_ablacion.csv",
              encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            curva[fila["subconjunto"]].append(
                (int(fila["n_bytes"]), float(fila["accuracy"])))
    medio = []
    with open(RES / "resultados_ablacion_extendida" / "b_bloque_medio.csv",
              encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            medio.append((fila["config"], float(fila["accuracy"])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    for clave, etiqueta, color, marcador in (
            ("todos", "Corpus completo (15.000)", AZUL, "o"),
            ("sin_relleno", "Solo archivos sin relleno (14.783)", VERDE, "s")):
        xs = [b for b, _ in sorted(curva[clave])]
        ys = [a for _, a in sorted(curva[clave])]
        ax1.plot(xs, ys, marker=marcador, ms=5, lw=1.6, color=color, label=etiqueta)
    ax1.axvline(1024, color=GRIS, ls=":", lw=1)
    ax1.text(1024 * 1.08, 0.80, "máximo:\n512+512", fontsize=8.2, color=GRIS)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([b for b, _ in sorted(curva["todos"])],
                   ["64+64", "128+128", "256+256", "512+512",
                    "1024+1024", "2048+2048", "4096+4096"],
                   rotation=45, fontsize=8)
    ax1.set_xlabel("Ventana (bytes de cabecera + cola)")
    ax1.set_ylabel("Exactitud (30 familias)")
    ax1.set_title("(a) La curva satura en 512+512 bytes", fontsize=10)
    ax1.legend(fontsize=8.2, loc="lower right")
    ax1.grid(alpha=0.25, lw=0.6)
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)

    nombres = {"solo medio 1024": "Solo medio\n(1.024 B)",
               "solo cabecera 512": "Solo cabecera\n(512 B)",
               "solo cola 512": "Solo cola\n(512 B)",
               "cabecera+cola 512": "Cabecera\n+ cola",
               "cabecera+cola+medio 512": "Cabecera + cola\n+ medio"}
    etiq = [nombres[c] for c, _ in medio]
    vals = [v for _, v in medio]
    colores = [ROJO, NARANJA, NARANJA, VERDE, AZUL]
    barras = ax2.bar(range(len(vals)), vals, color=colores, width=0.62, edgecolor="white")
    for b, v in zip(barras, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}".replace(".", ","),
                 ha="center", fontsize=8.6, fontweight="bold")
    ax2.axhline(1 / 30, color=ROJO, ls="--", lw=1)
    ax2.text(len(vals) - 0.45, 0.05, "azar", color=ROJO, fontsize=8, ha="right")
    ax2.set_xticks(range(len(etiq)), etiq, fontsize=7.8)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("(b) Los bytes del medio no llevan información", fontsize=10)
    ax2.grid(axis="y", alpha=0.25, lw=0.6)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(IMG / "fig_ablacion_extendida.png", dpi=220)
    print("  fig_ablacion_extendida.png")


if __name__ == "__main__":
    print(f"Generando figuras en {IMG}")
    f1 = leer_por_familia_2c()
    acc_2c, _ = leer_resumen_2c()
    con_firma, solo_ext, sin_marca, abl = leer_marcas_2b()
    p1, p2 = leer_notas()
    print(f"  fuentes: 2c acc={acc_2c} | 2b firmas={len(con_firma)} ext={len(solo_ext)} "
          f"sin={len(sin_marca)} | notas P1={p1:.3f} P2={p2:.3f}")
    fig_progresion(acc_2c, abl)
    fig_por_familia(f1, con_firma, solo_ext, sin_marca)
    fig_simetria(p1, p2, abl, acc_2c)
    fig_ablacion_extendida()
    print("Listo.")
