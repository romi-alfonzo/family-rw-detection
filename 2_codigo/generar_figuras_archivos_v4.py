#!/usr/bin/env python3
"""
generar_figuras_archivos_v4.py — Curva de aprendizaje del frente de archivos, LEÍDA de los resultados, con el mismo
estilo que la del frente de notas (generar_figuras_notas_v4.py).

  fig_curva_archivos.png   Macro-F1 medio con intervalo del 95 % según los archivos por familia (10 a 500), con
                           subconjuntos anidados, solo bytes, 30 familias y tres semillas (Exp. 2c, Sección 5.1.8).
                           Fuente: 4_resultados/resultados_exp2d/resultados_exp2d_job3937/a3_curva_por_semilla.csv.

  fig_marcas_bytes.png     La marca de la familia en los bytes reales de tres archivos cifrados (Exp. 2b, Sección 5.1.3):
                           el prefijo de WANNACRY («WANACRY!») y el sufijo de PHOBOS («LOCK96»). Pedida el 2026-10-31.

Pedida por Romina el 2026-10-30 para la guía de defensa y anotada para la tesis (observaciones de la v4). El intervalo
se calcula igual que en la figura de notas: 1,96 · desvío / √n sobre las semillas. Salida:
1_documento/tesis_v4_en_preparacion/images/
"""
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
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


def figura_ventanas():
    """fig_ventanas_archivos.png: macro-F1 según los bytes leídos por archivo (cabecera + cola), sobre los 15.000 archivos
    de la ablación de ventanas (Sección 5.1.4, «Ubicación de la señal»). Una sola línea: el control con los archivos que
    no necesitan relleno queda en el CSV y no se dibuja. Fuente: resultados_ablacion_extendida/a_curva_ablacion.csv."""
    puntos = []
    fuente = RES / "resultados_ablacion_extendida" / "a_curva_ablacion.csv"
    for f in csv.DictReader(open(fuente, encoding="utf-8")):
        if f["subconjunto"] == "todos":
            puntos.append((int(f["n_bytes"]), float(f["f1_macro"])))
    puntos.sort()
    xs = [x for x, _ in puntos]
    ms = [m for _, m in puntos]
    # control contra el texto de la tesis (experimentos.tex, subsec:exp2c_ventana): máximo 0,905 con 512 bytes por
    # extremo, 0,901 con 4.096, y el 87 % del máximo con 128 bytes leídos
    medias = dict(puntos)
    assert max(ms) == medias[1024] and round(medias[1024], 3) in (0.904, 0.905) and round(medias[8192], 3) == 0.901, medias
    assert round(medias[128] / medias[1024], 2) == 0.87, medias

    fig, eje = plt.subplots(figsize=(6.4, 3.4))
    eje.plot(xs, ms, color=AZUL, linewidth=2, marker="o", markersize=5)
    eje.axvline(1024, color=GRIS, linewidth=1, linestyle=(0, (3, 3)))
    eje.annotate("máximo: 512 bytes\nde cada extremo", xy=(1024, medias[1024]), xytext=(1150, 0.83),
                 fontsize=8, color="#333333", va="center")
    eje.set_xscale("log", base=2)
    eje.xaxis.set_major_locator(FixedLocator(xs))
    eje.xaxis.set_minor_locator(NullLocator())
    eje.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v)):,}".replace(",", ".")))
    eje.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    eje.set_xlabel("Bytes leídos por archivo (cabecera + cola, escala logarítmica)", fontsize=9)
    eje.set_ylabel("Macro-F1", fontsize=9)
    estilo(eje)
    fig.tight_layout()
    salida = IMG / "fig_ventanas_archivos.png"
    fig.savefig(salida, dpi=300)
    plt.close(fig)
    print("figura:", salida)


def figura_marcas_bytes():
    """fig_marcas_bytes.png: la marca de la familia en los bytes reales de tres archivos cifrados (Sección 5.1.3, Exp. 2b).
    (a) WANNACRY, los primeros 16 bytes: el prefijo común de 12 bytes se lee «WANACRY!». (b) PHOBOS, los últimos 10
    bytes: el sufijo común de 7 bytes se lee «LOCK96». Es la misma idea que los paneles de la presentación para Cappo,
    pero con bytes reales: allí solo la marca era real y el resto de los bytes, ilustrativo.

    Fuentes: los bytes de cada archivo salen del censo del job 4082 (resultados_diagnostico_dificiles_job4082/
    por_archivo.csv, columnas pre16 y suf16: los 16 primeros y los 16 últimos bytes de cada archivo); el largo y los
    bytes de la marca, del Experimento 2b (resultados_estructural/marcas_por_familia_umbral_90.csv, como la Tabla A.3).
    En PHOBOS se muestran 10 bytes y no 16: los seis anteriores (24 56 A8 42 A3 23) se repiten en sus 1.000 archivos,
    pero no forman parte del sufijo común, porque los separan del final tres bytes que cambian entre archivos, y el
    procedimiento solo cuenta los bytes comunes contiguos al final."""
    archivos = list(csv.DictReader(open(RES / "resultados_diagnostico_dificiles_job4082" / "por_archivo.csv",
                                        encoding="utf-8")))
    marcas = {f["familia"]: f for f in csv.DictReader(open(RES / "resultados_estructural" /
                                                           "marcas_por_familia_umbral_90.csv", encoding="utf-8"))}

    def original(nombre):  # el nombre del documento antes de que el ransomware le agregue nada
        return re.match(r"\d{4}-[^.]+\.\w+", nombre).group(0)

    def bytes_de(familia, nombres, columna, n):
        filas = []
        for nombre in nombres:
            f = next(a for a in archivos if a["familia"] == familia and original(a["archivo"]) == nombre)
            b = bytes.fromhex(f[columna])
            filas.append((nombre, b[:n] if columna == "pre16" else b[-n:]))
        return filas

    wannacry = bytes_de("WANNACRY", ["0001-doc.doc", "0001-jpg-fromweb.jpg", "0001-pdf.pdf"], "pre16", 16)
    phobos = bytes_de("PHOBOS", ["0001-doc.doc", "0001-jpg-fromweb.jpg", "0002-docx.docx"], "suf16", 10)
    prefijo = bytes.fromhex(marcas["WANNACRY"]["prefijo_hex"])
    sufijo = bytes.fromhex(marcas["PHOBOS"]["sufijo_hex"])
    # control contra el Exp. 2b y el texto de la tesis (Tabla A.3: prefijo de 12 bytes y sufijo de 7; 5.1.3)
    assert len(prefijo) == 12 and prefijo[:8] == b"WANACRY!" and all(b[:12] == prefijo for _, b in wannacry)
    assert len(sufijo) == 7 and sufijo[1:] == b"LOCK96" and all(b[-7:] == sufijo for _, b in phobos)
    # lo que no es marca cambia de un archivo a otro
    assert len({b[12:] for _, b in wannacry}) == 3 and len({b[:3] for _, b in phobos}) == 3

    ETIQ, ALTO_FILA = 4.6, 1.0  # ancho de la columna de nombres y alto de una fila, en celdas
    MONO = "DejaVu Sans Mono"
    CELESTE, TINTA_MARCA, TINTA, BORDE = "#dce8f6", "#1b3a5c", "#555555", "#d4d4d4"
    fig = plt.figure(figsize=(6.4, 3.9))
    eje = fig.add_axes([0, 0, 1, 1])
    eje.set_axis_off()
    ancho = ETIQ + 16 + 0.4
    eje.set_xlim(0, ancho)
    eje.set_ylim(0, ancho * 3.9 / 6.4)

    def panel(y, titulo, filas, marca, posiciones):
        """y: borde superior del panel; marca: columnas de la marca (inicio, fin incluido)."""
        eje.text(0.15, y, titulo, fontsize=8, color="#222222", va="top")
        y_pos = y - 1.05
        for j, p in enumerate(posiciones):
            eje.text(ETIQ + j + 0.46, y_pos, p, fontsize=5.5, color="#8a8a8a", ha="center", va="center")
        y0 = y_pos - 0.45
        for i, (nombre, b) in enumerate(filas):
            yf = y0 - (i + 1) * ALTO_FILA
            eje.text(ETIQ - 0.25, yf + 0.41, nombre, fontsize=7, color=TINTA, ha="right", va="center")
            for j, byte in enumerate(b):
                es_marca = marca[0] <= j <= marca[1]
                eje.add_patch(Rectangle((ETIQ + j + 0.04, yf), 0.84, 0.82, facecolor=CELESTE if es_marca else "white",
                                        edgecolor="none" if es_marca else BORDE, linewidth=0.6))
                eje.text(ETIQ + j + 0.46, yf + 0.41, f"{byte:02X}", fontsize=7.2, family=MONO, ha="center", va="center",
                         color=TINTA_MARCA if es_marca else TINTA, fontweight="bold" if es_marca else "normal")
        y_ult = y0 - len(filas) * ALTO_FILA
        eje.add_patch(FancyBboxPatch((ETIQ + marca[0] - 0.04, y_ult - 0.06), marca[1] - marca[0] + 1.0, len(filas) * ALTO_FILA + 0.12,
                                     boxstyle="round,pad=0,rounding_size=0.18", facecolor="none", edgecolor=AZUL,
                                     linewidth=1.4))
        y_lect = y_ult - 0.62
        eje.text(ETIQ - 0.25, y_lect, "se lee", fontsize=7, color="#8a8a8a", ha="right", va="center", style="italic")
        for j in range(marca[0], marca[1] + 1):
            c = filas[0][1][j]
            eje.text(ETIQ + j + 0.46, y_lect, chr(c) if 32 <= c < 127 else "·", fontsize=8, family=MONO,
                     fontweight="bold", color=AZUL, ha="center", va="center")
        return y_lect - 0.85

    y = eje.get_ylim()[1] - 0.25
    y = panel(y, "(a) WANNACRY: los primeros 16 bytes de tres archivos cifrados. En azul, el prefijo común de 12 bytes.",
              wannacry, (0, 11), [str(j) for j in range(16)])
    panel(y, "(b) PHOBOS: los últimos 10 bytes de tres archivos cifrados. En azul, el sufijo común de 7 bytes.",
          phobos, (3, 9), [f"−{10 - j}" for j in range(10)])
    salida = IMG / "fig_marcas_bytes.png"
    fig.savefig(salida, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("figura:", salida)


if __name__ == "__main__":
    figura_curva_archivos()
    figura_ventanas()
    figura_marcas_bytes()
