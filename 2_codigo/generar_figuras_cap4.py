#!/usr/bin/env python3
"""
generar_figuras_cap4.py — Figuras del capítulo 4, LEÍDAS de los archivos de resultados.

Reescrito el 2026-08-17 (auditoría de la descarga): la versión anterior tenía las cifras
hardcodeadas de corridas superadas (29 familias, Exp. 2b pre-corrección). Esta versión lee:

  4_resultados/resultados_bytes_multisemilla/..._job3648/    Exp. 2c, diez semillas (desde 2026-09-28;
                                                             antes, la corrida única del job 3639)
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
from collections import defaultdict
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
# Exactitud y macro-F1 en validación cruzada (media de 5 semillas) del Exp. 2e (job 4058) y del
# sistema completo del Exp. 2g (job 4091), transcritas del log pegado por Romina y registradas en
# ESTADO_TESIS.md. Si las carpetas de resultados están bajadas en 4_resultados/, se leen de ahí,
# de la fila explícita, y se verifica que coincidan: si no coinciden, el script se detiene.
EXACTITUD_2E, EXACTITUD_2G = 0.9357, 0.9998
MACROF1_2E, MACROF1_2G = 0.9359, 0.9998
# Frente de notas, cifra canónica (P2bal, 149 notas, 30 familias, 50 semillas), acordada con la
# sesión de notas el 2026-09-29: se lee de la fila explícita; si falta, el script se detiene.
P2BAL = RES / "resultados_protocolo_p2bal_149" / "p2bal_resumen.csv"


# ---------------------------------------------------------------- lectura de fuentes
# 2026-09-28: las dos lecturas del Exp. 2c pasan de la corrida única del job 3639 al promedio de
# DIEZ semillas del job 3648, que es la cifra que cita el capítulo (0,912 ± 0,002). La figura de
# progresión mostraba 0,909 al lado de un texto que decía 0,912.
MULTI_2C = RES / "resultados_bytes_multisemilla" / "resultados_bytes_multisemilla_job3648"


def leer_por_familia_2c():
    """F1 por familia del Exp. 2c, media de diez semillas (job 3648, 30 familias)."""
    suma, n = defaultdict(float), defaultdict(int)
    with open(MULTI_2C / "bytes_multisemilla_por_familia.csv", encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            suma[fila["familia"]] += float(fila["f1"])
            n[fila["familia"]] += 1
    f1 = {k: suma[k] / n[k] for k in suma}
    assert len(f1) == 30 and set(n.values()) == {10}, "se esperaban 30 familias x 10 semillas"
    return f1


def leer_resumen_2c():
    """Exactitud y macro-F1 del Exp. 2c, media de diez semillas (job 3648)."""
    with open(MULTI_2C / "bytes_multisemilla_resumen.csv", encoding="utf-8") as fh:
        filas = {f["metrica"]: float(f["media"]) for f in csv.DictReader(fh)}
    return filas["accuracy"], filas["f1_macro"]


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
    return con_firma, solo_ext, sin_marca, leer_ablacion_2b_10semillas()


def leer_ablacion_2b_10semillas():
    """Ablación del Exp. 2b con criterio 0,90: media de las diez semillas del job 3651, que es la
    base de la tabla del capítulo (57,2 % de cobertura, 0,563 de exactitud global para las firmas).
    Hasta el 2026-09-28 la figura leía la corrida única del job 3638 (54 % y 0,533)."""
    import glob
    mans = sorted(glob.glob(str(RES / "resultados_estructural_10semillas" /
                                "resultados_estructural_s*_job3651" / "manifiesto.json")))
    assert len(mans) == 10, f"se esperaban 10 semillas del job 3651, hay {len(mans)}"
    abl = defaultdict(lambda: defaultdict(float))
    for m in mans:
        d = json.loads(Path(m).read_text(encoding="utf-8"))["criterios"]["umbral_90"]["ablacion"]
        for modalidad in ("solo_extension", "solo_firmas_binarias", "combinado"):
            for clave in ("exactitud", "cobertura"):
                abl[modalidad][clave] += d[modalidad][clave] / len(mans)
    return abl


def leer_notas_p2bal():
    """Macro-F1 sobre 30 familias del frente de notas bajo P2bal: texto solo y cascada.

    Hasta el 2026-09-29 se leía el MÁXIMO de corrida_canonica_resumen.csv (base de 146 notas), que
    podía levantar una vista que el texto no cita. Ahora se lee la fila explícita y, si falta, se
    detiene: una figura generada con otra cifra es peor que una que no se genera."""
    with open(P2BAL, encoding="utf-8-sig") as fh:   # el CSV trae BOM
        filas = {(f["protocolo"], f["capa"]): float(f["f1_macro_30"]) for f in csv.DictReader(fh)}
    faltan = [k for k in (("P2bal", "texto solo"), ("P2bal", "M.6 (cascada)")) if k not in filas]
    assert not faltan, f"faltan filas en {P2BAL.name}: {faltan}"
    return filas[("P2bal", "texto solo")], filas[("P2bal", "M.6 (cascada)")]


def leer_referencia_19feat():
    man = json.loads((RES / "resultados_gridsearch_estadisticas" /
                      "gridsearch_estadisticas_manifiesto.json").read_text(encoding="utf-8"))
    return float(man["referencia_sin_ajustar"])


def leer_cv_2e_2g():
    """Exactitud y macro-F1 medios del Exp. 2e (bytes + estructura) y del sistema completo del
    Exp. 2g, de la fila explícita de cada resumen. Devuelve {(exp, métrica): valor}."""
    import pandas as pd
    valores = {}
    for clave, carpeta, archivo, fila, fijos in (
            ("2e", "resultados_exp2e_job4058", "exp2e_resumen.csv", "2_bytes_mas_estructura",
             {"accuracy": EXACTITUD_2E, "f1_macro": MACROF1_2E}),
            ("2g", "resultados_exp2g_job4091", "cv_resumen.csv", "5_bytes_estructura_extension",
             {"accuracy": EXACTITUD_2G, "f1_macro": MACROF1_2G})):
        ruta = RES / carpeta / archivo
        if not ruta.exists():
            print(f"  {clave}: {fijos} (transcritas del log; {carpeta} no está bajada)")
            valores.update({(clave, m): v for m, v in fijos.items()})
            continue
        t = pd.read_csv(ruta, header=[0, 1], index_col=0)
        assert fila in t.index, f"{clave}: falta la fila {fila} en {archivo}"
        for m, fijo in fijos.items():
            v = float(t.loc[fila, (m, "mean")])
            assert abs(v - fijo) < 5e-5, f"{clave}/{m}: el CSV dice {v}, el log decía {fijo}"
            valores[(clave, m)] = v
        print(f"  {clave}: leídas de {carpeta}/{archivo}, coinciden con el log")
    return valores


# ---------------------------------------------------------------- Figura 1
def fig_progresion(acc_2c, abl, acc_2e, acc_2g):
    """Progresión de enfoques sobre los archivos cifrados, hasta el sistema completo."""
    v_19 = leer_referencia_19feat()
    v_firmas = abl["solo_firmas_binarias"]["exactitud"]
    c_firmas = abl["solo_firmas_binarias"]["cobertura"]
    v_ext = abl["solo_extension"]["exactitud"]
    c_ext = abl["solo_extension"]["cobertura"]

    etiquetas = ["Entropía global\n+ tamaño\n(2 caract., 29 fam.)",
                 "Estadísticas\nregionales\n(19 caract., 29 fam.)",
                 "Firmas binarias\nexactas\n(Exp. 2b)",
                 "Extensión\ndel archivo\n(Exp. 2b)",
                 "Aprendizaje\nsobre bytes\n(Exp. 2c)",
                 "Bytes + rasgos\nestructurales\n(Exp. 2e)",
                 "Sistema completo:\n+ forma de la\nextensión (Exp. 2g)"]
    valores = [EXACTITUD_2FEAT_29FAM, v_19, v_firmas, v_ext, acc_2c, acc_2e, acc_2g]
    cobertura = [1.00, 1.00, c_firmas, c_ext, 1.00, 1.00, 1.00]
    colores = [GRIS, AZUL, NARANJA, "#b0b0b0", VERDE, VERDE, "#1f3f5c"]

    fig, ax = plt.subplots(figsize=(11.0, 4.9))
    x = np.arange(len(valores))
    barras = ax.bar(x, valores, color=colores, width=0.62, edgecolor="white")
    barras[-1].set_hatch("//")

    ax.axhline(1 / 30, color=ROJO, ls="--", lw=1.2, label="azar (1/30 = 0,033)")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)

    for i, (b, v, c) in enumerate(zip(barras, valores, cobertura)):
        cx = b.get_x() + b.get_width() / 2
        # cuatro decimales cerca del techo: 0,9998 redondeado a tres dice «1,000», que no es cierto
        txt = (f"{v:.4f}" if v > 0.995 else f"{v:.3f}").replace(".", ",")
        ax.text(cx, v + 0.018, txt, ha="center", fontsize=10, fontweight="bold")
        if c < 1.0:
            ax.text(cx, v / 2, f"cobertura\n{c:.0%}", ha="center", va="center",
                    fontsize=8, color="white")
        if i in (3, 6):
            ax.text(cx, v + 0.075, "usa el nombre\ndel archivo", ha="center", fontsize=8,
                    color=ROJO, style="italic")

    ax.set_xticks(x, etiquetas, fontsize=8.4)
    ax.set_ylabel("Exactitud multiclase")
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title("Identificación de familia a partir de archivos cifrados: la información no está\n"
                 "en la aleatoriedad del cifrado sino en lo que cada familia agrega al archivo",
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
    ax.set_xlabel("F1 por familia (Experimento 2c: aprendizaje sobre bytes, 30 familias,\n"
                  "media de diez semillas)")
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
def fig_simetria(notas_texto, notas_cascada, arch_contenido, arch_completo):
    """En cada frente, la señal del contenido sola y con la señal ligada a la campaña, macro-F1.

    Rediseñada el 2026-09-29 con la sesión del frente de notas: la versión anterior contrastaba
    tipo de señal en archivos (extensión contra bytes) y protocolo en notas (P1 contra P2, la misma
    señal en las dos barras), bajo una misma leyenda, y mezclaba macro-F1 con exactitud."""
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    grupos = ["Notas de rescate", "Archivos cifrados"]
    contenido = [notas_texto, arch_contenido]
    con_campana = [notas_cascada, arch_completo]
    etiq_contenido = ["Texto (TF-IDF + SVC)", "Contenido (bytes + estructura)"]
    etiq_campana = ["Texto + reglas de campaña", "Contenido + extensión (sistema completo)"]

    x = np.arange(2)
    ancho = 0.34
    b1 = ax.bar(x - ancho / 2, contenido, ancho, color=AZUL, label="Señal del contenido",
                edgecolor="white")
    b2 = ax.bar(x + ancho / 2, con_campana, ancho, color="#8a8a8a",
                label="Contenido + señal ligada a la campaña", edgecolor="white")

    for barras, vals, etiq, color in ((b1, contenido, etiq_contenido, "white"),
                                      (b2, con_campana, etiq_campana, "white")):
        for b, v, e in zip(barras, vals, etiq):
            txt = (f"{v:.4f}" if v > 0.995 else f"{v:.3f}").replace(".", ",")
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, txt,
                    ha="center", fontsize=9.5, fontweight="bold")
            ax.text(b.get_x() + b.get_width() / 2, 0.03, e, ha="center", fontsize=7.6,
                    rotation=90, va="bottom", color=color)

    ax.set_xticks(x, grupos, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Macro-F1")
    ax.set_title("Aporte de la señal ligada a la campaña sobre la señal del contenido,\n"
                 "en los dos frentes", fontsize=10, pad=12)
    ax.legend(fontsize=8.6, loc="upper left", ncol=2, frameon=False)
    fig.text(0.5, 0.012, "Archivos: validación cruzada, una campaña por familia (la ganancia es una "
             "cota superior). Notas: P2bal, plantilla nunca vista.", ha="center", fontsize=7.8,
             style="italic", color="#333333")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
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
    notas_texto, notas_cascada = leer_notas_p2bal()
    print(f"  fuentes: 2c acc={acc_2c} | 2b firmas={len(con_firma)} ext={len(solo_ext)} "
          f"sin={len(sin_marca)} | notas P2bal texto={notas_texto} cascada={notas_cascada}")
    cv = leer_cv_2e_2g()
    fig_progresion(acc_2c, abl, cv[("2e", "accuracy")], cv[("2g", "accuracy")])
    fig_por_familia(f1, con_firma, solo_ext, sin_marca)
    fig_simetria(notas_texto, notas_cascada, cv[("2e", "f1_macro")], cv[("2g", "f1_macro")])
    fig_ablacion_extendida()
    print("Listo.")
