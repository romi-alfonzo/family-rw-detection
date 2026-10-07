#!/usr/bin/env python3
"""
generar_figuras_cap2.py — Figuras del capítulo 2 (marco teórico).

Figura de histogramas de bytes: muestra por qué la entropía reconoce un archivo cifrado pero no lo
distingue de uno comprimido. Usa tres versiones de un mismo contenido:

  texto       los primeros 256 KiB del código fuente de este repositorio (los .py de 2_codigo,
              en orden alfabético): texto plano, sin datos del corpus;
  comprimido  ese texto comprimido con DEFLATE (zlib, nivel 9), el algoritmo de los ZIP;
  cifrado     ese texto combinado por XOR con una secuencia pseudoaleatoria del mismo largo
              (semilla 0), como hace un cifrado de flujo.

Imprime la entropía de Shannon (bits por byte) y el estadístico chi-cuadrado de cada versión, que
son las cifras que cita el capítulo. No usa archivos de NapierOne ni notas de rescate.

Salida: 1_documento/tesis_v4_en_preparacion/images/fig_mt_histogramas_bytes.png
"""
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

RAIZ = Path(__file__).resolve().parent.parent
IMG = RAIZ / "1_documento" / "tesis_v4_en_preparacion" / "images"
IMG.mkdir(parents=True, exist_ok=True)

AZUL = "#2c5f8a"
TAM = 256 * 1024


def entropia(datos: bytes) -> float:
    cuentas = np.bincount(np.frombuffer(datos, dtype=np.uint8), minlength=256)
    p = cuentas[cuentas > 0] / len(datos)
    return float(-(p * np.log2(p)).sum())


def chi_cuadrado(datos: bytes) -> float:
    cuentas = np.bincount(np.frombuffer(datos, dtype=np.uint8), minlength=256)
    esperado = len(datos) / 256
    return float(((cuentas - esperado) ** 2 / esperado).sum())


def main():
    fuentes = sorted((RAIZ / "2_codigo").glob("*.py"))
    texto = b"".join(f.read_bytes() for f in fuentes)[:TAM]
    assert len(texto) == TAM, "no alcanza el código fuente para 256 KiB"
    comprimido = zlib.compress(texto, 9)
    clave = np.random.default_rng(0).integers(0, 256, size=len(texto), dtype=np.uint8).tobytes()
    cifrado = bytes(a ^ b for a, b in zip(texto, clave))

    versiones = [("Texto plano", texto), ("Comprimido (DEFLATE)", comprimido), ("Cifrado", cifrado)]
    fig, ejes = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=False)
    for eje, (nombre, datos) in zip(ejes, versiones):
        h, chi = entropia(datos), chi_cuadrado(datos)
        print(f"{nombre:22s} {len(datos):8d} bytes   H = {h:.3f} bits/byte   chi2 = {chi:,.0f}")
        frec = np.bincount(np.frombuffer(datos, dtype=np.uint8), minlength=256) / len(datos)
        eje.bar(np.arange(256), frec, width=1.0, color=AZUL)
        eje.set_title(f"{nombre}\nH = {h:.2f} bits por byte".replace(".", ","), fontsize=10)
        eje.set_xlim(0, 255)
        eje.set_xticks([0, 64, 128, 192, 255])
        eje.set_xlabel("Valor del byte", fontsize=9)
        eje.tick_params(labelsize=8)
        eje.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        eje.spines[["top", "right"]].set_visible(False)
    ejes[0].set_ylabel("Frecuencia relativa", fontsize=9)
    fig.tight_layout()
    salida = IMG / "fig_mt_histogramas_bytes.png"
    fig.savefig(salida, dpi=300)
    print("figura:", salida)


if __name__ == "__main__":
    main()
