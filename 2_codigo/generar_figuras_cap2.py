#!/usr/bin/env python3
"""
generar_figuras_cap2.py — Figuras del capítulo 2 (marco teórico).

Las dos figuras usan tres versiones de un mismo contenido:

  texto       los primeros 256 KiB del código fuente de este repositorio (los .py de 2_codigo,
              en orden alfabético): texto plano, sin datos del corpus;
  comprimido  ese texto comprimido con DEFLATE (zlib, nivel 9), el algoritmo de los ZIP;
  cifrado     ese texto combinado por XOR con una secuencia pseudoaleatoria del mismo largo
              (semilla 0), como hace un cifrado de flujo.

Figura de histogramas de bytes: muestra por qué la entropía reconoce un archivo cifrado pero no lo
distingue de uno comprimido. Imprime la entropía de Shannon (bits por byte) y el estadístico
chi-cuadrado de cada versión, que son las cifras que cita el capítulo.

Figura de la estimación de pi por Monte Carlo (10-11-2026, pedido del tutor): cada par de bytes
consecutivos da un punto (x, y), con cada byte dividido por 255, y el punto es un acierto si cae
dentro del cuarto de círculo de radio 1. La estimación se calcula con la misma función del módulo
de los Experimentos 1 y 2 (family-rw-detection/advanced_features.py), y se dibuja una muestra de
3.000 puntos de cada versión (semilla 0).

No usa archivos de NapierOne ni notas de rescate.

Uso: python generar_figuras_cap2.py [carpeta_de_imagenes]
Salida por defecto: 1_documento/tesis_v4_en_preparacion/images/
"""
import importlib.util
import sys
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

RAIZ = Path(__file__).resolve().parent.parent
IMG = RAIZ / "1_documento" / "tesis_v4_en_preparacion" / "images"

AZUL = "#2c5f8a"
GRIS = "#b8b8b8"
TAM = 256 * 1024
MUESTRA = 3000


def entropia(datos: bytes) -> float:
    cuentas = np.bincount(np.frombuffer(datos, dtype=np.uint8), minlength=256)
    p = cuentas[cuentas > 0] / len(datos)
    return float(-(p * np.log2(p)).sum())


def chi_cuadrado(datos: bytes) -> float:
    cuentas = np.bincount(np.frombuffer(datos, dtype=np.uint8), minlength=256)
    esperado = len(datos) / 256
    return float(((cuentas - esperado) ** 2 / esperado).sum())


def versiones():
    fuentes = sorted((RAIZ / "2_codigo").glob("*.py"))
    texto = b"".join(f.read_bytes() for f in fuentes)[:TAM]
    assert len(texto) == TAM, "no alcanza el código fuente para 256 KiB"
    comprimido = zlib.compress(texto, 9)
    clave = np.random.default_rng(0).integers(0, 256, size=len(texto), dtype=np.uint8).tobytes()
    cifrado = bytes(a ^ b for a, b in zip(texto, clave))
    return [("Texto plano", texto), ("Comprimido (DEFLATE)", comprimido), ("Cifrado", cifrado)]


def coma(v, _=None):
    return f"{v:g}".replace(".", ",")


def figura_histogramas(vers, carpeta):
    fig, ejes = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=False)
    for eje, (nombre, datos) in zip(ejes, vers):
        h, chi = entropia(datos), chi_cuadrado(datos)
        print(f"{nombre:22s} {len(datos):8d} bytes   H = {h:.3f} bits/byte   chi2 = {chi:,.0f}")
        frec = np.bincount(np.frombuffer(datos, dtype=np.uint8), minlength=256) / len(datos)
        eje.bar(np.arange(256), frec, width=1.0, color=AZUL)
        eje.set_title(f"{nombre}\nH = {h:.2f} bits por byte".replace(".", ","), fontsize=10)
        eje.set_xlim(0, 255)
        eje.set_xticks([0, 64, 128, 192, 255])
        eje.set_xlabel("Valor del byte", fontsize=9)
        eje.tick_params(labelsize=8)
        eje.yaxis.set_major_formatter(FuncFormatter(coma))
        eje.spines[["top", "right"]].set_visible(False)
    ejes[0].set_ylabel("Frecuencia relativa", fontsize=9)
    fig.tight_layout()
    salida = carpeta / "fig_mt_histogramas_bytes.png"
    fig.savefig(salida, dpi=300)
    plt.close(fig)
    print("figura:", salida)


def modulo_estadistico():
    """El módulo con el que se calcularon las medidas de los Experimentos 1 y 2."""
    ruta = RAIZ / "2_codigo" / "family-rw-detection" / "advanced_features.py"
    spec = importlib.util.spec_from_file_location("advanced_features", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def figura_pi(vers, carpeta):
    af = modulo_estadistico()
    rng = np.random.default_rng(0)
    arco = np.linspace(0, np.pi / 2, 200)
    fig, ejes = plt.subplots(1, 3, figsize=(10.5, 3.9))
    for eje, (nombre, datos) in zip(ejes, vers):
        est = af.calculate_monte_carlo_pi(datos)
        a = np.frombuffer(datos, dtype=np.uint8).astype(float) / 255.0
        n = len(a) // 2
        x, y = a[0:2 * n:2], a[1:2 * n:2]
        dentro = x * x + y * y <= 1.0
        assert abs(4 * dentro.mean() - est) < 1e-9, "los puntos no coinciden con los del módulo"
        print(f"{nombre:22s} {n:7d} pares   estimación de pi = {est:.4f}   dentro = {dentro.mean():.4f}")
        idx = rng.choice(n, size=min(MUESTRA, n), replace=False)
        fuera_m, dentro_m = idx[~dentro[idx]], idx[dentro[idx]]
        eje.scatter(x[fuera_m], y[fuera_m], s=2, color=GRIS, linewidths=0, label="Fuera del cuarto de círculo")
        eje.scatter(x[dentro_m], y[dentro_m], s=2, color=AZUL, linewidths=0, label="Dentro (acierto)")
        eje.plot(np.cos(arco), np.sin(arco), color="#1a1a1a", linewidth=1)
        eje.set_xlim(0, 1)
        eje.set_ylim(0, 1)
        eje.set_aspect("equal")
        eje.set_title(f"{nombre}\n" + r"$\hat{\pi}$ = " + f"{est:.2f}".replace(".", ","), fontsize=10)
        eje.set_xticks([0, 0.5, 1])
        eje.set_yticks([0, 0.5, 1])
        eje.xaxis.set_major_formatter(FuncFormatter(coma))
        eje.yaxis.set_major_formatter(FuncFormatter(coma))
        eje.set_xlabel("Primer byte del par / 255", fontsize=9)
        eje.tick_params(labelsize=8)
    ejes[0].set_ylabel("Segundo byte del par / 255", fontsize=9)
    manejadores, rotulos = ejes[-1].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(manejadores[::-1], rotulos[::-1], loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2,
               fontsize=9, frameon=False, markerscale=4)
    salida = carpeta / "fig_mt_pi_montecarlo.png"
    fig.savefig(salida, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("figura:", salida)


def main():
    carpeta = Path(sys.argv[1]) if len(sys.argv) > 1 else IMG
    carpeta.mkdir(parents=True, exist_ok=True)
    vers = versiones()
    figura_histogramas(vers, carpeta)
    figura_pi(vers, carpeta)


if __name__ == "__main__":
    main()
