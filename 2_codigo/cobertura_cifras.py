#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cobertura_cifras.py -- que cifras de la tesis NO estan en la lista del verificador.

POR QUE EXISTE. `cifras_finales.py` responde «las cifras de mi lista aparecen en su corrida».
Eso no dice nada de las cifras que NO estan en la lista: son invisibles para el, y un «87 de 87»
invita a creer que esta todo cubierto. El 83,9 % de P1 estuvo citado en la tesis durante dias
sin haber sido contrastado nunca con su corrida, y el verificador daba 67 de 67 mientras tanto.
Este script mide lo otro: la cobertura.

QUE MIRA. Solo las cifras que aparecen en PROSA, no las celdas de tabla. Una celda es un dato
entre muchos; una cifra en prosa sostiene una afirmacion, que es donde un error hace dano. Las
tablas las cubre el inventario de la revision.

COMO SE USA.
    python cobertura_cifras.py                      # las secciones del frente de notas
    python cobertura_cifras.py archivo.tex ...      # los archivos que se le pasen
    python cobertura_cifras.py --todas              # incluye tambien las celdas de tabla

LO QUE NO HACE. No dice que una cifra descubierta este mal: dice que nadie la verifico. Cada una
hay que ir a buscarla a su corrida a mano, que es como salio el agujero de P1.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parent.parent
DOC = RAIZ / "1_documento" / "Plantilla_de_Tesis___Romina_Carlos"
VERIFICADOR = RAIZ / "2_codigo" / "cifras_finales.py"

# Por defecto, el frente de notas: su archivo propio y su tramo de resultados.tex.
POR_DEFECTO = ["resultados_notas_ampliacion.tex"]

RE_CIFRA = re.compile(r"\d+[,.]\d+")
# cifras que no son una medicion: version de un modelo, rango de n-gramas, fechas...
IGNORAR = {"0,90", "0,95", "3.11", "1.0", "0.90"}


def lineas_de_prosa(texto: str):
    """Devuelve (numero, linea) de lo que no es tabla, caption ni comentario."""
    dentro = False
    for i, linea in enumerate(texto.split("\n"), 1):
        if "\\begin{tabular}" in linea:
            dentro = True
        if "\\end{tabular}" in linea:
            dentro = False
            continue
        if dentro:
            continue
        pelada = linea.strip()
        if pelada.startswith("%") or pelada.startswith("\\caption") or pelada.startswith("\\label"):
            continue
        yield i, linea


def cifras_cubiertas() -> set[str]:
    """Las cifras de la lista de cifras_finales.py, normalizadas con coma decimal."""
    t = VERIFICADOR.read_text(encoding="utf-8")
    ini = t.index("CIFRAS = [")
    fin = t.index("\n]", ini)
    cubiertas = set()
    for m in RE_CIFRA.finditer(t[ini:fin]):
        cubiertas.add(m.group(0).replace(".", ","))
    return cubiertas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archivos", nargs="*", default=None)
    ap.add_argument("--todas", action="store_true",
                    help="incluye las celdas de tabla, no solo la prosa")
    args = ap.parse_args()

    archivos = args.archivos or POR_DEFECTO
    cubiertas = cifras_cubiertas()
    print("=" * 92)
    print("  COBERTURA DEL VERIFICADOR -- cifras en prosa que NADIE contrasto con una corrida")
    print("=" * 92)
    print(f"  La lista de cifras_finales.py tiene {len(cubiertas)} valores distintos.\n")

    total_descubiertas = 0
    for nombre in archivos:
        ruta = Path(nombre)
        if not ruta.is_file():
            ruta = DOC / nombre
        if not ruta.is_file():
            print(f"  !! no encuentro {nombre}")
            continue
        texto = ruta.read_text(encoding="utf-8")
        fuente = ((i, l) for i, l in enumerate(texto.split("\n"), 1)) if args.todas \
            else lineas_de_prosa(texto)
        sin_cubrir: dict[str, list[int]] = {}
        for num, linea in fuente:
            for m in RE_CIFRA.finditer(linea):
                v = m.group(0).replace(".", ",")
                if v in IGNORAR or v in cubiertas:
                    continue
                sin_cubrir.setdefault(v, []).append(num)
        print(f"--- {ruta.name}: {len(sin_cubrir)} cifras distintas sin cubrir " + "-" * 20)
        for v, nums in sorted(sin_cubrir.items()):
            donde = ", ".join(f"l.{n}" for n in nums[:4])
            if len(nums) > 4:
                donde += f" y {len(nums) - 4} mas"
            print(f"    {v:>10}   {donde}")
        total_descubiertas += len(sin_cubrir)
        print()

    print("=" * 92)
    print(f"  TOTAL SIN CUBRIR: {total_descubiertas} cifras distintas.")
    print("  Cada una hay que ir a buscarla a su corrida. Que no este cubierta no significa que")
    print("  este mal: significa que nadie lo comprobo.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
