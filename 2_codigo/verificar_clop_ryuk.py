#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verificar_clop_ryuk.py -- recalcula las dos cifras del par CLOP-RYUK que la tesis cita
sin respaldo en ninguna corrida guardada.

POR QUE EXISTE. La medicion de cobertura (`cobertura_cifras.py`) encontro que dos cifras del
capitulo 4 no aparecen en ningun archivo de `4_resultados/`:

  · el coseno de 0,8018 entre las notas de CLOP y de RYUK;
  · el 11,3 % del error total del sistema atribuido a esas confusiones.

Las dos salieron de calculos sueltos que nunca quedaron guardados, que es justo lo que este
proyecto se propuso no hacer. Este script las recalcula desde el corpus con las mismas funciones
canonicas del clasificador, imprime lo obtenido junto a lo citado y avisa si no coinciden.

NO IMPRIME NI GUARDA TEXTO DE LAS NOTAS: solo nombres de archivo y numeros.

USO:  python verificar_clop_ryuk.py
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parent.parent
CORPUS = RAIZ / "3_datos" / "corpus_v2"

# --- lo que la tesis cita hoy, para contrastar ---
CITADO_COSENO = 0.8018
CITADO_CONTENCION = 0.811
CITADO_PREFIJO = 423
CITADO_PORCENTAJE_ERROR = 11.3

_spec = importlib.util.spec_from_file_location("clas", Path(__file__).with_name("clasificador_notas_v2.py"))
clas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clas)


def shingles(t: str, k: int = 3) -> set[tuple[str, ...]]:
    """La misma definicion canonica del proyecto (revision_logo.py): 3-shingles de
    palabras sobre el texto en minusculas, quedandose solo con los \\w+."""
    w = re.findall(r"\w+", t.lower())
    if len(w) < k:
        return {tuple(w)} if w else set()
    return {tuple(w[i:i + k]) for i in range(len(w) - k + 1)}


def prefijo_comun(a: str, b: str) -> int:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def subcadena_comun_mas_larga(a: str, b: str) -> int:
    """Longitud de la subcadena comun mas larga, por programacion dinamica sobre dos filas."""
    if not a or not b:
        return 0
    ant = [0] * (len(b) + 1)
    mejor = 0
    for ca in a:
        act = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                act[j] = ant[j - 1] + 1
                if act[j] > mejor:
                    mejor = act[j]
        ant = act
    return mejor


def main() -> int:
    textos, familias, nombres, _ = clas.cargar_corpus(CORPUS)
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(**clas.TFIDF_CHAR).fit_transform(textos)
    sim = (X @ X.T).toarray()

    idx_clop = [i for i, f in enumerate(familias) if f.upper() == "CLOP"]
    idx_ryuk = [i for i, f in enumerate(familias) if f.upper() == "RYUK"]
    print("=" * 88)
    print("  PAR CLOP-RYUK -- recalculo de las cifras que no estaban en ninguna corrida")
    print("=" * 88)
    print(f"  Corpus: {len(textos)} notas · CLOP: {len(idx_clop)} · RYUK: {len(idx_ryuk)}\n")

    # --- 1. el coseno maximo entre una nota de CLOP y una de RYUK ---
    mejor = max(((sim[i, j], i, j) for i in idx_clop for j in idx_ryuk), key=lambda t: t[0])
    cos, i, j = mejor
    print("--- coseno maximo entre las dos familias " + "-" * 44)
    print(f"    {Path(nombres[i]).name}  x  {Path(nombres[j]).name}")
    print(f"    recalculado: {cos:.4f}      citado en la tesis: {CITADO_COSENO:.4f}"
          f"      {'COINCIDE' if abs(cos - CITADO_COSENO) < 5e-4 else '!! NO COINCIDE'}")

    # --- 2. la contencion de la nota de RYUK dentro de la de CLOP ---
    sa, sb = shingles(textos[i]), shingles(textos[j])
    cont = len(sa & sb) / len(sb) if sb else 0.0
    pref = prefijo_comun(textos[i], textos[j])
    print("\n--- contencion de la nota de RYUK dentro de la de CLOP " + "-" * 30)
    print(f"    recalculada: {cont:.4f}      citada: {CITADO_CONTENCION:.4f}"
          f"      {'COINCIDE' if abs(cont - CITADO_CONTENCION) < 5e-3 else '!! NO COINCIDE'}")
    print(f"    prefijo identico: {pref} caracteres   citado: {CITADO_PREFIJO}"
          f"      {'COINCIDE' if pref == CITADO_PREFIJO else '!! NO COINCIDE'}")

    # El «prefijo identico de 423 caracteres» no da con la definicion literal. Se prueban
    # las variantes razonables para ver si alguna lo reproduce, antes de declararlo erroneo.
    a_n = " ".join(textos[i].lower().split())
    b_n = " ".join(textos[j].lower().split())
    variantes = {
        "prefijo literal": prefijo_comun(textos[i], textos[j]),
        "prefijo normalizado (minusculas, espacios colapsados)": prefijo_comun(a_n, b_n),
        "subcadena comun mas larga, literal": subcadena_comun_mas_larga(textos[i], textos[j]),
        "subcadena comun mas larga, normalizada": subcadena_comun_mas_larga(a_n, b_n),
    }
    print("\n    variantes probadas para el «prefijo identico de 423 caracteres»:")
    for nombre, valor in variantes.items():
        marca = "  <-- reproduce el 423" if abs(valor - CITADO_PREFIJO) <= 2 else ""
        print(f"      {valor:>6}   {nombre}{marca}")

    # --- 3. que fraccion del error del sistema son esas confusiones ---
    # Se toman de las salidas ya guardadas del experimento de linaje, que son las que
    # sostienen la Tabla de confusion por linaje del capitulo.
    print("\n--- fraccion del error total atribuible al par " + "-" * 38)
    # Todo sale de corridas guardadas:
    #   _log_p2bal_149.txt          -> exactitud del sistema, 0.8123 sobre 50 semillas
    #   _log_jerarquica_linaje_149  -> decisiones y acierto del binario CLOP-RYUK
    #   tab:confusion_linaje        -> errores de la cascada y confusiones dentro de un par
    n_notas, n_semillas, exactitud = 149, 50, 0.8123
    decisiones = n_notas * n_semillas
    errores_sistema = decisiones * (1 - exactitud)
    dec_par, acierto_par = 393, 0.5878
    err_par_binario = dec_par * (1 - acierto_par)
    err_par_confusion = 116          # tab:confusion_linaje, cascada, par CLOP-RYUK

    candidatos = {
        "errores del binario del par / errores del sistema": err_par_binario / errores_sistema,
        "confusiones CLOP-RYUK de la cascada / errores del sistema": err_par_confusion / errores_sistema,
    }
    print(f"    decisiones del sistema: {decisiones}   errores: {errores_sistema:.0f}"
          f"   (exactitud {exactitud})")
    for nombre, v in candidatos.items():
        pct = 100 * v
        marca = "  <-- reproduce lo citado" if abs(pct - CITADO_PORCENTAJE_ERROR) < 0.05 else ""
        print(f"      {pct:5.1f} %   {nombre}{marca}")
    print(f"\n    citado en la tesis: {CITADO_PORCENTAJE_ERROR} %")
    if not any(abs(100 * v - CITADO_PORCENTAJE_ERROR) < 0.05 for v in candidatos.values()):
        print("    !! NINGUNA definicion razonable reproduce la cifra citada.")
        print("       La mas cercana, y la que corresponde a la frase que la precede (el acierto")
        print(f"       de 0,5878 del binario del par), da {100 * err_par_binario / errores_sistema:.1f} %.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
