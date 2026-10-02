#!/usr/bin/env python3
"""
rigidez_plantillas.py — ¿Cuán rígida es la plantilla de nota de cada familia?

Mide algo que ni B.1 ni B.3 miran, y que hace falta para ordenar la recolección:
**la probabilidad de que una nota nueva de una familia colapse contra las que ya
están en el corpus.**

B.1 dice cuánto vale un texto nuevo (el tramo 1→2 de la curva es el más empinado).
B.3 dice qué explica el desempeño de una familia (la cohesión entre sus plantillas).
Ninguno de los dos dice si la familia PRODUCE textos distintos. Una familia que usa
un molde fijo y solo cambia el correo y el nombre de la víctima no puede aportar
plantillas nuevas por más notas que se junten: es el caso de WASTEDLOCKER, cuyas 4
notas —de 3 víctimas y 2 fuentes distintas— colapsan en una sola plantilla.

Señales que devuelve, por familia:
  - notas y plantillas (componentes de casi-duplicados, criterio canónico 0,90)
  - coseno máximo entre notas de plantillas DISTINTAS: qué tan cerca del umbral
    quedan los textos que la familia sí varía
  - colapsos observados, con el coseno al que ocurrieron: evidencia directa de
    molde rígido cuando notas de fuentes distintas caen en la misma plantilla
  - largo en caracteres: los moldes rígidos son cortos

Criterio idéntico al del resto del frente de notas: agrupar_neardups() de
clasificador_notas_v2.py con UMBRAL_NEARDUP = 0,90 (coseno TF-IDF char_wb 3-5).
Solo lee: no modifica el corpus ni el manifiesto.

Uso:
    python rigidez_plantillas.py                    # todas las familias
    python rigidez_plantillas.py WASTEDLOCKER MAZE  # solo algunas
"""

import collections
import sys

from sklearn.feature_extraction.text import TfidfVectorizer

from clasificador_notas_v2 import (
    CORPUS_DIR,
    TFIDF_CHAR,
    UMBRAL_NEARDUP,
    agrupar_neardups,
    cargar_corpus,
)

# Las 9 familias de la lista de recolección de B.1 (F1 por familia < 0,70 y
# plantillas faltantes para llegar a 4), en el orden del plan operativo.
PRIORIDAD_B1 = ["WASTEDLOCKER", "CHIMERA", "MAZE", "MEDUZALOCKER", "WANNACRY",
                "RYUK", "CRYPTOLOCKER", "JIGSAW", "NOTPETYA"]


def main():
    textos, etiquetas, archivos, _ = cargar_corpus(CORPUS_DIR)
    grupos, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    X = TfidfVectorizer(**TFIDF_CHAR).fit_transform(textos)
    sim = (X @ X.T).toarray()

    if len(sys.argv) > 1:
        familias = sys.argv[1:]
    else:
        familias = PRIORIDAD_B1 + sorted(set(etiquetas) - set(PRIORIDAD_B1))

    print(f"Corpus: {len(textos)} notas -> {len(set(grupos))} plantillas "
          f"(umbral {UMBRAL_NEARDUP})\n")
    print(f"{'familia':<14} {'notas':>5} {'plant':>5} {'cos.max':>8}  chars")

    for fam in familias:
        idx = [i for i in range(len(textos)) if etiquetas[i] == fam]
        if not idx:
            print(f"{fam:<14}   (sin notas en el corpus)")
            continue

        plantillas = collections.defaultdict(list)
        for i in idx:
            plantillas[grupos[i]].append(i)

        # Coseno máximo entre notas de plantillas distintas de la MISMA familia:
        # qué tan cerca del umbral quedan los textos que la familia sí varía.
        entre = [sim[i, j] for i in idx for j in idx
                 if i < j and grupos[i] != grupos[j]]
        cos_max = f"{max(entre):.3f}" if entre else "—"
        largos = [len(textos[i].strip()) for i in idx]

        print(f"{fam:<14} {len(idx):>5} {len(plantillas):>5} {cos_max:>8}  "
              f"{min(largos)}-{max(largos)}")

        for miembros in plantillas.values():
            nombres = [archivos[i].split("\\")[-1].split("/")[-1] for i in miembros]
            if len(miembros) > 1:
                # Coseno mínimo del componente: puede estar por debajo del umbral,
                # porque las componentes conexas unen por cadena.
                minimo = min(sim[i, j] for i in miembros for j in miembros if i < j)
                print(f"      COLAPSAN {len(miembros)} notas (coseno mínimo del "
                      f"componente {minimo:.3f}): {', '.join(nombres)}")
            else:
                print(f"      sola: {nombres[0]}")

    print("\nLectura: pocas plantillas para muchas notas + notas cortas + colapsos "
          "entre fuentes\ndistintas = molde rígido = recolectar no va a aportar "
          "plantillas nuevas.")


if __name__ == "__main__":
    main()
