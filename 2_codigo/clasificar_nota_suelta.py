"""¿Qué familia le asigna el modelo a UNA nota que no está en el corpus?

Es el script de inferencia que faltaba: los demás miden el modelo con validación cruzada;
este lo USA sobre un caso nuevo, que es lo que se hace en un incidente real.

QUÉ HACE, en el orden en que se debe leer la salida
  1. Entrena la configuración canónica (TF-IDF combinado + LinearSVC, C=1,
     class_weight=balanced) sobre TODO el corpus. No hay pliegues: la nota de entrada es
     el conjunto de prueba.
  2. Predice familia y calcula el **margen** entre la primera y la segunda clase de
     `decision_function`, que es la misma medida de confianza de M.3.
  3. Ubica ese margen en la curva de abstención medida (M.3, 149 notas, 50 semillas) para
     decir si el sistema, en su punto de operación, CONTESTARÍA o se abstendría.
  4. Muestra las 5 familias mejor puntuadas y los vecinos más cercanos por coseno, que es
     lo que permite ver si la predicción se apoya en un parecido real o es un empate.
  5. Extrae los IOCs de la nota con los patrones canónicos y dice si la regla exacta de
     M.6 aplicaría (es decir, si algún IOC de la nota ya se vio en el corpus).

⚠️ LÍMITE QUE NO SE PUEDE ESQUIVAR: el modelo es de **mundo cerrado**. Está obligado a
contestar una de las 30 familias de NapierOne. Si la nota es de una familia que no está en
esas 30, la respuesta va a ser incorrecta por construcción, y el único indicio disponible
es un margen bajo. Por eso este script SIEMPRE imprime el margen junto a la predicción:
una familia sin su margen no es un resultado, es una adivinanza.

Uso:
    python clasificar_nota_suelta.py <ruta_de_la_nota> [otra ...]
    python clasificar_nota_suelta.py nota.txt --corpus ../3_datos/corpus_v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, TFIDF_CHAR, cargar_corpus,
                                   obtener_modelos, vectorizador)
from extractor_notas import extraer_texto
from grafo_marcadores import extraer_marcadores
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Curva de abstención medida en M.3 (149 notas, P2, 50 semillas):
# resultados_abstencion_149/. umbral -> (cobertura, acierto donde contesta)
CURVA_M3 = [(0.00, 1.0000, 0.6601), (0.10, 0.8244, 0.7796), (0.30, 0.7019, 0.8649),
            (0.50, 0.6459, 0.9000), (1.00, 0.5495, 0.9687), (1.50, 0.4681, 0.9763)]
UMBRAL_OPERACION = 0.50   # el punto de operación reportado: contesta 65 %, acierta 90 %


def tramo_de_la_curva(margen):
    """Dónde cae el margen en la curva de M.3: qué acierto tuvo ese nivel de confianza."""
    ultimo = CURVA_M3[0]
    for u, cob, acierto in CURVA_M3:
        if margen >= u:
            ultimo = (u, cob, acierto)
        else:
            break
    return ultimo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notas", nargs="+", type=Path, help="nota(s) a clasificar")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="corpus de entrenamiento (por defecto, el canónico)")
    ap.add_argument("--top", type=int, default=5, help="cuántas familias mostrar")
    args = ap.parse_args()
    corpus = args.corpus or CORPUS_DIR

    print("=" * 78)
    print("  CLASIFICACIÓN DE UNA NOTA NUEVA (inferencia, no validación)")
    print("=" * 78)
    textos, y, archivos, _ = cargar_corpus(corpus)
    familias = sorted(set(y))
    print(f"Entrenamiento: {corpus}")
    print(f"  {len(textos)} notas · {len(familias)} familias")

    vec = vectorizador("combinado")
    X = vec.fit_transform(textos)
    clf = obtener_modelos(0)["LinearSVC"]
    clf.fit(X, y)

    # vectorizador aparte, solo para el coseno con el corpus (char 3-5, como el agrupador)
    vec_cos = TfidfVectorizer(**TFIDF_CHAR)
    Xcos = vec_cos.fit_transform(textos)

    # IOCs vistos en el corpus -> familia (para saber si la regla de M.6 aplicaría)
    dicc = {}
    for t, fam in zip(textos, y):
        for clave in extraer_marcadores(t):
            dicc.setdefault(clave, set()).add(fam)

    for ruta in args.notas:
        if not ruta.is_file():
            print(f"\n⚠️  no existe: {ruta}")
            continue
        texto, metodo = extraer_texto(ruta)
        print("\n" + "-" * 78)
        print(f"NOTA: {ruta.name}  ({len(texto)} caracteres, extracción: {metodo})")
        print("-" * 78)

        dec = clf.decision_function(vec.transform([texto]))[0]
        orden = np.argsort(-dec)
        clases = np.array(clf.classes_)
        pred = clases[orden[0]]
        margen = float(dec[orden[0]] - dec[orden[1]])

        u, cob, acierto = tramo_de_la_curva(margen)
        contesta = margen >= UMBRAL_OPERACION

        print(f"  PREDICCIÓN: {pred}")
        print(f"  MARGEN (1ª vs 2ª clase): {margen:.4f}")
        print(f"  En el punto de operación de M.3 (umbral {UMBRAL_OPERACION:.2f}): "
              f"{'CONTESTA' if contesta else 'SE ABSTIENE'}")
        print(f"  Notas con margen >= {u:.2f} en la medición de M.3: acierto {acierto:.3f} "
              f"(cobertura {cob:.3f})")

        print(f"\n  Las {args.top} familias mejor puntuadas:")
        for k in orden[:args.top]:
            print(f"     {clases[k]:<16} {dec[k]:+.4f}")

        v = vec_cos.transform([texto])
        sim = (Xcos @ v.T).toarray().ravel()
        top = np.argsort(-sim)[:args.top]
        print(f"\n  Vecinos más cercanos del corpus (coseno char 3-5):")
        for j in top:
            print(f"     {sim[j]:.4f}  {y[j]:<16} {archivos[j]}")
        print(f"  El vecino más cercano está a {sim[top[0]]:.4f}; el umbral de "
              f"casi-duplicado es 0,90.")

        iocs = set(extraer_marcadores(texto))
        vistos = {c: dicc[c] for c in iocs if c in dicc}
        print(f"\n  IOCs en la nota: {len(iocs)}")
        for tipo, valor in sorted(iocs):
            corte = valor if len(valor) <= 46 else valor[:43] + "..."
            print(f"     {tipo:<10} {corte}")
        if vistos:
            fams = set().union(*vistos.values())
            print(f"  La regla exacta de M.6 APLICARÍA: {len(vistos)} IOC(s) ya visto(s), "
                  f"apuntan a {sorted(fams)}")
        else:
            print("  La regla exacta de M.6 NO aplicaría: ningún IOC de esta nota está en "
                  "el corpus. La decisión queda 100 % en manos del texto.")

    print("\n" + "=" * 78)
    print("  RECORDATORIO: mundo cerrado. El modelo elige entre las familias del corpus y")
    print("  NO puede responder «ninguna de estas». Un margen bajo es el único aviso.")
    print("=" * 78)


if __name__ == "__main__":
    main()
