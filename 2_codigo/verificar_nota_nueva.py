#!/usr/bin/env python3
"""
verificar_nota_nueva.py — ¿Una nota candidata aporta un TEXTO NUEVO al corpus?

Regla medida en B.1: lo que mueve el macro-F1 es el texto distinto, no la nota.
Una nota que repite un contenido ya presente en el corpus no aporta.

Criterio: EXACTAMENTE el mismo con el que se midió todo el frente de notas —
agrupar_neardups() de clasificador_notas_v2.py con UMBRAL_NEARDUP = 0.90
(similitud coseno de TF-IDF char_wb 3-5, componentes conexas). Si al sumar la
candidata el número de textos distintos (grupos) no sube, la nota no aporta.

Uso:
    python verificar_nota_nueva.py <nota_candidata> [otra_candidata ...]

Acepta cualquier formato que entienda extraer_texto() (.txt/.html/.hta/...).
Solo lee y dictamina: NO modifica el corpus ni el manifiesto.
"""

import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from clasificador_notas_v2 import (
    CORPUS_DIR,
    MIN_CHARS_NOTA,
    TFIDF_CHAR,
    UMBRAL_NEARDUP,
    agrupar_neardups,
    cargar_corpus,
)
from extractor_notas import extraer_texto


def main():
    if len(sys.argv) < 2:
        sys.exit("Uso: python verificar_nota_nueva.py <nota_candidata> [otra ...]")

    # ---- Corpus base
    textos, etiquetas, archivos, _ = cargar_corpus(CORPUS_DIR)
    n_corpus = len(textos)
    grupos_base, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    n_grupos_base = len(set(grupos_base))
    print(f"\nCorpus base: {n_corpus} notas -> {n_grupos_base} textos distintos "
          f"(umbral {UMBRAL_NEARDUP})")

    # ---- Candidatas
    candidatas = []  # (ruta, texto)
    for arg in sys.argv[1:]:
        ruta = Path(arg)
        if not ruta.is_file():
            print(f"\n[!] {ruta}: no existe, se omite")
            continue
        texto, metodo = extraer_texto(ruta)
        if metodo.startswith("error") or len(texto.strip()) < MIN_CHARS_NOTA:
            print(f"\n[!] {ruta}: extracción fallida u insuficiente "
                  f"(metodo={metodo}, chars={len(texto.strip())}), se omite")
            continue
        candidatas.append((ruta, texto))

    if not candidatas:
        sys.exit("\nNinguna candidata utilizable.")

    # ---- Agrupamiento conjunto: corpus + candidatas, mismo criterio
    todos = textos + [t for _, t in candidatas]
    grupos, _ = agrupar_neardups(todos, UMBRAL_NEARDUP)

    # Similitud solo para INFORMAR la vecina más cercana (el veredicto sale del
    # agrupamiento de arriba, no de esta matriz).
    X = TfidfVectorizer(**TFIDF_CHAR).fit_transform(todos)
    sim = (X @ X.T).toarray()

    for k, (ruta, texto) in enumerate(candidatas):
        idx = n_corpus + k
        miembros_corpus = [i for i in range(n_corpus) if grupos[i] == grupos[idx]]
        otras_candidatas = [j for j in range(len(candidatas))
                            if j != k and grupos[n_corpus + j] == grupos[idx]]

        vecina = max(range(n_corpus), key=lambda i: sim[idx, i])
        print(f"\n[{k + 1}] {ruta.name}  ({len(texto.strip())} caracteres)")

        if miembros_corpus:
            fams = sorted({etiquetas[i] for i in miembros_corpus})
            print(f"    VEREDICTO: COPIA de plantilla existente — NO aporta. "
                  f"Familias del grupo: {', '.join(fams)}")
            for i in sorted(miembros_corpus, key=lambda i: -sim[idx, i]):
                print(f"      - {archivos[i]} (coseno {sim[idx, i]:.3f})")
        else:
            print(f"    VEREDICTO: TEXTO NUEVO (vecina más cercana: "
                  f"{archivos[vecina]}, coseno {sim[idx, vecina]:.3f})")
            if otras_candidatas:
                print(f"    ADVERTENCIA: casi-duplicada de otra candidata de esta "
                      f"tanda ({', '.join(candidatas[j][0].name for j in otras_candidatas)}) "
                      f"— entre sí cuentan como UN solo texto")

    # Textos nuevos = grupos del agrupamiento conjunto formados SOLO por candidatas
    # (dos candidatas casi-duplicadas entre sí cuentan una sola vez).
    grupos_solo_candidatas = {
        grupos[n_corpus + k] for k in range(len(candidatas))
    } - {grupos[i] for i in range(n_corpus)}
    aporte = len(grupos_solo_candidatas)
    print(f"\nResumen: {len(candidatas)} candidata(s) -> {aporte} texto(s) nuevo(s). "
          f"Textos distintos: {n_grupos_base} -> {n_grupos_base + aporte}.")

    # Deriva del criterio: el TF-IDF se reajusta con las candidatas adentro y el IDF
    # se mueve, así que pares del corpus al borde del umbral pueden cambiar de lado.
    # Eso NO es aporte (ni culpa) de las candidatas: se informa aparte. La cifra
    # oficial de textos distintos se recalcula sobre el corpus una vez incorporadas.
    n_grupos_corpus_conjunto = len({grupos[i] for i in range(n_corpus)})
    if n_grupos_corpus_conjunto != n_grupos_base:
        print(f"AVISO: al reajustar el TF-IDF con las candidatas, los grupos del "
              f"corpus pasan de {n_grupos_base} a {n_grupos_corpus_conjunto} "
              f"(pares al borde del umbral). No afecta los veredictos de arriba.")


if __name__ == "__main__":
    main()
