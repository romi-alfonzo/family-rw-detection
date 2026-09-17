"""¿Reconoce la nota si se entrena con MÁS familias que las 30 del núcleo canónico?

PARA QUÉ: cuando llega una nota de un incidente real, lo más probable es que su familia NO
esté entre las 30 de NapierOne. Este script amplía el espacio de etiquetas con los repos de
notas que están en disco (Lemmou: 68 familias; ThreatLabz: más), entrena la misma
configuración canónica y clasifica la nota. Contesta la pregunta operativa: **con más
familias, ¿la reconoce?**

⚠️ ESTO NO ES EL EXPERIMENTO «Ext.» PREREGISTRADO. Ext. está diseñado como validación fuera
de muestra de la cohesión, con método congelado y decisión de Cappo pendiente
(`EXPERIMENTOS_PENDIENTES.md`). Acá no hay validación cruzada ni métrica reportable: es
**inferencia sobre un caso**, para saber si el modelo puede nombrar la familia de una nota
concreta. Ninguna cifra de acá va a la tesis como resultado del método.

⚠️ Y LA PROCEDENCIA DE LAS ETIQUETAS NO ESTÁ AUDITADA. Las familias salen del nombre de la
carpeta del repo. Para el corpus canónico esas etiquetas se verificaron una por una (y ahí
aparecieron los errores de HELLOKITTY, MEDUZALOCKER y CRYPTOLOCKER); acá se toman como
vienen. Sirve para orientar, no para afirmar.

Uso:
    python reconocer_con_expansion.py <nota> --fuente ../3_datos/fuentes_notas/RansomNoteFiles
    python reconocer_con_expansion.py <nota> --fuente A --fuente B --con-corpus
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, MIN_CHARS_NOTA, TFIDF_CHAR, UMBRAL_NEARDUP,
                                   agrupar_neardups, cargar_corpus, obtener_modelos,
                                   vectorizador)
from extractor_notas import extraer_texto
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def cargar_repo(raiz: Path):
    """Cada subcarpeta de primer nivel es una familia; se busca recursivamente dentro."""
    textos, y, rutas, saltados = [], [], [], Counter()
    for fam_dir in sorted(p for p in raiz.iterdir() if p.is_dir()):
        familia = fam_dir.name.upper().replace(" ", "")
        for f in sorted(fam_dir.rglob("*")):
            if not f.is_file():
                continue
            try:
                t, _m = extraer_texto(f)
            except Exception:
                saltados["error de extraccion"] += 1
                continue
            if not t or len(t) < MIN_CHARS_NOTA:
                saltados["texto corto o vacio"] += 1
                continue
            textos.append(t)
            y.append(familia)
            rutas.append(str(f.relative_to(raiz)))
    return textos, y, rutas, saltados


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nota", type=Path)
    ap.add_argument("--fuente", type=Path, action="append", required=True,
                    help="repo de notas con estructura familia/…  (se puede repetir)")
    ap.add_argument("--con-corpus", action="store_true",
                    help="sumar además las 149 notas del corpus canónico")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    print("=" * 80)
    print("  RECONOCIMIENTO CON ESPACIO DE ETIQUETAS AMPLIADO (exploratorio)")
    print("=" * 80)

    textos, y, rutas = [], [], []
    for fuente in args.fuente:
        if not fuente.is_dir():
            sys.exit(f"ABORTA: no existe {fuente}")
        t, yy, rr, salt = cargar_repo(fuente)
        print(f"  {fuente.name}: {len(t)} notas · {len(set(yy))} familias"
              + (f"  (saltados: {dict(salt)})" if salt else ""))
        textos += t
        y += yy
        rutas += [f"{fuente.name}/{r}" for r in rr]
    if args.con_corpus:
        t, yy, aa, _ = cargar_corpus(CORPUS_DIR)
        print(f"  corpus canónico: {len(t)} notas · {len(set(yy))} familias")
        textos += list(t)
        y += list(yy)
        rutas += [f"corpus/{a}" for a in aa]

    y = np.array(y)
    grupos, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    fams = sorted(set(y))
    print(f"\n  TOTAL: {len(textos)} notas · {len(fams)} familias · "
          f"{len(set(grupos))} plantillas distintas")
    # familias con una sola plantilla: no aportan separabilidad, pero sí ocupan una clase
    plant = {f: len({g for g, ff in zip(grupos, y) if ff == f}) for f in fams}
    print(f"  familias con 1 sola plantilla: {sum(1 for v in plant.values() if v == 1)}"
          f" de {len(fams)}")

    vec = vectorizador("combinado")
    X = vec.fit_transform(textos)
    clf = obtener_modelos(0)["LinearSVC"]
    clf.fit(X, y)

    texto, metodo = extraer_texto(args.nota)
    print("\n" + "-" * 80)
    print(f"  NOTA: {args.nota.name} ({len(texto)} caracteres, {metodo})")
    print("-" * 80)

    dec = clf.decision_function(vec.transform([texto]))[0]
    orden = np.argsort(-dec)
    clases = np.array(clf.classes_)
    margen = float(dec[orden[0]] - dec[orden[1]])
    positivas = int((dec > 0).sum())

    print(f"  PREDICCIÓN: {clases[orden[0]]}")
    print(f"  margen 1ª vs 2ª: {margen:.4f}")
    print(f"  clases con decision_function POSITIVA: {positivas} de {len(dec)}"
          + ("   <-- ninguna la reclama: el modelo no la tiene" if positivas == 0 else ""))
    print(f"  puntaje máximo: {dec.max():+.4f}")

    print(f"\n  Las {args.top} familias mejor puntuadas:")
    for k in orden[:args.top]:
        print(f"     {clases[k]:<22} {dec[k]:+.4f}")

    vec_cos = TfidfVectorizer(**TFIDF_CHAR)
    Xc = vec_cos.fit_transform(textos)
    sim = (Xc @ vec_cos.transform([texto]).T).toarray().ravel()
    top = np.argsort(-sim)[:args.top]
    print(f"\n  Notas más parecidas de TODAS las fuentes (coseno char 3-5):")
    for j in top:
        print(f"     {sim[j]:.4f}  {y[j]:<22} {rutas[j][:52]}")
    print(f"\n  Máximo coseno: {sim[top[0]]:.4f}. El umbral de casi-duplicado es 0,90:")
    if sim[top[0]] > UMBRAL_NEARDUP:
        print("     => HAY una nota casi idéntica: la familia de esa nota es la respuesta.")
    elif sim[top[0]] > 0.70:
        print("     => parecido alto pero por debajo del umbral: mismo estilo, no la misma nota.")
    else:
        print("     => NO hay ninguna nota parecida en ninguna fuente. Sin atribución.")


if __name__ == "__main__":
    main()
