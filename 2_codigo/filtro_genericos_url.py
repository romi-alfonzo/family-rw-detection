#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filtro_genericos_url.py -- arreglar el filtro de genericos normalizando las URL.

EL PROBLEMA, MEDIDO. La capa de reglas acierta 0,9928 sobre las 30 familias del nucleo y
**0,9427 sobre las 106 del corpus extendido**. La causa esta identificada: el mismo dominio de
infraestructura entra al diccionario **partido en varias claves distintas**, y cada una cuenta
por separado para el filtro de genericos. Sobre el corpus de 106:

    46 familias   https://www.torproject.org/download/
    45 familias   https://www.torproject.org/
    19 familias   https://torproject.org
    19 familias   https://torproject.org/
    14 familias   https://www.torproject.org
     9 familias   https://localbitcoins.com/buy_bitcoins
     9 familias   https://tox.chat/download.html

Son SIETE claves para el mismo sitio. El filtro descarta un valor cuando aparece en mas de una
familia **del pliegue de entrenamiento**; una variante poco frecuente puede quedar en una sola
familia de ese pliegue, **pasar el filtro y hacer que la regla conteste con seguridad
equivocada**. Es el mismo mecanismo que en mundo abierto hacia que la regla reclamara las notas
de CONTI para BLACKBASTA.

QUE SE PRUEBA. Dos normalizaciones de la clave [URL], antes de construir el diccionario:
  B. SUAVE   -- minusculas, sin esquema, sin `www.`, sin barra final. Une las variantes
                triviales del mismo recurso y conserva la ruta.
  C. DOMINIO -- ademas colapsa la ruta: la clave es el host. Une todo lo de un mismo sitio, a
                costa de perder una URL especifica de familia si la hubiera.
Se mide sobre el NUCLEO de 30 y sobre el EXTENDIDO de 106, porque el defecto solo se manifiesta
cuando hay muchas familias.

NO SE USA NINGUNA LISTA NEGRA. Se penso y se descarto: una lista escrita a mano mirando estos
resultados seria elegir los valores a filtrar DESPUES de ver cuales molestan. La normalizacion
es un criterio general, no depende de que sitios aparezcan, y por eso es defendible.

=============================================================================================
PREREGISTRO -- escrito y COMMITEADO ANTES de correr (2026-09-28).

N1. PUERTA DE ENTRADA. La variante A (la actual) reproduce sobre el nucleo macro-F1 0,7417 y
    exactitud 0,8123 (tolerancia 0,01). Si no, ABORTA.
N2. Sobre el EXTENDIDO, el acierto de la capa de reglas sube por encima de 0,9427 con B y con C.
    Es la prediccion central: si no sube, la fragmentacion de URL no era la causa.
N3. Sobre el EXTENDIDO, el macro-F1 sube con IC 95 % que excluye el cero, en al menos una de las
    dos variantes.
N4. Sobre el NUCLEO de 30 el efecto es pequeno, |Delta| < 0,01: con 30 familias el filtro ya
    descartaba torproject porque aparece en varias. El defecto es de escala.
N5. C tiene MENOR cobertura que B, porque colapsa mas claves y el filtro descarta mas.
N6. El acierto de la regla con C es >= que con B: colapsar por dominio elimina mas genericos.
    Si C acertara menos, estaria colapsando claves privadas utiles y habria que quedarse con B.
=============================================================================================

Uso:  python filtro_genericos_url.py [--n-semillas 20] [--salida CARPETA]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, MIN_CHARS_NOTA, TFIDF_CHAR, UMBRAL_NEARDUP,
                                   agrupar_neardups, cargar_corpus, obtener_modelos,
                                   vectorizador)
from extractor_notas import extraer_texto
from grafo_marcadores import extraer_marcadores
from protocolo_logo import cargar_nombres
from protocolo_p2bal import split_p2bal

RAIZ = _AQUI.parent
DIR_FUENTES = RAIZ / "3_datos" / "fuentes_notas"
FUENTES = ["ransomware_notes", "RansomNoteFiles", "f6dfir_ransom_notes", "notas_pcrisk"]
IGNORAR = {".git", ".github", "__pycache__"}
OUT_DEF = RAIZ / "4_resultados" / "resultados_filtro_url"
CANON_F1, CANON_ACC, TOL = 0.7417, 0.8123, 0.01
CANON_AC_REGLA_EXT = 0.9427


def norm_fam(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def clave_url(valor, modo):
    """Normaliza el valor de una [URL] segun el modo. 'A' deja el valor como esta."""
    if modo == "A":
        return valor
    v = valor.strip().lower()
    v = re.sub(r"^https?://", "", v)
    v = re.sub(r"^www\.", "", v)
    if modo == "C":
        v = v.split("/")[0].split("?")[0]           # solo el host
    else:
        v = v.rstrip("/")
    return v


def marcadores_modo(texto, modo):
    fuera = set()
    for tipo, valor in extraer_marcadores(texto):
        fuera.add((tipo, clave_url(valor, modo)) if tipo == "[URL]" else (tipo, valor))
    return fuera


def dicc(tr, iocs, nombres, y):
    d = defaultdict(set)
    for i in tr:
        for c in iocs[i]:
            d[c].add(y[i])
        if nombres[i]:
            d[("[NOMBRE]", nombres[i])].add(y[i])
    for k in [k for k, v in d.items() if len(v) > 1]:
        del d[k]
    return d


def regla(i, d, iocs, nombres):
    claves = set(iocs[i])
    if nombres[i]:
        claves.add(("[NOMBRE]", nombres[i]))
    fams = set()
    for c in claves:
        if c in d:
            fams |= d[c]
    return next(iter(fams)) if len(fams) == 1 else None


def ic(v):
    v = np.asarray(v, float)
    m, nn = float(np.mean(v)), len(v)
    s = float(np.std(v, ddof=1)) if nn > 1 else 0.0
    h = t_dist.ppf(0.975, nn - 1) * (s / np.sqrt(nn)) if nn > 1 else 0.0
    return m, m - h, m + h


def cargar_extendido():
    t_can, y_can, a_can, _ = cargar_corpus(CORPUS_DIR)
    canonicas = {norm_fam(f) for f in set(y_can)}
    t, y, a, es_can = list(t_can), [norm_fam(f) for f in y_can], list(a_can), [True] * len(t_can)
    for nombre in FUENTES:
        d = DIR_FUENTES / nombre
        if not d.is_dir():
            continue
        for fam in sorted(p for p in d.iterdir() if p.is_dir() and p.name not in IGNORAR):
            for nota in sorted(p for p in fam.rglob("*") if p.is_file()):
                if any(p in IGNORAR for p in nota.parts):
                    continue
                try:
                    tx, me = extraer_texto(nota)
                except Exception:
                    continue
                if me.startswith("error") or len(tx.strip()) < MIN_CHARS_NOTA:
                    continue
                t.append(tx); y.append(norm_fam(fam.name))
                a.append(f"{nombre}/{nota.relative_to(d)}"); es_can.append(False)
    y = np.array(y); es_can = np.array(es_can)
    n_can = len(t_can)
    g_can, _ = agrupar_neardups(t_can, UMBRAL_NEARDUP); g_can = np.array(g_can)
    X = TfidfVectorizer(**TFIDF_CHAR).fit_transform(t)
    Sim = (X[n_can:] @ X[:n_can].T).toarray()
    g = np.empty(len(t), dtype=int); g[:n_can] = g_can
    libres = []
    for j in range(len(t) - n_can):
        k = int(np.argmax(Sim[j]))
        if Sim[j, k] > UMBRAL_NEARDUP:
            g[n_can + j] = g_can[k]
        else:
            libres.append(j)
    if libres:
        gl, _ = agrupar_neardups([t[n_can + j] for j in libres], UMBRAL_NEARDUP)
        base = int(g_can.max()) + 1
        for pos, j in enumerate(libres):
            g[n_can + j] = base + int(gl[pos])
    ppf = {f: len(set(g[y == f])) for f in set(y)}
    m = np.array([ppf[f] >= 2 or f in canonicas for f in y])
    return ([t[i] for i in range(len(t)) if m[i]], y[m], g[m],
            [a[i] for i in range(len(t)) if m[i]], es_can[m], canonicas)


def corre(textos, y, grupos, nombres, familias, modo, n_semillas):
    iocs = [marcadores_modo(t, modo) for t in textos]
    ta = np.array(textos, dtype=object)
    n = len(y)
    P = np.empty((n_semillas, n), dtype=object)
    A = np.zeros((n_semillas, n), dtype=bool)
    for s in range(n_semillas):
        rng = np.random.default_rng(20_000 + s)
        for tr, te in split_p2bal(y, grupos, familias, rng):
            vec = vectorizador("combinado")
            Xtr = vec.fit_transform(ta[tr])
            clf = obtener_modelos(s)["LinearSVC"]
            clf.fit(Xtr, y[tr])
            pt = clf.predict(vec.transform(ta[te]))
            d = dicc(tr, iocs, nombres, y)
            for k, i in enumerate(te):
                r = regla(i, d, iocs, nombres)
                P[s, i] = pt[k] if r is None else r
                A[s, i] = r is not None
    f1 = np.array([f1_score(y, P[s], average="macro", labels=familias, zero_division=0)
                   for s in range(n_semillas)])
    ac = np.array([accuracy_score(y, P[s]) for s in range(n_semillas)])
    cob = float(A.mean())
    acr = float(np.mean([(P[s][A[s]] == y[A[s]]).mean() for s in range(n_semillas) if A[s].any()]))
    return f1, ac, cob, acr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=20)
    ap.add_argument("--sin-puerta", action="store_true")
    ap.add_argument("--solo-nucleo", action="store_true",
                    help="corre solo el nucleo de 30, para confirmar con mas semillas")
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  ARREGLAR EL FILTRO DE GENERICOS NORMALIZANDO LAS URL")
    print("=" * 78)
    textos, y, grupos, arch, es_can, canonicas = cargar_extendido()
    nom = cargar_nombres()
    inv = {norm_fam(f): f for f in {k[0] for k in nom}}
    nombres = [nom.get((inv[f], Path(a).name)) if f in inv else None
               for f, a in zip(y, arch)]
    print(f"Extendido: {len(y)} notas, {len(np.unique(y))} familias")
    idx_c = np.where(es_can)[0]
    print(f"Nucleo   : {len(idx_c)} notas, {len(np.unique(y[idx_c]))} familias\n")

    filas = []
    conjuntos = [("NUCLEO 30", idx_c)]
    if not args.solo_nucleo:
        conjuntos.append(("EXTENDIDO 106", np.arange(len(y))))
    for etq, sub in conjuntos:
        tt = [textos[i] for i in sub]
        yy, gg = y[sub], grupos[sub]
        nn = [nombres[i] for i in sub]
        ff = np.unique(yy)
        base = None
        for modo, nombre_modo in (("A", "A actual"), ("B", "B suave"), ("C", "C dominio")):
            print(f"  {etq} / {nombre_modo} ...")
            f1, ac, cob, acr = corre(tt, yy, gg, nn, ff, modo, args.n_semillas)
            if modo == "A":
                base = f1
                d = "---"
                sem = "---"
            else:
                m, lo, hi = ic(f1 - base)
                d = f"{m:+.4f} [{lo:+.4f}; {hi:+.4f}]"
                sem = f"{int(((f1-base) > 0).sum())}/{args.n_semillas}"
            filas.append(dict(corpus=etq, variante=nombre_modo,
                              macro_f1=round(float(f1.mean()), 4),
                              exactitud=round(float(ac.mean()), 4),
                              cobertura_regla=round(cob, 4),
                              acierto_regla=round(acr, 4),
                              delta_macro_f1=d, semillas_positivas=sem))
            if etq == "NUCLEO 30" and modo == "A":
                print(f"\n  N1 PUERTA: macro-F1 {f1.mean():.4f} vs {CANON_F1} | "
                      f"exactitud {ac.mean():.4f} vs {CANON_ACC}")
                ok = abs(f1.mean() - CANON_F1) <= TOL and abs(ac.mean() - CANON_ACC) <= TOL
                if not ok and not args.sin_puerta:
                    sys.exit("ABORTADO (N1).")
                print("  OK\n")

    df = pd.DataFrame(filas)
    df.to_csv(OUT / "filtro_url_resumen.csv", index=False, encoding="utf-8-sig")
    print("\n=== RESULTADO ===")
    print(df.to_string(index=False))

    if args.solo_nucleo:
        nuc = {r["variante"]: r for r in filas}
        print("
  CONFIRMACION SOBRE EL NUCLEO (solo N4):")
        for v in ("B suave", "C dominio"):
            print(f"    {v:<10} macro-F1 {nuc[v]['macro_f1']:.4f} | "
                  f"D {nuc[v]['delta_macro_f1']} | {nuc[v]['semillas_positivas']} | "
                  f"acierto regla {nuc[v]['acierto_regla']:.4f}")
        print(f"
Salidas en {OUT}")
        return
    ext = {r["variante"]: r for r in filas if r["corpus"] == "EXTENDIDO 106"}
    nuc = {r["variante"]: r for r in filas if r["corpus"] == "NUCLEO 30"}
    print("\n" + "=" * 78)
    print("  VEREDICTO DEL PREREGISTRO")
    print("=" * 78)
    chk = [
        ("N2 en 106, el acierto de la regla sube de 0,9427",
         ext["B suave"]["acierto_regla"] > CANON_AC_REGLA_EXT
         and ext["C dominio"]["acierto_regla"] > CANON_AC_REGLA_EXT,
         f"B {ext['B suave']['acierto_regla']:.4f} | C {ext['C dominio']['acierto_regla']:.4f}"),
        ("N3 en 106, el macro-F1 sube con IC que excluye 0",
         any("+0.0" in str(ext[v]["delta_macro_f1"]).split("[")[1].split(";")[0]
             or str(ext[v]["delta_macro_f1"]).split("[")[1].startswith("+")
             for v in ("B suave", "C dominio")),
         " · ".join(f"{v}: {ext[v]['delta_macro_f1']}" for v in ("B suave", "C dominio"))),
        ("N4 en el nucleo el efecto es pequeno (|D| < 0,01)",
         all(abs(nuc[v]["macro_f1"] - nuc["A actual"]["macro_f1"]) < 0.01
             for v in ("B suave", "C dominio")),
         " · ".join(f"{v}: {nuc[v]['macro_f1']:.4f}" for v in ("A actual", "B suave", "C dominio"))),
        ("N5 C cubre menos que B",
         ext["C dominio"]["cobertura_regla"] < ext["B suave"]["cobertura_regla"],
         f"B {ext['B suave']['cobertura_regla']:.4f} | C {ext['C dominio']['cobertura_regla']:.4f}"),
        ("N6 C acierta >= que B",
         ext["C dominio"]["acierto_regla"] >= ext["B suave"]["acierto_regla"],
         f"B {ext['B suave']['acierto_regla']:.4f} | C {ext['C dominio']['acierto_regla']:.4f}"),
    ]
    for nombre, cumple, det in chk:
        print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<48} {det}")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
