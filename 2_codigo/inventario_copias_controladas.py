#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inventario_copias_controladas.py -- cuántas copias reales de las plantillas del corpus hay en las
fuentes locales, para ampliar el catálogo de notas de forma controlada.

POR QUÉ. Romina (2026-10-01): «ampliá el corpus de forma controlada para mejorar los datos» y
«probá con duplicados controlados». Bajo P1, 76 de las 149 notas son el único ejemplar de su
plantilla, y mejoras_cascada_p1.py mostró que la técnica ya está en su techo: lo que puede subir
«nota conocida» son datos. Una COPIA CONTROLADA es otra nota real de la MISMA plantilla (otra
víctima u otra campaña de la misma familia: cambian el identificador o el contacto), tomada de
una fuente con procedencia.

QUÉ HACE. Solo lee: no copia ni mueve nada. Recorre las fuentes locales y clasifica cada archivo:
  EN_CORPUS        texto idéntico (espacios normalizados) a una nota del corpus: es el mismo documento.
  REPETIDA         idéntica a otra candidata anterior: se cuenta una sola vez.
  COPIA            coseno char 3-5 > 0,90 con al menos una nota del corpus, y todas las notas del
                   corpus por encima de 0,90 son de la familia de la carpeta y de una sola plantilla.
  COPIA_AMBIGUA    supera 0,90 con notas de otra familia o de más de una plantilla.
  TEXTO_NUEVO      ninguna nota del corpus supera 0,90: sería una plantilla nueva, no una copia.
  FUERA_DE_LAS_30  la carpeta no corresponde a ninguna de las 30 familias.
  ILEGIBLE         el extractor falla o deja menos de 10 caracteres.
Para cada COPIA se informa además si sus marcadores (correos, BTC, .onion, etc., con
extraer_marcadores) coinciden con los de su vecina más parecida: si coinciden y el texto es casi
igual, puede ser el MISMO documento que el corpus ya tiene con alguna redacción, y hay que mirarla
a mano antes de usarla.

FUENTES. Las que el corpus ya cita: Lemmou (RansomNoteFiles), ThreatLabz (ransomware_notes),
PCrisk (notas_pcrisk) y la recolección de agosto. Las que no cita (f6dfir, MISP, zsigovits) se
inventarían aparte, marcadas «fuente_nueva»: la decisión de Romina del 01-10 es no usar otra fuente.
El vectorizador es TFIDF_CHAR, el mismo que define «plantilla» (agrupar_neardups), ajustado sobre
corpus + candidatas.

SALIDA: 4_resultados/resultados_copias_controladas/inventario.csv y un resumen por pantalla.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clasificador_notas_v2 import (CORPUS_DIR, MIN_CHARS_NOTA, TFIDF_CHAR,  # noqa: E402
                                   UMBRAL_NEARDUP, agrupar_neardups, cargar_corpus)
from extractor_notas import extraer_texto  # noqa: E402
from grafo_marcadores import extraer_marcadores  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "3_datos"
SALIDA = Path(os.environ.get("SALIDA_COPIAS", RAIZ / "4_resultados" / "resultados_copias_controladas"))  # 2026-10-02: configurable para correr otros grupos sin pisar este

# (carpeta raíz, nombre de la fuente, ¿el corpus ya la cita?)
FUENTES = [
    (DATOS / "fuentes_notas" / "RansomNoteFiles", "lemmou", True),
    (DATOS / "fuentes_notas" / "ransomware_notes", "ThreatLabz", True),
    (DATOS / "fuentes_notas" / "notas_pcrisk", "pcrisk", True),
    (DATOS / "recoleccion_2026-08", "recoleccion_2026-08", True),
    (DATOS / "fuentes_notas" / "f6dfir_ransom_notes", "f6dfir", False),
    (DATOS / "candidatas_fuentes_nuevas", "candidatas_fuentes_nuevas", False),
]
# 2026-10-02: fuentes adicionales por variable de entorno, «ruta::nombre» separadas por «;».
for _f in filter(None, os.environ.get("FUENTES_EXTRA", "").split(";")):
    _ruta, _nombre = _f.split("::")
    FUENTES.append((Path(_ruta), _nombre, False))
NO_TEXTO = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".pdf", ".csv", ".json", ".md",
            ".zip", ".7z", ".exe", ".dll", ".bin"}

# Alias seguros. Petya NO es NotPetya, Medusa NO es MedusaLocker y Crypt0l0cker es TorrentLocker:
# por eso no hay prefijos sueltos para esos nombres.
ALIAS = {"alphv": "BLACKCAT", "revil": "SODINOKIBI", "cl0p": "CLOP", "medusalocker": "MEDUZALOCKER",
         "wanacry": "WANNACRY", "wannacrypt": "WANNACRY", "wcry": "WANNACRY"}


def normalizar(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def familia_de_carpeta(nombre: str, familias: list[str]) -> str | None:
    k = normalizar(nombre)
    if k in ALIAS:
        return ALIAS[k]
    for f in familias:
        if k == f.lower():
            return f
    # prefijo: «lockbit3», «LockBit 3.0», «blackbasta_2» -> la familia, si es la única que calza
    calzan = [f for f in familias if k.startswith(f.lower())]
    for a, f in ALIAS.items():
        if k.startswith(a):
            calzan.append(f)
    calzan = sorted(set(calzan))
    return calzan[0] if len(calzan) == 1 else None


def espacios(t: str) -> str:
    return " ".join(t.split())


def main():
    textos, y, archivos, _ = cargar_corpus(CORPUS_DIR)
    y = np.asarray(y)
    familias = sorted(set(y))
    grupos, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    grupos = np.asarray(grupos)
    tam_grupo = Counter(grupos)
    print(f"Corpus: {len(textos)} notas, {len(set(grupos))} plantillas, {len(familias)} familias")
    print(f"Notas que son el único ejemplar de su plantilla: {sum(tam_grupo[g] == 1 for g in grupos)}")

    corpus_norm = {espacios(t): i for i, t in enumerate(textos)}
    filas, cand_textos = [], []
    vistas = {}
    for raiz, fuente, citada in FUENTES:
        if not raiz.is_dir():
            print(f"  (no existe {raiz})")
            continue
        for p in sorted(raiz.rglob("*")):
            if not p.is_file() or ".git" in p.parts or p.suffix.lower() in NO_TEXTO:
                continue
            rel = p.relative_to(raiz)
            if len(rel.parts) < 2:
                continue  # archivos sueltos en la raíz de la fuente (README, procedencia)
            carpeta = rel.parts[0]
            fam = familia_de_carpeta(carpeta, familias)
            fila = {"fuente": fuente, "fuente_citada": "si" if citada else "no",
                    "ruta": str(p.relative_to(DATOS)), "carpeta": carpeta, "familia": fam or "",
                    "estado": "", "chars": 0, "coseno_max": "", "vecina": "", "familia_vecina": "",
                    "n_corpus_sobre_090": 0, "plantilla_vecina_tam": "", "marcadores_iguales": ""}
            try:
                texto, metodo = extraer_texto(p)
            except Exception as e:  # noqa: BLE001
                texto, metodo = "", f"error {e}"
            fila["chars"] = len(texto.strip())
            if metodo.startswith("error") or len(texto.strip()) < MIN_CHARS_NOTA:
                fila["estado"] = "ILEGIBLE"
            elif fam is None:
                fila["estado"] = "FUERA_DE_LAS_30"
            elif espacios(texto) in corpus_norm:
                fila["estado"] = "EN_CORPUS"
                fila["vecina"] = archivos[corpus_norm[espacios(texto)]]
            elif espacios(texto) in vistas:
                fila["estado"] = "REPETIDA"
                fila["vecina"] = vistas[espacios(texto)]
            else:
                vistas[espacios(texto)] = fila["ruta"]
                fila["_texto"] = texto
                cand_textos.append(texto)
            filas.append(fila)

    # coseno de cada candidata contra el corpus, con el vectorizador que define «plantilla»
    pend = [f for f in filas if "_texto" in f]
    if pend:
        vec = TfidfVectorizer(**TFIDF_CHAR).fit(list(textos) + cand_textos)
        Xc = vec.transform(textos)
        Xk = vec.transform([f["_texto"] for f in pend])
        sim = (Xk @ Xc.T).toarray()
        for k, f in enumerate(pend):
            j = int(sim[k].argmax())
            sobre = np.where(sim[k] > UMBRAL_NEARDUP)[0]
            f["coseno_max"] = f"{sim[k, j]:.4f}"
            f["vecina"] = archivos[j]
            f["familia_vecina"] = y[j]
            f["n_corpus_sobre_090"] = len(sobre)
            f["plantilla_vecina_tam"] = tam_grupo[grupos[j]]
            if len(sobre) == 0:
                f["estado"] = "TEXTO_NUEVO"
            elif set(y[sobre]) == {f["familia"]} and len(set(grupos[sobre])) == 1:
                f["estado"] = "COPIA"
                f["marcadores_iguales"] = ("si" if set(extraer_marcadores(f["_texto"]))
                                           == set(extraer_marcadores(textos[j])) else "no")
            else:
                f["estado"] = "COPIA_AMBIGUA"

    SALIDA.mkdir(parents=True, exist_ok=True)
    campos = [k for k in filas[0] if not k.startswith("_")] if filas else []
    with open(SALIDA / "inventario.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)

    print("\nEstado por fuente:")
    tabla = defaultdict(Counter)
    for f in filas:
        tabla[(f["fuente"], f["fuente_citada"])][f["estado"]] += 1
    estados = ["EN_CORPUS", "REPETIDA", "COPIA", "COPIA_AMBIGUA", "TEXTO_NUEVO", "FUERA_DE_LAS_30",
               "ILEGIBLE"]
    print(f"  {'fuente':28s} {'citada':6s} " + " ".join(f"{e[:13]:>13s}" for e in estados))
    for (fu, ci), c in tabla.items():
        print(f"  {fu:28s} {ci:6s} " + " ".join(f"{c[e]:>13d}" for e in estados))

    for citada in ("si", "no"):
        copias = [f for f in filas if f["estado"] == "COPIA" and f["fuente_citada"] == citada]
        print(f"\nCOPIAS de fuentes {'ya citadas' if citada == 'si' else 'NUEVAS'}: {len(copias)}")
        for f in sorted(copias, key=lambda f: (f["familia"], f["vecina"])):
            print(f"  {f['familia']:13s} cos {f['coseno_max']}  plantilla de {f['plantilla_vecina_tam']}"
                  f"  marcadores iguales: {f['marcadores_iguales']:2s}  {f['ruta']}  ~ {f['vecina']}")
        unicas = {f["vecina"] for f in copias if f["plantilla_vecina_tam"] == 1}
        print(f"  notas únicas del corpus que ganarían una hermana: {len(unicas)}")

    ambiguas = [f for f in filas if f["estado"] == "COPIA_AMBIGUA"]
    print(f"\nCOPIAS AMBIGUAS (no se usan): {len(ambiguas)}")
    for f in ambiguas:
        print(f"  {f['familia']:13s} cos {f['coseno_max']} vecina {f['familia_vecina']}  {f['ruta']}")
    nuevas = Counter((f["familia"], f["fuente_citada"]) for f in filas if f["estado"] == "TEXTO_NUEVO")
    print("\nTEXTOS NUEVOS por familia (plantillas que el corpus no tiene; no son copias):")
    for (fam, ci), c in sorted(nuevas.items()):
        print(f"  {fam:13s} fuente citada: {ci}  {c}")
    fuera = Counter(f["carpeta"] for f in filas if f["estado"] == "FUERA_DE_LAS_30")
    print(f"\nCarpetas fuera de las 30 (revisar que ninguna sea un alias): {len(fuera)}")
    print("  " + ", ".join(sorted(fuera)))
    print(f"\nSalida: {SALIDA / 'inventario.csv'}")


if __name__ == "__main__":
    main()
