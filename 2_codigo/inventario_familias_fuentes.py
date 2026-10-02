#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inventario_familias_fuentes.py -- cuantas familias NUEVAS soportarian de verdad el protocolo
del frente de notas, contando plantillas con el criterio del proyecto.

PARA QUE. La pregunta «¿podemos ampliar las familias?» no se contesta con cuantas carpetas
tienen las fuentes. Se contesta con cuantas familias llegan a DOS PLANTILLAS DISTINTAS: una
familia de una sola plantilla nunca aparece en entrenamiento y prueba a la vez bajo un corte
por plantilla, asi que saca F1 = 0 estructural y lo unico que hace es bajar el macro-F1. Es
exactamente lo que le pasa a BADRABBIT y CRYPTOLOCKER en el corpus canonico. Este script
cuenta, fuente por fuente y en conjunto, cuantas familias pasan ese piso.

QUE CUENTA COMO PLANTILLA. El mismo criterio del proyecto y de ningun otro modo: agrupamiento
por casi-duplicado, coseno de caracteres 3-5 > 0,90 (agrupar_neardups, UMBRAL_NEARDUP). El
agrupamiento se hace sobre TODAS las notas juntas, no familia por familia, para que aparezcan
tambien los grupos que cruzan familias: dos notas casi identicas con etiquetas distintas son un
problema de etiqueta, y hay que verlo antes y no despues.

ESTE SCRIPT NO INCORPORA NADA. Solo cuenta y deja la lista. El emparejamiento de nombres
externos con familias canonicas se normaliza aca de forma laxa (minusculas, sin separadores)
SOLO para el inventario; para incorporar una familia hace falta la tabla ALIAS explicita de
candidatas_de_fuentes_nuevas.py, porque la trampa del homonimo en este proyecto ya pego tres
veces (Medusa vs MedusaLocker, Crypt0l0cker vs CryptoLocker, HelloKitty vs Dharma).

LIMITACION QUE HAY QUE DECIR AL CITAR CUALQUIER NUMERO DE ACA. «Plantilla» es coseno de
caracteres 0,90, criterio que NO detecta contencion (una nota contenida literalmente dentro de
otra). La revision del 2026-09-17 midio que, con contencion >= 0,8, el corpus canonico pasa de
99 a 81 plantillas y de 28 a 21 familias evaluables. Un inventario hecho con el coseno es por
lo tanto un TECHO OPTIMISTA: el numero real de familias con dos plantillas de verdad
independientes es menor. Ver 6_notas_trabajo/REVISION_LOGO_2026-09-17_informe.md.

Uso:  python inventario_familias_fuentes.py [--salida CARPETA] [--contencion]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, MIN_CHARS_NOTA, UMBRAL_NEARDUP,
                                   agrupar_neardups, cargar_corpus)
from extractor_notas import extraer_texto

RAIZ = _AQUI.parent
DIR_FUENTES = RAIZ / "3_datos" / "fuentes_notas"
OUT_DEF = RAIZ / "4_resultados" / "inventario_familias_fuentes"
# imagenes_notas no tiene estructura familia/nota: son capturas sueltas, se excluye.
FUENTES = ["ransomware_notes", "RansomNoteFiles", "f6dfir_ransom_notes", "notas_pcrisk"]
IGNORAR = {".git", ".github", "__pycache__"}


def normalizar(nombre):
    """Normalizacion LAXA, solo para el inventario. No sirve para incorporar: ver docstring."""
    return re.sub(r"[^a-z0-9]", "", nombre.lower())


def cargar_fuente(d: Path):
    """Lee familia/nota de una fuente. Devuelve (textos, familias, archivos, omitidas)."""
    textos, fams, archs, omitidas = [], [], [], 0
    for fam_dir in sorted(p for p in d.iterdir() if p.is_dir() and p.name not in IGNORAR):
        for nota in sorted(p for p in fam_dir.rglob("*") if p.is_file()):
            if any(parte in IGNORAR for parte in nota.parts):
                continue
            try:
                texto, metodo = extraer_texto(nota)
            except Exception:
                omitidas += 1
                continue
            if metodo.startswith("error") or len(texto.strip()) < MIN_CHARS_NOTA:
                omitidas += 1
                continue
            textos.append(texto)
            fams.append(fam_dir.name)
            archs.append(str(nota.relative_to(d)))
    return textos, fams, archs, omitidas


def plantillas_por_familia(fams, grupos):
    d = defaultdict(set)
    for f, g in zip(fams, grupos):
        d[f].add(g)
    return {f: len(v) for f, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  INVENTARIO: cuantas familias llegan a 2 plantillas (coseno char 0,90)")
    print("=" * 78)

    # ---------- corpus canonico, para saber que ya esta ----------
    txt_can, y_can, arch_can, _ = cargar_corpus(CORPUS_DIR)
    canonicas = {normalizar(f) for f in set(y_can)}
    print(f"\nCorpus canonico: {len(txt_can)} notas, {len(canonicas)} familias\n")

    todo_txt, todo_fam, todo_src, todo_arch = [], [], [], []
    resumen_fuente = []
    for nombre in FUENTES:
        d = DIR_FUENTES / nombre
        if not d.is_dir():
            print(f"  (no esta: {nombre})")
            continue
        textos, fams, archs, omit = cargar_fuente(d)
        if not textos:
            continue
        grupos, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
        ppf = plantillas_por_familia(fams, grupos)
        nuevas = {f for f in ppf if normalizar(f) not in canonicas}
        resumen_fuente.append(dict(
            fuente=nombre, notas=len(textos), notas_omitidas=omit,
            familias=len(ppf), plantillas=len(set(grupos)),
            familias_2mas=sum(1 for v in ppf.values() if v >= 2),
            familias_3mas=sum(1 for v in ppf.values() if v >= 3),
            familias_nuevas=len(nuevas),
            familias_nuevas_2mas=sum(1 for f, v in ppf.items() if v >= 2 and f in nuevas)))
        todo_txt += textos
        todo_fam += fams
        todo_src += [nombre] * len(textos)
        todo_arch += archs
        print(f"  {nombre}: {len(textos)} notas leidas ({omit} omitidas), {len(ppf)} familias")

    df_src = pd.DataFrame(resumen_fuente)
    df_src.to_csv(OUT / "por_fuente.csv", index=False, encoding="utf-8-sig")
    print("\n=== POR FUENTE (cada una agrupada por separado) ===")
    print(df_src.to_string(index=False))

    # ---------- todas las fuentes juntas + el corpus canonico ----------
    print("\nAgrupando todo junto (fuentes + corpus canonico) ...")
    txt_all = todo_txt + list(txt_can)
    fam_all = [normalizar(f) for f in todo_fam] + [normalizar(f) for f in y_can]
    src_all = todo_src + ["CORPUS_CANONICO"] * len(txt_can)
    arch_all = todo_arch + list(arch_can)
    grupos_all, _ = agrupar_neardups(txt_all, UMBRAL_NEARDUP)
    ppf_all = plantillas_por_familia(fam_all, grupos_all)

    filas = []
    for f, k in sorted(ppf_all.items(), key=lambda kv: (-kv[1], kv[0])):
        idx = [i for i, ff in enumerate(fam_all) if ff == f]
        fuentes_f = sorted({src_all[i] for i in idx})
        filas.append(dict(familia_normalizada=f, n_notas=len(idx), n_plantillas=k,
                          ya_en_corpus="SI" if f in canonicas else "NO",
                          fuentes="|".join(fuentes_f)))
    df_fam = pd.DataFrame(filas)
    df_fam.to_csv(OUT / "familias_unificadas.csv", index=False, encoding="utf-8-sig")

    nuevas = df_fam[df_fam.ya_en_corpus == "NO"]
    print("\n" + "=" * 78)
    print("  LO QUE CONTESTA LA PREGUNTA")
    print("=" * 78)
    print(f"  Familias distintas en total (fuentes + corpus):        {len(df_fam)}")
    print(f"  De ellas, NUEVAS (no estan en las 30 canonicas):       {len(nuevas)}")
    for k in (2, 3, 4):
        print(f"  Familias nuevas con {k} o mas plantillas:               "
              f"{int((nuevas.n_plantillas >= k).sum())}")
    print(f"\n  Total de familias con 2+ plantillas (canonicas + nuevas): "
          f"{int((df_fam.n_plantillas >= 2).sum())}")

    # ---------- grupos que cruzan familias: etiquetas en conflicto ----------
    por_grupo = defaultdict(set)
    for g, f in zip(grupos_all, fam_all):
        por_grupo[g].add(f)
    conflictos = {g: fs for g, fs in por_grupo.items() if len(fs) > 1}
    cfilas = []
    for g, fs in conflictos.items():
        idx = [i for i, gg in enumerate(grupos_all) if gg == g]
        cfilas.append(dict(grupo=int(g), familias="|".join(sorted(fs)), n_notas=len(idx),
                           fuentes="|".join(sorted({src_all[i] for i in idx})),
                           ejemplo=arch_all[idx[0]]))
    df_cf = pd.DataFrame(cfilas).sort_values("n_notas", ascending=False) if cfilas else pd.DataFrame()
    if len(df_cf):
        df_cf.to_csv(OUT / "conflictos_de_etiqueta.csv", index=False, encoding="utf-8-sig")
    print(f"\n  Grupos de casi-duplicados que cruzan DOS O MAS familias: {len(conflictos)}")
    print("  (notas casi identicas con etiquetas distintas: hay que resolverlas a mano antes")
    print("   de incorporar nada; es la trampa del homonimo, ya pego tres veces en el proyecto)")
    if len(df_cf):
        print(df_cf.head(12).to_string(index=False))

    # ---------- cuanto de lo nuevo colapsa con el corpus canonico ----------
    g_can = {g for g, s in zip(grupos_all, src_all) if s == "CORPUS_CANONICO"}
    n_colapsan = sum(1 for g, s in zip(grupos_all, src_all)
                     if s != "CORPUS_CANONICO" and g in g_can)
    n_fuera = len(todo_txt) - n_colapsan
    print(f"\n  De las {len(todo_txt)} notas de fuentes, {n_colapsan} colapsan con una plantilla")
    print(f"  que el corpus canonico YA tiene, y {n_fuera} aportan texto distinto.")

    print("\n  RECORDAR AL CITAR: «plantilla» es coseno char 0,90, criterio que no detecta")
    print("  contencion; este inventario es un TECHO OPTIMISTA (informe del 2026-09-17).")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
