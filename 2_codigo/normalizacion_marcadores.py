#!/usr/bin/env python3
"""
normalizacion_marcadores.py -- Abstracción de los marcadores variables de las notas.

HIPÓTESIS
Bajo el protocolo P2 (variante nunca vista) el clasificador debe atribuir la familia de una
plantilla que no vio. Los datos de contacto concretos ---direcciones de correo, billeteras
Bitcoin, servicios .onion, identificadores de víctima--- cambian entre campañas de una misma
familia, de modo que memorizarlos no ayuda y puede desplazar peso desde señales estables.
Si en cambio se reemplaza cada valor por una etiqueta de su tipo, el modelo puede aprender
el PATRÓN de marcadores, que sí resulta característico.

EVIDENCIA PREVIA (medida sobre corpus_v2 el 2026-08-05)
DHARMA emplea 17 direcciones de correo y ningún servicio .onion; CERBER, 18 .onion y ningún
correo; RYUK combina correo y Bitcoin; PHOBOS solo correo; CONTI solo .onion. El perfil de
tipos de marcador discrimina entre familias aunque los valores concretos no se repitan.

FUNDAMENTO BIBLIOGRÁFICO
Lemmou et al. (2021) aplican una sustitución equivalente en su pre-análisis, reemplazando
por marcadores genéricos la extensión, el identificador de víctima y las cadenas aleatorias.
Trujillo (2022) define variables ##URL## y ##EMAIL## con el mismo propósito.

USO
    python normalizacion_marcadores.py            # compara sin normalizar vs normalizado
    python normalizacion_marcadores.py --ver 5    # además muestra 5 ejemplos de la sustitución
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, N_FOLDS, N_SEMILLAS, UMBRAL_NEARDUP,
                                   agrupar_neardups, cargar_corpus, evaluar)

OUT_DIR = (_AQUI.parent / "4_resultados" / "resultados_normalizacion"
           if (_AQUI.parent / "4_resultados").is_dir()
           else _AQUI / "resultados_normalizacion")

# El orden importa: los patrones más específicos van primero, porque una dirección .onion
# es también una URL y un correo puede aparecer dentro de una URL.
PATRONES = [
    ("[ONION]", re.compile(r"\b[a-z2-7]{16,56}\.onion\b", re.I)),
    ("[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("[BTC]",   re.compile(r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b")),
    ("[URL]",   re.compile(r"https?://[^\s<>\"')\]]+")),
    ("[ID]",    re.compile(r"\b[A-Fa-f0-9]{16,}\b")),
    ("[CLAVE]", re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b")),   # bloques tipo base64
]


def normalizar(texto):
    """Sustituye cada marcador variable por una etiqueta de su tipo.

    Devuelve (texto_normalizado, conteo_por_tipo).
    """
    conteo = Counter()
    for etiqueta, rx in PATRONES:
        texto, n = rx.subn(etiqueta, texto)
        if n:
            conteo[etiqueta] = n
    return texto, conteo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", type=int, default=0,
                    help="mostrar N ejemplos de la sustitución")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)

    log("=" * 72)
    log("  ABSTRACCIÓN DE MARCADORES VARIABLES EN LAS NOTAS")
    log("=" * 72)
    textos, y, archivos, _ = cargar_corpus(Path(CORPUS_DIR))
    familias = np.unique(y)
    log(f"Corpus: {len(textos)} notas | {len(familias)} familias\n")

    normalizados, conteos = [], []
    for t in textos:
        tn, c = normalizar(t)
        normalizados.append(tn)
        conteos.append(c)

    total = Counter()
    for c in conteos:
        total.update(c)
    afectadas = sum(1 for c in conteos if c)
    log(f"Notas modificadas: {afectadas}/{len(textos)} "
        f"({100 * afectadas / len(textos):.0f} %)")
    log(f"Sustituciones por tipo: {dict(total)}\n")

    if args.ver:
        for i in range(min(args.ver, len(textos))):
            if not conteos[i]:
                continue
            log(f"--- {archivos[i]} ---")
            for lin_o, lin_n in zip(textos[i].splitlines(), normalizados[i].splitlines()):
                if lin_o != lin_n:
                    log(f"  antes : {lin_o.strip()[:90]}")
                    log(f"  después: {lin_n.strip()[:90]}")
                    break
        log("")

    # Los grupos de casi-duplicados se calculan sobre el texto ORIGINAL en ambos casos,
    # para que la partición sea idéntica y la comparación aísle el efecto de normalizar.
    grupos, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    grupos = np.array(grupos)
    log(f"Grupos de contenido (calculados sobre el original): {len(set(grupos))}\n")

    filas = []
    for variante, corpus in (("original", textos), ("normalizado", normalizados)):
        for protocolo in ("grupos", "estratificado"):
            for vista in ("palabras", "caracteres", "combinado"):
                resumen, _, _ = evaluar(corpus, y, grupos, vista, "LinearSVC",
                                        protocolo, familias)
                filas.append(dict(variante=variante, protocolo=protocolo, vista=vista,
                                  **resumen))
                log(f"[{variante:<11} | {protocolo:<13} | {vista:<10}] "
                    f"macro-F1 {resumen['f1_macro_mean']:.3f} "
                    f"± {resumen['f1_macro_std']:.3f} | "
                    f"acc {resumen['accuracy_mean']:.3f}")

    df = pd.DataFrame(filas)
    df.to_csv(OUT_DIR / "normalizacion_resumen.csv", index=False)

    log("\n" + "=" * 72)
    log("  COMPARACIÓN (macro-F1, LinearSVC)")
    log("=" * 72)
    log(f"{'protocolo':<15}{'vista':<12}{'original':>10}{'normalizado':>14}{'dif.':>9}")
    mejoras = []
    for protocolo in ("grupos", "estratificado"):
        for vista in ("palabras", "caracteres", "combinado"):
            o = df[(df.variante == "original") & (df.protocolo == protocolo) &
                   (df.vista == vista)].f1_macro_mean.iloc[0]
            nz = df[(df.variante == "normalizado") & (df.protocolo == protocolo) &
                    (df.vista == vista)].f1_macro_mean.iloc[0]
            d = nz - o
            mejoras.append((protocolo, d))
            log(f"{protocolo:<15}{vista:<12}{o:>10.3f}{nz:>14.3f}{d:>+9.3f}")

    d_p2 = np.mean([d for p, d in mejoras if p == "grupos"])
    d_p1 = np.mean([d for p, d in mejoras if p == "estratificado"])
    log(f"\nDiferencia media  P2 (variante nueva): {d_p2:+.3f}")
    log(f"Diferencia media  P1 (plantilla conocida): {d_p1:+.3f}")
    log("\nNota: el desvío entre semillas ronda 0,03-0,06; una diferencia menor a eso NO "
        "es concluyente.")

    (OUT_DIR / "normalizacion_manifiesto.json").write_text(json.dumps(dict(
        fecha=str(date.today()), corpus=str(CORPUS_DIR), n_notas=len(textos),
        n_familias=int(len(familias)), notas_modificadas=afectadas,
        sustituciones=dict(total), n_folds=N_FOLDS, n_semillas=N_SEMILLAS,
        diferencia_media_P2=round(float(d_p2), 4),
        diferencia_media_P1=round(float(d_p1), 4),
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\nSalidas en: {OUT_DIR}")


if __name__ == "__main__":
    main()
