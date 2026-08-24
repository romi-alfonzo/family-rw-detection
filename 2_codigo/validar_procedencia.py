#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validar_procedencia.py -- Control de que NINGUNA nota nueva entre al corpus sin procedencia.

POR QUE EXISTE. El 24 % del corpus (34 notas) figura con la etiqueta «NapierOne/varios», que
no es una fuente: es un marcador de «no se de donde salio». Y NapierOne es un dataset de
ARCHIVOS, no un corpus de notas, asi que esa etiqueta nunca fue procedencia. El problema esta
detectado desde `DIAGNOSTICO_2026-07-27.md` (punto A4) y quedo como item B.2 del plan.

Nada impedia que volviera a pasar: el manifiesto es un CSV que se edita a mano. Este script es
ese impedimento. Corre sobre el manifiesto y **falla (exit 1)** si aparece una nota sin
procedencia que no este en la deuda declarada.

LA IDEA CLAVE ES LA LINEA BASE. Las 34 notas heredadas son una **deuda conocida**: se listan
en `3_datos/deuda_procedencia.json` y NO hacen fallar el control. Lo que hace fallar es:
  - una nota NUEVA sin fuente citable (lo que este script previene);
  - que la deuda CREZCA;
  - una transcripcion sin URL;
  - manifiesto y disco que no coinciden 1:1 (el problema de las «filas fantasma» de agosto).
Y si la deuda BAJA, el script lo celebra y avisa que hay que regrabar la linea base.

CRITERIO DE «FUENTE CITABLE», por tipo:
  bruto           tiene que nombrar un repositorio conocido (ThreatLabz, lemmou/Lemmou).
                  Su procedencia es el archivo mismo y se puede verificar por MD5.
  transcripcion   tiene que traer una URL (http/https). Es lo minimo para citar.
  corpus-existente  no es un tipo valido para notas nuevas: es la etiqueta del lote
                  fundacional. Cualquier nota nueva con este tipo es un error.

Uso:
    python validar_procedencia.py                # controla; exit 1 si hay violaciones nuevas
    python validar_procedencia.py --grabar-linea-base   # regraba la deuda (solo si bajo)

Pensado para correr a mano antes de tocar cifras, o como hook de pre-commit.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parent.parent
MANIFIESTO = RAIZ / "3_datos" / "manifiesto_corpus_v2.csv"
CORPUS = RAIZ / "3_datos" / "corpus_v2"
DEUDA = RAIZ / "3_datos" / "deuda_procedencia.json"

# Etiquetas que NO son una fuente: son marcadores de procedencia desconocida.
GENERICAS = {"napierone/varios", "napierone", "varios", "vario", "?", "-", "n/a", "na",
             "desconocido", "desconocida", "sin fuente", ""}
REPOS_VALIDOS = ("threatlabz", "lemmou")
RE_URL = re.compile(r"https?://", re.I)


def clasificar_fila(r):
    """Devuelve (ok, motivo). ok=False significa que la fila no tiene procedencia citable."""
    tipo = (r.get("tipo") or "").strip().lower()
    fuente = (r.get("fuente") or "").strip()
    primero = fuente.split("|")[0].strip().lower()

    if primero in GENERICAS:
        return False, f"fuente generica o vacia: «{fuente[:40]}»"
    if tipo == "bruto":
        if any(x in fuente.lower() for x in REPOS_VALIDOS):
            return True, ""
        if RE_URL.search(fuente):
            return True, ""
        return False, f"tipo bruto sin repositorio conocido ni URL: «{fuente[:40]}»"
    if tipo == "transcripcion":
        if RE_URL.search(fuente):
            return True, ""
        return False, "transcripcion SIN URL (no se puede citar)"
    if tipo == "corpus-existente":
        if RE_URL.search(fuente):
            return True, ""   # ya se le encontro procedencia: sale de la deuda
        return False, "tipo corpus-existente sin URL (lote fundacional, deuda B.2)"
    return False, f"tipo desconocido: «{tipo}»"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grabar-linea-base", action="store_true",
                    help="regraba 3_datos/deuda_procedencia.json con la deuda actual")
    args = ap.parse_args()

    if not MANIFIESTO.is_file():
        sys.exit(f"ABORTA: no encuentro el manifiesto en {MANIFIESTO}")

    filas = []
    with open(MANIFIESTO, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            filas.append(r)
    print(f"Manifiesto: {len(filas)} filas ({MANIFIESTO.relative_to(RAIZ)})")

    # ---- 1. procedencia por fila
    sin_proc, motivos = [], {}
    por_tipo = Counter()
    for r in filas:
        clave = f"{r['familia']}/{r['archivo']}"
        por_tipo[(r.get("tipo") or "?").strip()] += 1
        ok, motivo = clasificar_fila(r)
        if not ok:
            sin_proc.append(clave)
            motivos[clave] = motivo
    print("Por tipo: " + " | ".join(f"{k} {v}" for k, v in por_tipo.most_common()))

    # ---- 2. linea base de deuda conocida
    base = set()
    if DEUDA.is_file():
        base = set(json.load(open(DEUDA, encoding="utf-8"))["deuda"])
    nuevas = sorted(set(sin_proc) - base)
    saldadas = sorted(base - set(sin_proc))

    # ---- 3. manifiesto vs disco, 1:1
    en_disco = {f"{p.parent.name}/{p.name}"
                for p in CORPUS.rglob("*") if p.is_file()} if CORPUS.is_dir() else set()
    en_man = {f"{r['familia']}/{r['archivo']}" for r in filas}
    fantasma = sorted(en_man - en_disco)     # fila sin archivo
    huerfano = sorted(en_disco - en_man)     # archivo sin fila

    print(f"\nSin procedencia citable: {len(sin_proc)}  "
          f"(deuda declarada {len(base)}, nuevas {len(nuevas)}, saldadas {len(saldadas)})")
    print(f"Manifiesto vs disco: {len(en_man)} filas / {len(en_disco)} archivos "
          f"| filas sin archivo {len(fantasma)} | archivos sin fila {len(huerfano)}")

    if args.grabar_linea_base:
        if base and len(sin_proc) > len(base):
            sys.exit(f"NO se graba: la deuda CRECIO de {len(base)} a {len(sin_proc)}. "
                     f"Resolver las nuevas primero.")
        DEUDA.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"comentario": "Deuda de procedencia B.2: lote fundacional del corpus, "
                                 "etiquetado «NapierOne/varios». Detectado en "
                                 "DIAGNOSTICO_2026-07-27.md punto A4. Solo puede BAJAR.",
                   "n": len(sin_proc), "deuda": sorted(sin_proc)},
                  open(DEUDA, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\nLinea base grabada: {len(sin_proc)} notas en {DEUDA.relative_to(RAIZ)}")
        return

    fallos = []
    if nuevas:
        fallos.append(f"{len(nuevas)} nota(s) NUEVA(S) sin procedencia citable")
        print("\n⛔ NOTAS NUEVAS SIN PROCEDENCIA (esto es lo que el control previene):")
        for k in nuevas:
            print(f"   {k}  -> {motivos[k]}")
    if fantasma:
        fallos.append(f"{len(fantasma)} fila(s) del manifiesto sin archivo en disco")
        print("\n⛔ FILAS SIN ARCHIVO (las «filas fantasma»):")
        for k in fantasma[:10]:
            print(f"   {k}")
    if huerfano:
        fallos.append(f"{len(huerfano)} archivo(s) en disco sin fila en el manifiesto")
        print("\n⛔ ARCHIVOS SIN FILA EN EL MANIFIESTO:")
        for k in huerfano[:10]:
            print(f"   {k}")

    if saldadas:
        print(f"\n✅ {len(saldadas)} nota(s) de la deuda YA tienen procedencia. "
              f"Regrabar la linea base con --grabar-linea-base:")
        for k in saldadas[:15]:
            print(f"   {k}")

    if not base:
        print("\n[!] No hay linea base todavia. Correr una vez con --grabar-linea-base "
              "para congelar la deuda heredada; desde ahi el control detecta lo nuevo.")

    if fallos:
        print("\n" + "=" * 70)
        sys.exit("FALLA EL CONTROL DE PROCEDENCIA:\n  - " + "\n  - ".join(fallos))
    print("\n✅ Control OK: ninguna nota nueva sin procedencia, y manifiesto y disco "
          "coinciden 1:1.")
    if base:
        print(f"   (queda la deuda heredada de {len(base)} notas, item B.2 del plan)")


if __name__ == "__main__":
    main()
