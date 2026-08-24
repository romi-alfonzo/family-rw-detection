#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
completar_urls_manifiesto.py -- Completa la URL de las transcripciones que la tenian implicita.

QUE ARREGLA. `validar_procedencia.py` detecto 47 notas sin procedencia citable, no 34: a las
34 heredadas («NapierOne/varios») se suman **13 transcripciones de pcrisk cuyo campo `fuente`
dice solo «pcrisk», sin URL**. Esas 13 SI son rastreables —la guia de pcrisk existe— pero el
manifiesto no guardaba el enlace, asi que no se podian citar tal como estaban.

DE DONDE SALEN LAS URLs. Del workflow del 2026-08-23 que recupero el nombre de archivo de cada
nota abriendo su fuente (`3_datos/nombres_notas/nombres_por_nota_2026-08-23.json`, campo
`url_usada`). O sea: son las URLs que un agente **abrio y leyo** para esa nota concreta, no
URLs deducidas del nombre de la familia.

QUE HACE, y nada mas que eso:
  - solo toca filas con tipo `transcripcion` cuyo `fuente` NO tenga ya un http;
  - solo si hay una URL verificada para ESA nota (familia + archivo);
  - APENDE ` | URL verificada 2026-08-23 | <url>` al campo `fuente`. No borra ni reescribe nada.
  - deja respaldo con fecha antes de escribir, y aborta si el respaldo ya existe.

Uso:
    python completar_urls_manifiesto.py --simular   # muestra el cambio, no escribe (default)
    python completar_urls_manifiesto.py --aplicar
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parent.parent
MANIFIESTO = RAIZ / "3_datos" / "manifiesto_corpus_v2.csv"
NOMBRES = RAIZ / "3_datos" / "nombres_notas" / "nombres_por_nota_2026-08-23.json"
SELLO = "URL verificada 2026-08-23"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true", help="escribe el manifiesto")
    args = ap.parse_args()

    urls = {}
    with open(NOMBRES, encoding="utf-8") as f:
        for r in json.load(f):
            u = (r.get("url_usada") or "").strip()
            if u.startswith("http"):
                urls[(r["familia"], r["archivo_corpus"])] = u
    print(f"URLs verificadas disponibles: {len(urls)}")

    with open(MANIFIESTO, encoding="utf-8-sig", newline="") as f:
        lector = csv.DictReader(f)
        campos = lector.fieldnames
        filas = list(lector)

    cambios = []
    for r in filas:
        if (r.get("tipo") or "").strip() != "transcripcion":
            continue
        fuente = (r.get("fuente") or "").strip()
        if "http" in fuente:
            continue
        u = urls.get((r["familia"], r["archivo"]))
        if not u:
            continue
        nuevo = f"{fuente} | {SELLO} | {u}"
        cambios.append((r["familia"], r["archivo"], fuente, nuevo))
        if args.aplicar:
            r["fuente"] = nuevo

    print(f"\nFilas a completar: {len(cambios)}\n")
    for fam, arch, viejo, nuevo in cambios:
        print(f"  {fam}/{arch}")
        print(f"    antes:   {viejo}")
        print(f"    despues: {nuevo}")

    if not args.aplicar:
        print("\n[SIMULACION] No se escribio nada. Correr con --aplicar para guardar.")
        return
    if not cambios:
        print("Nada que hacer.")
        return

    respaldo = MANIFIESTO.with_name(
        MANIFIESTO.stem + "_respaldo_antes_de_urls_2026-08-23.csv")
    if respaldo.exists():
        sys.exit(f"ABORTA: ya existe {respaldo.name}. Revisar antes de sobrescribir.")
    shutil.copy2(MANIFIESTO, respaldo)
    print(f"\nRespaldo: {respaldo.name}")

    with open(MANIFIESTO, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    print(f"Manifiesto actualizado: {len(cambios)} filas completadas, "
          f"{len(filas)} filas en total (sin altas ni bajas).")


if __name__ == "__main__":
    main()
