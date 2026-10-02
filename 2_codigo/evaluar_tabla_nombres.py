#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluar_tabla_nombres.py -- ¿Sirve un dataset de NOMBRES POR FAMILIA? Medicion directa.

LA PREGUNTA. La tabla `3_datos/nombres_notas/tabla_nombres_notas.csv` (30/30 familias, 137
filas con fuente) NO se puede usar como feature por nota: asignarle a cada nota el nombre
documentado de SU familia convierte el nombre en una funcion de la etiqueta, o sea
circularidad perfecta. Pero SI se puede usar de la otra forma, que es como funciona ID
Ransomware de verdad: **como diccionario EXTERNO de consulta**.

POR QUE ESTA EVALUACION ES LIMPIA (y mas limpia que la de M.6):
  - El diccionario nombre->familia sale de LITERATURA EXTERNA (advisories, CERT, id-ransomware,
    pcrisk, MISP). No se construye con el corpus.
  - Los nombres de prueba salen de los ARTEFACTOS auditados: 64 notas cuyo nombre esta
    verificado por MD5 contra el repo de Lemmou o declarado por la fuente de esa nota.
  - Ninguna de las dos puntas usa la etiqueta de la nota evaluada. **No hace falta train/test
    split**: no hay nada que se pueda filtrar. Por eso mide el techo real de la senal.

SE REPORTA con las tres columnas del Exp. 2b:
    cobertura (¿el nombre de la nota esta en la tabla?)
  x acierto donde aplica (¿la tabla acierta la familia?)
  x y el desglose de por que falla cuando falla.

DOS FORMAS DE EMPAREJAR, se reportan las dos:
  exacto   el nombre de la nota es igual (minusculas) a un nombre de la tabla.
  patron   ademas se aceptan los nombres de la tabla que son PLANTILLAS con parte variable
           (README.[victim's_ID].TXT, RECOVER-XXX-FILES.txt, <encrypted_filename>_info,
           {ID}-readme.txt, _HELP_DECRYPT_[A-Z0-9]{4-8}_.hta). Se convierten a expresion
           regular reemplazando el hueco por .+  Es mas generoso y mas realista, pero
           SOBRE-EMPAREJA, y esta medido: el UNICO error del modo patron es
           `Recovery_README.html` (MEDUZALOCKER) capturado por `RECOVER<5_chars>.html` de
           TESLACRYPT, porque .+ se come «y_readme». El modo exacto no tiene ese problema y
           por eso es la cifra limpia; el modo patron se reporta como cota optimista.

Uso:  python evaluar_tabla_nombres.py [--salida CARPETA]
Solo lee. No modifica corpus, manifiesto ni resultados.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR = RAIZ / "3_datos" / "nombres_notas"
TABLA = DIR / "tabla_nombres_notas.csv"
AUDIT = DIR / "auditoria_nombres_corpus.csv"

# Huecos de plantilla que aparecen en la tabla, en orden de especificidad.
HUECOS = [
    r"\[a-z0-9\]\{4-8\}", r"\[A-Z0-9\]\{4-8\}",
    r"<[^>]{1,60}>", r"\[[^\]]{1,60}\]", r"\{[^}]{1,60}\}",
    r"%random%", r"\bXXX\b", r"\(seven-digit extension\)",
    r"<random \d+ chars>", r"<\d+_chars>",
]


def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def a_regex(nombre: str):
    """Convierte un nombre de tabla con huecos en regex. None si no tiene huecos."""
    n = norm(nombre)
    # Los ejemplos del tipo "X , e.g. Y" se cortan en la coma del e.g.
    n = re.split(r",\s*e\.?g\.?", n)[0].strip()
    tiene = False
    partes, resto = [], n
    while resto:
        mejor = None
        for h in HUECOS:
            m = re.search(h, resto, re.I)
            if m and (mejor is None or m.start() < mejor.start()):
                mejor = m
        if not mejor:
            partes.append(re.escape(resto))
            break
        partes.append(re.escape(resto[:mejor.start()]))
        partes.append(".+")
        resto = resto[mejor.end():]
        tiene = True
    if not tiene:
        return None
    try:
        return re.compile("^" + "".join(partes) + "$", re.I)
    except re.error:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=DIR)
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)

    # ---- diccionario EXTERNO: nombre -> familias
    exactos = defaultdict(set)
    patrones = []          # (regex, familia, nombre_original)
    filas_tabla = 0
    with open(TABLA, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            fam, nom = r["familia"].strip(), r["nombre_archivo"].strip()
            if not nom or nom.upper() == "SIN_ARCHIVO":
                continue
            filas_tabla += 1
            rx = a_regex(nom)
            if rx is None:
                exactos[norm(nom)].add(fam)
            else:
                patrones.append((rx, fam, nom))
    print(f"Tabla externa: {filas_tabla} filas utiles -> "
          f"{len(exactos)} nombres literales + {len(patrones)} plantillas")

    # ---- nombres de prueba: los 64 auditados del corpus
    pruebas = []
    with open(AUDIT, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if r["nombre_para_m2"]:
                pruebas.append((r["familia"], r["archivo_corpus"],
                                norm(r["nombre_para_m2"]), r["procedencia"]))
    print(f"Nombres de prueba (auditados del corpus): {len(pruebas)} "
          f"en {len({p[0] for p in pruebas})} familias\n")

    filas, resumen = [], {}
    for modo in ("exacto", "exacto+patron"):
        est = Counter()
        detalle = []
        for fam_real, arch, nom, proc in pruebas:
            cands = set(exactos.get(nom, set()))
            via = "literal" if cands else ""
            if modo == "exacto+patron":
                for rx, fam_p, nom_p in patrones:
                    if rx.match(nom):
                        cands.add(fam_p)
                        via = via or "plantilla"
            if not cands:
                est["sin_cobertura"] += 1
                res = "sin_cobertura"
            elif len(cands) > 1:
                if fam_real in cands:
                    est["ambiguo_contiene_la_correcta"] += 1
                    res = "ambiguo_contiene_la_correcta"
                else:
                    est["ambiguo_sin_la_correcta"] += 1
                    res = "ambiguo_sin_la_correcta"
            elif next(iter(cands)) == fam_real:
                est["acierto"] += 1
                res = "acierto"
            else:
                est["error"] += 1
                res = "error"
            detalle.append(dict(modo=modo, familia_real=fam_real, archivo=arch,
                                nombre=nom, procedencia=proc, resultado=res,
                                via=via, familias_propuestas="|".join(sorted(cands))))
        filas += detalle
        n = len(pruebas)
        decide = est["acierto"] + est["error"]
        cobertura = (n - est["sin_cobertura"]) / n
        acierto_unico = est["acierto"] / decide if decide else float("nan")
        resumen[modo] = dict(
            modo=modo, n=n, cobertura=round(cobertura, 4),
            decision_unica=decide, acierto_donde_decide=round(acierto_unico, 4),
            acierto=est["acierto"], error=est["error"],
            ambiguo_con_la_correcta=est["ambiguo_contiene_la_correcta"],
            ambiguo_sin_la_correcta=est["ambiguo_sin_la_correcta"],
            sin_cobertura=est["sin_cobertura"])

    df_res = list(resumen.values())
    with open(args.salida / "eval_tabla_resumen.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(df_res[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(df_res)
    with open(args.salida / "eval_tabla_detalle.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(filas)

    print("=== LA TABLA COMO DICCIONARIO EXTERNO (sin train/test: no hay fuga posible) ===")
    for r in df_res:
        print(f"\n  modo {r['modo']}")
        print(f"    cobertura                    {r['cobertura']:.4f}  "
              f"({r['n'] - r['sin_cobertura']} de {r['n']} nombres estan en la tabla)")
        print(f"    acierto donde decide         {r['acierto_donde_decide']:.4f}  "
              f"({r['acierto']} aciertos / {r['decision_unica']} decisiones unicas)")
        print(f"    errores                      {r['error']}")
        print(f"    ambiguos CON la correcta     {r['ambiguo_con_la_correcta']}")
        print(f"    ambiguos SIN la correcta     {r['ambiguo_sin_la_correcta']}")
        print(f"    sin cobertura                {r['sin_cobertura']}")

    print("\n=== ERRORES Y AMBIGUEDADES (modo exacto+patron) ===")
    for d in filas:
        if d["modo"] == "exacto+patron" and d["resultado"] in (
                "error", "ambiguo_sin_la_correcta", "ambiguo_contiene_la_correcta"):
            print(f"  {d['familia_real']:<13} {d['nombre'][:38]:<38} "
                  f"{d['resultado']:<28} -> {d['familias_propuestas']}")

    print("\n=== COBERTURA POR FAMILIA (modo exacto+patron) ===")
    porfam = defaultdict(lambda: [0, 0])
    for d in filas:
        if d["modo"] != "exacto+patron":
            continue
        porfam[d["familia_real"]][0] += 1
        if d["resultado"] == "acierto":
            porfam[d["familia_real"]][1] += 1
    for fam in sorted(porfam):
        t, ok = porfam[fam]
        print(f"  {fam:<14} {ok}/{t} aciertos")
    print(f"\nSalidas en {args.salida}")


if __name__ == "__main__":
    main()
