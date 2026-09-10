#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anotar_manifiesto.py -- Deja registrado en el manifiesto todo lo verificado el 2026-08-23.

ES ADITIVO Y NO CAMBIA NINGUNA CIFRA. No borra ni fusiona notas: solo agrega columnas con lo
que la auditoria establecio, para que la informacion viva en el manifiesto y no en un chat.

DECISION DE ROMINA (2026-08-23): las notas duplicadas o reescritas SE QUEDAN mientras tengan
fuente citable. Lo que queda como problema abierto son las que NO tienen fuente. Por eso este
script no propone bajas: documenta.

COLUMNAS NUEVAS
  url_procedencia     URL que publica el texto de ESA nota, verificada abriendola.
  nombre_genuino      nombre de archivo que la fuente declara para ESE texto (o SIN_ARCHIVO).
  origen_nombre       md5_lemmou | fuente_de_la_nota | (vacio)
  reescritura_de      si el texto es reescritura o duplicado de otra nota del corpus, cual.
                      ⚠️ Una reescritura tiene coseno < 0,90 con su original, asi que el
                      agrupador de casi-duplicados NO la detecta y cuenta como plantilla
                      aparte. Bajo P2 eso puede poner el original en entrenamiento y la
                      reescritura en prueba: la tarea queda MAS FACIL de lo que aparenta y el
                      resultado se INFLA. Declararlo al reportar cifras.
  placeholders        marcadores sinteticos presentes ([EMAIL_ADDRESS], [BITCOIN_ADDRESS]...)
                      que reemplazan el marcador real. No los produjo el malware: los puso
                      quien armo el corpus fundacional. Consecuencia medida: varias de estas
                      notas quedan con CERO IOCs extraibles, lo que explica su nula cobertura
                      en M.1/M.6. NO es motivo de baja (es redaccion), pero hay que declararlo.
  redaccion_fuente    `[snip]` = la fuente publica (ThreatLabz) redacto parte del contenido.

Uso:
    python anotar_manifiesto.py            # simula
    python anotar_manifiesto.py --aplicar  # escribe, con respaldo previo
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parent.parent
MANIFIESTO = RAIZ / "3_datos" / "manifiesto_corpus_v2.csv"
CORPUS = RAIZ / "3_datos" / "corpus_v2"
DIRN = RAIZ / "3_datos" / "nombres_notas"
NUEVAS = ["url_procedencia", "nombre_genuino", "origen_nombre", "reescritura_de",
          "placeholders", "redaccion_fuente"]
RE_PH = re.compile(r"\[[A-Z][A-Z0-9_]{3,40}\]")


def cargar():
    """Devuelve dicts (familia, archivo) -> dato, de las tres fuentes de la auditoria."""
    url, nom, orig, reesc = {}, {}, {}, {}
    # nombres por MD5 (auditoria)
    aud = DIRN / "auditoria_nombres_corpus.csv"
    if aud.is_file():
        with open(aud, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f, delimiter=";"):
                k = (r["familia"], r["archivo_corpus"])
                if r["nombre_para_m2"]:
                    nom[k] = r["nombre_para_m2"]
                    orig[k] = ("md5_lemmou" if r["procedencia"].startswith("genuino")
                               and r["procedencia"] != "genuino_de_la_fuente"
                               else "fuente_de_la_nota")
    # nombres recuperados de la fuente de cada nota
    js = DIRN / "nombres_por_nota_2026-08-23.json"
    if js.is_file():
        for r in json.load(open(js, encoding="utf-8")):
            nm = (r.get("nombre_archivo") or "").strip()
            k = (r["familia"], r["archivo_corpus"])
            if r.get("encontrado") and nm:
                nom[k] = nm
                orig[k] = "fuente_de_la_nota"
            if (r.get("url_usada") or "").startswith("http"):
                url[k] = r["url_usada"]
    # procedencia (las dos tandas)
    for f in ("procedencia_2026-08-23.json", "procedencia_lote2_2026-08-23.json"):
        p = DIRN / f
        if not p.is_file():
            continue
        d = json.load(open(p, encoding="utf-8"))
        conf = {(v["familia"], v["archivo"]): v for v in d["veredictos"] if v.get("confirmado")}
        for r in d["resultados"]:
            k = (r["familia"], r["archivo"])
            if r.get("es_parafrasis_de"):
                reesc[k] = r["es_parafrasis_de"]
            if k in conf:
                if (r.get("url") or "").startswith("http"):
                    url.setdefault(k, r["url"].split(";")[0].strip())
                nm = (conf[k].get("nombre_archivo_confirmado") or r.get("nombre_archivo") or "").strip()
                if nm:
                    nom.setdefault(k, nm)
                    orig.setdefault(k, "fuente_de_la_nota")
    return url, nom, orig, reesc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    url, nom, orig, reesc = cargar()
    print(f"Datos de la auditoria: {len(url)} URLs | {len(nom)} nombres | "
          f"{len(reesc)} reescrituras detectadas")

    with open(MANIFIESTO, encoding="utf-8-sig", newline="") as f:
        lector = csv.DictReader(f)
        campos = list(lector.fieldnames)
        filas = list(lector)
    for c in NUEVAS:
        if c not in campos:
            campos.append(c)

    n_url = n_nom = n_re = n_ph = n_snip = 0
    for r in filas:
        k = (r["familia"], r["archivo"])
        ruta = CORPUS / r["familia"] / r["archivo"]
        texto = ruta.read_text(encoding="utf-8", errors="replace") if ruta.is_file() else ""
        phs = sorted(x for x in set(RE_PH.findall(texto)) if x != "[ENTER]")
        r["url_procedencia"] = url.get(k, "")
        r["nombre_genuino"] = nom.get(k, "")
        r["origen_nombre"] = orig.get(k, "")
        r["reescritura_de"] = reesc.get(k, "")
        r["placeholders"] = " ".join(phs)
        r["redaccion_fuente"] = "[snip] x%d" % texto.count("[snip]") if "[snip]" in texto else ""
        n_url += bool(r["url_procedencia"]); n_nom += bool(r["nombre_genuino"])
        n_re += bool(r["reescritura_de"]); n_ph += bool(phs)
        n_snip += bool(r["redaccion_fuente"])

    print(f"\nFilas anotadas sobre {len(filas)}:")
    print(f"   url_procedencia  {n_url}")
    print(f"   nombre_genuino   {n_nom}")
    print(f"   reescritura_de   {n_re}")
    print(f"   placeholders     {n_ph}")
    print(f"   redaccion_fuente {n_snip}")

    if not args.aplicar:
        print("\n[SIMULACION] No se escribio. Correr con --aplicar.")
        return
    resp = MANIFIESTO.with_name(MANIFIESTO.stem + "_respaldo_antes_de_anotar_2026-08-23.csv")
    if resp.exists():
        sys.exit(f"ABORTA: ya existe {resp.name}.")
    shutil.copy2(MANIFIESTO, resp)
    with open(MANIFIESTO, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    print(f"\nRespaldo: {resp.name}")
    print(f"Manifiesto anotado: {len(filas)} filas, {len(campos)} columnas "
          f"(sin altas ni bajas de notas).")


if __name__ == "__main__":
    main()
