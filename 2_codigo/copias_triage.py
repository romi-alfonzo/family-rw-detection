#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
copias_triage.py -- recolecta notas de rescate de informes PÚBLICOS de tria.ge (Recorded Future
Triage) para ampliar el catálogo con otras víctimas de las plantillas del corpus.

POR QUÉ. Romina (2026-10-02, «dale nomás» a buscar datos en fuentes nuevas). objetivos_p1.py mostró
que 12 notas ÚNICAS se fallan bajo P1 en más de la mitad de las semillas, y que con una hermana de
su plantilla en entrenamiento la cascada las acierta casi siempre (P1cat). Las fuentes que el corpus
ya cita no tienen más copias (inventario_copias_controladas.py). Los informes públicos de tria.ge
muestran, para varias familias, la nota que dejó cada muestra ejecutada, con su identificador de
víctima, sin iniciar sesión.

QUÉ HACE. Solo lee páginas públicas, sin iniciar sesión y a ritmo lento (una cada 2 segundos).
Para cada familia de la lista toma los informes de la búsqueda pública «family:<etiqueta>» y,
opcionalmente, los identificadores de informe de un archivo de texto. De cada informe extrae los
bloques «Ransom Note» y guarda el texto en
    3_datos/fuentes_notas/triage_2026-10/<FAMILIA>/<informe>__<archivo>.txt
(3_datos no se versiona y tiene la exclusión de Defender puesta por Romina). El manifiesto
manifiesto_triage.csv registra informe, URL, ruta original de la nota en la muestra, sha256 y
largo. Textos repetidos dentro de la recolección se guardan una sola vez.

QUÉ NO HACE. No decide si una nota es copia: eso lo decide inventario_copias_controladas.py con su
regla, sin cambios (coseno > 0,90 contra UNA plantilla de la misma familia, texto no idéntico,
marcadores distintos). No baja muestras ni archivos: solo el HTML del informe. La etiqueta de familia
es la de tria.ge, y se usa solo si la plantilla del corpus más parecida es de esa misma familia
(lo exige la regla del inventario).

Uso:  python copias_triage.py LOCKBIT=lockbit CLOP=clop ... [--ids archivo.txt] [--max 40]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import re
import sys
import time
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "3_datos" / "fuentes_notas" / "triage_2026-10"
BASE = "https://tria.ge"
AGENTE = "Mozilla/5.0 (investigacion academica, tesis de grado FP-UNA, lectura de informes publicos)"
PAUSA = 2.0
# Etiquetas de familia de tria.ge aceptadas para cada familia del corpus. La etiqueta del informe es
# la fuente INDEPENDIENTE del rótulo: un informe sin la etiqueta esperada se descarta aunque su nota
# se parezca a una del corpus (si no, el rótulo saldría de la propia similitud, y sería circular).
# «medusaransomware» es Medusa, OTRA familia: no es MedusaLocker.
ETIQUETAS = {"LOCKBIT": {"lockbit"}, "RYUK": {"ryuk"}, "HELLOKITTY": {"hellokitty"},
             # ransomexx_win: agregada tras la 1.a recolección (es la etiqueta de la variante Windows)
             "RANSOMEXX": {"ransomexx", "ransomexx_win", "defray777"},
             "MEDUZALOCKER": {"medusalocker"},
             "CLOP": {"clop", "cl0p"}}
assert all(ETIQUETAS[f] for f in ("LOCKBIT", "RYUK", "HELLOKITTY", "RANSOMEXX", "MEDUZALOCKER"))


def leer(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": AGENTE})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read().decode("utf-8", "replace")


def ids_de_busqueda(etiqueta: str) -> list[str]:
    pag = leer(f"{BASE}/s?q=family%3A{etiqueta}")
    return list(dict.fromkeys(re.findall(r'href="/(\d{6}-[a-z0-9]{10})"', pag)))


def notas_de_informe(pag: str) -> list[tuple[str, str]]:
    """(ruta de la nota en la muestra, texto) de cada bloque «Ransom Note» del informe."""
    salida = []
    for bloque in re.split(r'(?=<div class="config-entry-heading">\s*Path)', pag):
        if "Ransom Note" not in bloque:
            continue
        ruta = re.search(r'Path\s*</div>.*?<li[^>]*>(.*?)</li>', bloque, re.S)
        nota = re.search(r'Ransom Note\s*</div>.*?<li class="prewrap[^"]*">(.*?)</li>', bloque, re.S)
        if nota:
            limpiar = lambda s: html.unescape(re.sub(r"<[^>]+>", "", s)).strip()  # noqa: E731
            salida.append((limpiar(ruta.group(1)) if ruta else "", limpiar(nota.group(1))))
    return salida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("familias", nargs="*", help="FAMILIA=etiqueta_de_tria.ge (búsqueda pública)")
    ap.add_argument("--ids", type=Path, help="archivo con líneas «FAMILIA informe»")
    ap.add_argument("--max", type=int, default=40, help="informes por familia")
    args = ap.parse_args()

    pedidos: dict[str, list[str]] = {}
    for par in args.familias:
        fam, etiqueta = par.split("=")
        ids = ids_de_busqueda(etiqueta)
        time.sleep(PAUSA)
        pedidos[fam] = ids[: args.max]
        print(f"{fam}: {len(ids)} informes en la búsqueda pública, se leen {len(pedidos[fam])}")
    if args.ids:
        for linea in args.ids.read_text(encoding="utf-8").splitlines():
            if linea.strip() and not linea.startswith("#"):
                fam, inf = linea.split()[:2]
                inf = inf.split("/")[0]  # siempre la vista general del informe
                pedidos.setdefault(fam, [])
                if inf not in pedidos[fam]:
                    pedidos[fam].append(inf)

    DESTINO.mkdir(parents=True, exist_ok=True)
    man = DESTINO / "manifiesto_triage.csv"
    vistos = set()
    filas, descartes = [], []
    for fam, ids in pedidos.items():
        (DESTINO / fam).mkdir(exist_ok=True)
        # se borran las notas que ESTE recolector guardó antes para la familia (son salidas suyas y se
        # regeneran): así no quedan restos de corridas viejas que el inventario leería igual
        for viejo in (DESTINO / fam).glob("*__*.txt"):
            viejo.unlink()
        guardadas = 0
        for inf in ids:
            url = f"{BASE}/{inf}"
            try:
                pag = leer(url)
            except Exception as e:  # noqa: BLE001
                print(f"  {fam} {inf}: no se pudo leer ({e})")
                continue
            finally:
                time.sleep(PAUSA)
            etiquetas = set(re.findall(r"/s/family:([a-z0-9_.-]+)", pag))
            if not etiquetas & ETIQUETAS.get(fam, {fam.lower()}):
                print(f"  {fam} {inf}: descartado, etiquetas de tria.ge {sorted(etiquetas) or '(ninguna)'}")
                descartes.append(dict(familia=fam, informe=inf, etiquetas=" ".join(sorted(etiquetas))))
                continue
            for ruta, texto in notas_de_informe(pag):
                clave = " ".join(texto.split())
                if len(clave) < 80 or clave in vistos:
                    continue
                vistos.add(clave)
                base = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(ruta.replace("\\", "/")).name or "nota")[:60]
                sha = hashlib.sha256(texto.encode("utf-8")).hexdigest()
                # la huella va en el nombre: un mismo informe puede traer dos notas DISTINTAS con el
                # mismo nombre de archivo, y sin ella la segunda pisaba a la primera (corrida del 02-10)
                destino = DESTINO / fam / f"{inf}__{sha[:10]}__{base}.txt"
                assert not destino.exists(), f"choque de nombres: {destino}"
                destino.write_text(texto, encoding="utf-8")
                filas.append(dict(familia=fam, informe=inf, url=url, ruta_en_la_muestra=ruta,
                                  archivo=str(destino.relative_to(RAIZ / "3_datos")),
                                  sha256=sha, chars=len(texto), leido=time.strftime("%Y-%m-%d")))
                guardadas += 1
        print(f"  {fam}: {guardadas} notas distintas guardadas")
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]) if filas else ["familia"])
        w.writeheader()
        w.writerows(filas)
    with open(DESTINO / "descartados_por_etiqueta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["familia", "informe", "etiquetas"])
        w.writeheader()
        w.writerows(descartes)
    print(f"\nTotal: {len(filas)} notas distintas de {sum(len(v) for v in pedidos.values()) - len(descartes)} "
          f"informes con la etiqueta esperada | {len(descartes)} informes descartados por etiqueta")
    print(f"Manifiesto: {man}")


if __name__ == "__main__":
    main()
