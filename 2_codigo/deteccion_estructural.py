#!/usr/bin/env python3
"""
deteccion_estructural.py — Experimento 2b: clasificación de familias por
ARTEFACTOS ESTRUCTURALES de los archivos cifrados (magic bytes / metadatos).

Motivación (pedido del tutor 11/01/25 + hallazgo de Pruebas.xlsx): las propiedades
estadísticas del cifrado no discriminan familias (~9,9 %), pero varias familias
insertan marcas deliberadas en el archivo cifrado (ej. WannaCry: b"WANACRY!" en los
primeros 8 bytes). ID Ransomware identifica 9/30 familias por estas firmas incluso
renombrando el archivo. Este experimento las descubre y evalúa automáticamente.

Dos análisis:
  1. DESCUBRIMIENTO: por familia, prefijo y sufijo binario común a todos sus
     archivos (longest common prefix/suffix de los primeros/últimos 64 bytes)
     y extensión añadida común. Familias con marca >= MIN_MARCA bytes se
     consideran estructuralmente identificables.
  2. CLASIFICACIÓN leave-one-out: para cada archivo, las marcas se aprenden con
     los archivos RESTANTES de cada familia (sin ver el archivo evaluado) y se
     predice la familia cuya marca (más larga) coincida. Reporta exactitud
     multiclase y cobertura — comparable con el 9,9 % del enfoque estadístico.

Estructura esperada del dataset (NapierOne tiny en el servidor):
    RAIZ/<FAMILIA>-tiny/*   o   RAIZ/<FAMILIA>/*
Uso:
    python deteccion_estructural.py RUTA_RAIZ [--max-archivos 50]
Salidas en .\resultados_estructural\:
    marcas_por_familia.csv, clasificacion_loo.csv, manifiesto.json

Nota metodológica: NapierOne genera los archivos cifrados ejecutando las muestras
reales de ransomware, por lo que las extensiones y marcas son artefactos auténticos
del atacante (verificar y citar Davies et al. 2022) — a diferencia de los nombres
de archivo de notas curados por repositorios, aquí el nombre/extensión SÍ es evidencia.
"""

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

N_BYTES = 64        # ventana de cabecera y cola a analizar
MIN_MARCA = 4       # bytes mínimos de prefijo/sufijo común para considerarlo marca
_AQUI = Path(__file__).resolve().parent
OUT_DIR = (_AQUI.parent / "4_resultados" / "resultados_estructural"
           if (_AQUI.parent / "4_resultados").is_dir()
           else _AQUI / "resultados_estructural")


def leer_ventanas(path):
    """Devuelve (primeros N_BYTES, últimos N_BYTES) del archivo."""
    with open(path, "rb") as f:
        head = f.read(N_BYTES)
        f.seek(0, 2)
        n = f.tell()
        f.seek(max(0, n - N_BYTES))
        tail = f.read(N_BYTES)
    return head, tail


def prefijo_comun(bloques):
    """Longest common prefix de una lista de bytes."""
    if not bloques:
        return b""
    p = bloques[0]
    for b in bloques[1:]:
        i = 0
        m = min(len(p), len(b))
        while i < m and p[i] == b[i]:
            i += 1
        p = p[:i]
        if not p:
            break
    return p


def sufijo_comun(bloques):
    return prefijo_comun([b[::-1] for b in bloques])[::-1]


def extension_comun(nombres):
    """Extensión final común (la que añade el ransomware), si existe."""
    exts = [Path(n).suffix.lower() for n in nombres]
    c = Counter(exts).most_common(1)[0]
    return c[0] if c[1] == len(exts) and c[0] else ""


def cargar_dataset(raiz, max_archivos):
    """{familia: [(nombre, head, tail), ...]}"""
    datos = {}
    raiz = Path(raiz)
    for d in sorted(p for p in raiz.iterdir() if p.is_dir()):
        familia = d.name.upper()
        for suf in ("-TINY", "_TINY", "-SMALL", "_SMALL"):
            familia = familia.removesuffix(suf)
        archivos = sorted(p for p in d.iterdir() if p.is_file())[:max_archivos]
        if len(archivos) < 3:
            print(f"  ADVERTENCIA: {familia} tiene {len(archivos)} archivos (<3), omitida.")
            continue
        datos[familia] = [(p.name, *leer_ventanas(p)) for p in archivos]
    return datos


def marcas(entradas):
    """(prefijo, sufijo, extensión) comunes de una lista de (nombre, head, tail)."""
    return (prefijo_comun([h for _, h, _ in entradas]),
            sufijo_comun([t for _, _, t in entradas]),
            extension_comun([n for n, _, _ in entradas]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz", help="carpeta con subcarpetas <FAMILIA>[-tiny]")
    ap.add_argument("--max-archivos", type=int, default=50)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    print(f"Dataset: {args.raiz}")
    datos = cargar_dataset(args.raiz, args.max_archivos)
    print(f"Familias: {len(datos)} | Archivos: {sum(len(v) for v in datos.values())}")

    # ---- 1. Descubrimiento de marcas por familia (con todos los archivos) ----
    filas = []
    for familia, entradas in sorted(datos.items()):
        pre, suf, ext = marcas(entradas)
        filas.append(dict(
            familia=familia, n_archivos=len(entradas),
            prefijo_len=len(pre), prefijo_hex=pre.hex(),
            sufijo_len=len(suf), sufijo_hex=suf.hex(),
            extension=ext,
            marca_detectable=(len(pre) >= MIN_MARCA or len(suf) >= MIN_MARCA
                              or bool(ext)),
        ))
    df_marcas = pd.DataFrame(filas)
    df_marcas.to_csv(OUT_DIR / "marcas_por_familia.csv", index=False)
    detectables = df_marcas[df_marcas.marca_detectable]
    print(f"\nFamilias con marca estructural (prefijo/sufijo >= {MIN_MARCA} bytes "
          f"o extensión propia): {len(detectables)}/{len(df_marcas)}")
    for _, r in detectables.iterrows():
        partes = []
        if r.prefijo_len >= MIN_MARCA:
            partes.append(f"prefijo {r.prefijo_len}B {r.prefijo_hex[:24]}")
        if r.sufijo_len >= MIN_MARCA:
            partes.append(f"sufijo {r.sufijo_len}B ...{r.sufijo_hex[-24:]}")
        if r.extension:
            partes.append(f"ext {r.extension}")
        print(f"  {r.familia:<15} {' | '.join(partes)}")

    # ---- 2. Clasificación leave-one-out, con ABLACIÓN por tipo de marca ----
    # Un revisor preguntará cuánto del rendimiento aporta la extensión del archivo
    # (que ya explotan las herramientas por reglas) frente a las firmas binarias
    # insertadas en el contenido. Se evalúan los tres modos por separado.
    MODOS = ("solo_extension", "solo_firmas_binarias", "combinado")
    resultados = {}
    detalle_por_familia = {}

    for modo in MODOS:
        aciertos, sin_marca, total = 0, 0, 0
        por_familia, ok_familia = Counter(), Counter()
        for familia, entradas in datos.items():
            for i, (nombre, head, tail) in enumerate(entradas):
                total += 1
                por_familia[familia] += 1
                mejor, mejor_score = None, 0
                for fam2, entradas2 in datos.items():
                    resto = ([e for j, e in enumerate(entradas2) if j != i]
                             if fam2 == familia else entradas2)
                    pre, suf, ext = marcas(resto)
                    score = 0
                    if modo in ("solo_firmas_binarias", "combinado"):
                        if len(pre) >= MIN_MARCA and head.startswith(pre):
                            score += len(pre)
                        if len(suf) >= MIN_MARCA and tail.endswith(suf):
                            score += len(suf)
                    if modo in ("solo_extension", "combinado"):
                        if ext and Path(nombre).suffix.lower() == ext:
                            score += 2  # pesa menos que una firma binaria
                    if score > mejor_score:
                        mejor, mejor_score = fam2, score
                if mejor is None:
                    sin_marca += 1
                elif mejor == familia:
                    aciertos += 1
                    ok_familia[familia] += 1
        acc = aciertos / total if total else 0
        cobertura = 1 - (sin_marca / total) if total else 0
        resultados[modo] = dict(exactitud=round(acc, 4), aciertos=aciertos,
                                total=total, sin_marca=sin_marca,
                                cobertura=round(cobertura, 4))
        detalle_por_familia[modo] = (por_familia, ok_familia)
        print(f"  [{modo:<21}] exactitud {acc:.3f} ({aciertos}/{total}) | "
              f"sin marca {sin_marca} | cobertura {cobertura:.3f}")

    # CSV por familia con las tres variantes
    filas = []
    for f in sorted(datos):
        fila = dict(familia=f, n=detalle_por_familia["combinado"][0][f])
        for m in MODOS:
            pf, okf = detalle_por_familia[m]
            fila[f"recall_{m}"] = round(okf.get(f, 0) / pf[f], 3) if pf[f] else 0
        filas.append(fila)
    pd.DataFrame(filas).to_csv(OUT_DIR / "clasificacion_loo.csv", index=False)

    (OUT_DIR / "manifiesto.json").write_text(json.dumps(dict(
        fecha=str(date.today()), raiz=str(args.raiz),
        n_familias=len(datos), n_archivos=resultados["combinado"]["total"],
        n_bytes_ventana=N_BYTES, min_marca=MIN_MARCA,
        familias_sin_marca=sorted(df_marcas[~df_marcas.marca_detectable].familia),
        ablacion=resultados, python=sys.version.split()[0],
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFamilias sin marca estructural: "
          f"{sorted(df_marcas[~df_marcas.marca_detectable].familia)}")
    print(f"Salidas en: {OUT_DIR}")


if __name__ == "__main__":
    main()
