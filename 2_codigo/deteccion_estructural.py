#!/usr/bin/env python3
"""
deteccion_estructural.py -- Experimento 2b: clasificación de familias por
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
     multiclase y cobertura -- comparable con el 9,9 % del enfoque estadístico.

Estructura esperada del dataset (NapierOne tiny en el servidor):
    RAIZ/<FAMILIA>-tiny/*   o   RAIZ/<FAMILIA>/*
Uso:
    python deteccion_estructural.py RUTA_RAIZ [--max-archivos 50] [--umbral 0.90]
    python deteccion_estructural.py RUTA_RAIZ --semilla 1     # otra muestra
Salidas en 4_resultados/resultados_estructural_s<semilla>[_job<SLURM_JOB_ID>]/, un juego
por criterio de extensión: marcas_por_familia_<criterio>.csv,
clasificacion_loo_<criterio>.csv, manifiesto.json. La carpeta lleva la semilla porque los
nombres de los CSV solo distinguen el criterio: al correr varias semillas seguidas contra
una carpeta fija, cada corrida pisaba la anterior (pasó el 2026-08-17). Si la carpeta ya
tiene CSV, el script aborta salvo --forzar.

Correcciones 2026-08-16 (halladas al inventariar extensiones del dataset):
  * Se excluye el .pdf de documentación presente en cada carpeta de NapierOne
    (clasificador_bytes.py y ablacion_ventana_extendida.py ya lo hacían; acá no:
    era una inconsistencia entre scripts propios, no un criterio distinto).
  * Muestreo ALEATORIO con semilla fija en lugar de los primeros N alfabéticos.
    El orden de Python pone los nombres que empiezan con '-' antes que los dígitos,
    y en CERBER eso sesgaba la muestra hacia los archivos renombrados.
  * La extensión común pasa de exigir unanimidad a un umbral de mayoría (--umbral,
    por defecto 0,90). Con unanimidad, 3 archivos sueltos anulaban la extensión .fun
    de JIGSAW (990/1000). Se reportan LAS DOS cifras, unanimidad y umbral, para que
    la elección del criterio quede documentada y no parezca elegida por conveniencia.

Corrección 2026-08-17 (el caso BADRABBIT):
  * El umbral se aplica AHORA TAMBIÉN al prefijo y al sufijo. Antes solo lo llevaba la
    extensión y las firmas binarias seguían exigiendo unanimidad byte a byte, que es
    aún más frágil: un único archivo discrepante lleva el trozo común a CERO. Se vio
    con BADRABBIT, cuyo sufijo «encrypted» en UTF-16LE está en 965 de ~1000 archivos
    (96,5 %): P(los 50 sorteados lo lleven todos) = 0,965^50 ~ 0,17, o sea que en
    ~83 % de las semillas el detector NO encontraba una firma que SÍ existe. Con
    umbral 0,90 aparece siempre y se reporta con su cobertura real.
  * Cada marca sale acompañada de su COBERTURA (fracción de archivos que la llevan).
    «La familia tiene marca» y «todos sus archivos la tienen» son cosas distintas, y
    para el detector la que manda es la segunda.
  * El LOO ya no recalcula las marcas de las otras familias por cada archivo (no
    dependen del archivo evaluado): mismo resultado exacto, mucho más rápido.

Nota metodológica: NapierOne genera los archivos cifrados ejecutando las muestras
reales de ransomware, por lo que las extensiones y marcas son artefactos auténticos
del atacante (verificar y citar Davies et al. 2022) -- a diferencia de los nombres
de archivo de notas curados por repositorios, aquí el nombre/extensión SÍ es evidencia.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import date
from math import ceil
from pathlib import Path
from typing import NamedTuple

import pandas as pd

N_BYTES = 64        # ventana de cabecera y cola a analizar
MIN_MARCA = 4       # bytes mínimos de prefijo/sufijo común para considerarlo marca
_AQUI = Path(__file__).resolve().parent
BASE_SALIDA = (_AQUI.parent / "4_resultados"
               if (_AQUI.parent / "4_resultados").is_dir() else _AQUI)


def carpeta_salida(args):
    """Carpeta de salida ÚNICA por corrida, con la semilla en el nombre.

    Este script escribía siempre en `resultados_estructural/`. El 2026-08-17, al correr
    las 10 semillas que demuestran el parpadeo del criterio de unanimidad, las diez
    corridas se pisaron entre sí: sobrevivieron solo los CSV de la última y se perdió la
    ablación por semilla. Como los CSV llevan el criterio en el nombre pero no la
    semilla, dos corridas con semillas distintas eran indistinguibles.
    """
    if args.salida:
        return BASE_SALIDA / args.salida
    partes = ["resultados_estructural", f"s{args.semilla}"]
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        partes.append(f"job{job}")
    return BASE_SALIDA / "_".join(partes)


def leer_ventanas(path):
    """Devuelve (primeros N_BYTES, últimos N_BYTES) del archivo."""
    with open(path, "rb") as f:
        head = f.read(N_BYTES)
        f.seek(0, 2)
        n = f.tell()
        f.seek(max(0, n - N_BYTES))
        tail = f.read(N_BYTES)
    return head, tail


class Marcas(NamedTuple):
    """Marcas de una familia, cada una con la FRACCIÓN de archivos que la lleva.

    La cobertura importa: «la familia tiene marca» y «todos sus archivos la tienen»
    son cosas distintas, y la segunda es la que determina si el detector la encuentra.
    BADRABBIT lo mostró: su sufijo «encrypted» está en el 96,5 % de los archivos, no en
    todos, así que bajo unanimidad aparecía o desaparecía según la muestra sorteada.
    """
    prefijo: bytes
    cobertura_prefijo: float
    sufijo: bytes
    cobertura_sufijo: float
    extension: str
    cobertura_extension: float


def _minimo(umbral, n):
    """Cuántos archivos deben compartir la marca. El épsilon evita que 0,90*50 = 45,000...4
    exija 46 por error de coma flotante."""
    return ceil(umbral * n - 1e-9)


def prefijo_mayoritario(bloques, umbral=1.0):
    """Prefijo más largo compartido por al menos `umbral` de los bloques.

    Con umbral=1.0 devuelve exactamente el longest common prefix (criterio original):
    un solo bloque discrepante lo lleva a cero. Con umbral<1.0 tolera esa minoría, que
    es el mismo criterio de mayoría que ya se le aplicaba a la extensión.
    Devuelve (prefijo, fracción de bloques que lo comparte)."""
    if not bloques:
        return b"", 0.0
    n = len(bloques)
    minimo = _minimo(umbral, n)
    for largo in range(min(len(b) for b in bloques), 0, -1):
        cand, cnt = Counter(b[:largo] for b in bloques).most_common(1)[0]
        if cnt >= minimo:
            return cand, cnt / n
    return b"", 0.0


def sufijo_mayoritario(bloques, umbral=1.0):
    pre, cob = prefijo_mayoritario([b[::-1] for b in bloques], umbral)
    return pre[::-1], cob


def extension_comun(nombres, umbral=1.0):
    """Extensión más frecuente, si al menos `umbral` de los archivos la comparten.
    Con umbral=1.0 se recupera el criterio original (unanimidad).
    Devuelve (extensión, fracción que la comparte)."""
    exts = [Path(n).suffix.lower() for n in nombres]
    ext, n = Counter(exts).most_common(1)[0]
    cob = n / len(exts)
    return (ext, cob) if ext and n >= _minimo(umbral, len(exts)) else ("", cob)


def cargar_dataset(raiz, max_archivos, semilla=42):
    """{familia: [(nombre, head, tail), ...]}
    Muestreo aleatorio con semilla fija, excluyendo el .pdf de documentación."""
    rng = random.Random(semilla)
    datos = {}
    raiz = Path(raiz)
    for d in sorted(p for p in raiz.iterdir() if p.is_dir()):
        familia = d.name.upper()
        for suf in ("-TINY", "_TINY", "-SMALL", "_SMALL"):
            familia = familia.removesuffix(suf)
        archivos = sorted(p for p in d.iterdir()
                          if p.is_file() and p.suffix.lower() != ".pdf")
        if len(archivos) < 3:
            print(f"  ADVERTENCIA: {familia} tiene {len(archivos)} archivos (<3), omitida.")
            continue
        if len(archivos) > max_archivos:
            archivos = sorted(rng.sample(archivos, max_archivos))
        datos[familia] = [(p.name, *leer_ventanas(p)) for p in archivos]
    return datos


def marcas(entradas, umbral=1.0):
    """Marcas de una lista de (nombre, head, tail).

    El umbral se aplica a LAS TRES marcas. Hasta el 2026-08-17 solo lo llevaba la
    extensión, mientras prefijo y sufijo seguían exigiendo unanimidad byte a byte: era
    la misma inestabilidad, sin corregir, en el otro eje del detector."""
    pre, cob_pre = prefijo_mayoritario([h for _, h, _ in entradas], umbral)
    suf, cob_suf = sufijo_mayoritario([t for _, _, t in entradas], umbral)
    ext, cob_ext = extension_comun([n for n, _, _ in entradas], umbral)
    return Marcas(pre, cob_pre, suf, cob_suf, ext, cob_ext)


def descubrir(datos, umbral):
    """Marcas por familia (con todos los archivos de la muestra)."""
    filas = []
    for familia, entradas in sorted(datos.items()):
        m = marcas(entradas, umbral)
        filas.append(dict(
            familia=familia, n_archivos=len(entradas),
            prefijo_len=len(m.prefijo), prefijo_hex=m.prefijo.hex(),
            prefijo_cobertura=round(m.cobertura_prefijo, 4),
            sufijo_len=len(m.sufijo), sufijo_hex=m.sufijo.hex(),
            sufijo_cobertura=round(m.cobertura_sufijo, 4),
            extension=m.extension,
            extension_cobertura=round(m.cobertura_extension, 4),
            marca_detectable=(len(m.prefijo) >= MIN_MARCA
                              or len(m.sufijo) >= MIN_MARCA or bool(m.extension)),
        ))
    return pd.DataFrame(filas)


def clasificar_loo(datos, umbral):
    """Clasificación leave-one-out con ABLACIÓN por tipo de marca.
    Un revisor preguntará cuánto del rendimiento aporta la extensión del archivo
    (que ya explotan las herramientas por reglas) frente a las firmas binarias
    insertadas en el contenido. Se evalúan los tres modos por separado."""
    MODOS = ("solo_extension", "solo_firmas_binarias", "combinado")
    resultados = {}
    detalle_por_familia = {}

    # Las marcas de las OTRAS familias no dependen del archivo que se está evaluando,
    # así que se calculan una sola vez. Antes se recalculaban 30 veces por archivo
    # (45.000 veces en total): es el mismo resultado, exacto, sin el costo.
    marcas_completas = {f: marcas(e, umbral) for f, e in datos.items()}

    for modo in MODOS:
        aciertos, sin_marca, total = 0, 0, 0
        por_familia, ok_familia = Counter(), Counter()
        for familia, entradas in datos.items():
            for i, (nombre, head, tail) in enumerate(entradas):
                total += 1
                por_familia[familia] += 1
                mejor, mejor_score = None, 0
                for fam2, entradas2 in datos.items():
                    if fam2 == familia:
                        m = marcas([e for j, e in enumerate(entradas2) if j != i],
                                   umbral)
                    else:
                        m = marcas_completas[fam2]
                    pre, suf, ext = m.prefijo, m.sufijo, m.extension
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

    filas = []
    for f in sorted(datos):
        fila = dict(familia=f, n=detalle_por_familia["combinado"][0][f])
        for m in MODOS:
            pf, okf = detalle_por_familia[m]
            fila[f"recall_{m}"] = round(okf.get(f, 0) / pf[f], 3) if pf[f] else 0
        filas.append(fila)
    return resultados, pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz", help="carpeta con subcarpetas <FAMILIA>[-tiny]")
    ap.add_argument("--max-archivos", type=int, default=50)
    ap.add_argument("--umbral", type=float, default=0.90,
                    help="fracción mínima de archivos que deben compartir la "
                         "extensión para contarla como marca (1.0 = unanimidad)")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--salida", default="",
                    help="nombre de la carpeta de salida dentro de 4_resultados/ "
                         "(por defecto se arma con la semilla y el job de SLURM)")
    ap.add_argument("--forzar", action="store_true",
                    help="permitir escribir en una carpeta que ya tiene resultados")
    args = ap.parse_args()

    OUT_DIR = carpeta_salida(args)
    previos = sorted(OUT_DIR.glob("*.csv")) if OUT_DIR.is_dir() else []
    if previos and not args.forzar:
        sys.exit(f"ABORTA: {OUT_DIR} ya tiene {len(previos)} CSV de una corrida anterior "
                 f"({previos[0].name}...). Usá --salida OTRO_NOMBRE, o --forzar si de "
                 f"verdad querés sobrescribirlos.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {args.raiz} | semilla {args.semilla} | "
          f"umbral de extensión {args.umbral} | sin .pdf | muestreo aleatorio")
    datos = cargar_dataset(args.raiz, args.max_archivos, args.semilla)
    print(f"Familias: {len(datos)} | Archivos: {sum(len(v) for v in datos.values())}")

    # Se evalúan LOS DOS criterios sobre la misma muestra: el original (unanimidad)
    # y el umbral declarado. Registrar ambos evita que la elección del criterio se
    # lea como hecha por conveniencia; la comparación va al capítulo 4.
    criterios = [("unanimidad", 1.0)]
    if args.umbral < 1.0:
        criterios.append((f"umbral_{int(round(args.umbral * 100))}", args.umbral))

    manifiesto = dict(
        fecha=str(date.today()), raiz=str(args.raiz),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        n_familias=len(datos), n_archivos=sum(len(v) for v in datos.values()),
        n_bytes_ventana=N_BYTES, min_marca=MIN_MARCA,
        semilla=args.semilla, umbral=args.umbral,
        muestreo="aleatorio con semilla fija, excluyendo .pdf",
        criterios={}, python=sys.version.split()[0],
    )

    for etiqueta, umbral in criterios:
        print("\n" + "=" * 70)
        print(f"  CRITERIO DE EXTENSIÓN: {etiqueta}")
        print("=" * 70)

        df_marcas = descubrir(datos, umbral)
        df_marcas.to_csv(OUT_DIR / f"marcas_por_familia_{etiqueta}.csv", index=False)
        detectables = df_marcas[df_marcas.marca_detectable]
        print(f"\nFamilias con marca estructural (prefijo/sufijo >= {MIN_MARCA} bytes "
              f"o extensión propia): {len(detectables)}/{len(df_marcas)}")
        for _, r in detectables.iterrows():
            partes = []
            if r.prefijo_len >= MIN_MARCA:
                partes.append(f"prefijo {r.prefijo_len}B {r.prefijo_hex[:24]}"
                              f" ({r.prefijo_cobertura:.0%})")
            if r.sufijo_len >= MIN_MARCA:
                partes.append(f"sufijo {r.sufijo_len}B ...{r.sufijo_hex[-24:]}"
                              f" ({r.sufijo_cobertura:.0%})")
            if r.extension:
                partes.append(f"ext {r.extension} ({r.extension_cobertura:.0%})")
            print(f"  {r.familia:<15} {' | '.join(partes)}")

        resultados, df_loo = clasificar_loo(datos, umbral)
        df_loo.to_csv(OUT_DIR / f"clasificacion_loo_{etiqueta}.csv", index=False)

        sin_marca = sorted(df_marcas[~df_marcas.marca_detectable].familia)
        manifiesto["criterios"][etiqueta] = dict(
            umbral=umbral, familias_con_marca=int(len(detectables)),
            familias_sin_marca=sin_marca, ablacion=resultados)
        print(f"\nFamilias sin marca estructural ({etiqueta}): {sin_marca}")

    (OUT_DIR / "manifiesto.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSalidas en: {OUT_DIR}")


if __name__ == "__main__":
    main()
