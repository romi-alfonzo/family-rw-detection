#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""revision_tesis_archivos.py -- REPRODUCCIÓN INDEPENDIENTE de la cifra de cabecera del frente
de archivos y censo del conjunto, en UNA sola corrida de clúster.

QUIÉN Y POR QUÉ
---------------
Lo escribe la revisión científica independiente de la tesis (skill `revisar-tesis`, 2026-09-29),
como haría el comité de evaluación de artefactos de una conferencia: la prueba más fuerte de que
un número es real es que otra persona lo obtenga con OTRA implementación. Por eso este script
NO importa ninguna función de los scripts de la tesis (exp2e/2f/2g/2h, clasificador_bytes): los
rasgos se reimplementan a partir de la DESCRIPCIÓN del capítulo 4 (§Exp. 2c, 2e y 2g), no del
código. Lo único compartido, a propósito, es la definición del modelo (Random Forest 300 árboles,
profundidad 20, hoja 2, max_features 0,3, class_weight balanced; 5 pliegues estratificados) y la
regla de muestreo (500 archivos por familia, permutación con `default_rng(semilla)` sobre la lista
ordenada), para que las cifras sean comparables semilla a semilla con las publicadas.

QUÉ MIDE, EN ORDEN DE IMPORTANCIA (el log sale en este orden)
------------------------------------------------------------
(A) CENSO INDEPENDIENTE del conjunto completo, sin aprendizaje: familias, muestras por familia y
    por tipo de documento, archivos con cabecera de documento en claro (magia de tipo), extensiones
    por familia y compartidas, duplicados exactos por SHA-256 dentro y entre familias.
(B) VALIDACIÓN CRUZADA, 5 semillas x 5 pliegues, 500 archivos por familia:
      (1) bytes 512+512 (solo semilla 0, control contra el Exp. 2c)
      (2) bytes + 44 rasgos estructurales           (Exp. 2e: 0,9359 ± 0,0004 de macro-F1)
      (5) bytes + estructura + forma de la extensión (Exp. 2g, CANÓNICO: 0,9998 ± 0,0001)
      (8) bytes + forma de la extensión             (Exp. 2h: 0,9999 ± 0,0001)
      (6) solo forma de la extensión                (Exp. 2h: 0,8781 ± 0,0013)
    con F1 por familia y delta pareado por semilla.
(C) DEJAR-UN-TIPO-FUERA con ponderación de clases: (5) en 5 semillas de muestreo (Exp. 2h:
    0,9983 ± 0,0011), y (2) y (8) en la semilla 0.
(D) CONTROLES DE FUGA en la muestra: pares duplicados repartidos entre pliegues; y la tabla de
    consulta por extensión (Exp. 2d: 0,9724 de exactitud en muestra).

PREREGISTRO -- escrito y commiteado ANTES de correr (2026-09-29)
----------------------------------------------------------------
Censo (conjunto completo; referencia: Exp. 2h, job 4096, transcripto en ESTADO_TESIS.md)
  C1. 30 familias y 29.948 muestras (sin la documentación `<FAMILIA>.pdf` ni los subdirectorios).
  C2. Duplicados SHA-256: 16 grupos / 32 archivos, TODOS dentro de una familia; 0 entre familias.
  C3. Cabecera de documento en claro: CERBER 988, JIGSAW 2, las demás 0.
  C4. BLACKMATTER: 988 jpg; NOTPETYA: 0 jpg.
  C5. 25 familias con una sola extensión; extensiones en más de una familia: solo las de documento.
Validación cruzada (referencias: Exp. 2c/2e/2g/2h)
  R1. (1) semilla 0: exactitud y macro-F1 a ±0,005 de 0,9123 / 0,9114.
  R2. (2) media de 5 semillas: macro-F1 a ±0,005 de 0,9359.
  R3. (5) media de 5 semillas: macro-F1 >= 0,9990 y las 30 familias con F1 medio >= 0,99.
  R4. (8) >= 0,9990; (6) entre 0,85 y 0,90.
  R5. Delta (5)-(2) de macro-F1 >= +0,060 en cada una de las 5 semillas.
Tipos no vistos (referencia: Exp. 2h con ponderación)
  T1. (5), 5 semillas: promedio >= 0,995 y ningún pliegue < 0,975 (publicado: 0,9983 ± 0,0011; pdf 0,9800).
  T2. (2) semilla 0: promedio a ±0,010 de 0,8814; (8) semilla 0: promedio >= 0,995.
Fuga y atajos
  F1. Pares duplicados de la muestra repartidos entre pliegues: <= 3 por semilla (efecto <= 0,0002).
  L1. Tabla de consulta por extensión, en muestra (semilla 0): exactitud entre 0,96 y 0,98.

Lectura acordada de antemano
  - R3 cumple -> la cifra de cabecera queda REPRODUCIDA por reimplementación independiente
    (nivel «Results Reproduced»). R3 falla y R2 cumple -> la diferencia está en la capa de la
    extensión: se reporta la cifra obtenida y se compara rasgo por rasgo.
  - C2, C3 o C4 fallan -> el censo del capítulo 4 se corrige con estos números (y se investiga
    por qué difieren dos censos del mismo conjunto).
  - F1 falla -> hay fuga por duplicados en la validación cruzada: se cuantifica su efecto
    reentrenando sin los duplicados (no en esta corrida).
  - Todo lo que falle se reporta igual.

USO
---
    python3.11 revision_tesis_archivos.py /scratch/ralfonzo/Napierone-small --por-familia 500 --semillas 0,1,2,3,4
    python3.11 revision_tesis_archivos.py --sintetico            # prueba de punta a punta con datos falsos

Salidas en `resultados_revision_archivos_job<id>/` junto al script (aborta si ya existe):
  censo_familias.csv, censo_duplicados.csv, censo_extensiones.csv, cv_por_semilla.csv,
  cv_por_familia_y_semilla.csv, cv_resumen.csv, cv_por_familia_resumen.csv, tipos_por_pliegue.csv,
  fuga_duplicados_por_semilla.csv, log.txt, manifiesto.json (con el sha256 de este script).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ----------------------------------------------------------------------------- constantes
N_HEAD = N_TAIL = 512                 # bytes posicionales (Exp. 2c)
PROFUNDIDADES = (16, 32, 64, 128, 256, 512, 1024, 4096)   # entropía a ocho profundidades (Exp. 2e)
N_BLOQUES_MEDIO, BLOQUE_MEDIO = 8, 512                     # ocho bloques repartidos por el archivo
BLOQUE_COLA, UMBRAL_COLA = 32, 4.4                         # bloque final no aleatorio
HIPER = dict(n_estimators=300, max_depth=20, min_samples_leaf=2, max_features=0.3)
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or max(1, (os.cpu_count() or 2) - 1)
EXT_DOCUMENTO = {"doc", "docx", "xls", "xlsx", "ppt", "pptx", "pdf", "jpg", "jpeg", "png", "gif",
                 "bmp", "tif", "tiff", "txt", "rtf", "csv", "odt", "ods", "odp", "html", "htm",
                 "xml", "zip", "mp3", "mp4", "avi", "wav"}
MAGIAS = {b"\xff\xd8\xff": "JPEG", b"%PDF": "PDF", b"PK\x03\x04": "ZIP/OOXML",
          b"\xd0\xcf\x11\xe0": "OLE", b"\x89PNG": "PNG", b"GIF8": "GIF", b"{\\rtf": "RTF",
          b"\x1f\x8b": "GZIP"}
HEX = set("0123456789abcdefABCDEF")
DIFICILES = ["NOTPETYA", "JIGSAW", "CRYPTOLOCKER", "DARKSIDE", "WASTEDLOCKER", "SUNCRYPT"]
COLUMNAS = {"1": "bytes", "2": "bytes+estructura", "5": "bytes+estructura+extension",
            "8": "bytes+extension", "6": "solo_extension"}
# referencias publicadas (transcriptas de ESTADO_TESIS.md; sirven para el veredicto, no para calcular)
REF = dict(cv_1=(0.9123, 0.9114), cv_2=0.9359, cv_5=0.9998, cv_8=0.9999, cv_6=0.8781,
           tipos_5=0.9983, tipos_2=0.8814, lookup=0.9724, censo_total=29948)


# ----------------------------------------------------------------------------- utilidades
def entropia(b: bytes) -> float:
    """Entropía de Shannon en bits por byte, sobre la distribución empírica de los bytes."""
    if not b:
        return 0.0
    cuenta = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256).astype(np.float64)
    p = cuenta[cuenta > 0] / len(b)
    return float(-(p * np.log2(p)).sum())


def chi2_uniforme(b: bytes) -> float:
    if not b:
        return 0.0
    esperado = len(b) / 256.0
    cuenta = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256).astype(np.float64)
    return float(((cuenta - esperado) ** 2 / esperado).sum())


def familia_de(d: Path) -> str:
    fam = d.name.upper()
    for suf in ("-SMALL", "_SMALL", "-TINY", "_TINY"):
        if fam.endswith(suf):
            fam = fam[: -len(suf)]
    return fam


def es_documentacion(p: Path, fam: str) -> bool:
    """El PDF descriptivo de NapierOne (`<FAMILIA>.pdf`), que no es una muestra."""
    return p.suffix.lower() == ".pdf" and p.stem.upper() == fam


def tipo_documento(nombre: str) -> str:
    m = re.match(r"^\d+-([a-z0-9]+)", nombre.lower())
    return m.group(1) if m else "desconocido"


def extension_final(nombre: str) -> str:
    return nombre.rpartition(".")[2] if "." in nombre else ""


def magia_en_claro(cab: bytes):
    for firma, etiqueta in MAGIAS.items():
        if cab.startswith(firma):
            return etiqueta
    return None


# ----------------------------------------------------------------------------- lectura y rasgos
def leer(path: Path):
    """Devuelve (cabecera 4096, cola 4096, bloques del medio, tamaño). Una sola pasada parcial."""
    with open(path, "rb") as fh:
        tam = os.fstat(fh.fileno()).st_size
        cab = fh.read(4096)
        if tam > 4096:
            fh.seek(tam - 4096)
            cola = fh.read(4096)
        else:
            cola = cab
        medios = []
        if tam > 3 * BLOQUE_MEDIO:
            paso = tam // (N_BLOQUES_MEDIO + 1)
            for i in range(1, N_BLOQUES_MEDIO + 1):
                fh.seek(min(i * paso, tam - BLOQUE_MEDIO))
                medios.append(fh.read(BLOQUE_MEDIO))
    return cab, cola, medios, tam


def bytes_posicionales(cab: bytes, cola: bytes) -> bytes:
    """512 primeros y 512 últimos bytes; los archivos más cortos se rellenan con ceros (§4.5.6)."""
    return cab[:N_HEAD].ljust(N_HEAD, b"\x00") + cola[-N_TAIL:].rjust(N_TAIL, b"\x00")


def largo_bloque_final_no_aleatorio(cola: bytes) -> int:
    """Bytes del final que forman bloques de 32 con entropía baja, contados de atrás hacia adelante."""
    n = 0
    for i in range(len(cola) - BLOQUE_COLA, -1, -BLOQUE_COLA):
        if entropia(cola[i:i + BLOQUE_COLA]) < UMBRAL_COLA:
            n += BLOQUE_COLA
        else:
            break
    return n


def rasgos_estructurales(cab: bytes, cola: bytes, medios: list, tam: int) -> list:
    """44 rasgos que describen la FORMA del archivo, según §Exp. 2e: entropía a ocho profundidades
    en cada extremo, ocho bloques del medio con sus cuatro estadísticos, bloque final no aleatorio,
    tamaño y restos módulo 16/512/4096, chi cuadrado, valores distintos, frecuencia máxima, ceros e
    imprimibles. Ningún valor de byte en ninguna posición."""
    f = [float(tam), math.log1p(tam), float(tam % 16), float(tam % 512), float(tam % 4096)]
    f += [entropia(cab[:n]) for n in PROFUNDIDADES]
    f += [entropia(cola[-n:]) for n in PROFUNDIDADES]
    hm = [entropia(b) for b in medios]
    hm = (hm + [0.0] * N_BLOQUES_MEDIO)[:N_BLOQUES_MEDIO]
    f += hm
    arr = np.asarray(hm)
    f += [float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())]
    c, t = cab[:N_HEAD], cola[-N_TAIL:]
    f += [chi2_uniforme(c), chi2_uniforme(t)]
    f += [float(len(set(c))), float(len(set(t)))]
    f += [float(max(Counter(c).values())) if c else 0.0, float(max(Counter(t).values())) if t else 0.0]
    f += [float(c.count(0)), float(t.count(0))]
    f += [sum(1 for b in t if 32 <= b < 127) / max(1, len(t))]
    f += [float(largo_bloque_final_no_aleatorio(cola))]
    f += [entropia(c) - entropia(t)]
    assert len(f) == 44, len(f)
    return f


def forma_de_la_extension(nombre: str) -> list:
    """14 rasgos de la forma de la extensión FINAL (§Exp. 2g): longitud; cuenta y proporción de
    dígitos, letras y mayúsculas; proporción hexadecimal; entropía; tres indicadores; si es la
    extensión de un tipo de documento conocido; y la cantidad de puntos del nombre."""
    ext = extension_final(nombre)
    L = max(1, len(ext))
    d = sum(ch.isdigit() for ch in ext)
    a = sum(ch.isalpha() for ch in ext)
    u = sum(ch.isupper() for ch in ext)
    h = sum(ch in HEX for ch in ext)
    cuenta = Counter(ext)
    ent = -sum((v / L) * math.log2(v / L) for v in cuenta.values()) if ext else 0.0
    f = [float(len(ext)), float(d), float(a), float(u), d / L, a / L, u / L, h / L, ent,
         1.0 if ext and all(ch in HEX for ch in ext) else 0.0,
         1.0 if ext and ext.isalpha() else 0.0,
         1.0 if any(ch.isdigit() for ch in ext) else 0.0,
         1.0 if ext.lower() in EXT_DOCUMENTO else 0.0,
         float(nombre.count("."))]
    assert len(f) == 14
    return f


# ----------------------------------------------------------------------------- (A) censo
def censo(raiz: Path, out: Path, log):
    """Recorre el conjunto COMPLETO (sin muestrear): recuentos, magia en claro, extensiones y
    duplicados SHA-256. Lee cada archivo entero una vez."""
    t0 = time.time()
    filas, dups, ext_por_fam = [], defaultdict(list), defaultdict(Counter)
    total = 0
    for d in sorted(p for p in raiz.iterdir() if p.is_dir()):
        fam = familia_de(d)
        archivos = sorted(p for p in d.iterdir() if p.is_file() and not es_documentacion(p, fam))
        tipos, claro = Counter(), Counter()
        for p in archivos:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                primero = fh.read(1 << 20)
                h.update(primero)
                while True:
                    trozo = fh.read(1 << 20)
                    if not trozo:
                        break
                    h.update(trozo)
            dups[h.hexdigest()].append((fam, p.name))
            tipos[tipo_documento(p.name)] += 1
            m = magia_en_claro(primero[:8])
            if m:
                claro[m] += 1
            ext_por_fam[fam][extension_final(p.name).lower()] += 1
        total += len(archivos)
        filas.append(dict(familia=fam, muestras=len(archivos), en_claro=sum(claro.values()),
                          en_claro_detalle=";".join(f"{k}:{v}" for k, v in sorted(claro.items())),
                          extensiones_distintas=len(ext_por_fam[fam]),
                          **{f"n_{t}": tipos.get(t, 0) for t in
                             ("doc", "docx", "jpg", "pdf", "pptx", "xls", "xlsx", "desconocido")}))
        log(f"  {fam:<14} {len(archivos):>5} muestras · en claro {sum(claro.values()):>4} · "
            f"ext. distintas {len(ext_por_fam[fam]):>5} · jpg {tipos.get('jpg', 0):>4} · "
            f"sin tipo {tipos.get('desconocido', 0):>4}")
    dfc = pd.DataFrame(filas)
    dfc.to_csv(out / "censo_familias.csv", index=False)

    grupos = [(h, v) for h, v in dups.items() if len(v) > 1]
    filas_d = [dict(sha256=h, n=len(v), familias=";".join(sorted({f for f, _ in v})),
                    entre_familias=len({f for f, _ in v}) > 1,
                    archivos=";".join(f"{f}/{n}" for f, n in v)) for h, v in grupos]
    dfd = pd.DataFrame(filas_d, columns=["sha256", "n", "familias", "entre_familias", "archivos"])
    dfd.to_csv(out / "censo_duplicados.csv", index=False)
    n_grupos = len(grupos)
    n_arch = sum(len(v) for _, v in grupos)
    n_entre = int(dfd.entre_familias.sum()) if len(dfd) else 0

    ext_fams = defaultdict(set)
    for fam, c in ext_por_fam.items():
        for e in c:
            ext_fams[e].add(fam)
    compartidas = {e: sorted(f) for e, f in ext_fams.items() if len(f) > 1}
    pd.DataFrame([dict(extension=e, familias=";".join(f)) for e, f in sorted(compartidas.items())],
                 columns=["extension", "familias"]).to_csv(out / "censo_extensiones.csv", index=False)
    una_sola = int((dfc.extensiones_distintas == 1).sum())

    log(f"\n  TOTAL: {len(dfc)} familias, {total} muestras ({round(time.time() - t0)} s)")
    log(f"  Duplicados SHA-256: {n_grupos} grupos / {n_arch} archivos; entre familias: {n_entre} grupos")
    for h, v in grupos:
        log("     " + " = ".join(f"{f}/{n}" for f, n in v))
    log(f"  Familias con una sola extensión: {una_sola} de {len(dfc)}; extensiones distintas en total: "
        f"{len(ext_fams)}; compartidas por más de una familia: {len(compartidas)} "
        f"({', '.join(sorted(compartidas)) if compartidas else 'ninguna'})")
    en_claro = {r.familia: r.en_claro for r in dfc.itertuples()}
    jpg = {r.familia: r.n_jpg for r in dfc.itertuples()}
    return dict(familias=len(dfc), total=total, dup_grupos=n_grupos, dup_archivos=n_arch,
                dup_entre=n_entre, una_sola=una_sola, compartidas=sorted(compartidas),
                en_claro=en_claro, jpg=jpg)


# ----------------------------------------------------------------------------- carga de la muestra
def cargar_muestra(raiz: Path, por_familia: int, semilla: int, log):
    rng = np.random.default_rng(semilla)
    Xb, Xe, Xr, y, nombres, tipos, hashes = [], [], [], [], [], [], []
    for d in sorted(p for p in raiz.iterdir() if p.is_dir()):
        fam = familia_de(d)
        archivos = sorted(p for p in d.iterdir() if p.is_file() and not es_documentacion(p, fam))
        if len(archivos) < 6:
            log(f"  ADVERTENCIA: {fam} tiene {len(archivos)} archivos, omitida")
            continue
        if len(archivos) < por_familia:
            log(f"  ADVERTENCIA: {fam} tiene {len(archivos)} < {por_familia} archivos")
        for p in (archivos[i] for i in rng.permutation(len(archivos))[:por_familia]):
            cab, cola, medios, tam = leer(p)
            Xb.append(bytes_posicionales(cab, cola))
            Xe.append(rasgos_estructurales(cab, cola, medios, tam))
            Xr.append(forma_de_la_extension(p.name))
            y.append(fam)
            nombres.append(p.name)
            tipos.append(tipo_documento(p.name))
            hashes.append(hashlib.sha256(open(p, "rb").read()).hexdigest())
    Xb = np.frombuffer(b"".join(Xb), dtype=np.uint8).reshape(len(Xb), N_HEAD + N_TAIL).astype(np.float32)
    Xe = np.nan_to_num(np.asarray(Xe, dtype=np.float32))
    Xr = np.asarray(Xr, dtype=np.float32)
    return Xb, Xe, Xr, np.array(y), nombres, np.array(tipos), hashes


def matriz(col: str, Xb, Xe, Xr):
    return {"1": Xb, "2": np.hstack([Xb, Xe]), "5": np.hstack([Xb, Xe, Xr]),
            "8": np.hstack([Xb, Xr]), "6": Xr}[col]


def ic95(d):
    from scipy import stats
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return float(d.mean()), float("nan"), float("nan")
    t = stats.t.ppf(0.975, len(d) - 1)
    ee = d.std(ddof=1) / math.sqrt(len(d))
    return float(d.mean()), float(d.mean() - t * ee), float(d.mean() + t * ee)


def veredicto(log, ok, codigo, texto, valor):
    log(f"  [{'CUMPLE' if ok else 'FALLA '}] {codigo} {texto:<62} {valor}")
    return bool(ok)


# ----------------------------------------------------------------------------- datos sintéticos
def crear_sintetico(destino: Path, n_familias=12, por_familia=40, semilla=0):
    """Conjunto falso con la forma de NapierOne-small, para probar el script de punta a punta:
    carpetas `<FAM>-small`, nombres `NNNN-<tipo>.<tipo>.<ext>` (jpg con `-fromweb`), un `<FAM>.pdf`
    de documentación, firmas de familia (sufijo constante), extensiones propias, dos familias que no
    renombran, una con cabecera en claro, y tres documentos de origen duplicados."""
    rng = np.random.default_rng(semilla)
    tipos = ["doc", "docx", "jpg", "pdf", "pptx", "xls", "xlsx"]
    base = {}
    for i in range(por_familia):
        t = tipos[i % len(tipos)]
        n = f"{i:04d}-{t}-fromweb.{t}" if t == "jpg" else f"{i:04d}-{t}.{t}"
        base[n] = rng.integers(0, 256, size=int(rng.integers(600, 9000)), dtype=np.uint8).tobytes()
    # tres documentos de origen repetidos (como en NapierOne)
    nombres = sorted(base)
    for a, b in ((0, 1), (2, 3), (4, 5)):
        base[nombres[b]] = base[nombres[a]]
    for k in range(n_familias):
        fam = f"FAM{k:02d}"
        d = destino / f"{fam}-small"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{fam}.pdf").write_bytes(b"%PDF-1.4 documentacion " + bytes(3000))
        sufijo = bytes(rng.integers(0, 256, size=24, dtype=np.uint8)) if k % 3 else b""
        ext = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=4 + k % 3)) if k >= 2 else None
        for n, contenido in base.items():
            cuerpo = bytes(rng.integers(0, 256, size=len(contenido), dtype=np.uint8)) if k % 4 else contenido
            if k == 7:                       # cabecera en claro, como CERBER
                cuerpo = b"PK\x03\x04" + cuerpo[4:]
                nombre = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), size=10)) + f".{ext}"
            elif ext is None:                # no renombra, como BADRABBIT / NOTPETYA
                nombre = n
            else:
                nombre = f"{n}.{ext}"
            if k % 4 == 0:                   # determinista: el mismo original da el mismo cifrado
                cuerpo = bytes((b ^ (k + 1)) & 0xFF for b in contenido)
            (d / nombre).write_bytes(cuerpo + sufijo)
    return destino


# ----------------------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser(description="Reproducción independiente del frente de archivos")
    ap.add_argument("raiz", nargs="?", help="carpeta con las 30 familias (NapierOne-small)")
    ap.add_argument("--por-familia", type=int, default=500)
    ap.add_argument("--semillas", default="0,1,2,3,4")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--sin-censo", action="store_true", help="omitir (A), que lee el conjunto entero")
    ap.add_argument("--sintetico", action="store_true", help="prueba de punta a punta con datos falsos")
    args = ap.parse_args()

    min_tipo, min_fams = 200, 10
    if args.sintetico:
        tmp = Path(tempfile.mkdtemp(prefix="napier_sintetico_"))
        args.raiz = str(crear_sintetico(tmp))
        args.por_familia, args.semillas, args.folds = 30, "0,1", 3
        min_tipo, min_fams = 20, 3
        hiper = dict(HIPER, n_estimators=40)
    else:
        hiper = dict(HIPER)
    if not args.raiz:
        sys.exit("Falta la carpeta de datos (o --sintetico).")
    raiz = Path(args.raiz)
    semillas = [int(s) for s in args.semillas.split(",") if s.strip()]

    aqui = Path(__file__).resolve().parent
    base_out = aqui.parent / "4_resultados" if (aqui.parent / "4_resultados").is_dir() else aqui
    out = args.salida or base_out / ("resultados_revision_archivos_job" + os.environ.get("SLURM_JOB_ID", "local"))
    if out.exists() and any(out.iterdir()):
        sys.exit(f"ABORTA: {out} ya tiene resultados. Borrarla o pasar --salida.")
    out.mkdir(parents=True, exist_ok=True)
    reg = open(out / "log.txt", "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        reg.write(m + "\n")
        reg.flush()

    import sklearn
    log("=" * 78)
    log("  REVISIÓN INDEPENDIENTE -- frente de archivos: censo y reproducción del 0,9998")
    log("=" * 78)
    log(f"  {raiz} · {args.por_familia}/familia · semillas {semillas} · {args.folds} pliegues · núcleos {N_JOBS}")
    log(f"  python {sys.version.split()[0]} · sklearn {sklearn.__version__} · numpy {np.__version__} · pandas {pd.__version__}")
    log(f"  script sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}… · inicio {datetime.now():%Y-%m-%d %H:%M}")
    resultados = {}

    # ------------------------------------------------------------------ (A) censo
    if not args.sin_censo:
        log("\n" + "=" * 78 + "\n  (A) CENSO INDEPENDIENTE DEL CONJUNTO COMPLETO\n" + "=" * 78)
        c = censo(raiz, out, log)
        resultados["censo"] = c
        log("\n  Predicciones del censo:")
        c1 = veredicto(log, c["familias"] == 30 and c["total"] == REF["censo_total"], "C1",
                       "30 familias y 29.948 muestras", f"{c['familias']} / {c['total']}")
        c2 = veredicto(log, c["dup_grupos"] == 16 and c["dup_archivos"] == 32 and c["dup_entre"] == 0, "C2",
                       "duplicados 16 grupos / 32 archivos, 0 entre familias",
                       f"{c['dup_grupos']} / {c['dup_archivos']} / entre {c['dup_entre']}")
        ec = c["en_claro"]
        c3 = veredicto(log, ec.get("CERBER") == 988 and ec.get("JIGSAW") == 2 and
                       all(v == 0 for f, v in ec.items() if f not in ("CERBER", "JIGSAW")), "C3",
                       "en claro: CERBER 988, JIGSAW 2, resto 0",
                       ", ".join(f"{f} {v}" for f, v in sorted(ec.items()) if v))
        c4 = veredicto(log, c["jpg"].get("BLACKMATTER") == 988 and c["jpg"].get("NOTPETYA") == 0, "C4",
                       "BLACKMATTER 988 jpg; NOTPETYA 0 jpg",
                       f"BLACKMATTER {c['jpg'].get('BLACKMATTER')} · NOTPETYA {c['jpg'].get('NOTPETYA')}")
        c5 = veredicto(log, c["una_sola"] == 25 and all(e in EXT_DOCUMENTO for e in c["compartidas"]), "C5",
                       "25 familias con una extensión; compartidas solo las de documento",
                       f"{c['una_sola']} · {c['compartidas']}")
        resultados["preregistro_censo"] = dict(C1=c1, C2=c2, C3=c3, C4=c4, C5=c5)

    # ------------------------------------------------------------------ (B) validación cruzada
    log("\n" + "=" * 78 + "\n  (B) VALIDACIÓN CRUZADA -- reimplementación independiente\n" + "=" * 78)
    filas, porfam, fuga, muestras = [], [], [], {}
    for s in semillas:
        t0 = time.time()
        log(f"\n--- semilla {s} ---")
        Xb, Xe, Xr, y, nombres, tipos, hashes = cargar_muestra(raiz, args.por_familia, s, log)
        muestras[s] = (Xb, Xe, Xr, y, nombres, tipos)
        log(f"  {len(y)} archivos, {len(set(y))} familias, rasgos: bytes {Xb.shape[1]} · estructura {Xe.shape[1]} · extensión {Xr.shape[1]}")
        cv = StratifiedKFold(args.folds, shuffle=True, random_state=s)
        pliegue = np.empty(len(y), dtype=int)
        for k, (_, te) in enumerate(cv.split(Xb, y)):
            pliegue[te] = k
        # (D) control de fuga: pares duplicados de la muestra repartidos entre pliegues
        por_hash = defaultdict(list)
        for i, h in enumerate(hashes):
            por_hash[h].append(i)
        pares = [ix for ix in por_hash.values() if len(ix) > 1]
        partidos = sum(1 for ix in pares if len({pliegue[i] for i in ix}) > 1)
        fuga.append(dict(semilla=s, grupos_duplicados_en_muestra=len(pares), grupos_partidos_entre_pliegues=partidos))
        log(f"  duplicados en la muestra: {len(pares)} grupos, {partidos} repartidos entre pliegues")
        columnas = ["1", "2", "5", "8", "6"] if s == semillas[0] else ["2", "5", "8", "6"]
        for col in columnas:
            X = matriz(col, Xb, Xe, Xr)
            clf = RandomForestClassifier(random_state=s, n_jobs=1, class_weight="balanced", **hiper)
            yp = cross_val_predict(clf, X, y, cv=cv, n_jobs=N_JOBS)
            f = dict(semilla=s, columna=col, nombre=COLUMNAS[col], n_rasgos=X.shape[1],
                     exactitud=round(accuracy_score(y, yp), 4),
                     exactitud_balanceada=round(balanced_accuracy_score(y, yp), 4),
                     macro_f1=round(f1_score(y, yp, average="macro", zero_division=0), 4))
            filas.append(f)
            rep = classification_report(y, yp, zero_division=0, output_dict=True)
            porfam += [dict(semilla=s, columna=col, familia=fam, f1=round(rep[fam]["f1-score"], 4))
                       for fam in sorted(set(y))]
            log(f"    ({col}) {COLUMNAS[col]:<30} exactitud {f['exactitud']:.4f} | macro-F1 {f['macro_f1']:.4f}")
            pd.DataFrame(filas).to_csv(out / "cv_por_semilla.csv", index=False)
            pd.DataFrame(porfam).to_csv(out / "cv_por_familia_y_semilla.csv", index=False)
        pd.DataFrame(fuga).to_csv(out / "fuga_duplicados_por_semilla.csv", index=False)
        log(f"  ({round(time.time() - t0)} s)")

    df, pf = pd.DataFrame(filas), pd.DataFrame(porfam)
    res = df.groupby("columna")[["exactitud", "macro_f1"]].agg(["mean", "std"]).round(4)
    res.to_csv(out / "cv_resumen.csv")
    log("\n  RESUMEN CV (media ± desvío entre semillas):")
    log(res.to_string())
    piv = df.pivot(index="semilla", columns="columna", values="macro_f1")
    d52 = piv["5"] - piv["2"]
    m, lo, hi = ic95(d52)
    log(f"\n  Δ (5)-(2) macro-F1: {m:+.4f} [{lo:+.4f}; {hi:+.4f}]  {int((d52 > 0).sum())}/{len(d52)}  (publicado +0,0639)")
    if "8" in piv:
        m8, lo8, hi8 = ic95(piv["8"] - piv["5"])
        log(f"  Δ (8)-(5) macro-F1: {m8:+.4f} [{lo8:+.4f}; {hi8:+.4f}]  (publicado +0,0001, n.s.)")
    fr = pf[pf.columna == "5"].groupby("familia").f1.agg(["mean", "std"]).round(4)
    fr2 = pf[pf.columna == "2"].groupby("familia").f1.mean().round(4)
    orden = [f for f in DIFICILES if f in fr.index] + sorted(f for f in fr.index if f not in DIFICILES)
    tabla = pd.DataFrame({"b+estructura": fr2.loc[orden], "sistema_completo": fr.loc[orden, "mean"],
                          "desvio": fr.loc[orden, "std"]})
    tabla.to_csv(out / "cv_por_familia_resumen.csv")
    log("\n  F1 POR FAMILIA (media de semillas), las difíciles primero:")
    log(tabla.to_string())
    bajo = tabla[tabla.sistema_completo < 0.99].index.tolist()
    log(f"  Familias con F1 medio < 0,99 en (5): {bajo if bajo else 'ninguna'}")

    # (D) tabla de consulta por extensión, en muestra, semilla 0
    Xb0, Xe0, Xr0, y0, nombres0, tipos0 = muestras[semillas[0]]
    ext0 = [extension_final(n).lower() for n in nombres0]
    mayoria = {}
    for e, fam in zip(ext0, y0):
        mayoria.setdefault(e, Counter())[fam] += 1
    lookup = float(np.mean([mayoria[e].most_common(1)[0][0] == fam for e, fam in zip(ext0, y0)]))
    log(f"\n  Tabla de consulta por extensión (en muestra, semilla {semillas[0]}): exactitud {lookup:.4f} (Exp. 2d: 0,9724)")

    # ------------------------------------------------------------------ (C) tipos no vistos
    log("\n" + "=" * 78 + "\n  (C) DEJAR-UN-TIPO-FUERA con ponderación de clases\n" + "=" * 78)
    filas_t = []
    for s in semillas:
        Xb, Xe, Xr, y, nombres, tipos = muestras[s]
        cuenta = Counter(tipos)
        candidatos = sorted(t for t, n in cuenta.items() if n >= min_tipo and t != "desconocido")
        cols = ["5", "2", "8"] if s == semillas[0] else ["5"]
        for tp in candidatos:
            te, tr = np.flatnonzero(tipos == tp), np.flatnonzero(tipos != tp)
            fams = np.unique(y[te])
            if len(fams) < min_fams:
                continue
            fila = dict(semilla=s, tipo=tp, n_prueba=len(te), familias_prueba=len(fams))
            for col in cols:
                X = matriz(col, Xb, Xe, Xr)
                mdl = RandomForestClassifier(random_state=42, n_jobs=N_JOBS, class_weight="balanced", **hiper)
                mdl.fit(X[tr], y[tr])
                yp = mdl.predict(X[te])
                fila[f"f1_{col}"] = round(f1_score(y[te], yp, average="macro", labels=fams, zero_division=0), 4)
                fila[f"exact_{col}"] = round(accuracy_score(y[te], yp), 4)
            filas_t.append(fila)
            log(f"  semilla {s} · {tp:<6} n {len(te):>5} fam {len(fams):>2}  " +
                "  ".join(f"({c}) {fila[f'f1_{c}']:.4f}" for c in cols))
            pd.DataFrame(filas_t).to_csv(out / "tipos_por_pliegue.csv", index=False)
    dft = pd.DataFrame(filas_t)
    prom5 = dft.groupby("semilla").f1_5.mean()
    log(f"\n  (5) promedio por semilla: " + " · ".join(f"{v:.4f}" for v in prom5) +
        f"  → {prom5.mean():.4f} ± {prom5.std(ddof=1) if len(prom5) > 1 else 0:.4f} (publicado 0,9983 ± 0,0011)")
    log(f"  (5) pliegue mínimo: {dft.f1_5.min():.4f} ({dft.loc[dft.f1_5.idxmin(), 'tipo']}, semilla {dft.loc[dft.f1_5.idxmin(), 'semilla']})")
    s0 = dft[dft.semilla == semillas[0]]
    if "f1_2" in s0:
        log(f"  semilla {semillas[0]}: (2) promedio {s0.f1_2.mean():.4f} (publicado 0,8814) · (8) promedio {s0.f1_8.mean():.4f} (publicado 0,9990)")

    # ------------------------------------------------------------------ veredicto
    log("\n" + "-" * 78 + "\n  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)\n" + "-" * 78)
    v = {}
    f1s = df[df.columna == "1"]
    if len(f1s):
        v["R1"] = veredicto(log, abs(f1s.exactitud.iloc[0] - REF["cv_1"][0]) <= 0.005 and
                            abs(f1s.macro_f1.iloc[0] - REF["cv_1"][1]) <= 0.005, "R1",
                            "(1) bytes semilla 0 a ±0,005 de 0,9123 / 0,9114",
                            f"{f1s.exactitud.iloc[0]:.4f} / {f1s.macro_f1.iloc[0]:.4f}")
    m2 = res.loc["2", ("macro_f1", "mean")]
    v["R2"] = veredicto(log, abs(m2 - REF["cv_2"]) <= 0.005, "R2", "(2) macro-F1 a ±0,005 de 0,9359", f"{m2:.4f}")
    m5 = res.loc["5", ("macro_f1", "mean")]
    v["R3"] = veredicto(log, m5 >= 0.999 and not bajo, "R3", "(5) macro-F1 >= 0,9990 y 30 familias >= 0,99",
                        f"{m5:.4f}; bajo 0,99: {bajo if bajo else 'ninguna'}")
    m8 = res.loc["8", ("macro_f1", "mean")] if "8" in res.index else float("nan")
    m6 = res.loc["6", ("macro_f1", "mean")] if "6" in res.index else float("nan")
    v["R4"] = veredicto(log, m8 >= 0.999 and 0.85 <= m6 <= 0.90, "R4", "(8) >= 0,9990; (6) entre 0,85 y 0,90",
                        f"(8) {m8:.4f} · (6) {m6:.4f}")
    v["R5"] = veredicto(log, bool((d52 >= 0.060).all()), "R5", "Δ (5)-(2) >= +0,060 en todas las semillas",
                        " ".join(f"{x:+.4f}" for x in d52))
    v["T1"] = veredicto(log, prom5.mean() >= 0.995 and dft.f1_5.min() >= 0.975, "T1",
                        "(5) tipos: promedio >= 0,995 y mínimo >= 0,975",
                        f"{prom5.mean():.4f} / mín {dft.f1_5.min():.4f}")
    if "f1_2" in s0:
        v["T2"] = veredicto(log, abs(s0.f1_2.mean() - REF["tipos_2"]) <= 0.010 and s0.f1_8.mean() >= 0.995, "T2",
                            "(2) tipos a ±0,010 de 0,8814; (8) >= 0,995",
                            f"{s0.f1_2.mean():.4f} / {s0.f1_8.mean():.4f}")
    dff = pd.DataFrame(fuga)
    v["F1"] = veredicto(log, bool((dff.grupos_partidos_entre_pliegues <= 3).all()), "F1",
                        "duplicados repartidos entre pliegues <= 3 por semilla",
                        " ".join(str(x) for x in dff.grupos_partidos_entre_pliegues))
    v["L1"] = veredicto(log, 0.96 <= lookup <= 0.98, "L1", "tabla de consulta por extensión entre 0,96 y 0,98", f"{lookup:.4f}")
    cumplen = sum(v.values()) + sum(resultados.get("preregistro_censo", {}).values())
    total_p = len(v) + len(resultados.get("preregistro_censo", {}))
    log(f"\n  {cumplen} de {total_p} predicciones cumplen.")
    log(f"  NIVEL: {'RESULTS REPRODUCED (cifra de cabecera obtenida por reimplementación independiente)' if v['R3'] else 'NO REPRODUCIDA: ver la diferencia en (5)'}")

    (out / "manifiesto.json").write_text(json.dumps(dict(
        fecha=str(date.today()), raiz=str(raiz), por_familia=args.por_familia, semillas=semillas,
        folds=args.folds, hiper=hiper, n_jobs=N_JOBS, sintetico=args.sintetico,
        versiones=dict(python=sys.version.split()[0], sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        referencias=REF, preregistro=dict(**resultados.get("preregistro_censo", {}), **v),
        cv_resumen={c: dict(exactitud=float(res.loc[c, ("exactitud", "mean")]), macro_f1=float(res.loc[c, ("macro_f1", "mean")]))
                    for c in res.index},
        tipos_5=dict(promedio=float(prom5.mean()), minimo=float(dft.f1_5.min())),
        lookup_extension=lookup,
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n  Salidas en: {out} · fin {datetime.now():%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
