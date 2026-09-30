#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cifras_finales_archivos.py -- las cifras del FRENTE DE ARCHIVOS, cada una contra su fuente.

PARA QUE SIRVE. Es el equivalente de `cifras_finales.py` (frente de notas). Cada número que el
capítulo 4 reporta sobre archivos cifrados sale de una corrida guardada en `4_resultados/`. Este
script lee la fuente de cada cifra y la compara con lo que dice la tesis. Si una cifra no coincide,
lo dice.

TRES DIFERENCIAS CON EL VERIFICADOR DE NOTAS (acordadas con esa sesión el 2026-09-29):
  1. Casi todas las fuentes son CSV: no se busca el texto, se LEE LA CELDA y se compara con los
     decimales que cita la tesis. Si la cifra solo está en un log, se busca con un ANCLA (el nombre
     de la métrica o de la corrida) y el valor tiene que aparecer en esa línea o a pocas líneas.
  2. Los AGREGADOS SE RECALCULAN desde el nivel más bajo guardado: el macro-F1 desde el F1 por familia,
     la media y el desvío (ddof=1) desde cada semilla, el promedio de pliegues desde cada pliegue. Una
     cifra que solo se puede leer del agregado que calculó el propio experimento (por ejemplo, la
     exactitud: no se guardaron predicciones por archivo) se marca "del script", no "recalculada".
  3. No comprueba que la cifra siga en el .tex: eso lo hace inventario_cifras.py (skill revisar-tesis)
     antes y después de cada pase de redacción.

CÓMO SE USA.  python cifras_finales_archivos.py            listado completo
              python cifras_finales_archivos.py --solo-fallas

BASES. Cada cifra declara la suya y no se mezclan: 29 o 30 familias; 5.800, 14.500 o 15.000
archivos; validación cruzada (CV) o dejar-un-tipo-fuera; 1, 5 o 10 semillas; job de origen.
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "4_resultados"
D = dict(
    b2=RES / "resultados_estructural_10semillas",
    c2=RES / "resultados_bytes_multisemilla" / "resultados_bytes_multisemilla_job3648",
    c2_busq=RES / "_logs_slurm_2026-08-17" / "slurm-bytes-3557.out",
    c2_an=RES / "resultados_analisis_bytes",
    c2_ven=RES / "resultados_ablacion_extendida",
    d2=RES / "resultados_exp2d" / "resultados_exp2d_job3937",
    e2=RES / "resultados_exp2e_job4058",
    e2r=RES / "resultados_exp2e_rasgo_job4059",
    f2=RES / "resultados_exp2f_job4083",
    g2=RES / "resultados_exp2g_job4091",
    h2=RES / "resultados_exp2h_job4096",
    dup=RES / "resultados_duplicados_en_cv",
    exp=RES / "resultados_experimentos",
    gs=RES / "resultados_gridsearch_estadisticas",
)
TIPOS = ("doc", "docx", "jpg", "pdf", "pptx", "xls", "xlsx")


# ============================================================ lectura
@functools.lru_cache(maxsize=None)
def csv(ruta) -> pd.DataFrame:
    return pd.read_csv(ruta)


@functools.lru_cache(maxsize=None)
def lineas(ruta) -> tuple:
    return tuple(Path(ruta).read_text(encoding="utf-8", errors="replace").splitlines())


def log(ruta, ancla, patron, grupo=1, dentro=6):
    """Valor que sigue a un ANCLA: la primera línea que contiene `ancla` y las `dentro` siguientes."""
    ls = lineas(ruta)
    for i, l in enumerate(ls):
        if ancla in l:
            for x in ls[i:i + dentro + 1]:
                m = re.search(patron, x)
                if m:
                    return float(m.group(grupo))
            raise KeyError(f"ancla «{ancla}» sin «{patron}» a {dentro} líneas")
    raise KeyError(f"ancla «{ancla}» no está en {Path(ruta).name}")


def ms(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


# ---- 2b: diez manifiestos del job 3651
@functools.lru_cache(maxsize=None)
def man_2b():
    ms_ = sorted(glob.glob(str(D["b2"] / "resultados_estructural_s*_job3651" / "manifiesto.json")))
    assert len(ms_) == 10, f"2b: se esperaban 10 semillas, hay {len(ms_)}"
    return tuple(json.loads(Path(m).read_text(encoding="utf-8"))["criterios"] for m in ms_)


def b2(criterio, modalidad, que):
    vals = []
    for c in man_2b():
        a = c[criterio]["ablacion"][modalidad]
        vals.append(dict(cobertura=a["cobertura"], exactitud=a["exactitud"],
                         aplica=a["aciertos"] / (a["total"] - a["sin_marca"]))[que])
    return ms(vals)


def b2_familias(criterio):
    v = [c[criterio]["familias_con_marca"] for c in man_2b()]
    v = [x if isinstance(x, (int, float)) else len(x) for x in v]
    return float(np.mean(v)), float(min(v)), float(max(v))


# ---- validación cruzada por semilla, con macro-F1 recalculado desde el F1 por familia
def cv_semillas(carpeta, archivo_sem, archivo_fam, columna, metrica):
    s = csv(carpeta / archivo_sem)
    s = s[s.columna.astype(str) == str(columna)]
    if metrica == "f1_rec":
        f = csv(carpeta / archivo_fam)
        f = f[f.columna.astype(str) == str(columna)]
        por = f.groupby("semilla").f1.mean()
        assert len(por) == len(s), f"{carpeta.name}: semillas por familia {len(por)} != {len(s)}"
        return ms(por.values)
    return ms(s[metrica].values)


def familia_cv(carpeta, columna, familia):
    f = csv(carpeta / "cv_por_familia_y_semilla.csv")
    f = f[(f.columna.astype(str) == str(columna)) & (f.familia == familia)]
    assert len(f) == 5, f"{carpeta.name}: {familia} tiene {len(f)} semillas"
    return float(f.f1.mean())


def familias_cv_todas(carpeta, columna):
    f = csv(carpeta / "cv_por_familia_y_semilla.csv")
    return f[f.columna.astype(str) == str(columna)].groupby("familia").f1.mean()


# ---- dejar-un-tipo-fuera con ponderación (2h, semilla 0)
def pliegue(tipo, col, que="f1"):
    p = csv(D["h2"] / "tipos_por_pliegue.csv")
    r = p[(p.semilla == 0) & (p.tipo == tipo)]
    assert len(r) == 1
    return float(r.iloc[0][f"{que}_{col}"])


def pliegue_rec(tipo, col):
    """macro-F1 del pliegue recalculado: media del F1 de las familias de la prueba."""
    f = csv(D["h2"] / "tipos_por_familia.csv")
    g = f[(f.semilla == 0) & (f.tipo == tipo) & (f.columna == col)]
    assert len(g) == 28, f"{tipo}/{col}: {len(g)} familias en la prueba"
    return float(g.f1.mean())


def promedio_tipos(col, que="f1", rec=True):
    if que == "f1" and rec:
        return float(np.mean([pliegue_rec(t, col) for t in TIPOS]))
    return float(np.mean([pliegue(t, col, que) for t in TIPOS]))


def cinco_semillas_tipos():
    s0 = promedio_tipos(5)
    resto = csv(D["h2"] / "tipos_semillas.csv").sort_values("semilla").f1_5.tolist()
    assert len(resto) == 4
    return [s0] + resto


def minimo_tipos():
    s0 = min(pliegue_rec(t, 5) for t in TIPOS)
    return min([s0] + csv(D["h2"] / "tipos_semillas.csv").minimo.tolist())


# ---- ablación del 2h
def abl_2h(col, que):
    s = csv(D["h2"] / "cv_ablacion_por_semilla.csv")
    s = s[s.columna == col]
    if que == "f1_rec":
        f = csv(D["h2"] / "cv_ablacion_por_familia.csv")
        por = f[f.columna == col].groupby("semilla").f1.mean()
        return ms(por.values)
    return ms(s[que].values)


# ---- censo del 2h
def censo():
    return csv(D["h2"] / "censo_por_archivo.csv")


def cfam(fam, col="total"):
    t = csv(D["h2"] / "censo_por_familia.csv").set_index("familia")
    return float(t.loc[fam, col])


def pares_dup():
    dup = csv(D["h2"] / "censo_duplicados.csv")
    return dup.groupby(["familia", "sha"]).nombre.apply(lambda s: tuple(sorted(s)))


def pares_origen():
    """Pares de documentos de origen: el número de base de NapierOne de cada copia (0066, 0067…)."""
    base = lambda n: re.match(r"(\d{4}-[a-z]+)", n).group(1)
    return {tuple(sorted(base(n) for n in p)) for p in pares_dup().values}


# ---- selectores por semilla (uno por fuente: con un solo nombre, una lambda tomaría el último)
def sel_2d(c, m="accuracy"):
    return ms(csv(D["d2"] / "exp2d_por_semilla.csv").query("columna == @c")[m])


def sel_curva(k, m="accuracy"):
    return ms(csv(D["d2"] / "a3_curva_por_semilla.csv").query("por_familia == @k")[m])


def sel_2e(c, m="accuracy"):
    return ms(csv(D["e2"] / "exp2e_por_semilla.csv").query("columna == @c")[m])


# ============================================================ las cifras
CIFRAS = []


def C(grupo, que, metrica, citado, valor, fuente, modo="igual", rec=False, tol_extra=0.0):
    """tol_extra: para una diferencia recalculada desde valores ya redondeados (0,0001 si son de 4
    decimales): el redondeo de las entradas se propaga, y sin ella la falla sería del verificador."""
    CIFRAS.append(dict(grupo=grupo, que=que, metrica=metrica, citado=citado, valor=valor,
                       fuente=fuente, modo=modo, rec=rec, tol_extra=tol_extra))


def M(grupo, que, metrica, citado_media, citado_desvio, par, fuente, rec=True, escala=1.0):
    """Una cifra «media ± desvío»: dos comprobaciones, desde el mismo par recalculado."""
    C(grupo, que, metrica, citado_media, lambda: par()[0] * escala, fuente, rec=rec)
    C(grupo, que, metrica + " (desvío)", citado_desvio, lambda: par()[1] * escala, fuente, rec=rec)


# ---------------- cabecera y síntesis (§4.11, tabla comparativa)
g = "Cabecera"
M(g, "2c solo bytes, 10 semillas (job 3648)", "exactitud", "0,912", "0,002",
  lambda: ms(csv(D["c2"] / "bytes_multisemilla.csv").accuracy), "bytes_multisemilla.csv")
M(g, "2c solo bytes, 10 semillas (job 3648)", "macro-F1", "0,911", "0,001",
  lambda: ms(csv(D["c2"] / "bytes_multisemilla.csv").f1_macro), "bytes_multisemilla.csv")
M(g, "2e bytes + estructura, 5 semillas (job 4058)", "exactitud", "0,9357", "0,0005",
  lambda: ms(csv(D["e2"] / "exp2e_por_semilla.csv").query("columna == '2_bytes_mas_estructura'").accuracy),
  "exp2e_por_semilla.csv")
M(g, "2e bytes + estructura, 5 semillas (job 4058)", "macro-F1", "0,9359", "0,0004",
  lambda: ms(csv(D["e2"] / "exp2e_por_semilla.csv").query("columna == '2_bytes_mas_estructura'").f1_macro),
  "exp2e_por_semilla.csv")
C(g, "2e bytes + estructura (síntesis, 3 decimales)", "macro-F1", "0,936",
  lambda: ms(csv(D["e2"] / "exp2e_por_semilla.csv").query("columna == '2_bytes_mas_estructura'").f1_macro)[0],
  "exp2e_por_semilla.csv", rec=True)
M(g, "2g sistema completo, CV 5 semillas (job 4091)", "macro-F1 recalculado desde el F1 por familia",
  "0,9998", "0,0001",
  lambda: cv_semillas(D["g2"], "cv_por_semilla.csv", "cv_por_familia_y_semilla.csv",
                      "5_bytes_estructura_extension", "f1_rec"), "cv_por_familia_y_semilla.csv")
M(g, "2g sistema completo, CV 5 semillas (job 4091)", "exactitud (del script)", "0,9998", "0,0001",
  lambda: cv_semillas(D["g2"], "cv_por_semilla.csv", None, "5_bytes_estructura_extension", "accuracy"),
  "cv_por_semilla.csv", rec=False)
M(g, "sistema completo bajo tipo no visto, 5 semillas (2h)", "macro-F1, promedio de 7 pliegues",
  "0,9983", "0,0011", lambda: ms(cinco_semillas_tipos()),
  "tipos_por_familia.csv (s0, recalculado) + tipos_semillas.csv (s1–4)")
C(g, "sistema completo bajo tipo no visto: todos los pliegues, 5 semillas", "macro-F1 mínimo",
  "0,98", minimo_tipos, "tipos_por_familia.csv + tipos_semillas.csv", modo="min_ge", rec=True)
M(g, "solo forma de la extensión, CV 5 semillas (2h, col. 6)", "macro-F1 recalculado", "0,8781", "0,0013",
  lambda: abl_2h(6, "f1_rec"), "cv_ablacion_por_familia.csv")
C(g, "solo forma de la extensión (síntesis, 3 decimales)", "macro-F1", "0,878",
  lambda: abl_2h(6, "f1_rec")[0], "cv_ablacion_por_familia.csv", rec=True)
M(g, "estructura + extensión, CV 5 semillas (2h, col. 7)", "macro-F1 recalculado", "0,9997", "0,0001",
  lambda: abl_2h(7, "f1_rec"), "cv_ablacion_por_familia.csv")
M(g, "bytes + extensión, CV 5 semillas (2h, col. 8)", "macro-F1 recalculado", "0,9999", "0,0001",
  lambda: abl_2h(8, "f1_rec"), "cv_ablacion_por_familia.csv")
C(g, "seis difíciles, F1 medio solo bytes (una semilla, job 4058)", "F1 medio", "0,5671",
  lambda: float(csv(D["e2"] / "por_familia_1_bytes_canonico.csv").set_index("Unnamed: 0")
                .loc[["WASTEDLOCKER", "JIGSAW", "DARKSIDE", "NOTPETYA", "SUNCRYPT", "CRYPTOLOCKER"],
                     "f1-score"].mean()), "por_familia_1_bytes_canonico.csv", rec=True)
C(g, "seis difíciles, F1 medio con estructura (una semilla, job 4058)", "F1 medio", "0,6857",
  lambda: float(csv(D["e2"] / "por_familia_2_bytes_mas_estructura.csv").set_index("Unnamed: 0")
                .loc[["WASTEDLOCKER", "JIGSAW", "DARKSIDE", "NOTPETYA", "SUNCRYPT", "CRYPTOLOCKER"],
                     "f1-score"].mean()), "por_familia_2_bytes_mas_estructura.csv", rec=True)
C(g, "difíciles bajo 0,75 con estructura, una semilla (4058)", "cantidad", "4",
  lambda: float((csv(D["e2"] / "por_familia_2_bytes_mas_estructura.csv").set_index("Unnamed: 0")
                 .loc[["WASTEDLOCKER", "JIGSAW", "DARKSIDE", "NOTPETYA", "SUNCRYPT", "CRYPTOLOCKER"],
                      "f1-score"] < 0.75).sum()), "por_familia_2_bytes_mas_estructura.csv", rec=True)
C(g, "difíciles bajo 0,75 con estructura, media de 5 semillas (4091)", "cantidad", "3",
  lambda: float((familias_cv_todas(D["g2"], "2_bytes_estructura")
                 .loc[["WASTEDLOCKER", "JIGSAW", "DARKSIDE", "NOTPETYA", "SUNCRYPT", "CRYPTOLOCKER"]]
                 < 0.75).sum()), "cv_por_familia_y_semilla.csv", rec=True)
C(g, "DARKSIDE con estructura, media de 5 semillas (4091)", "F1", "0,7515",
  lambda: familia_cv(D["g2"], "2_bytes_estructura", "DARKSIDE"), "cv_por_familia_y_semilla.csv", rec=True)
C(g, "sistema completo: la familia más baja, media de 5 semillas", "F1 mínimo", "0,99",
  lambda: float(familias_cv_todas(D["g2"], "5_bytes_estructura_extension").min()),
  "cv_por_familia_y_semilla.csv", modo="min_ge", rec=True)
C(g, "2d solo bytes, 5 semillas (job 3937)", "macro-F1", "0,9117",
  lambda: ms(csv(D["d2"] / "exp2d_por_semilla.csv").query("columna == '1_solo_bytes'").f1_macro)[0],
  "exp2d_por_semilla.csv", rec=True)
C(g, "2d bytes + forma del nombre, 5 semillas (job 3937)", "macro-F1", "0,9998",
  lambda: ms(csv(D["d2"] / "exp2d_por_semilla.csv").query("columna == '2_bytes_mas_forma_del_nombre'").f1_macro)[0],
  "exp2d_por_semilla.csv", rec=True)
C(g, "tabla comparativa: Exp. 1 binaria, Random Forest", "exactitud %", "88,6",
  lambda: float(csv(D["exp"] / "exp1_binaria.csv").set_index("Modelo").loc["Random Forest", "Mean_Accuracy_%"]),
  "exp1_binaria.csv")
C(g, "tabla comparativa: Exp. 2, 2 características, mejor modelo", "exactitud %", "9,9",
  lambda: float(csv(D["exp"] / "exp2_multiclase.csv")["Accuracy_%"].max()), "exp2_multiclase.csv")
C(g, "tabla comparativa: Exp. 2, rasgos estadísticos sin ajustar (29 fam.)", "exactitud %", "60,3",
  lambda: 100 * json.loads((D["gs"] / "gridsearch_estadisticas_manifiesto.json").read_text(encoding="utf-8"))
  ["referencia_sin_ajustar"], "gridsearch_estadisticas_manifiesto.json")
C(g, "tabla comparativa: 2c", "exactitud %", "91,2",
  lambda: 100 * ms(csv(D["c2"] / "bytes_multisemilla.csv").accuracy)[0], "bytes_multisemilla.csv", rec=True)
C(g, "tabla comparativa: 2e", "exactitud %", "93,6",
  lambda: 100 * ms(csv(D["e2"] / "exp2e_por_semilla.csv").query("columna == '2_bytes_mas_estructura'").accuracy)[0],
  "exp2e_por_semilla.csv", rec=True)
C(g, "tabla comparativa: 2g", "exactitud %", "99,98",
  lambda: 100 * cv_semillas(D["g2"], "cv_por_semilla.csv", None, "5_bytes_estructura_extension", "accuracy")[0],
  "cv_por_semilla.csv")

# ---------------- 2b: marcas estructurales (diez semillas, job 3651)
g = "2b · marcas"
for mod, nombre, cob, ex, ex_d, ap, ap_d in (
        ("solo_extension", "solo extensión", "86,7", "0,867", "0,000", "1,000", "0,000"),
        ("solo_firmas_binarias", "solo firmas binarias", "57,2", "0,563", "0,002", "0,984", "0,003"),
        ("combinado", "combinado", "94,1", "0,932", "0,001", "0,991", "0,002")):
    C(g, nombre + " (criterio 0,90)", "cobertura %", cob,
      (lambda m=mod: 100 * b2("umbral_90", m, "cobertura")[0]), "10 manifiestos", rec=True)
    M(g, nombre + " (criterio 0,90)", "exactitud global", ex, ex_d,
      (lambda m=mod: b2("umbral_90", m, "exactitud")), "10 manifiestos")
    M(g, nombre + " (criterio 0,90)", "exactitud donde aplica (aciertos / con marca)", ap, ap_d,
      (lambda m=mod: b2("umbral_90", m, "aplica")), "10 manifiestos")
C(g, "tabla comparativa: firmas donde aplican", "exactitud %", "98,4",
  lambda: 100 * b2("umbral_90", "solo_firmas_binarias", "aplica")[0], "10 manifiestos", rec=True)
C(g, "tabla comparativa: firmas", "cobertura %", "57",
  lambda: 100 * b2("umbral_90", "solo_firmas_binarias", "cobertura")[0], "10 manifiestos", rec=True)
for crit, nombre, fam, ex, ex_d, cob, cob_d in (
        ("unanimidad", "unanimidad byte a byte", "27,0", "0,900", "0,016", "0,909", "0,014"),
        ("umbral_90", "mayoría declarada en 0,90", "28,0", "0,932", "0,001", "0,941", "0,002")):
    C(g, nombre, "familias con marca (media de 10)", fam, (lambda c=crit: b2_familias(c)[0]),
      "10 manifiestos", rec=True)
    M(g, nombre + ", combinado", "exactitud", ex, ex_d, (lambda c=crit: b2(c, "combinado", "exactitud")),
      "10 manifiestos")
    M(g, nombre + ", combinado", "cobertura", cob, cob_d, (lambda c=crit: b2(c, "combinado", "cobertura")),
      "10 manifiestos")
C(g, "unanimidad: familias con marca, mínimo de 10", "cantidad", "26", lambda: b2_familias("unanimidad")[1],
  "10 manifiestos", rec=True)
C(g, "unanimidad: familias con marca, máximo de 10", "cantidad", "28", lambda: b2_familias("unanimidad")[2],
  "10 manifiestos", rec=True)

# ---------------- 2c
g = "2c · bytes"
for ancla, nombre, ex, f1 in (("[posicional + RandomForest] búsqueda", "posicional + RF (búsqueda)", "0,897", "0,897"),
                              ("[n-gramas de bytes + LogReg]", "n-gramas + Regresión Logística", "0,856", "0,858"),
                              ("[n-gramas de bytes + LinearSVC]", "n-gramas + LinearSVC", "0,849", "0,846")):
    C(g, nombre + " — 5.800 archivos, 29 familias (job 3557)", "exactitud", ex,
      (lambda a=ancla: log(D["c2_busq"], a, r"=> exactitud ([0-9.]+)", dentro=8)), "slurm-bytes-3557.out")
    C(g, nombre + " — 5.800 archivos, 29 familias (job 3557)", "macro-F1", f1,
      (lambda a=ancla: log(D["c2_busq"], a, r"macro-F1 ([0-9.]+) \|", dentro=8)), "slurm-bytes-3557.out")
C(g, "búsqueda: tamaño del conjunto (job 3557)", "archivos", "5800",
  lambda: log(D["c2_busq"], "Total:", r"Total: (\d+) archivos"), "slurm-bytes-3557.out")
C(g, "búsqueda: familias (job 3557)", "familias", "29",
  lambda: log(D["c2_busq"], "Total:", r"\| (\d+) familias"), "slurm-bytes-3557.out")
tabla_2c = {"doc": ("1991", "27", "0,889", "0,881"), "docx": ("1978", "27", "0,887", "0,877"),
            "xls": ("1948", "27", "0,889", "0,883"), "xlsx": ("1962", "27", "0,884", "0,872"),
            "pptx": ("1926", "27", "0,867", "0,864"), "pdf": ("1798", "25", "0,863", "0,836"),
            "jpg": ("2397", "28", "0,874", "0,811")}
for tipo, (n, nf, ex, f1) in tabla_2c.items():
    fila = lambda t=tipo: csv(D["c2_an"] / "a_generalizacion_tipos.csv").set_index("tipo_excluido").loc[t]
    C(g, f"tipos, 29 familias: {tipo}", "archivos de prueba", n, (lambda f=fila: float(f()["n_prueba"])),
      "a_generalizacion_tipos.csv")
    C(g, f"tipos, 29 familias: {tipo}", "familias en la prueba", nf, (lambda f=fila: float(f()["n_familias"])),
      "a_generalizacion_tipos.csv")
    C(g, f"tipos, 29 familias: {tipo}", "exactitud", ex, (lambda f=fila: float(f()["accuracy"])),
      "a_generalizacion_tipos.csv")
    C(g, f"tipos, 29 familias: {tipo}", "macro-F1", f1, (lambda f=fila: float(f()["f1_macro"])),
      "a_generalizacion_tipos.csv")
C(g, "tipos, 29 familias: promedio de 7 pliegues", "exactitud", "0,879",
  lambda: float(csv(D["c2_an"] / "a_generalizacion_tipos.csv").accuracy.mean()), "a_generalizacion_tipos.csv", rec=True)
C(g, "tipos, 29 familias: promedio de 7 pliegues", "macro-F1", "0,861",
  lambda: float(csv(D["c2_an"] / "a_generalizacion_tipos.csv").f1_macro.mean()), "a_generalizacion_tipos.csv", rec=True)
C(g, "importancia concentrada en la cola", "% de la importancia", "79,6",
  lambda: float(100 * csv(D["c2_an"] / "b_importancia_por_posicion.csv").groupby("region").importancia.sum()
                .pipe(lambda s: s["cola"] / s.sum())), "b_importancia_por_posicion.csv", rec=True)
dif = {"CRYPTOLOCKER": ("99,4", "7,59", "7,59"), "WASTEDLOCKER": ("99,2", "7,59", "7,59"),
       "JIGSAW": ("98,6", "7,51", "7,55"), "DARKSIDE": ("97,2", "7,59", "7,49"),
       "SUNCRYPT": ("98,2", "7,59", "4,78"), "NOTPETYA": ("98,4", "7,34", "6,58")}
for fam, (conf, hc, hk) in dif.items():
    fila = lambda f_=fam: csv(D["c2_an"] / "d_familias_dificiles.csv").set_index("familia").loc[f_]
    C(g, f"difíciles, 29 familias: {fam}", "confusión interna %", conf,
      (lambda f=fila: float(f()["pct_confusion_interna"])), "d_familias_dificiles.csv")
    C(g, f"difíciles, 29 familias: {fam}", "entropía de cabecera", hc,
      (lambda f=fila: float(f()["entropia_cabecera"])), "d_familias_dificiles.csv")
    C(g, f"difíciles, 29 familias: {fam}", "entropía de cola", hk,
      (lambda f=fila: float(f()["entropia_cola"])), "d_familias_dificiles.csv")
for clave, cit in (("entropia_resto_cabecera", "7,03"), ("entropia_resto_cola", "7,44")):
    C(g, "difíciles: resto de familias", clave, cit,
      (lambda k=clave: float(json.loads((D["c2_an"] / "analisis_manifiesto.json").read_text(encoding="utf-8"))
                             ["resultados"]["d_dificiles"][k])), "analisis_manifiesto.json")
ven = {"64+64": ("0,790", "0,794", "0,796"), "128+128": ("0,792", "0,795", "0,798"),
       "256+256": ("0,851", "0,856", "0,851"), "512+512": ("0,905", "0,910", "0,904"),
       "1024+1024": ("0,904", "0,908", "0,903"), "2048+2048": ("0,903", "0,908", "0,902"),
       "4096+4096": ("0,901", "0,906", "0,901")}
for v_, (f_t, f_s, a_t) in ven.items():
    t = lambda v=v_, sub="todos": csv(D["c2_ven"] / "a_curva_ablacion.csv").query("ventana == @v and subconjunto == @sub").iloc[0]
    C(g, f"ventana {v_}, 15.000", "macro-F1", f_t, (lambda v=v_: float(t(v)["f1_macro"])), "a_curva_ablacion.csv")
    C(g, f"ventana {v_}, sin relleno", "macro-F1", f_s, (lambda v=v_: float(t(v, "sin_relleno")["f1_macro"])),
      "a_curva_ablacion.csv")
    C(g, f"ventana {v_}, 15.000", "exactitud", a_t, (lambda v=v_: float(t(v)["accuracy"])), "a_curva_ablacion.csv")
C(g, "sesgo PDF: solo bytes sin los 310 PDF (2d, job 3937)", "exactitud", "0,9128",
  lambda: ms(csv(D["d2"] / "exp2d_por_semilla.csv").query("columna == '1_solo_bytes'").accuracy)[0],
  "exp2d_por_semilla.csv", rec=True)
C(g, "sesgo PDF: solo bytes con los 310 PDF (2e, job 4058)", "exactitud", "0,9123",
  lambda: ms(csv(D["e2"] / "exp2e_por_semilla.csv").query("columna == '1_bytes_canonico'").accuracy)[0],
  "exp2e_por_semilla.csv", rec=True)
C(g, "PDF cifrados de BADRABBIT y NOTPETYA (censo 2h)", "archivos", "310",
  lambda: cfam("BADRABBIT", "pdf") + cfam("NOTPETYA", "pdf"), "censo_por_familia.csv", rec=True)

# ---------------- 2d
g = "2d · nombre"
t2d = {"0a_solo_forma_del_nombre": ("0,5830", "0,0024", "0,5771", "0,0033"),
       "0b_solo_extension_literal": ("0,9380", "0,0011", "0,9244", "0,0036"),
       "1_solo_bytes": ("0,9128", "0,0012", "0,9117", "0,0011"),
       "2_bytes_mas_forma_del_nombre": ("0,9998", "0,0002", "0,9998", "0,0002"),
       "3_bytes_mas_extension_literal": ("0,9718", "0,0020", "0,9699", "0,0024")}
for col, (a, ad, f, fd) in t2d.items():
    M(g, f"{col}, 5 semillas", "exactitud", a, ad, (lambda c=col: sel_2d(c)), "exp2d_por_semilla.csv")
    M(g, f"{col}, 5 semillas", "macro-F1", f, fd, (lambda c=col: sel_2d(c, "f1_macro")), "exp2d_por_semilla.csv")
C(g, "tabla de consulta que solo mira la extensión", "exactitud", "0,9724",
  lambda: log(D["d2"] / "log.txt", "TABLA DE CONSULTA", r"EXTENSI[OÓ]N: ([0-9.]+)|: ([0-9.]+)\s*$", grupo=0)
  if False else log(D["d2"] / "log.txt", "TABLA DE CONSULTA", r":\s*([0-9.]+)\s*$"), "log.txt (3937)")
C(g, "familias con una sola extensión", "cantidad", "25",
  lambda: log(D["d2"] / "log.txt", "familias con UNA sola extensión", r": (\d+) de 30"), "log.txt (3937)")
C(g, "extensiones distintas", "cantidad", "905",
  lambda: log(D["d2"] / "log.txt", "extensiones distintas", r"(\d+) extensiones distintas"), "log.txt (3937)")
C(g, "extensiones compartidas por más de una familia", "cantidad", "5",
  lambda: log(D["d2"] / "log.txt", "extensiones compartidas", r": (\d+) de 905"), "log.txt (3937)")
curva = {10: ("0,8111", "0,0158", "0,7809", "0,0184"), 25: ("0,8627", "0,0074", "0,8464", "0,0086"),
         50: ("0,8809", "0,0105", "0,8723", "0,0114"), 100: ("0,8918", "0,0070", "0,8880", "0,0054"),
         200: ("0,9055", "0,0029", "0,9045", "0,0024"), 350: ("0,9102", "0,0008", "0,9096", "0,0009"),
         500: ("0,9124", "0,0005", "0,9117", "0,0005")}
for k, (a, ad, f, fd) in curva.items():
    M(g, f"curva: {k} archivos/familia, 3 semillas", "exactitud", a, ad, (lambda k_=k: sel_curva(k_)),
      "a3_curva_por_semilla.csv")
    M(g, f"curva: {k} archivos/familia, 3 semillas", "macro-F1", f, fd, (lambda k_=k: sel_curva(k_, "f1_macro")),
      "a3_curva_por_semilla.csv")

# ---------------- 2e
g = "2e · estructura"
t2e = {"1_bytes_canonico": ("1024", "0,9123", "0,0003", "0,9114", "0,0004"),
       "2_bytes_mas_estructura": ("1068", "0,9357", "0,0005", "0,9359", "0,0004"),
       "3_solo_estructura": ("44", "0,8699", "0,0030", "0,8680", "0,0032")}
for col, (nc, a, ad, f, fd) in t2e.items():
    C(g, f"{col}", "características", nc,
      (lambda c=col: float(csv(D["e2"] / "exp2e_por_semilla.csv").query("columna == @c").n_caracteristicas.iloc[0])),
      "exp2e_por_semilla.csv")
    M(g, f"{col}, 5 semillas", "exactitud", a, ad, (lambda c=col: sel_2e(c)), "exp2e_por_semilla.csv")
    M(g, f"{col}, 5 semillas", "macro-F1", f, fd, (lambda c=col: sel_2e(c, "f1_macro")), "exp2e_por_semilla.csv")
tdif = {"WASTEDLOCKER": ("0,6397", "0,8317", "+0,1920"), "JIGSAW": ("0,4285", "0,5711", "+0,1426"),
        "DARKSIDE": ("0,5982", "0,7327", "+0,1345"), "NOTPETYA": ("0,3614", "0,4839", "+0,1224"),
        "SUNCRYPT": ("0,7719", "0,8394", "+0,0675"), "CRYPTOLOCKER": ("0,6026", "0,6552", "+0,0526"),
        "BADRABBIT": ("0,9827", "1,0000", "+0,0173")}
pf = lambda arch, fam: float(csv(D["e2"] / arch).set_index("Unnamed: 0").loc[fam, "f1-score"])
for fam, (b, e, dd) in tdif.items():
    C(g, f"difíciles, una semilla: {fam}", "F1 solo bytes", b,
      (lambda f=fam: pf("por_familia_1_bytes_canonico.csv", f)), "por_familia_1_bytes_canonico.csv", rec=True)
    C(g, f"difíciles, una semilla: {fam}", "F1 con estructura", e,
      (lambda f=fam: pf("por_familia_2_bytes_mas_estructura.csv", f)), "por_familia_2_bytes_mas_estructura.csv", rec=True)
    C(g, f"difíciles, una semilla: {fam}", "Δ F1", dd,
      (lambda f=fam: pf("por_familia_2_bytes_mas_estructura.csv", f) - pf("por_familia_1_bytes_canonico.csv", f)),
      "por_familia_{1,2}_*.csv", rec=True)
base_f1 = lambda: log(D["e2r"] / "log.txt", "completo (44 rasgos)", r"macro-F1 ([0-9.]+)")
base_d = lambda: log(D["e2r"] / "log.txt", "completo (44 rasgos)", r"seis difíciles ([0-9.]+)")
C(g, "ablación: representación completa (una semilla, job 4059)", "macro-F1", "0,8716", base_f1, "log.txt (4059)")
C(g, "ablación: representación completa (una semilla, job 4059)", "F1 de las seis difíciles", "0,5704",
  base_d, "log.txt (4059)")
abl = {"tamaño": ("5", "-0,1227", "-0,2247"), "entropia_cabecera": ("8", "-0,0103", "-0,0498"),
       "entropia_medio": ("12", "-0,0082", "-0,0303"), "distribucion": ("9", "-0,0174", "-0,0233"),
       "entropia_cola": ("8", "-0,0566", "-0,0177"), "salto_cab_cola": ("1", "-0,0029", "-0,0092"),
       "pie_no_aleatorio": ("1", "+0,0001", "-0,0005")}
fa = lambda gr: csv(D["e2r"] / "ablacion_por_grupo.csv").set_index("grupo").loc[gr]
for gr, (n, cg, cd) in abl.items():
    C(g, f"ablación: sin {gr}", "rasgos", n, (lambda x=gr: float(fa(x)["n_rasgos"])), "ablacion_por_grupo.csv")
    C(g, f"ablación: sin {gr}", "caída global (del script)", cg, (lambda x=gr: float(fa(x)["caida"])),
      "ablacion_por_grupo.csv")
    C(g, f"ablación: sin {gr}", "caída global (recalculada: F1 − completo)", cg,
      (lambda x=gr: float(fa(x)["f1_macro"]) - base_f1()), "ablacion_por_grupo.csv + log.txt", rec=True,
      tol_extra=1e-4)
    C(g, f"ablación: sin {gr}", "caída en las seis (del script)", cd, (lambda x=gr: float(fa(x)["caida_dificiles"])),
      "ablacion_por_grupo.csv")
    C(g, f"ablación: sin {gr}", "caída en las seis (recalculada)", cd,
      (lambda x=gr: float(fa(x)["f1_dificiles"]) - base_d()), "ablacion_por_grupo.csv + log.txt", rec=True,
      tol_extra=1e-4)
C(g, "solo los 5 rasgos de tamaño", "macro-F1", "0,3272", lambda: float(fa("SOLO_tamaño")["f1_macro"]),
  "ablacion_por_grupo.csv")
C(g, "solo los 5 rasgos de tamaño", "F1 de las seis difíciles", "0,1151",
  lambda: float(fa("SOLO_tamaño")["f1_dificiles"]), "ablacion_por_grupo.csv")
C(g, "tamaño frente al segundo grupo, caída en las seis", "cociente", "4,5",
  lambda: float(fa("tamaño")["caida_dificiles"] / fa("entropia_cabecera")["caida_dificiles"]),
  "ablacion_por_grupo.csv", rec=True)
C(g, "resto del tamaño módulo 16 sobre el segundo rasgo", "% por encima", "28",
  lambda: float(100 * (csv(D["e2r"] / "importancias_rasgos.csv").importancia.nlargest(2).pipe(lambda s: s.iloc[0] / s.iloc[1]) - 1)),
  "importancias_rasgos.csv", rec=True)
C(g, "rasgo más informativo", "es tam_mod16 (1 = sí)", "1",
  lambda: float(csv(D["e2r"] / "importancias_rasgos.csv").sort_values("importancia", ascending=False).rasgo.iloc[0] == "tam_mod16"),
  "importancias_rasgos.csv", rec=True)
C(g, "puesto del grupo de tamaño en las importancias", "puesto", "4",
  lambda: float(list(csv(D["e2r"] / "importancias_rasgos.csv").groupby("grupo").importancia.sum()
                     .sort_values(ascending=False).index).index("tamaño") + 1), "importancias_rasgos.csv", rec=True)

# ---------------- 2e y 2g bajo tipo no visto, con ponderación (2h, semilla 0)
g = "tipos · 2h"
t_e = {"doc": ("2071", "0,9005", "0,8887", "0,9358", "0,9321"), "docx": ("2007", "0,8954", "0,8833", "0,9138", "0,9071"),
       "jpg": ("2401", "0,8538", "0,8419", "0,7326", "0,8155"), "pdf": ("1977", "0,8017", "0,7744", "0,8336", "0,8137"),
       "pptx": ("2056", "0,8862", "0,8716", "0,9056", "0,8935"), "xls": ("1974", "0,8936", "0,8830", "0,9184", "0,9046"),
       "xlsx": ("2007", "0,8989", "0,8887", "0,9103", "0,9031")}
for tipo, (n, a1, f1, a2, f2) in t_e.items():
    C(g, f"{tipo}: archivos de prueba", "archivos", n, (lambda t=tipo: pliegue(t, 1, "acc") * 0 + float(
        csv(D["h2"] / "tipos_por_pliegue.csv").query("semilla == 0 and tipo == @t").n.iloc[0])), "tipos_por_pliegue.csv")
    C(g, f"{tipo}: bytes", "exactitud (del script)", a1, (lambda t=tipo: pliegue(t, 1, "acc")), "tipos_por_pliegue.csv")
    C(g, f"{tipo}: bytes", "macro-F1 recalculado", f1, (lambda t=tipo: pliegue_rec(t, 1)), "tipos_por_familia.csv", rec=True)
    C(g, f"{tipo}: bytes + estructura", "exactitud (del script)", a2, (lambda t=tipo: pliegue(t, 2, "acc")), "tipos_por_pliegue.csv")
    C(g, f"{tipo}: bytes + estructura", "macro-F1 recalculado", f2, (lambda t=tipo: pliegue_rec(t, 2)), "tipos_por_familia.csv", rec=True)
for col, que, cit in ((1, "acc", "0,8757"), (1, "f1", "0,8617"), (2, "acc", "0,8786"), (2, "f1", "0,8814")):
    C(g, f"promedio de 7 pliegues, columna {col}", "exactitud" if que == "acc" else "macro-F1 recalculado", cit,
      (lambda c=col, q=que: promedio_tipos(c, q)), "tipos_por_pliegue/familia.csv", rec=True)
C(g, "promedio: Δ exactitud (2)−(1)", "Δ", "+0,0029", lambda: promedio_tipos(2, "acc") - promedio_tipos(1, "acc"),
  "tipos_por_pliegue.csv", rec=True)
C(g, "promedio: Δ macro-F1 (2)−(1)", "Δ", "+0,0197", lambda: promedio_tipos(2) - promedio_tipos(1),
  "tipos_por_familia.csv", rec=True)
for col, cit in ((1, "0,8621"), (2, "0,8859")):
    C(g, f"sin BLACKMATTER en jpg, columna {col}", "macro-F1 (del script: sin predicciones no se recalcula)", cit,
      (lambda c=col: promedio_tipos(c, "f1ok", rec=False)), "tipos_por_pliegue.csv (f1ok)")
C(g, "sin BLACKMATTER en jpg: Δ (2)−(1)", "Δ", "+0,0238",
  lambda: promedio_tipos(2, "f1ok", rec=False) - promedio_tipos(1, "f1ok", rec=False), "tipos_por_pliegue.csv (f1ok)")
C(g, "jpg, excluido BLACKMATTER: la estructura mejora a los bytes", "Δ f1ok > 0 (1 = sí)", "1",
  lambda: float(pliegue("jpg", 2, "f1ok") - pliegue("jpg", 1, "f1ok") > 0), "tipos_por_pliegue.csv (f1ok)")
t_g = {"doc": ("0,9321", "0,9947", "0,9943"), "docx": ("0,9071", "1,0000", "1,0000"), "jpg": ("0,8155", "0,4047", "1,0000"),
       "pdf": ("0,8137", "0,9834", "0,9800"), "pptx": ("0,8935", "1,0000", "1,0000"), "xls": ("0,9046", "0,9995", "0,9995"),
       "xlsx": ("0,9031", "1,0000", "1,0000")}
for tipo, (b, nom, ext) in t_g.items():
    C(g, f"{tipo}: + forma del nombre (2f)", "macro-F1 recalculado", nom, (lambda t=tipo: pliegue_rec(t, 3)),
      "tipos_por_familia.csv", rec=True)
    C(g, f"{tipo}: + forma de la extensión (2g)", "macro-F1 recalculado", ext, (lambda t=tipo: pliegue_rec(t, 5)),
      "tipos_por_familia.csv", rec=True)
for col, cit in ((3, "0,9118"), (5, "0,9963"), (6, "0,8630"), (7, "0,9980"), (8, "0,9990")):
    C(g, f"promedio de 7 pliegues, columna {col}", "macro-F1 recalculado", cit, (lambda c=col: promedio_tipos(c)),
      "tipos_por_familia.csv", rec=True)
C(g, "promedio: Δ nombre", "Δ", "+0,0304", lambda: promedio_tipos(3) - promedio_tipos(2), "tipos_por_familia.csv", rec=True)
C(g, "promedio: Δ extensión", "Δ", "+0,1149", lambda: promedio_tipos(5) - promedio_tipos(2), "tipos_por_familia.csv", rec=True)
C(g, "2f sin ponderación, jpg: bytes + estructura", "macro-F1", "0,8052",
  lambda: float(csv(D["f2"] / "tipos_por_pliegue.csv").set_index("tipo").loc["jpg", "f1_2_bytes_estructura"]),
  "tipos_por_pliegue.csv (4083)")
C(g, "2f sin ponderación, jpg: + forma del nombre", "macro-F1", "0,2167",
  lambda: float(csv(D["f2"] / "tipos_por_pliegue.csv").set_index("tipo").loc["jpg", "f1_3_bytes_estructura_forma"]),
  "tipos_por_pliegue.csv (4083)")
C(g, "2f sin ponderación, jpg: caída", "Δ", "-0,5885",
  lambda: float(csv(D["f2"] / "tipos_por_pliegue.csv").set_index("tipo").loc["jpg", "delta"]), "tipos_por_pliegue.csv (4083)")

# ---------------- 2f y 2g en validación cruzada
g = "2f · 2g · CV"
t2f = {"1_bytes": ("0,9123", "0,0003", "0,9114", "0,0004"), "2_bytes_estructura": ("0,9357", "0,0005", "0,9359", "0,0004"),
       "3_bytes_estructura_forma": ("0,9998", "0,0001", "0,9998", "0,0001"),
       "4_todo_con_extension": ("0,9998", "0,0001", "0,9998", "0,0001")}
for col, (a, ad, f, fd) in t2f.items():
    M(g, f"2f {col}, 5 semillas", "exactitud (del script)", a, ad,
      (lambda c=col: cv_semillas(D["f2"], "cv_por_semilla.csv", None, c, "accuracy")), "cv_por_semilla.csv (4083)", rec=False)
    M(g, f"2f {col}, 5 semillas", "macro-F1 recalculado desde el F1 por familia", f, fd,
      (lambda c=col: cv_semillas(D["f2"], "cv_por_semilla.csv", "cv_por_familia_y_semilla.csv", c, "f1_rec")),
      "cv_por_familia_y_semilla.csv (4083)")
fams = {"NOTPETYA": ("0,3850", "0,4811", "0,9978"), "JIGSAW": ("0,4338", "0,5739", "0,9990"),
        "CRYPTOLOCKER": ("0,6128", "0,6576", "1,0000"), "DARKSIDE": ("0,5981", "0,7515", "1,0000"),
        "WASTEDLOCKER": ("0,6266", "0,8352", "1,0000"), "SUNCRYPT": ("0,7590", "0,8296", "1,0000")}
for fam, (b, e, x) in fams.items():
    C(g, f"familia {fam}, 5 semillas: bytes (2f)", "F1", b, (lambda f=fam: familia_cv(D["f2"], "1_bytes", f)),
      "cv_por_familia_y_semilla.csv (4083)", rec=True)
    C(g, f"familia {fam}, 5 semillas: + estructura (2g)", "F1", e,
      (lambda f=fam: familia_cv(D["g2"], "2_bytes_estructura", f)), "cv_por_familia_y_semilla.csv (4091)", rec=True)
    C(g, f"familia {fam}, 5 semillas: + extensión (2g)", "F1", x,
      (lambda f=fam: familia_cv(D["g2"], "5_bytes_estructura_extension", f)), "cv_por_familia_y_semilla.csv (4091)", rec=True)
C(g, "las otras 24 familias en el sistema completo", "F1 mínimo", "0,99",
  lambda: float(familias_cv_todas(D["g2"], "5_bytes_estructura_extension").drop(list(fams)).min()),
  "cv_por_familia_y_semilla.csv (4091)", modo="min_ge", rec=True)

# ---------------- 2h: censo, duplicados, preregistro
g = "2h · censo"
C(g, "muestras del conjunto corregido", "archivos", "29948", lambda: float(len(censo())), "censo_por_archivo.csv", rec=True)
C(g, "derivación: 29.676 + 310 + 2 − 40", "archivos", "29948", lambda: 29676 + 310 + 2 - 40.0, "aritmética")
for fam, cit in (("NOTPETYA", "968"), ("JIGSAW", "992"), ("CERBER", "988"), ("DARKSIDE", "1000"), ("BLACKMATTER", "1000")):
    C(g, f"{fam}", "muestras", cit, (lambda f=fam: float((censo().familia == f).sum())), "censo_por_archivo.csv", rec=True)
C(g, "familias con 1.000 muestras repartidas por igual en los 7 tipos", "familias", "25",
  lambda: float(sum(1 for f, gr in censo().groupby("familia")
                    if len(gr) == 1000 and set(gr.tipo.value_counts().reindex(TIPOS).fillna(0)) <= {142, 143})),
  "censo_por_archivo.csv", rec=True)
C(g, "BLACKMATTER", "jpg", "988", lambda: float(((censo().familia == "BLACKMATTER") & (censo().tipo == "jpg")).sum()),
  "censo_por_archivo.csv", rec=True)
C(g, "BLACKMATTER", "sin tipo legible", "12",
  lambda: float(((censo().familia == "BLACKMATTER") & (censo().tipo == "sin_tipo")).sum()), "censo_por_archivo.csv", rec=True)
C(g, "CERBER con la cabecera del documento en claro", "archivos", "988",
  lambda: float(((censo().familia == "CERBER") & censo().en_claro.notna()).sum()), "censo_por_archivo.csv", rec=True)
C(g, "NOTPETYA", "jpg", "0", lambda: float(((censo().familia == "NOTPETYA") & (censo().tipo == "jpg")).sum()),
  "censo_por_archivo.csv", rec=True)
C(g, "JIGSAW sin la extensión .fun", "archivos", "2",
  lambda: float(((censo().familia == "JIGSAW") & (censo().ext != "fun")).sum()), "censo_por_archivo.csv", rec=True)
C(g, "JIGSAW en claro", "archivos", "2", lambda: float(((censo().familia == "JIGSAW") & censo().en_claro.notna()).sum()),
  "censo_por_archivo.csv", rec=True)
C(g, "familias con una sola extensión", "familias", "25", lambda: float((censo().groupby("familia").ext.nunique() == 1).sum()),
  "censo_por_archivo.csv", rec=True)
C(g, "SUNCRYPT: una extensión distinta por archivo (1 = sí)", "sí/no", "1",
  lambda: float(censo().query("familia == 'SUNCRYPT'").ext.nunique() == (censo().familia == "SUNCRYPT").sum()),
  "censo_por_archivo.csv", rec=True)
C(g, "MAZE", "extensiones distintas", "573", lambda: float(censo().query("familia == 'MAZE'").ext.nunique()),
  "censo_por_archivo.csv", rec=True)
C(g, "extensiones del conjunto", "distintas", "1606", lambda: float(censo().ext.nunique()), "censo_por_archivo.csv", rec=True)
C(g, "extensiones en más de una familia", "cantidad", "6",
  lambda: float((censo().groupby("ext").familia.nunique() > 1).sum()), "censo_por_archivo.csv", rec=True)
C(g, "extensiones en más de una familia: todas de documento (1 = sí)", "sí/no", "1",
  lambda: float(censo().groupby("ext").familia.nunique().pipe(lambda s: set(s[s > 1].index) <= set(TIPOS))),
  "censo_por_archivo.csv", rec=True)
g = "2h · duplicados"
C(g, "pares de archivos idénticos (SHA-256)", "pares", "16", lambda: float(len(pares_dup())), "censo_duplicados.csv", rec=True)
C(g, "pares entre familias distintas", "pares", "0",
  lambda: float(csv(D["h2"] / "censo_duplicados.csv").groupby("sha").familia.nunique().gt(1).sum()),
  "censo_duplicados.csv", rec=True)
C(g, "pares de documentos de origen", "pares", "3", lambda: float(len(pares_origen())), "censo_duplicados.csv", rec=True)
C(g, "familias con duplicados", "familias", "6",
  lambda: float(pares_dup().index.get_level_values(0).nunique()), "censo_duplicados.csv", rec=True)
C(g, "NOTPETYA", "pares", "1", lambda: float((pares_dup().index.get_level_values(0) == "NOTPETYA").sum()),
  "censo_duplicados.csv", rec=True)
C(g, "las otras cinco familias, cada una", "pares (mínimo)", "3",
  lambda: float(pares_dup().groupby(level=0).size().drop("NOTPETYA").min()), "censo_duplicados.csv", rec=True)
C(g, "pares repartidos entre entrenamiento y prueba, máximo por semilla (tesis: «a lo sumo uno o dos»)",
  "pares", "2", lambda: float(csv(D["dup"] / "duplicados_en_cv.csv").pares_repartidos.max()),
  "duplicados_en_cv.csv (réplica del muestreo)", modo="max_le", rec=True)
C(g, "efecto máximo sobre la exactitud (tesis: «del orden de una diezmilésima»)", "exactitud", "0,0001",
  lambda: float(csv(D["dup"] / "duplicados_en_cv.csv").efecto_max_exactitud.max()),
  "duplicados_en_cv.csv (réplica del muestreo)", modo="orden", rec=True)
g = "2h · preregistro"
C(g, "predicciones registradas", "cantidad", "16",
  lambda: float(len(json.loads((D["h2"] / "manifiesto.json").read_text(encoding="utf-8"))["preregistro"])),
  "manifiesto.json (4096)", rec=True)
C(g, "predicciones cumplidas", "cantidad", "13",
  lambda: float(sum(json.loads((D["h2"] / "manifiesto.json").read_text(encoding="utf-8"))["preregistro"].values())),
  "manifiesto.json (4096)", rec=True)
C(g, "predicciones fallidas", "cantidad", "3",
  lambda: float(sum(not v for v in json.loads((D["h2"] / "manifiesto.json").read_text(encoding="utf-8"))["preregistro"].values())),
  "manifiesto.json (4096)", rec=True)
C(g, "P3 en jpg: la estructura empeora a los bytes", "Δ macro-F1", "-0,0264",
  lambda: pliegue_rec("jpg", 2) - pliegue_rec("jpg", 1), "tipos_por_familia.csv", rec=True, tol_extra=1e-4)
C(g, "P4: pliegue pdf, semilla 0, sistema completo", "macro-F1", "0,980", lambda: pliegue_rec("pdf", 5),
  "tipos_por_familia.csv", rec=True)

# Cifras que NO salen de una corrida guardada: se citan con su fuente, no se verifican aquí.
SIN_CORRIDA = [
    ("Censo de integridad (tab:censo_integridad): 1.028 de 29.676, NOTPETYA 32/833, JIGSAW 8/998",
     "recuento del 16-08 sin los .pdf. Coherente con el censo del 2h: NOTPETYA 1.000 − 32 = 968 y JIGSAW 1.000 − 8 = 992 (verificados arriba)"),
    ("Herramientas: Crypto Sheriff 5/30, ID Ransomware 20/30 (archivos), 41/57 (notas), 30,0 % al renombrar",
     "7_compartido_carlos/Tesis Carlos y Romina/Pruebas.xlsx (pruebas manuales)"),
    ("Pruebas preliminares (350 y 1.600 archivos) y Exp. 2 ampliado (275 rasgos, 29 familias)",
     "corridas de agosto: 4_resultados/_logs_slurm_2026-08-17/ y _historico/; no incluidas en este verificador"),
    ("Composición de los pliegues del 2h (28 familias por pliegue; CERBER solo entrena; BLACKMATTER solo en jpg)",
     "tipos_por_pliegue.csv, columnas familias y fuera_de_la_prueba; control de entrada del propio 2h"),
]


# ============================================================ verificación
def decimales(s: str) -> int:
    s = s.replace("+", "").replace("-", "")
    return len(s.split(",")[1]) if "," in s else 0


def numero(s: str) -> float:
    return float(s.replace(".", "").replace(",", ".").replace("+", ""))


def verificar(c):
    v = c["valor"]()
    cit = numero(c["citado"])
    d = decimales(c["citado"])
    tol = 0.5 * 10 ** -d + 1e-9 + c["tol_extra"]
    if c["modo"] == "igual":
        ok = abs(v - cit) <= tol
    elif c["modo"] == "min_ge":
        ok = v >= cit - 1e-12
    elif c["modo"] == "max_le":
        ok = v <= cit + 1e-12
    elif c["modo"] == "orden":      # «del orden de»: dentro de un factor 3
        ok = cit / 3 <= v <= cit * 3
    else:
        raise ValueError(c["modo"])
    return v, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo-fallas", action="store_true")
    args = ap.parse_args()

    fallas, errores, grupo_actual = [], [], None
    n_rec = 0
    print("=" * 110)
    print("  CIFRAS DEL FRENTE DE ARCHIVOS, CONTRA SU FUENTE")
    print("=" * 110)
    for c in CIFRAS:
        try:
            v, ok = verificar(c)
        except Exception as e:                          # fuente ausente o mal leída: se informa
            errores.append((c, repr(e)))
            continue
        n_rec += c["rec"]
        if not ok:
            fallas.append((c, v))
        if args.solo_fallas and ok:
            continue
        if c["grupo"] != grupo_actual:
            grupo_actual = c["grupo"]
            print(f"\n  --- {grupo_actual}")
        marca = "ok" if ok else "FALLA"
        origen = "recalculada" if c["rec"] else "del script"
        print(f"   {c['que'][:64]:64} {c['metrica'][:30]:30} tesis {c['citado']:>8}  fuente {v:>10.4f}  "
              f"[{marca}] ({origen})")
    total = len(CIFRAS)
    print("\n" + "=" * 110)
    print(f"  VERIFICADAS: {total - len(fallas) - len(errores)} de {total}   ·   recalculadas desde un nivel "
          f"más bajo: {n_rec}   ·   fallas: {len(fallas)}   ·   fuentes que no se pudieron leer: {len(errores)}")
    if fallas:
        print("\n  FALLAS (la tesis dice una cosa y la fuente otra):")
        for c, v in fallas:
            print(f"   · {c['grupo']} · {c['que']} · {c['metrica']}: tesis {c['citado']}, fuente {v:.5f} ({c['fuente']})")
    if errores:
        print("\n  FUENTES QUE NO SE PUDIERON LEER:")
        for c, e in errores:
            print(f"   · {c['grupo']} · {c['que']} · {c['metrica']}: {e}")
    print("\n  CIFRAS QUE NO SALEN DE UNA CORRIDA GUARDADA -- se citan con su fuente y su aclaración")
    for que, fuente in SIN_CORRIDA:
        print(f"   · {que}\n       fuente: {fuente}")
    print("\n  Exactitud: sin predicciones por archivo guardadas, solo se lee del script (marcada así).")
    print("  Ningún número se cita sin su métrica y su base.")
    return 1 if (fallas or errores) else 0


if __name__ == "__main__":
    sys.exit(main())
