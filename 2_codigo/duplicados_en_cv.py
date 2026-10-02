#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
duplicados_en_cv.py -- ¿cuántos pares de archivos duplicados quedan repartidos entre
entrenamiento y prueba en la validación cruzada del frente de archivos? (2026-09-29)

POR QUÉ. La Sección del Exp. 2h dice que «en la validación cruzada a lo sumo uno o dos pares por
semilla quedan repartidos entre entrenamiento y prueba, del orden de una diezmilésima de
exactitud». Esa frase no sale de ninguna medición: el log del 2h no la mide. Una cuenta rápida la
contradice: si se toman 500 de cada 1.000 archivos y se reparten en cinco pliegues, lo esperable son
unos tres pares por semilla. Este script la MIDE sin clúster, reproduciendo el muestreo exacto.

CÓMO. El cargador del 2g y del 2h (`exp2g_nombre_robusto.cargar`) recorre las carpetas ordenadas,
los archivos ordenados de cada una sin el PDF descriptivo, y elige 500 por familia con UNA sola
secuencia `np.random.default_rng(semilla)`: `rng.permutation(len(arch))[:500]`, familia tras
familia. El censo del 2h (`censo_por_archivo.csv`) recorre exactamente igual y guarda los nombres
en ese orden. Con él se reproduce qué archivos entraron en cada semilla, y con
`StratifiedKFold(5, shuffle=True, random_state=semilla)`, que depende solo de las etiquetas, en qué
pliegue cayó cada uno.

CONTROL DE ENTRADA (aborta si falla). La réplica tiene que dar, para la semilla 0, exactamente los
recuentos de prueba y de entrenamiento por familia y tipo que el 2h guardó en
`tipos_por_familia.csv` (196 filas por columna). Si no coinciden, el muestreo no está reproducido
y el resultado no vale.

PREDICCIÓN (escrita antes de correr): entre 2 y 5 pares repartidos por semilla (esperanza ≈ 3,2),
es decir, la frase de la tesis es optimista pero la cota de efecto, una diezmilésima, se sostiene:
cada par repartido puede cambiar a lo sumo dos predicciones de 15.000.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parent.parent
H = RAIZ / "4_resultados" / "resultados_exp2h_job4096"
POR_FAMILIA = 500
SEMILLAS = (0, 1, 2, 3, 4)
TIPOS = ("doc", "docx", "jpg", "pdf", "pptx", "xls", "xlsx")


def muestra(censo: pd.DataFrame, semilla: int):
    """Réplica de exp2g_nombre_robusto.cargar: mismo orden, misma secuencia aleatoria."""
    rng = np.random.default_rng(semilla)
    nombres, familias, tipos = [], [], []
    for fam in dict.fromkeys(censo.familia):          # orden de aparición = orden de las carpetas
        g = censo[censo.familia == fam]
        arch = list(zip(g.nombre, g.tipo))           # ya ordenados por nombre, como sorted(iterdir())
        if len(arch) < 6:
            continue
        for i in rng.permutation(len(arch))[:POR_FAMILIA]:
            nombres.append(arch[i][0])
            familias.append(fam)
            tipos.append(arch[i][1])
    return nombres, np.array(familias), np.array(tipos)


def control_semilla0(familias, tipos):
    """Los recuentos de la réplica contra los que guardó el 2h para la semilla 0."""
    ref = pd.read_csv(H / "tipos_por_familia.csv")
    ref = ref[(ref.semilla == 0) & (ref.columna == 1)]
    tot = Counter(familias)
    por = Counter(zip(familias, tipos))
    malas = [(r.tipo, r.familia, r.n_prueba, por[(r.familia, r.tipo)], r.n_entrenamiento,
              tot[r.familia] - por[(r.familia, r.tipo)])
             for r in ref.itertuples()
             if r.n_prueba != por[(r.familia, r.tipo)] or r.n_entrenamiento != tot[r.familia] - por[(r.familia, r.tipo)]]
    return len(ref), malas


def main():
    censo = pd.read_csv(H / "censo_por_archivo.csv")
    dup = pd.read_csv(H / "censo_duplicados.csv")
    pares = []
    for (fam, sha), g in dup.groupby(["familia", "sha"]):
        assert len(g) == 2, f"grupo de {len(g)} archivos idénticos en {fam}: el script supone pares"
        a, b = g.nombre.tolist()
        pares.append((fam, a, b, g.tipo.iloc[0] == g.tipo.iloc[1]))
    print(f"Censo: {len(censo)} archivos · duplicados: {len(pares)} pares, "
          f"{sum(p[3] for p in pares)} con los dos del mismo tipo")

    nombres, familias, tipos = muestra(censo, 0)
    n_ref, malas = control_semilla0(familias, tipos)
    print(f"Control de la réplica (semilla 0): {n_ref - len(malas)} de {n_ref} recuentos coinciden "
          f"con tipos_por_familia.csv")
    if malas:
        for m in malas[:10]:
            print("   no coincide:", m)
        sys.exit("ABORTA: la réplica del muestreo no reproduce el 2h; el resultado no vale.")

    filas = []
    for s in SEMILLAS:
        nombres, familias, tipos = muestra(censo, s)
        pliegue = np.empty(len(nombres), dtype=int)
        for k, (_, prueba) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s)
                                        .split(np.zeros(len(familias)), familias)):
            pliegue[prueba] = k
        donde = {(f, n): pliegue[i] for i, (f, n) in enumerate(zip(familias, nombres))}
        ambos = [(f, a, b) for f, a, b, _ in pares if (f, a) in donde and (f, b) in donde]
        repartidos = [(f, a, b) for f, a, b in ambos if donde[(f, a)] != donde[(f, b)]]
        filas.append(dict(semilla=s, archivos=len(nombres), pares_en_la_muestra=len(ambos),
                          pares_repartidos=len(repartidos),
                          predicciones_afectadas_max=2 * len(repartidos),
                          efecto_max_exactitud=round(2 * len(repartidos) / len(nombres), 5),
                          familias=";".join(sorted({f for f, _, _ in repartidos}))))
    df = pd.DataFrame(filas)
    print()
    print(df.to_string(index=False))
    print(f"\nPares repartidos por semilla: entre {df.pares_repartidos.min()} y "
          f"{df.pares_repartidos.max()} (media {df.pares_repartidos.mean():.1f}); efecto máximo sobre "
          f"la exactitud: {df.efecto_max_exactitud.max():.5f}")
    salida = RAIZ / "4_resultados" / "resultados_duplicados_en_cv"
    salida.mkdir(exist_ok=True)
    df.to_csv(salida / "duplicados_en_cv.csv", index=False)
    print(f"Salida: {salida / 'duplicados_en_cv.csv'}")


if __name__ == "__main__":
    main()
