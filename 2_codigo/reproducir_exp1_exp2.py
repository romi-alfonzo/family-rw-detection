#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reproducir_exp1_exp2.py -- reproduce las tablas del Experimento 1 (detección binaria) y del
Experimento 2 (multiclase, 31 clases) a partir de 2_codigo/family-rw-detection/features.csv.

POR QUÉ EXISTE
--------------
El código que escribió 4_resultados/resultados_experimentos/exp1_binaria.csv, exp1_por_familia.csv y
exp2_multiclase.csv (2026-04-07) no se conservó: ningún script ni cuaderno del proyecto los escribe, y
el CSV de «Reporte 28-07» (cuaderno «Family vs Safe», partición única con 70 % de prueba) NO es su
fuente (0 de 30 exactitudes por familia coinciden). La revisión independiente del 2026-09-30 lo marcó
como hallazgo (A-04 + C-01). Este script reconstruye el procedimiento y lo deja en el repositorio.

LO QUE SE RECONSTRUYÓ (verificado el 2026-10-01 contra los CSV publicados)
------------------------------------------------------------------------
- Datos: features.csv, 1.600 archivos = 50 por familia (etiquetas 0-29) + 100 seguros (etiqueta 30),
  con DOS características: entropía de Shannon del archivo completo y tamaño. **No seis métricas.**
- Escalado: StandardScaler ajustado ANTES de la validación cruzada, sobre los 1.600 archivos en el
  Exp. 2 y sobre los 150 archivos de cada par familia-segura en el Exp. 1 (fuga menor: la media y el
  desvío del escalado ven la partición de prueba; no afecta a los árboles).
- Validación: StratifiedKFold(5, shuffle=True, random_state=42); exactitud media de los 5 pliegues.
- Exp. 1: cada familia contra la clase segura (50 + 100 = 150 archivos); media, mínimo y máximo de esas
  30 exactitudes por modelo. Exp. 2: las 31 clases juntas; media y desvío (ddof=0) entre pliegues.
- Modelos: DecisionTree, RandomForest y GradientBoosting con random_state=42; KNN con k=5; regresión
  logística (max_iter=1000); SVM lineal = LinearSVC.

Coincidencia con lo publicado, a un decimal (corrida del 2026-10-01, scikit-learn 1.6.1):
  Exp. 1: los seis modelos idénticos (media, mínimo y máximo); RF por familia: 30 de 30.
  Exp. 2: GB, DT, KNN y SVM idénticos (media y desvío); RF da 9,9 ± 0,9 contra 9,8 ± 0,9 publicado.

Uso:  python 2_codigo/reproducir_exp1_exp2.py   (≈ 1 minuto; solo lee, no escribe nada)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parent.parent
FEATURES = RAIZ / "2_codigo" / "family-rw-detection" / "features.csv"
PUBLICADOS = RAIZ / "4_resultados" / "resultados_experimentos"

# nombres tal como figuran en los CSV publicados
MODELOS = {
    "Random Forest": lambda: RandomForestClassifier(random_state=42),
    "Gradient Boost.": lambda: GradientBoostingClassifier(random_state=42),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
    "KNN (k=5)": lambda: KNeighborsClassifier(n_neighbors=5),
    "SVM Lineal": lambda: LinearSVC(),
    "Reg. Logística": lambda: LogisticRegression(max_iter=1000),
}
def escalar(X):
    """StandardScaler ajustado sobre el conjunto que se va a validar, ANTES de la validación cruzada:
    sobre los 1.600 archivos en el Exp. 2 y sobre los 150 de cada par familia-segura en el Exp. 1."""
    return StandardScaler().fit_transform(X)


def cv():
    return StratifiedKFold(5, shuffle=True, random_state=42)


def main():
    d = pd.read_csv(FEATURES)
    X_crudo = d[["Entropy", "Size"]].to_numpy(float)
    y = d["Label"].to_numpy()
    assert len(d) == 1600 and (pd.Series(y).value_counts().sort_index().tolist() == [50] * 30 + [100])

    print("EXPERIMENTO 2 -- multiclase, 31 clases, exactitud media ± desvío entre 5 pliegues (%)")
    X = escalar(X_crudo)
    filas2 = []
    for nombre, fab in MODELOS.items():
        s = cross_val_score(fab(), X, y, cv=cv()) * 100
        filas2.append((nombre, round(s.mean(), 1), round(s.std(), 1)))
    r2 = pd.DataFrame(filas2, columns=["Modelo", "Accuracy_%", "Std_%"])

    print("EXPERIMENTO 1 -- cada familia contra la clase segura, media/mínimo/máximo sobre 30 familias (%)")
    filas1, rf_fam = [], None
    for nombre, fab in MODELOS.items():
        v = []
        for k in range(30):
            sel = np.isin(y, [k, 30])
            v.append(cross_val_score(fab(), escalar(X_crudo[sel]), (y[sel] == k).astype(int), cv=cv()).mean() * 100)
        v = np.array(v)
        filas1.append((nombre, round(v.mean(), 1), round(v.min(), 1), round(v.max(), 1)))
        if nombre == "Random Forest":
            rf_fam = np.round(v, 1)
    r1 = pd.DataFrame(filas1, columns=["Modelo", "Mean_Accuracy_%", "Min_Accuracy_%", "Max_Accuracy_%"])

    pub1 = pd.read_csv(PUBLICADOS / "exp1_binaria.csv")
    pub2 = pd.read_csv(PUBLICADOS / "exp2_multiclase.csv")
    pubf = pd.read_csv(PUBLICADOS / "exp1_por_familia.csv").sort_values("Familia")

    def comparar(rep, pub, columnas):
        m = rep.merge(pub, on="Modelo", how="outer", suffixes=("_reproducido", "_publicado"))
        m["coincide"] = [all(f[f"{c}_reproducido"] == f[f"{c}_publicado"] for c in columnas)
                         for _, f in m.iterrows()]
        return m

    print("\n" + comparar(r2, pub2, ["Accuracy_%", "Std_%"]).to_string(index=False))
    print("\n" + comparar(r1, pub1, ["Mean_Accuracy_%", "Min_Accuracy_%", "Max_Accuracy_%"]).to_string(index=False))
    iguales = int((pubf["Accuracy_RF_%"].to_numpy() == rf_fam).sum())
    print(f"\nExactitud de Random Forest por familia (apéndice A.1): {iguales} de 30 idénticas a lo publicado.")
    print("Recordatorio: DOS características (entropía y tamaño), no seis; escalado ajustado antes de la validación.")


if __name__ == "__main__":
    main()
