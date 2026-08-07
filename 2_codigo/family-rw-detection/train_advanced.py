"""
Train and Evaluate Models with Advanced Features
=================================================
Carga el CSV generado por advanced_features.py y entrena
multiples modelos con cross-validation.

Uso:
  python train_advanced.py advanced_features.csv

Compara:
  1. Solo entropy + size (baseline, lo que teniamos)
  2. Features estadisticas (9 features)
  3. Byte frequency distribution (256 features)
  4. Todas las features combinadas (275 features)
  5. Features seleccionadas por importancia (top-K)
"""

import csv
import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Paralelismo controlado (corregido 2026-08-05 tras un OOM en el cluster del NIDTEC).
# Antes se usaba n_jobs=-1 en cross_val_score Y ADEMAS dentro de cada modelo, lo que
# genera paralelismo anidado: 32 procesos x 32 hilos, cada proceso con su copia de la
# matriz de features (~29.000 x 275). Con la memoria por defecto del cluster (2 GB) el
# sistema mata los workers. Ahora: el paralelismo va SOLO en la validacion cruzada,
# limitado a los nucleos que asigno SLURM, y los modelos corren con un hilo.
N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or -1


def load_features(filepath):
    """Load features from CSV file."""
    features = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            features.append([float(x) for x in row[:-1]])
            labels.append(int(row[-1]))
    return np.array(features), np.array(labels), header[:-1]


def evaluate_feature_sets(X, y, feature_names):
    """Evaluate different feature subsets."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define feature subsets
    subsets = {
        'Entropy+Size (baseline)': [0, 1],  # entropy_global, file_size
        'Statistical (9 feat)': list(range(9)),  # first 9 features
        'Byte Frequency (256 feat)': list(range(9, 265)),  # byte_freq_000 to byte_freq_255
        'All features (275 feat)': list(range(len(feature_names))),
        'Stats + Derived (19 feat)': list(range(9)) + list(range(265, 275)),
    }

    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1),
        'KNN-5': KNeighborsClassifier(n_neighbors=5),
        # HistGradientBoosting en lugar de GradientBoosting clasico: con 29 clases el
        # GradientBoosting original entrena n_clases x n_estimators = 2.900 arboles por
        # ajuste, lo que sobre 29.000 muestras es inviable (verificado localmente el
        # 2026-08-05: no termina). La documentacion de scikit-learn recomienda
        # HistGradientBoosting para n_samples > 10.000; es el mismo algoritmo con
        # histogramas de bins, ordenes de magnitud mas rapido. Declararlo en la tesis.
        'HistGradientBoosting': HistGradientBoostingClassifier(
            max_iter=100, random_state=42),
    }

    print("=" * 80)
    print("COMPARACIÓN DE FEATURE SETS Y MODELOS")
    print("=" * 80)
    print(f"{'Feature Set':<30} {'Model':<20} {'Accuracy':>10} {'Std':>8}")
    print("-" * 80)

    best_acc = 0
    best_config = ""

    import time
    for subset_name, indices in subsets.items():
        X_sub = X[:, indices]
        for model_name, model in models.items():
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model),
            ])
            # Aviso de progreso: este job puede tardar horas y conviene ver el avance
            # en el archivo de salida de SLURM mientras corre.
            print(f"  ... corriendo {subset_name} + {model_name}", flush=True)
            _t = time.time()
            scores = cross_val_score(pipe, X_sub, y, cv=cv, scoring='accuracy', n_jobs=N_JOBS)
            acc = scores.mean()
            std = scores.std()
            print(f"  {subset_name:<28} {model_name:<22} {acc:>8.3f}   {std:>6.3f}"
                  f"   [{time.time()-_t:.0f}s]", flush=True)
            if acc > best_acc:
                best_acc = acc
                best_config = f"{subset_name} + {model_name}"
        print()

    print(f"\n  MEJOR: {best_config} -> {best_acc:.3f}")

    # Feature selection: Top-K with SelectKBest
    print(f"\n{'='*80}")
    print("FEATURE SELECTION: Top-K features por importancia (ANOVA F-test)")
    print("=" * 80)

    for k in [10, 20, 50, 100]:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=min(k, X.shape[1]))),
            ('model', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)),
        ])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=N_JOBS)
        print(f"  Top-{k:3d} features + RF: {scores.mean():.3f} (+/- {scores.std():.3f})")

    return best_acc, best_config


def feature_importance_analysis(X, y, feature_names):
    """Show top most important features."""
    print(f"\n{'='*80}")
    print("TOP 20 FEATURES MÁS IMPORTANTES (Random Forest)")
    print("=" * 80)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf.fit(X_scaled, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    for i in range(min(20, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1:2d}. {feature_names[idx]:<25} importance: {importances[idx]:.4f}")


def detailed_report(X, y, feature_names):
    """Generate detailed classification report with best config."""
    print(f"\n{'='*80}")
    print("REPORTE DETALLADO (All features + RandomForest)")
    print("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f"\n  Accuracy: {np.mean(y_pred == y_test):.3f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


def main():
    if len(sys.argv) < 2:
        filepath = 'advanced_features.csv'
    else:
        filepath = sys.argv[1]

    print(f"Loading features from: {filepath}")
    X, y, feature_names = load_features(filepath)
    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"Samples per class: ~{X.shape[0] // len(np.unique(y))}")
    print()

    best_acc, best_config = evaluate_feature_sets(X, y, feature_names)
    feature_importance_analysis(X, y, feature_names)
    detailed_report(X, y, feature_names)

    print(f"\n{'='*80}")
    print("RESUMEN FINAL")
    print("=" * 80)
    print(f"""
  Con Entropy+Size (tu baseline):     ~10% accuracy
  Con features avanzadas ({X.shape[1]} feat): ~{best_acc:.0%} accuracy

  Si la accuracy con features avanzadas sigue siendo baja (<50%),
  eso confirma que los archivos cifrados por diferentes ransomware
  son estadisticamente muy similares, y NECESITAS complementar con
  datos adicionales como:
    - Notas de rescate (NLP/TF-IDF)
    - Extensiones de archivo
    - Metadata del cifrado

  Si la accuracy mejora significativamente (>60%), tenes un resultado
  publicable: que features de byte-frequency y entropia regional
  capturan diferencias sutiles entre algoritmos de cifrado.
""")


if __name__ == '__main__':
    main()
