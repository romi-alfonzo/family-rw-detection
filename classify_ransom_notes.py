"""
Ransomware Family Classification via Ransom Notes (NLP)
=======================================================
Clasifica familias de ransomware usando el texto de las notas de rescate.

Enfoque:
  1. Lee notas de rescate (txt, html, htm, hta, rtf) de carpetas por familia
  2. Extrae features con TF-IDF (n-gramas de palabras y caracteres)
  3. Entrena y evalua multiples clasificadores con cross-validation
  4. Reporta accuracy, F1, confusion matrix y features más importantes

Uso:
  python classify_ransom_notes.py <directorio_con_carpetas_por_familia>

  Estructura esperada:
    directorio/
      lockbit/
        nota1.txt
        nota2.html
      conti/
        nota1.txt
      ...

  El script también puede combinar múltiples fuentes:
  python classify_ransom_notes.py fuente1/ fuente2/ fuente3/
"""

import os
import sys
import re
import csv
import numpy as np
from collections import Counter

# Intentar importar sklearn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.preprocessing import LabelEncoder
    import warnings
    warnings.filterwarnings('ignore')
except ImportError:
    print("ERROR: sklearn no instalado. Ejecutar: pip install scikit-learn")
    sys.exit(1)


# ============================================================
# Mapeo de nombres de carpetas a familias normalizadas
# (para unificar diferentes fuentes con nombres distintos)
# ============================================================
FAMILY_ALIASES = {
    # ThreatLabz naming -> NapierOne naming
    'avoslocker': 'AVOSLOCKER',
    'badrabbit': 'BADRABBIT', 'bad_rabbit': 'BADRABBIT',
    'blackbasta': 'BLACKBASTA', 'black_basta': 'BLACKBASTA',
    'blackcat': 'BLACKCAT', 'black_cat': 'BLACKCAT', 'alphv': 'BLACKCAT',
    'blackmatter': 'BLACKMATTER', 'black_matter': 'BLACKMATTER',
    'cerber': 'CERBER',
    'chimera': 'CHIMERA',
    'clop': 'CLOP', 'cl0p': 'CLOP',
    'conti': 'CONTI',
    'cryptolocker': 'CRYPTOLOCKER', 'crypto_locker': 'CRYPTOLOCKER',
    'cuba': 'CUBA',
    'darkside': 'DARKSIDE', 'dark_side': 'DARKSIDE',
    'dharma': 'DHARMA', 'crysis': 'DHARMA',
    'gandcrab': 'GANDCRAB',
    'hellokitty': 'HELLOKITTY', 'hello_kitty': 'HELLOKITTY', 'hellocrypt': 'HELLOKITTY',
    'jigsaw': 'JIGSAW',
    'lockbit': 'LOCKBIT', 'lockbit2': 'LOCKBIT', 'lockbit3': 'LOCKBIT',
    'lorenz': 'LORENZ',
    'maze': 'MAZE',
    'medusalocker': 'MEDUZALOCKER', 'meduzalocker': 'MEDUZALOCKER', 'medusa_locker': 'MEDUZALOCKER',
    'netwalker': 'NETWALKER', 'mailto': 'NETWALKER',
    'notpetya': 'NOTPETYA', 'petya': 'NOTPETYA',
    'phobos': 'PHOBOS',
    'ransomexx': 'RANSOMEXX', 'ransom_exx': 'RANSOMEXX',
    'ryuk': 'RYUK',
    'sodinokibi': 'SODINOKIBI', 'revil': 'SODINOKIBI',
    'suncrypt': 'SUNCRYPT',
    'teslacrypt': 'TESLACRYPT', 'tesla_crypt': 'TESLACRYPT',
    'wannacry': 'WANNACRY', 'wanna_cry': 'WANNACRY', 'wcry': 'WANNACRY',
    'wastedlocker': 'WASTEDLOCKER', 'wasted_locker': 'WASTEDLOCKER',
}

# Familias que nos interesan (las 30 de NapierOne)
TARGET_FAMILIES = set(FAMILY_ALIASES.values())

NOTE_EXTENSIONS = {'.txt', '.html', '.htm', '.hta', '.rtf', '.md'}


def clean_html(text):
    """Remove HTML tags from text."""
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#?\w+;', ' ', text)
    return text


def read_note(filepath):
    """Read and clean a ransom note file."""
    try:
        # Try multiple encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'ascii']:
            try:
                with open(filepath, 'r', encoding=encoding, errors='replace') as f:
                    text = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            return None

        # Clean HTML if needed
        if any(filepath.lower().endswith(ext) for ext in ['.html', '.htm', '.hta']):
            text = clean_html(text)

        # Basic cleaning
        text = re.sub(r'\s+', ' ', text).strip()

        # Skip very short notes (likely corrupted or empty)
        if len(text) < 20:
            return None

        return text

    except Exception as e:
        print(f"  Warning: Could not read {filepath}: {e}")
        return None


def normalize_family_name(folder_name):
    """Map folder name to normalized family name."""
    key = folder_name.lower().replace('-', '').replace('_', '').replace(' ', '')

    # Direct match
    if key in FAMILY_ALIASES:
        return FAMILY_ALIASES[key]

    # Partial match
    for alias, family in FAMILY_ALIASES.items():
        clean_alias = alias.replace('_', '')
        if clean_alias in key or key in clean_alias:
            return family

    return None  # Unknown family


def load_notes_from_directory(base_dir, target_families=None):
    """Load ransom notes from a directory structure."""
    notes = []
    labels = []
    sources = {}

    if not os.path.isdir(base_dir):
        print(f"  [SKIP] Not a directory: {base_dir}")
        return notes, labels, sources

    for item in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item)
        if not os.path.isdir(item_path):
            continue
        if item.startswith('.'):
            continue

        family = normalize_family_name(item)

        if target_families and family not in target_families:
            # Also check if we want ALL families (not just NapierOne ones)
            if family is None:
                family = item.upper()  # Use folder name as-is
            elif family not in target_families:
                continue

        if family is None:
            family = item.upper()

        # Read all note files in this directory (including subdirectories)
        family_notes = []
        for root, dirs, files in os.walk(item_path):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in NOTE_EXTENSIONS:
                    fpath = os.path.join(root, fname)
                    text = read_note(fpath)
                    if text:
                        family_notes.append(text)

        if family_notes:
            notes.extend(family_notes)
            labels.extend([family] * len(family_notes))
            sources[family] = sources.get(family, 0) + len(family_notes)

    return notes, labels, sources


def evaluate_classifiers(notes, labels, label_names=None):
    """Train and evaluate multiple classifiers."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = le.classes_

    print(f"\n{'='*70}")
    print(f"CLASIFICACIÓN DE FAMILIAS POR NOTAS DE RESCATE")
    print(f"{'='*70}")
    print(f"Total notas: {len(notes)}")
    print(f"Total familias: {len(classes)}")
    print(f"\nDistribución:")
    for cls_name in sorted(set(labels)):
        count = labels.count(cls_name) if isinstance(labels, list) else (np.array(labels) == cls_name).sum()
        print(f"  {cls_name}: {count} notas")

    # TF-IDF: Word n-grams + Character n-grams
    print(f"\n--- Extrayendo features TF-IDF ---")

    # Configuraciones de TF-IDF a probar
    tfidf_configs = {
        'Word 1-2gram': TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                                         max_features=5000, sublinear_tf=True,
                                         min_df=1, max_df=0.95),
        'Char 3-5gram': TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                                         max_features=5000, sublinear_tf=True,
                                         min_df=1, max_df=0.95),
        'Combined': 'combined',  # Will be handled specially
    }

    models = {
        'LinearSVC': LinearSVC(max_iter=5000, C=1.0),
        'LogisticReg': LogisticRegression(max_iter=2000, C=1.0),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'KNN-5': KNeighborsClassifier(n_neighbors=min(5, min(Counter(labels).values()))),
    }

    cv = StratifiedKFold(n_splits=min(5, min(Counter(labels).values())),
                         shuffle=True, random_state=42)

    print(f"\n{'TF-IDF Config':<20} {'Model':<18} {'Accuracy':>10} {'Std':>8}")
    print("-" * 60)

    best_acc = 0
    best_config = ""

    for tfidf_name, tfidf in tfidf_configs.items():
        if tfidf_name == 'Combined':
            # Combine word and char features
            from scipy.sparse import hstack
            tfidf_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                                          max_features=3000, sublinear_tf=True,
                                          min_df=1, max_df=0.95)
            tfidf_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                                          max_features=3000, sublinear_tf=True,
                                          min_df=1, max_df=0.95)
            X_word = tfidf_word.fit_transform(notes)
            X_char = tfidf_char.fit_transform(notes)
            X = hstack([X_word, X_char])

            for model_name, model in models.items():
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                acc = scores.mean()
                print(f"  {tfidf_name:<18} {model_name:<18} {acc:>8.3f}   {scores.std():>6.3f}")
                if acc > best_acc:
                    best_acc = acc
                    best_config = f"{tfidf_name} + {model_name}"
        else:
            X = tfidf.fit_transform(notes)
            for model_name, model in models.items():
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                acc = scores.mean()
                print(f"  {tfidf_name:<18} {model_name:<18} {acc:>8.3f}   {scores.std():>6.3f}")
                if acc > best_acc:
                    best_acc = acc
                    best_config = f"{tfidf_name} + {model_name}"
        print()

    print(f"\n  MEJOR: {best_config} -> {best_acc:.3f}")

    # Detailed report with best config
    print(f"\n{'='*70}")
    print(f"REPORTE DETALLADO ({best_config})")
    print(f"{'='*70}")

    # Use combined features for final report
    from scipy.sparse import hstack
    tfidf_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                                  max_features=3000, sublinear_tf=True)
    tfidf_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                                  max_features=3000, sublinear_tf=True)

    X_word = tfidf_word.fit_transform(notes)
    X_char = tfidf_char.fit_transform(notes)
    X = hstack([X_word, X_char])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    best_model = LinearSVC(max_iter=5000, C=1.0)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print(f"\n  Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))

    # Top features per family
    if hasattr(best_model, 'coef_'):
        print(f"\n{'='*70}")
        print(f"TOP 5 PALABRAS MÁS DISCRIMINATIVAS POR FAMILIA")
        print(f"{'='*70}")
        feature_names = (tfidf_word.get_feature_names_out().tolist() +
                        tfidf_char.get_feature_names_out().tolist())
        for i, cls_name in enumerate(classes):
            if i < best_model.coef_.shape[0]:
                top_indices = best_model.coef_[i].argsort()[-5:][::-1]
                top_words = [feature_names[j] for j in top_indices if j < len(feature_names)]
                print(f"  {cls_name}: {', '.join(top_words)}")

    return best_acc, best_config


def main():
    if len(sys.argv) < 2:
        print("Uso: python classify_ransom_notes.py <dir1> [dir2] [dir3] ...")
        print("")
        print("Ejemplo:")
        print("  python classify_ransom_notes.py ransomware_notes/")
        print("  python classify_ransom_notes.py ransomware_notes/ RansomNoteFiles/")
        print("")
        print("Cada directorio debe contener subcarpetas por familia.")
        print("Se pueden combinar múltiples fuentes.")
        sys.exit(1)

    directories = sys.argv[1:]

    all_notes = []
    all_labels = []
    all_sources = {}

    for directory in directories:
        print(f"\nCargando notas desde: {directory}")
        notes, labels, sources = load_notes_from_directory(directory, target_families=None)
        all_notes.extend(notes)
        all_labels.extend(labels)
        for k, v in sources.items():
            all_sources[k] = all_sources.get(k, 0) + v
        print(f"  Encontradas: {len(notes)} notas de {len(sources)} familias")

    if not all_notes:
        print("\nERROR: No se encontraron notas de rescate.")
        print("Verificá que las carpetas contienen archivos .txt/.html/.htm")
        sys.exit(1)

    # Filter families with at least 2 samples (needed for cross-validation)
    min_samples = 2
    family_counts = Counter(all_labels)
    valid_families = {f for f, c in family_counts.items() if c >= min_samples}
    filtered_notes = []
    filtered_labels = []
    skipped = {}
    for note, label in zip(all_notes, all_labels):
        if label in valid_families:
            filtered_notes.append(note)
            filtered_labels.append(label)
        else:
            skipped[label] = skipped.get(label, 0) + 1

    if skipped:
        print(f"\nFamilias con <{min_samples} muestras (excluidas):")
        for fam, count in sorted(skipped.items()):
            print(f"  {fam}: {count} nota(s)")

    print(f"\nDataset final: {len(filtered_notes)} notas, {len(set(filtered_labels))} familias")

    if len(set(filtered_labels)) < 2:
        print("ERROR: Se necesitan al menos 2 familias con 2+ muestras.")
        sys.exit(1)

    evaluate_classifiers(filtered_notes, filtered_labels)

    # Save results summary
    output_file = 'ransom_notes_results.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Family', 'Num_Notes', 'Source'])
        for family in sorted(all_sources.keys()):
            writer.writerow([family, all_sources[family], ','.join(directories)])

    print(f"\nResumen guardado en: {output_file}")


if __name__ == '__main__':
    main()
