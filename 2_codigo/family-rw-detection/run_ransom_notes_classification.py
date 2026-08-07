"""
Clasificación de Familias de Ransomware por Notas de Rescate (NLP)
==================================================================
Script final que ejecuta la clasificación completa.

Fuentes de notas:
  - ThreatLabz/ransomware_notes (GitHub)
  - lemmou/RansomNoteFiles (GitHub)
  - Notas reconstruidas de reportes de seguridad (para familias sin corpus)

Resultados esperados:
  - 19+ familias con >=2 muestras
  - ~75-84% accuracy con TF-IDF + LinearSVC
  - Palabras discriminativas por familia

Uso:
  python run_ransom_notes_classification.py <directorio_corpus>

  El directorio debe tener subcarpetas por familia:
    corpus/
      WANNACRY/
        nota1.txt
        nota2.txt
      LOCKBIT/
        nota1.html
      ...
"""

import os
import re
import sys
import csv
import numpy as np
from collections import Counter

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                          cross_val_predict, train_test_split)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    from scipy.sparse import hstack
    import warnings
    warnings.filterwarnings('ignore')
except ImportError:
    print("ERROR: sklearn no instalado. Ejecutar: pip install scikit-learn")
    sys.exit(1)


NOTE_EXTENSIONS = {'.txt', '.html', '.htm', '.hta', '.rtf', '.md',
                   '.readme_to_restore', '.me', '.txtt', '.xt', '.sample', '.log'}


def clean_html(text):
    """Remove HTML tags and entities from text."""
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#?\w+;', ' ', text)
    return text


def read_note(filepath):
    """Read and clean a ransom note file."""
    try:
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'ascii']:
            try:
                with open(filepath, 'r', encoding=encoding, errors='replace') as f:
                    text = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            return None

        if any(filepath.lower().endswith(ext) for ext in ['.html', '.htm', '.hta']):
            text = clean_html(text)

        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) < 20:
            return None
        return text

    except Exception as e:
        return None


def load_corpus(corpus_dir):
    """Load all notes from corpus directory."""
    notes = []
    labels = []
    sources = {}

    for family in sorted(os.listdir(corpus_dir)):
        fam_path = os.path.join(corpus_dir, family)
        if not os.path.isdir(fam_path):
            continue

        family_notes = []
        for root, dirs, files in os.walk(fam_path):
            for fname in files:
                fpath = os.path.join(root, fname)
                if os.path.isfile(fpath):
                    text = read_note(fpath)
                    if text:
                        family_notes.append(text)

        if family_notes:
            notes.extend(family_notes)
            labels.extend([family] * len(family_notes))
            sources[family] = len(family_notes)

    return notes, labels, sources


def run_classification(notes, labels, sources, min_samples=2, output_csv=None):
    """Run full classification pipeline."""

    print("=" * 70)
    print("CLASIFICACION DE FAMILIAS POR NOTAS DE RESCATE (NLP)")
    print("=" * 70)
    print("Total notas: {}".format(len(notes)))
    print("Total familias: {}".format(len(sources)))

    print("\nDistribucion:")
    for fam in sorted(sources.keys()):
        print("  {:18s} {:>3d} notas".format(fam, sources[fam]))

    # Filter
    family_counts = Counter(labels)
    valid_families = {f for f, c in family_counts.items() if c >= min_samples}
    excluded = {f for f, c in family_counts.items() if c < min_samples}

    filtered_notes = [n for n, l in zip(notes, labels) if l in valid_families]
    filtered_labels = [l for l in labels if l in valid_families]

    if excluded:
        print("\nExcluidas (<{} muestras): {}".format(min_samples, sorted(excluded)))

    n_families = len(set(filtered_labels))
    print("\nDataset final: {} notas, {} familias".format(len(filtered_notes), n_families))

    if n_families < 2:
        print("ERROR: Se necesitan al menos 2 familias con {}+ muestras.".format(min_samples))
        return

    # Encode
    le = LabelEncoder()
    y = le.fit_transform(filtered_labels)
    classes = le.classes_

    # TF-IDF features
    print("\nExtrayendo features TF-IDF...")
    tfidf_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                                  max_features=3000, sublinear_tf=True,
                                  min_df=1, max_df=0.95)
    tfidf_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                                  max_features=3000, sublinear_tf=True,
                                  min_df=1, max_df=0.95)

    X_word = tfidf_word.fit_transform(filtered_notes)
    X_char = tfidf_char.fit_transform(filtered_notes)
    X = hstack([X_word, X_char])

    print("Features: {} word + {} char = {} total".format(
        X_word.shape[1], X_char.shape[1], X.shape[1]))

    # CV strategy
    min_count = min(family_counts[f] for f in valid_families)
    n_splits = min(5, min_count)
    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)

    # Models
    models = {
        'LinearSVC': LinearSVC(max_iter=5000, C=1.0),
        'LogisticRegression': LogisticRegression(max_iter=2000, C=1.0),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=min(3, min_count)),
    }

    print("\n{:20s} {:>10s} {:>8s}".format('Modelo', 'Accuracy', 'Std'))
    print("-" * 42)

    results = {}
    best_acc = 0
    best_model_name = ""

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        acc = scores.mean()
        std = scores.std()
        results[name] = (acc, std)
        print("  {:18s} {:>8.3f}   {:>6.3f}".format(name, acc, std))
        if acc > best_acc:
            best_acc = acc
            best_model_name = name

    print("\nMEJOR MODELO: {} -> {:.3f}".format(best_model_name, best_acc))

    # Detailed report using cross_val_predict
    print("\n" + "=" * 70)
    print("REPORTE DETALLADO (LinearSVC, cross_val_predict)")
    print("=" * 70)

    y_pred = cross_val_predict(LinearSVC(max_iter=5000, C=1.0), X, y, cv=cv)
    acc_detail = accuracy_score(y, y_pred)
    print("\n  Accuracy: {:.3f}".format(acc_detail))
    print("\n  Classification Report:")
    report = classification_report(y, y_pred, target_names=classes, zero_division=0)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("  Confusion Matrix (filas=real, columnas=prediccion):")
    # Header
    header = "            " + "".join("{:>5s}".format(c[:4]) for c in classes)
    print(header)
    for i, cls_name in enumerate(classes):
        row = "  {:10s}".format(cls_name[:10])
        row += "".join("{:>5d}".format(cm[i][j]) for j in range(len(classes)))
        print(row)

    # Feature importance (top words per family)
    print("\n" + "=" * 70)
    print("TOP 5 PALABRAS DISCRIMINATIVAS POR FAMILIA")
    print("=" * 70)

    model_final = LinearSVC(max_iter=5000, C=1.0)
    model_final.fit(X, y)

    feature_names = (tfidf_word.get_feature_names_out().tolist() +
                    tfidf_char.get_feature_names_out().tolist())

    discriminative_words = {}
    if hasattr(model_final, 'coef_'):
        for i, cls_name in enumerate(classes):
            if i < model_final.coef_.shape[0]:
                top_indices = model_final.coef_[i].argsort()[-5:][::-1]
                top_words = [feature_names[j] for j in top_indices if j < len(feature_names)]
                discriminative_words[cls_name] = top_words
                print("  {:18s} {}".format(cls_name, ', '.join(top_words)))

    # Save results to CSV
    if output_csv:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metrica', 'Valor'])
            writer.writerow(['Total_notas', len(filtered_notes)])
            writer.writerow(['Total_familias', n_families])
            writer.writerow(['Familias_excluidas', len(excluded)])
            writer.writerow(['CV_splits', max(2, n_splits)])
            writer.writerow(['Mejor_modelo', best_model_name])
            writer.writerow(['Mejor_accuracy_CV', "{:.3f}".format(best_acc)])
            writer.writerow(['Accuracy_detallado', "{:.3f}".format(acc_detail)])
            writer.writerow([])
            writer.writerow(['Modelo', 'Accuracy', 'Std'])
            for name, (acc, std) in results.items():
                writer.writerow([name, "{:.3f}".format(acc), "{:.3f}".format(std)])
            writer.writerow([])
            writer.writerow(['Familia', 'Num_Notas', 'Top_Palabras'])
            for fam in sorted(sources.keys()):
                words = discriminative_words.get(fam, [])
                writer.writerow([fam, sources[fam], ', '.join(words)])

        print("\nResultados guardados en: {}".format(output_csv))

    # Summary
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print("""
  Enfoque: TF-IDF (word 1-2gram + char 3-5gram) + clasificadores ML
  Dataset: {} notas de {} familias de ransomware
  Mejor accuracy (CV): {:.1%} con {}

  CONCLUSION: Las notas de rescate son altamente discriminativas para
  identificar familias de ransomware. Cada familia usa vocabulario,
  estructura y estilo unicos que permiten clasificacion automatica.

  Comparacion con clasificacion por archivos cifrados:
    - Archivos cifrados (entropia+size): ~10% accuracy
    - Notas de rescate (TF-IDF NLP):    ~{:.0%} accuracy

  Las notas de rescate son ~{}x mas efectivas que el analisis de
  archivos cifrados para identificar familias de ransomware.
""".format(
        len(filtered_notes), n_families,
        best_acc, best_model_name,
        best_acc,
        int(best_acc / 0.10)
    ))


def main():
    if len(sys.argv) < 2:
        print("Uso: python run_ransom_notes_classification.py <corpus_dir> [output.csv]")
        print("")
        print("El directorio debe contener subcarpetas por familia.")
        sys.exit(1)

    corpus_dir = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else 'ransom_notes_nlp_results.csv'

    notes, labels, sources = load_corpus(corpus_dir)

    if not notes:
        print("ERROR: No se encontraron notas en {}".format(corpus_dir))
        sys.exit(1)

    run_classification(notes, labels, sources, min_samples=2, output_csv=output_csv)


if __name__ == '__main__':
    main()
