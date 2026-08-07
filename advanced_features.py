"""
Advanced Feature Extraction for Ransomware Family Classification
================================================================
Extrae features mucho mas ricas que solo entropia + tamaño.

Features extraidas por archivo:
  1. Entropia global (Shannon)
  2. Tamaño del archivo
  3. Entropia del header (primeros 1024 bytes)
  4. Entropia del footer (ultimos 1024 bytes)
  5. Entropia del medio (bytes centrales 1024)
  6. Chi-square statistic
  7. Monte Carlo Pi estimation
  8. Serial Byte Correlation Coefficient
  9. Mean byte value
  10-265. Byte frequency distribution (256 features - frecuencia de cada byte 0x00-0xFF)
  266. Byte frequency std deviation
  267. Byte frequency max
  268. Byte frequency min
  269. Ratio of zero bytes
  270. Ratio of high-entropy bytes (>0xF0)
  271. Ratio of printable ASCII bytes
  272. Longest run of same byte
  273. Number of unique byte values
  274. Diferencia de entropia header vs footer
  275. Entropia por bloques (std de entropias de 16 bloques)

Total: ~275 features por archivo

Uso:
  python advanced_features.py <directorio_con_carpetas_de_familias> <output_csv>

  Ejemplo:
  python advanced_features.py Pruebas2/ advanced_features.csv
"""

import os
import sys
import csv
import math
import numpy as np
from collections import Counter


def calculate_entropy(data):
    """Shannon entropy of byte data."""
    if len(data) == 0:
        return 0.0
    byte_counts = Counter(data)
    entropy = 0.0
    length = len(data)
    for count in byte_counts.values():
        p_x = count / length
        entropy -= p_x * math.log2(p_x)
    return entropy


def calculate_chi_square(data):
    """Chi-square test for uniform distribution."""
    if len(data) == 0:
        return 0.0
    byte_counts = Counter(data)
    expected = len(data) / 256.0
    chi_sq = sum((byte_counts.get(i, 0) - expected) ** 2 / expected for i in range(256))
    return chi_sq


def calculate_monte_carlo_pi(data):
    """Monte Carlo Pi estimation from byte pairs."""
    if len(data) < 2:
        return 0.0
    points = [(data[i], data[i+1]) for i in range(0, len(data) - 1, 2)]
    inside = sum(1 for x, y in points if (x/255.0)**2 + (y/255.0)**2 <= 1.0)
    return (inside / len(points)) * 4


def calculate_sbcc(data):
    """Serial Byte Correlation Coefficient."""
    if len(data) < 2:
        return 0.0
    mean = sum(data) / len(data)
    numerator = sum((data[i] - mean) * (data[i+1] - mean) for i in range(len(data) - 1))
    denominator = sum((data[i] - mean) ** 2 for i in range(len(data) - 1))
    return numerator / denominator if denominator != 0 else 0.0


def byte_frequency_distribution(data):
    """Returns normalized frequency of each byte value (0-255)."""
    counts = Counter(data)
    length = len(data) if len(data) > 0 else 1
    return [counts.get(i, 0) / length for i in range(256)]


def longest_byte_run(data):
    """Length of the longest consecutive run of the same byte."""
    if len(data) == 0:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, len(data)):
        if data[i] == data[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def block_entropy_features(data, n_blocks=16):
    """Divide file into n_blocks and compute entropy of each block.
    Returns std of entropies (measures uniformity of randomness)."""
    if len(data) < n_blocks:
        return 0.0
    block_size = len(data) // n_blocks
    entropies = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        block = data[start:end]
        entropies.append(calculate_entropy(block))
    return float(np.std(entropies))


def extract_features(filepath):
    """Extract all features from a single file."""
    with open(filepath, 'rb') as f:
        raw = f.read()

    data = list(raw)
    length = len(data)

    if length == 0:
        return None

    # Basic stats
    entropy_global = calculate_entropy(data)
    file_size = length

    # Regional entropy (header, middle, footer)
    header_size = min(1024, length)
    footer_size = min(1024, length)
    mid_start = max(0, length // 2 - 512)
    mid_end = min(length, mid_start + 1024)

    entropy_header = calculate_entropy(data[:header_size])
    entropy_footer = calculate_entropy(data[-footer_size:])
    entropy_middle = calculate_entropy(data[mid_start:mid_end])

    # Statistical tests
    chi_square = calculate_chi_square(data)
    monte_carlo = calculate_monte_carlo_pi(data)
    sbcc = calculate_sbcc(data)
    mean_byte = sum(data) / length

    # Byte frequency distribution (256 features)
    byte_freq = byte_frequency_distribution(data)
    byte_freq_std = float(np.std(byte_freq))
    byte_freq_max = max(byte_freq)
    byte_freq_min = min(byte_freq)

    # Byte category ratios
    zero_ratio = data.count(0) / length
    high_byte_ratio = sum(1 for b in data if b > 0xF0) / length
    printable_ratio = sum(1 for b in data if 32 <= b <= 126) / length

    # Structural features
    max_run = longest_byte_run(data)
    unique_bytes = len(set(data))
    entropy_diff_hf = abs(entropy_header - entropy_footer)
    block_entropy_std = block_entropy_features(data, n_blocks=16)

    # Assemble feature vector
    features = [
        entropy_global,      # 1
        file_size,           # 2
        entropy_header,      # 3
        entropy_footer,      # 4
        entropy_middle,      # 5
        chi_square,          # 6
        monte_carlo,         # 7
        sbcc,                # 8
        mean_byte,           # 9
    ]
    features.extend(byte_freq)  # 10-265 (256 features)
    features.extend([
        byte_freq_std,       # 266
        byte_freq_max,       # 267
        byte_freq_min,       # 268
        zero_ratio,          # 269
        high_byte_ratio,     # 270
        printable_ratio,     # 271
        max_run,             # 272
        unique_bytes,        # 273
        entropy_diff_hf,     # 274
        block_entropy_std,   # 275
    ])

    return features


def get_feature_names():
    """Returns list of feature names matching extract_features output."""
    names = [
        'entropy_global', 'file_size', 'entropy_header', 'entropy_footer',
        'entropy_middle', 'chi_square', 'monte_carlo_pi', 'sbcc', 'mean_byte',
    ]
    names.extend([f'byte_freq_{i:03d}' for i in range(256)])
    names.extend([
        'byte_freq_std', 'byte_freq_max', 'byte_freq_min',
        'zero_ratio', 'high_byte_ratio', 'printable_ratio',
        'longest_run', 'unique_bytes', 'entropy_diff_hf', 'block_entropy_std',
    ])
    return names


def process_directory(base_dir, folders):
    """Process all files in all family folders."""
    all_features = []
    all_labels = []

    for label, folder in enumerate(folders):
        dir_path = os.path.join(base_dir, folder)
        if not os.path.isdir(dir_path):
            print(f"  [SKIP] Folder not found: {dir_path}")
            continue

        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        print(f"  [{label:2d}] {folder}: {len(files)} files", end="", flush=True)

        count = 0
        for filename in files:
            filepath = os.path.join(dir_path, filename)
            try:
                feats = extract_features(filepath)
                if feats is not None:
                    all_features.append(feats)
                    all_labels.append(label)
                    count += 1
            except Exception as e:
                print(f"\n    Error processing {filename}: {e}")

        print(f" -> {count} extracted")

    return all_features, all_labels


def main():
    if len(sys.argv) < 3:
        # Default: use the same folder structure as the original code
        base_dir = 'Pruebas2'
        output_file = 'advanced_features.csv'
    else:
        base_dir = sys.argv[1]
        output_file = sys.argv[2]

    folders = [
        'AVOSLOCKER-tiny', 'BADRABBIT-tiny', 'BLACKBASTA-tiny', 'BLACKCAT-tiny',
        'BLACKMATTER-tiny', 'CERBER-tiny', 'CHIMERA-tiny', 'CLOP-tiny', 'CONTI-tiny',
        'CRYPTOLOCKER-tiny', 'CUBA-tiny', 'DARKSIDE-tiny', 'DHARMA-tiny', 'GANDCRAB-tiny',
        'HELLOKITTY-tiny', 'JIGSAW-tiny', 'LOCKBIT-tiny', 'LORENZ-tiny', 'MAZE-tiny',
        'MEDUZALOCKER-tiny', 'NETWALKER-tiny', 'NOTPETYA-tiny', 'PHOBOS-tiny',
        'RANSOMEXX-tiny', 'RYUK-tiny', 'SODINOKIBI-tiny', 'SUNCRYPT-tiny',
        'TESLACRYPT-tiny', 'WANNACRY-tiny', 'WASTEDLOCKER-tiny', 'Z-Safe'
    ]

    # Allow switching to -small suffix via environment variable
    suffix = os.environ.get('NAPIERONE_SUFFIX', 'tiny')
    if suffix != 'tiny':
        folders = [f.replace('-tiny', f'-{suffix}') for f in folders]

    print(f"Base directory: {base_dir}")
    print(f"Output file: {output_file}")
    print(f"Folder suffix: {suffix}")
    print(f"Number of families: {len(folders)}")
    print()

    features, labels = process_directory(base_dir, folders)

    if not features:
        print("No features extracted! Check your directory paths.")
        return

    # Save to CSV
    feature_names = get_feature_names()
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(feature_names + ['label'])
        for feat, label in zip(features, labels):
            writer.writerow(feat + [label])

    print(f"\nSaved {len(features)} samples with {len(feature_names)} features to {output_file}")
    print(f"Feature groups:")
    print(f"  - Statistical (9): entropy, size, chi-square, monte carlo, sbcc, mean")
    print(f"  - Regional entropy (3): header, footer, middle")
    print(f"  - Byte frequency (256): full distribution")
    print(f"  - Derived (7): freq stats, byte ratios, structural")
    print(f"  Total: {len(feature_names)} features")


if __name__ == '__main__':
    main()
