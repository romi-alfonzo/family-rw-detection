import os
import csv
import math
import numpy as np
from collections import Counter


def calculate_entropy(data):
    byte_counts = Counter(data)
    entropy = 0
    for count in byte_counts.values():
        p_x = count / len(data)
        entropy += - p_x * math.log2(p_x)
    return entropy


def process_files_in_directory(directory):
    features = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        # Check if it is a file
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as file:
                data = file.read()
                entropy = calculate_entropy(data)
                # print(f'File: {filename}, Entropy: {entropy}')
                size = len(data)
                features.append([entropy, size])  # Add more features as necessary
    return features


def get_features(folders):
    features = []
    labels = []
    features_file = 'features.csv'
    # check if features.csv exists
    if os.path.exists(features_file):
        with open(features_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                features.append([float(row[0]), int(row[1])])
                labels.append(int(row[2]))
    else:
        # Process the files in each ransomware family
        for i, folder in enumerate(folders):
            print(f'Processing folder: {folder}')
            directory_path = 'Pruebas2/' + folder
            aux = process_files_in_directory(directory_path)
            length = len(aux)
            features += aux
            # Add labels (in this case family id) to the files
            labels += [i] * length
        # Save proccessed data to a CSV file
        with open(features_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Entropy', 'Size', 'Label'])
            for feature, label in zip(features, labels):
                writer.writerow(feature + [label])
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    return features, labels
