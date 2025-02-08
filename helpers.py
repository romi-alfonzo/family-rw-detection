import os
import csv
import math
import numpy as np
from collections import Counter


def calculate_entropy(data):
    # byte_counts = Counter(data[:100])
    byte_counts = Counter(data)
    entropy = 0
    for count in byte_counts.values():
        p_x = count / len(data)
        entropy += - p_x * math.log2(p_x)
    return entropy

def calculate_chi_square(data):
    # Count the frequency of each character
    char_counts = Counter(data)
    total_chars = sum(char_counts.values())

    # Calculate the expected frequency assuming uniform distribution
    expected_frequency = total_chars / len(char_counts)

    # Create observed and expected frequency arrays
    observed = np.array(list(char_counts.values()))
    expected = np.full(len(char_counts), expected_frequency)

    # Calculate the chi-square statistic
    chi_square_statistic = ((observed - expected) ** 2 / expected).sum()

    return chi_square_statistic

def calculate_monte_carlo_pi(data):
    # Interpret the data as a sequence of random points
    points = [(data[i], data[i+1]) for i in range(0, len(data) - 1, 2)]

    # Count the number of points inside the unit circle
    inside_circle = sum(1 for x, y in points if (x / 255.0) ** 2 + (y / 255.0) ** 2 <= 1.0)

    # Estimate pi
    pi_estimate = (inside_circle / len(points)) * 4

    return pi_estimate

def calculate_sbcc(data):
    # Calculate the mean of the data
    mean = calculate_mean(data)

    # Calculate the numerator and denominator for the correlation coefficient
    numerator = sum((data[i] - mean) * (data[i+1] - mean) for i in range(len(data) - 1))
    denominator = sum((data[i] - mean) ** 2 for i in range(len(data) - 1))

    # Calculate the serial byte correlation coefficient
    sbcc = numerator / denominator if denominator != 0 else 0

    return sbcc

def calculate_mean(data):
    return sum(data) / len(data)


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
