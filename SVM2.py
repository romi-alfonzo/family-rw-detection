import os
import math
import numpy as np
from collections import Counter
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score


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


folders = ['AVOSLOCKER-tiny', 'BADRABBIT-tiny', 'BLACKBASTA-tiny', 'BLACKCAT-tiny', 'BLACKMATTER-tiny', 'CERBER-tiny',
           'CHIMERA-tiny', 'CLOP-tiny', 'CONTI-tiny', 'CRYPTOLOCKER-tiny', 'CUBA-tiny', 'DARKSIDE-tiny', 'DHARMA-tiny',
           'GANDCRAB-tiny', 'HELLOKITTY-tiny', 'JIGSAW-tiny', 'LOCKBIT-tiny', 'LORENZ-tiny', 'MAZE-tiny',
           'MEDUZALOCKER-tiny', 'NETWALKER-tiny', 'NOTPETYA-tiny', 'PHOBOS-tiny', 'RANSOMEXX-tiny', 'RYUK-tiny',
           'SODINOKIBI-tiny', 'SUNCRYPT-tiny', 'TESLACRYPT-tiny', 'WANNACRY-tiny', 'WASTEDLOCKER-tiny', 'Z-Safe']
features = []
labels = []

# Process the files in each ransomware family
for i, folder in enumerate(folders):
    print(f'Processing folder: {folder}')
    directory_path = 'Pruebas2/' + folder
    aux = process_files_in_directory(directory_path)
    length = len(aux)
    features += aux
    # Add labels (in this case family id) to the files
    labels += [i] * length

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=0)

# Normalizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM
model = svm.SVC(kernel='linear')  # or 'rbf', 'poly', etc.
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print(f'Confusion Matrix:\n{cm}')
print(f'Accuracy: {acc}')
