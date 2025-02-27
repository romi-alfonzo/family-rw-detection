import csv
import os
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def main():
    # features_file1 = 'features_shannon.csv'
    # features_file1 = 'features_100.csv'
    features_file1 = 'features_chi_square.csv'
    # features_file1 = 'features_mean.csv'
    # features_file1 = 'features_monte_carlo.csv'
    # features_file1 = 'features_sbcc.csv'

    # features_file2 = 'features_shannon.csv'
    # features_file2 = 'features_100.csv'
    # features_file2 = 'features_chi_square.csv'
    features_file2 = 'features_mean.csv'
    # features_file2 = 'features_monte_carlo.csv'
    # features_file2 = 'features_sbcc.csv'

    # features_file3 = 'features_shannon.csv'
    # features_file3 = 'features_100.csv'
    # features_file3 = 'features_chi_square.csv'
    # features_file3 = 'features_mean.csv'
    features_file3 = 'features_monte_carlo.csv'
    # features_file3 = 'features_sbcc.csv'

    # features_file4 = 'features_shannon.csv'
    # features_file4 = 'features_100.csv'
    # features_file4 = 'features_chi_square.csv'
    # features_file4 = 'features_mean.csv'
    # features_file4 = 'features_monte_carlo.csv'
    features_file4 = 'features_sbcc.csv'

    features = []
    labels = []
    if os.path.exists(features_file1) and os.path.exists(features_file2) and os.path.exists(features_file3) and os.path.exists(features_file4):
        with open(features_file1, 'r', newline='', encoding='utf-8') as csvfile1:
            with open(features_file2, 'r', newline='', encoding='utf-8') as csvfile2:
                with open(features_file3, 'r', newline='', encoding='utf-8') as csvfile3:
                    with open(features_file4, 'r', newline='', encoding='utf-8') as csvfile4:
                        reader1 = csv.reader(csvfile1)
                        reader2 = csv.reader(csvfile2)
                        reader3 = csv.reader(csvfile3)
                        reader4 = csv.reader(csvfile4)
                        next(reader1)
                        next(reader2)
                        next(reader3)
                        next(reader4)
                        for row1, row2, row3, row4 in zip(reader1, reader2, reader3, reader4):
                            features.append([float(row1[0]), float(row1[1]), float(row2[0]), float(row3[0]), float(row4[0])])
                            labels.append(int(row1[2]))
    else:
        print('File not found. Exiting...')
        return

    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    features = poly.fit_transform(features)

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=0)

    # Normalizing the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = []
    # Train the Logistic Regression
    models.append(LogisticRegression(max_iter=100, solver='liblinear')) # Set max_iter to handle convergence issues

    # Train the MLP
    models.append(MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1200,
                        alpha=0.0001, learning_rate='adaptive', random_state=42))

    # Train the SVM
    models.append(svm.SVC(kernel='linear'))  # or 'rbf', 'poly', etc.

    # print file names
    print(f'Features: {features_file1} and {features_file2} and {features_file3} and {features_file4}')

    for model in models:
        model.fit(X_train, y_train)
        # # Predict and evaluate
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f'Model: {type(model).__name__}')
        print(f'Accuracy: {round(acc, 3)}')


if __name__ == '__main__':
    main()
