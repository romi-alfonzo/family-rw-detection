import csv
import os, sys
from itertools import combinations
from datetime import datetime
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures

def main():
    number = int(sys.argv[1])
    feature_files = [
        'features_shannon.csv', 'features_100.csv', 'features_chi_square.csv', 
        'features_mean.csv', 'features_monte_carlo.csv', 'features_sbcc.csv'
    ]
    
    model_results = []
    for combo in combinations(feature_files, number):
        features = []
        labels = []
        
        if all(os.path.exists(f) for f in combo):
            readers = [csv.reader(open(f, 'r', newline='', encoding='utf-8')) for f in combo]
            for r in readers:
                next(r)
            
            for rows in zip(*readers):
                features.append([float(rows[i][0]) for i in range(number)])
                labels.append(int(rows[0][2]))
        else:
            print(f'File(s) not found: {combo}. Skipping...')
            continue

        # poly = PolynomialFeatures(degree=2, interaction_only=True)
        # features = poly.fit_transform(features)
        
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=0)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        models = [
            # LogisticRegression(max_iter=200, solver='liblinear'),
            # MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1200, alpha=0.0001, learning_rate='adaptive', random_state=42),
            # svm.SVC(kernel='linear'),
            DecisionTreeClassifier(random_state=42),
            KNeighborsClassifier(n_neighbors=5),
            # GradientBoostingClassifier(random_state=42),
            RandomForestClassifier(n_estimators=100, random_state=42)
        ]
        
        results = [" - ".join(combo)]
        for model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = round(accuracy_score(y_test, y_pred), 3)
            results.append(acc)
        
        model_results.append(results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_results_{timestamp}.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Combination", "DecisionTree", "KNN", "RandomForest"])
        writer.writerows(model_results)
    
    print("Results saved to model_results.csv")

if __name__ == '__main__':
    main()
