from helpers import get_features
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score


def main():
    folders = ['AVOSLOCKER-tiny', 'BADRABBIT-tiny', 'BLACKBASTA-tiny', 'BLACKCAT-tiny', 'BLACKMATTER-tiny',
               'CERBER-tiny', 'CHIMERA-tiny', 'CLOP-tiny', 'CONTI-tiny', 'CRYPTOLOCKER-tiny', 'CUBA-tiny',
               'DARKSIDE-tiny', 'DHARMA-tiny', 'GANDCRAB-tiny', 'HELLOKITTY-tiny', 'JIGSAW-tiny', 'LOCKBIT-tiny',
               'LORENZ-tiny', 'MAZE-tiny', 'MEDUZALOCKER-tiny', 'NETWALKER-tiny', 'NOTPETYA-tiny', 'PHOBOS-tiny',
               'RANSOMEXX-tiny', 'RYUK-tiny', 'SODINOKIBI-tiny', 'SUNCRYPT-tiny', 'TESLACRYPT-tiny', 'WANNACRY-tiny',
               'WASTEDLOCKER-tiny', 'Z-Safe']
    features, labels = get_features(folders)
    # iterate test size and get the highest acc
    test_sizes = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
    best_accuracy = 0
    best_test_size = 0
    for test_size in test_sizes:
        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=0)

        # Normalizing the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the SVM
        model = svm.SVC(kernel='linear')  # or 'rbf', 'poly', etc.
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        print(f'Test Size: {test_size}')
        print(f'Accuracy: {acc}')
        print(classification_report(y_test, y_pred))
        print()
        if acc > best_accuracy:
            best_accuracy = acc
            best_test_size = test_size
    print(f'Best Test Size: {best_test_size}')
    print(f'Best Accuracy: {best_accuracy}')


if __name__ == '__main__':
    main()
