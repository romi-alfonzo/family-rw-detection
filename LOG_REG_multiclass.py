from helpers import get_features
from sklearn.linear_model import LogisticRegression
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

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=0)

    # Normalizing the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the SVM
    model = LogisticRegression(max_iter=100, solver='liblinear')  # Set max_iter to handle convergence issues
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f'Confusion Matrix:\n{cm}')
    print(f'Accuracy: {acc}')


if __name__ == '__main__':
    main()
