import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Chargement des données
digits = load_digits()
X, y = digits.data, digits.target

# Normalisation des données
X = X / 16.0

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Ajout d'un biais (interception)
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Fonctions auxiliaires


def sigmoid(z):
    """Calcule la fonction sigmoïde."""
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, theta):
    """Calcule la fonction coût pour la régression logistique."""
    h = sigmoid(X @ theta)
    return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))


def gradient_descent(X, y, theta, learning_rate, num_iterations):
    """Applique la descente de gradient pour minimiser la fonction coût."""
    m = len(y)
    for _ in range(num_iterations):
        gradient = (X.T @ (sigmoid(X @ theta) - y)) / m
        theta -= learning_rate * gradient
    return theta


def predict(X, all_theta):
    """Prédiction des classes pour les données d'entrée X."""
    probabilities = sigmoid(X @ all_theta.T)
    return np.argmax(probabilities, axis=1)


# Implémentation One-vs-All
num_classes = len(np.unique(y_train))  # Nombre de classes (0-9)
# Matrice pour stocker les paramètres de chaque classe
all_theta = np.zeros((num_classes, X_train.shape[1]))

learning_rate = 0.1
num_iterations = 300

# Entraîner un classifieur pour chaque classe
for classe in range(num_classes):
    y_train_bin = (y_train == classe).astype(
        int)  # Binarisation des étiquettes
    # Initialisation de theta pour la classe
    theta = np.zeros(X_train.shape[1])
    theta = gradient_descent(X_train, y_train_bin, theta,
                             learning_rate, num_iterations)
    all_theta[classe, :] = theta  # Stocker les paramètres pour la classe

# Prédictions et calcul de la précision
y_pred = predict(X_test, all_theta)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy (Gradient Descent): {accuracy * 100:.2f}%")

# Comparaison avec Scikit-learn
model = LogisticRegression(max_iter=1000, multi_class='ovr')
# Supprimer la colonne de biais pour Scikit-learn
model.fit(X_train[:, 1:], y_train)
sklearn_accuracy = model.score(X_test[:, 1:], y_test)
print(f"Accuracy (Scikit-learn): {sklearn_accuracy * 100:.2f}%")
