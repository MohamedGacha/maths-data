import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from my_descent_td1 import GradientDescent

digits = datasets.load_digits()
X = digits.data
y = digits.target

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Fonction sigmoïde.

    Paramètre :
    - z : La valeur d'entrée.

    Retourne :
    - La sortie de la fonction sigmoïde appliquée à z.
    """
    return 1 / (1 + np.exp(-z))


def cost_function(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    Fonction de coût pour la régression logistique.

    Paramètres :
    - X : Les données d'entrée (matrice des caractéristiques).
    - y : Les étiquettes cibles.
    - theta : Les paramètres du modèle (poids).

    Retourne :
    - Le coût de la fonction de coût pour les paramètres donnés.
    """
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = -1/m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost


def gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Calcul du gradient de la fonction de coût pour la régression logistique.

    Paramètres :
    - X : Les données d'entrée (matrice des caractéristiques).
    - y : Les étiquettes cibles.
    - theta : Les paramètres du modèle (poids).

    Retourne :
    - Le gradient de la fonction de coût par rapport aux paramètres.
    """
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    grad = 1/m * np.dot(X.T, (h - y))
    return grad


# Initialisation des paramètres
m, n = X_train.shape
theta_init = np.zeros(n)
learning_rate = 0.01
iterations = 1000

# Création de l'instance de GradientDescent
gd = GradientDescent(gradient=lambda theta: gradient(X_train, y_train, theta),
                     learning_rate=learning_rate,
                     max_iterations=iterations)

# Entraînement du modèle
theta_optimized = gd.descent(theta_init)

# Prédictions
y_pred = sigmoid(np.dot(X_test, theta_optimized)) >= 0.5
y_pred = y_pred.astype(int)

# Évaluation des performances
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
# pew
