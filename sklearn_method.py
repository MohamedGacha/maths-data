import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from my_descent_td1 import GradientDescent

# Chargement des données
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)

# Implémentation de la régression logistique via la descente de gradient


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def cost_function(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = -1/m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost


def gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
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

# Prédictions de notre modèle
y_pred_gd = sigmoid(np.dot(X_test, theta_optimized)) >= 0.5
y_pred_gd = y_pred_gd.astype(int)

# Évaluation des performances de notre modèle
accuracy_gd = accuracy_score(y_test, y_pred_gd)
print(f'Accuracy (Gradient Descent): {accuracy_gd * 100:.2f}%')

# Implémentation de la régression logistique avec scikit-learn
model_sklearn = LogisticRegression(
    max_iter=1000, solver='lbfgs', random_state=42)
model_sklearn.fit(X_train, y_train)

# Prédictions du modèle sklearn
y_pred_sklearn = model_sklearn.predict(X_test)

# Évaluation des performances de scikit-learn
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(
    f'Accuracy (scikit-learn LogisticRegression): {accuracy_sklearn * 100:.2f}%')
