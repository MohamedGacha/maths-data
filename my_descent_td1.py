# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:45:29 2024

@author: JDION
"""

"""
Création de la classe GradientDescent utilisée durant l'intégralité de ce cours
"""


class GradientDescent:

    def __init__(self, gradient, learning_rate, max_iterations):
        """
        Initialise l'objet GradientDescent avec les paramètres nécessaires.

        Paramètres :
        - gradient : La fonction gradient de la fonction de coût.
        - learning_rate : Taux d'apprentissage (pas) pour la mise à jour des paramètres.
        - max_iterations : Nombre maximal d'itérations de l'algorithme de descente.
        """

        self.gradient = gradient
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def descent(self, initial_point):
        """
        Effectue l'algorithme de descente de gradient.

        Paramètres :
        - initial_point : Le point de départ de l'algorithme.

        Retourne :
        - Le point optimal trouvé par l'algorithme.
        """

        point = initial_point

        for _ in range(self.max_iterations):
            # Calcul du gradient à l'instant actuel
            gradient_value = self.gradient(point)

            # Mise à jour du point selon la règle de la descente de gradient
            point = self.update(point, gradient_value)

        return point

    def update(self, point, gradient_value):
        """
        Met à jour le point en utilisant le gradient et le taux d'apprentissage.

        Paramètres :
        - point : Le point à mettre à jour.
        - gradient_value : Le gradient de la fonction de coût au point donné.

        Retourne :
        - Le nouveau point après la mise à jour.
        """

        # Mise à jour du point en fonction du gradient et du taux d'apprentissage
        return point - self.learning_rate * gradient_value
