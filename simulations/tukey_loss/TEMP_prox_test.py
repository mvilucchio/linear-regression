import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def f(x):
    """La fonction f(x) = sqrt(|x|)"""
    return np.sqrt(np.abs(x))

def objective_function(x, y, lambda_val):
    """La fonction à minimiser pour trouver l'opérateur proximal"""
    return f(x) + (1 / (2 * lambda_val)) * (x - y)**2

def prox_sqrt_abs_x(y, lambda_val):
    """Calcule numériquement l'opérateur proximal de sqrt(|x|)"""
    # Utilise minimize_scalar car c'est une fonction d'une seule variable
    # La borne (-100, 100) est une valeur de départ, on pourrait l'ajuster si nécessaire
    # pour des valeurs de y très grandes ou très petites.
    res = minimize_scalar(objective_function, args=(y, lambda_val), bounds=(-200, 200), method='bounded')
    return res.x

# --- Configuration du graphique ---
def plot_proximal_operator(lambda_vals=[0.5, 1.0, 2.0], y_range=(-5, 5), num_points=5000):
    """
    Trace le graphe de l'opérateur proximal pour différentes valeurs de lambda.

    Args:
        lambda_vals (list): Liste des valeurs de lambda à afficher.
        y_range (tuple): Intervalle (min, max) pour l'axe des y (entrée de l'opérateur).
        num_points (int): Nombre de points pour le tracé.
    """
    y_values = np.linspace(y_range[0], y_range[1], num_points)

    plt.figure(figsize=(10, 7))
    plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    plt.plot(y_values, y_values, 'k--', label='$x=y$ (Opérateur identité)') # Ligne identité

    for lambda_val in lambda_vals:
        prox_values = [prox_sqrt_abs_x(y, lambda_val) for y in y_values]
        plt.plot(y_values, prox_values, label=f'$\\lambda = {lambda_val}$')

    plt.title("Opérateur Proximal de $f(x) = \\sqrt{|x|}$")
    plt.xlabel("$y$")
    plt.ylabel("$\\text{prox}_{\\lambda f}(y)$")
    plt.legend()
    plt.grid(True)
    plt.ylim(y_range[0], y_range[1]) # Assure que l'axe y est aligné avec l'axe x pour une meilleure visualisation
    plt.xlim(y_range[0], y_range[1])
    plt.gca().set_aspect('equal', adjustable='box') # Pour avoir des échelles égales
    plt.show()

# --- Exemples d'utilisation ---

# Pour afficher le graphique avec les valeurs par défaut de lambda
print("Affichage du graphique avec des valeurs de lambda par défaut (0.5, 1.0, 2.0)...")
plot_proximal_operator()

# Pour varier les valeurs de lambda et la fenêtre d'affichage :
# Exemple : lambda=0.1, 0.5, 5.0 et une fenêtre d'affichage plus large
# print("\nAffichage du graphique avec des valeurs de lambda et une fenêtre d'affichage personnalisées...")
# plot_proximal_operator(lambda_vals=[0.1, 0.5, 5.0], y_range=(-10, 10))

# Exemple : une seule valeur de lambda
# print("\nAffichage du graphique avec une seule valeur de lambda personnalisée...")
# plot_proximal_operator(lambda_vals=[0.8], y_range=(-7, 7))
