import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def f(x):
    """f(x) = sqrt(|x|)"""
    return np.sqrt(np.abs(x))

def objective_function(x, y, lambda_val, f=f):
    """Obeective function for the proximal operator of f"""
    return lambda_val*f(x) +  (x - y)**2 / 2

def prox(y, lambda_val, f=f):
    res = minimize_scalar(objective_function, args=(y, lambda_val, f), bounds=(-200, 200), method='bounded')
    return res.x

# --- graph config ---
def plot_proximal_operator(f=f,lambda_vals=[0.5, 1.0, 2.0], y_range=(-5, 5), num_points=5000):
    """
    plots the proximal operator of f over a specified range of y values.
    """
    y_values = np.linspace(y_range[0], y_range[1], num_points)

    plt.figure(figsize=(10, 7))
    plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    plt.plot(y_values, y_values, 'k--', label='$x=y$ (identity)')

    for lambda_val in lambda_vals:
        prox_values = [prox(y, lambda_val, f) for y in y_values]
        plt.plot(y_values, prox_values, label=f'$\\lambda = {lambda_val}$')

    plt.title("Proximal Operator of $f$")
    plt.xlabel("$y$")
    plt.ylabel("$\\text{prox}_{\\lambda f}(y)$")
    plt.legend()
    plt.grid(True)
    plt.ylim(y_range[0], y_range[1])
    plt.xlim(y_range[0], y_range[1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

plot_proximal_operator()
