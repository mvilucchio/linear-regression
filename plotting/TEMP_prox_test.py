# Matéo begins 
# This file is here for testing purposes, it is not used in the code.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

def f(x):
    """f(x) = sqrt(|x|)"""
    if np.abs(x) <1:
        return x #np.sqrt(np.abs(x))
    else:
        return 2*x-1

def objective_function(x, y, lambda_val, f=f):
    """Objective function for the proximal operator of f"""
    return lambda_val*f(x) +  (x - y)**2 / 2

def prox(y, lambda_val, f=f):
    res = minimize_scalar(objective_function, args=(y, lambda_val, f), bounds=(-200, 200), method='bounded')
    return res.x

# --- graph config ---
def plot_proximal_operator(f=f,lambda_vals=[ 1.0], y_range=(-5, 5), num_points=5000):
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
        f_values = [lambda_val*f(y)+y**2/2 for y in y_values]
        plt.plot(y_values, prox_values, label=f'$\\lambda = {lambda_val}$')
        plt.plot(y_values, f_values, linestyle='--', label=f'$\\lambda f(y)$')

    plt.title("Proximal Operator of $f$")
    plt.xlabel("$y$")
    plt.ylabel("$\\text{prox}_{\\lambda f}(y)$")
    plt.legend()
    plt.grid(True)
    plt.ylim(y_range[0], y_range[1])
    plt.xlim(y_range[0], y_range[1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# plot_proximal_operator()

def l1(x):
    return np.abs(x)
def l2(x):
    return x**2
def Huber(x):
    return np.where(np.abs(x) <= 1, x**2 / 2, np.abs(x) - 0.5)
def Tukey(x):
    return np.where(np.abs(x) <= 1, (1 - (1 - x**2)**3), 1)
def Cauchy(x):
    return np.log(1 + x**2)
def lp(x):
    return np.abs(x)**(1/2)
def oscillator(x):
    if np.abs(x) < np.sqrt(np.pi/2):
        return 2*np.sqrt(np.pi/2)
    else:
        return np.sign(x)*(1+np.sin(x**2))

def step(x):
    """ returns the integral ofthe oscillator function by using scipy's quad"""
    integral, _ = quad(oscillator, 0, x)
    return 0.2+integral/3

def negative_l2(x):
    """ returns the negative of the l2 norm"""
    return 2-(x-1.5)**2/6

# y_values = np.linspace(-1.5, 1.5, 10000)

# plt.figure(figsize=(10, 7))
# for g in [l1, l2, Huber, Tukey, Cauchy]:
#     plt.plot(y_values, [g(y) for y in y_values], label=g.__name__)
# plt.axvline(-1, color='gray', linestyle='--', linewidth=0.8)
# plt.axvline(1, color='gray', linestyle='--', linewidth=0.8)
# plt.title("Examples of weakly convex functions")
# plt.legend()
# plt.ylim(0,1.5)
# plt.xlim(-1.5, 1.5)
# plt.show()

y_values = np.linspace(-8, 8, 600)

plt.figure(figsize=(10, 7))
for g in [lp,step, negative_l2]:
    plt.plot(y_values, [g(y) for y in y_values], label=g.__name__)
plt.title("Weakly convex functions ?")
plt.legend()
plt.ylim(-1,3)
plt.xlim(-8, 8)
plt.show()
