import numpy as np
import matplotlib.pyplot as plt
from linear_regression.erm.adversarial_perturbation_finders import (
    proj_grad_descent_step,
    loss_vec_linear_data_perturb,
    stochastic_proj_grad_descent_step,
)
from numba import njit


@njit
def non_linearity(x):
    return np.tanh(x)


@njit
def d_non_linearity(x):
    return 1 - np.tanh(x) ** 2


@njit
def linearity(x):
    return x


@njit
def d_linearity(x):
    return np.ones_like(x)


# here we consider just the non linearity and a value
d = 100
epsilons = np.logspace(-1.5, 1.5, 100)

xs = np.random.normal(0, 1, (1, d))
w = np.random.normal(0, 1, d)
ys = np.array([[1]])

F = np.eye(d)

random_direction = np.random.normal(0, 1, d)
random_direction /= np.linalg.norm(random_direction)

w_direction = w / np.linalg.norm(w)

losses = {"non_linear": [], "linear": []}

for eps in epsilons:

    losses["linear"].append(
        loss_vec_linear_data_perturb(eps * w_direction, xs, ys, w, F, linearity)[0]
    )

    losses["non_linear"].append(
        loss_vec_linear_data_perturb(eps * w_direction, xs, ys, w, F, non_linearity)[0]
    )

plt.plot(epsilons, losses["non_linear"], label="non_linear")
plt.plot(epsilons, losses["linear"], label="linear")
plt.grid(True)
plt.legend()
plt.show()
