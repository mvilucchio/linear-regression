import numpy as np
from math import sqrt
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from linear_regression.utils.integration_utils import (
    find_integration_borders_square,
    domains_sep_hyperboles_inside,
    domains_sep_hyperboles_above,
)
from matplotlib import cm
from linear_regression.fixed_point_equations.classification.Logistic_loss import (
    m_int_Logistic_no_noise_classif,
    q_int_Logistic_no_noise_classif,
    Σ_int_Logistic_no_noise_classif,
)

BIG_NUMBER = 10


m = 0.3
q = 0.5
sigma = 0.9

domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]


plt.figure(figsize=(10, 10))
for y, d in domains:
    print(y, d)
    xs = np.linspace(d[0], d[1], 1000)

    if y == 1:
        plt.axvline(d[0], color="k", linestyle="--")
        plt.axvline(d[1], color="k", linestyle="--")
        plt.plot(xs, [m_int_Logistic_no_noise_classif(x, y, q, m, sigma) for x in xs], "b", label="m")
        plt.plot(xs, [m_int_Logistic_no_noise_classif(x, y, q, m, sigma) for x in xs], "g", label="q")
        plt.plot(xs, [Σ_int_Logistic_no_noise_classif(x, y, q, m, sigma) for x in xs], "r", label="Σ")
    # else:
    #     plt.axvline(d[0], color="y", linestyle="--")
    #     plt.axvline(d[1], color="y", linestyle="--")
    #     plt.plot(xs, [m_int_Hinge_no_noise_classif(x, y, q, m, sigma) for x in xs], "b--")
    #     plt.plot(xs, [q_int_Hinge_no_noise_classif(x, y, q, m, sigma) for x in xs], "g--")
    #     plt.plot(xs, [Σ_int_Hinge_no_noise_classif(x, y, q, m, sigma) for x in xs], "r--")

plt.xlabel(r"$\xi$")
plt.legend()
plt.grid()

plt.show()
