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
from linear_regression.fixed_point_equations.classification.Hinge_loss import (
    m_int_Hinge_single_noise_classif,
    q_int_Hinge_single_noise_classif,
    Σ_int_Hinge_single_noise_classif,
)


def hyperbole(x, const):
    return const / x


m = 0.21562244834593775
q = 0.721271735064917
sigma = 7.5354931610250615
delta = 1.0

borders = find_integration_borders_square(
    m_int_Hinge_single_noise_classif, sqrt(1 + delta), 1.0, args=(q, m, sigma, delta), mult=10
)
domain_xi_1, domain_y_1 = domains_sep_hyperboles_inside(
    borders, hyperbole, hyperbole, {"const": (1.0 - sigma) / sqrt(q)}, {"const": 1.0 / sqrt(q)}
)
domain_xi_2, domain_y_2 = domains_sep_hyperboles_above(borders, hyperbole, {"const": (1.0 - sigma) / sqrt(q)})
domain_xi, domain_y = domain_xi_1 + domain_xi_2, domain_y_1 + domain_y_2


bound = borders[0][1]
PTS = 3000
xi = np.linspace(-bound, bound, PTS)
y = np.linspace(-bound, bound, PTS)
X, Y = np.meshgrid(xi, y)

Z_m = m_int_Hinge_single_noise_classif(X, Y, q, m, sigma, delta)
Z_q = q_int_Hinge_single_noise_classif(X, Y, q, m, sigma, delta)
Z_sigma = Σ_int_Hinge_single_noise_classif(X, Y, q, m, sigma, delta)

print(domain_xi_2)
plt.figure(figsize=(10, 10))
plt.title("Inside hyperboles")
for jdx, (d_xi, d_y) in enumerate(zip(domain_xi, domain_y)):
    xs = np.linspace(d_xi[0], d_xi[1], 1000)
    ys0 = np.zeros_like(xs)
    ys1 = np.zeros_like(xs)

    for idx in range(len(xs)):
        ys0[idx] = d_y[0](xs[idx])
        ys1[idx] = d_y[1](xs[idx])

    zs = np.ones_like(xs)
    ax = plt.gca()
    color = ax._get_lines.get_next_color()
    
    # fill the area between the two curves
    plt.fill_between(xs, ys0, ys1, facecolor=color, alpha=0.7)
    plt.plot(xs, ys0, label="first {}".format(jdx), color=color, linestyle="--")
    plt.plot(xs, ys1, label="second {}".format(jdx), color=color, linestyle="-.")


domain_xi_Σ_hat, domain_y_Σ_hat = domain_xi_1, domain_y_1
integral_value_Σ_hat = 0.0
for xi_funs, y_funs in zip(domain_xi_Σ_hat, domain_y_Σ_hat):
    integral_value_Σ_hat += dblquad(
        Σ_int_Hinge_single_noise_classif,
        xi_funs[0],
        xi_funs[1],
        y_funs[0],
        y_funs[1],
        args=(q, m, sigma, delta),
        epsabs=1e-10,
    )[0]

print("integral_value_Σ_hat ", integral_value_Σ_hat)

plt.ylim(-bound - 0.1, bound + 0.1)
plt.xlim(-bound - 0.1, bound + 0.1)
plt.xlabel(r"$\xi$")
plt.ylabel(r"$y$")
plt.legend()
plt.show()


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z_q, cmap=cm.coolwarm, linewidth=0, antialiased=True)
# ax.plot_surface(X, Y, Z_m)
ax.set_xlabel(r"$\xi$")
ax.set_ylabel(r"$y$")
ax.set_zlabel("Integral")

plt.show()
