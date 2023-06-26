# write a script that swows plots 3d the function to be integrated on the plane

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from linear_regression.utils.integration_utils import find_integration_borders_square, domains_sep_hyperboles_inside, domains_sep_hyperboles_above
from matplotlib import cm
from linear_regression.fixed_point_equations.fpe_Hinge_loss import m_integral_Hinge_decorrelated_noise, q_integral_Hinge_decorrelated_noise, sigma_integral_Hinge_decorrelated_noise

def hyperbole(x, const):
    return const / x

# m, q, sigma = 0.7, 0.7, 1.5
m = 0.06907
q = 0.29514
sigma = 0.86263
delta_in, delta_out, percentage, beta = 1.0, 2.0, 0.3, 0.0

bound = 5
PTS = 3000

# plot 3d the function to be integrated on the plane
xi = np.linspace(-bound, bound, PTS)
y = np.linspace(-bound, bound, PTS)
X, Y = np.meshgrid(xi, y)

Z_m = m_integral_Hinge_decorrelated_noise(X, Y, q, m, sigma, delta_in, delta_out, percentage, beta)
Z_q = q_integral_Hinge_decorrelated_noise(X, Y, q, m, sigma, delta_in, delta_out, percentage, beta)
Z_sigma = sigma_integral_Hinge_decorrelated_noise(X, Y, q, m, sigma, delta_in, delta_out, percentage, beta)

borders = [[-bound, bound], [-bound, bound]]

domain_xi_1, domain_y_1 = domains_sep_hyperboles_inside(
    borders, hyperbole, hyperbole, {"const": (1.0 - sigma) / sqrt(q)}, {"const": 1.0 / sqrt(q)}
)
domain_xi_2, domain_y_2 = domains_sep_hyperboles_above(borders, hyperbole, {"const": (1.0 - sigma) / sqrt(q)})
# domain_xi, domain_y = domain_xi_1 + domain_xi_2, domain_y_1 + domain_y_2
domain_xi, domain_y = domain_xi_2, domain_y_2

print(domain_xi)
plt.figure(figsize=(10,10))
plt.title("Inside hyperboles")
for jdx, (d_xi, d_y) in enumerate(zip(domain_xi, domain_y)):
    xs = np.linspace(d_xi[0], d_xi[1], 100)
    ys0 = np.zeros_like(xs)
    ys1 = np.zeros_like(xs)

    for idx in range(len(xs)):
        ys0[idx] = d_y[0](xs[idx])
        ys1[idx] = d_y[1](xs[idx])

    zs = np.ones_like(xs)
    ax = plt.gca()
    color = ax._get_lines.get_next_color()
    plt.plot(xs, ys0, label="first {}".format(jdx), color=color)
    plt.plot(xs, ys1, label="second {}".format(jdx), color=color)

xis = np.linspace(0.05,5,100)

# plt.plot(xis, hyperbole(xis, (1.0 - sigma) / sqrt(q)), label="hyperbole min", linestyle="--")
# plt.plot(xis, hyperbole(xis, 1.0 / sqrt(q)), label="hyperbole max", linestyle="--")
# plt.plot(-xis, hyperbole(-xis, (1.0 - sigma) / sqrt(q)), label="hyperbole 1", linestyle="--")
# plt.plot(-xis, hyperbole(-xis, 1.0 / sqrt(q)), label="hyperbole 2", linestyle="--")

plt.ylim(-bound-0.1, bound+0.1)
plt.xlim(-bound-0.1, bound+0.1)
plt.legend()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z_m, cmap=cm.coolwarm, linewidth=0, antialiased=True)
# # ax.plot_surface(X, Y, Z_m)
# ax.set_xlabel(r"$\xi$")
# ax.set_ylabel(r"$y$")
# ax.set_zlabel(r"$m$")

plt.show()
