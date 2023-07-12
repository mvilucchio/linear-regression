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
from linear_regression.aux_functions.training_errors import integral_training_error_Hinge_no_noise, training_error_Hinge_loss_no_noise
from linear_regression.fixed_point_equations.classification.Hinge_loss import (
    m_int_Hinge_single_noise_classif,
    q_int_Hinge_single_noise_classif,
    Σ_int_Hinge_single_noise_classif,
)


def hyperbole(x, const):
    return const / x

m = 0.8463072405331419
q = 1.0719399594291075
sigma = 0.611168185322105
# m = 0.21562244834593775
# q = 0.721271735064917
# sigma = 7.5354931610250615
delta = 1.0


bound = BIG_NUMBER = 20

plt.figure(figsize=(7,7))
xs = np.linspace(-bound, bound, 1000)
plt.plot(xs, [integral_training_error_Hinge_no_noise(x, 1, q, m, sigma) for x in xs], "-")

print(training_error_Hinge_loss_no_noise(m, q, sigma))

# plt.ylim(-bound - 0.1, bound + 0.1)
plt.xlim(-bound - 0.1, bound + 0.1)
# plt.xlabel(r"$\xi$")
# plt.ylabel(r"$y$")
plt.show()

plt.show()
