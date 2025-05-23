from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic_adv_Linf_L2,
    find_coefficients_Logistic_adv_Linf_L1,
)
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.pstar_attacks_Lr_reg import (
    f_Lr_regularisation_Lpstar_attack,
)
from linear_regression.fixed_point_equations.classification.Adv_train_p_norm import (
    f_hat_Logistic_no_noise_Linf_adv_classif,
)
from linear_regression.fixed_point_equations.classification.Adversarial_Logistic_loss import (
    f_hat_Logistic_no_noise_classif,
)
import matplotlib.pyplot as plt
from linear_regression.data.generation import data_generation, measure_gen_probit_clasif
from linear_regression.erm.metrics import generalisation_error_classification
from cvxpy.error import SolverError
import numpy as np
import os
import sys

if len(sys.argv) > 1:
    alpha_min_se, alpha_max_se, n_alphas_se, d, delta, pstar, reg_p, metric_name_chosen = (
        float(sys.argv[1]),
        float(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
        float(sys.argv[5]),
        float(sys.argv[6]),
        float(sys.argv[7]),
        sys.argv[8],
    )
else:
    alpha_min_se, alpha_max_se, n_alphas_se = 0.25, 3.0, 50
    delta = 0.0
    pstar = 1.0
    reg_p = 1.0
    metric_name_chosen = "misclass"

d = 1000
alpha = 5.5

reg_param = 1.0e0

reps = 10
n_gen = 1000
eps_test = 1.0

f_hat_kwargs = {"alpha": alpha, "eps_t": 0.0}
f_kwargs = {"reg_param": reg_param, "reg_order": reg_p, "pstar": pstar}

m_se, q_se, V_se, P_se = fixed_point_finder(
    f_Lr_regularisation_Lpstar_attack,
    f_hat_Logistic_no_noise_Linf_adv_classif,
    (0.2, 1.0, 1.0, 1.0),
    f_kwargs,
    f_hat_kwargs,
    abs_tol=1e-6,
)

n = int(d * alpha)

xs, ys, xs_gen, ys_gen, wstar = data_generation(measure_gen_probit_clasif, d, n, n_gen, (delta,))

rho = np.mean(np.abs(wstar) ** 2)

try:
    w = find_coefficients_Logistic_adv_Linf_L1(ys, xs, reg_param, 0.0)
except (ValueError, SolverError) as e:
    print(f"minimization didn't converge: ")

plt.hist(wstar, bins=30, density=True, alpha=0.5, label="w")

xi = np.random.randn(d)
aaa = m_se * wstar / rho + np.sqrt(q_se - m_se**2 / rho) * xi
plt.hist(aaa, bins=30, density=True, alpha=0.5, label="aaa")

plt.legend()
plt.show()
