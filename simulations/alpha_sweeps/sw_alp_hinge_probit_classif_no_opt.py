import linear_regression.sweeps.alpha_sweeps as alsw
import matplotlib.pyplot as plt
import numpy as np
from linear_regression.fixed_point_equations.classification.Hinge_loss import (
    f_hat_Hinge_no_noise_classif,
    f_hat_Hinge_probit_classif
)
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.misc import angle_teacher_student
from linear_regression.fixed_point_equations.classification.BO import f_BO, f_hat_BO_no_noise_classif, f_hat_BO_probit_classif

def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


alpha_min, alpha_max, n_alpha_pts = 0.1, 100, 200
# reg_params = [0.001, 0.01, 0.1, 1.0, 10.0]
delta_noise = 0.1
reg_params = [0.001, 0.01, 0.1, 1.0, 10.0]

fname = "./simulations/data/{}_classification_fixed_lambda_{:.2e}_delta_{:.2e}"

plt.figure(figsize=(7.5, 7.5))
for reg_param in reg_params:
    print("reg_param = {:.2f}".format(reg_param))
    (
        alphas_Hinge,
        (angle_ts_Hinge, sigmas_Hinge, qs_Hinge, ms_Hinge),
    ) = alsw.sweep_alpha_fixed_point(
        f_L2_reg,
        f_hat_Hinge_probit_classif,
        alpha_min,
        alpha_max,
        n_alpha_pts,
        {"reg_param": reg_param},
        {"delta": delta_noise},
        initial_cond_fpe=(0.9, 0.9, 0.9),
        funs=[angle_teacher_student, sigma_order_param, q_order_param, m_order_param],
        funs_args=[{}, {}, {}, {}],
    )

    plt.plot(alphas_Hinge, angle_ts_Hinge, "-", label=r"$\lambda = {:.2f}$".format(reg_param))


# alphas_BO, (angle_ts_BO, qs_BO) = alsw.sweep_alpha_fixed_point(
#     f_BO,
#     f_hat_Hinge_probit_classif,
#     alpha_min,
#     alpha_max,
#     n_alpha_pts,
#     {},
#     {"delta": delta_noise},
#     initial_cond_fpe=(0.6, 0.6, 0.9),
#     funs=[angle_teacher_student, q_order_param],
#     funs_args=[{}, {}],
#     decreasing=False,
# )
# plt.plot(alphas_BO, angle_ts_BO, "k--", label=r"BO")

plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\theta / \pi$")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.xlim(alpha_min, alpha_max)
# plt.ylim(5e-3, 1e0)
plt.legend()

plt.show()
