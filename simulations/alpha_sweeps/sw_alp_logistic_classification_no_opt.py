import linear_regression.sweeps.alpha_sweeps as alsw
import matplotlib.pyplot as plt
import numpy as np
from linear_regression.fixed_point_equations.classification.Logistic_loss import (
    f_hat_Logistic_no_noise_classif
)
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.misc import angle_teacher_student
from linear_regression.fixed_point_equations.classification.BO import f_BO, f_hat_BO_no_noise_classif


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


alpha_min, alpha_max, n_alpha_pts = 0.1, 100, 20
reg_params = [0.1, 1.0, 10.0, 100.0]

fname = "./simulations/data/{}_classification_fixed_lambda_{:.2e}_no_noise"

plt.figure(figsize=(7.5, 7.5))
for reg_param in reg_params:
    print("reg_param = {:.2f}".format(reg_param))
    (
        alphas_Logistic,
        (angle_ts_Logistic, sigmas_Logistic, qs_Logistic, ms_Logistic),
    ) = alsw.sweep_alpha_fixed_point(
        f_L2_reg,
        f_hat_Logistic_no_noise_classif,
        alpha_min,
        alpha_max,
        n_alpha_pts,
        {"reg_param": reg_param},
        {},
        initial_cond_fpe=(0.3, 0.5, 0.9),
        funs=[angle_teacher_student, sigma_order_param, q_order_param, m_order_param],
        funs_args=[{}, {}, {}, {}],
    )

    plt.plot(alphas_Logistic, angle_ts_Logistic, "-", label=r"$\lambda = {:.2f}$".format(reg_param))

    np.savez(
        fname.format("Logistic", reg_param),
        alphas=alphas_Logistic,
        angle_ts=angle_ts_Logistic,
        sigmas=sigmas_Logistic,
        qs=qs_Logistic,
        ms=ms_Logistic,
    )

alphas_BO, (angle_ts_BO, qs_BO) = alsw.sweep_alpha_fixed_point(
    f_BO,
    f_hat_BO_no_noise_classif,
    alpha_min,
    alpha_max,
    n_alpha_pts,
    {},
    {},
    initial_cond_fpe=(0.6, 0.6, 0.9),
    funs=[angle_teacher_student, q_order_param],
    funs_args=[{}, {}],
    decreasing=False,
)
plt.plot(alphas_BO, angle_ts_BO, "k--", label=r"BO")

plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\theta / \pi$")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.xlim(alpha_min, alpha_max)
# plt.ylim(5e-3, 1e0)
plt.legend()

plt.show()
