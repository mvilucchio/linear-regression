import numpy as np
import matplotlib.pyplot as plt
import linear_regression.sweeps.alpha_sweeps as alsw
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.fixed_point_equations.fpe_L2_loss import (
    f_hat_L2_decorrelated_noise,
)
from linear_regression.aux_functions.stability_functions import stability_ridge


def condition_MP(alphas):
    return -((1 - np.sqrt(alphas)) ** 2)


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0


alphas, f_min_vals, reg_param_opt, (sigmas, qs, ms) = alsw.sweep_alpha_optimal_lambda_fixed_point(
    f_L2_reg,
    f_hat_L2_decorrelated_noise,
    0.01,
    100,
    100,
    3.0,
    {"reg_param": 3.0},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[sigma_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
)

first_idx = 0
for idx, rp in enumerate(reg_param_opt):
    if rp <= 0.0:
        first_idx = idx
        break


plt.figure(figsize=(10, 10))

plt.subplot(311)
plt.title(
    "Ridge regression, L2 loss, L2 noise, $\\alpha$ sweep, $\\Delta_{{in}} = {}$, $\\Delta_{{out}} = {}$, $\\beta = {}$".format(
        delta_in, delta_out, beta
    )
)
plt.plot(alphas, f_min_vals)
plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"$E_{gen}$")
plt.grid()

plt.subplot(312)
plt.plot(alphas, reg_param_opt, label=r"$\lambda_{opt}$")
plt.plot(alphas, condition_MP(alphas), label=r"$-(1-\sqrt{\alpha})^2$")
plt.axvline(alphas[first_idx], color="red")
plt.xscale("log")
plt.ylim([-30, 8])
plt.ylabel(r"$\lambda_{opt}$")
plt.legend()
plt.grid()

plt.subplot(313)
plt.plot(
    alphas,
    stability_ridge(
        ms, qs, sigmas, alphas, reg_param_opt, delta_in, delta_out, percentage, beta, 1.0
    ),
    label=r"Stability cond.",
)
plt.legend()
plt.axvline(alphas[first_idx], color="red")
plt.xscale("log")
plt.grid()
plt.xlabel(r"$\alpha$")

plt.show()
