import linear_regression.sweeps.alpha_sweeps as alsw
import matplotlib.pyplot as plt
from linear_regression.fixed_point_equations.fpe_L2_loss import (
    var_hat_func_L2_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_L2_regularization import var_func_L2
from linear_regression.aux_functions.misc import estimation_error_rescaled, estimation_error, estimation_error_oracle_rescaling
import numpy as np


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
alpha_min, alpha_max, n_alpha_pts = 0.01, 100_000, 1500
delta_eff = delta_in + percentage * (delta_out - delta_in)
norm_const = 1 - percentage + percentage * beta
small_value = 0.0
reg_param = 5.0


(alphas_rescaled, (f_min_vals_rescaled,)) = alsw.sweep_alpha_fixed_point(
    var_func_L2,
    var_hat_func_L2_decorrelated_noise,
    alpha_min,
    alpha_max,
    n_alpha_pts,
    {"reg_param": 1.0},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[estimation_error_oracle_rescaling],
    funs_args=[
        {
            "delta_in": delta_in,
            "delta_out": delta_out,
            "percentage": percentage,
            "beta": beta,
        }
    ],
    update_funs_args=None,
)

print("First sweep done")

(
    alphas_neg,
    f_min_vals_neg,
    reg_param_opt_neg,
    (sigmas_neg, qs_neg, ms_neg),
) = alsw.sweep_alpha_optimal_lambda_fixed_point(
    var_func_L2,
    var_hat_func_L2_decorrelated_noise,
    alpha_min,
    alpha_max,
    n_alpha_pts,
    3.0,
    {"reg_param": 3.0},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    f_min=estimation_error,
    f_min_args={"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta},
    update_f_min_args=False,
    funs=[sigma_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
    min_reg_param=None,
)

print("Negative sweep done")


plt.figure(figsize=(7, 7))

plt.subplot(211)
plt.title(
    "Ridge regression, L2 loss, L2 noise, $\\alpha$ sweep, $\\Delta_{{in}} = {}$, $\\Delta_{{out}} = {}$, $\\epsilon = {}$, $\\beta = {}$".format(
        delta_in, delta_out, percentage, beta
    )
)

plt.plot(alphas_rescaled, f_min_vals_rescaled, label="rescaled $(1-\\epsilon + \\epsilon \\beta)$")
# plt.plot(alphas_2, f_min_vals_2, label="no rescaled")
plt.plot(alphas_neg, f_min_vals_neg, label="neg $\\lambda$ procedure")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.ylabel(r"$\|w^{\star} - \hat{w}\|^2}$")
plt.grid()

plt.subplot(212)
plt.axhline(y=1 - percentage + percentage * beta)
# plt.plot(alphas_2, reg_param_opt_2, label=r"$\lambda_{opt}$ no rescaled")
plt.plot(alphas_neg, reg_param_opt_neg, label=r"$\lambda_{opt} neg$")
plt.xscale("log")
plt.ylim([-8, 20])
plt.ylabel(r"$\lambda_{opt}$")
plt.legend()
plt.grid()


plt.show()
