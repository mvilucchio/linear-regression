import linear_regression.sweeps.alpha_sweeps as alsw
import matplotlib.pyplot as plt
from linear_regression.fixed_point_equations.fpe_L2_loss import (
    f_hat_L2_decorrelated_noise,
)
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.misc import estimation_error_rescaled, estimation_error
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


(
    alphas_5,
    f_min_vals_5,
    reg_param_opt_5,
    (sigmas_5, qs_5, ms_5),
) = alsw.sweep_alpha_optimal_lambda_fixed_point(
    f_L2_reg,
    f_hat_L2_decorrelated_noise,
    alpha_min, alpha_max, n_alpha_pts,
    3.0,
    {"reg_param": 3.0},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    f_min=estimation_error_rescaled,
    f_min_args={
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
        "norm_const": np.abs(norm_const) + small_value,
    },
    update_f_min_args=False,
    funs=[sigma_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
    min_reg_param=1e-8,
)

print("First sweep done")

# (
#     alphas_2,
#     f_min_vals_2,
#     reg_param_opt_2,
#     (sigmas_2, qs_2, ms_2),
# ) = alsw.sweep_alpha_optimal_lambda_fixed_point(
#     f_L2_reg,
#     f_hat_L2_decorrelated_noise,
#     alpha_min, alpha_max, n_alpha_pts,
#     3.0,
#     {"reg_param": 3.0},
#     {
#         "delta_in": delta_in,
#         "delta_out": delta_out,
#         "percentage": percentage,
#         "beta": beta,
#     },
#     initial_cond_fpe=(0.6, 0.01, 0.9),
#     f_min=estimation_error,
#     f_min_args={},
#     update_f_min_args=False,
#     funs=[sigma_order_param, q_order_param, m_order_param],
#     funs_args=[{}, {}, {}],
#     min_reg_param=1e-8,
# )

# print("Second sweep done")

(
    alphas_neg,
    f_min_vals_neg,
    reg_param_opt_neg,
    (sigmas_neg, qs_neg, ms_neg),
) = alsw.sweep_alpha_optimal_lambda_fixed_point(
    f_L2_reg,
    f_hat_L2_decorrelated_noise,
    alpha_min, alpha_max, n_alpha_pts,
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

plt.subplot(311)
plt.title(
    "Ridge regression, L2 loss, L2 noise, $\\alpha$ sweep, $\\Delta_{{in}} = {}$, $\\Delta_{{out}} = {}$, $\\epsilon = {}$, $\\beta = {}$".format(
        delta_in, delta_out, percentage, beta
    )
)

plt.plot(alphas_5, f_min_vals_5, label="rescaled $(1-\\epsilon + \\epsilon \\beta)$")
# plt.plot(alphas_2, f_min_vals_2, label="no rescaled")
plt.plot(alphas_neg, f_min_vals_neg, label="neg $\\lambda$ procedure")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.ylabel(r"$\|w^{\star} - \hat{w}\|^2}$")
plt.grid()

plt.subplot(312)
# plt.axhline(y=1-percentage + percentage*beta)
plt.plot(alphas_5, reg_param_opt_5, label=r"$\lambda_{opt}$ rescaled")
# plt.plot(alphas_2, reg_param_opt_2, label=r"$\lambda_{opt}$ no rescaled")
plt.plot(alphas_neg, reg_param_opt_neg, label=r"$\lambda_{opt} neg$")
plt.xscale("log")
plt.ylim([-1, 20])
plt.ylabel(r"$\lambda_{opt}$")
plt.legend()
plt.grid()


plt.subplot(313)
plt.plot(alphas_5, f_min_vals_neg - f_min_vals_5, label="Error (neg $\\lambda$) - Error (rescaled)")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Diff")
plt.xlabel(r"$\alpha$")
plt.legend()
plt.grid()

plt.show()
