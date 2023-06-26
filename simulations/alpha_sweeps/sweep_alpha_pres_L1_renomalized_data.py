import linear_regression.sweeps.alpha_sweeps as alsw
import matplotlib.pyplot as plt
from linear_regression.fixed_point_equations.fpe_L1_loss import (
    var_hat_func_L1_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_L2_regularization import var_func_L2
from linear_regression.aux_functions.misc import estimation_error_rescaled, estimation_error, excess_gen_error, estimation_error_oracle_rescaling
import numpy as np


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
alpha_min, alpha_max, n_alpha_pts = 0.1, 1000, 100
delta_eff = delta_in + percentage * (delta_out - delta_in)
normalization_const = 1 - percentage + percentage * beta + 1.4649e-1
normalization_const_2 = 1 - percentage * (1 - beta) 

print(f"normalization_const = {normalization_const}")

(
    alphas,
    f_min_vals,
    reg_param_opt,
    (sigmas, qs, ms),
) = alsw.sweep_alpha_optimal_lambda_fixed_point(
    var_func_L2,
    var_hat_func_L1_decorrelated_noise,
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
    f_min=estimation_error_oracle_rescaling,
    f_min_args={
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
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
#     var_func_L2,
#     var_hat_func_L1_decorrelated_noise,
#     alpha_min,
#     alpha_max,
#     n_alpha_pts,
#     3.0,
#     {"reg_param": 3.0},
#     {
#         "delta_in": delta_in,
#         "delta_out": delta_out,
#         "percentage": percentage,
#         "beta": beta,
#     },
#     initial_cond_fpe=(0.6, 0.01, 0.9),
#     f_min=estimation_error_rescaled,
#     f_min_args={
#         "delta_in": delta_in,
#         "delta_out": delta_out,
#         "percentage": percentage,
#         "beta": beta,
#         "norm_const": normalization_const_2,
#     },
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
    var_func_L2,
    var_hat_func_L1_decorrelated_noise,
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

# (
#     alphas_3,
#     f_min_vals_gen_error,
#     reg_param_opt_gen_error,
#     (estim_error, sigmas_gen_error, qs_gen_error, ms_gen_error),
# ) = alsw.sweep_alpha_optimal_lambda_fixed_point(
#     var_func_L2,
#     var_hat_func_L1_decorrelated_noise,
#     alpha_min,
#     alpha_max,
#     n_alpha_pts,
#     3.0,
#     {"reg_param": 3.0},
#     {
#         "delta_in": delta_in,
#         "delta_out": delta_out,
#         "percentage": percentage,
#         "beta": beta,
#     },
#     initial_cond_fpe=(0.6, 0.01, 0.9),
#     f_min=excess_gen_error,
#     f_min_args={"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta},
#     update_f_min_args=False,
#     funs=[estimation_error, sigma_order_param, q_order_param, m_order_param],
#     funs_args=[
#         {},
#         {},
#         {},
#         {},
#     ],
#     min_reg_param=1e-8,
# )

# print("Last sweep done")

plt.figure(figsize=(7, 7))

plt.subplot(211)
plt.title(
    "L1 loss, L2 noise, $\\alpha$ sweep, $\\Delta_{{in}} = {}$, $\\Delta_{{out}} = {}$, $\\epsilon = {}$, $\\beta = {}$".format(
        delta_in, delta_out, percentage, beta
    )
)

plt.plot(alphas, f_min_vals, label="rescaled $1 - \\epsilon + \\epsilon \\beta$")
plt.plot(alphas_neg, f_min_vals_neg, label="neg $\\lambda$ procedure")
# plt.plot(alphas_2, f_min_vals_2, label="rescaled $1 - \\epsilon + \\epsilon \\beta + \\kappa$ ")
# plt.plot(alphas_3, estim_error, label="estim gen error")
# plt.plot(alphas, qs, label="q")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.ylabel(r"$\|w^{\star} - \hat{w}\|^2}$")
plt.grid()

plt.subplot(212)
plt.plot(alphas, reg_param_opt, label="$\\lambda_{opt}$ rescaled $1 - \\epsilon + \\epsilon \\beta$")
# plt.plot(alphas_2, reg_param_opt_2, label="$\\lambda_{opt}$ rescaled $1 - \\epsilon + \\epsilon \\beta + \\kappa$")
plt.plot(alphas_neg, reg_param_opt_neg, label=r"$\lambda_{opt}$ neg")
plt.xscale("log")
plt.ylim([-1, 8])
plt.ylabel(r"$\lambda_{opt}$")
plt.legend()
plt.grid()

# plt.subplot(313)
# plt.plot(alphas, f_min_vals - f_min_vals_neg, label="Error rescaled $1 - \\epsilon + \\epsilon \\beta$ - Error neg $\\lambda$")
# plt.plot(alphas_neg, -(f_min_vals - f_min_vals_2), label="Error rescaled $1 - \\epsilon + \\epsilon \\beta + \\kappa$ - Error rescaled $1 - \\epsilon + \\epsilon \\beta$")
# plt.plot(alphas_2, f_min_vals_2 - f_min_vals_neg, label="Error rescaled $1 - \\epsilon + \\epsilon \\beta + \\kappa$ - Error neg $\\lambda$")
# plt.xscale("log")
# plt.yscale("log")
# plt.ylabel("Error differences")
# plt.ylim([1e-8, 3e-2])
# plt.legend()
# plt.xlabel(r"$\alpha$")
# plt.grid()

plt.show()
