import linear_regression.sweeps.alpha_sweeps as alsw
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import linear_regression.regression_numerics.data_generation as dg
import linear_regression.regression_numerics.erm_solvers as erm
from linear_regression.fixed_point_equations.fpe_L2_loss import (
    f_hat_L2_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_L1_loss import (
    f_hat_L1_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    f_hat_Huber_decorrelated_noise,
)
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
import numpy as np
from linear_regression.aux_functions.misc import excess_gen_error, estimation_error_rescaled, estimation_error_oracle_rescaling
from linear_regression.aux_functions.stability_functions import (
    stability_ridge,
    stability_l1_l2,
    stability_huber,
)


def condition_MP(alphas):
    return -((1 - np.sqrt(alphas)) ** 2)


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
alpha_min, alpha_max, n_alpha_pts = 0.01, 100_000, 200
norm_constant = 1 - percentage + percentage * beta + 1e-2

(
    alphas,
    f_min_vals,
    (reg_param_opt, hub_params_opt),
    (sigmas, qs, ms),
) = alsw.sweep_alpha_optimal_lambda_hub_param_fixed_point(
    f_L2_reg,
    f_hat_Huber_decorrelated_noise,
    alpha_min, alpha_max, n_alpha_pts,
    [3.0, 3.0],
    {"reg_param": 3.0},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
        "a": 1.0,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    f_min=estimation_error_oracle_rescaling,
    f_min_args={
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
        # "norm_const": norm_constant,
    },
    funs=[sigma_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
    min_reg_param=1e-8
)

(
    _,
    f_min_vals_neg,
    (reg_param_opt_neg, hub_params_opt_neg),
    (sigmas_neg, qs_neg, ms_neg),
) = alsw.sweep_alpha_optimal_lambda_hub_param_fixed_point(
    f_L2_reg,
    f_hat_Huber_decorrelated_noise,
    alpha_min, alpha_max, n_alpha_pts,
    [3.0, 3.0],
    {"reg_param": 3.0},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
        "a": 1.0,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[sigma_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
)


plt.figure(figsize=(7, 7))

plt.subplot(211)
plt.title(
    "Huber regression, Huber loss, L2 noise, $\\alpha$ sweep, $\\Delta_{{in}} = {}$, $\\Delta_{{out}} = {}$, $\\beta = {}$".format(
        delta_in, delta_out, beta
    )
)

plt.plot(alphas, f_min_vals, label=r"rescaled")
plt.plot(alphas, f_min_vals_neg, label=r"negative")
# plt.plot(alphas, np.abs(f_min_vals - f_min_vals_neg), label=r"diff")
# plt.plot(alphas, 1 + qs - 2 * ms, label=r"$\|w^\star - \hat{w}\|^2$")
# plt.plot(alphas, np.arccos(ms / np.sqrt(qs)) / np.pi, label="angle")
plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"$E_{gen}$")
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(alphas, reg_param_opt, label=r"$\lambda_{opt}$")
plt.plot(alphas, hub_params_opt, label=r"$a_{opt}$")
plt.plot(alphas, reg_param_opt_neg, label=r"$\lambda_{opt}$ negative")
plt.plot(alphas, hub_params_opt_neg, label=r"$a_{opt}$ negative")
# plt.axvline(alphas[first_idx], color="red")
plt.ylim([-5, 5])
plt.xscale("log")
plt.ylabel(r"$\lambda_{opt}$")
plt.legend()
plt.xlabel(r"$\alpha$")
plt.grid()

plt.show()
