import linear_regression.sweeps.alpha_sweeps as alsw
import matplotlib.pyplot as plt
from linear_regression.fixed_point_equations.fpe_L2_loss import (
    f_hat_L2_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_BO import f_BO, f_hat_BO_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_L1_loss import (
    f_hat_L1_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    f_hat_Huber_decorrelated_noise,
)
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.misc import estimation_error, excess_gen_error, gen_error_BO
import numpy as np


def condition_MP(alphas):
    return -((1 - np.sqrt(alphas)) ** 2)


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


delta_in, delta_out, percentage, beta = 1.0, 3.0, 0.3, 0.0
alpha_min, alpha_max, n_alpha_pts = 0.01, 1000, 100
delta_eff = (1 - percentage) * delta_in + percentage * delta_out

(
    alphas_L2,
    f_min_vals_L2,
    reg_param_opt_L2,
    (sigmas_L2, qs_L2, ms_L2),
) = alsw.sweep_alpha_optimal_lambda_fixed_point(
    f_L2_reg,
    f_hat_L2_decorrelated_noise,
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
    f_min_args={},
    funs=[sigma_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
)

print("L2 done")

(
    alphas_L1,
    f_min_vals_L1,
    reg_param_opt_L1,
    (sigmas_L1, qs_L1, ms_L1),
) = alsw.sweep_alpha_optimal_lambda_fixed_point(
    f_L2_reg,
    f_hat_L1_decorrelated_noise,
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
    f_min_args={},
    funs=[sigma_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
)

print("L1 done")

(
    alphas_Hub,
    f_min_vals_Hub,
    (reg_param_opt_Hub, hub_params_opt_Hub),
    (sigmas_Hub, qs_Hub, ms_Hub),
) = alsw.sweep_alpha_optimal_lambda_hub_param_fixed_point(
    f_L2_reg,
    f_hat_Huber_decorrelated_noise,
    alpha_min,
    alpha_max,
    n_alpha_pts,
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
    f_min=estimation_error,
    f_min_args={},
    funs=[sigma_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
)

print("Huber done")

# compute the same with BO
alphas_BO, (gen_error_BO_old, qs_BO) = alsw.sweep_alpha_fixed_point(
    f_BO,
    f_hat_BO_decorrelated_noise,
    alpha_min, 
    alpha_max, 
    15,
    {"reg_param": 1e-5},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[estimation_error, q_order_param],
    funs_args=[{}, {}],
    decreasing=False,
)

print("BO done")

plt.figure(figsize=(7, 7))

plt.subplot(311)
plt.title(
    "$\\alpha$ sweep, $\\Delta_{{in}} = {}$, $\\Delta_{{out}} = {}$, $\\epsilon = {}$, $\\beta = {}$".format(
        delta_in, delta_out, percentage, beta
    )
)
plt.plot(alphas_L2, excess_gen_error(ms_L2, qs_L2, sigmas_L2, delta_in, delta_out, percentage, beta), label="L2")
plt.plot(alphas_L1, excess_gen_error(ms_L1, qs_L1, sigmas_L1, delta_in, delta_out, percentage, beta), label="L1")
plt.plot(alphas_Hub, excess_gen_error(ms_Hub, qs_Hub, sigmas_Hub, delta_in, delta_out, percentage, beta), label="Huber")
plt.plot(alphas_BO, gen_error_BO(qs_BO, qs_BO, 1 - qs_BO, delta_in, delta_out, percentage, beta), label="BO")
plt.axhline(y=percentage*(beta-1)**2, color="black", linestyle="--")
plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"$E_{gen}$")
plt.legend()
plt.grid()

plt.subplot(312)
plt.plot(alphas_L2, f_min_vals_L2, label="L2")
plt.plot(alphas_L1, f_min_vals_L1, label="L1")
plt.plot(alphas_Hub, f_min_vals_Hub, label="Huber")
plt.plot(alphas_BO, gen_error_BO_old, label="BO")
plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"$\|w^\star - \hat{w}\|^2$")
plt.legend()
plt.grid()

plt.subplot(313)
plt.plot(alphas_L2, reg_param_opt_L2, label="L2")
plt.plot(alphas_L1, reg_param_opt_L1, label="L1")
plt.plot(alphas_Hub, reg_param_opt_Hub, label="Huber $\\lambda$")
plt.plot(alphas_Hub, hub_params_opt_Hub, label="Huber $a$")
plt.ylim([0.0, 5.0])
plt.xscale("log")
plt.ylabel("Optimal params")
plt.legend()
plt.xlabel(r"$\alpha$")
plt.grid()

plt.show()
