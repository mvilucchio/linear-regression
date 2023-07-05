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
from linear_regression.aux_functions.misc import estimation_error, excess_gen_error, gen_error_BO, angle_teacher_student
import numpy as np


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


delta_in, delta_out, percentage, beta = 1.0, 3.0, 0.3, 1.2
alpha_min, alpha_max, n_alpha_pts = 0.1, 10000, 200
n_alpha_pts_BO = 100
delta_eff = (1 - percentage) * delta_in + percentage * delta_out
plateau_alpha_inf = (1 - percentage) * percentage**2 * (1 - beta) ** 2 + percentage * (1 - percentage) ** 2 * (
    beta - 1
) ** 2

fname_add = "_deltain_{}_deltaout_{}_perc_{}_beta_{}".format(delta_in, delta_out, percentage, beta)

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
    min_reg_param=1e-5, 
)

# (
#     alphas_L2_neg,
#     f_min_vals_L2_neg,
#     reg_param_opt_L2_neg,
#     (sigmas_L2_neg, qs_L2_neg, ms_L2_neg),
# ) = alsw.sweep_alpha_optimal_lambda_fixed_point(
#     f_L2_reg,
#     f_hat_L2_decorrelated_noise,
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
#     f_min=estimation_error,
#     f_min_args={},
#     funs=[sigma_order_param, q_order_param, m_order_param],
#     funs_args=[{}, {}, {}],
#     min_reg_param=None, 
# )

np.savez(
    "L2_Estimation_FIGURE_1" + fname_add + ".npz",
    alphas=alphas_L2,
    f_min_vals=f_min_vals_L2,
    reg_param_opt=reg_param_opt_L2,
    sigmas=sigmas_L2,
    qs=qs_L2,
    ms=ms_L2,
)

# np.savez(
#     "L2_Estimation_FIGURE_1_neg.npz",
#     alphas=alphas_L2_neg,
#     f_min_vals=f_min_vals_L2_neg,
#     reg_param_opt=reg_param_opt_L2_neg,
#     sigmas=sigmas_L2_neg,
#     qs=qs_L2_neg,
#     ms=ms_L2_neg,
# )

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
    min_reg_param=1e-5,
)

# (
#     alphas_L1_neg,
#     f_min_vals_L1_neg,
#     reg_param_opt_L1_neg,
#     (sigmas_L1_neg, qs_L1_neg, ms_L1_neg),
# ) = alsw.sweep_alpha_optimal_lambda_fixed_point(
#     f_L2_reg,
#     f_hat_L1_decorrelated_noise,
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
#     f_min=estimation_error,
#     f_min_args={},
#     funs=[sigma_order_param, q_order_param, m_order_param],
#     funs_args=[{}, {}, {}],
#     min_reg_param=None,
# )

np.savez(
    "L1_Estimation_FIGURE_1" + fname_add + ".npz",
    alphas=alphas_L1,
    f_min_vals=f_min_vals_L1,
    reg_param_opt=reg_param_opt_L1,
    sigmas=sigmas_L1,
    qs=qs_L1,
    ms=ms_L1,
)

# np.savez(
#     "L1_Estimation_FIGURE_1_neg.npz",
#     alphas=alphas_L1_neg,
#     f_min_vals=f_min_vals_L1_neg,
#     reg_param_opt=reg_param_opt_L1_neg,
#     sigmas=sigmas_L1_neg,
#     qs=qs_L1_neg,
#     ms=ms_L1_neg,
# )

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
    [3.0, 0.5],
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
    min_reg_param=1e-5,
)

# (
#     alphas_Hub_neg,
#     f_min_vals_Hub_neg,
#     (reg_param_opt_Hub_neg, hub_params_opt_Hub_neg),
#     (sigmas_Hub_neg, qs_Hub_neg, ms_Hub_neg),
# ) = alsw.sweep_alpha_optimal_lambda_hub_param_fixed_point(
#     f_L2_reg,
#     f_hat_Huber_decorrelated_noise,
#     alpha_min,
#     alpha_max,
#     n_alpha_pts,
#     [3.0, 0.5],
#     {"reg_param": 3.0},
#     {
#         "delta_in": delta_in,
#         "delta_out": delta_out,
#         "percentage": percentage,
#         "beta": beta,
#         "a": 1.0,
#     },
#     initial_cond_fpe=(0.6, 0.01, 0.9),
#     f_min=estimation_error,
#     f_min_args={},
#     funs=[sigma_order_param, q_order_param, m_order_param],
#     funs_args=[{}, {}, {}],
#     min_reg_param=None,
# )

np.savez(
    "Huber_Estimation_FIGURE_1" + fname_add + ".npz",
    alphas=alphas_Hub,
    f_min_vals=f_min_vals_Hub,
    reg_param_opt=reg_param_opt_Hub,
    hub_params_opt=hub_params_opt_Hub,
    sigmas=sigmas_Hub,
    qs=qs_Hub,
    ms=ms_Hub,
)

# np.savez(
#     "Huber_Estimation_FIGURE_1_neg.npz",
#     alphas=alphas_Hub_neg,
#     f_min_vals=f_min_vals_Hub_neg,
#     reg_param_opt=reg_param_opt_Hub_neg,
#     hub_params_opt=hub_params_opt_Hub_neg,
#     sigmas=sigmas_Hub_neg,
#     qs=qs_Hub_neg,
#     ms=ms_Hub_neg,
# )

print("Huber done")

alphas_BO, (gen_error_BO_old, qs_BO) = alsw.sweep_alpha_fixed_point(
    f_BO,
    f_hat_BO_decorrelated_noise,
    alpha_min,
    alpha_max,
    n_alpha_pts_BO,
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

np.savez(
    "BO_Estimation_FIGURE_1" + fname_add + ".npz",
    alphas=alphas_BO,
    gen_error=gen_error_BO_old,
    qs=qs_BO,
)

print("BO done")

plt.figure(figsize=(7, 7))

plt.subplot(211)
plt.title(
    "$\\alpha$ sweep, $\\Delta_{{in}} = {}$, $\\Delta_{{out}} = {}$, $\\epsilon = {}$, $\\beta = {}$".format(
        delta_in, delta_out, percentage, beta
    )
)

plt.plot(alphas_L2, f_min_vals_L2, label="L2", color="tab:blue")
# plt.plot(alphas_L2_neg, f_min_vals_L2_neg, color="tab:blue", linestyle="--")
plt.plot(alphas_L1, f_min_vals_L1, label="L1", color="tab:green")
# plt.plot(alphas_L1_neg, f_min_vals_L1_neg, color="tab:green", linestyle="--")
plt.plot(alphas_Hub, f_min_vals_Hub, label="Huber", color="tab:orange")
# plt.plot(alphas_Hub_neg, f_min_vals_Hub_neg, linestyle="--", color="tab:orange")
plt.plot(alphas_BO, gen_error_BO_old, label="BO", color="tab:red")
plt.ylim(1e-2, 1e0)
plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"$E_{estim}$")
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(alphas_L2, reg_param_opt_L2 / alphas_L2, label="L2", color="tab:blue")
# plt.plot(alphas_L2, reg_param_opt_L2_neg / alphas_L2_neg, color="tab:blue", linestyle="--")
plt.plot(alphas_L1, reg_param_opt_L1 / alphas_L1, label="L1", color="tab:green")
# plt.plot(alphas_L1, reg_param_opt_L1_neg / alphas_L1_neg, color="tab:green", linestyle="--")
plt.plot(alphas_Hub, reg_param_opt_Hub / alphas_Hub, label="Huber $\\lambda$", color="tab:orange")
# plt.plot(alphas_Hub, reg_param_opt_Hub_neg / alphas_Hub_neg, color="tab:orange", linestyle="--")
plt.plot(alphas_Hub, hub_params_opt_Hub, label="Huber $a$", color="grey")
# plt.plot(alphas_Hub, hub_params_opt_Hub_neg, color="grey", linestyle="--")

plt.xscale("log")
plt.ylim(-1, 2.5)
plt.ylabel(r"$(\lambda / \alpha, a)$")
plt.legend()
plt.grid()

plt.show()
