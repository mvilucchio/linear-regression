import linear_regression.sweeps.alpha_sweeps as alsw
import matplotlib.pyplot as plt
from scipy.special import erfc, erf
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


delta_in, delta_out, percentage, beta = 1.0, 0.5, 0.1, 0.2
alpha_min, alpha_max, n_alpha_pts = 0.1, 10000, 500
n_alpha_pts_BO = 100
delta_eff = (1 - percentage) * delta_in + percentage * delta_out
plateau_alpha_inf = (1 - percentage) * percentage**2 * (1 - beta) ** 2 + percentage * (1 - percentage) ** 2 * (
    beta - 1
) ** 2
a_hub_fixed = 1.0

fname_add = "deltain_{}_deltaout_{}_perc_{}_beta_{}".format(delta_in, delta_out, percentage, beta)

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
    f_min=excess_gen_error,
    f_min_args={"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta},
    funs=[sigma_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
    min_reg_param=1e-5
)

np.savez(
    "L2_FIGURE_1_left_" + fname_add + ".npz",
    alphas=alphas_L2,
    f_min_vals=f_min_vals_L2,
    reg_param_opt=reg_param_opt_L2,
    sigmas=sigmas_L2,
    qs=qs_L2,
    ms=ms_L2,
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
    f_min=excess_gen_error,
    f_min_args={"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta},
    funs=[sigma_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
    min_reg_param=1e-5,
)

np.savez(
    "L1_FIGURE_1_left_" + fname_add + ".npz",
    alphas=alphas_L1,
    f_min_vals=f_min_vals_L1,
    reg_param_opt=reg_param_opt_L1,
    sigmas=sigmas_L1,
    qs=qs_L1,
    ms=ms_L1,
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
    f_min=excess_gen_error,
    f_min_args={"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta},
    funs=[sigma_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
    min_reg_param=1e-5,
)

np.savez(
    "Huber_FIGURE_1_left_" + fname_add + ".npz",
    alphas=alphas_Hub,
    f_min_vals=f_min_vals_Hub,
    reg_param_opt=reg_param_opt_Hub,
    hub_params_opt=hub_params_opt_Hub,
    sigmas=sigmas_Hub,
    qs=qs_Hub,
    ms=ms_Hub,
)

print("Huber done")

(
    alphas_Hub_2,
    f_min_vals_Hub_2,
    reg_param_opt_Hub_2,
    (sigmas_Hub_2, qs_Hub_2, ms_Hub_2),
) = alsw.sweep_alpha_optimal_lambda_fixed_point(
    f_L2_reg,
    f_hat_Huber_decorrelated_noise,
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
        "a": a_hub_fixed,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    f_min=excess_gen_error,
    f_min_args={"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta},
    funs=[sigma_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
    min_reg_param=1e-5,
)

np.savez(
    "Huber_a_{:.1f}_FIGURE_1_left_".format(a_hub_fixed) + fname_add  + ".npz",
    alphas=alphas_Hub_2,
    f_min_vals=f_min_vals_Hub_2,
    reg_param_opt=reg_param_opt_Hub_2,
    sigmas=sigmas_Hub_2,
    qs=qs_Hub_2,
    ms=ms_Hub_2,
)

print("Huber 2 done")

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
    funs=[gen_error_BO, q_order_param],
    funs_args=[{"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta}, {}],
    decreasing=False,
)

np.savez(
    "BO_FIGURE_1_left_" + fname_add + ".npz",
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

# plt.plot(alphas_L2, angle_teacher_student(ms_L2, qs_L2, sigmas_L2), label="L2")
# plt.plot(alphas_L1, angle_teacher_student(ms_L1, qs_L1, sigmas_L1), label="L1")
# plt.plot(alphas_Hub, angle_teacher_student(ms_Hub, qs_Hub, sigmas_Hub), label="Huber")
# plt.plot(alphas_BO, angle_teacher_student(qs_BO, qs_BO, qs_BO), label="BO")
# plt.yscale("log")
# plt.xscale("log")
# plt.ylabel("Angle teacher student")
# plt.legend()
# plt.grid()
# plt.xlabel(r"$\alpha$")

# plt.plot(alphas_L2, qs_L2, label="L2")
# plt.plot(alphas_L1, qs_L1, label="L1")
# plt.plot(alphas_Hub, qs_Hub, label="Huber")
# plt.plot(alphas_BO, qs_BO, label="BO")

plt.plot(alphas_L2, f_min_vals_L2, label="L2")
plt.plot(alphas_L1, f_min_vals_L1, label="L1")
plt.plot(alphas_Hub, f_min_vals_Hub, label="Huber")
plt.plot(alphas_Hub_2, f_min_vals_Hub_2, label="Huber a={:.1f}".format(a_hub_fixed))
# plt.plot(alphas_L1, qs_L1, label="q")
plt.plot(alphas_BO, gen_error_BO_old, label="BO")

# plt.axhline(y=1 - percentage + percentage * beta**2, color="black", linestyle="--")
# plt.axhline(y=percentage * (beta - 1) ** 2, color="violet", linestyle="--")
# plt.axhline(y=plateau_alpha_inf, color="green", linestyle="--")
# plt.axhline(y = percentage * np.sqrt(delta_out + (1 - percentage) * delta_in + 1))

plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"$E_{gen}^{excess}$")
plt.legend()
plt.grid()

plt.subplot(212)
# plt.plot(alphas_L2, 1 + qs_L2 - 2 * ms_L2, label="L2")
# plt.axhline(y=plateau_val, color="black", linestyle="--")
# plt.plot(alphas_L2, np.arccos(ms_L1 / np.sqrt(qs_L1)) / np.pi, label="angle L1")
# # plt.plot(alphas_L2, np.arccos(ms_Hub / np.sqrt(qs_Hub)) / np.pi, label="angle Huber")
# plt.plot(alphas_L1, 1 + qs_L1 - 2 * ms_L1, label="L1")
# plt.plot(alphas_Hub, 1 + qs_Hub - 2 * ms_Hub, label="Huber")
# plt.plot(alphas_BO, 1 - qs_BO, label="BO")
plt.plot(alphas_L2, reg_param_opt_L2, label="L2")
plt.plot(alphas_L1, reg_param_opt_L1, label="L1")
plt.plot(alphas_Hub, reg_param_opt_Hub, label="Huber $\\lambda$")
plt.plot(alphas_Hub, hub_params_opt_Hub, label="Huber $a$")
plt.plot(alphas_Hub_2, reg_param_opt_Hub_2, label="Huber $a$ fixed")

# valori asintotici lambda opt
# m_0 = 1 - percentage + percentage * beta
# eps_beta_1_2 = (percentage * (1 - beta)) ** 2
# plt.plot(
#     alphas_L1,
#     (1 / m_0 - 1) * alphas_L1 * (1 - percentage) / np.sqrt(np.pi / 2 * (delta_in + eps_beta_1_2))
#     + (beta / m_0 - 1)
#     * alphas_L1
#     * percentage
#     / np.sqrt(np.pi / 2 * (delta_out + ((1 - percentage) * (1 - beta)) ** 2)),
#     label="Claim L1"
# )

# plt.plot(
#     alphas_Hub,
#     (1 / m_0 - 1) * alphas_Hub * (1 - percentage) * erf(hub_params_opt_Hub / np.sqrt(2 * (delta_in + eps_beta_1_2)))
#     + (beta / m_0 - 1)
#     * alphas_Hub
#     * percentage
#     * erf(hub_params_opt_Hub / np.sqrt(2 * (delta_out + ((1 - percentage) * (1 - beta)) ** 2))),
#     label="Claim Huber",
# )

# plt.axhline(
#     y=((delta_eff + (1 - beta) ** 2 * (1 - percentage) * percentage) / ((1 + percentage * (beta - 1)) ** 2)),
#     color="black",
#     linestyle="--",
# )

# plt.axhline(y = percentage * np.sqrt(delta_out + (1 - percentage) * delta_in + 1))
# plt.plot(
#     alphas_BO,
#     1 + (1 - percentage + percentage * beta) ** 2 * qs_BO - 2 * (1 - percentage + percentage * beta) * qs_BO,
#     label="BO",
# )

plt.yscale("log")
plt.xscale("log")
plt.ylabel("Optimal params")
plt.legend()
plt.grid()

# plt.subplot(313)

# plt.plot(alphas_L2, np.arccos(ms_L2 / np.sqrt(qs_L2)) / np.pi, label="L2")
# plt.plot(alphas_L1, np.arccos(ms_L1 / np.sqrt(qs_L1)) / np.pi, label="L1")
# plt.plot(alphas_Hub, np.arccos(ms_Hub / np.sqrt(qs_Hub)) / np.pi, label="Huber")
# # plt.plot(alphas_Hub_2, np.arccos(ms_Hub_2 / np.sqrt(qs_Hub_2)) / np.pi, label="Huber a={:.1f}".format(a_hub_fixed))

# # plt.ylim([0.0, 10.0])
# plt.yscale("log")
# plt.xscale("log")
# plt.ylabel("Angle teacher student")
# plt.legend()
# plt.xlabel(r"$\alpha$")
# plt.grid()

plt.show()
