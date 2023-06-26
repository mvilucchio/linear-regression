import linear_regression.sweeps.alpha_sweeps as alsw
import matplotlib.pyplot as plt
from linear_regression.fixed_point_equations.fpe_L2_loss import (
    var_hat_func_L2_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_BO import var_func_BO, var_hat_func_BO_num_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_L1_loss import (
    var_hat_func_L1_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    var_hat_func_Huber_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_L2_regularization import var_func_L2
from linear_regression.aux_functions.misc import (
    estimation_error,
    excess_gen_error,
    gen_error_BO,
    angle_teacher_student,
    estimation_error_oracle_rescaling,
)
import numpy as np


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
alpha_min, alpha_max, n_alpha_pts = 0.01, 100000, 1000
delta_eff = (1 - percentage) * delta_in + percentage * delta_out
plateau_alpha_inf = (1 - percentage) * percentage**2 * (1 - beta) ** 2 + percentage * (1 - percentage) ** 2 * (
    beta - 1
) ** 2
fixed_reg_param = 1.0
fixed_a_hub = 1.0

fname_files = "{}_Figure_1_rescaling_lambda_{}.npz"

(
    alphas_L2,
    (f_min_vals_L2, sigmas_L2, qs_L2, ms_L2),
) = alsw.sweep_alpha_fixed_point(
    var_func_L2,
    var_hat_func_L2_decorrelated_noise,
    alpha_min,
    alpha_max,
    n_alpha_pts,
    {"reg_param": fixed_reg_param},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[estimation_error_oracle_rescaling, sigma_order_param, q_order_param, m_order_param],
    funs_args=[
        {
            "delta_in": delta_in,
            "delta_out": delta_out,
            "percentage": percentage,
            "beta": beta,
        },
        {},
        {},
        {},
    ],
)

np.savez(
    fname_files.format("L2", fixed_reg_param),
    alphas=alphas_L2,
    f_min_vals=f_min_vals_L2,
    sigmas=sigmas_L2,
    qs=qs_L2,
    ms=ms_L2,
)

print("L2 done")

(
    alphas_L1,
    (f_min_vals_L1, sigmas_L1, qs_L1, ms_L1),
) = alsw.sweep_alpha_fixed_point(
    var_func_L2,
    var_hat_func_L1_decorrelated_noise,
    alpha_min,
    alpha_max,
    n_alpha_pts,
    {"reg_param": fixed_reg_param},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[estimation_error_oracle_rescaling, sigma_order_param, q_order_param, m_order_param],
    funs_args=[
        {
            "delta_in": delta_in,
            "delta_out": delta_out,
            "percentage": percentage,
            "beta": beta,
        },
        {},
        {},
        {},
    ],
)

np.savez(
    fname_files.format("L1", fixed_reg_param),
    alphas=alphas_L1,
    f_min_vals=f_min_vals_L1,
    sigmas=sigmas_L1,
    qs=qs_L1,
    ms=ms_L1,
)

print("L1 done")

(
    alphas_Hub,
    (f_min_vals_Hub, sigmas_Hub, qs_Hub, ms_Hub),
) = alsw.sweep_alpha_fixed_point(
    var_func_L2,
    var_hat_func_Huber_decorrelated_noise,
    alpha_min,
    alpha_max,
    n_alpha_pts,
    {"reg_param": fixed_reg_param},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
        "a": fixed_a_hub,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[estimation_error_oracle_rescaling, sigma_order_param, q_order_param, m_order_param],
    funs_args=[{
            "delta_in": delta_in,
            "delta_out": delta_out,
            "percentage": percentage,
            "beta": beta,
        }, {}, {}, {}],
)

np.savez(
    fname_files.format("Huber", [fixed_reg_param, fixed_a_hub]),
    alphas=alphas_Hub,
    f_min_vals=f_min_vals_Hub,
    sigmas=sigmas_Hub,
    qs=qs_Hub,
    ms=ms_Hub,
)

print("Huber done")


plt.figure(figsize=(7, 7))

plt.subplot(211)
plt.title(
    "$\\alpha$ sweep, $\\Delta_{{in}} = {}$, $\\Delta_{{out}} = {}$, $\\epsilon = {}$, $\\beta = {}$".format(
        delta_in, delta_out, percentage, beta
    )
)

plt.plot(alphas_L2, f_min_vals_L2, label="L2")
plt.plot(alphas_L1, f_min_vals_L1, label="L1")
plt.plot(alphas_Hub, f_min_vals_Hub, label="Huber")
# plt.plot(alphas_L1, qs_L1, label="q")
# plt.plot(alphas_BO, gen_error_BO_old, label="BO")

plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"$E_{gen}^{excess}$")
plt.legend()
plt.grid()



plt.subplot(212)

plt.plot(alphas_L2, np.arccos(ms_L2 / np.sqrt(qs_L2)) / np.pi, label="L2")
plt.plot(alphas_L1, np.arccos(ms_L1 / np.sqrt(qs_L1)) / np.pi, label="L1")
plt.plot(alphas_Hub, np.arccos(ms_Hub / np.sqrt(qs_Hub)) / np.pi, label="Huber")

plt.yscale("log")
plt.xscale("log")
plt.ylabel("Angle teacher student")
plt.legend()
plt.xlabel(r"$\alpha$")
plt.grid()

plt.show()
