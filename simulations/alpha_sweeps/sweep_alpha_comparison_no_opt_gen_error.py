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


def condition_MP(alphas):
    return -((1 - np.sqrt(alphas)) ** 2)


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


delta_in, delta_out, percentage, beta = 1.0, 0.5, 0.6, 0.2
alpha_min, alpha_max, n_alpha_pts = 0.01, 100000, 200
reg_param = 1e-3
a_hub = 1.0
delta_eff = (1 - percentage) * delta_in + percentage * delta_out
plateau_alpha_inf = (1 - percentage) * percentage**2 * (1 - beta) ** 2 + percentage * (1 - percentage) ** 2 * (
    beta - 1
) ** 2

fname = "{}_fixed_lambda_{:.2e}_delta_in_{:.2e}_delta_out_{:.2e}_percentage_{:.2e}_beta_{:.2e}"

(
    alphas_L2,
    (gen_error_L2, sigmas_L2, qs_L2, ms_L2),
) = alsw.sweep_alpha_fixed_point(
    f_L2_reg,
    f_hat_L2_decorrelated_noise,
    alpha_min,
    alpha_max,
    n_alpha_pts,
    {"reg_param": reg_param},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[excess_gen_error, sigma_order_param, q_order_param, m_order_param],
    funs_args=[{"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta}, {}, {}, {}],
)

np.savez(
    fname.format("L2", reg_param, delta_in, delta_out, percentage, beta),
    alphas=alphas_L2,
    gen_error=gen_error_L2,
    sigmas=sigmas_L2,
    qs=qs_L2,
    ms=ms_L2,
)

print("L2 done")

(
    alphas_L1,
    (gen_error_L1, sigmas_L1, qs_L1, ms_L1),
) = alsw.sweep_alpha_fixed_point(
    f_L2_reg,
    f_hat_L1_decorrelated_noise,
    alpha_min,
    alpha_max,
    n_alpha_pts,
    {"reg_param": reg_param},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[excess_gen_error, sigma_order_param, q_order_param, m_order_param],
    funs_args=[{"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta}, {}, {}, {}],
)

np.savez(
    fname.format("L1", reg_param, delta_in, delta_out, percentage, beta),
    alphas=alphas_L1,
    gen_error=gen_error_L1,
    sigmas=sigmas_L1,
    qs=qs_L1,
    ms=ms_L1,
)

print("L1 done")

(
    alphas_Hub,
    (gen_error_Hub, sigmas_Hub, qs_Hub, ms_Hub),
) = alsw.sweep_alpha_fixed_point(
    f_L2_reg,
    f_hat_Huber_decorrelated_noise,
    alpha_min,
    alpha_max,
    n_alpha_pts,
    {"reg_param": reg_param},
    {"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta, "a": a_hub},
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[excess_gen_error, sigma_order_param, q_order_param, m_order_param],
    funs_args=[{"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta}, {}, {}, {}],
)

np.savez(
    fname.format("Huber_a_1.0", reg_param, delta_in, delta_out, percentage, beta),
    alphas=alphas_Hub,
    gen_error=gen_error_Hub,
    sigmas=sigmas_Hub,
    qs=qs_Hub,
    ms=ms_Hub,
)

print("Huber done")

# alphas_BO, (gen_error_BO_old, qs_BO) = alsw.sweep_alpha_fixed_point(
#     f_BO,
#     f_hat_BO_decorrelated_noise,
#     alpha_min,
#     alpha_max,
#     20,
#     {"reg_param": 1e-5},
#     {
#         "delta_in": delta_in,
#         "delta_out": delta_out,
#         "percentage": percentage,
#         "beta": beta,
#     },
#     initial_cond_fpe=(0.6, 0.01, 0.9),
#     funs=[gen_error_BO, q_order_param],
#     funs_args=[{"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta}, {}],
#     decreasing=False,
# )

# print("BO done")

plt.figure(figsize=(7, 7))

plt.subplot(211)
plt.title(
    "$\\alpha$ sweep, $\\Delta_{{in}} = {}$, $\\Delta_{{out}} = {}$, $\\epsilon = {}$, $\\beta = {}$, $\\lambda = {}$, $a = {}$".format(
        delta_in, delta_out, percentage, beta, reg_param, a_hub
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


plt.plot(alphas_L2, gen_error_L2, label="L2")
plt.plot(alphas_L1, gen_error_L1, label="L1")
plt.plot(alphas_Hub, gen_error_Hub, label="Huber")
# plt.plot(alphas_L1, qs_L1, label="q")
# plt.plot(alphas_BO, gen_error_BO_old, label="BO")

# plt.axhline(y=1 - percentage + percentage * beta**2, color="black", linestyle="--")
# plt.axhline(y=percentage * (beta - 1) ** 2, color="violet", linestyle="--")
# plt.axhline(y=plateau_alpha_inf, color="green", linestyle="--")
# plt.axhline(y = percentage * np.sqrt(delta_out + (1 - percentage) * delta_in + 1))

plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"$E_{gen}^{excess}$")
plt.legend()
plt.grid()

plateau_val = percentage**2 * (beta - 1) ** 2

plt.subplot(212)
# plt.plot(alphas_L2, 1 + qs_L2 - 2 * ms_L2, label="L2")
# plt.plot(alphas_L1, 1 + qs_L1 - 2 * ms_L1, label="L1")
# plt.plot(alphas_Hub, 1 + qs_Hub - 2 * ms_Hub, label="Huber")
plt.plot(alphas_L2, np.arccos(ms_L2 / np.sqrt(qs_L2)) / np.pi, label="L2")
plt.plot(alphas_L1, np.arccos(ms_L1 / np.sqrt(qs_L1)) / np.pi, label="L1")
plt.plot(alphas_Hub, np.arccos(ms_Hub / np.sqrt(qs_Hub)) / np.pi, label="Huber")

plt.yscale("log")
plt.xscale("log")
# plt.ylabel(r"$\|w^\star - \hat{w}\|^2$")
plt.ylabel(r"$\theta$")
plt.xlabel(r"$\alpha$")
plt.legend()
plt.grid()

plt.show()
