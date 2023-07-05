import linear_regression.sweeps.alpha_sweeps as alsw
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import linear_regression.regression_numerics.data_generation as dg
import linear_regression.regression_numerics.erm_solvers as erm
from linear_regression.fixed_point_equations.fpe_L2_loss import (
    f_hat_L2_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_BO import f_BO, f_hat_BO_decorrelated_noise
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.misc import estimation_error, excess_gen_error
import numpy as np


def condition_MP(alphas):
    return -((1 - np.sqrt(alphas)) ** 2)


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


delta_in, delta_out, percentage, beta = 1.0, 1.0, 0.3, 0.0
delta_eff = (1 - percentage) * delta_in + percentage * delta_out

(
    alphas,
    f_min_vals,
    reg_param_opt,
    (sigmas, qs, ms),
) = alsw.sweep_alpha_optimal_lambda_fixed_point(
    f_L2_reg,
    f_hat_L2_decorrelated_noise,
    0.01,
    10000,
    150,
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
    f_min_args={"delta_in" : delta_in, "delta_out" : delta_out, "percentage" : percentage, "beta" : beta},
    funs=[sigma_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
)

# compute the same with BO
# alphas_BO, (gen_error_BO_old, qs_BO) = alsw.sweep_alpha_fixed_point(
#     f_BO,
#     f_hat_BO_decorrelated_noise,
#     0.01,
#     10000,
#     40,
#     {"reg_param": 1e-5},
#     {
#         "delta_in": delta_in,
#         "delta_out": delta_out,
#         "percentage": percentage,
#         "beta": beta,
#     },
#     initial_cond_fpe=(0.6, 0.01, 0.9),
#     funs=[excess_gen_error, q_order_param],
#     funs_args=[(delta_in, delta_out, percentage, beta), {}],
#     decreasing=False,
# )

first_idx = 0
for idx, rp in enumerate(reg_param_opt):
    if rp <= 0.0:
        first_idx = idx
        break

# Also compute the GD values at those lambdas
# d = 1000
# reps = 5

# alphas_num = []
# gen_error_mean = []
# gen_error_std = []

# for idx, (alpha, rp) in enumerate(zip(alphas, reg_param_opt)):
#     if alpha > 200:
#         continue

#     if idx % 12 != 0:
#         continue

#     all_gen_errors = []
#     for _ in tqdm(range(reps), desc=f"alpha = {alpha}"):
#         xs, ys, xs_train, ys_train, _ = dg.data_generation(
#             dg.measure_gen_decorrelated,
#             d,
#             max(int(np.around(d * alpha)), 1),
#             500,
#             (delta_in, delta_out, percentage, beta),
#         )

#         # xs_train, ys_train, _, _, _ = dg.data_generation(
#         #     dg.measure_gen_decorrelated, d, 100, 1, (delta_in, delta_out, percentage, beta)
#         # )

#         w_hat = erm.find_coefficients_L2(ys, xs, rp)
#         all_gen_errors.append(0.5 * np.mean(np.square(ys_train - (xs_train @ w_hat) / np.sqrt(d))))

#     alphas_num.append(alpha)
#     gen_error_mean.append(np.mean(all_gen_errors))
#     gen_error_std.append(np.std(all_gen_errors))

plt.figure(figsize=(10, 10))

plt.subplot(211)
plt.title(
    "Ridge regression, L2 loss, L2 noise, $\\alpha$ sweep, $\\Delta_{{in}} = {}$, $\\Delta_{{out}} = {}$, $\\epsilon = {}$, $\\beta = {}$".format(
        delta_in, delta_out, percentage, beta
    )
)

color = next(plt.gca()._get_lines.prop_cycler)["color"]
plt.plot(alphas, f_min_vals, color=color, label=r"$E_{gen}$")
plt.plot(alphas, np.arccos(ms / np.sqrt(qs)) / np.pi, label="angle")

# plt.plot(alphas_BO, gen_error_BO_old, color="red", label=r"$E_{gen}$ (BO)")

# plt.errorbar(alphas_num, gen_error_mean, yerr=gen_error_std, marker=".", color=color)
# plt.plot(alphas_BO, 1 - qs_BO, color="green", label=r"$\|w^\star - \hat{w}\|^2$ (BO)")
# plt.plot(alphas, 1 + qs - 2 * ms, label=r"$\|w^\star - \hat{w}\|^2$")

plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"$E_{gen}$")
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(alphas, reg_param_opt, label=r"$\lambda_{opt}$")
# plt.axhline(
#     y=((1 - percentage) * delta_in + percentage * delta_out) / (1 - percentage + beta**2 * percentage), color="red"
# )
# plt.axhline(
#     y=(percentage * (beta**2 - 1) + delta_eff) / (1 - percentage + percentage * beta**2),
#     color="green",
# )
plt.xscale("log")
plt.ylabel(r"$\lambda_{opt}$")
plt.legend()
plt.grid()

plt.show()
