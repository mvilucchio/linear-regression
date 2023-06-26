import linear_regression.sweeps.alpha_sweeps as alsw
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import linear_regression.regression_numerics.data_generation as dg
import linear_regression.regression_numerics.erm_solvers as erm
from linear_regression.fixed_point_equations.fpe_L2_loss import (
    var_hat_func_L2_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_L1_loss import (
    var_hat_func_L1_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    var_hat_func_Huber_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_L2_regularization import var_func_L2
import numpy as np
from linear_regression.aux_functions.misc import excess_gen_error
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


delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.1, 0.0

(
    alphas,
    f_min_vals,
    (reg_param_opt, hub_params_opt),
    (sigmas, qs, ms),
) = alsw.sweep_alpha_optimal_lambda_hub_param_fixed_point(
    var_func_L2,
    var_hat_func_Huber_decorrelated_noise,
    0.01,
    10000,
    150,
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
    f_min=excess_gen_error,
    f_min_args=(delta_in, delta_out, percentage, beta),
    funs=[sigma_order_param, q_order_param, m_order_param],
    funs_args=[{}, {}, {}],
)

first_idx = 0
for idx, rp in enumerate(reg_param_opt):
    if rp <= 0.0:
        first_idx = idx
        break

# Also compute the GD values at those lambdas
d = 1000
reps = 5

alphas_num = []
gen_error_mean = []
gen_error_std = []

for idx, (alpha, rp, a) in enumerate(zip(alphas, reg_param_opt, hub_params_opt)):
    if alpha > 200:
        continue

    if idx % 12 != 0:
        continue

    all_gen_errors = []
    for _ in tqdm(range(reps), desc=f"alpha = {alpha}"):
        xs, ys, xs_train, ys_train, _ = dg.data_generation(
            dg.measure_gen_decorrelated,
            d,
            max(int(np.around(d * alpha)), 1),
            500,
            (delta_in, delta_out, percentage, beta),
        )

        # xs_train, ys_train, _, _, _ = dg.data_generation(
        #     dg.measure_gen_decorrelated, d, 100, 1, (delta_in, delta_out, percentage, beta)
        # )

        w_hat = erm.find_coefficients_Huber(ys, xs, rp, a)
        all_gen_errors.append(0.5 * np.mean(np.square(ys_train - (xs_train @ w_hat) / np.sqrt(d))))

    alphas_num.append(alpha)
    gen_error_mean.append(np.mean(all_gen_errors))
    gen_error_std.append(np.std(all_gen_errors))

plt.figure(figsize=(10, 10))

plt.subplot(311)
plt.title("Huber regression, Huber loss, L2 noise, $\\alpha$ sweep, $\\Delta_{{in}} = {}$, $\\Delta_{{out}} = {}$, $\\beta = {}$".format(delta_in, delta_out, beta))

# plt.plot(alphas, f_min_vals)

color = next(plt.gca()._get_lines.prop_cycler)["color"]
plt.plot(alphas, f_min_vals, color=color, label=r"$E_{gen}$")
plt.errorbar(alphas_num, gen_error_mean, yerr=gen_error_std, marker=".", color=color)
plt.plot(alphas, 1 + qs - 2 * ms, label=r"$\|w^\star - \hat{w}\|^2$")
plt.plot(alphas, np.arccos(ms / np.sqrt(qs)) / np.pi, label="angle")
plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"$E_{gen}$")
plt.legend()
plt.grid()

plt.subplot(312)
plt.plot(alphas, reg_param_opt, label=r"$\lambda_{opt}$")
plt.plot(alphas, hub_params_opt, label=r"$a_{opt}$")
# plt.axvline(alphas[first_idx], color="red")
plt.xscale("log")
plt.ylabel(r"$\lambda_{opt}$")
plt.legend()
plt.grid()

plt.subplot(313)
# plt.plot(alphas, sigmas, label=r"$\Sigma$")
# plt.plot(alphas, 1 - alphas * (sigmas / (sigmas + 1))**2, label=r"$1 - \alpha \Sigma^2 / (\Sigma + 1)^2$")
plt.plot(alphas, stability_huber(ms, qs, sigmas, alphas, reg_param_opt, delta_in, delta_out, percentage, beta, hub_params_opt), label=r"Stability")
plt.legend()
# plt.axvline(alphas[first_idx], color="red")
plt.xscale("log")
plt.grid()
# plt.ylabel(r"$1 - \alpha \Sigma^2 / (\Sigma + 1)^2$")
plt.ylabel("Stability cond.")
plt.xlabel(r"$\alpha$")

plt.show()

np.savetxt(
    "./simulations/data/TEST_alpha_sweep_Huber.csv",
    np.array([alphas, f_min_vals, reg_param_opt, hub_params_opt]).T,
    delimiter=",",
    header="alpha,f_min,lambda_opt,hub_param_opt",
)
