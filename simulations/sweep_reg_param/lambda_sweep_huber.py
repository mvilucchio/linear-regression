import linear_regression.sweeps.reg_param_sweep as swreg
import matplotlib.pyplot as plt
import numpy as np
from linear_regression.fixed_point_equations.fpe_L2_regularization import var_func_L2
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    var_hat_func_Huber_decorrelated_noise,
)
from linear_regression.aux_functions.stability_functions import stability_huber


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


alpha = 2.0
alpha = 30.0
delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
a_hub = 0.02 # 0.5 # 0.962

reg_params, (ms, qs, sigmas) = swreg.sweep_reg_param_fixed_point(
    var_func_L2,
    var_hat_func_Huber_decorrelated_noise,
    -0.15, # -0.0345,
    0.01,
    1000,
    {"reg_param": 0.5},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
        "a": a_hub,
    },
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[list(), list(), list()],
    linear=True,
    decreasing=True,
)

plt.title(r"$\alpha = {:.1f} \, a = {:.3f}$ rest as fig 1 right".format(alpha, a_hub))

color = next(plt.gca()._get_lines.prop_cycler)["color"]

plt.plot(
    reg_params,
    stability_huber(
        ms, qs, sigmas, alpha, reg_params, delta_in, delta_out, percentage, beta, a_hub
    ),
    label="AT stability",
    color=color,
)

for idx, sigma in enumerate(sigmas):
    if not np.isnan(sigma):
        first_idx = idx
        break

found_first_non_nan = False
found_first_positive = False

for idx, sigma in enumerate(sigmas):
    if not np.isnan(sigma) and not found_first_non_nan:
        first_non_nan_idx = idx
        found_first_non_nan = True
    if (
        stability_huber(
            ms[idx],
            qs[idx],
            sigmas[idx],
            alpha,
            reg_params[idx],
            delta_in,
            delta_out,
            percentage,
            beta,
            a_hub
        )
        >= 0.0
        and not found_first_positive
    ):
        first_positive_idx = idx
        found_first_positive = True

if found_first_non_nan:
    color = next(plt.gca()._get_lines.prop_cycler)["color"]
    plt.axvline(
        reg_params[first_non_nan_idx],
        label="$\lambda$ = {:.3f}".format(reg_params[first_non_nan_idx]),
        color=color,
    )
    print("first non nan: ", reg_params[first_non_nan_idx])

if found_first_positive:
    color = next(plt.gca()._get_lines.prop_cycler)["color"]
    plt.axvline(
        reg_params[first_positive_idx],
        label="$\lambda$ = {:.3f}".format(reg_params[first_positive_idx]),
        color=color,
    )
    print("first positive: ", reg_params[first_positive_idx])

plt.xlabel(r"$\lambda$")
plt.grid()
plt.legend()

plt.show()
