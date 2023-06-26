import linear_regression.sweeps.reg_param_sweep as swreg
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from linear_regression.fixed_point_equations.fpe_L2_loss import (
    var_hat_func_L2_decorrelated_noise,
    order_parameters_ridge,
)
from linear_regression.fixed_point_equations.fpe_L1_loss import var_hat_func_L1_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    var_hat_func_Huber_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_L2_regularization import var_func_L2
from linear_regression.aux_functions.stability_functions import stability_l1_l2


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


alpha = 2.0  # 10
delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0

reg_params, (ms, qs, sigmas) = swreg.sweep_reg_param_fixed_point(
    var_func_L2,
    var_hat_func_L1_decorrelated_noise,
    -0.1,
    0.1,
    1000,
    {"reg_param": 0.5},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[list(), list(), list()],
    linear=True,
    decreasing=True,
)

plt.figure(figsize=(10, 7.5))
plt.title(
    r"L1 Loss L2 Regularization $\alpha$ = {:.2f} $\Delta_{{IN}}$ = {:.2f} $\Delta_{{OUT}}$ = {:.3f} $\epsilon$ = {:.2f} $\beta$ = {:.2f}".format(
        alpha, delta_in, delta_out, percentage, beta
    )
),

plt.plot(
    reg_params,
    stability_l1_l2(ms, qs, sigmas, alpha, reg_params, delta_in, delta_out, percentage, beta),
    label="Stability",
)

found_first_non_nan = False
found_first_positive = False

for idx, sigma in enumerate(sigmas):
    if not np.isnan(sigma) and not found_first_non_nan:
        first_non_nan_idx = idx
        found_first_non_nan = True
    if (
        stability_l1_l2(
            ms[idx],
            qs[idx],
            sigmas[idx],
            alpha,
            reg_params[idx],
            delta_in,
            delta_out,
            percentage,
            beta,
        )
        >= 0.0
        and not found_first_positive
    ):
        first_positive_idx = idx
        found_first_positive = True

if found_first_non_nan:
    plt.axvline(
        reg_params[first_non_nan_idx],
        color="g",
        linestyle="--",
        label="$\lambda$ = {:.3f}".format(reg_params[first_non_nan_idx]),
    )
    
if found_first_positive:
    plt.axvline(
        reg_params[first_positive_idx],
        color="r",
        linestyle="--",
        label="$\lambda$ = {:.3f}".format(reg_params[first_positive_idx]),
    )

print(reg_params[first_non_nan_idx], reg_params[first_positive_idx])

plt.xlabel(r"$\lambda$")
plt.grid()
plt.legend()

plt.show()
