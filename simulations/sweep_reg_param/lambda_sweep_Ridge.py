import linear_regression.sweeps.reg_param_sweep as swreg
import matplotlib.pyplot as plt
import numpy as np
from linear_regression.aux_functions.stability_functions import stability_ridge
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.fixed_point_equations.fpe_L2_loss import (
    f_hat_L2_decorrelated_noise,
    order_parameters_ridge
)

alpha = 2.0
delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0

reg_params, (sigmas,) = swreg.sweep_reg_param_fixed_point(
    f_L2_reg,
    f_hat_L2_decorrelated_noise,
    -0.2,
    0.01,
    500,
    {"reg_param": 0.5},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
    funs=[lambda m, q, sigma: sigma],
    funs_args=[{}],
    linear=True,
    decreasing=True
)

n_points = 1000
reg_params_anal = np.linspace(0.01, -0.2, n_points)

sigmas_anal = np.empty(n_points)
for idx, rp in enumerate(reg_params_anal):
    _, _, sigmas_anal[idx], _, _, _ = order_parameters_ridge(alpha, rp, delta_in, delta_out, percentage, beta)

plt.title(r"L2 Loss L2 Regularization $\alpha$ = {:.2f}".format(alpha))
plt.plot(reg_params, stability_ridge(1.0, 1.0, sigmas, alpha, 1.0, delta_in, delta_out, percentage, beta), label="Stability")
plt.xlabel(r"$\lambda$")
plt.grid()

for idx, sigma in enumerate(sigmas):
    if not np.isnan(sigma):
        first_idx = idx
        break

plt.axvline(reg_params[first_idx], color="g", label="$\lambda$ = {:.3f}".format(reg_params[first_idx]))
plt.legend()

plt.show()
