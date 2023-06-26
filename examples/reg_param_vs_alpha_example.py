import numpy as np
import matplotlib.pyplot as plt
import linear_regression.sweeps.alpha_sweeps as alsw
from linear_regression.fixed_point_equations.fpe_L2_regularization import var_func_L2
from linear_regression.fixed_point_equations.fpe_L2_loss import (
    var_hat_func_L2_decorrelated_noise,
)


def e_gen(m, q, sigma):
    return 1 - 2 * m + q**2


alpha_min, alpha_max = 0.1, 10
delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0


(alphas, reg_params), (eg,) = alsw.sweep_alpha_descend_lambda(
    var_func_L2,
    var_hat_func_L2_decorrelated_noise,
    alpha_min,
    alpha_max,
    100,
    -5,
    0.5,
    500,
    {"reg_param": 3.0},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
    funs=[e_gen],
    funs_args=[list()],
    initial_cond_fpe=(0.6, 0.01, 0.9),
)

alphas_ls, last_reg_param_stable = alsw.sweep_alpha_minimal_stable_reg_param(
    var_func_L2,
    var_hat_func_L2_decorrelated_noise,
    alpha_min,
    alpha_max,
    100,
    {"reg_param": 3.0},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
    bounds_reg_param_search=(-100.0, 0.01),
    points_per_run=10000,
)

fig, ax = plt.subplots()

AA, RR = np.meshgrid(alphas, reg_params)
ax.contourf(AA, RR, eg, cmap="jet")
fig.colorbar(ax.contourf(AA, RR, eg, 200, cmap="jet"))

ax.autoscale(False)
ax.plot(alphas, -((1 - np.sqrt(alphas)) ** 2), zorder=1, color="tab:blue", label="condition MP")
ax.plot(
    alphas_ls, last_reg_param_stable, zorder=1, color="tab:green", label=r"last $\lambda_{stab}$"
)

ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\lambda$")
ax.set_title(r"$\Sigma$")
ax.set_xscale("log")
ax.legend()

plt.show()
