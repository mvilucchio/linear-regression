import linear_regression.sweeps.alpha_sweeps as alsw
import linear_regression.sweeps.eps_sweep as epsw
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
from linear_regression.aux_functions.misc import estimation_error, excess_gen_error, gen_error_BO
import numpy as np


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


alpha, delta_in, delta_out, beta = 10.0, 1.0, 2.0, 0.1
eps_min, eps_max, n_eps_pts = 0.000001, 0.001, 700
n_eps_pts_BO = 200
# n_eps_pts = n_eps_pts_BO

while True:
    m = 0.89 * np.random.random() + 0.1
    q = 0.89 * np.random.random() + 0.1
    sigma = 0.89 * np.random.random() + 0.1
    if np.square(m) < q + delta_in * q and np.square(m) < q + delta_out * q:
        initial_condition = [m, q, sigma]
        break

epsilons_l2, e_gen_l2, reg_params_opt_l2, (ms_l2, qs_l2, sigmas_l2) = epsw.sweep_eps_optimal_lambda_fixed_point(
    f_L2_reg,
    f_hat_L2_decorrelated_noise,
    eps_min,
    eps_max,
    n_eps_pts,
    delta_in,
    {"reg_param": 3.0},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": 0.3,
        "beta": beta,
    },
    initial_cond_fpe=initial_condition,
    f_min=excess_gen_error,
    f_min_args={"delta_in": delta_in, "delta_out": delta_out, "percentage": 0.3, "beta": beta},
    update_f_min_args=True,
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[{}, {}, {}],
)

print("L2 done")


# ----------------------------

plt.figure(figsize=(7, 7))

plt.subplot(211)
plt.title(
    r"$\alpha = {}$, $\beta = {}$, $\Delta_{{in}} = {}$, $\Delta_{{in}} = {}$".format(alpha, beta, delta_in, delta_out)
)
plt.plot(epsilons_l2, e_gen_l2, label="L2")

plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$E_{gen}^{excess}$")
plt.ylim(1e-2, 1e0)
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()

reg_param_0 = delta_in

coeff_reg_1 = (
    (1 + beta**2 - delta_in + delta_out + 2 * reg_param_0 - 2 * beta * (1 + reg_param_0))
    * (alpha**2 + 2 * alpha * (-1 + reg_param_0) + (1 + reg_param_0) ** 2)
) / (alpha**2 + alpha * (-2 + 3 * delta_in - reg_param_0) + (1 + 3 * delta_in - 2 * reg_param_0) * (1 + reg_param_0))
print(coeff_reg_1)

x, y = epsilons_l2, reg_params_opt_l2 - delta_in

def fun(x, a, b):
    return a * x + b

popt, pcov = curve_fit(fun, x, y)

print(popt)

plt.subplot(212)
plt.plot(epsilons_l2, reg_params_opt_l2 - delta_in, label="L2")

plt.xlabel(r"$\epsilon$")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()


plt.show()
