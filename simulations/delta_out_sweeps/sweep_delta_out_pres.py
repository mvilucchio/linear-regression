import linear_regression.sweeps.delta_out_sweep as dosw
import matplotlib.pyplot as plt
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
from linear_regression.aux_functions.stability_functions import (
    stability_ridge,
    stability_l1_l2,
    stability_huber,
)

def sigma_order_param(m, q, sigma):
    return sigma

def q_order_param(m, q, sigma):
    return q

def m_order_param(m, q, sigma):
    return m

alpha, delta_in, epsilon, beta = 10.0, 1.0, 0.3, 0.0
delta_out_init = 100

while True:
    m = 0.89 * np.random.random() + 0.1
    q = 0.89 * np.random.random() + 0.1
    sigma = 0.89 * np.random.random() + 0.1
    if np.square(m) < q + delta_in * q and np.square(m) < q + delta_out_init * q:
        initial_condition = [m, q, sigma]
        break

print("begin")

delta_outs, e_gen_l2, reg_params_opt_l2, (ms_l2, qs_l2, sigmas_l2) = dosw.sweep_delta_out_optimal_lambda_fixed_point(
    var_func_L2,
    var_hat_func_L2_decorrelated_noise,
    0.01,
    10,
    300,
    0.1,
    {"reg_param": 3.0},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out_init,
        "percentage": 0.3,
        "beta": beta,
    },
    initial_cond_fpe=initial_condition,
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[list(), list(), list()],
    decreasing=True,
)

print("L2 done")

_, e_gen_l1, reg_params_opt_l1, (ms_l1, qs_l1, sigmas_l1) = dosw.sweep_delta_out_optimal_lambda_fixed_point(
    var_func_L2,
    var_hat_func_L1_decorrelated_noise,
    0.01,
    10,
    300,
    0.5,
    {"reg_param": 3.0},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out_init,
        "percentage": 0.3,
        "beta": beta,
    },
    initial_cond_fpe=initial_condition,
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[list(), list(), list()],
    decreasing=True
)

print("L1 done")

_, e_gen_hub, (reg_params_opt_hub, hub_params_opt), (ms_hub, qs_hub, sigmas_hub) = dosw.sweep_delta_out_optimal_lambda_hub_param_fixed_point(
    var_func_L2,
    var_hat_func_Huber_decorrelated_noise,
    0.01,
    10,
    300,
    [0.5, 1.0],
    {"reg_param": 3.0},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out_init,
        "percentage": 0.3,
        "beta": beta,
        "a": 1.0,
    },
    initial_cond_fpe=initial_condition,
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[list(), list(), list()],
    decreasing=True
)

print("Huber done")

stabs_l2 = stability_ridge(ms_l2, qs_l2, sigmas_l2, alpha, reg_params_opt_l2, delta_in, delta_outs, epsilon, beta)
stabs_l1 = stability_l1_l2(ms_l1, qs_l1, sigmas_l1, alpha, reg_params_opt_l1, delta_in, delta_outs, epsilon, beta)
stabs_hub = stability_huber(ms_hub, qs_hub, sigmas_hub, alpha, reg_params_opt_hub, delta_in, delta_outs, epsilon, beta, hub_params_opt)

# ----------------------------

plt.figure(figsize=(10, 10))

plt.subplot(311)
plt.plot(delta_outs, e_gen_l2, label="L2")
plt.plot(delta_outs, e_gen_l1, label="L1")
plt.plot(delta_outs, e_gen_hub, label="Huber")
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$E_{gen}$")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()

plt.subplot(312)
plt.plot(delta_outs, reg_params_opt_l2, label="L2")
plt.plot(delta_outs, reg_params_opt_l1, label="L1")
plt.plot(delta_outs, reg_params_opt_hub, label="Huber lambda")
plt.plot(delta_outs, hub_params_opt, label="Huber a")
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$\lambda_{opt}$")
plt.xscale("log")
plt.legend()
plt.grid()

plt.subplot(313)
plt.plot(delta_outs, stabs_l2, label="L2")
plt.plot(delta_outs, stabs_l1, label="L1")
plt.plot(delta_outs, stabs_hub, label="Huber")
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"Stability cond.")
plt.xscale("log")
plt.legend()
plt.grid()

plt.show()