import numpy as np
import matplotlib.pyplot as plt
import linear_regression.sweeps.alpha_sweeps as alsw
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.fixed_point_equations.fpe_L2_loss import (
    f_hat_L2_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    f_hat_Huber_decorrelated_noise,
)


def condition_MP(alphas):
    return -((1 - np.sqrt(alphas)) ** 2)


def sigma_order_param(m, q, sigma):
    return sigma


alpha_min, alpha_max = 0.1, 100
delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0


(
    alphas_Huber,
    e_gen_Huber,
    (reg_param_opt_Huber, _),
    (sigmas_Huber,),
) = alsw.sweep_alpha_optimal_lambda_hub_param_fixed_point(
    f_L2_reg,
    f_hat_Huber_decorrelated_noise,
    alpha_min,
    alpha_max,
    250,
    [1.0, 1.0],
    {"reg_param": 3.0},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
        "a": 1.0,
    },
    initial_cond_fpe=(0.6, 0.2, 0.9),
    funs=[sigma_order_param],
    funs_args=[{}],
)

alphas_L2, e_gen_L2, reg_param_opt_L2, (sigmas_L2,) = alsw.sweep_alpha_optimal_lambda_fixed_point(
    f_L2_reg,
    f_hat_L2_decorrelated_noise,
    alpha_min,
    alpha_max,
    200,
    3.0,
    {"reg_param": 3.0},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[sigma_order_param],
    funs_args=[{}],
)

plt.figure(figsize=(10, 10))

plt.subplot(311)
plt.plot(alphas_Huber, e_gen_Huber, label="Huber")
plt.plot(alphas_L2, e_gen_L2, label="L2")
plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"$E_{gen}$")
plt.legend()
plt.grid()

plt.subplot(312)
plt.plot(alphas_Huber, reg_param_opt_Huber, label=r"$\lambda_{opt}$ Huber")
plt.plot(alphas_L2, reg_param_opt_L2, label=r"$\lambda_{opt}$ L2")
plt.plot(alphas_L2, condition_MP(alphas_L2), label=r"$-(1-\sqrt{\alpha})^2$")
plt.xscale("log")
plt.ylim([-30, 8])
plt.ylabel(r"$\lambda_{opt}$")
plt.legend()
plt.grid()

plt.subplot(313)
plt.plot(alphas_Huber, sigmas_Huber, label="Huber")
plt.plot(alphas_L2, sigmas_L2, label="L2")
plt.xscale("log")
plt.ylabel(r"$\Sigma$")
plt.xlabel(r"$\alpha$")
plt.legend()
plt.grid()

plt.show()
