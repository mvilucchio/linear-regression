from linear_regression.aux_functions.free_energy import (
    free_energy,
    Psi_w_L2_reg,
    Psi_out_L2,
)
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.fpe_L2_regularization import var_func_L2
from linear_regression.fixed_point_equations.fpe_L2_loss import order_parameters_ridge, var_hat_func_L2_decorrelated_noise
import numpy as np
import matplotlib.pyplot as plt
from linear_regression.utils.errors import ConvergenceError
from linear_regression.aux_functions.misc import damped_update
from math import erf

blend = 1.0
max_iter = 100000
min_iter = 10
abs_tol = 1e-7

delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
# minimum lambda is -0.171 fora alpha = 2.0
reg_params = [-0.171, -0.1, 0.0, 0.1]
# reg_params = [-0.1, -0.05]
alpha = 2.0

# qs = np.linspace(0.01, 10.0, 1000)
N = 1000
n_decade = 5
ms = np.logspace(-2, 4, N)
free_energies = np.empty_like(ms)
qs = np.empty_like(ms)
sigmas = np.empty_like(ms)
m_hats = np.empty_like(ms)
q_hats = np.empty_like(ms)
sigma_hats = np.empty_like(ms)

delta_eff = (1 - percentage) * delta_in + percentage * delta_out
intermediate_val = 1 + percentage * (beta - 1)

plt.figure(figsize=(10, 7.5))

for reg_param in reg_params:
    print(reg_param)

    m = ms[0]
    while True:
        q = 100 * np.random.random() + 1e-8
        sigma = 100 * np.random.random() + 1e-8
        if (
            np.square(m) < q + delta_in * q
            and np.square(m) * beta**2 < q * beta**2 + delta_out * q
        ):
            break

    print("found initial values")

    for idx, m in enumerate(ms):

        iter_nb = 0
        err = 100.0
        while err > abs_tol or iter_nb < min_iter:
            m_hat, q_hat, sigma_hat = var_hat_func_L2_decorrelated_noise(
                m, q, sigma, alpha, delta_in, delta_out, percentage, beta
            )
            _, new_q, new_sigma = var_func_L2(m_hat, 0.0, sigma_hat, reg_param)

            err = max([abs(new_q - q), abs(new_sigma - sigma)])

            q = damped_update(new_q, q, blend)
            sigma = damped_update(new_sigma, sigma, blend)

            iter_nb += 1
            if iter_nb > max_iter:
                raise ConvergenceError("fixed_point_finder", iter_nb)
            
        qs[idx] = q
        sigmas[idx] = sigma
        m_hats[idx] = m_hat
        sigma_hats[idx] = sigma_hat
        q_hats[idx] = q_hat

        # print(m, q, sigma, m_hat, q_hat, sigma_hat)

        free_energies[idx] = free_energy(
            Psi_w_L2_reg,
            Psi_out_L2,
            alpha,
            m,
            q,
            sigma,
            m_hat,
            q_hat,
            sigma_hat,
            (reg_param,),
            (delta_in, delta_out, percentage, beta),
        )
        # print(idx, free_energies[idx])

    m_true, q_true, sigma_true, m_hat_true, q_hat_true, sigma_hat_true = order_parameters_ridge(alpha, reg_param, delta_in, delta_out, percentage, beta)

    free_energy_true = free_energy(
        Psi_w_L2_reg,
        Psi_out_L2,
        alpha,
        m_true, q_true, sigma_true, m_hat_true, q_hat_true, sigma_hat_true,
        (reg_param,),
        (delta_in, delta_out, percentage, beta),
    )

    color = next(plt.gca()._get_lines.prop_cycler)['color']

    # plt.plot(qs, free_energies - np.amin(free_energies), '-', label="$\\lambda$ = " + "{:.2f}".format(reg_param), color=color)
    plt.plot(ms, free_energies, '.', label="$\\lambda$ = " + "{:.2f}".format(reg_param), color=color)
    plt.axhline(free_energy_true, linestyle="--", color=color)
    plt.axvline(m_true, linestyle="--", color=color)

plt.title(
    "Ridge "
    + r"$\Delta_{in}$ = "
    + "{:.2f}".format(delta_in)
    + r" $\Delta_{out}$ ="
    + "{:.2f}".format(delta_out)
    + r" $\alpha$ = "
    + "{:.2f}".format(alpha)
    + r" $\beta$ = "
    + "{:.2f}".format(beta)
)

# plt.plot(qs, ms, label="m")
# plt.plot(qs, sigmas, label="sigma")
# plt.plot(qs, m_hats, label="m_hat")
# plt.plot(qs, q_hats, label="q_hat")
# plt.plot(qs, sigma_hats, label="sigma_hat")

plt.ylabel("Free energy")
plt.xlabel("m")
plt.xscale("log")
# plt.yscale("log")
plt.legend()
plt.grid()

plt.show()
