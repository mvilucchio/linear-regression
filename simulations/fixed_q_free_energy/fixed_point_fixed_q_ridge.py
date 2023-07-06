from linear_regression.aux_functions.free_energy import (
    free_energy,
    Psi_w_L2_reg,
    Psi_out_L2,
)
from linear_regression.fixed_point_equations.fpe_L2_loss import f_hat_L2_decorrelated_noise
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
import numpy as np
import matplotlib.pyplot as plt
from linear_regression.utils.errors import ConvergenceError
from linear_regression.aux_functions.misc import damped_update
from math import erf


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


blend = 1.0
max_iter = 100000
min_iter = 10
abs_tol = 1e-7

delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
# minimum lambda is -0.171 fora alpha = 2.0
reg_params = [0.001, 0.01]
# reg_params = [-0.1, -0.05]
alpha = 0.5

# qs = np.linspace(0.01, 10.0, 1000)
N = 1000
n_decade = 5
qs = np.logspace(0, 4, N)
free_energies = np.empty_like(qs)
ms = np.empty_like(qs)
sigmas = np.empty_like(qs)
m_hats = np.empty_like(qs)
q_hats = np.empty_like(qs)
Σ_hats = np.empty_like(qs)

delta_eff = (1 - percentage) * delta_in + percentage * delta_out
intermediate_val = 1 + percentage * (beta - 1)

plt.figure(figsize=(10, 7.5))

for reg_param in reg_params:
    # print("reg_param ", reg_param)
    q = qs[0]
    while True:
        m = 100 * np.random.random() + 1e-8
        sigma = 100 * np.random.random() + 1e-8
        if (
            np.square(m) < q + delta_in * q
            and np.square(m) * beta**2 < q * beta**2 + delta_out * q
        ):
            break
    # print("found initial values")

    for idx, q in enumerate(qs):

        iter_nb = 0
        err = 100.0
        while err > abs_tol or iter_nb < min_iter:
            m_hat = alpha * intermediate_val / (1 + sigma)
            Σ_hat = alpha / (1 + sigma)

            new_m = m_hat / (reg_param + Σ_hat)
            new_sigma = 1 / (reg_param + Σ_hat)

            err = max([abs(new_m - m), abs(new_sigma - sigma)])

            m = damped_update(new_m, m, blend)
            sigma = damped_update(new_sigma, sigma, blend)

            iter_nb += 1
            if iter_nb > max_iter:
                raise ConvergenceError("fixed_point_finder", iter_nb)

        q_hat = q * (reg_param + Σ_hat) ** 2 - m_hat**2

        ms[idx] = m
        sigmas[idx] = sigma
        m_hats[idx] = m_hat
        Σ_hats[idx] = Σ_hat
        q_hats[idx] = q_hat

        # print(m, q, sigma, m_hat, q_hat, Σ_hat)

        free_energies[idx] = free_energy(
            Psi_w_L2_reg,
            Psi_out_L2,
            alpha,
            m,
            q,
            sigma,
            m_hat,
            q_hat,
            Σ_hat,
            (reg_param,),
            (delta_in, delta_out, percentage, beta),
        )
        # print(idx, free_energies[idx])

    # m_true, q_true, sigma_true, m_hat_true, q_hat_true, Σ_hat_true = order_parameters_ridge(
    #     alpha, reg_param, delta_in, delta_out, percentage, beta
    # )
    m_true, q_true, sigma_true = fixed_point_finder(
        f_L2_reg,
        f_hat_L2_decorrelated_noise,
        (m,q,sigma),
        {"reg_param":reg_param},
        {"alpha" : alpha, "delta_in":delta_in, "delta_out":delta_out, "percentage":percentage, "beta":beta},
    )
    m_hat_true, q_hat_true, Σ_hat_true = f_hat_L2_decorrelated_noise(m_true, q_true, sigma_true, alpha, delta_in, delta_out, percentage, beta)

    _, closest_true_idx = find_nearest(qs, q_true)
    print("true values ", m_true, q_true, sigma_true, m_hat_true, q_hat_true, Σ_hat_true)
    print("difference ",  ms[closest_true_idx], m_true, abs(ms[closest_true_idx] - m_true))

    free_energy_true = free_energy(
        Psi_w_L2_reg,
        Psi_out_L2,
        alpha,
        m_true,
        q_true,
        sigma_true,
        m_hat_true,
        q_hat_true,
        Σ_hat_true,
        (reg_param,),
        (delta_in, delta_out, percentage, beta),
    )

    min_idx = np.argmin(free_energies)
    print("-- lambda {:.2f} --> ".format(reg_param), q_true, Σ_hat + reg_param, free_energy_true)
    print(abs(free_energies[min_idx] - free_energy_true), abs(qs[min_idx] - q_true))
    color = next(plt.gca()._get_lines.prop_cycler)["color"]

    # plt.plot(qs, free_energies - np.amin(free_energies), '-', label="$\\lambda$ = " + "{:.2f}".format(reg_param), color=color)
    plt.plot(
        qs, free_energies, ".", label="$\\lambda$ = " + "{:.2f}".format(reg_param), color=color
    )
    # plt.axhline(free_energy_true, linestyle="--", color=color)
    # plt.axvline(q_true, linestyle="--", color=color)

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
# plt.plot(qs, Σ_hats, label="Σ_hat")

plt.ylabel("Free energy")
plt.xlabel("q")
plt.xscale("log")
# plt.yscale("log")
plt.legend()
plt.grid()

plt.show()
