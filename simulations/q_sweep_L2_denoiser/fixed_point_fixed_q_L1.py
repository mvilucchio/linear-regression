from linear_regression.aux_functions.free_energy import (
    free_energy,
    Psi_w_L2_reg,
    Psi_out_L1,
    Psi_out_L2,
)
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.fixed_point_equations.fpe_L1_loss import (
    f_hat_L1_decorrelated_noise,
)
import numpy as np
import matplotlib.pyplot as plt
from linear_regression.utils.errors import ConvergenceError
from linear_regression.aux_functions.misc import damped_update



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

blend = 0.85
max_iter = 1000
min_iter = 50
abs_tol = 1e-8

delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
reg_params = [-0.03, -0.01, 0.0, 0.01, 0.03]
reg_params = [0.0]
alpha = 2.0

N = 1000
# qs = np.logspace(-2, 5, N)
qs = np.linspace(0.01, 1000, N)
free_energies = np.empty_like(qs)
ms = np.empty_like(qs)
sigmas = np.empty_like(qs)
m_hats = np.empty_like(qs)
q_hats = np.empty_like(qs)
sigma_hats = np.empty_like(qs)

plt.figure(figsize=(10, 7.5))

for reg_param in reg_params:
    print("reg param ", reg_param)
    q = qs[0]
    while True:
        m = 10 * np.random.random() + 0.01
        sigma = 10 * np.random.random() + 0.01
        if np.square(m) < q + delta_in * q and np.square(m) < q + delta_out * q:
            break

    for idx, q in enumerate(qs):
        try:
            iter_nb = 0
            err = 100.0
            m_hat, q_hat, sigma_hat = f_hat_L1_decorrelated_noise(
                m, q, sigma, alpha, delta_in, delta_out, percentage, beta
            )
            while err > abs_tol or iter_nb < min_iter:
                new_m_hat, new_q_hat, new_sigma_hat = f_hat_L1_decorrelated_noise(
                    m, q, sigma, alpha, delta_in, delta_out, percentage, beta
                )
                # print("hat    ", m_hat, q_hat, sigma_hat)
                new_m, _, new_sigma = f_L2_reg(new_m_hat, new_q_hat, new_sigma_hat, reg_param)
                # print("non hat ", m, q, sigma)

                err = max(
                    [
                        abs(new_m - m),
                        abs(new_sigma - sigma),
                        abs(new_sigma_hat - sigma_hat),
                        abs(new_m_hat - m_hat),
                        abs(new_q_hat - q_hat),
                    ]
                )

                m = damped_update(new_m, m, blend)
                sigma = damped_update(new_sigma, sigma, blend)
                m_hat = damped_update(new_m_hat, m_hat, blend)
                q_hat = damped_update(new_q_hat, q_hat, blend)
                sigma_hat = damped_update(new_sigma_hat, sigma_hat, blend)

                iter_nb += 1
                if iter_nb > max_iter:
                    print(new_m, new_sigma, new_m_hat, new_q_hat, new_sigma_hat)
                    new_m_hat, new_q_hat, new_sigma_hat = f_hat_L1_decorrelated_noise(
                        m, q, sigma, alpha, delta_in, delta_out, percentage, beta
                    )
                    new_m, _, new_sigma = f_L2_reg(new_m_hat, new_q_hat, new_sigma_hat, reg_param)
                                                      
                    print(new_m, new_sigma, new_m_hat, new_q_hat, new_sigma_hat)
                    raise ConvergenceError("fixed_point_finder", iter_nb)

            # q_hat = q * (reg_param + sigma_hat) ** 2 - m_hat**2

            ms[idx] = m
            sigmas[idx] = sigma
            m_hats[idx] = m_hat
            sigma_hats[idx] = sigma_hat
            q_hats[idx] = q_hat

            # print(m, q, sigma, m_hat, q_hat, sigma_hat)

            free_energies[idx] = free_energy(
                Psi_w_L2_reg,
                Psi_out_L1,
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
        except (ConvergenceError, ValueError) as e:
            print(e)

            ms[idx:] = np.nan
            sigmas[idx:] = np.nan
            m_hats[idx:] = np.nan
            sigma_hats[idx:] = np.nan
            q_hats[idx:] = np.nan
            free_energies[idx:] = np.nan
            break

    while True:
        m = 10 * np.random.random() + 0.01
        q = 10 * np.random.random() + 0.01
        sigma = 10 * np.random.random() + 0.01
        if np.square(m) < q + delta_in * q and np.square(m) < q + delta_out * q:
            break
    
    min_idx = np.argmin(free_energies)

    m_true, q_true, sigma_true = fixed_point_finder(
        f_L2_reg,
        f_hat_L1_decorrelated_noise,
        (ms[min_idx], qs[min_idx], sigmas[min_idx]),
        {"reg_param": reg_param},
        {
            "alpha": alpha,
            "delta_in": delta_in,
            "delta_out": delta_out,
            "percentage": percentage,
            "beta": beta,
        },
    )

    m_hat_true, q_hat_true, sigma_hat_true = f_hat_L1_decorrelated_noise(
        m_true, q_true, sigma_true, alpha, delta_in, delta_out, percentage, beta
    )

    free_energy_true = free_energy(
        Psi_w_L2_reg,
        Psi_out_L1,
        alpha,
        m_true,
        q_true,
        sigma_true,
        m_hat_true,
        q_hat_true,
        sigma_hat_true,
        (reg_param,),
        (delta_in, delta_out, percentage, beta),
    )

    print("difference from min and true", "{:.2E}".format(abs(free_energies[min_idx] - free_energy_true)), "{:.2E}".format(abs(qs[min_idx] - q_true)))
    _, closest_true_idx = find_nearest(qs, q_true)
    # print("true values ", m_true, q_true, sigma_true, m_hat_true, q_hat_true, sigma_hat_true)
    print("difference from the FPE (un)constrained ", m_true, ms[closest_true_idx], "{:.2E} {:.2E}".format(abs(ms[closest_true_idx] - m_true), abs(free_energies[closest_true_idx] - free_energy_true)))
    # print(np.amin(free_energies), free_energy_true)

    color = next(plt.gca()._get_lines.prop_cycler)["color"]
    # plt.plot(qs, free_energies - (free_energy_true - 1e-2), '.', label="$\\lambda$ = " + "{:.2f}".format(reg_param), color=color)

    # plot the true values
    plt.axhline(free_energy_true, linestyle="--", color=color, alpha=0.5)
    plt.axvline(q_true, linestyle="--", color=color, alpha=0.5)

    # plot the sweeps
    plt.plot(qs, free_energies, label="$\\lambda$ = " + "{:.2f}".format(reg_param), color=color)

    # plot the order parameters
    # plt.plot(qs, ms, label="m")
    # plt.plot(qs, sigmas, label="sigma")
    # plt.plot(qs, m_hats, label="m_hat")
    # plt.plot(qs, m_hats, label="q_hat")
    # plt.plot(qs, sigma_hats, label="sigma_hat")


plt.title(
    "L1 "
    + r"$\Delta_{in}$ = "
    + "{:.2f}".format(delta_in)
    + r" $\Delta_{out}$ ="
    + "{:.2f}".format(delta_out)
    + r" $\alpha$ = "
    + "{:.2f}".format(alpha)
    + r" $\beta$ = "
    + "{:.2f}".format(beta)
    + r" $\lambda$ = "
    + "{:.2f}".format(reg_param)
)

plt.ylabel("Free energy")
plt.xlabel("q")
plt.xscale("log")
plt.yscale("log")
plt.legend()
# plt.ylim([0.0, 30])
plt.grid()

plt.show()
