from linear_regression.aux_functions.free_energy import free_energy, Psi_w_L2_reg, Psi_out_L1
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.fixed_point_equations.fpe_L1_loss import (
    f_hat_L1_decorrelated_noise,
)
import numpy as np
import matplotlib.pyplot as plt
from linear_regression.utils.errors import ConvergenceError
from linear_regression.aux_functions.misc import damped_update

blend = 1.0
max_iter = 100000
min_iter = 500
abs_tol = 1e-8

delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
reg_params = [-0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
alpha = 2.0

N = 1000
qs = np.logspace(-2, 4, N)

free_energies = np.empty_like(qs)
ms = np.empty_like(qs)
sigmas = np.empty_like(qs)
m_hats = np.empty_like(qs)
q_hats = np.empty_like(qs)
Σ_hats = np.empty_like(qs)

plt.figure(figsize=(10, 7.5))

for reg_param in reg_params:
    print("Calculating λ = {:>5.2f}".format(reg_param))

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
            while err > abs_tol or iter_nb < min_iter:
                m_hat, _, Σ_hat = f_hat_L1_decorrelated_noise(
                    m, q, sigma, alpha, delta_in, delta_out, percentage, beta
                )
                new_m, _, new_sigma = f_L2_reg(m_hat, 0.0, Σ_hat, reg_param)

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

            free_energies[idx] = free_energy(
                Psi_w_L2_reg,
                Psi_out_L1,
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
        except (ConvergenceError, ValueError) as e:
            ms[idx:] = np.nan
            sigmas[idx:] = np.nan
            m_hats[idx:] = np.nan
            Σ_hats[idx:] = np.nan
            q_hats[idx:] = np.nan
            free_energies[idx:] = np.nan
            break

    while True:
        m = 5 * np.random.random() + 0.01
        q = 5 * np.random.random() + 0.01
        sigma = 5 * np.random.random() + 0.01
        if np.square(m) < q + delta_in * q and np.square(m) < q + delta_out * q:
            break

    m_true, q_true, sigma_true = fixed_point_finder(
        f_L2_reg,
        f_hat_L1_decorrelated_noise,
        (m, q, sigma),
        {"reg_param": reg_param},
        {
            "alpha": alpha,
            "delta_in": delta_in,
            "delta_out": delta_out,
            "percentage": percentage,
            "beta": beta,
        },
    )

    m_hat_true, q_hat_true, Σ_hat_true = f_hat_L1_decorrelated_noise(
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
        Σ_hat_true,
        (reg_param,),
        (delta_in, delta_out, percentage, beta),
    )

    color = next(plt.gca()._get_lines.prop_cycler)["color"]
    # plt.axhline(free_energy_true, linestyle="--", linewidth=0.75, color=color, alpha=0.5)
    # plt.axvline(q_true, linestyle="--", linewidth=0.75, color=color, alpha=0.5)
    plt.plot(q_true, free_energy_true, ".", color=color)
    plt.plot(qs, free_energies, label="$\\lambda$ = " + "{:.2f}".format(reg_param), color=color)


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
    + r" $\epsilon$ = "
    + "{:.2f}".format(percentage)
)

plt.ylabel("Free energy")
plt.xlabel("q")
plt.xscale("log")
plt.legend()
plt.ylim([0.0, 10])
plt.grid()

plt.show()
