from linear_regression.aux_functions.free_energy import (
    free_energy,
    Psi_w_L2_reg,
    Psi_w_projection_denoising,
    Psi_out_L1,
    Psi_out_L2
)
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.fpe_projection_denoising import f_projection_denoising
from linear_regression.fixed_point_equations.fpe_L2_loss import f_hat_L2_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_L1_loss import (
    f_hat_L1_decorrelated_noise,
)
import numpy as np
import matplotlib.pyplot as plt
from linear_regression.utils.errors import ConvergenceError
from linear_regression.aux_functions.misc import damped_update
from math import erf

blend = 1.0
max_iter = 100000
min_iter = 100
abs_tol = 1e-8

delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
alpha = 2.0

N = 1000
qs = np.logspace(-1, 2, N)
free_energies = np.empty_like(qs)
ms = np.empty_like(qs)
sigmas = np.empty_like(qs)
m_hats = np.empty_like(qs)
q_hats = np.empty_like(qs)
Σ_hats = np.empty_like(qs)

plt.figure(figsize=(10, 7.5))

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
            m_hat, q_hat, Σ_hat = f_hat_L2_decorrelated_noise(
                m, q, sigma, alpha, delta_in, delta_out, percentage, beta
            )
            new_m, new_q, new_sigma = f_projection_denoising(m_hat, q_hat, Σ_hat, q)

            # print(new_q, q)

            err = max([abs(new_m - m), abs(new_sigma - sigma)])

            m = damped_update(new_m, m, blend)
            sigma = damped_update(new_sigma, sigma, blend)

            iter_nb += 1
            if iter_nb > max_iter:
                raise ConvergenceError("fixed_point_finder", iter_nb)

        ms[idx] = m
        sigmas[idx] = sigma
        m_hats[idx] = m_hat
        Σ_hats[idx] = Σ_hat
        q_hats[idx] = q_hat

        # print(m, q, sigma, m_hat, q_hat, Σ_hat)

        free_energies[idx] = free_energy(
            Psi_w_projection_denoising,
            Psi_out_L2,
            alpha,
            m,
            q,
            sigma,
            m_hat,
            q_hat,
            Σ_hat,
            (q,),
            (delta_in, delta_out, percentage, beta),
        )
        # print(q, free_energies[idx])
        # print(idx, free_energies[idx])
    except (ConvergenceError, ValueError) as e:
        ms[idx:] = np.nan
        sigmas[idx:] = np.nan
        m_hats[idx:] = np.nan
        Σ_hats[idx:] = np.nan
        q_hats[idx:] = np.nan
        free_energies[idx:] = np.nan
        break

while True:
    m = 1 * np.random.random() + 0.01
    q = 1 * np.random.random() + 0.01
    sigma = 1 * np.random.random() + 0.01
    if np.square(m) < q + delta_in * q and np.square(m) < q + delta_out * q:
        break

print("---> ", qs[np.argmin(free_energies)])

m_true, q_true, sigma_true = fixed_point_finder(
    f_projection_denoising,
    f_hat_L2_decorrelated_noise,
    (m, q, sigma),
    {"q_fixed": q},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
)

m_hat_true, q_hat_true, Σ_hat_true = f_hat_L2_decorrelated_noise(
    m_true, q_true, sigma_true, alpha, delta_in, delta_out, percentage, beta
)

free_energy_true = free_energy(
    Psi_w_projection_denoising,
    Psi_out_L2,
    alpha,
    m_true,
    q_true,
    sigma_true,
    m_hat_true,
    q_hat_true,
    Σ_hat_true,
    (q,),
    (delta_in, delta_out, percentage, beta),
)
# print(np.amin(free_energies), free_energy_true)

color = next(plt.gca()._get_lines.prop_cycler)["color"]
# plt.plot(qs, free_energies - (free_energy_true - 1e-2), '.', label="$\\lambda$ = " + "{:.2f}".format(reg_param), color=color)
plt.axhline(free_energy_true, linestyle="--", color=color, alpha=0.5)
plt.axvline(q_true, linestyle="--", color=color, alpha=0.5)
plt.plot(qs, free_energies,  color=color)


plt.title(
    "L2 loss Projection Denoising "
    + r"$\Delta_{in}$ = "
    + "{:.2f}".format(delta_in)
    + r" $\Delta_{out}$ ="
    + "{:.2f}".format(delta_out)
    + r" $\alpha$ = "
    + "{:.2f}".format(alpha)
    + r" $\beta$ = "
    + "{:.2f}".format(beta)
)

plt.ylabel("Free energy")
plt.xlabel("q")
plt.xscale("log")
plt.legend()
# plt.ylim([0.0, 10])
plt.grid()

plt.show()
