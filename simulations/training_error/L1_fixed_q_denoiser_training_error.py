from linear_regression.aux_functions.free_energy import (
    free_energy,
    Psi_w_L2_reg,
    Psi_w_projection_denoising,
    Psi_out_L1,
    Psi_out_L2
)
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.aux_functions.training_errors import training_error_l1_loss
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
alpha = 5.0

N = 5000
qs =  np.logspace(-2, 4, N)
training_error = np.empty_like(qs)
ms = np.empty_like(qs)
sigmas = np.empty_like(qs)
m_hats = np.empty_like(qs)
q_hats = np.empty_like(qs)
sigma_hats = np.empty_like(qs)

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
            m_hat, q_hat, sigma_hat = f_hat_L2_decorrelated_noise(
                m, q, sigma, alpha, delta_in, delta_out, percentage, beta
            )
            new_m, new_q, new_sigma = f_projection_denoising(m_hat, q_hat, sigma_hat, q)

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
        sigma_hats[idx] = sigma_hat
        q_hats[idx] = q_hat

        # print(m, q, sigma, m_hat, q_hat, sigma_hat)

        training_error[idx] = training_error_l1_loss(m, q, sigma, 1.0, alpha, delta_in, delta_out, percentage, beta)
        # print(q, free_energies[idx])
        # print(idx, free_energies[idx])
    except (ConvergenceError, ValueError) as e:
        ms[idx:] = np.nan
        sigmas[idx:] = np.nan
        m_hats[idx:] = np.nan
        sigma_hats[idx:] = np.nan
        q_hats[idx:] = np.nan
        training_error[idx:] = np.nan
        break

while True:
    m = 1 * np.random.random() + 0.01
    q = 1 * np.random.random() + 0.01
    sigma = 1 * np.random.random() + 0.01
    if np.square(m) < q + delta_in * q and np.square(m) < q + delta_out * q:
        break

print("---> ", qs[np.argmin(training_error)])

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

m_hat_true, q_hat_true, sigma_hat_true = f_hat_L2_decorrelated_noise(
    m_true, q_true, sigma_true, alpha, delta_in, delta_out, percentage, beta
)

training_error_true = training_error_l1_loss(m, q, sigma, 1.0, alpha, delta_in, delta_out, percentage, beta)
# print(np.amin(free_energies), free_energy_true)

color = next(plt.gca()._get_lines.prop_cycler)["color"]
# plt.plot(qs, free_energies - (free_energy_true - 1e-2), '.', label="$\\lambda$ = " + "{:.2f}".format(reg_param), color=color)
# plt.axhline(training_error_true, linestyle="--", color=color, alpha=0.5)
# plt.axvline(q_true, linestyle="--", color=color, alpha=0.5)
plt.plot(qs, training_error,  color=color)


plt.title(
    "L1 loss Projection Denoising "
    + r"$\Delta_{in}$ = "
    + "{:.2f}".format(delta_in)
    + r" $\Delta_{out}$ ="
    + "{:.2f}".format(delta_out)
    + r" $\alpha$ = "
    + "{:.2f}".format(alpha)
    + r" $\beta$ = "
    + "{:.2f}".format(beta)
)

plt.ylabel("Training Error")
plt.xlabel("q")
plt.xscale("log")
plt.yscale("log")
plt.legend()
# plt.ylim([0.0, 10])
plt.grid()

plt.show()
