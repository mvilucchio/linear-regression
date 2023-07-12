from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.fpe_projection_denoising import (
    f_projection_denoising,
)
from scipy.signal import find_peaks
from linear_regression.aux_functions.stability_functions import stability_L1_decorrelated_regress, stability_Huber_decorrelated_regress, stability_L2_decorrelated_regress
from linear_regression.fixed_point_equations.fpe_L2_loss import f_hat_L2_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_L1_loss import f_hat_L1_decorrelated_noise
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.training_errors import training_error_l2_loss, training_error_l1_loss 
import numpy as np
import matplotlib.pyplot as plt
from linear_regression.utils.errors import ConvergenceError
from linear_regression.aux_functions.misc import damped_update

blend = 1.0
max_iter = 100000
min_iter = 100
abs_tol = 1e-8

delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
alpha = 2.0

N = 5000
qs = np.logspace(-1, 3, N)
training_error = np.empty_like(qs)
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
            m_hat, q_hat, Σ_hat = f_hat_L1_decorrelated_noise(
                m, q, sigma, alpha, delta_in, delta_out, percentage, beta
            )
            new_m, new_q, new_sigma = f_projection_denoising(m_hat, q_hat, Σ_hat, q)

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

        training_error[idx] = training_error_l1_loss(
            m, q, sigma, delta_in, delta_out, percentage, beta
        )
    except (ConvergenceError, ValueError) as e:
        ms[idx:] = np.nan
        sigmas[idx:] = np.nan
        m_hats[idx:] = np.nan
        Σ_hats[idx:] = np.nan
        q_hats[idx:] = np.nan
        training_error[idx:] = np.nan
        break

min_idx = np.argmin(training_error)

m_true, q_true, sigma_true = fixed_point_finder(
    f_projection_denoising,
    f_hat_L1_decorrelated_noise,
    (ms[min_idx], qs[min_idx], sigmas[min_idx]),
    {"q_fixed": qs[min_idx]},
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

training_error_true = training_error_l1_loss(m_true, q_true, sigma_true, delta_in, delta_out, percentage, beta)

print("q_true = ", q_true)

color = next(plt.gca()._get_lines.prop_cycler)["color"]

# peaks = find_peaks(-training_error)

# for p in peaks[0]:
#     plt.axvline(qs[p], linestyle="--", color=color, alpha=0.5)

stab_cond = stability_L1_decorrelated_regress(ms, qs, sigmas, alpha, 1.0, delta_in, delta_out, percentage, beta)
unstable_values = stab_cond <= 0.0

plt.plot(qs[unstable_values], training_error[unstable_values], color=color, linestyle="--")
plt.plot(qs[~unstable_values], training_error[~unstable_values], color=color, linestyle="-", label="Training Error")

color = next(plt.gca()._get_lines.prop_cycler)["color"]
plt.plot(qs, 1 + qs - 2.0 * ms, label="Generalization Error", color=color, linestyle="-")

plt.title(
    "L1 loss Projection Denoising "
    + r"$\Delta_{in}$ = "
    + "{:.2f}".format(delta_in)
    + r" $\Delta_{out}$ ="
    + "{:.2f}".format(delta_out)
    + r"$\epsilon$ = "
    + "{:.2f}".format(percentage)
    + r" $\alpha$ = "
    + "{:.2f}".format(alpha)
    + r" $\beta$ = "
    + "{:.2f}".format(beta)
)

plt.ylabel("Training Error")
plt.xlabel("q")
plt.xscale("log")
plt.yscale("log")
# plt.legend()
# plt.ylim([-0.5, 2])
plt.grid()

plt.show()
