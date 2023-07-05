from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.fpe_projection_denoising import (
    f_projection_denoising,
)
from scipy.signal import find_peaks
from linear_regression.aux_functions.stability_functions import (
    stability_huber,
    stability_ridge,
)
from linear_regression.fixed_point_equations.fpe_Huber_loss import f_hat_Huber_decorrelated_noise
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.training_errors import (
    training_error_l2_loss,
    training_error_huber_loss
)
import numpy as np
import matplotlib.pyplot as plt
from linear_regression.utils.errors import ConvergenceError
from linear_regression.aux_functions.misc import damped_update

blend = 1.0
max_iter = 100000
min_iter = 100
abs_tol = 1e-8

alpha = 30.0
delta_in, delta_out, percentage, beta, a = 1.0, 5.0, 0.3, 0.0, 0.02

N = 5000
qs = np.logspace(-1, 2, N)
training_error = np.empty_like(qs)
ms = np.empty_like(qs)
sigmas = np.empty_like(qs)
m_hats = np.empty_like(qs)
q_hats = np.empty_like(qs)
sigma_hats = np.empty_like(qs)

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
            m_hat, q_hat, sigma_hat = f_hat_Huber_decorrelated_noise(
                m, q, sigma, alpha, delta_in, delta_out, percentage, beta, a
            )
            new_m, new_q, new_sigma = f_projection_denoising(m_hat, q_hat, sigma_hat, q)

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

        training_error[idx] = training_error_huber_loss(
            m, q, sigma, delta_in, delta_out, percentage, beta, a
        )
    except (ConvergenceError, ValueError) as e:
        ms[idx:] = np.nan
        sigmas[idx:] = np.nan
        m_hats[idx:] = np.nan
        sigma_hats[idx:] = np.nan
        q_hats[idx:] = np.nan
        training_error[idx:] = np.nan
        break

plt.figure(figsize=(10, 7.5))

# plt.subplot(211)

reg_parmas = [-0.15, -0.14391463821273756, -0.142, -0.13895298432810935, -0.1]

for reg_param in reg_parmas:
    color = next(plt.gca()._get_lines.prop_cycler)["color"]

    peaks = find_peaks(-(training_error + reg_param / (2 * alpha) * qs))

    for p in peaks[0]:
        plt.axvline(qs[p], linestyle="--", color=color, alpha=0.75)
        plt.scatter(qs[p], training_error[p]+ reg_param / (2 * alpha) * qs[p], color=color, alpha=0.75)

    plt.plot(
        qs,
        training_error + reg_param / (2 * alpha) * qs,
        color=color,
        label="$\\lambda$ = {:.3f}".format(reg_param),
    )

# plt.plot(
#     qs,
#     stability_huber(ms, qs, sigmas, alpha, 1.0, delta_in, delta_out, percentage, beta, a),
#     label="stability",
#     color="black",
#     linestyle="--",
# )
plt.axhline(0, color="black", alpha=0.75)

plt.title(
    "Huber loss Projection Denoising "
    + r"$\Delta_{in}$ = "
    + "{:.2f}".format(delta_in)
    + r" $\Delta_{out}$ ="
    + "{:.2f}".format(delta_out)
    + r" $\epsilon$ = "
    + "{:.2f}".format(percentage)
    + r" $\alpha$ = "
    + "{:.2f}".format(alpha)
    + r" $\beta$ = "
    + "{:.2f}".format(beta)
    + r" $a$ = "
    + "{:.2f}".format(a)
)

plt.ylabel("Training Error")
plt.xlabel("q")
plt.xscale("log")
plt.legend()
plt.ylim([0.01, 0.03])
plt.grid()

plt.show()
