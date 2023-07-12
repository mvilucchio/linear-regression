from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.fpe_projection_denoising import (
    f_projection_denoising,
)
from linear_regression.regression_numerics.numerics import (
    erm_weight_finding,
    train_error_data,
)
from scipy.signal import find_peaks
from linear_regression.aux_functions.loss_functions import huber_loss
from linear_regression.aux_functions.stability_functions import stability_L1_decorrelated_regress, stability_Huber_decorrelated_regress, stability_L2_decorrelated_regress
from linear_regression.regression_numerics.data_generation import measure_gen_decorrelated
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    f_hat_Huber_decorrelated_noise,
)
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.training_errors import (
    training_error_huber_loss,
    training_error_huber_loss,
)
from linear_regression.regression_numerics.erm_solvers import find_coefficients_Huber_on_sphere
import numpy as np
import matplotlib.pyplot as plt
from linear_regression.utils.errors import ConvergenceError
from linear_regression.aux_functions.misc import damped_update

blend = 1.0
max_iter = 100000
min_iter = 100
abs_tol = 1e-8

delta_in, delta_out, percentage, beta, a = 1.0, 5.0, 0.3, 0.0, 1.0
alpha = 30.0
reg_param = -1.5

N = 3000
qs = np.logspace(-2, 3, N)
training_error = np.empty_like(qs)
generalization_error = np.empty_like(qs)
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
            m_hat, q_hat, Σ_hat = f_hat_Huber_decorrelated_noise(
                m, q, sigma, alpha, delta_in, delta_out, percentage, beta, a
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

        training_error[idx] = training_error_huber_loss(
            m, q, sigma, delta_in, delta_out, percentage, beta, a
        ) + reg_param / (2 * alpha) * q
        generalization_error[idx] = 1 - 2 * m + q

    except (ConvergenceError, ValueError) as e:
        ms[idx:] = np.nan
        sigmas[idx:] = np.nan
        m_hats[idx:] = np.nan
        Σ_hats[idx:] = np.nan
        q_hats[idx:] = np.nan
        training_error[idx:] = np.nan
        break

min_idx = np.argmin(training_error)

# m_true, q_true, sigma_true = fixed_point_finder(
#     f_projection_denoising,
#     f_hat_Huber_decorrelated_noise,
#     (ms[min_idx], qs[min_idx], sigmas[min_idx]),
#     {"q_fixed": qs[min_idx]},
#     {
#         "alpha": alpha,
#         "delta_in": delta_in,
#         "delta_out": delta_out,
#         "percentage": percentage,
#         "beta": beta,
#         "a": a,
#     },
# )

# m_hat_true, q_hat_true, Σ_hat_true = f_hat_Huber_decorrelated_noise(
#     m_true, q_true, sigma_true, alpha, delta_in, delta_out, percentage, beta, a
# )

# training_error_true = training_error_huber_loss(
#     m_true, q_true, sigma_true, delta_in, delta_out, percentage, beta, a
# )

# print("q_true = ", q_true, qs[min_idx])

color = next(plt.gca()._get_lines.prop_cycler)["color"]

# plt.axhline(training_error_true, linestyle="--", color=color, alpha=0.5)
# plt.axvline(q_true, linestyle="--", color=color, alpha=0.5)

# reg_param = -1.5

# peaks = find_peaks(-(training_error + reg_param / (2 * alpha) * qs))

peaks = find_peaks(-training_error)
print(peaks)
for p in peaks[0]:
    print("qs min ", qs[p])
    plt.axvline(qs[p], linestyle="--", color=color, alpha=0.5)

# plt.plot(qs, training_error + reg_param / (2 * alpha) * qs, color=color, label="training error lambda = {:.2f}".format(reg_param))
plt.plot(qs, training_error, color=color, label="training error lambda = {:.2f}".format(reg_param))

plt.plot(qs, stability_Huber_decorrelated_regress(ms, qs, sigmas, alpha, reg_param, delta_in, delta_out, percentage, beta, a), label="stability")

np.savetxt(
    "./simulations/data/Huber_decorrelated_noise_alpha={:.2f}_delta_in={:.2f}_delta_out={:.2f}_percentage={:.2f}_beta={:.2f}_a={:.2f}_reg_param={:.2f}.csv".format(
        alpha, delta_in, delta_out, percentage, beta, a, reg_param
    ),
    np.vstack([qs, training_error, generalization_error, ms, sigmas, m_hats, q_hats, Σ_hats]).T,
    delimiter=",",
    header="q,training_error,generalization_error,m,sigma,m_hat,q_hat,Σ_hat",
)

# plt.plot(qs, training_error, color=color)
# plt.plot(qs, sigmas, label="sigma")

# qss = np.logspace(3, 7, 100)
# plt.plot(qss, 0.01 * np.sqrt(qss), label="0.01 * sqrt(q)")

# reg_param = 0.0

# m_l2_reg, q_l2_reg, sigma_l2_reg = fixed_point_finder(
#     f_L2_reg,
#     f_hat_Huber_decorrelated_noise,
#     (ms[min_idx], qs[min_idx], sigmas[min_idx]),
#     {"reg_param": reg_param},
#     {
#         "alpha": alpha,
#         "delta_in": delta_in,
#         "delta_out": delta_out,
#         "percentage": percentage,
#         "beta": beta,
#         "a": a,
#     },
# )

# print(q_l2_reg, q_true, abs(q_l2_reg - q_true))




# n_features = 500
# repetitions = 5

# qs_numerical = np.linspace(2, 20, 10)
# E_train_mean = np.empty_like(qs_numerical)
# E_train_std = np.empty_like(qs_numerical)


# for q_fixed in qs_numerical:
#     E_train_mean[idx], E_train_std[idx] = erm_weight_finding(
#         alpha,
#         measure_gen_decorrelated,
#         find_coefficients_Huber_on_sphere,
#         [train_error_data],
#         [(huber_loss, (a,))],
#         n_features,
#         repetitions,
#         (delta_in, delta_out, percentage, beta),
#         (reg_param, q_fixed, a),
#     )

# plt.errorbar(
#     qs_numerical, E_train_mean, yerr=E_train_std, label="E_train", linestyle="", marker="."
# )

plt.title(
    "Huber loss Projection Denoising "
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
    + r"$a$ = "
    + "{:.2f}".format(a)
)

plt.ylabel("Training Error")
plt.xlabel("q")
plt.xscale("log")
# plt.yscale("log")
plt.legend()
plt.ylim([-1.0, 6])
plt.grid()

plt.show()
