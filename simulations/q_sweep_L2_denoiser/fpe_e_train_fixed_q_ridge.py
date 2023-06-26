from linear_regression.aux_functions.training_errors import training_error_l2_loss
from linear_regression.fixed_point_equations.fpe_L2_loss import var_hat_func_L2_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_L2_regularization import var_func_L2
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
import numpy as np
import matplotlib.pyplot as plt
from linear_regression.utils.errors import ConvergenceError
from linear_regression.aux_functions.misc import damped_update


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


blend = 0.85
max_iter = 100000
min_iter = 10
abs_tol = 1e-7

delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
# minimum lambda is -0.171 fora alpha = 2.0
reg_params = [-0.17, -0.1, 0.0, 0.1]
alpha = 2.0

N = 1000
qs = np.linspace(0.01, 2, N)
# qs = np.logspace(-2, 0, N)
training_error = np.empty_like(qs)
ms = np.empty_like(qs)
sigmas = np.empty_like(qs)
m_hats = np.empty_like(qs)
q_hats = np.empty_like(qs)
sigma_hats = np.empty_like(qs)

delta_eff = (1 - percentage) * delta_in + percentage * delta_out
intermediate_val = 1 + percentage * (beta - 1)

plt.figure(figsize=(10, 7.5))

for reg_param in reg_params:
    print("reg_param {:.2f}".format(reg_param))
    q = qs[0]
    while True:
        m = 100 * np.random.random() + 1e-8
        sigma = 100 * np.random.random() + 1e-8
        if (
            np.square(m) < q + delta_in * q
            and np.square(m) * beta**2 < q * beta**2 + delta_out * q
        ):
            break

    for idx, q in enumerate(qs):

        iter_nb = 0
        err = 100.0
        while err > abs_tol or iter_nb < min_iter:
            m_hat = alpha * intermediate_val / (1 + sigma)
            sigma_hat = alpha / (1 + sigma)

            new_m = m_hat / (reg_param + sigma_hat)
            new_sigma = 1 / (reg_param + sigma_hat)

            err = max([abs(new_m - m), abs(new_sigma - sigma)])

            m = damped_update(new_m, m, blend)
            sigma = damped_update(new_sigma, sigma, blend)

            iter_nb += 1
            if iter_nb > max_iter:
                raise ConvergenceError("fixed_point_finder", iter_nb)

        q_hat = q * (reg_param + sigma_hat) ** 2 - m_hat**2

        ms[idx] = m
        sigmas[idx] = sigma
        m_hats[idx] = m_hat
        sigma_hats[idx] = sigma_hat
        q_hats[idx] = q_hat

        training_error[idx] = training_error_l2_loss(m, q, sigma, reg_param, alpha, delta_in, delta_out, percentage, beta)

    m_true, q_true, sigma_true = fixed_point_finder(
        var_func_L2,
        var_hat_func_L2_decorrelated_noise,
        (m,q,sigma),
        {"reg_param":reg_param},
        {"alpha" : alpha, "delta_in":delta_in, "delta_out":delta_out, "percentage":percentage, "beta":beta},
    )
    m_hat_true, q_hat_true, sigma_hat_true = var_hat_func_L2_decorrelated_noise(m_true, q_true, sigma_true, alpha, delta_in, delta_out, percentage, beta)

    _, closest_true_idx = find_nearest(qs, q_true)
    print("true values ", m_true, q_true, sigma_true, m_hat_true, q_hat_true, sigma_hat_true)
    print("difference ",  ms[closest_true_idx], m_true, abs(ms[closest_true_idx] - m_true))

    training_error_true = training_error_l2_loss(m, q, sigma, reg_param, alpha, delta_in, delta_out, percentage, beta)

    min_idx = np.argmin(training_error)
    print("-- lambda {:.2f} --> ".format(reg_param), q_true, sigma_hat + reg_param, training_error_true)
    print(abs(training_error[min_idx] - training_error_true), abs(qs[min_idx] - q_true))

    color = next(plt.gca()._get_lines.prop_cycler)["color"]

    # plt.plot(qs, free_energies - np.amin(free_energies), '-', label="$\\lambda$ = " + "{:.2f}".format(reg_param), color=color)
    plt.plot(
        qs, training_error + reg_param / (2 * alpha) * qs, label="$\\lambda$ = " + "{:.2f}".format(reg_param), color=color
    )
    # plt.axhline(training_error_true, linestyle="--", color=color)
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
# plt.plot(qs, sigma_hats, label="sigma_hat")

plt.ylabel("Training Error")
plt.xlabel("q")
# plt.xscale("log")
# plt.yscale("log")
plt.legend()
plt.grid()

plt.show()
