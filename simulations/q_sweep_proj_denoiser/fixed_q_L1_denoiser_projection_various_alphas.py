from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.optimality_finding import find_optimal_reg_param_function
from linear_regression.fixed_point_equations.fpe_projection_denoising import (
    var_func_projection_denoising,
)
from scipy.signal import find_peaks
from linear_regression.aux_functions.stability_functions import stability_l1_l2, stability_huber, stability_ridge
from linear_regression.fixed_point_equations.fpe_L2_loss import var_hat_func_L2_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_L1_loss import var_hat_func_L1_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_L2_regularization import var_func_L2
from linear_regression.aux_functions.training_errors import training_error_l2_loss, training_error_l1_loss
from linear_regression.aux_functions.misc import excess_gen_error
import numpy as np
import matplotlib.pyplot as plt
from linear_regression.utils.errors import ConvergenceError
from linear_regression.aux_functions.misc import damped_update

blend = 1.0
max_iter = 100000
min_iter = 100
abs_tol = 1e-8

delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
reg_param = 0.0
alphas = [2.0, 10.0, 100.0, 200.0, 1000.0]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

plt.figure(figsize=(10, 7.5))

for alpha, color in zip(alphas, colors):
    N = 2000
    qs = np.logspace(-1, 1, N)
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
                m_hat, q_hat, sigma_hat = var_hat_func_L1_decorrelated_noise(
                    m, q, sigma, alpha, delta_in, delta_out, percentage, beta
                )
                new_m, new_q, new_sigma = var_func_projection_denoising(m_hat, q_hat, sigma_hat, q)

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

            training_error[idx] = training_error_l1_loss(m, q, sigma, delta_in, delta_out, percentage, beta)
        except (ConvergenceError, ValueError) as e:
            ms[idx:] = np.nan
            sigmas[idx:] = np.nan
            m_hats[idx:] = np.nan
            sigma_hats[idx:] = np.nan
            q_hats[idx:] = np.nan
            training_error[idx:] = np.nan
            break

    min_idx = np.argmin(training_error)

    m_true, q_true, sigma_true = fixed_point_finder(
        var_func_projection_denoising,
        var_hat_func_L1_decorrelated_noise,
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

    m_hat_true, q_hat_true, sigma_hat_true = var_hat_func_L1_decorrelated_noise(
        m_true, q_true, sigma_true, alpha, delta_in, delta_out, percentage, beta
    )

    training_error_true = training_error_l1_loss(m_true, q_true, sigma_true, delta_in, delta_out, percentage, beta)

    fun_min_val_true, reg_param_opt, (m_true, q_true, sigma_true), _ = find_optimal_reg_param_function(
        var_func_L2,
        var_hat_func_L1_decorrelated_noise,
        {"reg_param": reg_param},
        {
            "alpha": alpha,
            "delta_in": delta_in,
            "delta_out": delta_out,
            "percentage": percentage,
            "beta": beta,
        },
        3.0,
        (ms[min_idx], qs[min_idx], sigmas[min_idx]),
        f_min=excess_gen_error,
        f_min_args={"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta},
    )

    # m_true, q_true, sigma_true = fixed_point_finder(
    #     var_func_L2,
    #     var_hat_func_L1_decorrelated_noise,
    #     (ms[min_idx], qs[min_idx], sigmas[min_idx]),
    #     {"reg_param": reg_param},
    #     {
    #         "alpha": alpha,
    #         "delta_in": delta_in,
    #         "delta_out": delta_out,
    #         "percentage": percentage,
    #         "beta": beta,
    #     },
    # )

    min_idx = np.argmin(training_error + reg_param / (2 * alpha) * qs)

    print(f"q_true = {q_true}, q_min = {qs[min_idx]}")

    plt.subplot(2, 1, 1)
    plt.plot(qs, training_error + reg_param / (2 * alpha) * qs, color=color, label=f"$\\alpha$ = {alpha}")
    plt.plot(qs[min_idx], training_error[min_idx] + reg_param / (2 * alpha) * qs[min_idx], ".", color=color)

    plt.subplot(2, 1, 2)
    plt.axvline(q_true, color=color, linestyle="--", alpha=0.5)
    min_gen_error = np.argmin(excess_gen_error(ms, qs, sigmas, delta_in, delta_out, percentage, beta))
    plt.plot(qs, excess_gen_error(ms, qs, sigmas, delta_in, delta_out, percentage, beta), color=color, linestyle="-")
    plt.plot(qs[min_gen_error], excess_gen_error(ms[min_gen_error], qs[min_gen_error], sigmas[min_gen_error], delta_in, delta_out, percentage, beta), ".", color=color)

plt.subplot(2, 1, 1)
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
    + r" $\lambda$ = "
    + "{:.2f}".format(reg_param)
)

plt.ylabel("Training Error")
plt.xlabel("q")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.ylabel("Estimation Error")
plt.xlabel("q")
plt.xscale("log")
plt.yscale("log")
plt.grid()

plt.show()
