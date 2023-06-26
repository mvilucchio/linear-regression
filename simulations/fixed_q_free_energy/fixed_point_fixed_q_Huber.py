from linear_regression.aux_functions.free_energy import (
    free_energy,
    Psi_w_L2_reg,
    Psi_out_Huber
)
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.fpe_L2_regularization import var_func_L2
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    var_hat_func_Huber_decorrelated_noise,
)
import numpy as np
import matplotlib.pyplot as plt
from linear_regression.utils.errors import ConvergenceError
from linear_regression.aux_functions.misc import damped_update

blend = 1.0
max_iter = 100000
min_iter = 1000
abs_tol = 1e-10

delta_in, delta_out, percentage, beta, a = 1.0, 5.0, 0.3, 0.0, 0.5
reg_params = [-0.05, -0.01, 0.01, 0.05, 0.1]
reg_params = [0.0, 0.05]
alpha = 2.0

N_the = 1000
N_num = 10
# qs = np.linspace(0.01, 10.0, 1000)
qs_theoretical = np.logspace(-2, 2, N_the)
free_energies = np.empty_like(qs_theoretical)
ms = np.empty_like(qs_theoretical)
sigmas = np.empty_like(qs_theoretical)
m_hats = np.empty_like(qs_theoretical)
q_hats = np.empty_like(qs_theoretical)
sigma_hats = np.empty_like(qs_theoretical)

plt.figure(figsize=(10, 7.5))

for reg_param in reg_params[::-1]:
    print("reg_param {:.3f}".format(reg_param))
    q = qs_theoretical[0]
    while True:
        m = 10 * np.random.random() + 0.01
        sigma = 10 * np.random.random() + 0.01
        if (
            np.square(m) < q + delta_in * q
            and np.square(m) < q + delta_out * q
        ):
            break

    for idx, q in enumerate(qs_theoretical):
        try:
            iter_nb = 0
            err = 100.0
            while err > abs_tol or iter_nb < min_iter:
                m_hat, _, sigma_hat = var_hat_func_Huber_decorrelated_noise(
                    m, q, sigma, alpha, delta_in, delta_out, percentage, beta, a
                )
                new_m, _, new_sigma = var_func_L2(m_hat, 0.0, sigma_hat, reg_param)

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

            # print(m, q, sigma, m_hat, q_hat, sigma_hat)

            free_energies[idx] = free_energy(
                Psi_w_L2_reg,
                Psi_out_Huber,
                alpha,
                m,
                q,
                sigma,
                m_hat,
                q_hat,
                sigma_hat,
                (reg_param,),
                (delta_in, delta_out, percentage, beta, a),
            )
            # print(idx, free_energies[idx])
        except (ConvergenceError, ValueError) as e:
            print(e)
            ms[idx] = np.nan
            sigmas[idx] = np.nan
            m_hats[idx] = np.nan
            sigma_hats[idx] = np.nan
            q_hats[idx] = np.nan
            free_energies[idx] = np.nan
            break

    while True:
        m = 10 * np.random.random() + 0.01
        q = 10 * np.random.random() + 0.01
        sigma = 10 * np.random.random() + 0.01
        if (
            np.square(m) < q + delta_in * q
            and np.square(m) < q + delta_out * q
        ):
            break

    # train_error_mean, train_error_std = run_erm_weight_finding(alpha, measure_fun, find_coefficients_fun, n_features, repetitions, measure_fun_args, find_coefficients_fun_args)

    m_true, q_true, sigma_true = fixed_point_finder(
        var_func_L2,
        var_hat_func_Huber_decorrelated_noise,
        (m, q, sigma),
        {"reg_param": reg_param},
        {
            "alpha": alpha,
            "delta_in": delta_in,
            "delta_out": delta_out,
            "percentage": percentage,
            "beta": beta,
            "a": a,
        },
    )
    m_hat_true, q_hat_true, sigma_hat_true = var_hat_func_Huber_decorrelated_noise(
        m_true, q_true, sigma_true, alpha, delta_in, delta_out, percentage, beta, a
    )

    free_energy_true = free_energy(
        Psi_w_L2_reg,
        Psi_out_Huber,
        alpha,
        m_true,
        q_true,
        sigma_true,
        m_hat_true,
        q_hat_true,
        sigma_hat_true,
        (reg_param,),
        (delta_in, delta_out, percentage, beta, a),
    )
    min_idx = np.argmin(free_energies)
    print(abs(free_energies[min_idx] - free_energy_true), abs(qs_theoretical[min_idx] - q_true))

    color = next(plt.gca()._get_lines.prop_cycler)["color"]
    # print(np.amin(free_energies), free_energy_true)
    # plt.plot(qs, free_energies - (free_energy_true - 1e-2), '.', label="$\\lambda$ = " + "{:.2f}".format(reg_param), color=color)
    plt.axhline(free_energy_true, linestyle="--", color=color)
    plt.axvline(q_true, linestyle="--", color=color)
    plt.plot(qs_theoretical, free_energies, label="$\\lambda$ = " + "{:.2f}".format(reg_param), color=color)


plt.title(
    "Huber a = {:.1f}".format(a)
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
plt.legend()
plt.ylim([-1, 5])
plt.grid()

plt.show()
