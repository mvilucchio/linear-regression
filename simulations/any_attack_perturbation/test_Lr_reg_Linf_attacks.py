import linear_regression.sweeps.alpha_sweeps as alsw
import matplotlib.pyplot as plt
import numpy as np
from linear_regression.fixed_point_equations.classification.Hinge_loss import (
    f_hat_Hinge_no_noise_classif,
)
from linear_regression.aux_functions.training_errors import (
    training_error_Hinge_loss_no_noise,
)
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.misc import angle_teacher_student
from linear_regression.fixed_point_equations.classification.BO import (
    f_BO,
    f_hat_BO_no_noise_classif,
)
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder_adversiaral
from linear_regression.fixed_point_equations.regularisation.Linf_attacks_Lr_reg import (
    f_Lr_reg_Linf_attack,
)
from linear_regression.fixed_point_equations.classification.Adv_train_p_norm import (
    f_hat_Logistic_no_noise_Linf_adv_classif,
)
import pickle
from linear_regression.erm.erm_solvers import find_coefficients_Logistic_adv_Linf
from scipy.integrate import quad
from math import erfc, erf


def adversarial_error_test(m, q, P, eps_g):
    Iminus = quad(
        lambda x: 0.5
        / np.sqrt(2 * np.pi * q)
        * np.exp(-0.5 * x**2 / q)
        * erfc(m * x / np.sqrt(2 * q * (q - m**2))),
        -eps_g * P,
        20,
    )[0]
    # erf( m * x / np.sqrt(2 * q * (q - m ** 2)) )
    Iplus = quad(
        lambda x: 0.5
        / np.sqrt(2 * np.pi * q)
        * np.exp(-0.5 * x**2 / q)
        * (1 + erf(m * x / np.sqrt(2 * q * (q - m**2)))),
        -20,
        eps_g * P,
    )[0]
    return Iminus + Iplus


alpha_min, alpha_max, n_alpha_pts = 0.2, 0.3, 5
reg_orders = [1, 2]  # [2, 3, 4, 5]
eps_t = 0.1
eps_g = 0.1
reg_param = 1e-3

alpha_min_num, alpha_max_num, n_alpha_pts_num = 0.02, 0.3, 3

run_experiments = True

d = 1_500
n_gen = 1000
reps = 5

if __name__ == "__main__":
    if run_experiments:
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)
        alphas_num = np.logspace(
            np.log10(alpha_min_num), np.log10(alpha_max_num), n_alpha_pts_num
        )

        ms_found = np.empty((len(reg_orders), n_alpha_pts))
        qs_found = np.empty((len(reg_orders), n_alpha_pts))
        sigmas_found = np.empty((len(reg_orders), n_alpha_pts))
        Ps_found = np.empty((len(reg_orders), n_alpha_pts))
        estim_errors_found = np.empty((len(reg_orders), n_alpha_pts))
        adversarial_errors_found = np.empty((len(reg_orders), n_alpha_pts))
        generalisation_errors_found = np.empty((len(reg_orders), n_alpha_pts))

        estim_errors_mean = np.empty((len(reg_orders), n_alpha_pts_num))
        estim_errors_std = np.empty((len(reg_orders), n_alpha_pts_num))
        generalisation_errors_mean = np.empty((len(reg_orders), n_alpha_pts_num))
        generalisation_errors_std = np.empty((len(reg_orders), n_alpha_pts_num))
        adversarial_error_mean = np.empty((len(reg_orders), n_alpha_pts_num))
        adversarial_error_std = np.empty((len(reg_orders), n_alpha_pts_num))

        initial_condition = (1.539e+00, 1.356e+01, 4.500e+03, 0.933e+00)

        for i, reg_order in enumerate(reg_orders):
            for jprime, alpha in enumerate(alphas[::-1]):
                j = n_alpha_pts - jprime - 1
                print(f"SE reg_order = {reg_order}, alpha = {alpha}")

                f_kwargs = {"r": reg_order, "reg_param": reg_param}
                f_hat_kwargs = {"alpha": alpha, "eps_t": eps_t}

                ms_found[i, j], qs_found[i, j], sigmas_found[i, j], Ps_found[i, j] = (
                    fixed_point_finder_adversiaral(
                        f_Lr_reg_Linf_attack,
                        f_hat_Logistic_no_noise_Linf_adv_classif,
                        initial_condition,
                        f_kwargs,
                        f_hat_kwargs,
                        abs_tol=1e-4,
                        min_iter=10,
                    )
                )

                estim_errors_found[i, j] = 1 - 2 * ms_found[i, j] + qs_found[i, j]

                initial_condition = (
                    ms_found[i, j],
                    qs_found[i, j],
                    sigmas_found[i, j],
                    Ps_found[i, j],
                )

                adversarial_errors_found[i, j] = adversarial_error_test(
                    ms_found[i, j], qs_found[i, j], Ps_found[i, j], eps_g
                )
                generalisation_errors_found[i, j] = (
                    np.arccos(ms_found[i, j] / np.sqrt(qs_found[i, j])) / np.pi
                )

            # data_dict = {
            #     "alphas": alphas,
            #     "ms_found": ms_found[i],
            #     "qs_found": qs_found[i],
            #     "sigmas_found": sigmas_found[i],
            #     "Ps_found": Ps_found[i],
            #     "estim_errors_found": estim_errors_found[i],
            #     "adversarial_errors_found": adversarial_errors_found[i],
            #     "generalisation_errors_found": generalisation_errors_found[i],
            #     # "alphas_num": alphas_num,
            #     # "erm_estim_errors_mean": estim_errors_mean[i],
            #     # "erm_estim_errors_std": estim_errors_std[i],
            #     # "erm_gen_errors_mean": generalisation_errors_mean[i],
            #     # "erm_gen_errors_std": generalisation_errors_std[i],
            #     # "erm_adv_errors_mean": adversarial_error_mean[i],
            #     # "erm_adv_errors_std": adversarial_error_std[i],
            # }

            for j, alpha in enumerate(alphas_num):
                print(f"ERM reg_order = {reg_order}, alpha = {alpha}")
                n = int(alpha * d)

                tmp_estim_errors = []
                tmp_generalisation_errors = []
                tmp_adversarial_errors = []
                for _ in range(reps):
                    xs = np.random.normal(loc=0.0, scale=1.0, size=(n, d))
                    wstar = np.random.normal(loc=0.0, scale=1.0, size=(d,))
                    # wstar /= np.sqrt(np.sum(wstar ** 2))
                    ys = np.sign(xs @ wstar)

                    w = find_coefficients_Logistic_adv_Linf(
                        ys, xs, reg_param, eps_t, reg_order
                    )

                    tmp_estim_errors.append(np.mean((w - wstar) ** 2))

                    xs_gen = np.random.normal(loc=0.0, scale=1.0, size=(n_gen, d))
                    ys_gen = np.sign(xs_gen @ wstar)

                    tmp_generalisation_errors.append(
                        np.mean(np.where(ys_gen != np.sign(xs_gen @ w), 1, 0))
                    )
                    tmp_adversarial_errors.append(
                        np.mean(
                            np.where(
                                ys_gen * (xs_gen @ w / np.sqrt(d))
                                - eps_g * np.sum(np.abs(w)) / d
                                < 0,
                                1,
                                0,
                            )
                        )
                    )

                estim_errors_mean[i, j] = np.mean(tmp_estim_errors)
                estim_errors_std[i, j] = np.std(tmp_estim_errors)

                generalisation_errors_mean[i, j] = np.mean(tmp_generalisation_errors)
                generalisation_errors_std[i, j] = np.std(tmp_generalisation_errors)

                adversarial_error_mean[i, j] = np.mean(tmp_adversarial_errors)
                adversarial_error_std[i, j] = np.std(tmp_adversarial_errors)

            data_dict = {
                "alphas": alphas,
                "ms_found": ms_found[i],
                "qs_found": qs_found[i],
                "sigmas_found": sigmas_found[i],
                "Ps_found": Ps_found[i],
                "estim_errors_found": estim_errors_found[i],
                "adversarial_errors_found": adversarial_errors_found[i],
                "generalisation_errors_found": generalisation_errors_found[i],
                "alphas_num": alphas_num,
                "erm_estim_errors_mean": estim_errors_mean[i],
                "erm_estim_errors_std": estim_errors_std[i],
                "erm_gen_errors_mean": generalisation_errors_mean[i],
                "erm_gen_errors_std": generalisation_errors_std[i],
                "erm_adv_errors_mean": adversarial_error_mean[i],
                "erm_adv_errors_std": adversarial_error_std[i],
            }

            with open(f"test_lasso_data_Linf_reg_order_{reg_order:d}.pkl", "wb") as f:
                pickle.dump(data_dict, f)

    plt.figure(figsize=(10, 5))
    for i, reg_order in enumerate(reg_orders):
        with open(f"test_lasso_data_Linf_reg_order_{reg_order:d}.pkl", "rb") as f:
            data_dict = pickle.load(f)

        alphas = data_dict["alphas"]
        estim_errors_found = data_dict["estim_errors_found"]
        adversarial_error_test_found = data_dict["adversarial_errors_found"]
        generalisation_errors_found = data_dict["generalisation_errors_found"]

        alphas_num = data_dict["alphas_num"]
        estim_errors_mean = data_dict["erm_estim_errors_mean"]
        estim_errors_std = data_dict["erm_estim_errors_std"]

        generalisation_errors_mean = data_dict["erm_gen_errors_mean"]
        generalisation_errors_std = data_dict["erm_gen_errors_std"]

        adversarial_error_test_mean = data_dict["erm_adv_errors_mean"]
        adversarial_error_test_std = data_dict["erm_adv_errors_std"]

        plt.subplot(1, 2, 1)
        plt.plot(
            alphas,
            generalisation_errors_found,
            "-.",
            color=f"C{i}",
            label=f"r = {reg_order} Gen",
        )

        plt.errorbar(
            alphas_num,
            generalisation_errors_mean,
            yerr=generalisation_errors_std,
            fmt="x",
            color=f"C{i}",
        )

        plt.subplot(1, 2, 2)
        plt.plot(
            alphas,
            adversarial_error_test_found,
            "--",
            color=f"C{i}",
            label=f"r = {reg_order} Adv",
        )

        plt.errorbar(
            alphas_num,
            adversarial_error_test_mean,
            yerr=adversarial_error_test_std,
            fmt="x",
            color=f"C{i}",
        )

    # plt.subplot(1, 3, 1)
    # plt.title(r"L$\infty$ attack with regularisation $L r$")
    # plt.xlabel(r"$\alpha$")
    # # plt.ylabel(r"$E_{\mathrm{estim}}$")
    # plt.xscale("log")
    # # plt.yscale("log")
    # plt.legend()
    # plt.grid()

    plt.subplot(1, 2, 1)
    # plt.title(r"L$\infty$ attack with regularisation $L r$")
    plt.xlabel(r"$\alpha$")
    # plt.ylabel(r"$E_{\mathrm{gen}}$")
    plt.xscale("log")
    # plt.yscale("log")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    # plt.title(r"L$\infty$ attack with regularisation $L r$")
    plt.xlabel(r"$\alpha$")
    # plt.ylabel(r"$E_{\mathrm{adv}}$")
    plt.xscale("log")
    # plt.yscale("log")
    plt.legend()
    plt.grid()

    

    plt.show()
