import linear_regression.sweeps.alpha_sweeps as alsw
import matplotlib.pyplot as plt
import numpy as np
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder_adversiaral
from linear_regression.aux_functions.misc import classification_adversarial_error
from linear_regression.fixed_point_equations.regularisation.Linf_attacks_Lr_reg import (
    f_Lr_regularisation_Lpstar_attack,
    f_Lr_regularisation_attack_1,
)
from linear_regression.fixed_point_equations.classification.Adv_train_p_norm import (
    f_hat_Logistic_no_noise_Linf_adv_classif,
)
import pickle


alpha_min, alpha_max, n_alpha_pts = 0.1, 5, 55
reg_orders = [
    1,
    # 2,
    # 3,
]
eps_t = 0.1
eps_g = 0.1
reg_param = 1e-2
pstar = 1

run_experiments = True

file_name = f"SE_data_Linf_reg_order_{{}}_alpha_{alpha_min:.3f}_{alpha_max:.3f}_reg_param_{reg_param:.1e}_eps_t_g_{eps_t:.1e}_{eps_g:.1e}.pkl"

alpha_min_erm, alpha_max_erm, n_alpha_pts_erm = 0.1, 100, 22
d = 1000
reps = 10
file_name_erm = f"ERM_data_Linf_reg_order_{{:d}}_alpha_{alpha_min_erm:.3f}_{alpha_max_erm:.3f}_{n_alpha_pts_erm:d}_dim_{d:d}_reps_{reps:d}_reg_param_{reg_param:.1e}_eps_t_g_{eps_t:.1e}_{eps_g:.1e}.pkl"

if __name__ == "__main__":
    if run_experiments:
        alphas_se = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)

        ms_found = np.empty((len(reg_orders), n_alpha_pts))
        qs_found = np.empty((len(reg_orders), n_alpha_pts))
        sigmas_found = np.empty((len(reg_orders), n_alpha_pts))
        Ps_found = np.empty((len(reg_orders), n_alpha_pts))
        estim_errors_se = np.empty((len(reg_orders), n_alpha_pts))
        adversarial_errors_found = np.empty((len(reg_orders), n_alpha_pts))
        gen_errors_se = np.empty((len(reg_orders), n_alpha_pts))

        for i, reg_order in enumerate(reg_orders):
            # those are initial conditions for alpha = 0.1
            if reg_order == 1:
                initial_condition = (
                    0.6604238471542454,
                    13.959656849095602,
                    25767.132081452226,
                    1.3648584334765084,
                )
            elif reg_order == 2:
                m, q, sigma, P = (6.766e-01, 5.780e00, 4.442e03, 1.780e00)
                initial_condition = (m, q, sigma, P)
            elif reg_order == 3:
                m, q, sigma, P = (3.873e-01, 2.571e00, 1.710e03, 1.446e00)
                initial_condition = (m, q, sigma, P)
            elif reg_order == 4:
                m, q, sigma, P = (3.417e-01, 2.140e00, 8.220e02, 1.371e00)
                initial_condition = (m, q, sigma, P)
            else:
                m, q, sigma, P = (
                    0.6604238471542454,
                    13.959656849095602,
                    25767.132081452226,
                    1.3648584334765084,
                )
                initial_condition = (m, q, sigma, P)

            for jprime, alpha in enumerate(alphas_se):
                # j = n_alpha_pts - jprime - 1
                j = jprime
                print("\033[91m" + f"SE {reg_order = }, {alpha = :.3e}" + "\033[0m")

                f_kwargs = {"reg_param": reg_param, "reg_order": reg_order, "pstar": 1}
                f_hat_kwargs = {"alpha": alpha, "eps_t": eps_t}
                ms_found[i, j], qs_found[i, j], sigmas_found[i, j], Ps_found[i, j] = (
                    fixed_point_finder_adversiaral(
                        f_Lr_regularisation_Lpstar_attack,
                        f_hat_Logistic_no_noise_Linf_adv_classif,
                        initial_condition,
                        f_kwargs,
                        f_hat_kwargs,
                        abs_tol=1e-7,
                        min_iter=10,
                    )
                )

                # print("found fixed point")
                estim_errors_se[i, j] = 1 - 2 * ms_found[i, j] + qs_found[i, j]

                initial_condition = (
                    ms_found[i, j],
                    qs_found[i, j],
                    sigmas_found[i, j],
                    Ps_found[i, j],
                )

                adversarial_errors_found[i, j] = classification_adversarial_error(
                    ms_found[i, j], qs_found[i, j], Ps_found[i, j], eps_g, pstar
                )
                gen_errors_se[i, j] = np.arccos(ms_found[i, j] / np.sqrt(qs_found[i, j])) / np.pi

            data = {
                "alphas": alphas_se,
                "ms_found": ms_found[i],
                "qs_found": qs_found[i],
                "sigmas_found": sigmas_found[i],
                "Ps_found": Ps_found[i],
                "estim_errors_found": estim_errors_se[i],
                "adversarial_errors_found": adversarial_errors_found[i],
                "generalisation_errors_found": gen_errors_se[i],
            }

            with open(file_name.format(reg_order), "wb") as f:
                pickle.dump(data, f)

    plt.figure(figsize=(10, 5))
    for i, reg_order in enumerate(reg_orders):
        with open(file_name.format(reg_order), "rb") as f:
            data_se = pickle.load(f)

        alphas_se = data_se["alphas"]
        estim_errors_se = data_se["estim_errors_found"]
        adv_errors_se = data_se["adversarial_errors_found"]
        gen_errors_se = data_se["generalisation_errors_found"]

        ms_se = data_se["ms_found"]
        qs_se = data_se["qs_found"]
        ps_se = data_se["Ps_found"]
        sigmas_se = data_se["sigmas_found"]

        with open(file_name_erm.format(reg_order), "rb") as f:
            data_erm = pickle.load(f)

        alphas_num = data_erm["alphas"]
        estim_errors_mean = data_erm["estim_error_mean"]
        estim_errors_std = data_erm["estim_error_std"]

        gen_errors_mean = data_erm["gen_error_mean"]
        gen_errors_std = data_erm["gen_error_std"]

        adv_errors_mean = data_erm["adversarial_error_mean"]
        adv_errors_std = data_erm["adversarial_error_std"]

        ms_mean = data_erm["m_mean"]
        ms_std = data_erm["m_std"]

        qs_mean = data_erm["q_mean"]
        qs_std = data_erm["q_std"]

        ps_mean = data_erm["p_mean"]
        ps_std = data_erm["p_std"]

        plt.subplot(2, 3, 1)
        plt.plot(
            alphas_se,
            gen_errors_se,
            "-",
            color=f"C{i}",
            label=f"r = {reg_order} Gen",
        )
        plt.errorbar(
            alphas_num,
            gen_errors_mean,
            yerr=gen_errors_std,
            fmt=".",
            color=f"C{i}",
        )

        plt.subplot(2, 3, 2)
        plt.plot(
            alphas_se,
            adv_errors_se,
            "-",
            color=f"C{i}",
            label=f"r = {reg_order} Adv",
        )
        plt.errorbar(
            alphas_num,
            adv_errors_mean,
            yerr=adv_errors_std,
            fmt="x",
            color=f"C{i}",
        )

        plt.subplot(2, 3, 3)
        plt.plot(
            alphas_se,
            ps_se,
            "-",
            color=f"C{i}",
            label=f"r = {reg_order} P",
        )
        plt.errorbar(
            alphas_num,
            ps_mean,
            yerr=ps_std,
            fmt="x",
            color=f"C{i}",
        )

        plt.subplot(2, 3, 4)
        plt.plot(
            alphas_se,
            qs_se,
            "-",
            color=f"C{i}",
            label=f"r = {reg_order} q",
        )
        plt.errorbar(
            alphas_num,
            qs_mean,
            yerr=qs_std,
            fmt=".",
            color=f"C{i}",
        )

        plt.subplot(2, 3, 5)
        plt.plot(
            alphas_se,
            ms_se,
            "-",
            color=f"C{i}",
            label=f"r = {reg_order} m",
        )
        plt.errorbar(
            alphas_num,
            ms_mean,
            yerr=ms_std,
            fmt="x",
            color=f"C{i}",
        )

        plt.subplot(2, 3, 6)
        plt.plot(
            alphas_se,
            sigmas_se,
            "-",
            color=f"C{i}",
            label=f"r = {reg_order} sigma",
        )

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.xlabel(r"$\alpha$")
        plt.xscale("log")
        plt.legend()
        plt.grid()

    plt.show()
