import matplotlib.pyplot as plt
import numpy as np
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.aux_functions.misc import classification_adversarial_error
from linear_regression.fixed_point_equations.regularisation.pstar_attacks_Lr_reg import (
    f_Lr_regularisation_Lpstar_attack,
)
from linear_regression.fixed_point_equations.classification.Adv_train_p_norm import (
    f_hat_Logistic_no_noise_Linf_adv_classif,
)
import pickle
from os.path import join, exists


alpha_min, alpha_max, n_alpha_pts = 0.1, 50, 50
reg_orders = [1, 2]
eps_t = 0.3
eps_g = eps_t
reg_param = 1e-2
pstar = 1

run_experiments = True

data_folder_SE = "./data/SE_any_norm"

file_name = f"SE_data_pstar_{pstar}_reg_order_{{}}_pstar_{pstar}_alpha_{alpha_min:.3f}_{alpha_max:.3f}_reg_param_{reg_param:.1e}_eps_t_g_{eps_t:.1e}_{eps_g:.1e}.csv"

data_folder_ERM = "./data/ERM_any_norm"

reps = 10
alpha_min_erm, alpha_max_erm, n_alpha_pts_erm = 0.1, 10, 15
d = 500
file_name_erm = f"ERM_data_pstar_{pstar}_reg_order_{{:d}}_alpha_{alpha_min_erm:.3f}_{alpha_max_erm:.3f}_{n_alpha_pts_erm:d}_dim_{d:d}_reps_{reps:d}_reg_param_{reg_param:.1e}_eps_t_g_{eps_t:.1e}_{eps_g:.1e}.csv"

alpha_min_erm_1, alpha_max_erm_1, n_alpha_pts_erm_1 = 0.1, 26.827, 18
d_1 = 500
file_name_erm_1 = f"ERM_data_pstar_{pstar}_reg_order_{{:d}}_alpha_{alpha_min_erm_1:.3f}_{alpha_max_erm_1:.3f}_{n_alpha_pts_erm_1:d}_dim_{d_1:d}_reps_{reps:d}_reg_param_{reg_param:.1e}_eps_t_g_{eps_t:.1e}_{eps_g:.1e}.csv"

if __name__ == "__main__":
    for i, reg_order in enumerate(reg_orders):
        if run_experiments and not exists(join(data_folder_SE, file_name.format(reg_order))):

            alphas_se = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)

            ms_found = np.empty((n_alpha_pts,))
            qs_found = np.empty((n_alpha_pts,))
            Vs_found = np.empty((n_alpha_pts,))
            Ps_found = np.empty((n_alpha_pts,))
            estim_errors_se = np.empty((n_alpha_pts,))
            adversarial_errors_found = np.empty((n_alpha_pts,))
            gen_errors_se = np.empty((n_alpha_pts,))

            if reg_order == 1:
                m, q, V, P = (0.348751, 9.11468, 270.038, 0.615440)
                initial_condition = (m, q, V, P)
            elif reg_order == 2:
                m, q, V, P = (6.766e-01, 5.780e00, 4.442e03, 1.780e00)
                initial_condition = (m, q, V, P)
            elif reg_order == 3:
                m, q, V, P = (3.873e-01, 2.571e00, 1.710e03, 1.446e00)
                initial_condition = (m, q, V, P)
            elif reg_order == 4:
                m, q, V, P = (3.417e-01, 2.140e00, 8.220e02, 1.371e00)
                initial_condition = (m, q, V, P)
            else:
                m, q, V, P = (0.6604, 13.959, 25767.13, 1.364)
                initial_condition = (m, q, V, P)

            for jprime, alpha in enumerate(alphas_se):
                j = jprime
                print("\033[91m" + f"SE {reg_order = }, {alpha = :.3e}" + "\033[0m")

                f_kwargs = {"reg_param": reg_param, "reg_order": reg_order, "pstar": 1}
                f_hat_kwargs = {"alpha": alpha, "eps_t": eps_t}

                ms_found[j], qs_found[j], Vs_found[j], Ps_found[j] = fixed_point_finder(
                    f_Lr_regularisation_Lpstar_attack,
                    f_hat_Logistic_no_noise_Linf_adv_classif,
                    initial_condition,
                    f_kwargs,
                    f_hat_kwargs,
                    abs_tol=1e-7,
                    min_iter=10,
                )

                initial_condition = (ms_found[j], qs_found[j], Vs_found[j], Ps_found[j])

                estim_errors_se[j] = 1 - 2 * ms_found[j] + qs_found[j]
                adversarial_errors_found[j] = classification_adversarial_error(
                    ms_found[j], qs_found[j], Ps_found[j], eps_g, pstar
                )
                gen_errors_se[j] = np.arccos(ms_found[j] / np.sqrt(qs_found[j])) / np.pi

            data = {
                "alphas": alphas_se,
                "ms_found": ms_found,
                "qs_found": qs_found,
                "Vs_found": Vs_found,
                "Ps_found": Ps_found,
                "estim_errors_found": estim_errors_se,
                "adversarial_errors_found": adversarial_errors_found,
                "generalisation_errors_found": gen_errors_se,
            }

            with open(join(data_folder_SE, file_name.format(reg_order)), "wb") as f:
                # pickle.dump(data, f)
                data_array = np.column_stack([data[key] for key in data.keys()])
                header = ",".join(data.keys())
                np.savetxt(
                    join(data_folder_SE, file_name.format(reg_order)),
                    data_array,
                    header=header,
                    delimiter=",",
                    comments="",
                )

    plt.figure(figsize=(15, 8))
    for i, reg_order in enumerate(reg_orders):
        with open(join(data_folder_SE, file_name.format(reg_order)), "rb") as f:
            # data_se = pickle.load(f)
            data_se = np.genfromtxt(f, delimiter=",", names=True)

        alphas_se = data_se["alphas"]
        estim_errors_se = data_se["estim_errors_found"]
        adv_errors_se = data_se["adversarial_errors_found"]
        gen_errors_se = data_se["generalisation_errors_found"]

        ms_se = data_se["ms_found"]
        qs_se = data_se["qs_found"]
        ps_se = data_se["Ps_found"]
        Vs_se = data_se["Vs_found"]

        # alphas_se = data_se["alphas"]
        # estim_errors_se = data_se["estim_errors_found"]
        # adv_errors_se = data_se["adversarial_errors_found"]
        # gen_errors_se = data_se["generalisation_errors_found"]

        # ms_se = data_se["ms_found"]
        # qs_se = data_se["qs_found"]
        # ps_se = data_se["Ps_found"]
        # Vs_se = data_se["Vs_found"]

        if reg_order == 1:
            with open(join(data_folder_ERM, file_name_erm_1.format(reg_order)), "rb") as f:
                # data_erm = pickle.load(f)
                data_erm = np.genfromtxt(f, delimiter=",", names=True)
        else:
            with open(join(data_folder_ERM, file_name_erm.format(reg_order)), "rb") as f:
                # data_erm = pickle.load(f)
                data_erm = np.genfromtxt(f, delimiter=",", names=True)

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
            label=f"r = {reg_order}",
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
            label=f"r = {reg_order}",
        )
        plt.errorbar(
            alphas_num,
            adv_errors_mean,
            yerr=adv_errors_std,
            fmt="x",
            color=f"C{i}",
        )

        plt.subplot(2, 3, 3)
        # plt.plot(
        #     alphas_se,
        #     ps_se,
        #     "-",
        #     color=f"C{i}",
        #     label=f"r = {reg_order}",
        # )
        plt.errorbar(
            alphas_num,
            adv_errors_mean - gen_errors_mean,
            yerr=np.sqrt(adv_errors_std**2 + gen_errors_std**2),
            fmt="x",
            color=f"C{i}",
        )

        plt.subplot(2, 3, 4)
        plt.plot(
            alphas_se,
            qs_se,
            "-",
            color=f"C{i}",
            label=f"r = {reg_order}",
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
            label=f"r = {reg_order}",
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
            Vs_se,
            "-",
            color=f"C{i}",
            label=f"r = {reg_order}",
        )

    names = ["Gen", "Adv Err", "P", "q", "m", "V"]
    limits = [[0.2, 0.5], [0.2, 0.5], [0.4, 5], [0.4, 50], [0.1, 5], [0, 5e2]]
    for i, (nn, lms) in enumerate(zip(names, limits)):
        plt.subplot(2, 3, i + 1)
        plt.xlabel(r"$\alpha$")
        plt.ylabel(nn)
        plt.xscale("log")
        # plt.xlim([alpha_min, 5])
        plt.yscale("log")
        # plt.ylim(lms)
        plt.legend()
        plt.grid()

    plt.suptitle(
        r"$L_{{\infty}}$ attack with regularisation $L_r$ - $\varepsilon_t$ = {:.1e}, $\varepsilon_g$ = {:.1e}".format(
            eps_t, eps_g
        )
    )

    plt.show()
