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


alpha_min, alpha_max, n_alpha_pts = 0.05, 1, 50
reg_orders = [1, 2, 3]
eps_t = 0.3
eps_g = eps_t
reg_param = 1e-2
pstar = 1

run_experiments = False

data_folder = "./data/SE_any_norm"

file_name = f"SE_data_pstar_{pstar}_reg_order_{{}}_alpha_{alpha_min:.3f}_{alpha_max:.3f}_reg_param_{reg_param:.1e}_eps_t_g_{eps_t:.1e}_{eps_g:.1e}.csv"

if __name__ == "__main__":
    for i, reg_order in enumerate(reg_orders):
        if run_experiments and not exists(join(data_folder, file_name.format(reg_order))):

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

            for jprime, alpha in enumerate(reversed(alphas_se)):
                j = jprime
                j = n_alpha_pts - jprime - 1
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

            with open(join(data_folder, file_name.format(reg_order)), "wb") as f:
                # Create the array from the data in the order of the keys
                data_array = np.column_stack([data[key] for key in data.keys()])
                # Create the header for the csv file
                header = ",".join(data.keys())
                # Save the data array to a csv file
                np.savetxt(
                    join(data_folder, file_name.format(reg_order)),
                    data_array,
                    delimiter=",",
                    header=header,
                    comments="",
                )

    plt.style.use("./latex_ready.mplstyle")
    plt.figure(figsize=(10, 3))
    for i, reg_order in enumerate(reg_orders):
        with open(join(data_folder, file_name.format(reg_order)), "rb") as f:
            data_se = np.loadtxt(f, delimiter=",", skiprows=1)

        alphas_se = data_se[:, 0]
        ms_se = data_se[:, 1]
        qs_se = data_se[:, 2]
        Vs_se = data_se[:, 3]
        Ps_se = data_se[:, 4]
        estim_errors_se = data_se[:, 5]
        adv_errors_se = data_se[:, 6]
        gen_errors_se = data_se[:, 7]

        plt.subplot(1, 3, 1)
        plt.plot(
            alphas_se,
            gen_errors_se,
            "-",
            color=f"C{i}",
            label=f"$r$ = {reg_order}",
        )

        plt.subplot(1, 3, 2)
        plt.plot(
            alphas_se,
            adv_errors_se,
            "-",
            color=f"C{i}",
            label=f"$r$ = {reg_order}",
        )

        plt.subplot(1, 3, 3)
        plt.plot(
            alphas_se,
            adv_errors_se - gen_errors_se,
            "-",
            color=f"C{i}",
            label=f"$r$ = {reg_order}",
        )

    names = ["$E_{\\mathrm{gen}}$", "$E_{\\mathrm{adv}}$", "$E_{\\mathrm{bnd}}$"]
    limits = [[0.2, 0.5], [0.2, 0.5], [0.4, 5], [0.4, 50], [0.1, 5], [0, 5e2]]
    for i, (nn, lms) in enumerate(zip(names, limits)):
        plt.subplot(1, 3, i + 1)
        plt.xlabel(r"$\alpha$")
        plt.ylabel(nn)
        plt.xscale("log")
        # plt.xlim([alpha_min, 5])
        plt.yscale("log")
        # plt.ylim(lms)
        plt.legend()
        plt.grid(which="both")

    # plt.suptitle(
    #     r"$L_{{\infty}}$ attack with regularisation $L_r$ - $\varepsilon_t$ = {:.1f}, $\varepsilon_g$ = {:.1f}".format(
    #         eps_t, eps_g
    #     )
    # )

    plt.tight_layout()

    plt.show()
