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


def log_log_linear_fit(x, y, base=10, return_points=False, extend_percent=0.1):
    x = np.array(x)
    y = np.array(y)

    log_x = np.log(x) / np.log(base)
    log_y = np.log(y) / np.log(base)

    A = np.vstack([log_x, np.ones(len(log_x))]).T
    m, c = np.linalg.lstsq(A, log_y, rcond=None)[0]

    coefficient = base**c

    if return_points:
        log_x_min, log_x_max = np.log10(min(x)), np.log10(max(x))
        log_x_range = log_x_max - log_x_min
        extended_log_x_min = log_x_min - extend_percent * log_x_range
        extended_log_x_max = log_x_max + extend_percent * log_x_range

        x_fit = np.logspace(extended_log_x_min, extended_log_x_max, 100)
        y_fit = coefficient * x_fit**m

        return m, coefficient, (x_fit, y_fit)
    else:
        return m, coefficient


alpha_min, alpha_max, n_alpha_pts = 0.01, 1, 200
reg_orders = [1, 2, 3, 4]
eps_t = 0.3
eps_g = eps_t
reg_param = 1e-2
pstar = 1
alpha_cutoff = 0.01

run_experiments = True

data_folder = "./data/SE_any_norm"

file_name = f"SE_data_pstar_{pstar}_reg_order_{{}}_alpha_{alpha_min:.3f}_{alpha_max:.3f}_reg_param_{reg_param:.1e}_eps_t_g_{eps_t:.1e}_{eps_g:.1e}_overlaps.csv"

if __name__ == "__main__":
    for i, reg_order in enumerate(reg_orders):
        if run_experiments and not exists(join(data_folder, file_name.format(reg_order))):

            alphas_se = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)

            ms_found = np.empty((n_alpha_pts,))
            qs_found = np.empty((n_alpha_pts,))
            Vs_found = np.empty((n_alpha_pts,))
            Ps_found = np.empty((n_alpha_pts,))
            mhats_found = np.empty((n_alpha_pts,))
            qhats_found = np.empty((n_alpha_pts,))
            Phats_found = np.empty((n_alpha_pts,))
            Vhats_found = np.empty((n_alpha_pts,))

            if reg_order == 1:
                m, q, V, P = (4.809, 61.102, 318.050, 4.514)
                initial_condition = (m, q, V, P)
            elif reg_order == 2:
                m, q, V, P = (2.076, 9.021, 17.5593, 2.2070)
                initial_condition = (m, q, V, P)
            elif reg_order == 3:
                m, q, V, P = (1.1902, 2.976, 5.5135, 1.372)
                initial_condition = (m, q, V, P)
            elif reg_order == 4:
                m, q, V, P = (0.879, 1.6953, 3.378, 1.07302)
                initial_condition = (m, q, V, P)
            else:
                m, q, V, P = (0.6604, 13.959, 25767.13, 1.364)
                initial_condition = (m, q, V, P)

            for jprime, alpha in enumerate(reversed(alphas_se)):
                # j = jprime
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
                    abs_tol=1e-6,
                    min_iter=10,
                    args_update_function=(0.2,),
                )

                if jprime == 0:
                    print("Initial condition")
                    print(ms_found[j], qs_found[j], Vs_found[j], Ps_found[j])

                initial_condition = (ms_found[j], qs_found[j], Vs_found[j], Ps_found[j])

                mhats_found[j], qhats_found[j], Vhats_found[j], Phats_found[j] = (
                    f_hat_Logistic_no_noise_Linf_adv_classif(
                        ms_found[j], qs_found[j], Vs_found[j], Ps_found[j], eps_t, alpha
                    )
                )

            data = {
                "alphas": alphas_se,
                "ms_found": ms_found,
                "qs_found": qs_found,
                "Vs_found": Vs_found,
                "Ps_found": Ps_found,
                "mhats_found": mhats_found,
                "qhats_found": qhats_found,
                "Vhats_found": Vhats_found,
                "Phats_found": Phats_found,
            }

            with open(join(data_folder, file_name.format(reg_order)), "rb") as f:
                # Create the array from the data in the order of the keys
                data_array = np.column_stack([data[key] for key in data.keys()])
                # Create the header for the csv file
                header = ",".join(data.keys())
                # Save the file using np.savetxt
                np.savetxt(
                    join(data_folder, file_name.format(reg_order)),
                    data_array,
                    delimiter=",",
                    header=header,
                    comments="",
                )

    plt.figure(figsize=(10, 2.5 * len(reg_orders)))
    for i, reg_order in enumerate(reg_orders):
        with open(join(data_folder, file_name.format(reg_order)), "rb") as f:
            data_se = np.loadtxt(f, delimiter=",", skiprows=1)

        alphas_se = data_se[:, 0]

        ms_se = data_se[:, 1]
        qs_se = data_se[:, 2]
        ps_se = data_se[:, 3]
        Vs_se = data_se[:, 4]

        mhats_se = data_se[:, 5]
        qhats_se = data_se[:, 6]
        phats_se = data_se[:, 7]
        Vhats_se = data_se[:, 8]

        overlaps_se = [ms_se, qs_se, ps_se, Vs_se, ms_se / np.sqrt(qs_se)]
        overlaps_hats_se = [mhats_se, qhats_se, phats_se, Vhats_se]

        names = ["m", "q", "P", "$V$", "$\\frac{{m}}{\\sqrt{q}}$"]
        names_hat = ["$\\hat{{m}}$", "$\\hat{{q}}$", "$\\hat{{P}}$", "$\\hat{{V}}$"]

        plt.subplot(len(reg_orders), 2, i * 2 + 1)
        for j, ov in enumerate(overlaps_se):
            plt.plot(
                alphas_se,
                ov,
                linestyle="-",
                color=f"C{j}",
                alpha=0.5,
                label=names[j],
            )
            # ones_to_keep = np.where(alphas_se < alpha_cutoff)
            # m, c, (x_lin, y_lin) = log_log_linear_fit(
            #     alphas_se[ones_to_keep], ov[ones_to_keep], return_points=True
            # )
            # plt.plot(
            #     x_lin,
            #     y_lin,
            #     linestyle="--",
            #     color=f"C{j}",
            #     label=f"{names[j]} {c:.1f} $\\alpha^{{{m:.2f}}}$",
            # )

        plt.subplot(len(reg_orders), 2, i * 2 + 2)
        for j, hat_ov in enumerate(overlaps_hats_se):
            plt.plot(
                alphas_se,
                hat_ov,
                linestyle="-",
                color=f"C{j}",
                alpha=0.5,
                label=names_hat[j],
            )
            # ones_to_keep = np.where(alphas_se < alpha_cutoff)
            # m, c, (x_lin, y_lin) = log_log_linear_fit(
            #     alphas_se[ones_to_keep], hat_ov[ones_to_keep], return_points=True
            # )
            # plt.plot(
            #     x_lin,
            #     y_lin,
            #     linestyle="--",
            #     color=f"C{j}",
            #     label=f"{names_hat[j]} {c:.1f} $\\alpha^{{{m:.2f}}}$",
            # )

        names = [f"Non Hat reg_order = {reg_order:.1f}", f"Hat reg_order = {reg_order:.1f}"]
        limits = [[0.2, 0.5], [0.2, 0.5]]
        for k, (nn, lms) in enumerate(zip(names, limits)):
            plt.subplot(len(reg_orders), 2, i * 2 + k + 1)
            plt.xlabel(r"$\alpha$")
            plt.ylabel(nn)
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()
            plt.grid()

    plt.suptitle(
        r"$L_{{\infty}}$ attack with regularisation $L_r$ - $\varepsilon_t$ = {:.1e}, $\varepsilon_g$ = {:.1e}".format(
            eps_t, eps_g
        )
    )
    plt.tight_layout()

    plt.show()
