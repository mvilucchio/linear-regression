import matplotlib.pyplot as plt
import numpy as np
from linear_regression.erm.erm_solvers import find_coefficients_Logistic_adv
from linear_regression.data.generation import data_generation, measure_gen_no_noise_clasif
from linear_regression.erm.metrics import (
    estimation_error_data,
    generalisation_error_classification,
    adversarial_error_data,
)
import pickle
from tqdm.auto import tqdm

alpha_min, alpha_max, n_alpha_pts = 0.1, 100, 22
reg_orders = [
    1,
    2,
    3,
    4,
]
eps_t = 0.1
eps_g = 0.1
reg_param = 1e-4
pstar = 1.0

run_experiments = True

d = 1000
reps = 10
n_gen = 1000

file_name = f"ERM_data_Linf_reg_order_{{:d}}_alpha_{alpha_min:.3f}_{alpha_max:.3f}_{n_alpha_pts:d}_dim_{d:d}_reps_{reps:d}_reg_param_{reg_param:.1e}_eps_t_g_{eps_t:.1e}_{eps_g:.1e}.pkl"

if __name__ == "__main__":
    if run_experiments:
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)

        for i, reg_order in enumerate(reg_orders):
            q_mean = np.empty_like(alphas)
            q_std = np.empty_like(alphas)

            m_mean = np.empty_like(alphas)
            m_std = np.empty_like(alphas)

            p_mean = np.empty_like(alphas)
            p_std = np.empty_like(alphas)

            train_error_mean = np.empty_like(alphas)
            train_error_std = np.empty_like(alphas)

            gen_error_mean = np.empty_like(alphas)
            gen_error_std = np.empty_like(alphas)

            estim_errors_mean = np.empty_like(alphas)
            estim_errors_std = np.empty_like(alphas)

            adversarial_errors_mean = np.empty_like(alphas)
            adversarial_errors_std = np.empty_like(alphas)

            for j, alpha in enumerate(alphas):
                n = int(alpha * d)
                print(
                    f"Running reg_order = {reg_order}, alpha = {alpha:.4f} (= {n:d} samples / {d:d} features)"
                )

                tmp_estim_errors = []
                tmp_train_errors = []
                tmp_gen_errors = []
                tmp_adversarial_errors = []
                tmp_qs = []
                tmp_ms = []
                tmp_ps = []

                iter = 0
                # for _ in tqdm(range(reps), leave=False):
                pbar = tqdm(total=reps)
                while iter < reps:
                    xs_train, ys_train, xs_gen, ys_gen, wstar = data_generation(
                        measure_gen_no_noise_clasif, d, n, n_gen, tuple()
                    )

                    try:
                        w = find_coefficients_Logistic_adv(
                            ys_train, xs_train, reg_param, eps_t, reg_order, pstar, wstar
                        )
                    except ValueError as e:
                        print(e)
                        continue

                    tmp_estim_errors.append(estimation_error_data(ys_gen, xs_gen, w, wstar))
                    tmp_qs.append(np.sum(w**2) / d)
                    tmp_ms.append(np.dot(wstar, w) / d)
                    tmp_ps.append(np.sum(np.abs(w) ** pstar) / d)
                    tmp_train_errors.append(
                        adversarial_error_data(ys_train, xs_train, w, wstar, eps_t, pstar)
                    )
                    tmp_gen_errors.append(
                        generalisation_error_classification(ys_gen, xs_gen, w, wstar)
                    )
                    tmp_adversarial_errors.append(
                        adversarial_error_data(ys_gen, xs_gen, w, wstar, eps_g, pstar)
                    )

                    del w
                    del xs_gen
                    del ys_gen
                    del xs_train
                    del ys_train
                    del wstar

                    iter += 1
                    pbar.update(1)

                pbar.close()

                estim_errors_mean[j] = np.mean(tmp_estim_errors)
                estim_errors_std[j] = np.std(tmp_estim_errors)

                q_mean[j] = np.mean(tmp_qs)
                q_std[j] = np.std(tmp_qs)

                m_mean[j] = np.mean(tmp_ms)
                m_std[j] = np.std(tmp_ms)

                p_mean[j] = np.mean(tmp_ps)
                p_std[j] = np.std(tmp_ps)

                train_error_mean[j] = np.mean(tmp_train_errors)
                train_error_std[j] = np.std(tmp_train_errors)

                gen_error_mean[j] = np.mean(tmp_gen_errors)
                gen_error_std[j] = np.std(tmp_gen_errors)

                adversarial_errors_mean[j] = np.mean(tmp_adversarial_errors)
                adversarial_errors_std[j] = np.std(tmp_adversarial_errors)

            data_dict = {
                "alphas": alphas,
                "q_mean": q_mean,
                "q_std": q_std,
                "m_mean": m_mean,
                "m_std": m_std,
                "p_mean": p_mean,
                "p_std": p_std,
                "train_error_mean": train_error_mean,
                "train_error_std": train_error_std,
                "gen_error_mean": gen_error_mean,
                "gen_error_std": gen_error_std,
                "estim_error_mean": estim_errors_mean,
                "estim_error_std": estim_errors_std,
                "adversarial_error_mean": adversarial_errors_mean,
                "adversarial_error_std": adversarial_errors_std,
            }

            with open(file_name.format(reg_order), "wb") as f:
                pickle.dump(data_dict, f)

    plt.figure(figsize=(15, 5))

    for i, reg_order in enumerate(reg_orders):
        with open(file_name.format(reg_order), "rb") as f:
            data_dict = pickle.load(f)

        alphas = data_dict["alphas"]

        train_error_mean = data_dict["train_error_mean"]
        train_error_std = data_dict["train_error_std"]

        gen_error_mean = data_dict["gen_error_mean"]
        gen_error_std = data_dict["gen_error_std"]

        adversarial_errors_mean = data_dict["adversarial_error_mean"]
        adversarial_errors_std = data_dict["adversarial_error_std"]

        q_mean = data_dict["q_mean"]
        q_std = data_dict["q_std"]

        m_mean = data_dict["m_mean"]
        m_std = data_dict["m_std"]

        p_mean = data_dict["p_mean"]
        p_std = data_dict["p_std"]

        plt.subplot(1, 3, 1)
        plt.errorbar(
            alphas,
            q_mean,
            yerr=q_std,
            label=f"q r = {reg_order}",
            fmt="x",
            linestyle="-",
        )

        plt.subplot(1, 3, 2)
        plt.errorbar(
            alphas,
            m_mean,
            yerr=m_std,
            label=f"m r = {reg_order}",
            fmt="x",
            linestyle="-",
        )

        plt.subplot(1, 3, 3)
        plt.errorbar(
            alphas,
            p_mean,
            yerr=p_std,
            label=f"p r = {reg_order}",
            fmt="x",
            linestyle="-",
        )

        # plt.errorbar(
        #     alphas,
        #     train_error_mean,
        #     yerr=train_error_std,
        #     fmt=".",
        #     label=f"TRAIN r = {reg_order}",
        # )

    # plt.title(r"L$\infty$ attack with regularisation $L r$")
    plt.subplot(1, 3, 1)
    plt.xlabel(r"$\alpha$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.xlabel(r"$\alpha$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.xlabel(r"$\alpha$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()

    plt.show()
