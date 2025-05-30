from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic_adv,
    find_coefficients_Logistic_adv_Linf_L2,
)
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_linear_rf,
    find_adversarial_error_rf,
)
from linear_regression.data.generation import data_generation_hastie, measure_gen_probit_clasif
from linear_regression.erm.metrics import (
    generalisation_error_classification,
    adversarial_error_data,
    percentage_flipped_labels_estim,
    percentage_error_from_true,
)
from cvxpy.error import SolverError
from linear_regression.utils.errors import ConvergenceError
import numpy as np
import os
import sys

if len(sys.argv) > 1:
    gamma_min_se, gamma_max_se, n_gammas_se, d, alpha, delta = (
        float(sys.argv[1]),
        float(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
        float(sys.argv[5]),
        float(sys.argv[6]),
    )
else:
    gamma_min_se, gamma_max_se, n_gammas_se = 0.5, 2.0, 50
    alpha = 2.0
    delta = 0.0
    d = 500

reps = 10
n_gen = 1000

pstar = 1.0
reg = 2.0
eps_test = 1.0

gamma_min_erm, gamma_max_erm, n_gammas_erm = max(0.2, gamma_min_se), min(2.0, gamma_max_se), 10

data_folder = f"./data/hastie_model_training_optimal"

file_name_misclass = f"ERM_optimal_regp_misclass_alpha_{alpha:.2f}_gammas_{gamma_min_erm:.1f}_{gamma_max_erm:.1f}_{n_gammas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{pstar_t:.1f}_{pstar_g:.1f}_reg_{reg:.1f}.csv"
file_name_flipped = f"ERM_optimal_regp_flipped_alpha_{alpha:.2f}_gammas_{gamma_min_erm:.1f}_{gamma_max_erm:.1f}_{n_gammas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{pstar_t:.1f}_{pstar_g:.1f}_reg_{reg:.1f}.csv"
file_name_bound = f"ERM_optimal_regp_bound_alpha_{alpha:.2f}_gammas_{gamma_min_erm:.1f}_{gamma_max_erm:.1f}_{n_gammas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{pstar_t:.1f}_{pstar_g:.1f}_reg_{reg:.1f}.csv"
file_name_adverr = f"ERM_optimal_regp_adverr_alpha_{alpha:.2f}_gammas_{gamma_min_erm:.1f}_{gamma_max_erm:.1f}_{n_gammas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{pstar_t:.1f}_{pstar_g:.1f}_reg_{reg:.1f}.csv"

file_name_misclass_SE = f"SE_optimal_regp_misclass_alpha_{alpha:.2f}_gammas_{gamma_min_se:.1f}_{gamma_max_se:.1f}_{n_gammas_se:d}_pstar_{pstar_t:.1f}_{pstar_g:.1f}_reg_{reg:.1f}.csv"
file_name_flipped_SE = f"SE_optimal_regp_flipped_alpha_{alpha:.2f}_gammas_{gamma_min_se:.1f}_{gamma_max_se:.1f}_{n_gammas_se:d}_pstar_{pstar_t:.1f}_{pstar_g:.1f}_reg_{reg:.1f}.csv"
file_name_bound_SE = f"SE_optimal_regp_bound_alpha_{alpha:.2f}_gammas_{gamma_min_se:.1f}_{gamma_max_se:.1f}_{n_gammas_se:d}_pstar_{pstar_t:.1f}_{pstar_g:.1f}_reg_{reg:.1f}.csv"
file_name_adverr_SE = f"SE_optimal_regp_adverr_alpha_{alpha:.2f}_gammas_{gamma_min_se:.1f}_{gamma_max_se:.1f}_{n_gammas_se:d}_pstar_{pstar_t:.1f}_{pstar_g:.1f}_reg_{reg:.1f}.csv"

if not os.path.exists(data_folder):
    os.makedirs(data_folder)


def perform_sweep(metric_name, file_name_SE_template, file_name_output):
    """
    Perform a parameter sweep for a specific metric.

    Args:
        metric_name: Name of the metric (for display purposes)
        file_name_SE_template: Template for the SE file name
        file_name_output: Output file name
    """
    if os.path.exists(os.path.join(data_folder, file_name_SE_template)):
        print(f"SE file {file_name_SE_template} exists.")
    else:
        print(f"SE file {file_name_SE_template} does not exist. Exiting.")
        return {}

    SE_data = np.loadtxt(
        os.path.join(data_folder, file_name_SE_template),
        delimiter=",",
        skiprows=1,
    )

    gammas_SE = SE_data[:, 0]
    gamma_list = np.linspace(gamma_min_erm, gamma_max_erm, n_gammas_erm)
    indices = np.searchsorted(gammas_SE, gamma_list)
    gamma_list = SE_data[indices, 0]
    reg_param_list = SE_data[indices, -1]

    ms = np.empty((n_gammas_erm, 2))
    qs = np.empty((n_gammas_erm, 2))
    q_latent = np.empty((n_gammas_erm, 2))
    q_feature = np.empty((n_gammas_erm, 2))
    Ps = np.empty((n_gammas_erm, 2))
    gen_errs = np.empty((n_gammas_erm, 2))
    adv_errs = np.empty((n_gammas_erm, 2))
    flipped_fairs = np.empty((n_gammas_erm, 2))
    misclas_fairs = np.empty((n_gammas_erm, 2))
    bound_errs = np.empty((n_gammas_erm, 2))

    for i, (gamma, reg_param) in enumerate(zip(gamma_list, reg_param_list)):
        p = int(d / gamma)
        n = int(alpha * d)

        print(f"p {p} d {d} n {n}")

        m_vals = []
        q_vals = []
        q_latent_vals = []
        q_feature_vals = []
        P_vals = []
        gen_err_vals = []
        adv_err_vals = []
        flip_fair_vals = []
        misc_fair_vals = []
        bound_vals = []

        j = 0
        while j < reps:
            xs, ys, zs, xs_gen, ys_gen, zs_gen, wstar, F, noise, noise_gen = data_generation_hastie(
                measure_gen_probit_clasif, d, n, n_gen, (delta,), gamma, noi=True
            )

            try:
                w = find_coefficients_Logistic_adv_Linf_L2(ys, xs, 0.5 * reg_param, 0.0)
            except (ValueError, SolverError) as e:
                print(
                    f"minimization didn't converge on iteration {j} for gamma {gamma:.2f}. Trying again."
                )
                continue

            m_vals.append(np.dot(wstar, F.T @ w) / (p * np.sqrt(gamma)))
            q_vals.append(np.dot(F.T @ w, F.T @ w) / p + np.dot(w, w) / p)
            q_latent_vals.append(np.dot(F.T @ w, F.T @ w) / d)
            q_feature_vals.append(np.dot(w, w) / p)
            P_vals.append(np.mean(np.abs(w)))

            yhat_gen = np.sign(np.dot(xs_gen, w))

            gen_err_vals.append(generalisation_error_classification(ys_gen, xs_gen, w, wstar))

            adv_perturbation = find_adversarial_error_rf(
                ys_gen, zs_gen, w, F.T, wstar, eps_test / np.sqrt(d), "inf"
            )
            adv_err = np.mean(
                ys_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w)
            )
            adv_err_vals.append(adv_err)

            adv_perturbation = find_adversarial_perturbation_linear_rf(
                ys_gen, zs_gen, w, F.T, wstar, eps_test / np.sqrt(d), "inf"
            )
            bound = np.mean(
                (ys_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w))
                * (ys_gen == yhat_gen)
            )
            bound_vals.append(bound)

            adv_perturbation = find_adversarial_perturbation_linear_rf(
                yhat_gen, zs_gen, w, F.T, wstar, eps_test / np.sqrt(d), "inf"
            )
            flipped = np.mean(
                yhat_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w)
            )
            flip_fair_vals.append(flipped)

            adv_perturbation = find_adversarial_perturbation_linear_rf(
                ys_gen, zs_gen, w, F.T, wstar, eps_test / np.sqrt(d), "inf"
            )
            misclass = np.mean(
                ys_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w)
            )
            misc_fair_vals.append(misclass)

            print(f"repetition {j} for gamma {gamma:.2f} done.")
            j += 1

        ms[i, 0], ms[i, 1] = np.mean(m_vals), np.std(m_vals)
        qs[i, 0], qs[i, 1] = np.mean(q_vals), np.std(q_vals)
        q_latent[i, 0], q_latent[i, 1] = np.mean(q_latent_vals), np.std(q_latent_vals)
        q_feature[i, 0], q_feature[i, 1] = np.mean(q_feature_vals), np.std(q_feature_vals)
        Ps[i, 0], Ps[i, 1] = np.mean(P_vals), np.std(P_vals)
        gen_errs[i, 0], gen_errs[i, 1] = np.mean(gen_err_vals), np.std(gen_err_vals)
        adv_errs[i, 0], adv_errs[i, 1] = np.mean(adv_err_vals), np.std(adv_err_vals)
        flipped_fairs[i, 0], flipped_fairs[i, 1] = np.mean(flip_fair_vals), np.std(flip_fair_vals)
        misclas_fairs[i, 0], misclas_fairs[i, 1] = np.mean(misc_fair_vals), np.std(misc_fair_vals)
        bound_errs[i, 0], bound_errs[i, 1] = np.mean(bound_vals), np.std(bound_vals)

        print(f"gamma {gamma:.2f} done.")

    columns = (
        gamma_list,
        ms[:, 0],
        ms[:, 1],
        qs[:, 0],
        qs[:, 1],
        q_latent[:, 0],
        q_latent[:, 1],
        q_feature[:, 0],
        q_feature[:, 1],
        Ps[:, 0],
        Ps[:, 1],
        gen_errs[:, 0],
        gen_errs[:, 1],
        adv_errs[:, 0],
        adv_errs[:, 1],
        flipped_fairs[:, 0],
        flipped_fairs[:, 1],
        misclas_fairs[:, 0],
        misclas_fairs[:, 1],
        bound_errs[:, 0],
        bound_errs[:, 1],
        reg_param_list,
    )

    header = (
        "gamma, m_mean, m_std, q_mean, q_std, q_latent_mean, q_latent_std, "
        "q_feature_mean, q_feature_std, P_mean, P_std, gen_err_mean, gen_err_std, "
        "adv_err_mean, adv_err_std, flipped_fairs_mean, flipped_fairs_std, "
        "misclas_fairs_mean, misclas_fairs_std, bound_err_mean, bound_err_std,"
        "reg_param"
    )

    # Save results to file
    np.savetxt(
        os.path.join(data_folder, file_name_output),
        np.column_stack(columns),
        delimiter=",",
        header=header,
    )

    print(f"Data saved for {metric_name}.")
    return {"gamma_list": gamma_list, "gen_errs": gen_errs, "adv_errs": adv_errs}


# Run the sweeps for each metric
misclass_results = perform_sweep("misclassification", file_name_misclass_SE, file_name_misclass)
# flipped_results = perform_sweep("flipped labels", file_name_flipped_SE, file_name_flipped)
bound_results = perform_sweep("boundary error", file_name_bound_SE, file_name_bound)
adverr_results = perform_sweep("adversarial error", file_name_adverr_SE, file_name_adverr)
