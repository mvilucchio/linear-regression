from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic_adv_Linf_L2,
    find_coefficients_Logistic_adv_Linf_L1,
)
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_direct_space_noteacher,
    find_adversarial_perturbation_direct_space,
)
from linear_regression.data.generation import data_generation, measure_gen_probit_clasif
from linear_regression.erm.metrics import generalisation_error_classification
from cvxpy.error import SolverError
import numpy as np
import os
import sys

if len(sys.argv) > 1:
    alpha_min_se, alpha_max_se, n_alphas_se, d, delta, pstar, reg_p, metric_name_chosen = (
        float(sys.argv[1]),
        float(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
        float(sys.argv[5]),
        float(sys.argv[6]),
        float(sys.argv[7]),
        sys.argv[8],
    )
else:
    alpha_min_se, alpha_max_se, n_alphas_se = 0.25, 3.0, 50
    delta = 0.0
    d = 500
    pstar = 1.0
    reg_p = 2.0
    metric_name_chosen = "misclass"

reps = 10
n_gen = 1000
eps_test = 1.0

if pstar == 2.0:
    adv_geometry = 2.0
elif pstar == 1.0:
    adv_geometry = "inf"

alpha_min_erm, alpha_max_erm, n_alphas_erm = max(0.5, alpha_min_se), min(5.0, alpha_max_se), 10

data_folder = f"./data/direct_space_model_training_optimal"

file_name_misclass = f"ERM_optimal_regp_misclass_direct_alphas_{alpha_min_erm:.1f}_{alpha_max_erm:.1f}_{n_alphas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{pstar:.1f}_reg_{reg_p:.1f}.csv"
file_name_adverr = f"ERM_optimal_regp_adverr_direct_alphas_{alpha_min_erm:.1f}_{alpha_max_erm:.1f}_{n_alphas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{pstar:.1f}_reg_{reg_p:.1f}.csv"
file_name_bound = f"ERM_optimal_regp_bound_direct_alphas_{alpha_min_erm:.1f}_{alpha_max_erm:.1f}_{n_alphas_erm:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{pstar:.1f}_reg_{reg_p:.1f}.csv"


file_name_misclass_SE = f"SE_optimal_regp_misclass_direct_alphas_{alpha_min_se:.1f}_{alpha_max_se:.1f}_{n_alphas_se:d}_pstar_{pstar:.1f}_reg_{reg_p:.1f}.csv"
file_name_adverr_SE = f"SE_optimal_regp_adverr_direct_alphas_{alpha_min_se:.1f}_{alpha_max_se:.1f}_{n_alphas_se:d}_pstar_{pstar:.1f}_reg_{reg_p:.1f}.csv"
file_name_bound_SE = f"SE_optimal_regp_bound_direct_alphas_{alpha_min_se:.1f}_{alpha_max_se:.1f}_{n_alphas_se:d}_pstar_{pstar:.1f}_reg_{reg_p:.1f}.csv"

if not os.path.exists(data_folder):
    os.makedirs(data_folder)


def perform_sweep(metric_name, file_name_SE_template, file_name_output):
    """
    Perform a parameter sweep for a specific metric.

    Args:
        metric_name: Name of the metric (for display purposes)
        file_name_SE_template: Template for the SE file name
        file_name_output: Output file name
        eps_t: Epsilon parameter for the solver (default 0.0)
    """
    print(os.path.join(data_folder, file_name_SE_template))
    if os.path.exists(os.path.join(data_folder, file_name_SE_template)):
        print(f"SE file {file_name_SE_template} exists.")
    else:
        print(f"SE file {file_name_SE_template} does not exist.")
        return {}

    SE_data = np.loadtxt(
        os.path.join(data_folder, file_name_SE_template),
        delimiter=",",
        skiprows=1,
    )

    alphas_SE = SE_data[:, 0]
    alpha_list = np.linspace(alpha_min_erm, alpha_max_erm, n_alphas_erm)
    indices = np.searchsorted(alphas_SE, alpha_list)
    alpha_list = SE_data[indices, 0]
    reg_param_list = SE_data[indices, -1]

    # Initialize arrays to hold results
    ms = np.empty((n_alphas_erm, 2))
    qs = np.empty((n_alphas_erm, 2))
    Ps = np.empty((n_alphas_erm, 2))
    gen_errs = np.empty((n_alphas_erm, 2))
    adv_errs = np.empty((n_alphas_erm, 2))
    flipped_fairs = np.empty((n_alphas_erm, 2))
    misclas_fairs = np.empty((n_alphas_erm, 2))
    bound_errs = np.empty((n_alphas_erm, 2))

    if pstar == 2.0:
        eps_test_tilde = eps_test
    elif pstar == 1.0:
        eps_test_tilde = eps_test / np.sqrt(d)

    # Loop through alpha values
    for i, (alpha, reg_param) in enumerate(zip(alpha_list, reg_param_list)):
        print(f"Calculating alpha: {alpha:.2f} / {alpha_max_erm:.2f}")
        n = int(alpha * d)

        m_vals = []
        q_vals = []
        P_vals = []
        gen_err_vals = []
        adv_err_vals = []
        flip_fair_vals = []
        misc_fair_vals = []
        bound_err_vals = []

        j = 0
        while j < reps:
            xs, ys, xs_gen, ys_gen, wstar = data_generation(
                measure_gen_probit_clasif, d, n, n_gen, (delta,)
            )

            try:
                if pstar == 2.0:
                    w = find_coefficients_Logistic_adv_Linf_L2(ys, xs, reg_param, 0.0)
                else:
                    w = find_coefficients_Logistic_adv_Linf_L1(ys, xs, reg_param, 0.0)
            except (ValueError, SolverError) as e:
                print(
                    f"minimization didn't converge on iteration {j} for alpha {alpha:.2f}. Trying again."
                )
                continue

            m_vals.append(np.dot(wstar, w) / d)
            q_vals.append(np.dot(w, w) / d)
            P_vals.append(np.sum(np.abs(w) ** pstar) / d)

            yhat_gen = np.sign(np.dot(xs_gen, w))

            gen_err_vals.append(generalisation_error_classification(ys_gen, xs_gen, w, wstar))

            adv_perturbation = find_adversarial_perturbation_direct_space_noteacher(
                ys_gen, xs_gen, w, wstar, eps_test_tilde, adv_geometry
            )
            adv_err = np.mean(ys_gen != np.sign((xs_gen + adv_perturbation) @ w))
            adv_err_vals.append(adv_err)

            adv_perturbation = find_adversarial_perturbation_direct_space(
                yhat_gen, xs_gen, w, wstar, eps_test_tilde, adv_geometry
            )
            flipped = np.mean(yhat_gen != np.sign((xs_gen + adv_perturbation) @ w))
            flip_fair_vals.append(flipped)

            adv_perturbation = find_adversarial_perturbation_direct_space(
                ys_gen, xs_gen, w, wstar, eps_test_tilde, adv_geometry
            )
            misclass = np.mean(ys_gen != np.sign((xs_gen + adv_perturbation) @ w))
            misc_fair_vals.append(misclass)

            adv_perturbation = find_adversarial_perturbation_direct_space(
                ys_gen, xs_gen, w, wstar, eps_test_tilde, adv_geometry
            )
            bound_err = np.mean(
                (ys_gen != np.sign((xs_gen + adv_perturbation) @ w)) * (ys_gen != yhat_gen)
            )
            bound_err_vals.append(bound_err)

            print(f"repetition {j} for alpha {alpha:.2f} done.")
            j += 1

        ms[i, 0], ms[i, 1] = np.mean(m_vals), np.std(m_vals)
        qs[i, 0], qs[i, 1] = np.mean(q_vals), np.std(q_vals)
        Ps[i, 0], Ps[i, 1] = np.mean(P_vals), np.std(P_vals)
        gen_errs[i, 0], gen_errs[i, 1] = np.mean(gen_err_vals), np.std(gen_err_vals)
        adv_errs[i, 0], adv_errs[i, 1] = np.mean(adv_err_vals), np.std(adv_err_vals)
        flipped_fairs[i, 0], flipped_fairs[i, 1] = np.mean(flip_fair_vals), np.std(flip_fair_vals)
        misclas_fairs[i, 0], misclas_fairs[i, 1] = np.mean(misc_fair_vals), np.std(misc_fair_vals)
        bound_errs[i, 0], bound_errs[i, 1] = np.mean(bound_err_vals), np.std(bound_err_vals)

        print(f"alpha {alpha:.2f} done.")

    columns = (
        alpha_list,
        ms[:, 0],
        ms[:, 1],
        qs[:, 0],
        qs[:, 1],
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
        "alpha,m_mean,m_std,q_mean,q_std,P_mean,P_std,"
        "gen_err_mean,gen_err_std,"
        "adv_err_mean,adv_err_std,"
        "flipped_fair_mean,flipped_fair_std,"
        "misclass_fair_mean,misclass_fair_std,"
        "bound_err_mean,bound_err_std,"
        "reg_param"
    )

    results = {"alpha_list": alpha_list, "gen_errs": gen_errs, "adv_errs": adv_errs}
    np.savetxt(
        os.path.join(data_folder, file_name_output),
        np.column_stack(columns),
        delimiter=",",
        header=header,
    )

    print(f"Data saved for {metric_name}.")
    return results


if metric_name_chosen == "misclass":
    print("Performing sweep for misclassification error.")
    misclass_results = perform_sweep("misclass", file_name_misclass_SE, file_name_misclass)
elif metric_name_chosen == "bound":
    print("Performing sweep for boundary error.")
    bound_results = perform_sweep("bound", file_name_bound_SE, file_name_bound)
elif metric_name_chosen == "adverr":
    print("Performing sweep for adversarial error.")
    adverr_results = perform_sweep("adversarial error", file_name_adverr_SE, file_name_adverr)
