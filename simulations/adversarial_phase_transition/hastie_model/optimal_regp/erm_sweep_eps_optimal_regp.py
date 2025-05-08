import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfc
from linear_regression.data.generation import (
    measure_gen_no_noise_clasif,
    data_generation,
    data_generation_hastie,
)
from linear_regression.aux_functions.percentage_flipped import percentage_misclassified_hastie_model
from linear_regression.aux_functions.misc import classification_adversarial_error
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_linear_rf,
    find_adversarial_error_rf,
    find_adversarial_perturbation_direct_space_noteacher,
    find_adversarial_perturbation_direct_space,
)
from linear_regression.data.generation import data_generation_hastie, measure_gen_probit_clasif
from linear_regression.erm.metrics import (
    generalisation_error_classification,
    adversarial_error_data,
    percentage_flipped_labels_estim,
    percentage_error_from_true,
    percentage_flipped_labels_estim,
    percentage_error_from_true,
    adversarial_error_data,
)
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic,
    find_coefficients_Logistic_adv,
    find_coefficients_Logistic_adv_Linf_L2,
)
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_linear_rf,
    find_adversarial_error_rf,
)
from tqdm.auto import tqdm
import os
import sys
from cvxpy.error import SolverError

if len(sys.argv) > 1:
    eps_min, eps_max, n_epss, alpha, gamma, eps_training = (
        float(sys.argv[1]),
        float(sys.argv[2]),
        int(sys.argv[3]),
        float(sys.argv[4]),
        float(sys.argv[5]),
        float(sys.argv[6]),
    )
else:
    eps_min, eps_max, n_epss, alpha, gamma, eps_training = (0.1, 10.0, 15, 1.5, 0.5, 0.0)

pstar = 1.0
reg = 2.0

d = 500
reps = 10
n_gen = 1000

data_folder = "./data/hastie_model_training"

file_name_misclass_ERM = f"ERM_optimal_regp_misclass_Linf_d_{d:d}_alpha_{alpha:.1f}_gamma_{gamma:.1f}_reps_{reps:d}_epss_{eps_min:.1f}_{eps_max:.1f}_{n_epss:d}_pstar_{pstar:.1f}_reg_{reg:.1f}.csv"
file_name_flipped_ERM = f"ERM_optimal_regp_flipped_Linf_d_{d:d}_alpha_{alpha:.1f}_gamma_{gamma:.1f}_reps_{reps:d}_epss_{eps_min:.1f}_{eps_max:.1f}_{n_epss:d}_pstar_{pstar:.1f}_reg_{reg:.1f}.csv"
file_name_adverr_ERM = f"ERM_optimal_regp_adverr_Linf_d_{d:d}_alpha_{alpha:.1f}_gamma_{gamma:.1f}_reps_{reps:d}_epss_{eps_min:.1f}_{eps_max:.1f}_{n_epss:d}_pstar_{pstar:.1f}_reg_{reg:.1f}.csv"

file_name_misclass_SE = f"SE_optimal_regp_misclass_Linf_alpha_{alpha:.1f}_gamma_{gamma:.1f}_epss_{eps_min:.1f}_{eps_max:.1f}_{n_epss:d}_pstar_{pstar:.1f}_reg_{reg:.1f}.csv"
file_name_flipped_SE = f"SE_optimal_regp_flipped_Linf_alpha_{alpha:.1f}_gamma_{gamma:.1f}_epss_{eps_min:.1f}_{eps_max:.1f}_{n_epss:d}_pstar_{pstar:.1f}_reg_{reg:.1f}.csv"
file_name_adverr_SE = f"SE_optimal_regp_adverr_Linf_alpha_{alpha:.1f}_gamma_{gamma:.1f}_epss_{eps_min:.1f}_{eps_max:.1f}_{n_epss:d}_pstar_{pstar:.1f}_reg_{reg:.1f}.csv"

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

    epss_SE = SE_data[:, 0]
    epss = np.logspace(np.log10(eps_min), np.log10(eps_max), n_epss)
    indices = np.searchsorted(epss_SE, epss)
    epss_list = SE_data[indices, 0]
    reg_param_list = SE_data[indices, -1]

    ms = np.empty((n_epss, 2))
    qs = np.empty((n_epss, 2))
    Ps = np.empty((n_epss, 2))
    gen_errors = np.empty((n_epss, 2))
    adversarial_errors = np.empty((n_epss, 2))
    flipped_fairs = np.empty((n_epss, 2))
    misclas_fairs = np.empty((n_epss, 2))

    n = int(d * alpha)

    for i, eps in enumerate(epss_list):
        m_vals = np.empty((reps,))
        q_vals = np.empty((reps,))
        P_vals = np.empty((reps,))
        gen_errors_vals = np.empty((reps,))
        adversarial_errors_vals = np.empty((reps,))
        flipped_fairs_vals = np.empty((reps,))
        misclas_fairs_vals = np.empty((reps,))

        j = 0
        while j < reps:
            xs, ys, xs_gen, ys_gen, wstar = data_generation(
                measure_gen_no_noise_clasif, d, n, n_gen, ()
            )

            try:
                w = find_coefficients_Logistic(ys, xs, reg_param_list[i])
            except (ValueError, UserWarning, SolverError) as e:
                print("Error in finding coefficients:", e)
                continue

            m_vals[j] = np.dot(wstar, w) / d
            q_vals[j] = np.dot(w, w) / d
            P_vals[j] = np.mean(np.abs(w))
            gen_errors_vals[j] = generalisation_error_classification(ys_gen, xs_gen, w, wstar)
            adversarial_errors_vals[j] = classification_adversarial_error(
                m_vals[j], q_vals[j], P_vals[j], eps, pstar
            )

            yhats_gen = np.dot(xs_gen, w)

            try:
                adv_pert = find_adversarial_perturbation_direct_space(
                    yhats_gen, xs_gen, w, wstar, eps, "inf"
                )
            except (ValueError, UserWarning, SolverError) as e:
                print("Error in finding adversarial perturbation:", e)
                continue

            xs_pert_gen = xs_gen + adv_pert
            flipped = np.mean(yhats_gen != np.sign(np.dot(xs_pert_gen, w)))
            flipped_fairs_vals[j] = flipped

            try:
                adv_pert = find_adversarial_perturbation_direct_space(
                    ys_gen, xs_gen, w, wstar, eps, "inf"
                )
            except (ValueError, UserWarning, SolverError) as e:
                print("Error in finding adversarial perturbation:", e)
                continue

            xs_pert_gen = xs_gen + adv_pert
            misclass = np.mean(ys_gen != np.sign(np.dot(xs_pert_gen, w)))
            misclas_fairs_vals[j] = misclass

            j += 1

        ms[i, 0], ms[i, 1] = np.mean(m_vals), np.std(m_vals)
        qs[i, 0], qs[i, 1] = np.mean(q_vals), np.std(q_vals)
        Ps[i, 0], Ps[i, 1] = np.mean(P_vals), np.std(P_vals)
        gen_errors[i, 0], gen_errors[i, 1] = np.mean(gen_errors_vals), np.std(gen_errors_vals)
        adversarial_errors[i, 0], adversarial_errors[i, 1] = np.mean(
            adversarial_errors_vals
        ), np.std(adversarial_errors_vals)
        flipped_fairs[i, 0], flipped_fairs[i, 1] = np.mean(flipped_fairs_vals), np.std(
            flipped_fairs_vals
        )
        misclas_fairs[i, 0], misclas_fairs[i, 1] = np.mean(misclas_fairs_vals), np.std(
            misclas_fairs_vals
        )

    # Save results to file
    data = {
        "eps": epss,
        "m_mean": ms[:, 0],
        "m_std": ms[:, 1],
        "q_mean": qs[:, 0],
        "q_std": qs[:, 1],
        "P_mean": Ps[:, 0],
        "P_std": Ps[:, 1],
        "gen_error_mean": gen_errors[:, 0],
        "gen_error_std": gen_errors[:, 1],
        "adversarial_error_mean": adversarial_errors[:, 0],
        "adversarial_error_std": adversarial_errors[:, 1],
        "flipped_fair_mean": flipped_fairs[:, 0],
        "flipped_fair_std": flipped_fairs[:, 1],
        "misclass_fair_mean": misclas_fairs[:, 0],
        "misclass_fair_std": misclas_fairs[:, 1],
    }
    data_array = np.column_stack([data[key] for key in data.keys()])
    header = ",".join(data.keys())
    np.savetxt(
        os.path.join(data_folder, file_name_output),
        data_array,
        delimiter=",",
        header=header,
    )


# Perform the sweep for the specified metric
perform_sweep("misclass", file_name_misclass_SE, file_name_misclass_ERM)
perform_sweep("flipped", file_name_flipped_SE, file_name_flipped_ERM)
perform_sweep("adverr", file_name_adverr_SE, file_name_adverr_ERM)
