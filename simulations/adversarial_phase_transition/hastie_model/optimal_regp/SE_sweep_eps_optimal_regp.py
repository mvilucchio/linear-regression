import numpy as np
import matplotlib.pyplot as plt
from linear_regression.aux_functions.percentage_flipped import (
    percentage_misclassified_hastie_model,
    percentage_flipped_hastie_model,
)
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.hastie_model_pstar_attacks import (
    f_hastie_L2_reg_Linf_attack,
    q_latent_hastie_L2_reg_Linf_attack,
    q_features_hastie_L2_reg_Linf_attack,
)
from linear_regression.aux_functions.misc import classification_adversarial_error_latent
from linear_regression.fixed_point_equations.classification.Adv_train_p_norm_hastie import (
    f_hat_Logistic_no_noise_Linf_adv_classif,
)
from scipy.optimize import minimize_scalar
from tqdm.auto import tqdm
from os.path import join
import os
import sys
from cvxpy.error import SolverError
import pickle

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

# DO NOT CHANGE
pstar = 1.0
reg = 2.0


def compute_theory_overlaps(reg_param, alpha, gamma, init_cond):
    f_kwargs = {"reg_param": reg_param, "gamma": gamma}
    f_hat_kwargs = {"alpha": alpha, "gamma": gamma, "ε": 0.0}

    m_se, q_se, V_se, P_se = fixed_point_finder(
        f_hastie_L2_reg_Linf_attack,
        f_hat_Logistic_no_noise_Linf_adv_classif,
        init_cond,
        f_kwargs,
        f_hat_kwargs,
        abs_tol=1e-6,
    )

    m_hat, q_hat, V_hat, P_hat = f_hat_Logistic_no_noise_Linf_adv_classif(
        m_se, q_se, V_se, P_se, 0.0, alpha, gamma
    )

    q_latent_se = q_latent_hastie_L2_reg_Linf_attack(m_hat, q_hat, V_hat, P_hat, reg_param, gamma)
    q_features_se = q_features_hastie_L2_reg_Linf_attack(
        m_hat, q_hat, V_hat, P_hat, reg_param, gamma
    )

    return m_se, q_se, q_latent_se, q_features_se, V_se, P_se


def fun_to_min(reg_param, alpha, gamma, init_cond, eps_test, error_metric="misclass"):
    f_kwargs = {"reg_param": reg_param, "gamma": gamma}
    f_hat_kwargs = {"alpha": alpha, "gamma": gamma, "ε": 0.0}

    m_se, q_se, V_se, P_se = fixed_point_finder(
        f_hastie_L2_reg_Linf_attack,
        f_hat_Logistic_no_noise_Linf_adv_classif,
        init_cond,
        f_kwargs,
        f_hat_kwargs,
        abs_tol=1e-5,
    )

    m_hat, q_hat, V_hat, P_hat = f_hat_Logistic_no_noise_Linf_adv_classif(
        m_se, q_se, V_se, P_se, 0.0, alpha, gamma
    )

    q_latent_se = q_latent_hastie_L2_reg_Linf_attack(m_hat, q_hat, V_hat, P_hat, reg_param, gamma)
    q_features_se = q_features_hastie_L2_reg_Linf_attack(
        m_hat, q_hat, V_hat, P_hat, reg_param, gamma
    )
    if error_metric == "adv":
        return classification_adversarial_error_latent(
            m_se, q_se, q_features_se, q_latent_se, 1.0, P_se, eps_test, gamma, pstar
        )
    elif error_metric == "misclass":
        return percentage_misclassified_hastie_model(
            m_se, q_se, q_latent_se, q_features_se, 1.0, eps_test, gamma, "inf"
        )
    else:
        return percentage_flipped_hastie_model(
            m_se, q_se, q_latent_se, q_features_se, 1.0, eps_test, gamma, "inf"
        )


data_folder = "./data/hastie_model_training"
file_name_misclass = f"SE_optimal_regp_misclass_Linf_alpha_{alpha:.1f}_gamma_{gamma:.1f}_epss_{eps_min:.1f}_{eps_max:.1f}_{n_epss:d}_pstar_{pstar:.1f}_reg_{reg:.1f}.csv"
file_name_flipped = f"SE_optimal_regp_flipped_Linf_alpha_{alpha:.1f}_gamma_{gamma:.1f}_epss_{eps_min:.1f}_{eps_max:.1f}_{n_epss:d}_pstar_{pstar:.1f}_reg_{reg:.1f}.csv"
file_name_adverr = f"SE_optimal_regp_adverr_Linf_alpha_{alpha:.1f}_gamma_{gamma:.1f}_epss_{eps_min:.1f}_{eps_max:.1f}_{n_epss:d}_pstar_{pstar:.1f}_reg_{reg:.1f}.csv"

if not os.path.exists(data_folder):
    os.makedirs(data_folder)


def perform_sweep(error_metric_type, output_file):
    """
    Performs an alpha sweep with optimization for regularization parameter.

    Parameters:
    - error_metric_type: String, metric used for optimization ('misclass', 'flipped', or 'adv')
    - output_file: String, file path to save results
    """
    epss = np.logspace(np.log10(eps_min), np.log10(eps_max), n_epss)

    # Initialize arrays
    ms_found = np.empty((n_epss,))
    qs_found = np.empty((n_epss,))
    qs_latent_found = np.empty((n_epss,))
    qs_features_found = np.empty((n_epss,))
    Vs_found = np.empty((n_epss,))
    Ps_found = np.empty((n_epss,))
    estim_errors_se = np.empty((n_epss,))
    adversarial_errors_found = np.empty((n_epss,))
    gen_errors_se = np.empty((n_epss,))
    flipped_fairs_se = np.empty((n_epss,))
    misclas_fairs_se = np.empty((n_epss,))
    reg_param_found = np.empty((n_epss,))

    initial_condition = (0.6, 1.6, 1.05, 1.1)

    for j, eps in enumerate(epss):
        print(f"Calculating epss: {eps:.2f} / {eps_max:.2f}")

        if j == 0:
            res = minimize_scalar(
                fun_to_min,
                args=(alpha, gamma, initial_condition, eps, error_metric_type),
                bounds=(1e-5, 1e1),
                method="bounded",
            )
            reg_param = res.x
        else:
            prev_param = reg_param_found[j - 1]
            lower = max(1e-5, prev_param * 0.5)
            upper = min(1e1, prev_param * 2.0)

            res = minimize_scalar(
                fun_to_min,
                args=(alpha, gamma, initial_condition, eps, error_metric_type),
                bounds=(lower, upper),
                method="bounded",
            )
            reg_param = res.x
            reg_param = max(1e-5, min(1e1, reg_param))

        reg_param_found[j] = reg_param

        (
            ms_found[j],
            qs_found[j],
            qs_latent_found[j],
            qs_features_found[j],
            Vs_found[j],
            Ps_found[j],
        ) = compute_theory_overlaps(reg_param, alpha, gamma, initial_condition)

        initial_condition = (ms_found[j], qs_found[j], Vs_found[j], Ps_found[j])

        estim_errors_se[j] = 1 - 2 * ms_found[j] + qs_found[j]

        adversarial_errors_found[j] = classification_adversarial_error_latent(
            ms_found[j],
            qs_found[j],
            qs_features_found[j],
            qs_latent_found[j],
            1.0,
            Ps_found[j],
            eps,
            gamma,
            pstar,
        )

        gen_errors_se[j] = np.arccos(ms_found[j] / np.sqrt(qs_found[j])) / np.pi

        flipped_fairs_se[j] = percentage_flipped_hastie_model(
            ms_found[j],
            qs_found[j],
            qs_latent_found[j],
            qs_features_found[j],
            1.0,
            eps,
            gamma,
            "inf",
        )
        misclas_fairs_se[j] = percentage_misclassified_hastie_model(
            ms_found[j],
            qs_found[j],
            qs_latent_found[j],
            qs_features_found[j],
            1.0,
            eps,
            gamma,
            "inf",
        )

    # Save results to file
    data = {
        "eps": epss,
        "m": ms_found,
        "q": qs_found,
        "q_latent": qs_latent_found,
        "q_features": qs_features_found,
        "V": Vs_found,
        "P": Ps_found,
        "estim_errors_found": estim_errors_se,
        "adversarial_errors_found": adversarial_errors_found,
        "generalisation_errors_found": gen_errors_se,
        "flipped_fairs_found": flipped_fairs_se,
        "misclas_fairs_found": misclas_fairs_se,
        "reg_param_found": reg_param_found,
    }

    data_array = np.column_stack([data[key] for key in data.keys()])
    header = ",".join(data.keys())
    np.savetxt(
        join(data_folder, output_file),
        data_array,
        delimiter=",",
        header=header,
        comments="",
    )


perform_sweep("misclass", file_name_misclass)  # For misclassification error
perform_sweep("flipped", file_name_flipped)  # For flipped error
perform_sweep("adv", file_name_adverr)  # For adversarial error
