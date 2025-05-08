import numpy as np
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.aux_functions.misc import (
    classification_adversarial_error,
    misclassification_error_direct_space,
    flipped_error_direct_space,
)
from linear_regression.fixed_point_equations.regularisation.pstar_attacks_Lr_reg import (
    f_Lr_regularisation_Lpstar_attack,
)
from linear_regression.fixed_point_equations.classification.Adv_train_p_norm import (
    f_hat_Logistic_no_noise_Linf_adv_classif,
)
from linear_regression.aux_functions.percentage_flipped import (
    percentage_flipped_hastie_model,
    percentage_misclassified_hastie_model,
)
from os.path import join, exists
import os
import sys
from scipy.optimize import minimize_scalar

if len(sys.argv) > 1:
    alpha_min, alpha_max, n_alphas = (
        float(sys.argv[1]),
        float(sys.argv[2]),
        int(sys.argv[3]),
        float(sys.argv[4]),
    )
else:
    alpha_min, alpha_max, n_alphas = (0.5, 2.0, 100)

pstar = 1.0
reg_p = 2.0
eps_test = 1.0


def compute_theory_overlaps(reg_param, alpha, init_cond):
    f_kwargs = {"reg_param": reg_param, "reg_order": reg_p, "pstar": pstar}
    f_hat_kwargs = {"alpha": alpha, "eps_t": 0.0}

    m_se, q_se, V_se, P_se = fixed_point_finder(
        f_Lr_regularisation_Lpstar_attack,
        f_hat_Logistic_no_noise_Linf_adv_classif,
        init_cond,
        f_kwargs,
        f_hat_kwargs,
        abs_tol=1e-6,
    )

    return m_se, q_se, V_se, P_se


def fun_to_min(reg_param, alpha, init_cond, error_metric="misclass"):
    f_kwargs = {"reg_param": reg_param, "reg_order": reg_p, "pstar": pstar}
    f_hat_kwargs = {"alpha": alpha, "eps_t": 0.0}

    m_se, q_se, V_se, P_se = fixed_point_finder(
        f_Lr_regularisation_Lpstar_attack,
        f_hat_Logistic_no_noise_Linf_adv_classif,
        init_cond,
        f_kwargs,
        f_hat_kwargs,
        abs_tol=1e-5,
    )

    if error_metric == "adv":
        return classification_adversarial_error(m_se, q_se, P_se, eps_test, pstar)
    elif error_metric == "misclass":
        return misclassification_error_direct_space(m_se, q_se, P_se, eps_test, pstar)
    else:
        return flipped_error_direct_space(m_se, q_se, P_se, eps_test, pstar)


data_folder = "./data/direct_space_model_training_optimal"

file_name_misclass = f"SE_optimal_regp_misclass_direct_alphas_{alpha_min:.1f}_{alpha_max:.1f}_{n_alphas:d}_pstar_{pstar:.1f}_reg_{reg_p:.1f}.csv"
file_name_flipped = f"SE_optimal_regp_flipped_direct_alphas_{alpha_min:.1f}_{alpha_max:.1f}_{n_alphas:d}_pstar_{pstar:.1f}_reg_{reg_p:.1f}.csv"
file_name_adverr = f"SE_optimal_regp_adverr_direct_alphas_{alpha_min:.1f}_{alpha_max:.1f}_{n_alphas:d}_pstar_{pstar:.1f}_reg_{reg_p:.1f}.csv"

if not exists(data_folder):
    os.makedirs(data_folder)


def perform_sweep(error_metric_type, output_file):
    """
    Performs an alpha sweep with optimization for regularization parameter.

    Parameters:
    - error_metric_type: String, metric used for optimization ('misclass', 'flipped', or 'adv')
    - output_file: String, file path to save results
    """
    alphas = np.linspace(alpha_min, alpha_max, n_alphas)

    # Initialize arrays
    ms_found = np.empty((n_alphas,))
    qs_found = np.empty((n_alphas,))
    qs_latent_found = np.empty((n_alphas,))
    qs_features_found = np.empty((n_alphas,))
    Vs_found = np.empty((n_alphas,))
    Ps_found = np.empty((n_alphas,))
    estim_errors_se = np.empty((n_alphas,))
    adversarial_errors_found = np.empty((n_alphas,))
    gen_errors_se = np.empty((n_alphas,))
    flipped_fairs_se = np.empty((n_alphas,))
    misclas_fairs_se = np.empty((n_alphas,))
    reg_param_found = np.empty((n_alphas,))

    initial_condition = (0.6, 1.6, 1.05, 1.1)

    for j, alpha in enumerate(alphas):
        print(f"Calculating alpha: {alpha:.2f} / {alpha_max:.2f}")

        # Optimize regularization parameter
        if j == 0:
            res = minimize_scalar(
                fun_to_min,
                args=(alpha, initial_condition, error_metric_type),
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
                args=(alpha, initial_condition, error_metric_type),
                bounds=(lower, upper),
                method="bounded",
            )
            reg_param = res.x
            reg_param = max(1e-5, min(1e1, reg_param))

        reg_param_found[j] = reg_param

        ms_found[j], qs_found[j], Vs_found[j], Ps_found[j] = compute_theory_overlaps(
            reg_param, alpha, initial_condition
        )

        initial_condition = (ms_found[j], qs_found[j], Vs_found[j], Ps_found[j])

        estim_errors_se[j] = 1 - 2 * ms_found[j] + qs_found[j]

        # Calculate adversarial error using the latent version
        adversarial_errors_found[j] = classification_adversarial_error(
            ms_found[j], qs_found[j], Ps_found[j], eps_test, pstar
        )

        gen_errors_se[j] = np.arccos(ms_found[j] / np.sqrt(qs_found[j])) / np.pi

        flipped_fairs_se[j] = flipped_error_direct_space(
            ms_found[j], qs_found[j], Ps_found[j], eps_test, pstar
        )
        misclas_fairs_se[j] = misclassification_error_direct_space(
            ms_found[j], qs_found[j], Ps_found[j], eps_test, pstar
        )

    # Save results to file
    data = {
        "alpha": alphas,
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


# Call the function three times with appropriate parameters
perform_sweep("misclass", file_name_misclass)  # For misclassification error
perform_sweep("flipped", file_name_flipped)  # For flipped error
perform_sweep("adv", file_name_adverr)  # For adversarial error
