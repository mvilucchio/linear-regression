import numpy as np
import matplotlib.pyplot as plt
from linear_regression.data.generation import (
    measure_gen_no_noise_clasif,
    data_generation,
    data_generation_hastie,
)
from linear_regression.erm.metrics import (
    percentage_flipped_labels_estim,
    percentage_error_from_true,
)
from linear_regression.aux_functions.misc import classification_adversarial_error
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic,
    find_coefficients_Logistic_adv,
)
from linear_regression.aux_functions.percentage_flipped import (
    percentage_flipped_hastie_model,
    percentage_misclassified_hastie_model,
)
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_linear_rf,
)
from scipy.optimize import minimize, minimize_scalar
from scipy.special import erf
from tqdm.auto import tqdm
import os
import sys
import warnings
import pickle
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.regularisation.hastie_model_pstar_attacks import (
    f_hastie_L2_reg_Linf_attack,
    q_latent_hastie_L2_reg_Linf_attack,
    q_features_hastie_L2_reg_Linf_attack,
)
from linear_regression.fixed_point_equations.classification.Adv_train_p_norm_hastie import (
    f_hat_Logistic_no_noise_Linf_adv_classif,
)


alpha_min, alpha_max, n_alphas = 0.1, 2.0, 30
gamma_min, gamma_max, n_gammas = 0.1, 2.0, 30
eps_test = 0.1


def fun_to_min(reg_param, alpha, gamma, init_cond):

    f_kwargs = {"reg_param": reg_param, "gamma": gamma}
    f_hat_kwargs = {"alpha": alpha, "gamma": gamma, "ε": 0.0}

    # print(f_kwargs, f_hat_kwargs)

    m_se, q_se, V_se, P_se = fixed_point_finder(
        f_hastie_L2_reg_Linf_attack,
        f_hat_Logistic_no_noise_Linf_adv_classif,
        init_cond,
        f_kwargs,
        f_hat_kwargs,
        abs_tol=1e-5,
    )

    # return classification_adversarial_error(m_se, q_se, P_se, eps_test, 1.0)

    m_hat, q_hat, V_hat, P_hat = f_hat_Logistic_no_noise_Linf_adv_classif(
        m_se, q_se, V_se, P_se, 0.0, alpha, gamma
    )

    q_latent_se = q_latent_hastie_L2_reg_Linf_attack(m_hat, q_hat, V_hat, P_hat, reg_param, gamma)
    q_features_se = q_features_hastie_L2_reg_Linf_attack(
        m_hat, q_hat, V_hat, P_hat, reg_param, gamma
    )

    # return percentage_flipped_hastie_model(
    #     m_se, q_se, q_latent_se, q_features_se, 1.0, eps_test, gamma, "inf"
    # )

    return percentage_misclassified_hastie_model(
        m_se, q_se, q_latent_se, q_features_se, 1.0, eps_test, gamma, "inf"
    )


alphas = np.linspace(alpha_min, alpha_max, n_alphas)
gammas = np.linspace(gamma_min, gamma_max, n_gammas)

optimal_reg_param_adv = np.zeros((n_alphas, n_gammas))
flipped_percentages_adv = np.zeros((n_alphas, n_gammas))

init_c = (0.1, 1.0, 1.0, 1.0)

for i, alpha in enumerate(alphas):
    for j, gamma in enumerate(gammas):
        res = minimize_scalar(
            fun_to_min,
            bounds=(1e-5, 1e1),
            method="bounded",
            options={"xatol": 1e-4, "disp": False},
            args=(alpha, gamma, init_c),
        )
        reg_param_opt = res.x

        old_reg_param = reg_param_opt

        f_kwargs = {"reg_param": reg_param_opt, "gamma": gamma}
        f_hat_kwargs = {"alpha": alpha, "gamma": gamma, "ε": 0.0}

        m_se, q_se, V_se, P_se = fixed_point_finder(
            f_hastie_L2_reg_Linf_attack,
            f_hat_Logistic_no_noise_Linf_adv_classif,
            init_c,
            f_kwargs,
            f_hat_kwargs,
            abs_tol=1e-5,
        )

        init_c = (m_se, q_se, V_se, P_se)

        # res = minimize(
        #     fun_to_min_2,
        #     # (old_reg_param, old_eps_training),
        #     (np.random.uniform(1e-6, 1e1), np.random.uniform(0.0, 0.5)),
        #     bounds=((1e-6, 1e0), (0.0, 0.5)),
        #     method="Nelder-Mead",
        #     args=(alpha, gamma),
        #     options={"xatol": 1e-4, "disp": False},
        # )
        # reg_param_opt, eps_training_opt = res.x
        print(f"alpha: {alpha:.2f}, gamma: {gamma:.2f}, reg_param: {reg_param_opt:.1e}")

        optimal_reg_param_adv[i, j] = reg_param_opt
        flipped_percentages_adv[i, j] = res.fun

ALPHAS, GAMMAS = np.meshgrid(alphas, gammas)
# create three subplots and show with controurf and contour the difference of the values
# fig, axs = plt.subplots(1, 2, figsize=(15, 5))
# im0 = axs[0].contourf(ALPHAS, GAMMAS, flipped_percentages_adv.T, cmap="viridis")
# fig.colorbar(im0, ax=axs[0], label="Value")

# axs[0].set_title("Flipped percentage difference")
# axs[0].set_xlabel("alpha")
# axs[0].set_ylabel("gamma")
# axs[0].set_xscale("log")
# axs[0].set_yscale("log")

# im1 = axs[1].contourf(
#     ALPHAS,
#     GAMMAS,
#     optimal_reg_param_adv.T,
#     cmap="viridis",
# )
# fig.colorbar(im1, ax=axs[1], label="Value")

# axs[1].set_title("Optimal reg_param difference")
# axs[1].set_xlabel("alpha")
# axs[1].set_ylabel("gamma")
# axs[1].set_xscale("log")
# axs[1].set_yscale("log")

# plt.tight_layout()

# plt.show()

# Save the results
data_folder = "./data/hastie_model_optimal_values/"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

file_name_just_reg = "PD_optimal_reg_p_alphas_{:.2f}_{:.2f}_{:d}_gammas_{:.2f}_{:.2f}_{:d}.csv"
np.savetxt(
    os.path.join(
        data_folder,
        file_name_just_reg.format(alpha_min, alpha_max, n_alphas, gamma_min, gamma_max, n_gammas),
    ),
    np.column_stack(
        (
            ALPHAS.flatten(),
            GAMMAS.flatten(),
            optimal_reg_param_adv.flatten(),
            flipped_percentages_adv.flatten(),
        )
    ),
    delimiter=",",
    header="alpha,gamma,optimal_reg_param,flipped_percentage",
)
