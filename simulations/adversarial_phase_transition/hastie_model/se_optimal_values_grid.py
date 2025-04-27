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
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic,
    find_coefficients_Logistic_adv,
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

alpha_min, alpha_max, n_alphas = 0.1, 1.5, 15
gamma_min, gamma_max, n_gammas = 0.1, 1.5, 15


def fun_to_min(reg_param, alpha, gamma):
    init_cond = (0.1, 1.0, 1.0, 1.0)

    f_kwargs = {"reg_param": reg_param, "gamma": gamma}
    f_hat_kwargs = {"alpha": alpha, "gamma": gamma, "ε": 0.0}

    # print(f_kwargs, f_hat_kwargs)

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

    if gamma <= 1:
        return (
            np.sqrt(q_latent_se - m_se**2 / gamma)
            * np.sqrt(1 / np.pi)
            / np.sqrt(q_se)
            * np.sqrt(gamma)
        )
    else:
        return (
            np.sqrt(q_features_se - m_se**2 / gamma)
            / np.sqrt(gamma)
            * np.sqrt(1 / np.pi)
            / np.sqrt(q_se)
        )


def fun_to_min_2(x, alpha, gamma):
    reg_param, eps_training = x
    init_cond = (0.1, 1.0, 1.0, 1.0)

    f_kwargs = {"reg_param": reg_param, "gamma": gamma}
    f_hat_kwargs = {"alpha": alpha, "gamma": gamma, "ε": eps_training}

    # print(f_kwargs, f_hat_kwargs)

    m_se, q_se, V_se, P_se = fixed_point_finder(
        f_hastie_L2_reg_Linf_attack,
        f_hat_Logistic_no_noise_Linf_adv_classif,
        init_cond,
        f_kwargs,
        f_hat_kwargs,
        abs_tol=1e-6,
    )

    m_hat, q_hat, V_hat, P_hat = f_hat_Logistic_no_noise_Linf_adv_classif(
        m_se, q_se, V_se, P_se, eps_training, alpha, gamma
    )

    q_latent_se = q_latent_hastie_L2_reg_Linf_attack(m_hat, q_hat, V_hat, P_hat, reg_param, gamma)
    q_features_se = q_features_hastie_L2_reg_Linf_attack(
        m_hat, q_hat, V_hat, P_hat, reg_param, gamma
    )

    if gamma <= 1:
        return (
            np.sqrt(q_latent_se - m_se**2 / gamma)
            * np.sqrt(1 / np.pi)
            / np.sqrt(q_se)
            * np.sqrt(gamma)
        )
    else:
        return (
            np.sqrt(q_features_se - m_se**2 / gamma)
            / np.sqrt(gamma)
            * np.sqrt(1 / np.pi)
            / np.sqrt(q_se)
        )


alphas = np.linspace(alpha_min, alpha_max, n_alphas)
gammas = np.linspace(gamma_min, gamma_max, n_gammas)

optimal_reg_param_noadv = np.zeros((n_alphas, n_gammas))
optimal_reg_param_adv = np.zeros((n_alphas, n_gammas))
optimal_eps_training = np.zeros((n_alphas, n_gammas))
flipped_percentages_noadv = np.zeros((n_alphas, n_gammas))
flipped_percentages_adv = np.zeros((n_alphas, n_gammas))

old_reg_param = 0.1
old_eps_training = 0.1

for i, alpha in enumerate(alphas):
    for j, gamma in enumerate(gammas):
        print(f"alpha: {alpha}, gamma: {gamma}")
        res = minimize_scalar(
            fun_to_min,
            args=(alpha, gamma),
            bounds=(1e-6, 1e1),
            method="bounded",
            options={"xatol": 1e-8, "disp": False},
        )
        reg_param_opt = res.x
        print("Optimal reg_param:", reg_param_opt, "Optimal flipped percentage:", res.fun)

        optimal_reg_param_noadv[i, j] = reg_param_opt
        flipped_percentages_noadv[i, j] = res.fun

        res = minimize(
            fun_to_min_2,
            # (old_reg_param, old_eps_training),
            (np.random.uniform(1e-6, 1e1), np.random.uniform(1.0, 1e1)),
            bounds=((1e-6, 1e1), (0.0, 1e1)),
            method="Nelder-Mead",
            args=(alpha, gamma),
            options={"xatol": 1e-8, "disp": False},
        )
        reg_param_opt, eps_training_opt = res.x
        print(
            "Optimal reg_param:",
            reg_param_opt,
            "Optimal eps_training:",
            eps_training_opt,
            "Optimal flipped percentage:",
            res.fun,
        )

        optimal_reg_param_adv[i, j] = reg_param_opt
        optimal_eps_training[i, j] = eps_training_opt
        flipped_percentages_adv[i, j] = res.fun

ALPHAS, GAMMAS = np.meshgrid(alphas, gammas)
# create three subplots and show with controurf and contour the difference of the values
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
im0 = axs[0].contourf(
    ALPHAS,
    GAMMAS,
    flipped_percentages_noadv.T - flipped_percentages_adv.T,
    levels=np.linspace(0, 1, 20),
    cmap="viridis",
)
fig.colorbar(im0, ax=axs[0], label="Difference")

axs[0].set_title("Flipped percentage difference")
axs[0].set_xlabel("alpha")
axs[0].set_ylabel("gamma")
axs[0].set_xscale("log")
axs[0].set_yscale("log")

im1 = axs[1].contourf(
    ALPHAS,
    GAMMAS,
    optimal_reg_param_noadv.T - optimal_reg_param_adv.T,
    levels=np.linspace(-1, 1, 20),
    cmap="viridis",
)
fig.colorbar(im1, ax=axs[1], label="Difference")

axs[1].set_title("Optimal reg_param difference")
axs[1].set_xlabel("alpha")
axs[1].set_ylabel("gamma")
axs[1].set_xscale("log")
axs[1].set_yscale("log")

im2 = axs[2].contourf(
    ALPHAS,
    GAMMAS,
    optimal_eps_training.T,
    levels=np.linspace(0, 3, 20),
    cmap="viridis",
)
fig.colorbar(im2, ax=axs[2], label="Value")

axs[2].set_title("Optimal eps_training")
axs[2].set_xlabel("alpha")
axs[2].set_ylabel("gamma")
axs[2].set_xscale("log")
axs[2].set_yscale("log")

plt.tight_layout()

plt.show()

# Save the results
data_folder = "./data/hastie_model_optimal_values/"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
file_name = "optimal_values_alpha_{:.2f}_gamma_{:.2f}.pkl"
with open(os.path.join(data_folder, file_name.format(alpha_max, gamma_max)), "wb") as f:
    pickle.dump(
        {
            "optimal_reg_param_noadv": optimal_reg_param_noadv,
            "optimal_reg_param_adv": optimal_reg_param_adv,
            "optimal_eps_training": optimal_eps_training,
            "flipped_percentages_noadv": flipped_percentages_noadv,
            "flipped_percentages_adv": flipped_percentages_adv,
        },
        f,
    )
print("Results saved to", os.path.join(data_folder, file_name.format(alpha_max, gamma_max)))
