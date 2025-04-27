import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
from tqdm.auto import tqdm
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
from linear_regression.fixed_point_equations.classification.Adv_train_p_norm_hastie import (
    f_hat_Logistic_no_noise_Linf_adv_classif,
)

gammas = [2.0]
alphas = [1.5]
reps = 10
eps_training = 0.0
pstar_t = 1.0
p = "inf"
reg_param = 1e-2
eps_min, eps_max, n_epss = 0.1, 10, 15

data_folder = "./data/hastie_model_training"
file_name = f"ERM_flipped_Hastie_Linf_d_{{:d}}_alpha_{{alpha:.1f}}_gamma_{{gamma:.1f}}_reps_{reps:d}_epss_{{eps_min:.1f}}_{{eps_max:.1f}}_{{n_epss:d}}_reg_param_{reg_param:.1e}_eps_t_{eps_training:.2f}.pkl"

dimensions = [int(2**a) for a in range(11, 12)]

markers = [".", "x", "1", "2", "+", "3", "4"]

eps_dense = np.logspace(-1.5, 1.5, 50)
out = np.empty_like(eps_dense)

plt.figure(figsize=(13, 5))
for k, alpha in enumerate(alphas):

    plt.subplot(1, 3, k + 1)
    for idx, gamma in enumerate(gammas):

        for i, (d, mks) in enumerate(zip(dimensions, markers)):
            with open(
                os.path.join(
                    data_folder,
                    file_name.format(
                        d, alpha=alpha, gamma=gamma, eps_min=eps_min, eps_max=eps_max, n_epss=n_epss
                    ),
                ),
                "rb",
            ) as f:
                data = pickle.load(f)

                epss_g = data["eps"]
                mean_m = data["mean_m"]
                std_m = data["std_m"]
                mean_q = data["mean_q"]
                std_q = data["std_q"]
                mean_q_latent = data["mean_q_latent"]
                std_q_latent = data["std_q_latent"]
                mean_q_feature = data["mean_q_feature"]
                std_q_feature = data["std_q_feature"]
                mean_P = data["mean_P"]
                std_P = data["std_P"]
                mean_rho = data["mean_rho"]
                std_rho = data["std_rho"]
                mean_flipped = data["mean_flipped"]
                std_flipped = data["std_flipped"]

            plt.errorbar(
                epss_g,
                mean_flipped,
                yerr=std_flipped,
                linestyle="",
                color=f"C{idx}",
                marker=mks,
                label=f"$\\gamma = $ {gamma:.1f}",
            )

        init_cond = (mean_m, mean_q, 1.0, mean_P)

        f_kwargs = {"reg_param": reg_param, "gamma": gamma}
        f_hat_kwargs = {"alpha": alpha, "gamma": gamma, "ε": eps_training}

        print(f_kwargs, f_hat_kwargs)

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

        q_latent_se = q_latent_hastie_L2_reg_Linf_attack(
            m_hat, q_hat, V_hat, P_hat, reg_param, gamma
        )
        q_features_se = q_features_hastie_L2_reg_Linf_attack(
            m_hat, q_hat, V_hat, P_hat, reg_param, gamma
        )

        for j, eps_i in enumerate(tqdm(eps_dense)):
            out[j] = percentage_flipped_hastie_model(
                m_se,
                q_se,
                q_latent_se,
                q_features_se,
                1,
                eps_i,
                gamma,
                "inf",
            )

        plt.plot(eps_dense, out, color=f"C{idx}", label=f"$\\gamma = $ {gamma:.1f}")

    plt.legend()

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.xlabel(r"$\epsilon_g (\sqrt[p]{d} / \sqrt{d})$")
    plt.ylabel("Percentage of flipped labels")
    plt.grid(which="both")
    plt.title(f"$\\alpha$ = {alpha:.1f} $\\epsilon_t = {eps_training:.2f}$")

plt.suptitle(f"Linear Random Features p = {p} $\\lambda$ = {reg_param:.1e}")
plt.tight_layout()

plt.show()
