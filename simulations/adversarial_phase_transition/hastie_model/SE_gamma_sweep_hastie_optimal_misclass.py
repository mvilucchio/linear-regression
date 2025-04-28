import numpy as np
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.aux_functions.misc import classification_adversarial_error
from linear_regression.fixed_point_equations.regularisation.hastie_model_pstar_attacks import (
    f_hastie_L2_reg_Linf_attack,
    q_latent_hastie_L2_reg_Linf_attack,
    q_features_hastie_L2_reg_Linf_attack,
)
from linear_regression.fixed_point_equations.classification.Adv_train_p_norm_hastie import (
    f_hat_Logistic_no_noise_Linf_adv_classif,
)
from linear_regression.aux_functions.percentage_flipped import (
    percentage_flipped_hastie_model,
    percentage_misclassified_hastie_model,
)
from os.path import join, exists
import os
import sys
from scipy.optimize import minimize

if len(sys.argv) > 1:
    gamma_min, gamma_max, n_gammas, alpha = (
        float(sys.argv[1]),
        float(sys.argv[2]),
        int(sys.argv[3]),
        float(sys.argv[4]),
    )
else:
    gamma_min, gamma_max, n_gammas, alpha = (0.5, 2.0, 100, 1.5)

pstar = 1
eps_test = 1.0

data_folder = "./data/hastie_model_training_optimal"
file_name = f"SE_optimal_misclass_training_alpha_{alpha:.2f}_gammas_{gamma_min:.1f}_{gamma_max:.1f}_{n_gammas:d}.csv"

if not exists(data_folder):
    os.makedirs(data_folder)


gammas = np.linspace(gamma_min, gamma_max, n_gammas)

ms_found = np.empty((n_gammas,))
qs_found = np.empty((n_gammas,))
qs_latent_found = np.empty((n_gammas,))
qs_features_found = np.empty((n_gammas,))
Vs_found = np.empty((n_gammas,))
Ps_found = np.empty((n_gammas,))
estim_errors_se = np.empty((n_gammas,))
adversarial_errors_found = np.empty((n_gammas,))
gen_errors_se = np.empty((n_gammas,))
flipped_fairs_se = np.empty((n_gammas,))
misclas_fairs_se = np.empty((n_gammas,))
reg_param_found = np.empty((n_gammas,))
eps_t_found = np.empty((n_gammas,))

initial_condition = (0.6, 1.6, 1.05, 1.1)


def hybrid_minimize_2d(fun, bounds, n_samples=100, method="Nelder-Mead", options=None):
    samples = np.random.uniform(
        low=[bounds[0][0], bounds[1][0]], high=[bounds[0][1], bounds[1][1]], size=(n_samples, 2)
    )

    values = np.array([fun(p) for p in samples])

    best_start = samples[np.argmin(values)]

    print("Best start point:", best_start)

    result = minimize(fun, best_start, bounds=bounds, method=method, options=options)

    return result


for j, gamma in enumerate(gammas):
    print(f"Gamma: {gamma:.2f} / {gamma_max:.2f}")

    def fun_to_min_2(x):
        reg_param, eps_training = x
        init_cond = (0.1, 1.0, 1.0, 1.0)

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

        return percentage_misclassified_hastie_model(
            m_se, q_se, q_latent_se, q_features_se, 1.0, eps_test, gamma, "inf"
        )

    # res = minimize(
    #     fun_to_min_2,
    #     (0.1, 0.1),
    #     bounds=((1e-5, 1e0), (0.0, 5e-1)),
    #     method="Nelder-Mead",
    #     options={"xatol": 1e-5, "disp": True},
    # )
    # reg_param, eps_t = res.x
    res = hybrid_minimize_2d(
        fun=fun_to_min_2,
        bounds=((1e-5, 1.0), (0.0, 0.5)),
        n_samples=200,
        options={"xatol": 1e-5, "disp": True},
    )
    reg_param, eps_t = res.x
    print(res.fun, res.x)

    reg_param_found[j] = reg_param
    eps_t_found[j] = eps_t

    f_kwargs = {"reg_param": reg_param, "gamma": gamma}
    f_hat_kwargs = {"alpha": alpha, "gamma": gamma, "ε": eps_t}

    ms_found[j], qs_found[j], Vs_found[j], Ps_found[j] = fixed_point_finder(
        f_hastie_L2_reg_Linf_attack,
        f_hat_Logistic_no_noise_Linf_adv_classif,
        initial_condition,
        f_kwargs,
        f_hat_kwargs,
        abs_tol=1e-5,
        min_iter=10,
        verbose=False,
        print_every=1,
    )

    initial_condition = (ms_found[j], qs_found[j], Vs_found[j], Ps_found[j])

    print(f"({ms_found[j]:.6f}, {qs_found[j]:.6f}, {Vs_found[j]:.6f}, {Ps_found[j]:.6f})")
    m_hat, q_hat, V_hat, P_hat = f_hat_Logistic_no_noise_Linf_adv_classif(
        ms_found[j], qs_found[j], Vs_found[j], Ps_found[j], eps_t, alpha, gamma
    )
    print(f"({m_hat:.6f}, {q_hat:.6f}, {V_hat:.6f}, {P_hat:.6f})")

    qs_latent_found[j] = q_latent_hastie_L2_reg_Linf_attack(
        m_hat, q_hat, V_hat, P_hat, reg_param, gamma
    )
    qs_features_found[j] = q_features_hastie_L2_reg_Linf_attack(
        m_hat, q_hat, V_hat, P_hat, reg_param, gamma
    )

    estim_errors_se[j] = 1 - 2 * ms_found[j] + qs_found[j]
    adversarial_errors_found[j] = classification_adversarial_error(
        ms_found[j], qs_found[j], Ps_found[j], eps_t, pstar
    )
    gen_errors_se[j] = np.arccos(ms_found[j] / np.sqrt(qs_found[j])) / np.pi

    flipped_fairs_se[j] = percentage_flipped_hastie_model(
        ms_found[j],
        qs_found[j],
        qs_latent_found[j],
        qs_features_found[j],
        1.0,
        eps_test,
        gamma,
        "inf",
    )
    misclas_fairs_se[j] = percentage_misclassified_hastie_model(
        ms_found[j],
        qs_found[j],
        qs_latent_found[j],
        qs_features_found[j],
        1.0,
        eps_test,
        gamma,
        "inf",
    )

data = {
    "gamma": gammas,
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
}

data_array = np.column_stack([data[key] for key in data.keys()])
header = ",".join(data.keys())
np.savetxt(
    join(data_folder, file_name),
    data_array,
    delimiter=",",
    header=header,
    comments="",
)
