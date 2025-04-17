import matplotlib.pyplot as plt
import numpy as np
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.aux_functions.misc import classification_adversarial_error
from linear_regression.fixed_point_equations.regularisation.hastie_model_pstar_attacks import (
    f_hastie_L2_reg_Lp_attack,
)
from linear_regression.fixed_point_equations.classification.Adv_train_p_norm_hastie import (
    f_hat_Logistic_no_noise_Linf_adv_classif,
)
import pickle
from os.path import join, exists
import os
import sys

if len(sys.argv) > 1:
    gamma_min, gamma_max, n_gammas, alpha, eps_t, reg_param = (
        float(sys.argv[1]),
        float(sys.argv[2]),
        int(sys.argv[3]),
        float(sys.argv[4]),
        float(sys.argv[5]),
        float(sys.argv[6]),
    )
else:
    gamma_min, gamma_max, n_gammas, alpha, eps_t, reg_param = (
        0.5,
        2.0,
        60,
        2.0,
        0.1,
        1e-3,
    )

pstar = 1

data_folder = "./data/hastie_model_training"
file_name = f"SE_training_alpha_{alpha:.2f}_gammas_{gamma_min:.1f}_{gamma_max:.1f}_{n_gammas:d}_eps_{eps_t:.2f}_reg_param_{reg_param:.1e}_pstar_{pstar:.1f}.csv"

if not exists(data_folder):
    os.makedirs(data_folder)

gammas = np.linspace(gamma_min, gamma_max, n_gammas)

ms_found = np.empty((n_gammas,))
qs_found = np.empty((n_gammas,))
Vs_found = np.empty((n_gammas,))
Ps_found = np.empty((n_gammas,))
estim_errors_se = np.empty((n_gammas,))
adversarial_errors_found = np.empty((n_gammas,))
gen_errors_se = np.empty((n_gammas,))
flipped_fairs_se = np.empty((n_gammas,))
misclas_fairs_se = np.empty((n_gammas,))

initial_condition = (0.6, 1.6, 1.05, 1.1)

for j, gamma in enumerate(gammas):
    print(f"Gamma: {gamma:.2f} / {gamma_max:.2f}")

    f_kwargs = {"reg_param": reg_param, "pstar": pstar, "gamma": gamma}
    f_hat_kwargs = {"alpha": alpha, "gamma": gamma, "ε": eps_t}

    ms_found[j], qs_found[j], Vs_found[j], Ps_found[j] = fixed_point_finder(
        f_hastie_L2_reg_Lp_attack,
        f_hat_Logistic_no_noise_Linf_adv_classif,
        initial_condition,
        f_kwargs,
        f_hat_kwargs,
        abs_tol=1e-4,
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

    estim_errors_se[j] = 1 - 2 * ms_found[j] + qs_found[j]
    adversarial_errors_found[j] = classification_adversarial_error(
        ms_found[j], qs_found[j], Ps_found[j], eps_t, pstar
    )
    gen_errors_se[j] = np.arccos(ms_found[j] / np.sqrt(qs_found[j])) / np.pi

    flipped_fairs_se[j] = 0.0
    misclas_fairs_se[j] = 0.0

data = {
    "gamma": gammas,
    "m": ms_found,
    "q": qs_found,
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

plt.plot(
    gammas,
    adversarial_errors_found,
    label="Adversarial Error",
    color="blue",
)
plt.plot(gammas, gen_errors_se, label="Generalization Error", color="orange")
plt.legend()
plt.xlabel("$\\gamma$")
plt.ylabel("Error")
plt.show()
