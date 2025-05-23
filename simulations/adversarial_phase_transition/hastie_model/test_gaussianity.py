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
    find_coefficients_Logistic_adv_Linf_L2,
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
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_linear_rf,
)
from tqdm.auto import tqdm
import os
import sys
from itertools import product
import warnings
import pickle

warnings.filterwarnings("error")

if len(sys.argv) > 1:
    eps_min, eps_max, n_epss, alpha, gamma, reg_param, eps_training = (
        float(sys.argv[1]),
        float(sys.argv[2]),
        int(sys.argv[3]),
        float(sys.argv[4]),
        float(sys.argv[5]),
        float(sys.argv[6]),
        float(sys.argv[7]),
    )
else:
    eps_min, eps_max, n_epss, alpha, gamma, reg_param, eps_training = (
        0.1,
        10.0,
        25,
        0.3,
        1.0,
        1e-2,
        0.5,
    )

gamma = 1.0

# DO NOT CHANGE, NOT IMPLEMENTED FOR OTHERS
pstar_t = 1.0

d = 1000
reps = 25

epss = np.logspace(np.log10(eps_min), np.log10(eps_max), n_epss)

data_folder = "./data/hastie_model_training"
file_name = f"ERM_flipped_Hastie_Linf_d_{{:d}}_alpha_{alpha:.1f}_gamma_{gamma:.1f}_reps_{reps:d}_epss_{eps_min:.1f}_{eps_max:.1f}_{n_epss:d}_reg_param_{reg_param:.1e}_eps_t_{eps_training:.2f}.pkl"

print("starting")

init_cond = (0.1, 1.0, 1.0, 1.0)

f_kwargs = {"reg_param": reg_param, "gamma": gamma}
f_hat_kwargs = {"alpha": alpha, "gamma": gamma, "ε": eps_training}

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
q_features_se = q_features_hastie_L2_reg_Linf_attack(m_hat, q_hat, V_hat, P_hat, reg_param, gamma)

print("theoretical values done")

p = int(d / gamma)
n = int(d * alpha)

print(f"p: {p}, d: {d}, n: {n}")
# works for p = "inf"
epss_rescaled = epss * (d ** (-1 / 2))

m_vals = []
rep = 0
# while rep < reps:
xs, ys, zs, xs_gen, ys_gen, zs_gen, wstar, F, noise, noise_gen = data_generation_hastie(
    measure_gen_no_noise_clasif,
    d=d,
    n=max(n, 1),
    n_gen=1000,
    measure_fun_args={},
    gamma=gamma,
    noi=True,
)

try:
    # if eps_training == 0.0:
    #     w = find_coefficients_Logistic(ys, xs, reg_param)
    # else:
    #     w = find_coefficients_Logistic_adv(
    #         ys, xs, 0.5 * reg_param, eps_training, 2.0, pstar_t, F @ wstar
    #     )
    w = find_coefficients_Logistic_adv_Linf_L2(ys, xs, 0.5 * reg_param, eps_training)
except ValueError as e:
    print("Error in finding coefficients:", e)

    # m_vals.append(np.dot(wstar, F.T @ w) / (p * np.sqrt(gamma)))
    # print(f"rep {rep} done")
    # rep += 1

# print(F)
wtilde = F.T @ w

print(np.count_nonzero(np.abs(wtilde) < 1e-2) / p)

xi = np.random.randn(d)

plt.hist(wtilde[:p], bins=50, density=True, alpha=0.5, label=f"w")
aaa = m_se / np.sqrt(gamma) * wstar + np.sqrt(q_latent_se - m_se**2 / gamma) * xi

plt.hist(
    aaa,
    bins=50,
    density=True,
    alpha=0.5,
    label=f"m_se / sqrt(gamma) * wstar + sqrt(q_latent_se - m_se**2 / gamma) * xi",
)

print(np.std(wtilde[:p]))
print(np.std(aaa))
print(q_features_se)

# plt.hist(
#     m_vals,
#     density=True,
#     alpha=0.5,
#     label=f"w, d={d}, eps={eps_training:.2f}",
# )

# plt.hist(
#     np.dot(zs_gen, F.T @ w) / np.sqrt(p),
#     bins=50,
#     density=True,
#     alpha=0.5,
#     label=f"F.T @ w, d={d}, eps={eps_training:.2f}",
# )

# plt.hist(
#     noise_gen @ w / np.sqrt(p),
#     bins=50,
#     density=True,
#     alpha=0.5,
#     label=f"noise @ w, d={d}, eps={eps_training:.2f}",
# )

# plot the mean and std of the distribution
# plt.axvline(
#     np.mean(m_vals),
#     color="black",
#     linestyle="-",
#     label=f"$\\mu$ (empirical)",
# )
# plt.axvline(
#     np.mean(m_vals) - np.std(m_vals),
#     color="black",
#     linestyle="--",
#     label=f"$\\mu - \\sigma$ (empirical)",
# )
# plt.axvline(
#     np.mean(m_vals) + np.std(m_vals),
#     color="black",
#     linestyle="--",
#     label=f"$\\mu + \\sigma$ (empirical)",
# )

# plt.axvline(
#     m_se,
#     color="red",
#     linestyle="--",
# )
# plt.axvline(
#     m_se / np.sqrt(gamma),
#     color="blue",
#     linestyle="--",
# )
# plt.axvline(
#     m_se * np.sqrt(gamma),
#     color="green",
#     linestyle="--",
# )

# xx = np.linspace(-5, 5, 100)
# q = np.dot(w, w) / p
# plt.plot(
#     xx,
#     np.exp(-(xx**2) / (2 * q)) / np.sqrt(2 * np.pi * q),
#     label="Gaussian",
#     color="black",
# )

# plt.plot(
#     xx,
#     np.exp(-(xx**2) / (2 * m_se)) / np.sqrt(2 * np.pi * m_se),
#     label="q_latent",
#     color="red",
# )

plt.legend()

plt.show()
