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
from linear_regression.aux_functions.percentage_flipped import (
    percentage_misclassified_hastie_model,
    percentage_flipped_hastie_model,
)
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_linear_rf,
)
from cvxpy.error import SolverError
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

warnings.filterwarnings("error")

if len(sys.argv) > 1:
    eps_min, eps_max, n_epss, alpha, gamma = (
        float(sys.argv[1]),
        float(sys.argv[2]),
        int(sys.argv[3]),
        float(sys.argv[4]),
        float(sys.argv[5]),
    )
else:
    eps_min, eps_max, n_epss, alpha, gamma = (0.1, 10.0, 15, 1.2, 1.0)

pstar_t = 1.0
eps_test = 1.0

dimensions = [int(2**a) for a in range(9, 10)]
reps = 10

epss = np.logspace(np.log10(eps_min), np.log10(eps_max), n_epss)

data_folder = "./data/hastie_model_training_optimal"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

file_name_flipped_no_adv = f"ERM_flipped_optimal_noadvtrain_Hastie_Linf_d_{{:d}}_alpha_{alpha:.1f}_gamma_{gamma:.1f}_reps_{reps:d}_epss_{eps_min:.1f}_{eps_max:.1f}_{n_epss:d}.csv"
file_name_flipped_adv = f"ERM_flipped_optimal_advtrain_Hastie_Linf_d_{{:d}}_alpha_{alpha:.1f}_gamma_{gamma:.1f}_reps_{reps:d}_epss_{eps_min:.1f}_{eps_max:.1f}_{n_epss:d}.csv"


def fun_to_min(reg_param):
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

    return percentage_flipped_hastie_model(
        m_se, q_se, q_latent_se, q_features_se, 1.0, eps_test, gamma, "inf"
    )


def fun_to_min_2(x):
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

    return percentage_flipped_hastie_model(
        m_se, q_se, q_latent_se, q_features_se, 1.0, eps_test, gamma, "inf"
    )


def hybrid_minimize_2d(fun, bounds, n_samples=100, method="Nelder-Mead", options=None):
    # Generate random samples within bounds
    samples = np.random.uniform(
        low=[bounds[0][0], bounds[1][0]], high=[bounds[0][1], bounds[1][1]], size=(n_samples, 2)
    )

    # Evaluate function at each sample
    values = np.array([fun(p) for p in samples])

    # Select best starting point
    best_start = samples[np.argmin(values)]

    print("Best start point:", best_start)

    # Run local optimization
    result = minimize(fun, best_start, bounds=bounds, method=method, options=options)

    return result


# Find the optimal reg_param
res = minimize_scalar(
    fun_to_min,
    bounds=(1e-5, 1e0),
    method="bounded",
    options={"xatol": 1e-5, "disp": True},
)
reg_param_noadv_opt = res.x
print(res.fun, res.x)


# res = minimize(
#     fun_to_min_2,
#     # (reg_param_noadv_opt, 0.05),
#     (np.random.uniform(0.0, 1e-1), np.random.uniform(0.0, 1e-1)),
#     bounds=((1e-5, 1e0), (0.0, 5e-1)),
#     method="Nelder-Mead",
#     options={"xatol": 1e-5, "disp": True},
# )
# reg_param_opt, eps_training_opt = res.x
# print(res.fun, res.x)

res = hybrid_minimize_2d(
    fun=fun_to_min_2,
    bounds=((1e-5, 1.0), (0.0, 0.5)),
    n_samples=200,
    options={"xatol": 1e-5, "disp": True},
)
reg_param_opt, eps_training_opt = res.x
print(res.fun, res.x)

print("theory done")

for d in tqdm(dimensions, desc="dim", leave=False):
    p = int(d / gamma)
    n = int(d * alpha)

    print(f"p: {p}, d: {d}, n: {n}")

    epss_rescaled = epss / np.sqrt(d)

    vals = np.empty((reps, len(epss)))
    estim_vals_m = np.empty((reps,))
    estim_vals_q = np.empty((reps,))
    estim_vals_q_latent = np.empty((reps,))
    estim_vals_q_feature = np.empty((reps,))
    estim_vals_rho = np.empty((reps,))
    estim_vals_P = np.empty((reps,))

    j = 0
    while j < reps:
        xs, ys, zs, xs_gen, ys_gen, zs_gen, wstar, F, noise, noise_gen = data_generation_hastie(
            measure_gen_no_noise_clasif,
            d=d,
            n=max(n, 1),
            n_gen=5000,
            measure_fun_args={},
            gamma=gamma,
            noi=True,
        )

        assert xs.shape == (n, p)
        assert ys.shape == (n,)
        assert zs.shape == (n, d)
        assert F.shape == (p, d)

        try:
            w = find_coefficients_Logistic(ys, xs, reg_param_noadv_opt)
            print("j", j)
        except (ValueError, UserWarning) as e:
            print(f"Error in finding coefficients {j}:", e)
            continue

        estim_vals_rho[j] = np.sum(wstar**2) / d
        estim_vals_m[j] = np.dot(wstar, F.T @ w) / (p * np.sqrt(gamma))
        estim_vals_q[j] = np.dot(F.T @ w, F.T @ w) / p + np.dot(w, w) / p
        estim_vals_q_latent[j] = np.dot(F.T @ w, F.T @ w) / d
        estim_vals_q_feature[j] = np.dot(w, w) / p
        estim_vals_P[j] = np.mean(np.abs(w))

        yhat = np.repeat(np.sign(xs @ w).reshape(-1, 1), d, axis=1)

        yhat_gen = np.sign(np.dot(xs_gen, w))

        i = 0
        while i < len(epss_rescaled):
            print("i", i)
            eps_i = epss_rescaled[i]
            try:
                adv_perturbation = find_adversarial_perturbation_linear_rf(
                    yhat_gen, zs_gen, w, F.T, wstar, eps_i, "inf"
                )
            except (ValueError, UserWarning) as e:
                print("Error in finding adversarial perturbation:", e)
                break  # Restart the outer loop
            flipped = np.mean(
                yhat_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w)
            )

            vals[j, i] = flipped
            i += 1
        else:
            j += 1  # Only increment j if the inner loop completes successfully

    mean_m, std_m = np.mean(estim_vals_m), np.std(estim_vals_m)
    mean_q, std_q = np.mean(estim_vals_q), np.std(estim_vals_q)
    mean_q_latent, std_q_latent = np.mean(estim_vals_q_latent), np.std(estim_vals_q_latent)
    mean_q_feature, std_q_feature = np.mean(estim_vals_q_feature), np.std(estim_vals_q_feature)
    mean_P, std_P = np.mean(estim_vals_P), np.std(estim_vals_P)
    mean_rho, std_rho = np.mean(estim_vals_rho), np.std(estim_vals_rho)
    mean_flipped, std_flipped = np.mean(vals, axis=0), np.std(vals, axis=0)

    data = {
        "optimal_reg_param": reg_param_noadv_opt,
        "eps": epss,
        "vals": vals,
        "mean_m": mean_m,
        "std_m": std_m,
        "mean_q": mean_q,
        "std_q": std_q,
        "mean_q_latent": mean_q_latent,
        "std_q_latent": std_q_latent,
        "mean_q_feature": mean_q_feature,
        "std_q_feature": std_q_feature,
        "mean_P": mean_P,
        "std_P": std_P,
        "mean_rho": mean_rho,
        "std_rho": std_rho,
        "mean_flipped": mean_flipped,
        "std_flipped": std_flipped,
    }

    data_file = os.path.join(data_folder, file_name_flipped_no_adv.format(d))

    with open(data_file, "wb") as f:
        pickle.dump(data, f)

    print("saved data to", data_file)

#     plt.errorbar(epss, mean_flipped, yerr=std_flipped, linestyle="", marker=".", label=f"$d = {d}$")

# if gamma <= 1:
#     plt.plot(
#         epss,
#         erf(
#             epss
#             * np.sqrt(mean_q_latent - mean_m**2 / gamma)
#             * np.sqrt(1 / np.pi)
#             / np.sqrt(mean_q)
#             * np.sqrt(gamma)
#         ),
#         label="theoretical",
#         linestyle="--",
#     )
# else:
#     plt.plot(
#         epss,
#         erf(
#             epss
#             * np.sqrt(mean_q_feature - mean_m**2 / gamma)
#             / np.sqrt(gamma)
#             * np.sqrt(1 / np.pi)
#             / np.sqrt(mean_q)
#         ),
#         label="theoretical",
#         linestyle="--",
#     )

# plt.xlabel(r"$\epsilon$")
# plt.ylabel(r"$\mathbb{P}(\hat{y} \neq y)$")
# plt.xscale("log")
# plt.grid(which="both")
# plt.legend()
# plt.show()


for d in tqdm(dimensions, desc="dim", leave=False):
    p = int(d / gamma)
    n = int(d * alpha)

    print(f"p: {p}, d: {d}, n: {n}")

    epss_rescaled = epss / np.sqrt(d)

    vals = np.empty((reps, len(epss)))
    estim_vals_m = np.empty((reps,))
    estim_vals_q = np.empty((reps,))
    estim_vals_q_latent = np.empty((reps,))
    estim_vals_q_feature = np.empty((reps,))
    estim_vals_rho = np.empty((reps,))
    estim_vals_P = np.empty((reps,))

    j = 0
    while j < reps:
        xs, ys, zs, xs_gen, ys_gen, zs_gen, wstar, F, noise, noise_gen = data_generation_hastie(
            measure_gen_no_noise_clasif,
            d=d,
            n=max(n, 1),
            n_gen=5000,
            measure_fun_args={},
            gamma=gamma,
            noi=True,
        )

        assert xs.shape == (n, p)
        assert ys.shape == (n,)
        assert zs.shape == (n, d)
        assert F.shape == (p, d)

        try:
            # if eps_training_opt == 0.0:
            #     w = find_coefficients_Logistic(ys, xs, reg_param_opt)
            # else:
            #     w = find_coefficients_Logistic_adv(
            #         ys, xs, 0.5 * reg_param_opt, eps_training_opt, 2.0, pstar_t, F @ wstar
            #     )
            w = find_coefficients_Logistic_adv_Linf_L2(ys, xs, reg_param_opt, eps_training_opt)
            print("j", j)
        except (ValueError, UserWarning, SolverError) as e:
            print(f"Error in finding coefficients {j}:", e)
            continue

        estim_vals_rho[j] = np.sum(wstar**2) / d
        estim_vals_m[j] = np.dot(wstar, F.T @ w) / (p * np.sqrt(gamma))
        estim_vals_q[j] = np.dot(F.T @ w, F.T @ w) / p + np.dot(w, w) / p
        estim_vals_q_latent[j] = np.dot(F.T @ w, F.T @ w) / d
        estim_vals_q_feature[j] = np.dot(w, w) / p
        estim_vals_P[j] = np.mean(np.abs(w))

        yhat = np.repeat(np.sign(xs @ w).reshape(-1, 1), d, axis=1)

        yhat_gen = np.sign(np.dot(xs_gen, w))

        i = 0
        while i < len(epss_rescaled):
            print("i", i)
            eps_i = epss_rescaled[i]
            try:
                adv_perturbation = find_adversarial_perturbation_linear_rf(
                    yhat_gen, zs_gen, w, F.T, wstar, eps_i, "inf"
                )
            except (ValueError, UserWarning) as e:
                print("Error in finding adversarial perturbation:", e)
                break  # Restart the outer loop
            flipped = np.mean(
                yhat_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w)
            )

            vals[j, i] = flipped
            i += 1
        else:
            j += 1  # Only increment j if the inner loop completes successfully

    mean_m, std_m = np.mean(estim_vals_m), np.std(estim_vals_m)
    mean_q, std_q = np.mean(estim_vals_q), np.std(estim_vals_q)
    mean_q_latent, std_q_latent = np.mean(estim_vals_q_latent), np.std(estim_vals_q_latent)
    mean_q_feature, std_q_feature = np.mean(estim_vals_q_feature), np.std(estim_vals_q_feature)
    mean_P, std_P = np.mean(estim_vals_P), np.std(estim_vals_P)
    mean_rho, std_rho = np.mean(estim_vals_rho), np.std(estim_vals_rho)
    mean_flipped, std_flipped = np.mean(vals, axis=0), np.std(vals, axis=0)

    data = {
        "optimal_reg_param": reg_param_opt,
        "optimal_eps_training": eps_training_opt,
        "eps": epss,
        "vals": vals,
        "mean_m": mean_m,
        "std_m": std_m,
        "mean_q": mean_q,
        "std_q": std_q,
        "mean_q_latent": mean_q_latent,
        "std_q_latent": std_q_latent,
        "mean_q_feature": mean_q_feature,
        "std_q_feature": std_q_feature,
        "mean_P": mean_P,
        "std_P": std_P,
        "mean_rho": mean_rho,
        "std_rho": std_rho,
        "mean_flipped": mean_flipped,
        "std_flipped": std_flipped,
    }

    data_file = os.path.join(data_folder, file_name_flipped_adv.format(d))

    with open(data_file, "wb") as f:
        pickle.dump(data, f)

    print("saved data to", data_file)

#     plt.errorbar(epss, mean_flipped, yerr=std_flipped, linestyle="", marker=".", label=f"$d = {d}$")

# if gamma <= 1:
#     plt.plot(
#         epss,
#         erf(
#             epss
#             * np.sqrt(mean_q_latent - mean_m**2 / gamma)
#             * np.sqrt(1 / np.pi)
#             / np.sqrt(mean_q)
#             * np.sqrt(gamma)
#         ),
#         label="theoretical",
#         linestyle="--",
#     )
# else:
#     plt.plot(
#         epss,
#         erf(
#             epss
#             * np.sqrt(mean_q_feature - mean_m**2 / gamma)
#             / np.sqrt(gamma)
#             * np.sqrt(1 / np.pi)
#             / np.sqrt(mean_q)
#         ),
#         label="theoretical",
#         linestyle="--",
#     )

# plt.xlabel(r"$\epsilon$")
# plt.ylabel(r"$\mathbb{P}(\hat{y} \neq y)$")
# plt.xscale("log")
# plt.grid(which="both")
# plt.legend()
# plt.show()
