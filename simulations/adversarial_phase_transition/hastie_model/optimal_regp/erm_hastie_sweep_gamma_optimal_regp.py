from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic_adv,
    find_coefficients_Logistic_adv_Linf_L2,
)
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_linear_rf,
    find_adversarial_error_rf,
)
from linear_regression.data.generation import data_generation_hastie, measure_gen_probit_clasif
from linear_regression.erm.metrics import (
    generalisation_error_classification,
    adversarial_error_data,
    percentage_flipped_labels_estim,
    percentage_error_from_true,
)
from cvxpy.error import SolverError
from linear_regression.utils.errors import ConvergenceError
import numpy as np
import os
import sys

if len(sys.argv) > 1:
    gamma_min, gamma_max, n_gammas, d, alpha, eps_t, delta, reg_param = (
        float(sys.argv[1]),
        float(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
        float(sys.argv[5]),
        float(sys.argv[6]),
        float(sys.argv[7]),
        float(sys.argv[8]),
    )
else:
    gamma_min, gamma_max, n_gammas = 0.5, 3.0, 10
    alpha = 2.0
    delta = 0.0
    reg_param = 1e-2
    d = 500

eps_t = 0.0
reps = 10
n_gen = 1000

pstar = 1.0

eps_test = 1.0

data_folder = f"./data/hastie_model_training_optimal"
file_name_misclass = f"ERM_optimal_regp_misclass_alpha_{alpha:.2f}_gammas_{gamma_min:.1f}_{gamma_max:.1f}_{n_gammas:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{pstar:.1f}_reg_{pstar:.1f}.csv"
file_name_flipped = f"ERM_optimal_regp_flipped_alpha_{alpha:.2f}_gammas_{gamma_min:.1f}_{gamma_max:.1f}_{n_gammas:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{pstar:.1f}_reg_{pstar:.1f}.csv"
file_name_adverr = f"ERM_optimal_regp_adverr_alpha_{alpha:.2f}_gammas_{gamma_min:.1f}_{gamma_max:.1f}_{n_gammas:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_pstar_{pstar:.1f}_reg_{pstar:.1f}.csv"

gamma_min_se, gamma_max_se, n_gammas_se = 0.5, 3.0, 100

file_name_misclass_SE = f"SE_optimal_regp_misclass_alpha_{{alpha:.2f}}_gammas_{gamma_min:.1f}_{gamma_max:.1f}_{n_gammas:d}_pstar_{pstar:.1f}_reg_{pstar:.1f}.csv"
file_name_flipped_SE = f"SE_optimal_regp_flipped_alpha_{{alpha:.2f}}_gammas_{gamma_min:.1f}_{gamma_max:.1f}_{n_gammas:d}_pstar_{pstar:.1f}_reg_{pstar:.1f}.csv"
file_name_adverr_SE = f"SE_optimal_regp_adverr_alpha_{{alpha:.2f}}_gammas_{gamma_min:.1f}_{gamma_max:.1f}_{n_gammas:d}_pstar_{pstar:.1f}_reg_{pstar:.1f}.csv"

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# ---------------------------------------------------------------------------- #
#                                 misclassified                                #
# ---------------------------------------------------------------------------- #

if os.path.exists(os.path.join(data_folder, file_name_misclass_SE.format(alpha=alpha))):
    print(f"SE file {file_name_misclass_SE.format(alpha=alpha)} exists.")

SE_data = np.loadtxt(
    os.path.join(data_folder, file_name_misclass_SE.format(alpha=alpha)),
    delimiter=",",
    skiprows=1,
)

gammas_SE = SE_data[:, 0]

gamma_list = np.linspace(gamma_min, gamma_max, n_gammas)

indices = np.searchsorted(gammas_SE, gamma_list)
gamma_list = SE_data[indices, 0]
reg_param_list = SE_data[indices, -1]

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

ms = np.empty((n_gammas, 2))
qs = np.empty((n_gammas, 2))
q_latent = np.empty((n_gammas, 2))
q_feature = np.empty((n_gammas, 2))
Ps = np.empty((n_gammas, 2))
gen_errs = np.empty((n_gammas, 2))
adv_errs = np.empty((n_gammas, 2))
flipped_fairs = np.empty((n_gammas, 2))
misclas_fairs = np.empty((n_gammas, 2))

# do a for loop on the alpha_list and compute for each the p = d / gamma and n = alpha * d
for i, (gamma, reg_param) in enumerate(gamma_list, reg_param_list):
    p = int(d / gamma)
    n = int(alpha * d)

    print(f"p {p} d {d} n {n}")

    m_vals = []
    q_vals = []
    q_latent_vals = []
    q_feature_vals = []
    P_vals = []
    gen_err_vals = []
    adv_err_vals = []
    flip_fair_vals = []
    misc_fair_vals = []

    j = 0
    while j < reps:
        xs, ys, zs, xs_gen, ys_gen, zs_gen, wstar, F, noise, noise_gen = data_generation_hastie(
            measure_gen_probit_clasif, d, n, n_gen, (delta,), gamma, noi=True
        )

        try:
            w = find_coefficients_Logistic_adv_Linf_L2(ys, xs, 0.5 * reg_param, eps_t)
        except (ValueError, SolverError) as e:
            print(
                f"minimization didn't converge on iteration {j} for gamma {gamma:.2f}. Trying again."
            )
            continue

        m_vals.append(np.dot(wstar, F.T @ w) / (p * np.sqrt(gamma)))
        q_vals.append(np.dot(F.T @ w, F.T @ w) / p + np.dot(w, w) / p)
        q_latent_vals.append(np.dot(F.T @ w, F.T @ w) / d)
        q_feature_vals.append(np.dot(w, w) / p)
        P_vals.append(np.mean(np.abs(w)))

        yhat_gen = np.sign(np.dot(xs_gen, w))

        gen_err_vals.append(generalisation_error_classification(ys_gen, xs_gen, w, wstar))

        adv_pert = find_adversarial_error_rf(
            ys_gen, zs_gen, w, F.T, wstar, eps_test / np.sqrt(d), "inf"
        )
        adv_err = np.mean(ys_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w))
        adv_err_vals.append(adv_err)

        # calculation of flipped perturbation
        adv_perturbation = find_adversarial_perturbation_linear_rf(
            yhat_gen, zs_gen, w, F.T, wstar, eps_test / np.sqrt(d), "inf"
        )
        flipped = np.mean(
            yhat_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w)
        )
        flip_fair_vals.append(flipped)

        # calculation of perturbation
        adv_perturbation = find_adversarial_perturbation_linear_rf(
            ys_gen, zs_gen, w, F.T, wstar, eps_test / np.sqrt(d), "inf"
        )

        misclass = np.mean(ys_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w))
        misc_fair_vals.append(misclass)

        print(f"repetition {j} for gamma {gamma:.2f} done.")
        j += 1

    ms[i, 0], ms[i, 1] = np.mean(m_vals), np.std(m_vals)
    qs[i, 0], qs[i, 1] = np.mean(q_vals), np.std(q_vals)
    q_latent[i, 0], q_latent[i, 1] = np.mean(q_latent_vals), np.std(q_latent_vals)
    q_feature[i, 0], q_feature[i, 1] = np.mean(q_feature_vals), np.std(q_feature_vals)
    Ps[i, 0], Ps[i, 1] = np.mean(P_vals), np.std(P_vals)
    gen_errs[i, 0], gen_errs[i, 1] = np.mean(gen_err_vals), np.std(gen_err_vals)
    adv_errs[i, 0], adv_errs[i, 1] = np.mean(adv_err_vals), np.std(adv_err_vals)
    flipped_fairs[i, 0], flipped_fairs[i, 1] = np.mean(flip_fair_vals), np.std(flip_fair_vals)
    misclas_fairs[i, 0], misclas_fairs[i, 1] = np.mean(misc_fair_vals), np.std(misc_fair_vals)

    print(f"gamma {gamma:.2f} done.")

np.savetxt(
    os.path.join(data_folder, file_name_misclass),
    np.column_stack(
        (
            gamma_list,
            ms[:, 0],
            ms[:, 1],
            qs[:, 0],
            qs[:, 1],
            q_latent[:, 0],
            q_latent[:, 1],
            q_feature[:, 0],
            q_feature[:, 1],
            Ps[:, 0],
            Ps[:, 1],
            gen_errs[:, 0],
            gen_errs[:, 1],
            adv_errs[:, 0],
            adv_errs[:, 1],
            flipped_fairs[:, 0],
            flipped_fairs[:, 1],
            misclas_fairs[:, 0],
            misclas_fairs[:, 1],
            reg_param_list,
        )
    ),
    delimiter=",",
    header="gamma,m_mean,m_std,q_mean,q_std,q_latent_mean,q_latent_std,q_feature_mean,q_feature_std,P_mean,P_std,gen_err_mean,gen_err_std,adv_err_mean,adv_err_std,flipped_fair_mean,flipped_fair_std,misclas_fair_mean,misclas_fair_std,reg_param",
)

print("data saved for misclass.")


# ---------------------------------------------------------------------------- #
#                                    flipped                                   #
# ---------------------------------------------------------------------------- #


if os.path.exists(os.path.join(data_folder, file_name_flipped_SE.format(alpha=alpha))):
    print(f"SE file {file_name_misclass_SE.format(alpha=alpha)} exists.")

SE_data = np.loadtxt(
    os.path.join(data_folder, file_name_flipped_SE.format(alpha=alpha)),
    delimiter=",",
    skiprows=1,
)

gammas_SE = SE_data[:, 0]

gamma_list = np.linspace(gamma_min, gamma_max, n_gammas)

indices = np.searchsorted(gammas_SE, gamma_list)
gamma_list = SE_data[indices, 0]
reg_param_list = SE_data[indices, -1]

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

ms = np.empty((n_gammas, 2))
qs = np.empty((n_gammas, 2))
q_latent = np.empty((n_gammas, 2))
q_feature = np.empty((n_gammas, 2))
Ps = np.empty((n_gammas, 2))
gen_errs = np.empty((n_gammas, 2))
adv_errs = np.empty((n_gammas, 2))
flipped_fairs = np.empty((n_gammas, 2))
misclas_fairs = np.empty((n_gammas, 2))

# do a for loop on the alpha_list and compute for each the p = d / gamma and n = alpha * d
for i, (gamma, reg_param) in enumerate(gamma_list, reg_param_list):
    p = int(d / gamma)
    n = int(alpha * d)

    print(f"p {p} d {d} n {n}")

    m_vals = []
    q_vals = []
    q_latent_vals = []
    q_feature_vals = []
    P_vals = []
    gen_err_vals = []
    adv_err_vals = []
    flip_fair_vals = []
    misc_fair_vals = []

    j = 0
    while j < reps:
        xs, ys, zs, xs_gen, ys_gen, zs_gen, wstar, F, noise, noise_gen = data_generation_hastie(
            measure_gen_probit_clasif, d, n, n_gen, (delta,), gamma, noi=True
        )

        try:
            w = find_coefficients_Logistic_adv_Linf_L2(ys, xs, 0.5 * reg_param, eps_t)
        except (ValueError, SolverError) as e:
            print(
                f"minimization didn't converge on iteration {j} for gamma {gamma:.2f}. Trying again."
            )
            continue

        m_vals.append(np.dot(wstar, F.T @ w) / (p * np.sqrt(gamma)))
        q_vals.append(np.dot(F.T @ w, F.T @ w) / p + np.dot(w, w) / p)
        q_latent_vals.append(np.dot(F.T @ w, F.T @ w) / d)
        q_feature_vals.append(np.dot(w, w) / p)
        P_vals.append(np.mean(np.abs(w)))

        yhat_gen = np.sign(np.dot(xs_gen, w))

        gen_err_vals.append(generalisation_error_classification(ys_gen, xs_gen, w, wstar))
        # adv_err_vals.append(adversarial_error_data(ys_gen, xs_gen, w, wstar, eps_test, pstar))
        adv_pert = find_adversarial_error_rf(
            ys_gen, zs_gen, w, F.T, wstar, eps_test / np.sqrt(d), "inf"
        )
        adv_err = np.mean(ys_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w))
        adv_err_vals.append(adv_err)

        # calculation of flipped perturbation
        adv_perturbation = find_adversarial_perturbation_linear_rf(
            yhat_gen, zs_gen, w, F.T, wstar, eps_test / np.sqrt(d), "inf"
        )
        flipped = np.mean(
            yhat_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w)
        )
        flip_fair_vals.append(flipped)

        # calculation of perturbation
        adv_perturbation = find_adversarial_perturbation_linear_rf(
            ys_gen, zs_gen, w, F.T, wstar, eps_test / np.sqrt(d), "inf"
        )

        misclass = np.mean(ys_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w))
        misc_fair_vals.append(misclass)

        print(f"repetition {j} for gamma {gamma:.2f} done.")
        j += 1

    ms[i, 0], ms[i, 1] = np.mean(m_vals), np.std(m_vals)
    qs[i, 0], qs[i, 1] = np.mean(q_vals), np.std(q_vals)
    q_latent[i, 0], q_latent[i, 1] = np.mean(q_latent_vals), np.std(q_latent_vals)
    q_feature[i, 0], q_feature[i, 1] = np.mean(q_feature_vals), np.std(q_feature_vals)
    Ps[i, 0], Ps[i, 1] = np.mean(P_vals), np.std(P_vals)
    gen_errs[i, 0], gen_errs[i, 1] = np.mean(gen_err_vals), np.std(gen_err_vals)
    adv_errs[i, 0], adv_errs[i, 1] = np.mean(adv_err_vals), np.std(adv_err_vals)
    flipped_fairs[i, 0], flipped_fairs[i, 1] = np.mean(flip_fair_vals), np.std(flip_fair_vals)
    misclas_fairs[i, 0], misclas_fairs[i, 1] = np.mean(misc_fair_vals), np.std(misc_fair_vals)

    print(f"gamma {gamma:.2f} done.")

np.savetxt(
    os.path.join(data_folder, file_name_flipped),
    np.column_stack(
        (
            gamma_list,
            ms[:, 0],
            ms[:, 1],
            qs[:, 0],
            qs[:, 1],
            q_latent[:, 0],
            q_latent[:, 1],
            q_feature[:, 0],
            q_feature[:, 1],
            Ps[:, 0],
            Ps[:, 1],
            gen_errs[:, 0],
            gen_errs[:, 1],
            adv_errs[:, 0],
            adv_errs[:, 1],
            flipped_fairs[:, 0],
            flipped_fairs[:, 1],
            misclas_fairs[:, 0],
            misclas_fairs[:, 1],
            reg_param_list,
        )
    ),
    delimiter=",",
    header="gamma,m_mean,m_std,q_mean,q_std,q_latent_mean,q_latent_std,q_feature_mean,q_feature_std,P_mean,P_std,gen_err_mean,gen_err_std,adv_err_mean,adv_err_std,flipped_fair_mean,flipped_fair_std,misclas_fair_mean,misclas_fair_std,reg_param",
)

print("data saved for misclass.")


# ---------------------------------------------------------------------------- #
#                              adversaraial error                              #
# ---------------------------------------------------------------------------- #

if os.path.exists(os.path.join(data_folder, file_name_adverr_SE.format(alpha=alpha))):
    print(f"SE file {file_name_misclass_SE.format(alpha=alpha)} exists.")

SE_data = np.loadtxt(
    os.path.join(data_folder, file_name_adverr_SE.format(alpha=alpha)),
    delimiter=",",
    skiprows=1,
)

gammas_SE = SE_data[:, 0]

gamma_list = np.linspace(gamma_min, gamma_max, n_gammas)

indices = np.searchsorted(gammas_SE, gamma_list)
gamma_list = SE_data[indices, 0]
reg_param_list = SE_data[indices, -1]

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

ms = np.empty((n_gammas, 2))
qs = np.empty((n_gammas, 2))
q_latent = np.empty((n_gammas, 2))
q_feature = np.empty((n_gammas, 2))
Ps = np.empty((n_gammas, 2))
gen_errs = np.empty((n_gammas, 2))
adv_errs = np.empty((n_gammas, 2))
flipped_fairs = np.empty((n_gammas, 2))
misclas_fairs = np.empty((n_gammas, 2))

# do a for loop on the alpha_list and compute for each the p = d / gamma and n = alpha * d
for i, (gamma, reg_param) in enumerate(gamma_list, reg_param_list):
    p = int(d / gamma)
    n = int(alpha * d)

    print(f"p {p} d {d} n {n}")

    m_vals = []
    q_vals = []
    q_latent_vals = []
    q_feature_vals = []
    P_vals = []
    gen_err_vals = []
    adv_err_vals = []
    flip_fair_vals = []
    misc_fair_vals = []

    j = 0
    while j < reps:
        xs, ys, zs, xs_gen, ys_gen, zs_gen, wstar, F, noise, noise_gen = data_generation_hastie(
            measure_gen_probit_clasif, d, n, n_gen, (delta,), gamma, noi=True
        )

        try:
            w = find_coefficients_Logistic_adv_Linf_L2(ys, xs, 0.5 * reg_param, eps_t)
        except (ValueError, SolverError) as e:
            print(
                f"minimization didn't converge on iteration {j} for gamma {gamma:.2f}. Trying again."
            )
            continue

        m_vals.append(np.dot(wstar, F.T @ w) / (p * np.sqrt(gamma)))
        q_vals.append(np.dot(F.T @ w, F.T @ w) / p + np.dot(w, w) / p)
        q_latent_vals.append(np.dot(F.T @ w, F.T @ w) / d)
        q_feature_vals.append(np.dot(w, w) / p)
        P_vals.append(np.mean(np.abs(w)))

        yhat_gen = np.sign(np.dot(xs_gen, w))

        gen_err_vals.append(generalisation_error_classification(ys_gen, xs_gen, w, wstar))
        # adv_err_vals.append(adversarial_error_data(ys_gen, xs_gen, w, wstar, eps_test, pstar))

        adv_pert = find_adversarial_error_rf(
            ys_gen, zs_gen, w, F.T, wstar, eps_test / np.sqrt(d), "inf"
        )
        adv_err = np.mean(ys_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w))
        adv_err_vals.append(adv_err)

        # calculation of flipped perturbation
        adv_perturbation = find_adversarial_perturbation_linear_rf(
            yhat_gen, zs_gen, w, F.T, wstar, eps_test / np.sqrt(d), "inf"
        )
        flipped = np.mean(
            yhat_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w)
        )
        flip_fair_vals.append(flipped)

        # calculation of perturbation
        adv_perturbation = find_adversarial_perturbation_linear_rf(
            ys_gen, zs_gen, w, F.T, wstar, eps_test / np.sqrt(d), "inf"
        )

        misclass = np.mean(ys_gen != np.sign((zs_gen + adv_perturbation) @ F.T @ w + noise_gen @ w))
        misc_fair_vals.append(misclass)

        print(f"repetition {j} for gamma {gamma:.2f} done.")
        j += 1

    ms[i, 0], ms[i, 1] = np.mean(m_vals), np.std(m_vals)
    qs[i, 0], qs[i, 1] = np.mean(q_vals), np.std(q_vals)
    q_latent[i, 0], q_latent[i, 1] = np.mean(q_latent_vals), np.std(q_latent_vals)
    q_feature[i, 0], q_feature[i, 1] = np.mean(q_feature_vals), np.std(q_feature_vals)
    Ps[i, 0], Ps[i, 1] = np.mean(P_vals), np.std(P_vals)
    gen_errs[i, 0], gen_errs[i, 1] = np.mean(gen_err_vals), np.std(gen_err_vals)
    adv_errs[i, 0], adv_errs[i, 1] = np.mean(adv_err_vals), np.std(adv_err_vals)
    flipped_fairs[i, 0], flipped_fairs[i, 1] = np.mean(flip_fair_vals), np.std(flip_fair_vals)
    misclas_fairs[i, 0], misclas_fairs[i, 1] = np.mean(misc_fair_vals), np.std(misc_fair_vals)

    print(f"gamma {gamma:.2f} done.")

np.savetxt(
    os.path.join(data_folder, file_name_adverr),
    np.column_stack(
        (
            gamma_list,
            ms[:, 0],
            ms[:, 1],
            qs[:, 0],
            qs[:, 1],
            q_latent[:, 0],
            q_latent[:, 1],
            q_feature[:, 0],
            q_feature[:, 1],
            Ps[:, 0],
            Ps[:, 1],
            gen_errs[:, 0],
            gen_errs[:, 1],
            adv_errs[:, 0],
            adv_errs[:, 1],
            flipped_fairs[:, 0],
            flipped_fairs[:, 1],
            misclas_fairs[:, 0],
            misclas_fairs[:, 1],
            reg_param_list,
        )
    ),
    delimiter=",",
    header="gamma,m_mean,m_std,q_mean,q_std,q_latent_mean,q_latent_std,q_feature_mean,q_feature_std,P_mean,P_std,gen_err_mean,gen_err_std,adv_err_mean,adv_err_std,flipped_fair_mean,flipped_fair_std,misclas_fair_mean,misclas_fair_std,reg_param",
)

print("data saved for misclass.")
