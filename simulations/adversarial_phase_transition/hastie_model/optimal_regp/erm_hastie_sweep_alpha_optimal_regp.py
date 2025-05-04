from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic_adv,
    find_coefficients_Logistic_adv_Linf_L2,
)
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_linear_rf,
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
    alpha_min, alpha_max, n_alphas, d, gamma, eps_t, delta, reg_param = (
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
    alpha_min, alpha_max, n_alphas = 0.3, 3.0, 10
    gamma = 2.0
    eps_t = 0.1
    delta = 0.0
    reg_param = 1e-2
    d = 500

reps = 10
n_gen = 1000

pstar = 1.0
eps_test = 1.0

data_folder = f"./data/hastie_model_training"
file_name = f"ERM_training_gamma_{gamma:.2f}_alphas_{alpha_min:.1f}_{alpha_max:.1f}_{n_alphas:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_eps_{eps_t:.2f}_reg_param_{reg_param:.1e}_pstar_{pstar:.1f}.csv"

alpha_list = np.linspace(alpha_min, alpha_max, n_alphas)

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

ms = np.empty((n_alphas, 2))
qs = np.empty((n_alphas, 2))
q_latent = np.empty((n_alphas, 2))
q_feature = np.empty((n_alphas, 2))
Ps = np.empty((n_alphas, 2))
gen_errs = np.empty((n_alphas, 2))
adv_errs = np.empty((n_alphas, 2))
flipped_fairs = np.empty((n_alphas, 2))
misclas_fairs = np.empty((n_alphas, 2))

# do a for loop on the alpha_list and compute for each the p = d / gamma and n = alpha * d
for i, alpha in enumerate(alpha_list):
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
                f"minimization didn't converge on iteration {j} for alpha {alpha:.2f}. Trying again."
            )
            continue

        m_vals.append(np.dot(wstar, F.T @ w) / (p * np.sqrt(gamma)))
        q_vals.append(np.dot(F.T @ w, F.T @ w) / p + np.dot(w, w) / p)
        P_vals.append(np.mean(np.abs(w)))
        q_latent_vals.append(np.dot(F.T @ w, F.T @ w) / d)
        q_feature_vals.append(np.dot(w, w) / p)

        yhat_gen = np.sign(np.dot(xs_gen, w))

        gen_err_vals.append(generalisation_error_classification(ys_gen, xs_gen, w, wstar))
        adv_err_vals.append(adversarial_error_data(ys_gen, xs_gen, w, wstar, eps_test, pstar))

        # calculation of flipped perturbation
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

        print(f"repetition {j} for alpha {alpha:.2f} done.")
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

    print(f"alpha {alpha:.2f} done.")

np.savetxt(
    os.path.join(data_folder, file_name),
    np.column_stack(
        (
            alpha_list,
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
        )
    ),
    delimiter=",",
    header="alpha,m_mean,m_std,q_mean,q_std,q_latent_mean,q_latent_std,q_feature_mean,q_feature_std,P_mean,P_std,gen_err_mean,gen_err_std,adv_err_mean,adv_err_std,flipped_fair_mean,flipped_fair_std,misclas_fair_mean,misclas_fair_std",
)

print("data saved.")
