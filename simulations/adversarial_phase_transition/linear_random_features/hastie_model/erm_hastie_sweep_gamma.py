from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic_adv,
    find_coefficients_Logistic_adv_Linf_L2,
)
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_direct_space,
)
from linear_regression.data.generation import data_generation_hastie, measure_gen_probit_clasif
from linear_regression.erm.metrics import (
    generalisation_error_classification,
    adversarial_error_data,
    percentage_flipped_labels_estim,
    percentage_error_from_true,
)
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
    gamma_min, gamma_max, n_gammas = 0.5, 5.0, 20
    alpha = 0.5
    eps_t = 0.1
    delta = 0.0
    reg_param = 1e-3
    d = 300
    
reps = 20
n_gen = 1000

pstar = 1.0

data_folder = f"./data/hastie_model_training"
file_name = f"ERM_training_alpha_{alpha:.2f}_gammas_{gamma_min:.1f}_{gamma_max:.1f}_{n_gammas:d}_delta_{delta:.2f}_d_{d:d}_reps_{reps:d}_eps_{eps_t:.2f}_reg_param_{reg_param:.1e}_pstar_{pstar:.1f}.csv"

gamma_list = np.linspace(gamma_min, gamma_max, n_gammas)

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

ms = np.empty((n_gammas, 2))
qs = np.empty((n_gammas, 2))
Ps = np.empty((n_gammas, 2))
gen_errs = np.empty((n_gammas, 2))
adv_errs = np.empty((n_gammas, 2))
flipped_fairs = np.empty((n_gammas, 2))
misclas_fairs = np.empty((n_gammas, 2))

# do a for loop on the alpha_list and compute for each the p = d / gamma and n = alpha * d
for i, gamma in enumerate(gamma_list):
    p = int(d / gamma)
    n = int(alpha * d)

    m_vals = []
    q_vals = []
    P_vals = []
    gen_err_vals = []
    adv_err_vals = []
    flip_fair_vals = []
    misc_fair_vals = []

    j = 0
    while j < reps:
        xs, ys, zs, xs_gen, ys_gen, zs_gen, wstar, F = data_generation_hastie(
            measure_gen_probit_clasif, d, n, n_gen, (delta,), gamma
        )

        try:
            # w = find_coefficients_Logistic_adv(ys, xs, 0.5 * reg_param, eps_t, 2.0, pstar)
            w = find_coefficients_Logistic_adv_Linf_L2(ys, xs, 0.5 * reg_param, eps_t)
        except ValueError as e:
            print(
                f"minimization didn't converge on iteration {j} for gamma {gamma:.2f}. Trying again."
            )
            continue

        m_vals.append(np.dot(wstar, F.T @ w) / (p * np.sqrt(gamma)))
        q_vals.append(np.dot(F.T @ w, F.T @ w) / p + np.dot(w, w) / p)
        P_vals.append(np.sum(np.abs(w) ** pstar) / p)

        yhat_gen = np.sign(np.dot(xs_gen, w))

        gen_err_vals.append(generalisation_error_classification(ys_gen, xs_gen, w, wstar))
        adv_err_vals.append(adversarial_error_data(ys_gen, xs_gen, w, wstar, eps_t, pstar))

        # calculation of flipped perturbation
        adv_perturbation = find_adversarial_perturbation_direct_space(
            yhat_gen, xs_gen, w, F @ wstar, eps_t * p ** (-1 / 2), "inf"
        )
        flipped = percentage_flipped_labels_estim(
            yhat_gen,
            xs_gen,
            w,
            wstar,
            xs_gen + adv_perturbation,
        )
        flip_fair_vals.append(flipped)

        # calculation of perturbation
        adv_perturbation = find_adversarial_perturbation_direct_space(
            ys_gen, xs_gen, w, F @ wstar, eps_t * p ** (-1 / 2), "inf"
        )
        misclass = percentage_error_from_true(
            ys_gen,
            xs_gen,
            w,
            F @ wstar,
            xs_gen + adv_perturbation,
        )
        misc_fair_vals.append(misclass)

        print(f"repetition {j} for gamma {gamma:.2f} done.")
        j += 1

    ms[i, 0], ms[i, 1] = np.mean(m_vals), np.std(m_vals)
    qs[i, 0], qs[i, 1] = np.mean(q_vals), np.std(q_vals)
    Ps[i, 0], Ps[i, 1] = np.mean(P_vals), np.std(P_vals)
    gen_errs[i, 0], gen_errs[i, 1] = np.mean(gen_err_vals), np.std(gen_err_vals)
    adv_errs[i, 0], adv_errs[i, 1] = np.mean(adv_err_vals), np.std(adv_err_vals)
    flipped_fairs[i, 0], flipped_fairs[i, 1] = np.mean(flip_fair_vals), np.std(flip_fair_vals)
    misclas_fairs[i, 0], misclas_fairs[i, 1] = np.mean(misc_fair_vals), np.std(misc_fair_vals)

    print(f"gamma {gamma:.2f} done.")

np.savetxt(
    os.path.join(data_folder, file_name),
    np.column_stack(
        (
            gamma_list,
            ms[:, 0],
            ms[:, 1],
            qs[:, 0],
            qs[:, 1],
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
    header="gamma,m-mean,m-std,q-mean,q-std,P-mean,P-std,gen_err-mean,gen_err-std,adv_err-mean,adv_err-std,flipped_fair-mean,flipped_fair-std,misclass_fair-mean,misclass_fair-std",
)

print("data saved.")
