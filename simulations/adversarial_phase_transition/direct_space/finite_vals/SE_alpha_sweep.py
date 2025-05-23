import numpy as np
import matplotlib.pyplot as plt
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.aux_functions.misc import (
    classification_adversarial_error,
    misclassification_error_direct_space,
    flipped_error_direct_space,
    boundary_error_direct_space,
)
from linear_regression.fixed_point_equations.regularisation.pstar_attacks_Lr_reg import (
    f_Lr_regularisation_Lpstar_attack,
)
from linear_regression.fixed_point_equations.classification.Adv_train_p_norm import (
    f_hat_Logistic_no_noise_Linf_adv_classif,
)
from linear_regression.fixed_point_equations.classification.Adversarial_Logistic_loss import (
    f_hat_Logistic_no_noise_classif,
)
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic_adv_Linf_L2,
    find_coefficients_Logistic_adv_Linf_L1,
)
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_direct_space_noteacher,
    find_adversarial_perturbation_direct_space,
)
from linear_regression.data.generation import data_generation, measure_gen_probit_clasif
from linear_regression.erm.metrics import generalisation_error_classification
from cvxpy.error import SolverError
from os.path import exists
import os
import sys

if len(sys.argv) > 1:
    alpha_min, alpha_max, n_alphas, pstar, reg_p = (
        float(sys.argv[1]),
        float(sys.argv[2]),
        int(sys.argv[3]),
        float(sys.argv[4]),
        float(sys.argv[5]),
        sys.argv[6],
    )
else:
    alpha_min, alpha_max, n_alphas = (0.2, 2.0, 20)
    pstar = 2.0
    reg_p = 2.0

eps_test = 1.0
reg_param = 0.5


def compute_theory_overlaps(reg_param, alpha, init_cond):
    if pstar == 2.0:
        f_hat = f_hat_Logistic_no_noise_classif
        f_hat_kwargs = {"alpha": alpha, "eps_t": 0.0}
    else:
        f_hat = f_hat_Logistic_no_noise_Linf_adv_classif
        f_hat_kwargs = {"alpha": alpha, "eps_t": 0.0}

    f_kwargs = {"reg_param": reg_param, "reg_order": reg_p, "pstar": pstar}

    m_se, q_se, V_se, P_se = fixed_point_finder(
        f_Lr_regularisation_Lpstar_attack,
        f_hat,
        init_cond,
        f_kwargs,
        f_hat_kwargs,
        abs_tol=1e-6,
    )

    return m_se, q_se, V_se, P_se


if pstar == 2.0:
    adv_geometry = 2.0
elif pstar == 1.0:
    adv_geometry = "inf"

reps = 10
d = 500
n_gen = 1000
delta = 0.0
n_alphas_erm = 5


def perform_sweep():
    alphas = np.linspace(alpha_min, alpha_max, n_alphas)

    ms_found = np.empty((n_alphas,))
    qs_found = np.empty((n_alphas,))
    Vs_found = np.empty((n_alphas,))
    Ps_found = np.empty((n_alphas,))
    estim_errors_se = np.empty((n_alphas,))
    adversarial_errors_found = np.empty((n_alphas,))
    gen_errors_se = np.empty((n_alphas,))
    flipped_fairs_se = np.empty((n_alphas,))
    misclas_fairs_se = np.empty((n_alphas,))
    bound_errs_se = np.empty((n_alphas,))

    initial_condition = (0.6, 1.6, 1.05, 1.1)

    for j, alpha in enumerate(alphas):
        print(f"Calculating alpha: {alpha:.2f} / {alpha_max:.2f}")

        ms_found[j], qs_found[j], Vs_found[j], Ps_found[j] = compute_theory_overlaps(
            reg_param, alpha, initial_condition
        )

        initial_condition = (ms_found[j], qs_found[j], Vs_found[j], Ps_found[j])

        estim_errors_se[j] = 1 - 2 * ms_found[j] + qs_found[j]

        adversarial_errors_found[j] = classification_adversarial_error(
            ms_found[j], qs_found[j], Ps_found[j], eps_test, pstar
        )

        bound_errs_se[j] = boundary_error_direct_space(
            ms_found[j], qs_found[j], Ps_found[j], eps_test, pstar
        )

        gen_errors_se[j] = np.arccos(ms_found[j] / np.sqrt(qs_found[j])) / np.pi

        flipped_fairs_se[j] = flipped_error_direct_space(
            ms_found[j], qs_found[j], Ps_found[j], eps_test, pstar
        )
        misclas_fairs_se[j] = misclassification_error_direct_space(
            ms_found[j], qs_found[j], Ps_found[j], eps_test, pstar
        )

    plt.subplot(3, 2, 1)
    plt.plot(alphas, adversarial_errors_found, label="Adversarial error")

    plt.subplot(3, 2, 2)
    plt.plot(alphas, bound_errs_se, label="Bound error")

    plt.subplot(3, 2, 3)
    plt.plot(alphas, misclas_fairs_se, label="Misclassification fairness")

    plt.xlabel("Alpha")

    plt.subplot(3, 2, 4)
    plt.plot(alphas, ms_found, label="m")

    plt.subplot(3, 2, 5)
    plt.plot(alphas, qs_found, label="q")

    plt.subplot(3, 2, 6)
    plt.plot(alphas, Ps_found, label="V")
    plt.xlabel("Alpha")

    ms = np.empty((n_alphas_erm, 2))
    qs = np.empty((n_alphas_erm, 2))
    Ps = np.empty((n_alphas_erm, 2))
    gen_errs = np.empty((n_alphas_erm, 2))
    adv_errs = np.empty((n_alphas_erm, 2))
    flipped_fairs = np.empty((n_alphas_erm, 2))
    misclas_fairs = np.empty((n_alphas_erm, 2))
    bound_errs = np.empty((n_alphas_erm, 2))

    alpha_list = np.linspace(alpha_min, alpha_max, n_alphas_erm)

    for i, alpha in enumerate(alpha_list):
        if pstar == 2.0:
            eps_test_tilde = eps_test
        elif pstar == 1.0:
            eps_test_tilde = eps_test / np.sqrt(d)
        print(f"Calculating alpha {i} / {n_alphas_erm}...")
        n = int(alpha * d)

        m_vals = []
        q_vals = []
        P_vals = []
        gen_err_vals = []
        adv_err_vals = []
        flip_fair_vals = []
        misc_fair_vals = []
        bound_err_vals = []

        j = 0
        while j < reps:
            xs, ys, xs_gen, ys_gen, wstar = data_generation(
                measure_gen_probit_clasif, d, n, n_gen, (delta,)
            )

            try:
                if pstar == 2.0:
                    w = find_coefficients_Logistic_adv_Linf_L2(ys, xs, reg_param, 0.0)
                else:
                    w = find_coefficients_Logistic_adv_Linf_L1(ys, xs, reg_param, 0.0)
            except (ValueError, SolverError) as e:
                print(
                    f"minimization didn't converge on iteration {j} for alpha {alpha:.2f}. Trying again."
                )
                continue

            m_vals.append(np.dot(wstar, w) / d)
            q_vals.append(np.dot(w, w) / d)
            P_vals.append(np.sum(np.abs(w) ** pstar) / d)

            yhat_gen = np.sign(np.dot(xs_gen, w))

            gen_err_vals.append(generalisation_error_classification(ys_gen, xs_gen, w, wstar))

            adv_perturbation = find_adversarial_perturbation_direct_space_noteacher(
                ys_gen, xs_gen, w, wstar, eps_test_tilde, adv_geometry
            )
            adv_err = np.mean(ys_gen != np.sign((xs_gen + adv_perturbation) @ w))
            adv_err_vals.append(adv_err)

            adv_perturbation = find_adversarial_perturbation_direct_space(
                ys_gen, xs_gen, w, wstar, eps_test_tilde, adv_geometry
            )
            misclass = np.mean(ys_gen != np.sign((xs_gen + adv_perturbation) @ w))
            misc_fair_vals.append(misclass)

            adv_perturbation = find_adversarial_perturbation_direct_space(
                ys_gen, xs_gen, w, wstar, eps_test_tilde, adv_geometry
            )
            bound_err = np.mean(
                (ys_gen != np.sign((xs_gen + adv_perturbation) @ w)) * (ys_gen == yhat_gen)
            )
            bound_err_vals.append(bound_err)

            print(f"repetition {j} for alpha {alpha:.2f} done.")
            j += 1

        ms[i, 0], ms[i, 1] = np.mean(m_vals), np.std(m_vals)
        qs[i, 0], qs[i, 1] = np.mean(q_vals), np.std(q_vals)
        Ps[i, 0], Ps[i, 1] = np.mean(P_vals), np.std(P_vals)
        gen_errs[i, 0], gen_errs[i, 1] = np.mean(gen_err_vals), np.std(gen_err_vals)
        adv_errs[i, 0], adv_errs[i, 1] = np.mean(adv_err_vals), np.std(adv_err_vals)
        misclas_fairs[i, 0], misclas_fairs[i, 1] = np.mean(misc_fair_vals), np.std(misc_fair_vals)
        bound_errs[i, 0], bound_errs[i, 1] = np.mean(bound_err_vals), np.std(bound_err_vals)

        print(f"alpha {alpha:.2f} done.")

    plt.subplot(3, 2, 1)
    plt.errorbar(
        alpha_list,
        adv_errs[:, 0],
        yerr=adv_errs[:, 1],
        label="Adversarial error",
        fmt="o",
        markersize=3,
    )
    plt.ylabel("Adversarial")
    plt.subplot(3, 2, 2)
    plt.errorbar(
        alpha_list,
        bound_errs[:, 0],
        yerr=bound_errs[:, 1],
        label="Bound error",
        fmt="o",
        markersize=3,
    )
    plt.ylabel("Bound")
    plt.subplot(3, 2, 3)
    plt.errorbar(
        alpha_list,
        misclas_fairs[:, 0],
        yerr=misclas_fairs[:, 1],
        label="Misclassification fairness",
        fmt="o",
        markersize=3,
    )
    plt.ylabel("Misclass")
    plt.xlabel("Alpha")

    plt.subplot(3, 2, 4)
    plt.errorbar(
        alpha_list,
        ms[:, 0],
        yerr=ms[:, 1],
        label="m",
        fmt="o",
        markersize=3,
    )
    plt.ylabel("m")
    plt.subplot(3, 2, 5)
    plt.errorbar(
        alpha_list,
        qs[:, 0],
        yerr=qs[:, 1],
        label="q",
        fmt="o",
        markersize=3,
    )
    plt.ylabel("q")
    plt.subplot(3, 2, 6)
    plt.errorbar(
        alpha_list,
        Ps[:, 0],
        yerr=Ps[:, 1],
        label="V",
        fmt="o",
        markersize=3,
    )
    plt.ylabel("P")
    plt.tight_layout()


perform_sweep()
plt.show()
