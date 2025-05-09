import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfc
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic_adv_Linf_L2,
    find_coefficients_Logistic_adv_Linf_L1,
)
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_direct_space_noteacher,
    find_adversarial_perturbation_direct_space,
)
from linear_regression.data.generation import data_generation, measure_gen_no_noise_clasif
from linear_regression.erm.metrics import generalisation_error_classification
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic_adv_Linf_L2,
    find_coefficients_Logistic_adv_Linf_L1,
    find_coefficients_Logistic,
)
from linear_regression.erm.adversarial_perturbation_finders import (
    find_adversarial_perturbation_linear_rf,
    find_adversarial_error_rf,
)
from tqdm.auto import tqdm
import os
import sys
from cvxpy.error import SolverError

if len(sys.argv) > 1:
    eps_min, eps_max, n_epss, alpha, eps_training, reg_param, reg, pstar, d = (
        float(sys.argv[1]),
        float(sys.argv[2]),
        int(sys.argv[3]),
        float(sys.argv[4]),
        float(sys.argv[5]),
        float(sys.argv[6]),
        float(sys.argv[7]),
        float(sys.argv[8]),
        int(sys.argv[9]),
    )
else:
    eps_min, eps_max, n_epss, alpha, eps_training, reg_param, reg, pstar, d = (
        0.1,
        10.0,
        15,
        1.5,
        0.0,
        1e-2,
        2.0,
        2.0,
        500,
    )

reps = 10
n_gen = 1000

if pstar == 2.0:
    adv_geometry = 2.0
if pstar == 1.0:
    adv_geometry = "inf"

data_folder = "./data/direct_space_model_training"

file_name = f"ERM_sweep_eps_direct_model_d_{d:d}_alpha_{alpha:.1f}_reps_{reps:d}_epss_{eps_min:.1f}_{eps_max:.1f}_{n_epss:d}_pstar_{pstar:.1f}_reg_{reg:.1f}_regparam_{reg_param:.1e}.csv"

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

epss = np.logspace(np.log10(eps_min), np.log10(eps_max), n_epss)

n = int(d * alpha)

vals_misclass = np.empty((reps, len(epss)))
vals_flipped = np.empty((reps, len(epss)))
vals_adverr = np.empty((reps, len(epss)))
vals_bound = np.empty((reps, len(epss)))

estim_vals_m = np.empty((reps,))
estim_vals_q = np.empty((reps,))
estim_vals_rho = np.empty((reps,))
estim_vals_P = np.empty((reps,))

j = 0
while j < reps:
    print(f"Calculating repetition: {j + 1} / {reps}")
    xs, ys, xs_gen, ys_gen, wstar = data_generation(
        measure_gen_no_noise_clasif, d, max(n, 1), 1000, ()
    )

    try:
        w = find_coefficients_Logistic(ys, xs, reg_param)
    except (ValueError, UserWarning, SolverError) as e:
        print("Error in finding coefficients:", e)
        continue

    estim_vals_rho[j] = np.sum(wstar**2) / d
    estim_vals_m[j] = np.dot(wstar, w) / d
    estim_vals_q[j] = np.dot(w, w) / d
    estim_vals_P[j] = np.sum(np.abs(w) ** pstar) / d

    yhat = np.repeat(np.sign(xs @ w).reshape(-1, 1), d, axis=1)

    yhat_gen = np.sign(xs_gen @ w)

    i = 0
    while i < len(epss):
        eps_i = epss[i]

        if pstar == 2.0:
            eps_i = eps_i
        elif pstar == 1.0:
            eps_i = eps_i / np.sqrt(d)

        # flipped error
        try:
            adv_perturbation = find_adversarial_perturbation_direct_space(
                yhat_gen, xs_gen, w, wstar, eps_i, adv_geometry
            )
        except (ValueError, UserWarning, SolverError) as e:
            print("Error in finding adversarial perturbation:", e)
            break

        flipped = np.mean(yhat_gen != np.sign((xs_gen + adv_perturbation) @ w))
        vals_flipped[j, i] = flipped

        # misclassification error
        try:
            adv_perturbation = find_adversarial_perturbation_direct_space(
                ys_gen, xs_gen, w, wstar, eps_i, adv_geometry
            )
        except (ValueError, UserWarning, SolverError) as e:
            print("Error in finding adversarial perturbation:", e)
            break

        misclass = np.mean(ys_gen != np.sign((xs_gen + adv_perturbation) @ w))
        vals_misclass[j, i] = misclass

        # adversarial error
        try:
            adv_perturbation = find_adversarial_perturbation_direct_space_noteacher(
                ys_gen, xs_gen, w, wstar, eps_i, adv_geometry
            )
        except (ValueError, UserWarning, SolverError) as e:
            print("Error in finding adversarial perturbation:", e)
            break

        adv_err = np.mean(ys_gen != np.sign((xs_gen + adv_perturbation) @ w))
        vals_adverr[j, i] = adv_err

        # bound error
        try:
            adv_perturbation = find_adversarial_perturbation_direct_space(
                ys_gen, xs_gen, w, wstar, eps_i, adv_geometry
            )
        except (ValueError, UserWarning, SolverError) as e:
            print("Error in finding adversarial perturbation:", e)
            break
        bound_err = np.mean(
            (ys_gen != np.sign((xs_gen + adv_perturbation) @ w)) * (ys_gen == yhat_gen)
        )
        vals_bound[j, i] = bound_err
        i += 1
    else:
        j += 1

mean_m, std_m = np.mean(estim_vals_m), np.std(estim_vals_m)
mean_q, std_q = np.mean(estim_vals_q), np.std(estim_vals_q)
mean_P, std_P = np.mean(estim_vals_P), np.std(estim_vals_P)
mean_rho, std_rho = np.mean(estim_vals_rho), np.std(estim_vals_rho)
mean_misclass, std_misclass = np.mean(vals_misclass, axis=0), np.std(vals_misclass, axis=0)
mean_flipped, std_flipped = np.mean(vals_flipped, axis=0), np.std(vals_flipped, axis=0)
mean_adverr, std_adverr = np.mean(vals_adverr, axis=0), np.std(vals_adverr, axis=0)
mean_bound, std_bound = np.mean(vals_bound, axis=0), np.std(vals_bound, axis=0)

# Create a structured array for CSV output
data_dict = {
    "eps": epss,
    "mean_m": np.full_like(epss, mean_m),
    "std_m": np.full_like(epss, std_m),
    "mean_q": np.full_like(epss, mean_q),
    "std_q": np.full_like(epss, std_q),
    "mean_P": np.full_like(epss, mean_P),
    "std_P": np.full_like(epss, std_P),
    "mean_rho": np.full_like(epss, mean_rho),
    "std_rho": np.full_like(epss, std_rho),
    "mean_misclass": mean_misclass,
    "std_misclass": std_misclass,
    "mean_flipped": mean_flipped,
    "std_flipped": std_flipped,
    "mean_adverr": mean_adverr,
    "std_adverr": std_adverr,
    "mean_bound": mean_bound,
    "std_bound": std_bound,
}

# Convert to structured array for saving
header = ",".join(data_dict.keys())
data_array = np.column_stack([data_dict[key] for key in data_dict.keys()])

# Save to CSV
data_file = os.path.join(data_folder, file_name)
np.savetxt(data_file, data_array, delimiter=",", header=header, comments="")
