import matplotlib.pyplot as plt
import numpy as np
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic_adv,
    find_coefficients_Logistic_adv_Linf_L1,
)
from linear_regression.data.generation import data_generation, measure_gen_no_noise_clasif
from linear_regression.erm.metrics import (
    estimation_error_data,
    generalisation_error_classification,
    adversarial_error_data,
)
import pickle
from tqdm.auto import tqdm
from mpi4py import MPI
import os

alpha_min, alpha_max, n_alpha_pts = 0.01, 1, 22
reg_orders = [
    1,
    2,
    3,
]
eps_t = 0.2
eps_g = 0.2
pstar = 1

d = 300
reps = 10
n_gen = 1000

data_folder = "./data/ERM_sweep_opt_lambda"

file_name_save = f"ERM_data_optimal_lambda_pstar_{pstar:d}_reg_order_{{:d}}_alpha_{alpha_min:.3f}_{alpha_max:.3f}_{n_alpha_pts:d}_dim_{d:d}_reps_{reps:d}_eps_{eps_t:.1e}.csv"
file_name_SE = f"SE_alpha_sweep_optimal_lambda_pstar_{pstar:d}_reg_order_{{:d}}_alpha_{alpha_min:.3f}_{alpha_max:.3f}_eps_{eps_t:.1e}.csv"

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

assert len(reg_orders) >= size

reg_order = reg_orders[rank]

# open the file with the SE data in it and get the optimal lambda for every alpha of the file
with open(os.path.join(data_folder, file_name_SE.format(reg_order)), "rb") as f:
    data_SE = np.loadtxt(f, delimiter=",", skiprows=1)

alphas_SE = data_SE[:, 0]
reg_param_opts = data_SE[:, 9]

print(f"process {rank}/{size} loaded data correctly")

alphas_EPR_test = np.linspace(alpha_min, alpha_max, n_alpha_pts)

alphas = []

q_mean = []
q_std = []

m_mean = []
m_std = []

p_mean = []
p_std = []

train_error_mean = []
train_error_std = []

gen_error_mean = []
gen_error_std = []

estim_errors_mean = []
estim_errors_std = []

adversarial_errors_mean = []
adversarial_errors_std = []

for j, alpha_test in enumerate(alphas_EPR_test):
    # find the closest alpha in the SE data
    idx = np.argmin(np.abs(alphas_SE - alpha_test))
    reg_param = reg_param_opts[idx]
    alpha = alphas_SE[idx]

    n = int(alpha * d)

    print(
        f"process {rank}/{size} running reg_order = {reg_order}, reg_param = {reg_param:.1e}, alpha = {alpha:.4f} (= {n:d} samples / {d:d} features)"
    )

    tmp_estim_errors = []
    tmp_train_errors = []
    tmp_gen_errors = []
    tmp_adversarial_errors = []
    tmp_qs = []
    tmp_ms = []
    tmp_ps = []

    iter = 0
    pbar = tqdm(total=reps)
    while iter < reps:
        xs_train, ys_train, xs_gen, ys_gen, wstar = data_generation(
            measure_gen_no_noise_clasif, d, n, n_gen, tuple()
        )

        try:
            if reg_order == 1:
                w = find_coefficients_Logistic_adv_Linf_L1(ys_train, xs_train, reg_param, eps_t)
            else:
                w = find_coefficients_Logistic_adv(
                    ys_train, xs_train, reg_param, eps_t, reg_order, pstar, wstar
                )
        except ValueError as e:
            print(e)
            continue

        tmp_qs.append(np.sum(w**2) / d)
        tmp_ms.append(np.dot(wstar, w) / d)
        tmp_ps.append(np.sum(np.abs(w) ** pstar) / d)
        tmp_estim_errors.append(estimation_error_data(ys_gen, xs_gen, w, wstar))
        tmp_train_errors.append(adversarial_error_data(ys_train, xs_train, w, wstar, eps_t, pstar))
        tmp_gen_errors.append(generalisation_error_classification(ys_gen, xs_gen, w, wstar))
        tmp_adversarial_errors.append(
            adversarial_error_data(ys_gen, xs_gen, w, wstar, eps_g, pstar)
        )

        del w
        del xs_gen
        del ys_gen
        del xs_train
        del ys_train
        del wstar

        iter += 1
        pbar.update(1)

    pbar.close()

    estim_errors_mean.append(np.mean(tmp_estim_errors))
    estim_errors_std.append(np.std(tmp_estim_errors))

    q_mean.append(np.mean(tmp_qs))
    q_std.append(np.std(tmp_qs))

    m_mean.append(np.mean(tmp_ms))
    m_std.append(np.std(tmp_ms))

    p_mean.append(np.mean(tmp_ps))
    p_std.append(np.std(tmp_ps))

    train_error_mean.append(np.mean(tmp_train_errors))
    train_error_std.append(np.std(tmp_train_errors))

    gen_error_mean.append(np.mean(tmp_gen_errors))
    gen_error_std.append(np.std(tmp_gen_errors))

    adversarial_errors_mean.append(np.mean(tmp_adversarial_errors))
    adversarial_errors_std.append(np.std(tmp_adversarial_errors))

data_dict = {
    "alphas": alphas,
    "q_mean": q_mean,
    "q_std": q_std,
    "m_mean": m_mean,
    "m_std": m_std,
    "p_mean": p_mean,
    "p_std": p_std,
    "train_error_mean": train_error_mean,
    "train_error_std": train_error_std,
    "gen_error_mean": gen_error_mean,
    "gen_error_std": gen_error_std,
    "estim_error_mean": estim_errors_mean,
    "estim_error_std": estim_errors_std,
    "adversarial_error_mean": adversarial_errors_mean,
    "adversarial_error_std": adversarial_errors_std,
}

with open(os.path.join(data_folder, file_name_save.format(reg_order)), "wb") as f:
    pickle.dump(data_dict, f)
