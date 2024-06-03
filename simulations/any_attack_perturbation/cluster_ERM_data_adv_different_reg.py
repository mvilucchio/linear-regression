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

alpha_min, alpha_max, n_alpha_pts = 0.1, 100, 22
reg_orders = [
    1,
    2,
    3,
    4,
]
eps_t = 0.1
eps_g = 0.1
reg_params = [1e-1, 1e-2, 1e-3, 1e-4]
pstar = 1.0

d = 100
reps = 10
n_gen = 1000

data_folder = "./data"
file_name = f"ERM_data_Linf_reg_order_{{:d}}_alpha_{alpha_min:.3f}_{alpha_max:.3f}_{n_alpha_pts:d}_dim_{d:d}_reps_{reps:d}_reg_param_{{:.1e}}_eps_t_g_{eps_t:.1e}_{eps_g:.1e}.pkl"

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

assert len(reg_orders) >= size

reg_order = reg_orders[rank]

for reg_param in reg_params:
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)

    q_mean = np.empty_like(alphas)
    q_std = np.empty_like(alphas)

    m_mean = np.empty_like(alphas)
    m_std = np.empty_like(alphas)

    p_mean = np.empty_like(alphas)
    p_std = np.empty_like(alphas)

    train_error_mean = np.empty_like(alphas)
    train_error_std = np.empty_like(alphas)

    gen_error_mean = np.empty_like(alphas)
    gen_error_std = np.empty_like(alphas)

    estim_errors_mean = np.empty_like(alphas)
    estim_errors_std = np.empty_like(alphas)

    adversarial_errors_mean = np.empty_like(alphas)
    adversarial_errors_std = np.empty_like(alphas)

    for j, alpha in enumerate(alphas):
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
            tmp_train_errors.append(
                adversarial_error_data(ys_train, xs_train, w, wstar, eps_t, pstar)
            )
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

        estim_errors_mean[j] = np.mean(tmp_estim_errors)
        estim_errors_std[j] = np.std(tmp_estim_errors)

        q_mean[j] = np.mean(tmp_qs)
        q_std[j] = np.std(tmp_qs)

        m_mean[j] = np.mean(tmp_ms)
        m_std[j] = np.std(tmp_ms)

        p_mean[j] = np.mean(tmp_ps)
        p_std[j] = np.std(tmp_ps)

        train_error_mean[j] = np.mean(tmp_train_errors)
        train_error_std[j] = np.std(tmp_train_errors)

        gen_error_mean[j] = np.mean(tmp_gen_errors)
        gen_error_std[j] = np.std(tmp_gen_errors)

        adversarial_errors_mean[j] = np.mean(tmp_adversarial_errors)
        adversarial_errors_std[j] = np.std(tmp_adversarial_errors)

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

    with open(os.path.join(data_folder, file_name.format(reg_order, reg_param)), "wb") as f:
        pickle.dump(data_dict, f)
