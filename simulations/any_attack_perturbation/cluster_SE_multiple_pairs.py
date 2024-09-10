import matplotlib.pyplot as plt
import numpy as np
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.aux_functions.misc import classification_adversarial_error
from linear_regression.fixed_point_equations.regularisation.pstar_attacks_Lr_reg import (
    f_Lr_regularisation_Lpstar_attack,
)
from linear_regression.fixed_point_equations.classification.Adv_train_p_norm import (
    f_hat_Logistic_no_noise_Linf_adv_classif,
)
from os.path import join, exists
from mpi4py import MPI
from itertools import product
import os

alpha_min, alpha_max, n_alpha_pts = 0.001, 100, 200
reg_orders = [
    1,
    2,
    3,
    4,
]
epss = [0.1, 0.2, 0.3]
reg_params = [1e-1, 1e-2, 1e-3]
pstars = [1, 2, 3]

pairs = list(product(epss, reg_params, reg_orders, pstars))

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

assert len(pairs) >= size

eps_t, reg_param, reg_order, pstar = pairs[rank]
eps_g = eps_t

data_folder_SE = "./data/SE_any_norm"

if not exists(data_folder_SE):
    os.makedirs(data_folder_SE)

file_name = f"SE_data_pstar_{pstar}_reg_order_{{}}_pstar_{pstar}_alpha_{alpha_min:.3f}_{alpha_max:.3f}_reg_param_{reg_param:.1e}_eps_t_g_{eps_t:.1e}_{eps_g:.1e}.csv"

alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)

ms_found = np.empty((n_alpha_pts,))
qs_found = np.empty((n_alpha_pts,))
Vs_found = np.empty((n_alpha_pts,))
Ps_found = np.empty((n_alpha_pts,))

mhats_found = np.empty((n_alpha_pts,))
qhats_found = np.empty((n_alpha_pts,))
Vhats_found = np.empty((n_alpha_pts,))
Phats_found = np.empty((n_alpha_pts,))

estim_errors_se = np.empty((n_alpha_pts,))
adversarial_errors_found = np.empty((n_alpha_pts,))
gen_errors_se = np.empty((n_alpha_pts,))

if reg_order == 1:
    m, q, V, P = (0.348751, 9.11468, 270.038, 0.615440)
    initial_condition = (m, q, V, P)
elif reg_order == 2:
    m, q, V, P = (6.766e-01, 5.780e00, 4.442e03, 1.780e00)
    initial_condition = (m, q, V, P)
elif reg_order == 3:
    m, q, V, P = (3.873e-01, 2.571e00, 1.710e03, 1.446e00)
    initial_condition = (m, q, V, P)
elif reg_order == 4:
    m, q, V, P = (3.417e-01, 2.140e00, 8.220e02, 1.371e00)
    initial_condition = (m, q, V, P)
else:
    m, q, V, P = (0.6604, 13.959, 25767.13, 1.364)
    initial_condition = (m, q, V, P)

for jprime, alpha in enumerate(alphas):
    j = jprime
    # print("\033[91m" + f"SE {reg_order = }, {alpha = :.3e}" + "\033[0m")

    f_kwargs = {"reg_param": reg_param, "reg_order": reg_order, "pstar": 1}
    f_hat_kwargs = {"alpha": alpha, "eps_t": eps_t}

    ms_found[j], qs_found[j], Vs_found[j], Ps_found[j] = fixed_point_finder(
        f_Lr_regularisation_Lpstar_attack,
        f_hat_Logistic_no_noise_Linf_adv_classif,
        initial_condition,
        f_kwargs,
        f_hat_kwargs,
        abs_tol=1e-7,
        min_iter=10,
        args_update_function=(0.4,),
        max_iter=1_000_000,
    )

    initial_condition = (ms_found[j], qs_found[j], Vs_found[j], Ps_found[j])

    estim_errors_se[j] = 1 - 2 * ms_found[j] + qs_found[j]
    adversarial_errors_found[j] = classification_adversarial_error(
        ms_found[j], qs_found[j], Ps_found[j], eps_g, pstar
    )
    gen_errors_se[j] = np.arccos(ms_found[j] / np.sqrt(qs_found[j])) / np.pi

data = {
    "alpha": alphas,
    "m": ms_found,
    "q": qs_found,
    "V": Vs_found,
    "P": Ps_found,
    "mhat": mhats_found,
    "qhat": qhats_found,
    "Vhat": Vhats_found,
    "Phat": Phats_found,
    "estim_error": estim_errors_se,
    "adversarial_error": adversarial_errors_found,
    "generalisation_error": gen_errors_se,
}

with open(join(data_folder_SE, file_name.format(reg_order)), "wb") as f:
    data_array = np.column_stack([data[key] for key in data.keys()])
    header = ",".join(data.keys())
    np.savetxt(
        join(data_folder_SE, file_name.format(reg_order)),
        data_array,
        header=header,
        delimiter=",",
        comments="",
    )
