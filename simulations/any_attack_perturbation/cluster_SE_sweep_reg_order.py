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

reg_order_min, reg_order_max, n_reg_orders = 1, 3, 40
epss = [0.1, 0.2, 0.3]
alphas = [0.01, 0.1, 1]
reg_params = [1e-2, 1e-3]
pstars = [1, 2, 3]

pairs = list(product(alphas, reg_params, epss, pstars))

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

assert len(pairs) >= size

alpha, reg_param, eps_t, pstar = pairs[rank]
eps_g = eps_t

data_folder_SE = "./data/SE_reg_order_sweep"

if not exists(data_folder_SE):
    os.makedirs(data_folder_SE)

file_name = f"SE_eps_sweep_pstar_{pstar}_reg_order_{reg_order_min:.1f}_{reg_order_max:.1f}_alpha_{alpha:.3f}_reg_param_{reg_param:.1e}_eps_{eps_t:.2f}.csv"

reg_orders = np.linspace(reg_order_min, reg_order_max, n_reg_orders)

ms_found = np.empty((n_reg_orders,))
qs_found = np.empty((n_reg_orders,))
Vs_found = np.empty((n_reg_orders,))
Ps_found = np.empty((n_reg_orders,))

mhats_found = np.empty((n_reg_orders,))
qhats_found = np.empty((n_reg_orders,))
Vhats_found = np.empty((n_reg_orders,))
Phats_found = np.empty((n_reg_orders,))

estim_errors_se = np.empty((n_reg_orders,))
adversarial_errors_found = np.empty((n_reg_orders,))
gen_errors_se = np.empty((n_reg_orders,))

m, q, V, P = (0.348751, 9.11468, 270.038, 0.615440)
initial_condition = (m, q, V, P)

print(
    f"Starting the sweep for alpha = {alpha}, reg_param = {reg_param}, eps = {eps_t}, pstar = {pstar}"
)

for jprime, reg_order in enumerate(reg_orders):
    j = jprime

    f_kwargs = {"reg_param": reg_param, "reg_order": reg_order, "pstar": pstar}
    f_hat_kwargs = {"alpha": alpha, "eps_t": eps_t}

    ms_found[j], qs_found[j], Vs_found[j], Ps_found[j] = fixed_point_finder(
        f_Lr_regularisation_Lpstar_attack,
        f_hat_Logistic_no_noise_Linf_adv_classif,
        initial_condition,
        f_kwargs,
        f_hat_kwargs,
        abs_tol=1e-6,
        min_iter=10,
        args_update_function=(0.4,),
        max_iter=10_000_000,
    )

    initial_condition = (ms_found[j], qs_found[j], Vs_found[j], Ps_found[j])

    estim_errors_se[j] = 1 - 2 * ms_found[j] + qs_found[j]
    adversarial_errors_found[j] = classification_adversarial_error(
        ms_found[j], qs_found[j], Ps_found[j], eps_g, pstar
    )
    gen_errors_se[j] = np.arccos(ms_found[j] / np.sqrt(qs_found[j])) / np.pi

print(
    f"Starting the sweep for alpha = {alpha}, reg_param = {reg_param}, eps = {eps_t}, pstar = {pstar}"
)

# Save the data
data = {
    "reg_order": reg_orders,
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

with open(join(data_folder_SE, file_name), "wb") as f:
    data_array = np.column_stack([data[key] for key in data.keys()])
    header = ",".join(data.keys())
    np.savetxt(
        join(data_folder_SE),
        data_array,
        header=header,
        delimiter=",",
        comments="",
    )
