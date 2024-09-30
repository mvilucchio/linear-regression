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
from scipy.optimize import minimize

XATOL = 1e-7
FATOL = 1e-7
MIN_REG_PARAM = 1e-7

reg_order_min, reg_order_max, n_reg_orders = 1, 3, 50
epss = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
# epss = [0.1]
alpha = [0.01, 0.1, 1.0]
pstar = 1

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

# size = 1
# rank = 0

pairs = list(product(epss, alpha))

assert len(pairs) >= size

eps_t, alpha = pairs[rank]
eps_g = eps_t

data_folder_SE = "./data/SE_reg_order_sweep_optimal_lambda"

if not exists(data_folder_SE):
    os.makedirs(data_folder_SE)

file_name = f"SE_reg_order_optimal_lambda_sweep_pstar_{pstar}_reg_order_{reg_order_min:.1f}_{reg_order_max:.1f}_alpha_{alpha:.3f}_eps_{eps_t:.2f}.csv"

reg_orders = np.linspace(reg_order_min, reg_order_max, n_reg_orders)

ms_found = np.empty((n_reg_orders,))
qs_found = np.empty((n_reg_orders,))
Vs_found = np.empty((n_reg_orders,))
Ps_found = np.empty((n_reg_orders,))

mhats_found = np.empty((n_reg_orders,))
qhats_found = np.empty((n_reg_orders,))
Vhats_found = np.empty((n_reg_orders,))
Phats_found = np.empty((n_reg_orders,))

reg_param_opts = np.empty((n_reg_orders,))

estim_errors_se = np.empty((n_reg_orders,))
adversarial_errors_found = np.empty((n_reg_orders,))
gen_errors_se = np.empty((n_reg_orders,))

m, q, V, P = (0.348751, 9.11468, 270.038, 0.615440)
reg_param_init = 1e-1
initial_condition = (m, q, V, P)

for jprime, reg_order in enumerate(reversed(reg_orders)):
    # j = jprime
    j = n_reg_orders - jprime - 1

    print(f"Starting the sweep for reg_order = {reg_order}, alpha = {alpha}, eps_t = {eps_t}")

    f_kwargs = {"reg_param": reg_param_init / alpha, "reg_order": reg_order, "pstar": pstar}
    f_hat_kwargs = {"alpha": alpha, "eps_t": eps_t}

    def minimize_fun(reg_param):
        print(
            f"Testing reg_param = {reg_param}. Optimising for reg_order = {reg_order}, alpha = {alpha}, eps_t = {eps_t}"
        )
        f_kwargs.update({"reg_param": float(reg_param / alpha)})
        m, q, V, P = fixed_point_finder(
            f_Lr_regularisation_Lpstar_attack,
            f_hat_Logistic_no_noise_Linf_adv_classif,
            initial_condition,
            f_kwargs,
            f_hat_kwargs,
            abs_tol=1e-7,
            min_iter=10,
            args_update_function=(0.4,),
            max_iter=10_000_000,
        )
        return classification_adversarial_error(m, q, P, eps_g, pstar)

    bnds = [(MIN_REG_PARAM, None)]
    obj = minimize(
        minimize_fun,
        x0=reg_param_init,
        method="Nelder-Mead",
        bounds=bnds,
        options={"xatol": XATOL, "fatol": FATOL},
    )

    if obj.success:
        print(
            f"Optimisation successful for reg_order = {reg_order:.2f}, alpha = {alpha}, eps_t = {eps_t}"
        )
        fun_min_val = obj.fun
        reg_param_opt = obj.x

        f_kwargs.update({"reg_param": float(reg_param_opt / alpha)})

        ms_found[j], qs_found[j], Vs_found[j], Ps_found[j] = fixed_point_finder(
            f_Lr_regularisation_Lpstar_attack,
            f_hat_Logistic_no_noise_Linf_adv_classif,
            initial_condition,
            f_kwargs,
            f_hat_kwargs,
            abs_tol=1e-6,
            min_iter=10,
            args_update_function=(0.2,),
            max_iter=10_000_000,
        )

        reg_param_opts[j] = reg_param_opt
        reg_param_init = reg_param_opt

        initial_condition = (ms_found[j], qs_found[j], Vs_found[j], Ps_found[j])

        estim_errors_se[j] = 1 - 2 * ms_found[j] + qs_found[j]
        adversarial_errors_found[j] = classification_adversarial_error(
            ms_found[j], qs_found[j], Ps_found[j], eps_g, pstar
        )
        gen_errors_se[j] = np.arccos(ms_found[j] / np.sqrt(qs_found[j])) / np.pi

    else:
        print(f"Optimisation failed for reg_order = {reg_order}, alpha = {alpha}, eps_t = {eps_t}")

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
    "reg_param": reg_param_opts,
    "estim_error": estim_errors_se,
    "adversarial_error": adversarial_errors_found,
    "generalisation_error": gen_errors_se,
}

data_array = np.column_stack([data[key] for key in data.keys()])
header = ",".join(data.keys())
np.savetxt(
    join(data_folder_SE, file_name),
    data_array,
    header=header,
    delimiter=",",
    comments="",
)
