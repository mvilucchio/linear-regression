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
from scipy.optimize import minimize_scalar

MIN_REG_PARAM = 1e-6
XATOL = 1e-8

alpha_min, alpha_max, n_alpha_pts = 0.005, 1, 100
eps_min, eps_max, n_eps_pts = 0.01, 0.3, 100
reg_orders = [1, 2]
n_division_of_epss_array = 15
pstar = 1

alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)
epss = np.linspace(eps_min, eps_max, n_eps_pts)

epss_divisions = np.array_split(epss, n_division_of_epss_array)

pairs = list(product(epss_divisions, reg_orders))

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

assert len(pairs) >= size

epss, reg_order = pairs[rank]
n_eps_pts_rank = len(epss)

data_folder_SE = "./data/SE_alpha_eps_sweep_optimal_lambda"

if not exists(data_folder_SE):
    os.makedirs(data_folder_SE)

file_name = f"SE_alpha_eps_sweep_optimal_lambda_pstar_{pstar}_reg_order_{reg_order}_alpha_{alpha_min:.3f}_{alpha_max:.3f}_eps_{min(epss):.1e}_{min(epss):.1e}.csv"

ms_found = np.empty((n_alpha_pts, n_eps_pts_rank))
qs_found = np.empty((n_alpha_pts, n_eps_pts_rank))
Vs_found = np.empty((n_alpha_pts, n_eps_pts_rank))
Ps_found = np.empty((n_alpha_pts, n_eps_pts_rank))

mhats_found = np.empty((n_alpha_pts, n_eps_pts_rank))
qhats_found = np.empty((n_alpha_pts, n_eps_pts_rank))
Vhats_found = np.empty((n_alpha_pts, n_eps_pts_rank))
Phats_found = np.empty((n_alpha_pts, n_eps_pts_rank))

reg_param_opts = np.empty((n_alpha_pts, n_eps_pts_rank))

estim_errors_se = np.empty((n_alpha_pts, n_eps_pts_rank))
adversarial_errors_found = np.empty((n_alpha_pts, n_eps_pts_rank))
gen_errors_se = np.empty((n_alpha_pts, n_eps_pts_rank))

for i, eps_t in enumerate(epss):
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

    reg_param_init = 1e-3

    eps_g = eps_t
    print(f"Starting the sweep for {reg_order = }, {eps_t = }")

    for jprime, alpha in enumerate(reversed(alphas)):
        j = n_alpha_pts - jprime - 1

        f_kwargs = {"reg_param": reg_param_init / alpha, "reg_order": reg_order, "pstar": pstar}
        f_hat_kwargs = {"alpha": alpha, "eps_t": eps_t}

        def minimize_fun(reg_param: float, alpha: float):
            print(
                f"Testing reg_param = {reg_param}, for reg_order = {reg_order}, alpha = {alpha}, eps_t = {eps_t}"
            )
            f_kwargs.update({"reg_param": reg_param / alpha})
            m, q, V, P = fixed_point_finder(
                f_Lr_regularisation_Lpstar_attack,
                f_hat_Logistic_no_noise_Linf_adv_classif,
                initial_condition,
                f_kwargs,
                f_hat_kwargs,
                abs_tol=1e-7,
                min_iter=10,
                args_update_function=(0.2,),
                max_iter=10_000_000,
            )
            return classification_adversarial_error(m, q, P, eps_g, pstar)

        obj = minimize_scalar(
            minimize_fun,
            args=(alpha,),
            method="bounded",
            bounds=(MIN_REG_PARAM, 1e0),
            options={"xatol": XATOL, "maxiter": 1000},
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
                abs_tol=1e-8,
                min_iter=10,
                args_update_function=(0.4,),
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
            print(
                f"Optimisation failed for reg_order = {reg_order}, alpha = {alpha}, eps_t = {eps_t}"
            )

    print(f"Saving data for reg_order = {reg_order}, eps_t = {eps_t}")

# Save data
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
