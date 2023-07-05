import matplotlib.pyplot as plt
import numpy as np
from linear_regression.sweeps.eps_delta_out_sweeps import (
    sweep_eps_delta_out_optimal_lambda_fixed_point,
    sweep_eps_delta_out_optimal_lambda_hub_param_fixed_point,
)
from linear_regression.fixed_point_equations.fpe_Huber_loss import f_hat_Huber_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_L2_loss import f_hat_L2_decorrelated_noise
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.misc import estimation_error, excess_gen_error, gen_error_BO, angle_teacher_student

alpha, delta_in, beta = 10.0, 1.0, 0.0
eps_min, eps_max, n_eps_pts = 1e-3, 0.99, 100
delta_out_min, delta_out_max, n_delta_out_pts = 1e0, 5e2, 100

(
    epsilons_l2,
    delta_out_l2,
    f_min_vals_l2,
    reg_params_opt_l2,
    _,
) = sweep_eps_delta_out_optimal_lambda_fixed_point(
    f_L2_reg,
    f_hat_L2_decorrelated_noise,
    eps_min,
    eps_max,
    n_eps_pts,
    delta_out_min,
    delta_out_max,
    n_delta_out_pts,
    {"reg_param": 3.0},
    {"alpha": alpha, "delta_in": delta_in, "delta_out": 10.0, "percentage": 0.3, "beta": beta},
    delta_in,
    (0.6, 0.01, 0.9),
    funs=[],
    funs_args=[],
    update_funs_args=None,
    f_min=excess_gen_error,
    f_min_args={"delta_in": delta_in, "delta_out": 10.0, "percentage": 0.3, "beta": beta},
    update_f_min_args=True,
    min_reg_param=1e-5,
)

print("L2 done")

(
    epsilons_hub,
    delta_out_hub,
    f_min_vals_hub,
    (reg_params_opt_hub, huber_params_opt),
    _,
) = sweep_eps_delta_out_optimal_lambda_hub_param_fixed_point(
    f_L2_reg,
    f_hat_Huber_decorrelated_noise,
    eps_min,
    eps_max,
    n_eps_pts,
    delta_out_min,
    delta_out_max,
    n_delta_out_pts,
    {"reg_param": 3.0},
    {"alpha": alpha, "delta_in": delta_in, "delta_out": 10.0, "percentage": 0.3, "beta": beta, "a": 1.0},
    delta_in,
    1.0,
    (0.6, 0.01, 0.9),
    funs=[],
    funs_args=[],
    update_funs_args=None,
    f_min=excess_gen_error,
    f_min_args={"delta_in": delta_in, "delta_out": 10.0, "percentage": 0.3, "beta": beta},
    update_f_min_args=True,
    min_reg_param=1e-5,
)

print("Huber done")

print("L2", delta_out_l2)
print("Huber", delta_out_hub)

difference_hub_l2 = f_min_vals_l2 - f_min_vals_hub

# plt.figure(figsize=(7, 7))

# cs = plt.contourf(
#     epsilons_hub, delta_out_hub, huber_params_opt, levels=13
# )
# plt.contour(
#     epsilons_hub, delta_out_hub,
#     huber_params_opt,
#     levels=13,
#     colors="black",
#     alpha=0.7,
#     linewidths=0.5,
# )
# plt.colorbar(cs)

# plt.xlabel(r"$\epsilon$", labelpad=1.0)
# plt.ylabel(r"$\Delta_{OUT}$", labelpad=1.0)
# plt.xscale("log")
# plt.yscale("log")

# plt.show()

# ------------------------------

plt.figure(figsize=(7, 7))

cs = plt.contourf(
    epsilons_hub, delta_out_hub, difference_hub_l2, levels=13
)
plt.contour(
    epsilons_hub, delta_out_hub,
    difference_hub_l2,
    levels=13,
    colors="black",
    alpha=0.7,
    linewidths=0.5,
)
plt.colorbar(cs)

plt.xlabel(r"$\epsilon$", labelpad=1.0)
plt.ylabel(r"$\Delta_{OUT}$", labelpad=1.0)
plt.xscale("log")
plt.yscale("log")

plt.show()