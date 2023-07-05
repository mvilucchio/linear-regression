import matplotlib.pyplot as plt
import numpy as np
from linear_regression.sweeps.eps_delta_out_sweeps import (
    sweep_eps_delta_out_optimal_lambda_fixed_point,
    sweep_eps_delta_out_optimal_lambda_hub_param_fixed_point,
)
from tqdm.auto import tqdm
from linear_regression.fixed_point_equations.fpe_Huber_loss import f_hat_Huber_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_L2_loss import f_hat_L2_decorrelated_noise
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.misc import estimation_error, excess_gen_error, gen_error_BO, angle_teacher_student


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


alpha, delta_in, beta = 10.0, 1.0, 0.0
eps_min, eps_max, n_eps_pts = 1e-3, 0.99, 150
delta_out_min, delta_out_max, n_delta_out_pts = 1e-2, 1e2, 150

alphas = [10.0]
# colors = ["r", "g", "b"]

# plt.figure(figsize=(7, 7))

for alpha in tqdm(alphas):
    (
        epsilons_l2,
        delta_out_l2,
        f_min_vals_l2,
        reg_params_opt_l2,
        (ms_l2, qs_l2, sigmas_l2),
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
        funs=[m_order_param, q_order_param, sigma_order_param],
        funs_args=[{}, {}, {}],
        update_funs_args=None,
        f_min=excess_gen_error,
        f_min_args={"delta_in": delta_in, "delta_out": 10.0, "percentage": 0.3, "beta": beta},
        update_f_min_args=True,
        min_reg_param=1e-5,
    )

    # print("L2 done")

    # np.savez(
    #     "phase_diagram_l2_alpha_{:.2f}.npz".format(alpha),
    #     epsilons=epsilons_l2,
    #     delta_out=delta_out_l2,
    #     f_min_vals=f_min_vals_l2,
    #     reg_params_opt=reg_params_opt_l2,
    #     ms=ms_l2,
    #     qs=qs_l2,
    #     sigmas=sigmas_l2,
    # )

    (
        epsilons_hub,
        delta_out_hub,
        f_min_vals_hub,
        (reg_params_opt_hub, huber_params_opt),
        (ms_hub, qs_hub, sigmas_hub),
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
        funs=[m_order_param, q_order_param, sigma_order_param],
        funs_args=[{}, {}, {}],
        update_funs_args=None,
        f_min=excess_gen_error,
        f_min_args={"delta_in": delta_in, "delta_out": 10.0, "percentage": 0.3, "beta": beta},
        update_f_min_args=True,
        min_reg_param=1e-5,
    )

    # print("Huber done")

    # np.savez(
    #     "phase_diagram_huber_alpha_{:.2f}.npz".format(alpha),
    #     epsilons=epsilons_hub,
    #     delta_out=delta_out_hub,
    #     f_min_vals=f_min_vals_hub,
    #     reg_params_opt=reg_params_opt_hub,
    #     huber_params_opt=huber_params_opt,
    #     ms=ms_hub,
    #     qs=qs_hub,
    #     sigmas=sigmas_hub,
    # )

    # print("L2", delta_out_l2)
    # print("Huber", delta_out_hub)

    difference_hub_l2 = f_min_vals_l2 - f_min_vals_hub

    np.savez(
        "phase_digaram_alpha_{:.2f}_deltain_{:.1f}.npz".format(alpha, delta_in),
        epsilons_hub=epsilons_hub,
        delta_out_hub=delta_out_hub,
        f_min_vals_hub=f_min_vals_hub,
        reg_params_opt_hub=reg_params_opt_hub,
        huber_params_opt=huber_params_opt,
        ms_hub=ms_hub,
        qs_hub=qs_hub,
        sigmas_hub=sigmas_hub,
        epsilons_l2=epsilons_l2,
        delta_out_l2=delta_out_l2,
        f_min_vals_l2=f_min_vals_l2,
        reg_params_opt_l2=reg_params_opt_l2,
        ms_l2=ms_l2,
        qs_l2=qs_l2,
        sigmas_l2=sigmas_l2,
        difference_hub_l2=difference_hub_l2,
    )

# it = np.nditer(difference_hub_l2, flags=["multi_index"])
# for x in it:
#     difference_hub_l2[it.multi_index] = 1.0 / (1.0 + np.exp(-(x - 1)))

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


# cs = plt.contourf(
#     epsilons_hub, delta_out_hub, difference_hub_l2, levels=[0.03]# 13
# )
# plt.contour(
#     epsilons_hub, delta_out_hub,
#     difference_hub_l2,
#     levels=[0.03],
#     colors=c,
#     # alpha=0.7,
#     linewidths=0.5,
#     label=r"$\alpha = {}$".format(alpha),
# )
# plt.contour(
#     epsilons_hub, delta_out_hub,
#     difference_hub_l2,
#     levels=[0.03],
#     colors="blue",
#     alpha=0.7,
#     linewidths=0.5,
# )
# plt.contour(
#     epsilons_hub, delta_out_hub,
#     difference_hub_l2,
#     levels=[0.03],
#     colors="red",
#     alpha=0.7,
#     linewidths=0.5,
# )
# plt.contour(
#     epsilons_hub, delta_out_hub,
#     difference_hub_l2,
#     levels=[0.03],
#     colors="green",
#     alpha=0.7,
#     linewidths=0.5,
# )
# plt.colorbar(cs)

# plt.xlabel(r"$\epsilon$", labelpad=1.0)
# plt.ylabel(r"$\Delta_{OUT}$", labelpad=1.0)
# plt.xscale("log")
# plt.yscale("log")
# plt.legend

# plt.show()
