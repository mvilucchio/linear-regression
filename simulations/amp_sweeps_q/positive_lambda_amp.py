import linear_regression.regression_numerics.amp_funcs as amp
import linear_regression.sweeps.alpha_sweeps as alsw
from linear_regression.regression_numerics.data_generation import measure_gen_decorrelated
from linear_regression.regression_numerics.numerics import gen_error_data, train_error_data
import linear_regression.aux_functions.prior_regularization_funcs as priors
import linear_regression.aux_functions.likelihood_channel_functions as like
import numpy as np
import matplotlib.pyplot as plt
from linear_regression.aux_functions.stability_functions import (
    stability_ridge,
    stability_l1_l2,
    stability_huber,
)
from linear_regression.sweeps.alpha_sweeps import sweep_alpha_fixed_point
from linear_regression.fixed_point_equations.fpe_L2_regularization import var_func_L2
from linear_regression.fixed_point_equations.fpe_L2_loss import var_hat_func_L2_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_L1_loss import var_hat_func_L1_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    var_hat_func_Huber_decorrelated_noise,
)
from linear_regression.aux_functions.misc import estimation_error
from linear_regression.aux_functions.loss_functions import l2_loss, l1_loss, huber_loss
from linear_regression.aux_functions.training_errors import (
    training_error_l2_loss,
    training_error_l1_loss,
    training_error_huber_loss,
)


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
a_hub = 1.0

reg_params = [0.1]

alphas_l2 = [None for _ in range(len(reg_params))]
alphas_l1 = [None for _ in range(len(reg_params))]
alphas_Hub = [None for _ in range(len(reg_params))]

gen_error_l2 = [None for _ in range(len(reg_params))]
gen_error_l1 = [None for _ in range(len(reg_params))]
gen_error_Hub = [None for _ in range(len(reg_params))]

train_error_l2 = [None for _ in range(len(reg_params))]
train_error_l1 = [None for _ in range(len(reg_params))]
train_error_Hub = [None for _ in range(len(reg_params))]

ms_l2 = [None for _ in range(len(reg_params))]
qs_l2 = [None for _ in range(len(reg_params))]
sigmas_l2 = [None for _ in range(len(reg_params))]
ms_l1 = [None for _ in range(len(reg_params))]
qs_l1 = [None for _ in range(len(reg_params))]
sigmas_l1 = [None for _ in range(len(reg_params))]
ms_Hub = [None for _ in range(len(reg_params))]
qs_Hub = [None for _ in range(len(reg_params))]
sigmas_Hub = [None for _ in range(len(reg_params))]

for idx, reg_param in enumerate(reg_params):
    while True:
        q_init = 10 * np.random.random() + 0.01
        m_init = 10 * np.random.random() + 0.01
        sigma_init = 10 * np.random.random() + 0.01
        if (
            np.square(m_init) < q_init + delta_in * q_init
            and np.square(m_init) < q_init + delta_out * q_init
        ):
            break

    print("FPE reg param: ", reg_param)
    alphas_l2[idx], (
        gen_error_l2[idx],
        train_error_l2[idx],
        ms_l2[idx],
        qs_l2[idx],
        sigmas_l2[idx],
    ) = sweep_alpha_fixed_point(
        var_func_L2,
        var_hat_func_L2_decorrelated_noise,
        0.1,
        300,
        100,
        {"reg_param": reg_param},
        {"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta},
        initial_cond_fpe=(m_init, q_init, sigma_init),
        funs=[estimation_error, training_error_l2_loss, m_order_param, q_order_param, sigma_order_param],
        funs_args=[list(), (delta_in, delta_out, percentage, beta), list(), list(), list()],
    )

    alphas_l1[idx], (
        gen_error_l1[idx],
        train_error_l1[idx],
        ms_l1[idx],
        qs_l1[idx],
        sigmas_l1[idx],
    ) = sweep_alpha_fixed_point(
        var_func_L2,
        var_hat_func_L1_decorrelated_noise,
        0.1,
        100,
        100,
        {"reg_param": reg_param},
        {"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta},
        initial_cond_fpe=(m_init, q_init, sigma_init),
        funs=[estimation_error, training_error_l1_loss, m_order_param, q_order_param, sigma_order_param],
        funs_args=[list(), (delta_in, delta_out, percentage, beta), list(), list(), list()],
    )

    # alphas_Hub[idx], (
    #     gen_error_Hub[idx],
    #     train_error_Hub[idx],
    #     ms_Hub[idx],
    #     qs_Hub[idx],
    #     sigmas_Hub[idx],
    # ) = sweep_alpha_fixed_point(
    #     var_func_L2,
    #     var_hat_func_Huber_decorrelated_noise,
    #     0.1,
    #     30,
    #     100,
    #     {"reg_param": reg_param},
    #     {
    #         "delta_in": delta_in,
    #         "delta_out": delta_out,
    #         "percentage": percentage,
    #         "beta": beta,
    #         "a": a_hub,
    #     },
    #     initial_cond_fpe=(m_init, q_init, sigma_init),
    #     funs=[
    #         estimation_error,
    #         training_error_huber_loss,
    #         m_order_param,
    #         q_order_param,
    #         sigma_order_param,
    #     ],
    #     funs_args=[list(), (delta_in, delta_out, percentage, beta, a_hub), list(), list(), list()],
    # )

n_features = 10000
repetitions = 5

alphas_l2_amp = [None for _ in range(len(reg_params))]
alphas_l1_amp = [None for _ in range(len(reg_params))]
alphas_Hub_amp = [None for _ in range(len(reg_params))]

gen_error_mean_l2_amp = [None for _ in range(len(reg_params))]
gen_error_std_l2_amp = [None for _ in range(len(reg_params))]
gen_error_mean_l1_amp = [None for _ in range(len(reg_params))]
gen_error_std_l1_amp = [None for _ in range(len(reg_params))]
gen_error_mean_Hub_amp = [None for _ in range(len(reg_params))]
gen_error_std_Hub_amp = [None for _ in range(len(reg_params))]

train_error_mean_l2_amp = [None for _ in range(len(reg_params))]
train_error_std_l2_amp = [None for _ in range(len(reg_params))]
train_error_mean_l1_amp = [None for _ in range(len(reg_params))]
train_error_std_l1_amp = [None for _ in range(len(reg_params))]
train_error_mean_Hub_amp = [None for _ in range(len(reg_params))]
train_error_std_Hub_amp = [None for _ in range(len(reg_params))]


for idx, reg_param in enumerate(reg_params):
    print("AMP reg_param: ", reg_param)
    # for a total of 10 different alphas fun the AMP for L2, L1, and Huber loss
    print("L2")
    (
        alphas_l2_amp[idx],
        (gen_error_mean_l2_amp[idx], train_error_mean_l2_amp[idx]),
        (gen_error_std_l2_amp[idx], train_error_std_l2_amp[idx]),
    ) = alsw.sweep_alpha_GAMP(
        priors.f_w_L2_regularization,
        priors.Df_w_L2_regularization,
        like.f_out_L2,
        like.Df_out_L2,
        measure_gen_decorrelated,
        0.1,
        1,
        10,
        repetitions,
        n_features,
        (reg_param,),
        list(),
        (delta_in, delta_out, percentage, beta),
        funs=[gen_error_data, train_error_data],
        funs_args=[list(), [l2_loss, list()]],
    )

    print("L1")
    (
        alphas_l1_amp[idx],
        (gen_error_mean_l1_amp[idx], train_error_mean_l1_amp[idx]),
        (gen_error_std_l1_amp[idx], train_error_std_l1_amp[idx]),
    ) = alsw.sweep_alpha_GAMP(
        priors.f_w_L2_regularization,
        priors.Df_w_L2_regularization,
        like.f_out_L1,
        like.Df_out_L1,
        measure_gen_decorrelated,
        0.1,
        1,
        10,
        repetitions,
        n_features,
        (reg_param,),
        list(),
        (delta_in, delta_out, percentage, beta),
        funs=[gen_error_data, train_error_data],
        funs_args=[list(), [l1_loss, list()]],
    )

    # print("Huber")
    # (
    #     alphas_Hub_amp[idx],
    #     (gen_error_mean_Hub_amp[idx], train_error_mean_Hub_amp[idx]),
    #     (gen_error_std_Hub_amp[idx], train_error_std_Hub_amp[idx]),
    # ) = alsw.sweep_alpha_GAMP(
    #     priors.f_w_L2_regularization,
    #     priors.Df_w_L2_regularization,
    #     like.f_out_Huber,
    #     like.Df_out_Huber,
    #     measure_gen_decorrelated,
    #     0.1,
    #     100,
    #     10,
    #     repetitions,
    #     n_features,
    #     (reg_param,),
    #     (a_hub,),
    #     (delta_in, delta_out, percentage, beta),
    #     funs=[gen_error_data, train_error_data],
    #     funs_args=[list(), [huber_loss, (a_hub,)]],
    # )

# print(alphas_l2_amp, gen_error_mean_l2_amp, gen_error_std_l2_amp)

# from here on there is the plotting

plt.figure(figsize=(10, 8))

plt.subplot(211)
for idx in range(len(reg_params)):
    color = next(plt.gca()._get_lines.prop_cycler)["color"]
    # plt.plot(
    #     alphas_l2[idx],
    #     gen_error_l2[idx],
    #     color=color,
    #     label="L2, FPE, λ={}".format(reg_params[idx]),
    # )
    # plt.errorbar(
    #     alphas_l2_amp[idx],
    #     gen_error_mean_l2_amp[idx],
    #     yerr=gen_error_std_l2_amp[idx],
    #     color=color,
    #     linestyle="None",
    #     marker='.',
    #     label="L2, AMP, λ={}".format(reg_params[idx]),
    # )
    plt.plot(
        alphas_l2[idx],
        train_error_l2[idx],
        color=color,
        linestyle="-",
        label="L2, FPE, λ={}".format(reg_params[idx]),
    )
    plt.errorbar(
        alphas_l2_amp[idx],
        train_error_mean_l2_amp[idx],
        yerr=train_error_std_l2_amp[idx],
        color=color,
        linestyle="None",
        marker=".",
        label="L2, AMP, λ={}".format(reg_params[idx]),
    )

    color = next(plt.gca()._get_lines.prop_cycler)["color"]
    # plt.plot(
    #     alphas_l1[idx],
    #     gen_error_l1[idx],
    #     color=color,
    #     label="L1, FPE, λ={}".format(reg_params[idx]),
    # )
    # plt.errorbar(
    #     alphas_l1_amp[idx],
    #     gen_error_mean_l1_amp[idx],
    #     yerr=gen_error_std_l1_amp[idx],
    #     color=color,
    #     linestyle="None",
    #     marker=".",
    #     label="L1, AMP, λ={}".format(reg_params[idx]),
    # )
    plt.plot(
        alphas_l1[idx],
        train_error_l1[idx],
        color=color,
        linestyle="-",
        label="L1, FPE, λ={}".format(reg_params[idx]),
    )
    plt.errorbar(
        alphas_l1_amp[idx],
        train_error_mean_l1_amp[idx],
        yerr=train_error_std_l1_amp[idx],
        color=color,
        linestyle="None",
        marker=".",
        label="L1, AMP, λ={}".format(reg_params[idx]),
    )

    # color = next(plt.gca()._get_lines.prop_cycler)["color"]
    # plt.plot(
    #     alphas_Hub[idx],
    #     gen_error_Hub[idx],
    #     color=color,
    #     label="Huber, FPE, λ={}".format(reg_params[idx]),
    # )
    # plt.errorbar(
    #     alphas_Hub_amp[idx],
    #     gen_error_mean_Hub_amp[idx],
    #     yerr=gen_error_std_Hub_amp[idx],
    #     color=color,
    #     linestyle="None",
    #     marker=".",
    #     label="Huber, AMP, λ={}".format(reg_params[idx]),
    # )

plt.ylabel("E_gen")
plt.xscale("log")
plt.yscale("log")
# plt.xlim([0.1, 100])
# plt.ylim([1e-1, 1.5])
plt.legend()
plt.grid()

plt.subplot(212)
for idx in range(len(reg_params)):
    color = next(plt.gca()._get_lines.prop_cycler)["color"]
    plt.plot(
        alphas_l2[idx],
        stability_ridge(
            ms_l2[idx],
            qs_l2[idx],
            sigmas_l2[idx],
            alphas_l2[idx],
            reg_param,
            delta_in,
            delta_out,
            percentage,
            beta,
        ),
        color=color,
        label="L2, λ={}".format(reg_params[idx]),
    )

    color = next(plt.gca()._get_lines.prop_cycler)["color"]
    plt.plot(
        alphas_l1[idx],
        stability_l1_l2(
            ms_l1[idx],
            qs_l1[idx],
            sigmas_l1[idx],
            alphas_l1[idx],
            reg_param,
            delta_in,
            delta_out,
            percentage,
            beta,
        ),
        color=color,
        label="L1, λ={}".format(reg_params[idx]),
    )

    # color = next(plt.gca()._get_lines.prop_cycler)["color"]
    # plt.plot(
    #     alphas_l1[idx],
    #     stability_huber(
    #         ms_l1[idx], qs_l1[idx], sigmas_l1[idx], alphas_l1[idx], reg_param, delta_in, delta_out, percentage, beta, a_hub
    #     ),
    #     color=color,
    #     label="Huber, λ={}".format(reg_params[idx]),
    # )

plt.ylabel("Stability Condition")
plt.xscale("log")
# plt.xlim([0.1, 100])
# plt.ylim([1e-1, 1.5])
plt.legend()
plt.grid()

plt.show()
