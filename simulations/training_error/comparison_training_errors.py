import numpy as np
import matplotlib.pyplot as plt
import linear_regression.sweeps.alpha_sweeps as alsw
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.fixed_point_equations.fpe_L2_loss import (
    f_hat_L2_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_L1_loss import f_hat_L1_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    f_hat_Huber_decorrelated_noise,
)
from linear_regression.aux_functions.training_errors import (
    training_error_l2_loss,
    training_error_l1_loss,
    training_error_huber_loss,
)
from linear_regression.regression_numerics.numerics import (
    erm_weight_finding,
    train_error_data,
    gen_error_data,
)
from linear_regression.aux_functions.misc import l2_loss, l1_loss, huber_loss
from linear_regression.regression_numerics.data_generation import measure_gen_decorrelated
from linear_regression.regression_numerics.erm_solvers import (
    find_coefficients_L2,
    find_coefficients_L1,
    find_coefficients_Huber,
)

delta_in, delta_out, percentage, beta, a = 1.0, 5.0, 0.3, 0.0, 1.0
reg_param = 1.0

# theoretical training errors
alphas_theo_ridge, (training_error_theor_ridge,) = alsw.sweep_alpha_fixed_point(
    f_L2_reg,
    f_hat_L2_decorrelated_noise,
    0.01,
    100,
    100,
    {"reg_param": reg_param},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[training_error_l2_loss],
    funs_args=[(delta_in, delta_out, percentage, beta)],
)

alphas_theo_L1, (training_error_theor_L1,) = alsw.sweep_alpha_fixed_point(
    f_L2_reg,
    f_hat_L1_decorrelated_noise,
    0.01,
    100,
    100,
    {"reg_param": reg_param},
    {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": percentage,
        "beta": beta,
    },
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[training_error_l1_loss],
    funs_args=[(delta_in, delta_out, percentage, beta)],
)

alphas_theo_Huber, (training_error_theor_Huber,) = alsw.sweep_alpha_fixed_point(
    f_L2_reg,
    f_hat_Huber_decorrelated_noise,
    0.01,
    100,
    100,
    {"reg_param": reg_param},
    {"delta_in": delta_in, "delta_out": delta_out, "percentage": percentage, "beta": beta, "a": a},
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[training_error_huber_loss],
    funs_args=[(delta_in, delta_out, percentage, beta, a)],
)

print("Theoretical done.")

# numerical training errors
n_features = 500
repetitions = 10
alphas_ridge = np.logspace(-1, 2, 10)
alphas_L1 = alphas_ridge.copy()
alphas_Hub = alphas_ridge.copy()

E_train_mean_ridge = np.empty_like(alphas_ridge)
E_train_std_ridge = np.empty_like(alphas_ridge)
E_train_mean_L1 = np.empty_like(alphas_L1)
E_train_std_L1 = np.empty_like(alphas_L1)
E_train_mean_Hub = np.empty_like(alphas_Hub)
E_train_std_Hub = np.empty_like(alphas_Hub)

for idx, alpha in enumerate(alphas_ridge):
    E_train_mean_ridge[idx], E_train_std_ridge[idx] = erm_weight_finding(
        alpha,
        measure_gen_decorrelated,
        find_coefficients_L2,
        [train_error_data],
        [(l2_loss, tuple())],
        n_features,
        repetitions,
        (delta_in, delta_out, percentage, beta),
        (reg_param,),
    )

    # E_train_mean_L1[idx], E_train_std_L1[idx] = erm_weight_finding(
    #     alpha,
    #     measure_gen_decorrelated,
    #     find_coefficients_L1,
    #     [train_error_data],
    #     [(l1_loss, tuple())],
    #     n_features,
    #     repetitions,
    #     (delta_in, delta_out, percentage, beta),
    #     (reg_param,),
    # )

    E_train_mean_Hub[idx], E_train_std_Hub[idx] = erm_weight_finding(
        alpha,
        measure_gen_decorrelated,
        find_coefficients_Huber,
        [train_error_data],
        [(huber_loss, (a,))],
        n_features,
        repetitions,
        (delta_in, delta_out, percentage, beta),
        (reg_param, a),
    )

print("Numerical done.")

# np.savetxt(
#     "./simulations/data/sweep_numerical_training_errors.csv",
#     np.array(
#         [
#             alphas_ridge,
#             E_train_mean_ridge,
#             E_train_std_ridge,
#             E_train_mean_L1,
#             E_train_std_L1,
#             E_train_mean_Hub,
#             E_train_std_Hub,
#         ]
#     ).T,
#     delimiter=",",
#     header="alpha, E_train_mean_ridge, E_train_std_ridge, E_train_mean_L1, E_train_std_L1, E_train_mean_Hub, E_train_std_Hub",
# )

plt.figure(figsize=(10, 10))

plt.title(
    "$\\alpha$ sweep, $\\Delta_{{in}} = {}$, $\\Delta_{{out}} = {}$, $\\beta = {}$".format(
        delta_in, delta_out, beta
    )
)
plt.plot(
    alphas_theo_ridge,
    training_error_theor_ridge,
    label="Theoretical Training error Ridge",
    color="tab:blue",
)
plt.plot(
    alphas_theo_L1,
    training_error_theor_L1,
    label="Theoretical Training error L1",
    color="tab:green",
)
plt.plot(
    alphas_theo_Huber,
    training_error_theor_Huber,
    label="Theoretical Training error Huber",
    color="tab:orange",
)

plt.errorbar(
    alphas_ridge,
    E_train_mean_ridge,
    yerr=E_train_std_ridge,
    label="Numerical Training error Ridge",
    color="tab:blue",
    linestyle="",
)
# plt.errorbar(
#     alphas_L1,
#     E_train_mean_L1,
#     yerr=E_train_std_L1,
#     label="Numerical Training error L1",
#     color="tab:green",
#     linestyle="",
# )
plt.errorbar(
    alphas_Hub,
    E_train_mean_Hub,
    yerr=E_train_std_Hub,
    label="Numerical Training error Huber",
    color="tab:orange",
    linestyle="",
)

plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"$E_{train}$")
plt.xlabel(r"$\alpha$")
plt.legend()
plt.grid()

plt.show()
