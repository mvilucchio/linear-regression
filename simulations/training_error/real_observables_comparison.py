import numpy as np
import matplotlib.pyplot as plt
import linear_regression.sweeps.alpha_sweeps as alsw
from linear_regression.fixed_point_equations.fpe_L2_regularization import var_func_L2
from linear_regression.fixed_point_equations.fpe_L2_loss import (
    var_hat_func_L2_decorrelated_noise,
)
from linear_regression.fixed_point_equations.fpe_L1_loss import var_hat_func_L1_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    var_hat_func_Huber_decorrelated_noise,
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
    m_real_overlaps,
    q_real_overlaps
)
from linear_regression.aux_functions.misc import l2_loss, l1_loss, huber_loss
from linear_regression.regression_numerics.data_generation import measure_gen_decorrelated
from linear_regression.regression_numerics.erm_solvers import (
    find_coefficients_L2,
    find_coefficients_L1,
    find_coefficients_Huber,
)

delta_in, delta_out, percentage, beta, a = 1.0, 5.0, 0.3, 0.0, 1.0
reg_param = 0.0
alpha = 2.0

# numerical training errors
n_features = 1000
repetitions = 20
alphas_num_ridge = np.logspace(-1, 2, 10)
alphas_num_L1 = alphas_num_ridge.copy()
alphas_num_Hub = alphas_num_ridge.copy()

E_train_mean_ridge = np.empty_like(alphas_num_ridge)
E_train_std_ridge = np.empty_like(alphas_num_ridge)
E_train_mean_L1 = np.empty_like(alphas_num_L1)
E_train_std_L1 = np.empty_like(alphas_num_L1)
E_train_mean_Hub = np.empty_like(alphas_num_Hub)
E_train_std_Hub = np.empty_like(alphas_num_Hub)

# for idx, alpha in enumerate(alphas_num_ridge):

[(m_mean, m_std), (q_mean, q_std)] = erm_weight_finding(
    alpha,
    measure_gen_decorrelated,
    find_coefficients_L2,
    [m_real_overlaps, q_real_overlaps],
    [tuple(), tuple()],
    n_features,
    repetitions,
    (delta_in, delta_out, percentage, beta),
    (reg_param,),
)

print("m ", m_mean, " +/- ", m_std)
print("q ", q_mean, " +/- ", q_std)

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

    # E_train_mean_Hub[idx], E_train_std_Hub[idx] = erm_weight_finding(
    #     alpha,
    #     measure_gen_decorrelated,
    #     find_coefficients_Huber,
    #     [train_error_data],
    #     [(huber_loss, (a,))],
    #     n_features,
    #     repetitions,
    #     (delta_in, delta_out, percentage, beta),
    #     (reg_param, a),
    # )

print("Numerical done.")

# np.savetxt(
#     "./simulations/data/sweep_numerical_training_errors.csv",
#     np.array(
#         [
#             alphas_num_ridge,
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

# plt.figure(figsize=(10, 10))

# plt.title(
#     "$\\alpha$ sweep, $\\Delta_{{in}} = {}$, $\\Delta_{{out}} = {}$, $\\beta = {}$".format(
#         delta_in, delta_out, beta
#     )
# )

# plt.errorbar(
#     alphas_num_ridge,
#     E_train_mean_ridge,
#     yerr=E_train_std_ridge,
#     label="Numerical Training error Ridge",
#     color="tab:blue",
#     linestyle="",
# )
# # plt.errorbar(
# #     alphas_num_L1,
# #     E_train_mean_L1,
# #     yerr=E_train_std_L1,
# #     label="Numerical Training error L1",
# #     color="tab:green",
# #     linestyle="",
# # )
# plt.errorbar(
#     alphas_num_Hub,
#     E_train_mean_Hub,
#     yerr=E_train_std_Hub,
#     label="Numerical Training error Huber",
#     color="tab:orange",
#     linestyle="",
# )

# plt.yscale("log")
# plt.xscale("log")
# plt.ylabel(r"$E_{train}$")
# plt.xlabel(r"$\alpha$")
# plt.legend()
# plt.grid()

# plt.show()
