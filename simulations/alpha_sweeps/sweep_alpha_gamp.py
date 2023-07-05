import linear_regression.regression_numerics.amp_funcs as amp
import linear_regression.sweeps.alpha_sweeps as alsw
import linear_regression.regression_numerics.data_generation as dg
import linear_regression.aux_functions.prior_regularization_funcs as priors
import linear_regression.aux_functions.likelihood_channel_functions as like
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from linear_regression.aux_functions.stability_functions import (
    stability_ridge,
    stability_l1_l2,
    stability_huber,
)
from linear_regression.sweeps.alpha_sweeps import sweep_alpha_fixed_point
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.fixed_point_equations.fpe_L2_loss import f_hat_L2_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_L1_loss import f_hat_L1_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_Huber_loss import (
    f_hat_Huber_decorrelated_noise,
)
from linear_regression.aux_functions.misc import estimation_error
from linear_regression.aux_functions.loss_functions import l2_loss, l1_loss, huber_loss
from linear_regression.aux_functions.training_errors import (
    training_error_l2_loss,
    training_error_l1_loss,
    training_error_huber_loss,
)
from linear_regression.utils.errors import ConvergenceError

delta_in, delta_out, percentage, beta = 1.0, 5.0, 0.3, 0.0
repetitions = 10
d = 4000
# here the stuff to load the values in alpha

# alphas = np.logspace(-2, 2, 100)
alphas = np.array(
    [
        1.33545156e-01,
        1.84642494e-01,
        2.55290807e-01,
        3.52970730e-01,
        4.88025158e-01,
        6.74754405e-01,
        9.32930403e-01,
        1.28989026e00,
        1.78343088e00,
        2.46581108e00,
        3.40928507e00,
        4.71375313e00,
        6.51733960e00,
        9.01101825e00,
        1.24588336e01,
        1.72258597e01,
        2.38168555e01,
        3.29297126e01,
        4.55293507e01,
        6.29498899e01,
        8.70359136e01,
        # 1.20337784e02,
        # 1.66381689e02,
        # 2.30043012e02,
        # 3.18062569e02,
        # 4.39760361e02,
    ]
)

means_gen_error = np.ones_like(alphas)
stds_gen_error = np.ones_like(alphas)

means_estimation_error = np.ones_like(alphas)
stds_estimation_error = np.ones_like(alphas)

# means_estimation_error = []  # np.ones_like(alphas)
# stds_estimation_error = []  # np.ones_like(alphas)

# means_gen_error = []
# stds_gen_error = []

# alphas_2 = []

# for idx, alpha in enumerate(tqdm(alphas)):
#     # if alpha > 30:
#     #     continue

#     print(f"alpha = {alpha}")
#     all_gen_errors = []  # np.empty((repetitions,))
#     all_estim_errors = []  # np.empty((repetitions,))

#     for idx in range(repetitions):
#         try:
#             xs, ys, xs_test, ys_test, ground_truth_theta = dg.data_generation(
#                 dg.measure_gen_decorrelated,
#                 n_features=d,
#                 n_samples=max(int(np.around(d * alpha)), 1),
#                 n_generalization=2 * d,
#                 measure_fun_args=(delta_in, delta_out, percentage, beta),
#             )

#             # code to estimate the theta with GAMP algorithm
#             estimated_theta, _ = amp.GAMP_unsimplified_iters(
#                 priors.f_w_Bayes_gaussian_prior,
#                 priors.Df_w_Bayes_gaussian_prior,
#                 like.f_out_Bayes_decorrelated_noise,
#                 like.Df_out_Bayes_decorrelated_noise,
#                 ys,
#                 xs,
#                 (0.0, 1.0),
#                 (delta_in, delta_out, percentage, beta),
#                 np.random.normal(size=d),
#                 1.0,
#                 max_iter=10_000,
#                 blend=0.85,
#             )

#             # two lines need to be checked
#             all_gen_errors.append(
#                 np.mean((ys_test - (1 - percentage + percentage * beta) * xs_test @ estimated_theta / np.sqrt(d)) ** 2)
#                 - np.mean(
#                     (ys_test - (1 - percentage + percentage * beta) * xs_test @ ground_truth_theta / np.sqrt(d)) ** 2
#                 )
#                 # - ((1-percentage) * delta_in + percentage * delta_out)
#             )

#             all_estim_errors.append(np.mean(np.square(estimated_theta - ground_truth_theta)))

#             del xs
#             del ys
#             del xs_test
#             del ys_test
#             del estimated_theta
#             del ground_truth_theta

#         except ConvergenceError:
#             continue

#     # check if one needs the additional normalisation
#     means_gen_error.append(np.mean(all_gen_errors))
#     stds_gen_error.append(np.std(all_gen_errors) / np.sqrt(repetitions))

#     alphas_2.append(alpha)
#     means_estimation_error.append(np.mean(all_estim_errors))
#     stds_estimation_error.append(np.std(all_estim_errors) / np.sqrt(repetitions))

# # print(means_estimation_error, stds_estimation_error)

# np.savez(
#     "GAMP_alpha_sweep_gen_error_d4000_tol_5e-3.npz",
#     alphas=alphas,
#     means_gen_error=means_gen_error,
#     stds_gen_error=stds_gen_error,
#     means_estimation_error=means_estimation_error,
#     stds_estimation_error=stds_estimation_error,
# )

dat_gamp = np.load("GAMP_alpha_sweep_gen_error_d4000_tol_5e-3.npz")

means_gen_error = dat_gamp["means_gen_error"]
stds_gen_error = dat_gamp["stds_gen_error"]

means_estimation_error = dat_gamp["means_estimation_error"]
stds_estimation_error = dat_gamp["stds_estimation_error"]

alphas_2 = dat_gamp["alphas"]

dat_gen = np.load("BO_FIGURE_1_generalization_error.npz")
dat_estim = np.load("BO_Estimation_FIGURE_1.npz")

dat_huber = np.load("Huber_FIGURE_1_generalization_error.npz")
dat_l1 = np.load("L1_FIGURE_1_generalization_error.npz")
dat_l2 = np.load("L2_FIGURE_1_generalization_error.npz")

plt.errorbar(alphas_2, means_gen_error, yerr=stds_gen_error, label="gen", color="tab:blue", marker=".", linestyle=None)
plt.plot(dat_gen["alphas"], dat_gen["gen_error"], color="tab:blue", label="gen")
plt.plot(dat_huber["alphas"], dat_huber["f_min_vals"], color="tab:green", label="huber")
plt.plot(dat_l1["alphas"], dat_l1["f_min_vals"], color="tab:red", label="l1")
plt.plot(dat_l2["alphas"], dat_l2["f_min_vals"], color="tab:purple", label="l2")

# plt.errorbar(
#     alphas_2,
#     means_estimation_error,
#     yerr=stds_estimation_error,
#     label="estim",
#     color="tab:orange",
#     marker=".",
#     linestyle=None,
# )
# plt.plot(dat_estim["alphas"], dat_estim["gen_error"], color="tab:orange", label="estim")

plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.legend()
plt.xlim(0.1, 100)
plt.ylim(0.08, 1.1)

plt.show()
