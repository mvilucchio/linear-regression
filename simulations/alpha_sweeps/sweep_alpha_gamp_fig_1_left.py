import linear_regression.regression_numerics.amp_funcs as amp
import linear_regression.sweeps.alpha_sweeps as alsw
import linear_regression.regression_numerics.data_generation as dg
import linear_regression.aux_functions.prior_regularization_funcs as priors
import linear_regression.aux_functions.likelihood_channel_functions as like
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from linear_regression.aux_functions.stability_functions import (
    stability_L2_decorrelated_regress,
    stability_L1_decorrelated_regress,
    stability_Huber_decorrelated_regress,
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

delta_in, delta_out, percentage, beta = 1.0, 0.5, 0.6, 0.0
repetitions = 50
d = 1000
# here the stuff to load the values in alpha

alphas = np.array(
    [
        1.34977233e-01,
        1.82188534e-01,
        2.45913043e-01,
        3.31926620e-01,
        4.48025368e-01,
        6.04732245e-01,
        8.16250851e-01,
        1.10175281e00,
        1.48711546e00,
        2.00726730e00,
        2.70935387e00,
        3.65701088e00,
        4.93613210e00,
        6.66265452e00,
        8.99306672e00,
        1.21385926e01,
        1.63843365e01,
        2.21151240e01,
        2.98503825e01,
        4.02912203e01,
        5.43839743e01,
        7.34059837e01,
        9.90813657e01,
        1.33737286e02,
        1.80514888e02,
        2.43654001e02,
        3.28877429e02,
    ]
)

means_estimation_error = []
stds_estimation_error = []

means_gen_error = []
stds_gen_error = []

alphas_2 = []

for idx, alpha in enumerate(tqdm(alphas)):
    print(f"alpha = {alpha}")
    all_gen_errors = []
    all_estim_errors = []

    for idx in range(repetitions):
        try:
            xs, ys, xs_test, ys_test, ground_truth_theta = dg.data_generation(
                dg.measure_gen_decorrelated,
                n_features=d,
                n_samples=max(int(np.around(d * alpha)), 1),
                n_generalization=2 * d,
                measure_fun_args=(delta_in, delta_out, percentage, beta),
            )

            # code to estimate the theta with GAMP algorithm
            estimated_theta, _ = amp.GAMP_unsimplified_iters(
                priors.f_w_Bayes_gaussian_prior,
                priors.Df_w_Bayes_gaussian_prior,
                like.f_out_Bayes_decorrelated_noise,
                like.Df_out_Bayes_decorrelated_noise,
                ys,
                xs,
                (0.0, 1.0),
                (delta_in, delta_out, percentage, beta),
                np.random.normal(size=d),
                1.0,
                max_iter=10_000,
                blend=0.85,
            )

            # two lines need to be checked
            all_gen_errors.append(
                np.mean((ys_test - (1 - percentage + percentage * beta) * xs_test @ estimated_theta / np.sqrt(d)) ** 2)
                - np.mean(
                    (ys_test - (1 - percentage + percentage * beta) * xs_test @ ground_truth_theta / np.sqrt(d)) ** 2
                )
                # - ((1-percentage) * delta_in + percentage * delta_out)
            )

            all_estim_errors.append(np.mean(np.square(estimated_theta - ground_truth_theta)))

            del xs
            del ys
            del xs_test
            del ys_test
            del estimated_theta
            del ground_truth_theta

        except ConvergenceError:
            continue

    # check if one needs the additional normalisation
    means_gen_error.append(np.mean(all_gen_errors))
    stds_gen_error.append(np.std(all_gen_errors) / np.sqrt(repetitions))

    alphas_2.append(alpha)
    means_estimation_error.append(np.mean(all_estim_errors))
    stds_estimation_error.append(np.std(all_estim_errors) / np.sqrt(repetitions))

# print(means_estimation_error, stds_estimation_error)

fname = f"GAMP_alpha_sweep_gen_error_d{d}_reps_{repetitions}_tol_5e-3.npz"

np.savez(
    fname,
    alphas=alphas,
    means_gen_error=means_gen_error,
    stds_gen_error=stds_gen_error,
    means_estimation_error=means_estimation_error,
    stds_estimation_error=stds_estimation_error,
)

dat_gamp = np.load(fname)

means_gen_error = dat_gamp["means_gen_error"]
stds_gen_error = dat_gamp["stds_gen_error"]

means_estimation_error = dat_gamp["means_estimation_error"]
stds_estimation_error = dat_gamp["stds_estimation_error"]

alphas_2 = dat_gamp["alphas"]

dat_gen = np.load("BO_FIGURE_1_left.npz")
# dat_estim = np.load("BO_Estimation_FIGURE_1.npz")

dat_huber = np.load("Huber_FIGURE_1_generalization_error.npz")
dat_l1 = np.load("L1_FIGURE_1_generalization_error.npz")
dat_l2 = np.load("L2_FIGURE_1_generalization_error.npz")

plt.errorbar(alphas_2, means_gen_error, yerr=stds_gen_error, label="gen", color="tab:blue", marker=".", linestyle=None)
plt.plot(dat_gen["alphas"], dat_gen["gen_error"], color="tab:blue", label="gen")
# plt.plot(dat_huber["alphas"], dat_huber["f_min_vals"], color="tab:green", label="huber")
# plt.plot(dat_l1["alphas"], dat_l1["f_min_vals"], color="tab:red", label="l1")
# plt.plot(dat_l2["alphas"], dat_l2["f_min_vals"], color="tab:purple", label="l2")

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
# plt.xlim(0.1, 100)
plt.ylim(0.08, 1.1)

plt.show()
