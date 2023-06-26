import linear_regression.regression_numerics.amp_funcs as amp
import linear_regression.sweeps.alpha_sweeps as alsw
import linear_regression.regression_numerics.data_generation as dg
import linear_regression.aux_functions.prior_regularization_funcs as priors
import linear_regression.aux_functions.likelihood_channel_functions as like
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from linear_regression.utils.errors import ConvergenceError

alpha, delta_in, percentage, beta = 10.0, 1.0, 0.3, 0.0
delta_out_min, delta_out_max, n_delta_out_pts = 0.01, 10, 250
repetitions = 1000
d = 200

delta_out_list = np.logspace(np.log10(delta_out_min), np.log10(delta_out_max), n_delta_out_pts)

idx = np.linspace(0, len(delta_out_list)-1, 30, dtype=int)

delta_large_list_sim =  delta_out_list[idx]

means_estimation_error = []
stds_estimation_error = []

means_gen_error = []
stds_gen_error = []

delta_outs_2 = []

for idx, delta_out in enumerate(tqdm(delta_large_list_sim)):
    print(f"eps = {percentage}, alpha = {alpha}, delta_in = {delta_in}, delta_out = {delta_out}, beta = {beta}")

    all_gen_errors = []  # np.empty((repetitions,))
    all_estim_errors = []  # np.empty((repetitions,))

    for _ in range(repetitions):
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
                abs_tol=1e-3
            )

            # two lines need to be checked
            all_gen_errors.append(
                np.mean((ys_test - (1 - percentage + percentage * beta) * xs_test @ estimated_theta / np.sqrt(d)) ** 2)
                - np.mean(
                    (ys_test - (1 - percentage + percentage * beta) * xs_test @ ground_truth_theta / np.sqrt(d)) ** 2
                )
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

    delta_outs_2.append(delta_out)
    means_estimation_error.append(np.mean(all_estim_errors))
    stds_estimation_error.append(np.std(all_estim_errors) / np.sqrt(repetitions))

# print(means_estimation_error, stds_estimation_error)
fname = f"GAMP_delta_out_sweep_gen_error_d{d}_reps_{repetitions}_tol_1e-3.npz"
np.savez(
    fname,
    delta_outs=delta_outs_2,
    means_gen_error=means_gen_error,
    stds_gen_error=stds_gen_error,
    means_estimation_error=means_estimation_error,
    stds_estimation_error=stds_estimation_error,
)

dat_sim = np.load(fname)

dat_huber = np.load("delta_out_sweep_Huber.npz")
dat_l1 = np.load("delta_out_sweep_L1.npz")
dat_l2 = np.load("delta_out_sweep_L2.npz")
dat_BO = np.load("delta_out_sweep_BO.npz")


plt.errorbar(dat_sim["delta_outs"], dat_sim["means_gen_error"], yerr=dat_sim["stds_gen_error"], marker=None, color="tab:blue", linestyle=None)

plt.plot(dat_huber["delta_outs"], dat_huber["e_gen"], color="tab:orange", linestyle=None)
plt.plot(dat_l1["delta_outs"], dat_l1["e_gen"], color="tab:green", linestyle=None)
plt.plot(dat_l2["delta_outs"], dat_l2["e_gen"], color="tab:red", linestyle=None)
plt.plot(dat_BO["delta_outs"], dat_BO["gen_error"], color="tab:purple", linestyle=None)

plt.xscale("log")

plt.show()