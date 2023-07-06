import linear_regression.regression_numerics.amp_funcs as amp
from linear_regression.sweeps.q_sweeps import sweep_fw_first_arg_GAMP
import linear_regression.regression_numerics.data_generation as dg
from linear_regression.aux_functions.prior_regularization_funcs import (
    f_w_projection_on_sphere,
    Df_w_projection_on_sphere,
)
from linear_regression.fixed_point_equations.fpe_L1_loss import f_hat_L1_decorrelated_noise
from linear_regression.fixed_point_equations.regularisation.fpe_projection_denoising import (
    f_projection_denoising,
)
from linear_regression.aux_functions.misc import damped_update
from linear_regression.aux_functions.likelihood_channel_functions import f_out_L1, Df_out_L1
from linear_regression.aux_functions.loss_functions import l1_loss
from linear_regression.aux_functions.stability_functions import (
    stability_l1_l2,
    stability_huber,
    stability_ridge,
)
from linear_regression.aux_functions.training_errors import training_error_l1_loss
from linear_regression.regression_numerics.amp_funcs import (
    GAMP_algorithm_unsimplified,
    GAMP_algorithm_unsimplified_mod,
    GAMP_algorithm_unsimplified_mod_2,
    GAMP_algorithm_unsimplified_mod_3,
)
import numpy as np
import matplotlib.pyplot as plt
import linear_regression.fixed_point_equations as fpe
from linear_regression.utils.errors import ConvergenceError
from linear_regression.regression_numerics.numerics import gen_error_data, train_error_data

print("here")

abs_tol = 1e-8
min_iter = 100
max_iter = 10000
blend = 0.85

# f_out_args = (1.0, 5.0, 0.3, 0.0)

n_features = 2500
d = n_features
repetitions = 10

alpha, delta_in, delta_out, percentage, beta = 2.0, 1.0, 5.0, 0.3, 0.0
n_samples = max(int(np.around(n_features * alpha)), 1)

qs_amp = list()
gen_err_mean_amp = list()
gen_err_std_amp = list()
train_err_mean_amp = list()
train_err_std_amp = list()
iters_nb_mean_amp = list()
iters_nb_std_amp = list()

# qs_amp_test = np.logspace(-1, 1, 25)

# for idx, q in enumerate(qs_amp_test):
#     print(f"--- q = {q}")
#     all_gen_err = list()
#     all_train_err = list()
#     all_iters_nb = list()

#     for _ in range(repetitions):
#         xs, ys, _, _, ground_truth_theta = dg.data_generation(
#             dg.measure_gen_decorrelated,
#             n_features=n_features,
#             n_samples=max(int(np.around(n_features * alpha)), 1),
#             n_generalization=1,
#             measure_fun_args=(delta_in, delta_out, percentage, beta),
#         )

#         while True:
#             m = 10 * np.random.random() + 0.01
#             sigma = 10 * np.random.random() + 0.01
#             if np.square(m) < q + delta_in * q and np.square(m) < q + delta_out * q:
#                 break

#         iter_nb = 0
#         err = 100.0
#         while err > abs_tol or iter_nb < min_iter:
#             m_hat, q_hat, Σ_hat = f_hat_L1_decorrelated_noise(
#                 m, q, sigma, alpha, delta_in, delta_out, percentage, beta
#             )
#             new_m, _, new_sigma = f_projection_denoising(m_hat, q_hat, Σ_hat, q)

#             err = max([abs(new_m - m), abs(new_sigma - sigma)])

#             m = damped_update(new_m, m, blend)
#             sigma = damped_update(new_sigma, sigma, blend)

#             iter_nb += 1
#             if iter_nb > max_iter:
#                 raise ConvergenceError("fixed_point_finder", iter_nb)

#         print(d * q, d**2 * m**2)
#         # we want to initialize them at the fixed point so:
#         estimated_theta, iters_nb = GAMP_algorithm_unsimplified_mod_3(
#             sigma,
#             f_w_projection_on_sphere,
#             Df_w_projection_on_sphere,
#             f_out_L1,
#             Df_out_L1,
#             ys,
#             xs,
#             (q,),
#             list(),
#             m * ground_truth_theta + np.sqrt(q - m**2) * np.random.normal(size=n_features),
#             ground_truth_theta,
#             abs_tol=1e-2,
#             max_iter=5000,
#             blend=1.0,
#         )

#         print("iter_nb", iters_nb)

#         all_gen_err.append(gen_error_data(ys, xs, estimated_theta, ground_truth_theta))

#         all_train_err.append(
#             train_error_data(ys, xs, estimated_theta, ground_truth_theta, l1_loss, list())
#         )

#         all_iters_nb.append(iters_nb)

#         del xs
#         del ys
#         del ground_truth_theta

#     qs_amp.append(q)
#     gen_err_mean_amp.append(np.mean(all_gen_err))
#     gen_err_std_amp.append(np.std(all_gen_err))
#     train_err_mean_amp.append(np.mean(all_train_err))
#     train_err_std_amp.append(np.std(all_train_err))
#     iters_nb_mean_amp.append(np.mean(all_iters_nb))
#     iters_nb_std_amp.append(np.std(all_iters_nb))

# # save the results of AMP in a file with the delta_in, delta_out, percentage, beta, alpha,n_features, repetitions parameters
# np.savetxt(
#     f"./results/AMP_results_decorrelated_noise_{delta_in}_{delta_out}_{percentage}_{beta}_{alpha}_{n_features}_{repetitions}.csv",
#     np.array(
#         [
#             qs_amp,
#             gen_err_mean_amp,
#             gen_err_std_amp,
#             train_err_mean_amp,
#             train_err_std_amp,
#             iters_nb_mean_amp,
#             iters_nb_std_amp,
#         ]
#     ).T,
#     delimiter=",",
#     header="q,gen_err_mean,gen_err_std,train_err_mean,train_err_std,iters_nb_mean,iters_nb_std",
# )

# load the data from the file in ./results with 2500
data_2500 = np.loadtxt(
    f"./results/AMP_results_decorrelated_noise_20_params_{delta_in}_{delta_out}_{percentage}_{beta}_{alpha}_2500_{repetitions}.csv",
    delimiter=",",
    skiprows=1,
)

qs_amp_2500 = data_2500[:, 0]
gen_err_mean_amp_2500 = data_2500[:, 1]
gen_err_std_amp_2500 = data_2500[:, 2]
train_err_mean_amp_2500 = data_2500[:, 3]
train_err_std_amp_2500 = data_2500[:, 4]
iters_nb_mean_amp_2500 = data_2500[:, 5]
iters_nb_std_amp_2500 = data_2500[:, 6]

# load the data from the file in ./results with 1500
data_1500 = np.loadtxt(
    f"./results/AMP_results_decorrelated_noise_20_params_{delta_in}_{delta_out}_{percentage}_{beta}_{alpha}_1500_{repetitions}.csv",
    delimiter=",",
    skiprows=1,
)

qs_amp_1500 = data_1500[:, 0]
gen_err_mean_amp_1500 = data_1500[:, 1]
gen_err_std_amp_1500 = data_1500[:, 2]
train_err_mean_amp_1500 = data_1500[:, 3]
train_err_std_amp_1500 = data_1500[:, 4]
iters_nb_mean_amp_1500 = data_1500[:, 5]
iters_nb_std_amp_1500 = data_1500[:, 6]

# load the data from the file in ./results with 2000
data_2000 = np.loadtxt(
    f"./results/AMP_results_decorrelated_noise_20_params_{delta_in}_{delta_out}_{percentage}_{beta}_{alpha}_2000_{repetitions}.csv",
    delimiter=",",
    skiprows=1,
)

qs_amp_2000 = data_2000[:, 0]
gen_err_mean_amp_2000 = data_2000[:, 1]
gen_err_std_amp_2000 = data_2000[:, 2]
train_err_mean_amp_2000 = data_2000[:, 3]
train_err_std_amp_2000 = data_2000[:, 4]
iters_nb_mean_amp_2000 = data_2000[:, 5]
iters_nb_std_amp_2000 = data_2000[:, 6]

data_5000 = np.loadtxt(
    f"./results/AMP_results_decorrelated_noise_20_params_{delta_in}_{delta_out}_{percentage}_{beta}_{alpha}_5000_6.csv",
    delimiter=",",
    skiprows=1,
)

qs_amp_5000 = data_5000[:, 0]
gen_err_mean_amp_5000 = data_5000[:, 1]
gen_err_std_amp_5000 = data_5000[:, 2]
train_err_mean_amp_5000 = data_5000[:, 3]
train_err_std_amp_5000 = data_5000[:, 4]
iters_nb_mean_amp_5000 = data_5000[:, 5]
iters_nb_std_amp_5000 = data_5000[:, 6]

qs = np.logspace(-1, 1, 500)
ms = np.empty_like(qs)
sigmas = np.empty_like(qs)
m_hats = np.empty_like(qs)
q_hats = np.empty_like(qs)
Σ_hats = np.empty_like(qs)
training_error = np.empty_like(qs)

q = qs[0]
while True:
    m = 10 * np.random.random() + 0.01
    sigma = 10 * np.random.random() + 0.01
    if np.square(m) < q + delta_in * q and np.square(m) < q + delta_out * q:
        break
for idx, q in enumerate(qs):
    try:
        iter_nb = 0
        err = 100.0
        while err > abs_tol or iter_nb < min_iter:
            m_hat, q_hat, Σ_hat = f_hat_L1_decorrelated_noise(
                m, q, sigma, alpha, delta_in, delta_out, percentage, beta
            )
            new_m, _, new_sigma = f_projection_denoising(m_hat, q_hat, Σ_hat, q)

            err = max([abs(new_m - m), abs(new_sigma - sigma)])

            m = damped_update(new_m, m, blend)
            sigma = damped_update(new_sigma, sigma, blend)

            iter_nb += 1
            if iter_nb > max_iter:
                raise ConvergenceError("fixed_point_finder", iter_nb)

        ms[idx] = m
        sigmas[idx] = sigma
        m_hats[idx] = m_hat
        Σ_hats[idx] = Σ_hat
        q_hats[idx] = q_hat

        training_error[idx] = training_error_l1_loss(
            m, q, sigma, delta_in, delta_out, percentage, beta
        )
    except (ConvergenceError, ValueError) as e:
        print(e)
        ms[idx:] = np.nan
        sigmas[idx:] = np.nan
        m_hats[idx:] = np.nan
        Σ_hats[idx:] = np.nan
        q_hats[idx:] = np.nan
        training_error[idx:] = np.nan
        break


plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.title(
    "L1 loss Projective Denoiser"
    + " d = {:d}".format(n_features)
    + r" $\alpha$ = "
    + "{:.2f}".format(alpha)
    + r" $\Delta_{in}$ = "
    + "{:.2f}".format(delta_in)
    + r" $\Delta_{out}$ ="
    + "{:.2f}".format(delta_out)
    + r" $\epsilon$ = "
    + "{:.2f}".format(percentage)
    + r" $\beta$ = "
    + "{:.2f}".format(beta)
    + r" $\alpha$ = "
    + "{:.2f}".format(alpha)
)

color = next(plt.gca()._get_lines.prop_cycler)["color"]
plt.errorbar(
    qs_amp_2500,
    gen_err_mean_amp_2500,
    yerr=gen_err_std_amp_2500,
    label="AMP Gen. Error d={:d} reps={:d}".format(2500, repetitions),
    color=color,
    linestyle="",
    marker=".",
)
# plt.errorbar(
#     qs_amp_1500,
#     gen_err_mean_amp_1500,
#     yerr=gen_err_std_amp_1500,
#     label="AMP Gen. Error d={:d} reps={:d}".format(1500, repetitions),
#     color=color,
#     linestyle="",
#     marker=".",
# )
# plt.errorbar(
#     qs_amp_2000,
#     gen_err_mean_amp_2000,
#     yerr=gen_err_std_amp_2000,
#     label="AMP Gen. Error d={:d} reps={:d}".format(2000, repetitions),
#     color=color,
#     linestyle="",
#     marker=".",
# )
plt.errorbar(
    qs_amp_5000,
    gen_err_mean_amp_5000,
    yerr=gen_err_std_amp_5000,
    label="AMP Gen. Error d={:d} reps={:d}".format(5000, 6),
    color=color,
    linestyle="",
    marker=".",
)
plt.plot(qs, 1 + qs - 2 * ms, label="Theoretical Generalization Error", color=color, linestyle="-")

color = next(plt.gca()._get_lines.prop_cycler)["color"]
plt.errorbar(
    qs_amp_2500,
    train_err_mean_amp_2500,
    yerr=train_err_std_amp_2500,
    label="AMP Train. Error d={:d} reps={:d}".format(2500, repetitions),
    color=color,
    linestyle="",
    marker=".",
)
# plt.errorbar(
#     qs_amp_1500,
#     train_err_mean_amp_1500,
#     yerr=train_err_std_amp_1500,
#     label="AMP Train. Error d={:d} reps={:d}".format(1500, repetitions),
#     color=color,
#     linestyle="",
#     marker=".",
# )
# plt.errorbar(
#     qs_amp_2000,
#     train_err_mean_amp_2000,
#     yerr=train_err_std_amp_2000,
#     label="AMP Train. Error d={:d} reps={:d}".format(2000, repetitions),
#     color=color,
#     linestyle="",
#     marker=".",
# )
plt.errorbar(
    qs_amp_5000,
    train_err_mean_amp_5000,
    yerr=train_err_std_amp_5000,
    label="AMP Train. Error d={:d} reps={:d}".format(5000, 6),
    color=color,
    linestyle="",
    marker=".",
)
plt.plot(qs, training_error, label="Theoretical Training Error", color=color, linestyle="-")
stab_vals = stability_l1_l2(ms, qs, sigmas, alpha, 1.0, delta_in, delta_out, percentage, beta)
plt.plot(
    qs,
    stab_vals,
    label="stability",
)
idx = np.amin(np.arange(len(stab_vals))[stab_vals < 0])
plt.axvline(qs[idx], color="black", linestyle="--", label="stability threshold")


plt.xlabel("q")
# plt.yscale("log")
plt.xscale("log")
plt.ylim(-0.5, 7.5)
plt.legend()
plt.grid()


plt.subplot(212)
# plt.errorbar(
#     qs_amp_1500,
#     iters_nb_mean_amp_1500,
#     yerr=iters_nb_std_amp_1500,
#     label="AMP iterations d={:d} reps={:d}".format(1500, repetitions),
#     linestyle="",
#     marker=".",
# )
# plt.errorbar(
#     qs_amp_2000,
#     iters_nb_mean_amp_2000,
#     yerr=iters_nb_std_amp_2000,
#     label="AMP iterations d={:d} reps={:d}".format(2000, repetitions),
#     linestyle="",
#     marker=".",
# )
plt.errorbar(
    qs_amp_2500,
    iters_nb_mean_amp_2500,
    yerr=iters_nb_std_amp_2500,
    label="AMP iterations d={:d} reps={:d}".format(2500, repetitions),
    linestyle="",
    marker=".",
)
plt.errorbar(
    qs_amp_5000,
    iters_nb_mean_amp_5000,
    yerr=iters_nb_std_amp_5000,
    label="AMP iterations d={:d} reps={:d}".format(5000, 6),
    linestyle="",
    marker=".",
)

plt.xlabel("q")
plt.ylabel("iterations")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.grid()

plt.show()
