import linear_regression.regression_numerics.data_generation as dg
from linear_regression.aux_functions.prior_regularization_funcs import (
    f_w_projection_on_sphere,
    Df_w_projection_on_sphere,
)
from linear_regression.fixed_point_equations.classification.Hinge_loss import f_hat_Hinge_no_noise_classif
from linear_regression.fixed_point_equations.regularisation.fpe_projection_denoising import (
    f_projection_denoising,
)
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder, fixed_point_finder_loser
from linear_regression.aux_functions.misc import damped_update, sample_vector_informed, sample_vector_random
from linear_regression.aux_functions.stability_functions import stability_Hinge_no_noise_classif
from linear_regression.aux_functions.likelihood_channel_functions import f_out_Hinge, Df_out_Hinge
from linear_regression.aux_functions.training_errors import training_error_Hinge_loss_no_noise
from linear_regression.aux_functions.loss_functions import hinge_loss
from linear_regression.regression_numerics.amp_funcs import GAMP_algorithm_unsimplified_mod_3
from linear_regression.utils.errors import ConvergenceError
from linear_regression.regression_numerics.numerics import train_error_data, angle_teacher_student

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

abs_tol = 1e-7
min_iter = 200
max_iter_se = 5_000
blend = 0.85

n_features = 2500
d = n_features
repetitions = 5
max_iters_amp = 2_000

alpha = 2.0
n_samples = max(int(np.around(n_features * alpha)), 1)
q_min, q_max, n_q_pts_amp, n_q_pts_se = 1.0, 4.8, 10, 300

qs_amp = list()
gen_err_mean_amp = list()
gen_err_std_amp = list()
train_err_mean_amp = list()
train_err_std_amp = list()
iters_nb_mean_amp = list()
iters_nb_std_amp = list()

qs_amp_test = np.logspace(np.log10(q_min), np.log10(q_max), n_q_pts_amp)

for idx, q in enumerate(qs_amp_test):
    print(f"--- q = {q}")
    all_gen_err = list()
    all_train_err = list()
    all_iters_nb = list()

    for _ in range(repetitions):
        xs, ys, xs_test, ys_test, ground_truth_theta = dg.data_generation(
            dg.measure_gen_no_noise_clasif,
            n_features=n_features,
            n_samples=n_samples,
            n_generalization=n_samples,
            measure_fun_args=list(),
        )

        # while True:
        #     m_init = 10 * np.random.random() + 0.01
        #     sigma_init = 10 * np.random.random() + 0.01
        #     if np.square(m_init) < q:
        #         break

        # m, q, sigma = fixed_point_finder_loser(
        #     f_projection_denoising,
        #     f_hat_Hinge_no_noise_classif,
        #     (m_init, q, sigma_init),
        #     {"q_fixed": q},
        #     {"alpha": alpha},
        #     abs_tol=abs_tol,
        #     min_iter=min_iter,
        #     max_iter=max_iter_se,
        #     control_variate=(True, True, False),
        # )

        while True:
            m = 10 * np.random.random() + 0.01
            sigma = 10 * np.random.random() + 0.01
            if np.square(m) < q:
                break

        iter_nb = 0
        err = 100.0
        # print(type(err), type(abs_tol), type(iter_nb), type(min_iter))
        while err > abs_tol or iter_nb < min_iter:
            m_hat, q_hat, Σ_hat = f_hat_Hinge_no_noise_classif(m, q, sigma, alpha)
            new_m, _, new_sigma = f_projection_denoising(m_hat, q_hat, Σ_hat, q)

            # if q >= 4.8:
            #     err = abs(new_m - m)
            # else:
            err = max(abs(new_m - m), abs(new_sigma - sigma))

            m = damped_update(new_m, m, blend)
            sigma = damped_update(new_sigma, sigma, blend)

            iter_nb += 1
            if iter_nb > max_iter_se:
                break
                raise ConvergenceError("fixed_point_finder", iter_nb)

        # here I am initialising very close the fixed point
        # init_w = sample_vector_informed(ground_truth_theta, m, q)
        init_w = sample_vector_random(n_features, n_features * q)

        print(
            "\tm = {:.2f} true m = {:.2f} Δm = {:.2e} q = {:.2f} true q = {:.2f} Δq = {:.2e} Σ = {:.2f}".format(
                m,
                np.mean(init_w * ground_truth_theta),
                abs(m - np.mean(init_w * ground_truth_theta)),
                q,
                np.mean(init_w**2),
                abs(q - np.mean(init_w**2)),
                sigma,
            )
        )

        estimated_theta, iters_nb = GAMP_algorithm_unsimplified_mod_3(
            sigma,
            f_w_projection_on_sphere,
            Df_w_projection_on_sphere,
            f_out_Hinge,
            Df_out_Hinge,
            ys,
            xs,
            (q,),
            list(),
            init_w,
            ground_truth_theta,
            abs_tol=1e-2,
            max_iter=max_iters_amp,
            blend=1.0,
        )

        print("\titer_nb", iters_nb)

        if iters_nb != max_iters_amp + 1:
            print("here")
            all_gen_err.append(angle_teacher_student(ys_test, xs_test, estimated_theta, ground_truth_theta))
            all_train_err.append(train_error_data(ys, xs, estimated_theta, ground_truth_theta, hinge_loss, list()))
            all_iters_nb.append(iters_nb)

        del xs
        del ys
        del xs_test
        del ys_test
        del ground_truth_theta

    if len(all_gen_err) == 0:
        continue

    qs_amp.append(q)
    gen_err_mean_amp.append(np.mean(all_gen_err))
    gen_err_std_amp.append(np.std(all_gen_err))
    train_err_mean_amp.append(np.mean(all_train_err))
    train_err_std_amp.append(np.std(all_train_err))
    iters_nb_mean_amp.append(np.mean(all_iters_nb))
    iters_nb_std_amp.append(np.std(all_iters_nb))

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

print("--- finished AMP")

qs = np.logspace(np.log10(q_min), np.log10(5), n_q_pts_se)
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
    if np.square(m) < q:
        break
for idx, q in enumerate(tqdm(qs)):
    # print(f"--- q SE = {q}")
    try:
        iter_nb = 0
        err = 100.0
        # print(type(err), type(abs_tol), type(iter_nb), type(min_iter))
        while err > abs_tol or iter_nb < min_iter:
            m_hat, q_hat, Σ_hat = f_hat_Hinge_no_noise_classif(m, q, sigma, alpha)
            new_m, _, new_sigma = f_projection_denoising(m_hat, q_hat, Σ_hat, q)
            if np.isnan(new_m):
                print("m_hat = {:.2e} q_hat = {:.2e} Σ_hat = {:.2e} q = {:.2f}".format(m_hat, q_hat, Σ_hat, q))
            # if q >= 4.8:
            #     err = abs(new_m - m)
            # else:
            err = max(abs(new_m - m), abs(new_sigma - sigma))

            m = damped_update(new_m, m, blend)
            sigma = damped_update(new_sigma, sigma, blend)

            iter_nb += 1
            if iter_nb > max_iter_se:
                break
                raise ConvergenceError("fixed_point_finder", iter_nb)

        ms[idx] = m
        sigmas[idx] = sigma
        m_hats[idx] = m_hat
        Σ_hats[idx] = Σ_hat
        q_hats[idx] = q_hat

        training_error[idx] = training_error_Hinge_loss_no_noise(ms[idx], qs[idx], sigmas[idx])
    except (ConvergenceError, ValueError) as e:
        print(e)
        ms[idx:] = np.nan
        sigmas[idx:] = np.nan
        m_hats[idx:] = np.nan
        Σ_hats[idx:] = np.nan
        q_hats[idx:] = np.nan
        training_error[idx:] = np.nan
        break

# save the results of the AMP experiment in a csv file in the folder simulations/data
# np.savetxt(
#     "./simulations/data/projection_amp_sweep_q_L1.csv",
#     np.array([gen_err_q, gen_err_mean, gen_err_std, train_err_mean, train_err_std]).T,
#     delimiter=",",
#     header="q,gen_err_mean,gen_err_std,train_err_mean,train_err_std",
# )

# Plotting

plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.title("Hinge loss Proj. No Noise" + " d = {:d}".format(n_features) + r" $\alpha$ = " + "{:.2f}".format(alpha))

color = next(plt.gca()._get_lines.prop_cycler)["color"]
plt.errorbar(
    qs_amp,
    gen_err_mean_amp,
    yerr=gen_err_std_amp,
    label="AMP Gen Err reps={:d}".format(repetitions),
    color=color,
    linestyle="",
    marker=".",
)
plt.plot(qs, np.arccos(ms / np.sqrt(qs)) / np.pi, label="Theoretical Gen Err", color=color, linestyle="-")

color = next(plt.gca()._get_lines.prop_cycler)["color"]
plt.errorbar(
    qs_amp,
    train_err_mean_amp,
    yerr=train_err_std_amp,
    label="AMP Training Error reps={:d}".format(repetitions),
    color=color,
    linestyle="",
    marker=".",
)
plt.plot(qs, training_error, label="Theoretical Training Error", color=color, linestyle="-")

stab_vals = np.array([stability_Hinge_no_noise_classif(m, q, sigma, alpha) for m, q, sigma in zip(ms, qs, sigmas)])
plt.plot(
    qs,
    stab_vals,
    label="Stability",
)

plt.xlabel("q")
plt.xscale("log")
# plt.ylim(-0.5, 7.5)
plt.legend()
plt.grid()

plt.subplot(212)
plt.errorbar(
    qs_amp,
    iters_nb_mean_amp,
    yerr=iters_nb_std_amp,
    label="AMP reps={:d}".format(repetitions),
    linestyle="",
    marker=".",
)

plt.xlabel("q")
plt.ylabel("iterations")
plt.yscale("log")
plt.xscale("log")
# plt.ylim(-0.5, 7.5)
plt.legend()
plt.grid()

plt.show()
