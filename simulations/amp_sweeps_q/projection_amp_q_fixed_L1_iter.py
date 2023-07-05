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

n_features = 3000
d = n_features
repetitions = 10

alpha, delta_in, delta_out, percentage, beta = 2.0, 1.0, 5.0, 0.3, 0.0
n_samples = max(int(np.around(n_features * alpha)), 1)
q = 3.2

qs_amp = list()
gen_err_mean_amp = list()
gen_err_std_amp = list()
train_err_mean_amp = list()
train_err_std_amp = list()


plt.figure(figsize=(10, 7.5))
plt.title(
    "L1 loss Projective Denoiser"
    + " d = {:d}".format(n_features)
    + r" $q$ = "
    + "{:.2f}".format(q)
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
color_q = next(plt.gca()._get_lines.prop_cycler)["color"]
color_m = next(plt.gca()._get_lines.prop_cycler)["color"]

while True:
    m = 10 * np.random.random() + 0.01
    sigma = 10 * np.random.random() + 0.01
    if np.square(m) < q + delta_in * q and np.square(m) < q + delta_out * q:
        break

iter_nb = 0
err = 100.0
while err > abs_tol or iter_nb < min_iter:
    m_hat, q_hat, sigma_hat = f_hat_L1_decorrelated_noise(
        m, q, sigma, alpha, delta_in, delta_out, percentage, beta
    )
    new_m, _, new_sigma = f_projection_denoising(m_hat, q_hat, sigma_hat, q)

    err = max([abs(new_m - m), abs(new_sigma - sigma)])

    m = damped_update(new_m, m, blend)
    sigma = damped_update(new_sigma, sigma, blend)

    iter_nb += 1
    if iter_nb > max_iter:
        raise ConvergenceError("fixed_point_finder", iter_nb)

# plt.axhline(m, color=color_m, label="m theoretical", linestyle="--")
# plt.axhline(q, color=color_q, label="q theoretical", linestyle="--")

for _ in range(repetitions):
    xs, ys, _, _, ground_truth_theta = dg.data_generation(
        dg.measure_gen_decorrelated,
        n_features=n_features,
        n_samples=max(int(np.around(n_features * alpha)), 1),
        n_generalization=1,
        measure_fun_args=(delta_in, delta_out, percentage, beta),
    )

    # we want to initialize them at the fixed point so:
    print(f"q = {q}, m = {m}")
    init_w = m * ground_truth_theta + np.sqrt(q - m**2) * np.random.normal(size=n_features)

    while not np.isclose([np.mean(init_w **2), np.mean(init_w * ground_truth_theta)], [q, m], rtol=0.0, atol=1e-1).all():
        init_w = m * ground_truth_theta + np.sqrt(q - m**2) * np.random.normal(size=n_features)

    print("found ")

    estimated_theta, q_list, m_list, prev_dot_list, eps_list = GAMP_algorithm_unsimplified_mod_2(
        sigma,
        f_w_projection_on_sphere,
        Df_w_projection_on_sphere,
        f_out_L1,
        Df_out_L1,
        ys,
        xs,
        (q,),
        list(),
        init_w, # m * ground_truth_theta + np.sqrt(q - m**2) * np.random.normal(size=n_features),
        ground_truth_theta,
        abs_tol=5e-3,
        max_iter=1000,
        blend=1.0,
    )

    # plt.plot(q_list, marker=".", color=color_q, linestyle="-")
    # plt.plot(m_list, marker=".", color=color_m, linestyle="-")
    plt.plot(prev_dot_list, marker=".", linestyle="-")
    # plt.plot(eps_list, marker=".", color="red", linestyle="-")

    del xs
    del ys
    del ground_truth_theta

plt.xlabel("iter.")
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'$\|\hat{w}^{t+1} - \hat{w}^{t} \|_2^2$')
plt.legend()
plt.grid()

plt.show()
