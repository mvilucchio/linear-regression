import linear_regression.sweeps.alpha_sweeps as alsw
import matplotlib.pyplot as plt
import numpy as np
from linear_regression.fixed_point_equations.classification.Hinge_loss import (
    f_hat_Hinge_single_noise_classif,
    f_hat_Hinge_no_noise_classif,
)
from tqdm.auto import tqdm
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.misc import angle_teacher_student
import linear_regression.regression_numerics.data_generation as dg
import linear_regression.regression_numerics.erm_solvers as erm


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


alpha_min, alpha_max, n_alpha_pts = 0.1, 100, 100
reg_params = [0.1, 1.0, 10.0, 100.0]

fname = "./simulations/data/{}_classification_fixed_lambda_{:.2e}_delta_{:.2e}"

plt.figure(figsize=(7.5, 7.5))
for reg_param in reg_params:
    (
        alphas_Hinge,
        (angle_ts_Hinge, sigmas_Hinge, qs_Hinge, ms_Hinge),
    ) = alsw.sweep_alpha_fixed_point(
        f_L2_reg,
        f_hat_Hinge_no_noise_classif,
        alpha_min,
        alpha_max,
        n_alpha_pts,
        {"reg_param": reg_param},
        {},
        initial_cond_fpe=(0.1, 0.1, 0.9),
        funs=[angle_teacher_student, sigma_order_param, q_order_param, m_order_param],
        funs_args=[{}, {}, {}, {}],
    )

    plt.plot(alphas_Hinge, angle_ts_Hinge, "-", label=r"$\lambda = {:.2f}$".format(reg_param))


d = 1000
alphas_num = np.logspace(-1, 1.1, 10)

for reg_param in reg_params:
    e_gen_num = np.empty_like(alphas_num)

    for alpha_idx, alpha in enumerate(alphas_num):
        print(alpha)
        xs, ys, xs_test, ys_test, ground_truth_theta = dg.data_generation(
            dg.measure_gen_no_noise_clasif,
            n_features=d,
            n_samples=max(int(np.around(d * alpha)), 1),
            n_generalization=2 * d,
            measure_fun_args=list(),
        )

        estimated_theta = erm.find_coefficients_Hinge(ys, xs, reg_param)

        e_gen_num[alpha_idx] = ((np.sign(xs_test @ estimated_theta) - ys_test) ** 2 / 4).mean()

        del xs
        del ys
        del xs_test
        del ys_test
        del estimated_theta
        del ground_truth_theta

    plt.plot(alphas_num, e_gen_num, ".", label=r"$\lambda = {:.2f}$".format(reg_param))


plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\theta / \pi$")
plt.xscale("log")
plt.yscale("log")
plt.grid(which="both")
plt.legend()

plt.show()
