import linear_regression.sweeps.alpha_sweeps as alsw
import matplotlib.pyplot as plt
import numpy as np
from linear_regression.fixed_point_equations.classification.Hinge_loss import (
    f_hat_Hinge_no_noise_classif,
)
from linear_regression.aux_functions.training_errors import (
    training_error_Hinge_loss_no_noise,
)
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.misc import angle_teacher_student
from linear_regression.fixed_point_equations.classification.BO import (
    f_BO,
    f_hat_BO_no_noise_classif,
)
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder_adversiaral
from linear_regression.fixed_point_equations.regularisation.Linf_attacks_Lr_reg import (
    f_Lr_reg_Linf_attack,
)
from linear_regression.fixed_point_equations.classification.Adv_train_p_norm import (
    f_hat_Logistic_no_noise_Linf_adv_classif,
)
import pickle
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize as jax_minimize


def log1pexp_jax(x):
    return jnp.where(
        x <= -37,
        jnp.exp(x),  # For x <= -37
        jnp.where(
            x <= -2,
            jnp.log1p(jnp.exp(x)),  # For -37 < x <= -2
            jnp.where(
                x <= 18,
                jnp.log(1.0 + jnp.exp(x)),  # For -2 < x <= 18
                jnp.where(
                    x <= 33.3, jnp.exp(-x) + x, x  # For 18 < x <= 33.3  # For x > 33.3
                ),
            ),
        ),
    )


def _loss_Logistic_adv_Linf(w, xs_norm, ys, reg_param, eps_t, reg_order):
    n, d = xs_norm.shape
    margin = ys * jnp.dot(xs_norm, w) / jnp.sqrt(d)
    perturbed_margin = margin - (eps_t / d) * jnp.sum(jnp.abs(w))
    loss = jnp.sum(log1pexp_jax(-perturbed_margin)) + reg_param * jnp.sum(
        jnp.abs(w) ** reg_order
    )
    return loss


def find_coefficients_Logistic_adv_Linf(ys, xs, reg_param, eps_t, reg_order):
    _, d = xs.shape
    w = np.random.normal(loc=0.0, scale=1.0, size=(d,))
    xs_norm = np.divide(xs, np.sqrt(d))

    opt_res = jax_minimize(
        _loss_Logistic_adv_Linf,
        w,
        method="BFGS",
        args=(xs_norm, ys, reg_param, eps_t, reg_order),
        options={"maxiter": 1000},
    )

    if opt_res.status == 2:
        raise ValueError(
            "LogisticRegressor convergence failed: l-BFGS solver terminated with %s"
            % opt_res.message
        )

    return opt_res.x


alpha_min, alpha_max, n_alpha_pts = 0.01, 1, 12
reg_orders = [
    1,
    2,
    3,
    4,
]  # [2, 3, 4, 5]
eps_t = 0.1
eps_g = 0.1
reg_param = 1e-4

run_experiments = False

d = 1000
reps = 10
n_gen = 1000

if __name__ == "__main__":
    if run_experiments:
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)

        q_mean = np.empty((len(reg_orders), n_alpha_pts))
        q_std = np.empty((len(reg_orders), n_alpha_pts))

        m_mean = np.empty((len(reg_orders), n_alpha_pts))
        m_std = np.empty((len(reg_orders), n_alpha_pts))

        train_error_mean = np.empty((len(reg_orders), n_alpha_pts))
        train_error_std = np.empty((len(reg_orders), n_alpha_pts))

        gen_error_mean = np.empty((len(reg_orders), n_alpha_pts))
        gen_error_std = np.empty((len(reg_orders), n_alpha_pts))

        estim_errors_mean = np.empty((len(reg_orders), n_alpha_pts))
        estim_errors_std = np.empty((len(reg_orders), n_alpha_pts))

        adversarial_errors_mean = np.empty((len(reg_orders), n_alpha_pts))
        adversarial_errors_std = np.empty((len(reg_orders), n_alpha_pts))

        for i, reg_order in enumerate(reg_orders):
            for j, alpha in enumerate(alphas):
                print(f"Running reg_order = {reg_order}, alpha = {alpha}")
                n = int(alpha * d)

                tmp_estim_errors = []
                tmp_train_errors = []
                tmp_gen_errors = []
                tmp_adversarial_errors = []
                tmp_qs = []
                tmp_ms = []

                for _ in range(reps):
                    xs_train = np.random.normal(loc=0.0, scale=1.0, size=(n, d))
                    wstar = np.random.normal(loc=0.0, scale=1.0, size=(d,))
                    ys_train = np.sign(xs_train @ wstar)

                    w = find_coefficients_Logistic_adv_Linf(
                        ys_train, xs_train, reg_param, eps_t, reg_order
                    )

                    tmp_estim_errors.append(np.sum((w - wstar) ** 2 / d))
                    tmp_qs.append(np.sum(w**2) / d)
                    tmp_ms.append(np.dot(wstar, w) / d)
                    tmp_train_errors.append(
                        np.sum(
                            np.where(
                                ys_train
                                != np.sign(xs_train @ w - eps_t * np.sum(np.abs(w))),
                                1,
                                0,
                            )
                        )
                        / n
                    )

                    xs_gen = np.random.normal(loc=0.0, scale=1.0, size=(n_gen, d))
                    ys_gen = np.sign(xs_gen @ wstar)

                    tmp_gen_errors.append(
                        np.sum(np.where(ys_gen != np.sign(xs_gen @ w), 1, 0)) / n_gen
                    )

                    tmp_adversarial_errors.append(
                        np.sum(
                            np.where(
                                ys_gen
                                != np.sign(xs_gen @ w - eps_g * np.sum(np.abs(w))),
                                1,
                                0,
                            )
                        )
                        / n_gen
                    )

                estim_errors_mean[i, j] = np.mean(tmp_estim_errors)
                estim_errors_std[i, j] = np.std(tmp_estim_errors)

                q_mean[i, j] = np.mean(tmp_qs)
                q_std[i, j] = np.std(tmp_qs)

                m_mean[i, j] = np.mean(tmp_ms)
                m_std[i, j] = np.std(tmp_ms)

                train_error_mean[i, j] = np.mean(tmp_train_errors)
                train_error_std[i, j] = np.std(tmp_train_errors)

                gen_error_mean[i, j] = np.mean(tmp_gen_errors)
                gen_error_std[i, j] = np.std(tmp_gen_errors)

                adversarial_errors_mean[i, j] = np.mean(tmp_adversarial_errors)
                adversarial_errors_std[i, j] = np.std(tmp_adversarial_errors)

            data_dict = {
                "alphas": alphas,
                "q_mean": q_mean[i],
                "q_std": q_std[i],
                "m_mean": m_mean[i],
                "m_std": m_std[i],
                "train_error_mean": train_error_mean[i],
                "train_error_std": train_error_std[i],
                "gen_error_mean": gen_error_mean[i],
                "gen_error_std": gen_error_std[i],
                "estim_errors_mean": estim_errors_mean[i],
                "estim_errors_std": estim_errors_std[i],
                "adversarial_errors_mean": adversarial_errors_mean[i],
                "adversarial_errors_std": adversarial_errors_std[i],
            }

            with open(
                f"ERM_data_Linf_reg_order_{reg_order:d}_dim_{d:d}_reps_{reps:d}_reg_param_{reg_param:.1e}_eps_t_g_{eps_t:.1e}_{eps_g:.1e}.pkl",
                "wb",
            ) as f:
                pickle.dump(data_dict, f)

    for i, reg_order in enumerate(reg_orders):
        with open(
            f"ERM_data_Linf_reg_order_{reg_order:d}_dim_{d:d}_reps_{reps:d}_reg_param_{reg_param:.1e}_eps_t_g_{eps_t:.1e}_{eps_g:.1e}.pkl",
            "rb",
        ) as f:
            data_dict = pickle.load(f)

        alphas = data_dict["alphas"]

        train_error_mean = data_dict["train_error_mean"]
        train_error_std = data_dict["train_error_std"]

        gen_error_mean = data_dict["gen_error_mean"]
        gen_error_std = data_dict["gen_error_std"]

        adversarial_errors_mean = data_dict["adversarial_errors_mean"]
        adversarial_errors_std = data_dict["adversarial_errors_std"]

        # plt.errorbar(
        #     alphas,
        #     gen_error_mean,
        #     yerr=gen_error_std,
        #     fmt="1",
        #     linestyle="--",
        #     label=f"GEN r = {reg_order}",
        #     # color=f"C{i}",
        # )

        plt.errorbar(
            alphas,
            adversarial_errors_mean,
            yerr=adversarial_errors_std,
            label=f"ADV r = {reg_order}",
            fmt="x",
            linestyle="-",
            # color=f"C{i}",
        )

        # plt.errorbar(
        #     alphas,
        #     train_error_mean,
        #     yerr=train_error_std,
        #     fmt=".",
        #     label=f"TRAIN r = {reg_order}",
        #     # color=f"C{i}",
        # )

    plt.title(r"L$\infty$ attack with regularisation $L r$")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$E_{\mathrm{estim}}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()

    plt.show()
