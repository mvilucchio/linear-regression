import matplotlib.pyplot as plt
import numpy as np
from linear_regression.data.generation import (
    measure_gen_no_noise_clasif,
    data_generation,
)
from tqdm.auto import tqdm
from linear_regression.erm.metrics import percentage_flipped_labels
from linear_regression.erm.erm_solvers import find_coefficients_Logistic
from linear_regression.fixed_point_equations.fpeqs import fixed_point_finder
from linear_regression.fixed_point_equations.classification.Logistic_loss import (
    f_hat_Logistic_no_noise_classif,
    f_hat_Logistic_probit_classif,
)
from linear_regression.fixed_point_equations.regularisation.L2_reg import f_L2_reg
from linear_regression.aux_functions.percentage_flipped import (
    percentage_flipped_direct_space,
)
from math import gamma
import jax.numpy as jnp
import jax
from jax.scipy.optimize import minimize as jax_minimize
from jax import grad, vmap
import pickle


@jax.jit
def total_loss_logistic(w, Xs, ys, reg_param):
    ys = ys.reshape(-1, 1) if ys.ndim == 1 else ys
    scores = jnp.matmul(Xs, w)
    loss_part = jnp.sum(jnp.log(1 + jnp.exp(-ys * scores)))
    reg_part = reg_param * jnp.dot(w, w)
    return loss_part + reg_part


@jax.jit
def linear_loss_function_single(x, y, w):
    prediction = jnp.dot(x, w)
    return -y * prediction


linear_loss_all = jax.jit(vmap(linear_loss_function_single, in_axes=(0, 0, None)))

grad_linear_loss_single = jax.jit(grad(linear_loss_function_single, argnums=0))

grad_linear_loss_all = jax.jit(vmap(grad_linear_loss_single, in_axes=(0, 0, None)))


@jax.jit
def then_func(ops):
    x, eps_t, norm_x_projected = ops
    return eps_t * x / norm_x_projected


@jax.jit
def else_func(ops):
    return ops[0]


@jax.jit
def project_and_normalize(x, wstar, p, eps_t):
    x -= jnp.dot(x, wstar) * wstar / jnp.dot(wstar, wstar)
    # norm_x_projected = jnp.linalg.norm(x, ord=p)
    norm_x_projected = jnp.sum(jnp.abs(x) ** p) ** (1 / p)

    return jax.lax.cond(
        norm_x_projected > eps_t, then_func, else_func, (x, eps_t, norm_x_projected)
    )


vecorized_project_and_normalize = jax.jit(
    vmap(project_and_normalize, in_axes=(0, None, None, None))
)

# if __name__ == "__main__":
#     n = 6
#     d = 3
#     xs = np.random.normal(0, 1, (n, d))
#     w = np.random.normal(0, 1, d)
#     ys = np.sign(xs @ w)

#     print(grad_linear_loss_all(xs, ys, w))

#     for i in range(n):
#         print(grad_linear_loss_single(xs[i], ys[i], w))

#     print(grad_linear_loss_all(xs, ys, w).shape)

#     p = 3
#     eps_t = 0.3

#     adv_per = vecorized_project_and_normalize(xs, w, p, eps_t)
#     print(np.linalg.norm(adv_per, ord=p, axis=1))


@jax.jit
def projected_GA_step_jit(vs, ys, w, wstar, step_size, eps, p):
    g = grad_linear_loss_all(vs, ys, w)
    return vecorized_project_and_normalize(vs + step_size * g, wstar, p, eps)


def projected_GA(ys, w, wstar, step_size, n_steps, eps, p):
    adv_perturbation = jnp.zeros((len(ys), len(w)))

    for _ in range(n_steps):
        adv_perturbation = projected_GA_step_jit(
            adv_perturbation, ys, w, wstar, step_size, eps, p
        )
        # if np.allclose(jnp.linalg.norm(adv_perturbation, ord=p, axis=1) - eps, 0.0, atol=1e-5):
        #     break
    return adv_perturbation


# if __name__ == "__main__":
#     d = 1024
#     n = 1600
#     xs = np.random.normal(0, 1, (n, d))
#     w_star = np.random.normal(0, 1, d)
#     # w_star = w_star / np.sqrt(np.sum(w_star**2))

#     xi = np.random.normal(0, 1, d)
#     xi_perp_wstar = xi
#     # xi_perp_wstar = xi - np.dot(xi, w_star) * w_star / np.dot(w_star, w_star)
#     # xi_perp_wstar = xi_perp_wstar / np.sqrt(np.sum(xi_perp_wstar**2))

#     alpha = 0.6
#     beta = 0.3
#     w = alpha * w_star + beta * xi_perp_wstar
#     eps = 300.0

#     ys = np.sign(xs @ w)
#     p = 3
#     n_steps = 500
#     adv_perturbation = np.zeros_like(xs)  # np.random.normal(0, 1, (n, d))
#     losses_p3 = np.empty((n_steps, n))

#     for i in range(n_steps):
#         adv_perturbation = projected_GA_step(
#             adv_perturbation,
#             ys,
#             w,
#             0.5,
#             lambda x: vecorized_project_and_normalize(x, w_star, p, eps),
#         )

#         losses_p3[i, :] = np.sort(linear_loss_all(adv_perturbation, ys, w))

#     print(np.allclose(adv_perturbation @ w_star, 0.0, atol=1e-3))
#     print(np.linalg.norm(adv_perturbation, ord=p, axis=1) - eps)

#     p = 2
#     adv_perturbation = np.zeros_like(xs)  # np.random.normal(0, 1, (n, d))
#     losses_p2 = np.empty((n_steps, n))

#     for i in range(n_steps):
#         adv_perturbation = projected_GA_step(
#             adv_perturbation,
#             ys,
#             w,
#             0.5,
#             lambda x: vecorized_project_and_normalize(x, w_star, p, eps),
#         )

#         losses_p2[i, :] = np.sort(linear_loss_all(adv_perturbation, ys, w))

#     print(np.allclose(adv_perturbation @ w_star, 0.0, atol=1e-3))
#     print(np.linalg.norm(adv_perturbation, ord=p, axis=1) - eps)

#     p = 5
#     adv_perturbation = np.zeros_like(xs)  # np.random.normal(0, 1, (n, d))
#     losses_p5 = np.empty((n_steps, n))

#     for i in range(n_steps):
#         adv_perturbation = projected_GA_step(
#             adv_perturbation,
#             ys,
#             w,
#             0.5,
#             lambda x: vecorized_project_and_normalize(x, w_star, p, eps),
#         )

#         losses_p5[i, :] = np.sort(linear_loss_all(adv_perturbation, ys, w))

#     print(np.allclose(adv_perturbation @ w_star, 0.0, atol=1e-3))
#     print(np.linalg.norm(adv_perturbation, ord=p, axis=1) - eps)

#     plt.plot(losses_p3[:,0], "-", label="p = 3")
#     plt.plot(losses_p2[:,0], "--", label="p = 2")
#     plt.plot(losses_p5[:,0], "-.", label="p = 5")
#     plt.legend()
#     plt.show()


if __name__ == "__main__":
    alpha = 1.5
    reg_param = 1.0
    ps = [2, 3, 5]
    dimensions = [int(2**a) for a in range(15, 16)]
    print(dimensions)
    epss = np.logspace(-2, 2, 20)
    eps_dense = np.logspace(-2, 2, 100)
    reps = 25
    run_experiment = True

    colors = [f"C{i}" for i in range(len(dimensions))]
    linestyles = ["-", "--", "-.", ":"]
    markers = [".", "x", "1", "2", "+", "3", "4"]
    assert len(linestyles) >= len(ps)
    assert len(markers) >= len(ps)

    for p, ls, mrk in zip(tqdm(ps, desc="p", leave=False), linestyles, markers):

        for n_features, c in zip(tqdm(dimensions, desc="n", leave=False), colors):
            if run_experiment:
                epss_rescaled = epss / (n_features ** (1 / 2 - 1 / p))

                vals = np.empty((reps, len(epss)))
                estim_vals_m = np.empty((reps,))
                estim_vals_q = np.empty((reps,))
                estim_vals_rho = np.empty((reps,))

                for j in tqdm(range(reps), desc="rps", leave=False):
                    xs, ys, xs_gen, ys_gen, teacher_vector = data_generation(
                        measure_gen_no_noise_clasif,
                        n_features=n_features,
                        n_samples=max(int(n_features * alpha), 1),
                        n_generalization=1000,
                        measure_fun_args={},
                    )

                    estimated_theta = jax_minimize(
                        total_loss_logistic,
                        x0=teacher_vector,
                        args=(xs / np.sqrt(n_features), ys, reg_param),
                        method="BFGS",
                    ).x

                    estim_vals_rho[j] = np.sum(teacher_vector**2) / n_features
                    estim_vals_m[j] = (
                        np.sum(teacher_vector * estimated_theta) / n_features
                    )
                    estim_vals_q[j] = np.sum(estimated_theta**2) / n_features

                    # yhat = np.repeat(
                    #     np.sign(xs_gen @ estimated_theta).reshape(-1, 1), n_features, axis=1
                    # )
                    yhat = np.sign(xs_gen @ estimated_theta)  # .reshape(-1, 1)

                    for i, eps_i in enumerate(
                        tqdm(epss_rescaled, desc="eps", leave=False)
                    ):
                        # print(f"current eps_i = {eps_i:.3f}")
                        adv_perturbation = projected_GA(
                            yhat, estimated_theta, teacher_vector, 0.5, 200, eps_i, p
                        )

                        # check that it has the maximum allowed norm
                        assert np.allclose(
                            np.linalg.norm(adv_perturbation, ord=p, axis=1) - eps_i,
                            0.0,
                            atol=1e-5,
                            rtol=1e-5,
                        )

                        flipped = percentage_flipped_labels(
                            yhat,
                            xs_gen,
                            estimated_theta,
                            teacher_vector,
                            xs_gen + adv_perturbation,
                        )

                        vals[j, i] = flipped

                mean_m, std_m = np.mean(estim_vals_m), np.std(estim_vals_m)
                mean_q, std_q = np.mean(estim_vals_q), np.std(estim_vals_q)
                mean_rho, std_rho = np.mean(estim_vals_rho), np.std(estim_vals_rho)

                print(
                    f"Estimated m = {mean_m:.3f} ± {std_m:.3f} q = {mean_q:.3f} ± {std_q:.3f} rho = {mean_rho:.3f} ± {std_rho:.3f}"
                )

                # Save the data with pickle
                data = {
                    "epss": epss,
                    "vals": vals,
                    "estim_vals_m": estim_vals_m,
                    "estim_vals_q": estim_vals_q,
                    "estim_vals_rho": estim_vals_rho,
                    "mean_m": mean_m,
                    "std_m": std_m,
                    "mean_q": mean_q,
                    "std_q": std_q,
                    "mean_rho": mean_rho,
                    "std_rho": std_rho,
                }

                with open(
                    f"./data/n_features_{n_features:d}_reps_{reps:d}_p_{p:d}.pkl", "wb"
                ) as f:
                    pickle.dump(data, f)
            else:
                with open(
                    f"./data/n_features_{n_features:d}_reps_{reps:d}_p_{p:d}.pkl", "rb"
                ) as f:
                    data = pickle.load(f)
                    epss = data["epss"]
                    vals = data["vals"]
                    estim_vals_m = data["estim_vals_m"]
                    estim_vals_q = data["estim_vals_q"]
                    estim_vals_rho = data["estim_vals_rho"]
                    mean_m = data["mean_m"]
                    std_m = data["std_m"]
                    mean_q = data["mean_q"]
                    std_q = data["std_q"]
                    mean_rho = data["mean_rho"]

            plt.errorbar(
                epss,
                np.mean(vals, axis=0),
                yerr=np.std(vals, axis=0),
                # label=f"n_features = {n_features:d}",
                marker=mrk,
                linestyle="None",
                color=c,
            )

        # epss_rescaled = epss / (n_features ** (1 / 2 - 1 / p))

        # print("max eps_rescaled", np.max(epss_rescaled), "min eps_rescaled", np.min(epss_rescaled))

        out = np.empty_like(eps_dense)

        for i, eps_i in enumerate(eps_dense):
            out[i] = percentage_flipped_direct_space(
                mean_m, mean_q, mean_rho, eps_i, p / (p - 1)
            )

        plt.plot(eps_dense, out, linestyle=ls, color="black", linewidth=0.5)

    plt.title(f"Direct space $\\alpha$ = {alpha:.1f} $\\lambda$ = {reg_param:.1e}")
    plt.xscale("log")
    plt.xlabel(r"$\epsilon (\sqrt[p]{d} / \sqrt{d})$")
    plt.ylabel("Percentage of flipped labels")
    plt.grid()

    handles = []
    labels = []
    for p, ls, mrk in zip(ps, linestyles, markers):
        handle = plt.Line2D(
            [], [], linestyle=ls, linewidth=0.5, marker=mrk, color="black"
        )
        handles.append(handle)
        labels.append(f"p = {p:d}" if p != np.inf else "p = $\infty$")

    for dim, c in zip(dimensions, colors):
        handle_dim = plt.Line2D([], [], linestyle="None", marker="o", color=c)
        handles.append(handle_dim)
        labels.append(f"d = {dim:d}")

    plt.legend(handles, labels)

    plt.show()
