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
import sys
from mpi4py import MPI
import time
import datetime

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
def then_func_p(ops):
    x, eps_t, norm_x_projected = ops
    return eps_t * x / norm_x_projected


@jax.jit
def else_func_p(ops):
    return ops[0]


@jax.jit
def project_and_normalize_p(x, wstar, p, eps_t):
    x -= jnp.dot(x, wstar) * wstar / jnp.dot(wstar, wstar)
    # norm_x_projected = jnp.linalg.norm(x, ord=p)
    norm_x_projected = jnp.sum(jnp.abs(x) ** p) ** (1 / p)

    return jax.lax.cond(
        norm_x_projected > eps_t, then_func_p, else_func_p, (x, eps_t, norm_x_projected)
    )


@jax.jit
def then_func_inf(ops):
    x, eps_t, norm_x_projected = ops
    return eps_t * x / norm_x_projected


@jax.jit
def else_func_inf(ops):
    return ops[0]


@jax.jit
def project_and_normalize_inf(x, wstar, eps_t):
    x -= jnp.dot(x, wstar) * wstar / jnp.dot(wstar, wstar)
    # norm_x_projected = jnp.linalg.norm(x, ord=p)
    norm_x_projected = jnp.max(jnp.abs(x))

    return jax.lax.cond(
        norm_x_projected > eps_t, then_func_inf, else_func_inf, (x, eps_t, norm_x_projected)
    )


vecorized_project_and_normalize_p = jax.jit(
    vmap(project_and_normalize_p, in_axes=(0, None, None, None))
)

@jax.jit
def projected_GA_step_jit_p(vs, ys, w, wstar, step_size, eps, p):
    g = grad_linear_loss_all(vs, ys, w)
    return vecorized_project_and_normalize_p(vs + step_size * g, wstar, p, eps)


def projected_GA_p(ys, w, wstar, step_size, n_steps, eps, p):
    adv_perturbation = jnp.zeros((len(ys), len(w)))

    for _ in range(n_steps):
        adv_perturbation = projected_GA_step_jit_p(
            adv_perturbation, ys, w, wstar, step_size, eps, p
        )
    return adv_perturbation


vecorized_project_and_normalize_inf = jax.jit(
    vmap(project_and_normalize_inf, in_axes=(0, None, None))
)

@jax.jit
def projected_GA_step_jit_inf(vs, ys, w, wstar, step_size, eps):
    g = grad_linear_loss_all(vs, ys, w)
    return vecorized_project_and_normalize_inf(vs + step_size * g, wstar, eps)


def projected_GA_inf(ys, w, wstar, step_size, n_steps, eps):
    adv_perturbation = jnp.zeros((len(ys), len(w)))

    for _ in range(n_steps):
        adv_perturbation = projected_GA_step_jit_inf(
            adv_perturbation, ys, w, wstar, step_size, eps
        )
    return adv_perturbation


# if __name__ == "__main__":

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

alpha = 1.5
reg_param = 1.0
ps = ["inf", 2, 3, 5]
assert len(ps) >= size, "Number of processes should not exceed number of p values."

print(f"rank = {rank}, size = {size} and ps = {ps[rank]} started at {datetime.datetime.now()}")

dimensions = [int(2**a) for a in range(15, 16)]
print(dimensions)
epss = np.logspace(-2, 2, 15)
eps_dense = np.logspace(-2, 2, 100)
reps = 50

colors = [f"C{i}" for i in range(len(dimensions))]
linestyles = ["-", "--", "-.", ":"]
markers = [".", "x", "1", "2", "+", "3", "4"]
assert len(linestyles) >= len(ps)
assert len(markers) >= len(ps)

if ps[rank] == "inf":
    for n_features, c in zip(dimensions, colors):
        print(f"process {rank} starts with n_features = {n_features}")
        epss_rescaled = epss * (n_features ** (1 / 2 - 1))

        vals = np.empty((reps, len(epss)))
        estim_vals_m = np.empty((reps,))
        estim_vals_q = np.empty((reps,))
        estim_vals_rho = np.empty((reps,))

        for j in range(reps):
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

            yhat = np.sign(xs_gen @ estimated_theta)  # .reshape(-1, 1)

            for i, eps_i in enumerate(
                epss_rescaled
            ):
                # print(f"current eps_i = {eps_i:.3f}")
                adv_perturbation = projected_GA_inf(
                    yhat, estimated_theta, teacher_vector, 0.5, 500, eps_i
                )

                # check that it has the maximum allowed norm
                print(
                    np.allclose(
                        np.linalg.norm(adv_perturbation, ord=np.inf, axis=1) - eps_i,
                        0.0,
                        atol=1e-5,
                        rtol=1e-5,
                    )
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
            f"./data/n_features_{n_features:d}_reps_{reps:d}_p_inf_alpha_{alpha:.1f}.pkl", "wb"
        ) as f:
            pickle.dump(data, f)

        print(f"process {rank} finished with n_features = {n_features} time = {datetime.datetime.now()}")
else:
    p = ps[rank]  # each process gets a different p value

    for n_features, c in zip(dimensions, colors):
        print(f"process {rank} starts with n_features = {n_features}")
        # epss_rescaled = epss / (n_features ** (1 / 2 - 1 / p))
        epss_rescaled = epss * (n_features ** (- 1 / 2 + 1 / p))

        vals = np.empty((reps, len(epss)))
        estim_vals_m = np.empty((reps,))
        estim_vals_q = np.empty((reps,))
        estim_vals_rho = np.empty((reps,))

        for j in range(reps):
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

            yhat = np.sign(xs_gen @ estimated_theta)  # .reshape(-1, 1)

            for i, eps_i in enumerate(
                epss_rescaled
            ):
                # print(f"current eps_i = {eps_i:.3f}")
                adv_perturbation = projected_GA_p(
                    yhat, estimated_theta, teacher_vector, 0.5, 400, eps_i, p
                )

                # check that it has the maximum allowed norm
                print(
                    np.allclose(
                        np.linalg.norm(adv_perturbation, ord=p, axis=1) - eps_i,
                        0.0,
                        atol=1e-5,
                        rtol=1e-5,
                    )
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
            f"./data/n_features_{n_features:d}_reps_{reps:d}_p_{p:d}_alpha_{alpha:.1f}.pkl", "wb"
        ) as f:
            pickle.dump(data, f)

        print(f"process {rank} finished with n_features = {n_features} time = {datetime.datetime.now()}")