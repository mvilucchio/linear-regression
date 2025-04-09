from numpy import empty, where, divide, ndarray, float32, zeros, eye
from numpy.random import default_rng
from math import sqrt


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Classification                                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def measure_gen_no_noise_clasif(rng, teacher_vector: ndarray, xs: ndarray):
    _, n_features = xs.shape
    w_xs = divide(xs @ teacher_vector, sqrt(n_features), dtype=float32)
    ys = where(w_xs > 0.0, 1.0, -1.0)
    return ys


def measure_gen_probit_clasif(rng, teacher_vector, xs, delta):
    n_samples, n_features = xs.shape
    w_xs = divide(xs @ teacher_vector, sqrt(n_features), dtype=float32)
    noise = sqrt(delta) * rng.standard_normal(size=(n_samples,), dtype=float32)
    ys = where(w_xs + noise > 0.0, 1.0, -1.0)
    return ys


def measure_gen_single_noise_clasif(rng, teacher_vector, xs, delta: float):
    n_samples, n_features = xs.shape
    w_xs = divide(xs @ teacher_vector, sqrt(n_features), dtype=float32)
    error_sample = sqrt(delta) * rng.standard_normal(size=(n_samples,), dtype=float32)
    ys = where(w_xs > 0.0, 1.0, -1.0) + error_sample
    return ys


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Regression                                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def measure_gen_single(rng, teacher_vector, xs, delta: float):
    n_samples, n_features = xs.shape
    w_xs = divide(xs @ teacher_vector, sqrt(n_features), dtype=float32)
    error_sample = sqrt(delta) * rng.standard_normal(size=(n_samples,), dtype=float32)
    ys = w_xs + error_sample
    return ys


def measure_gen_double(
    rng,
    teacher_vector,
    xs,
    delta_in: float,
    delta_out: float,
    percentage: float,
):
    n_samples, n_features = xs.shape
    w_xs = divide(xs @ teacher_vector, sqrt(n_features), dtype=float32)
    c = rng.choice([0, 1], p=[1 - percentage, percentage], size=(n_samples,))
    error_sample = empty((n_samples, 2))
    error_sample[:, 0] = sqrt(delta_in) * rng.standard_normal(size=(n_samples,), dtype=float32)
    error_sample[:, 1] = sqrt(delta_out) * rng.standard_normal(size=(n_samples,), dtype=float32)
    total_error = where(c, error_sample[:, 1], error_sample[:, 0])
    ys = w_xs + total_error
    return ys


def measure_gen_decorrelated(
    rng,
    teacher_vector,
    xs,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
):
    n_samples, n_features = xs.shape
    w_xs = divide(xs @ teacher_vector, sqrt(n_features), dtype=float32)
    c = rng.choice([0, 1], p=[1 - percentage, percentage], size=(n_samples,))
    error_sample = empty((n_samples, 2))
    error_sample[:, 0] = sqrt(delta_in) * rng.standard_normal(size=(n_samples,), dtype=float32)
    error_sample[:, 1] = sqrt(delta_out) * rng.standard_normal(size=(n_samples,), dtype=float32)
    total_error = where(c, error_sample[:, 1], error_sample[:, 0])
    factor_in_front = where(c, beta, 1.0)
    ys = factor_in_front * w_xs + total_error
    return ys


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# General                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def data_generation(
    measure_fun: callable,
    n_features: int,
    n_samples: int,
    n_generalization: int,
    measure_fun_args,
    hidden_model: bool = False,
    overparam_ratio: float = 1.0,
    hidden_fun: callable = None,
    theta_0_teacher: ndarray = None,
    Σx: ndarray = None,
    Σθ: ndarray = None,
):
    rng = default_rng()

    if hidden_model and hidden_fun is None:
        hidden_fun = lambda x: x

    if theta_0_teacher is None:
        theta_0_teacher = rng.standard_normal(size=(n_features,), dtype=float32)

    if Σx is None:
        Σx = eye(n_features, dtype=float32)

    if Σθ is None:
        Σθ = eye(n_features, dtype=float32)

    if hidden_model:
        projector = rng.standard_normal(
            size=(int(overparam_ratio * n_features), n_features),
            dtype=float32,
        )

    zero_vec = zeros(n_features, dtype=float32)

    xs = rng.multivariate_normal(zero_vec, Σx, size=(n_samples,)).astype(float32)
    xs_gen = rng.multivariate_normal(zero_vec, Σx, size=(n_generalization,)).astype(float32)

    ys = measure_fun(rng, theta_0_teacher, xs, *measure_fun_args)
    ys_gen = measure_fun(rng, theta_0_teacher, xs_gen, *measure_fun_args)

    if hidden_model:
        n = sqrt(overparam_ratio * n_features)
        vs = hidden_fun(xs @ projector.T / sqrt(n_features)) / n
        vs_gen = hidden_fun(xs_gen @ projector.T / sqrt(n_features)) / n

        return vs, xs, ys, vs_gen, xs_gen, ys_gen, theta_0_teacher, projector
    else:
        return xs, ys, xs_gen, ys_gen, theta_0_teacher


def data_generation_hastie(
    measure_fun: callable,
    d: int,
    n: int,
    n_gen: int,
    measure_fun_args,
    gamma: float = 1.0,
    theta_0_teacher: ndarray = None,
    Σx: ndarray = None,
):
    rng = default_rng()

    if Σx is None:
        Σx = eye(d, dtype=float32)

    if theta_0_teacher is None:
        theta_0_teacher = rng.standard_normal(size=(d,), dtype=float32)

    p = int(d / gamma)

    projector = zeros((p, d), dtype=float32)
    if p >= d:
        projector[:d, :d] = sqrt(p / d) * eye(d)
    else:
        projector[:p, :p] = eye(p)

    print("projector shape", projector.shape)

    zero_vec = zeros(d, dtype=float32)

    zs = rng.multivariate_normal(zero_vec, Σx, size=(n,)).astype(float32)
    zs_gen = rng.multivariate_normal(zero_vec, Σx, size=(n_gen,)).astype(float32)

    ys = measure_fun(rng, theta_0_teacher, zs, *measure_fun_args)
    ys_gen = measure_fun(rng, theta_0_teacher, zs_gen, *measure_fun_args)

    xs = zs @ projector.T + rng.multivariate_normal(
        zeros(p, dtype=float32), eye(p, dtype=float32), size=(n,)
    ).astype(float32)
    xs_gen = zs_gen @ projector.T + rng.multivariate_normal(
        zeros(p, dtype=float32), eye(p, dtype=float32), size=(n_gen,)
    ).astype(float32)

    return xs, ys, zs, xs_gen, ys_gen, zs_gen, theta_0_teacher, projector


def data_generation_correalted(
    measure_fun,
    n_features: int,
    n_samples: int,
    n_generalization: int,
    measure_fun_args,
    Sigmax_cov: ndarray,
    Sigmatheta_cov: ndarray,
):
    rng = default_rng()
    mean = zeros(n_features)
    theta_0_vector = rng.multivariate_normal(mean, Sigmatheta_cov, size=(1,)).astype(float32)[0]

    xs = rng.multivariate_normal(mean, Sigmax_cov, size=(n_samples,)).astype(float32)
    xs_gen = rng.multivariate_normal(mean, Sigmax_cov, size=(n_generalization,)).astype(float32)

    ys = measure_fun(rng, theta_0_vector, xs, *measure_fun_args)
    ys_gen = measure_fun(rng, theta_0_vector, xs_gen, *measure_fun_args)

    return xs, ys, xs_gen, ys_gen, theta_0_vector
