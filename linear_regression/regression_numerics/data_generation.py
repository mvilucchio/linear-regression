from numpy.random import normal, choice
from numpy import empty, sqrt, where, divide


def measure_gen_single(generalization: bool, teacher_vector, xs, delta: float):
    n_samples, n_features = xs.shape
    w_xs = divide(xs @ teacher_vector, sqrt(n_features))
    if generalization:
        ys = w_xs
    else:
        error_sample = sqrt(delta) * normal(loc=0.0, scale=1.0, size=(n_samples,))
        ys = w_xs + error_sample
    return ys


def measure_gen_double(
    generalization: bool, teacher_vector, xs, delta_in: float, delta_out: float, percentage: float
):
    n_samples, n_features = xs.shape
    w_xs = divide(xs @ teacher_vector, sqrt(n_features))
    if generalization:
        ys = w_xs
    else:
        c = choice([0, 1], p=[1 - percentage, percentage], size=(n_samples,))
        error_sample = empty((n_samples, 2))
        error_sample[:, 0] = sqrt(delta_in) * normal(loc=0.0, scale=1.0, size=(n_samples,))
        error_sample[:, 1] = sqrt(delta_out) * normal(loc=0.0, scale=1.0, size=(n_samples,))
        total_error = where(c, error_sample[:, 1], error_sample[:, 0])
        ys = w_xs + total_error
    return ys


def measure_gen_decorrelated(
    generalization: bool,
    teacher_vector,
    xs,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
):
    n_samples, n_features = xs.shape
    w_xs = divide(xs @ teacher_vector, sqrt(n_features))
    if generalization:
        ys = w_xs
    else:
        c = choice([0, 1], p=[1 - percentage, percentage], size=(n_samples,))
        error_sample = empty((n_samples, 2))
        error_sample[:, 0] = sqrt(delta_in) * normal(loc=0.0, scale=1.0, size=(n_samples,))
        error_sample[:, 1] = sqrt(delta_out) * normal(loc=0.0, scale=1.0, size=(n_samples,))
        total_error = where(c, error_sample[:, 1], error_sample[:, 0])
        factor_in_front = where(c, beta, 1.0)
        ys = factor_in_front * w_xs + total_error
    return ys


def data_generation(
    measure_fun, n_features: int, n_samples: int, n_generalization: int, measure_fun_args
):
    theta_0_teacher = normal(loc=0.0, scale=1.0, size=(n_features,))

    xs = normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    xs_gen = normal(loc=0.0, scale=1.0, size=(n_generalization, n_features))

    ys = measure_fun(False, theta_0_teacher, xs, *measure_fun_args)
    ys_gen = measure_fun(False, theta_0_teacher, xs_gen, *measure_fun_args)
    # ys_gen = measure_fun(True, theta_0_teacher, xs_gen, *measure_fun_args)

    return xs, ys, xs_gen, ys_gen, theta_0_teacher
