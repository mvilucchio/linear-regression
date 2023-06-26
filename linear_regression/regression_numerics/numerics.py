from numpy import around, empty, sum, mean, std, square, divide, sqrt, dot
from .data_generation import data_generation


def gen_error_data(ys, xs, estimated_theta, ground_truth_theta):
    _, d = xs.shape
    estimation_error = sum((ground_truth_theta - estimated_theta) ** 2) / d
    return estimation_error


def train_error_data(
    ys, xs, estimated_theta, ground_truth_theta, loss_function, loss_function_args
):
    n, d = xs.shape
    xs_norm = xs / sqrt(d)
    train_error = sum(loss_function(ys, xs_norm @ estimated_theta, *loss_function_args)) / n
    return train_error


def m_real_overlaps(ys, xs, estimated_theta, ground_truth_theta):
    d = xs.shape[1]
    m = dot(estimated_theta, ground_truth_theta) / d
    return m


def q_real_overlaps(ys, xs, estimated_theta, ground_truth_theta):
    d = xs.shape[1]
    q = sum(square(estimated_theta)) / d
    return q


def erm_weight_finding(
    alpha: float,
    measure_fun: callable,
    find_coefficients_fun: callable,
    funs,
    funs_args,
    n_features: int,
    repetitions: int,
    measure_fun_args,
    find_coefficients_fun_args,
):
    if alpha <= 0:
        raise ValueError("alpha should be positive, in this case is {:f}".format(alpha))

    if len(funs) != len(funs_args):
        raise ValueError(
            "The length of funs and funs_args should be the same, in this case is {:d} and {:d}".format(
                len(funs), len(funs_args)
            )
        )

    if n_features <= 0:
        raise ValueError("n_features should be positive, in this case is {:d}".format(n_features))

    if repetitions <= 0:
        raise ValueError("repetitions should be positive, in this case is {:d}".format(repetitions))

    out_list = [empty(repetitions) for _ in range(len(funs))]
    out_list_mean = empty(len(funs))
    out_list_std = empty(len(funs))

    for idx in range(repetitions):
        xs, ys, _, _, ground_truth_theta = data_generation(
            measure_fun,
            n_features=n_features,
            n_samples=max(int(around(n_features * alpha)), 1),
            n_generalization=1,
            measure_fun_args=measure_fun_args,
        )

        estimated_theta = find_coefficients_fun(ys, xs, *find_coefficients_fun_args)

        for jdx, (f, f_args) in enumerate(zip(funs, funs_args)):
            out_list[jdx][idx] = f(ys, xs, estimated_theta, ground_truth_theta, *f_args)

        del xs
        del ys
        del ground_truth_theta

    for idx, out_vals in enumerate(out_list):
        out_list_mean[idx], out_list_std[idx] = mean(out_vals), std(out_vals)

    print(alpha, " Done.")

    del out_vals

    return out_list_mean, out_list_std


# def run_erm_weight_finding(
#     alpha: float,
#     measure_fun,
#     find_coefficients_fun,
#     n_features: int,
#     repetitions: int,
#     measure_fun_args,
#     find_coefficients_fun_args,
# ):
#     all_gen_errors = empty((repetitions,))

#     for idx in range(repetitions):
#         xs, ys, _, _, ground_truth_theta = data_generation(
#             measure_fun,
#             n_features=n_features,
#             n_samples=max(int(around(n_features * alpha)), 1),
#             n_generalization=1,
#             measure_fun_args=measure_fun_args,
#         )

#         print(xs.shape, ys.shape)

#         estimated_theta = find_coefficients_fun(ys, xs, *find_coefficients_fun_args)

#         all_gen_errors[idx] = divide(sum(square(ground_truth_theta - estimated_theta)), n_features)

#         del xs
#         del ys
#         del ground_truth_theta

#     error_mean, error_std = mean(all_gen_errors), std(all_gen_errors)
#     print(alpha, " Done.")

#     del all_gen_errors

#     return error_mean, error_std
