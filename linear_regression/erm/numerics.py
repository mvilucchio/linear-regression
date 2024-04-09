from numpy import around, empty, mean, std
from ..data.generation import data_generation

# from numpy import around, empty, sum, mean, std, square, divide, sqrt, dot
# from math import acos, pi


# def estimation_error_data(ys, xs, estimated_theta, ground_truth_theta):
#     _, d = xs.shape
#     estimation_error = sum((ground_truth_theta - estimated_theta) ** 2) / d
#     return estimation_error


# def train_error_data(
#     ys, xs, estimated_theta, ground_truth_theta, loss_function, loss_function_args
# ):
#     n, d = xs.shape
#     xs_norm = xs / sqrt(d)
#     tmp = loss_function(ys, xs_norm @ estimated_theta, *loss_function_args)
#     return sum(tmp) / n


# def angle_teacher_student_data(ys, xs, estimated_theta, ground_truth_theta):
#     tmp = dot(estimated_theta, ground_truth_theta) / sqrt(
#         dot(estimated_theta, estimated_theta)
#         * dot(ground_truth_theta, ground_truth_theta)
#     )
#     return acos(tmp) / pi


# def m_real_overlaps(ys, xs, estimated_theta, ground_truth_theta):
#     d = xs.shape[1]
#     m = dot(estimated_theta, ground_truth_theta) / d
#     return m


# def q_real_overlaps(ys, xs, estimated_theta, ground_truth_theta):
#     d = xs.shape[1]
#     q = sum(square(estimated_theta)) / d
#     return q


def erm_weight_finding_2(
    sample_complexity: float,
    measure_fun: callable,
    find_coefficients_fun: callable,
    funs_train_data,
    funs_args_train_data,
    n_features: int,
    repetitions: int,
    measure_fun_args,
    find_coefficients_fun_args,
    verbose: bool = False,
    hidden_ratio: float = 1.0,
    hidden_model: bool = False,
    hidden_fun: callable = None,
):
    if hidden_model and hidden_fun is None:
        hidden_fun = lambda x: x
        if verbose:
            print("Hidden function is None, using identity function.")

    if sample_complexity <= 0:
        raise ValueError("sample_complexity should be positive, in this case is {:f}".format(sample_complexity))

    if len(funs_train_data) != len(funs_args_train_data):
        raise ValueError(
            "The length of funs and funs_args should be the same, in this case is {:d} and {:d}".format(
                len(funs_train_data), len(funs_args_train_data)
            )
        )

    if n_features <= 0:
        raise ValueError(
            "n_features should be positive, in this case is {:d}".format(n_features)
        )

    if repetitions <= 0:
        raise ValueError(
            "repetitions should be positive, in this case is {:d}".format(repetitions)
        )

    out_list = [empty(repetitions) for _ in range(len(funs_train_data))]
    out_list_mean = empty(len(funs_train_data))
    out_list_std = empty(len(funs_train_data))

    if verbose:
        print("sample_complexity = {:f} rep : ".format(sample_complexity), end="")

    for idx in range(repetitions):
        if verbose:
            print("{:d}".format(idx), end=",")

        out = data_generation(
            measure_fun,
            n_features=n_features,
            n_samples=max(int(around(n_features * sample_complexity)), 1),
            n_generalization=1,
            measure_fun_args=measure_fun_args,
            hidden_model=hidden_model,
            overparam_ratio=hidden_ratio,
            hidden_fun=hidden_fun,
        )

        if hidden_model:
            xs, _, ys, _, _, _, ground_truth_theta, _ = out
        else:
            xs, ys, _, _, ground_truth_theta = out

        estimated_theta = find_coefficients_fun(ys, xs, *find_coefficients_fun_args)

        for jdx, (f, f_args) in enumerate(zip(funs_train_data, funs_args_train_data)):
            out_list[jdx][idx] = f(ys, xs, estimated_theta, ground_truth_theta, *f_args)

        del xs
        del ys
        del ground_truth_theta

    for idx, out_vals in enumerate(out_list):
        out_list_mean[idx], out_list_std[idx] = mean(out_vals), std(out_vals)

    if verbose:
        print(" Done.")

    del out_vals

    return out_list_mean, out_list_std


def erm_weight_finding(
    sample_complexity: float,
    measure_fun: callable,
    find_coefficients_fun: callable,
    funs_train_data,
    funs_args_train_data,
    n_features: int,
    repetitions: int,
    measure_fun_args,
    find_coefficients_fun_args,
    verbose: bool = False,
):
    if sample_complexity <= 0:
        raise ValueError("sample_complexity should be positive, in this case is {:f}".format(sample_complexity))

    if len(funs_train_data) != len(funs_args_train_data):
        raise ValueError(
            "The length of funs and funs_args should be the same, in this case is {:d} and {:d}".format(
                len(funs_train_data), len(funs_args_train_data)
            )
        )

    if n_features <= 0:
        raise ValueError(
            "n_features should be positive, in this case is {:d}".format(n_features)
        )

    if repetitions <= 0:
        raise ValueError(
            "repetitions should be positive, in this case is {:d}".format(repetitions)
        )

    out_list = [empty(repetitions) for _ in range(len(funs_train_data))]
    out_list_mean = empty(len(funs_train_data))
    out_list_std = empty(len(funs_train_data))

    if verbose:
        print("sample_complexity = {:f} rep : ".format(sample_complexity), end="")

    for idx in range(repetitions):
        if verbose:
            print("{:d}".format(idx), end=",")
        xs, ys, _, _, ground_truth_theta = data_generation(
            measure_fun,
            n_features=n_features,
            n_samples=max(int(around(n_features * sample_complexity)), 1),
            n_generalization=1,
            measure_fun_args=measure_fun_args,
        )

        estimated_theta = find_coefficients_fun(ys, xs, *find_coefficients_fun_args)

        for jdx, (f, f_args) in enumerate(zip(funs_train_data, funs_args_train_data)):
            out_list[jdx][idx] = f(ys, xs, estimated_theta, ground_truth_theta, *f_args)

        del xs
        del ys
        del ground_truth_theta

    for idx, out_vals in enumerate(out_list):
        out_list_mean[idx], out_list_std[idx] = mean(out_vals), std(out_vals)

    if verbose:
        print(" Done.")

    del out_vals

    return out_list_mean, out_list_std


# def run_erm_weight_finding(
#     sample_complexity: float,
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
#             n_samples=max(int(around(n_features * sample_complexity)), 1),
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
#     print(sample_complexity, " Done.")

#     del all_gen_errors

#     return error_mean, error_std
