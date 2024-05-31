from numpy import around, empty, sum, mean, std, square, divide, sqrt, dot, sign
from math import acos, pi


def estimation_error_data(ys, xs, w, wstar):
    _, d = xs.shape
    return sum((wstar - w) ** 2) / d


def train_error_data(ys, xs, w, wstar, loss_function, loss_function_args):
    n, d = xs.shape
    xs_norm = xs / sqrt(d)
    tmp = loss_function(ys, xs_norm @ w, *loss_function_args)
    return sum(tmp) / n


def angle_teacher_student_data(ys, xs, w, wstar):
    tmp = dot(w, wstar) / sqrt(dot(w, w) * dot(wstar, wstar))
    return acos(tmp) / pi


def generalisation_error_classification(ys, xs, w, wstar):
    return mean(ys != sign(xs @ w))


def adversarial_error_data(ys, xs, w, wstar, eps, pstar):
    _, d = xs.shape
    tmp = sign(xs @ w / sqrt(d) - eps * ys * sum(abs(w) ** pstar) ** (1 / pstar) / d**pstar)
    return mean(ys != tmp)


# Adversarial Errors

# def adversarial_error_data(ys, xs, w, wstar):
#     n, d = xs.shape
#     xs_norm = xs / sqrt(d)
#     tmp = ys - xs_norm @ w
#     return sum(square(tmp)) / n


# def percentage_flipped_labels(
#     ys, xs, w, wstar, xs_perturbation
# ):
#     return mean(
#         sign(xs @ w) != sign((xs + xs_perturbation) @ w)
#     )


def percentage_flipped_labels(
    ys,
    xs,
    w,
    wstar,
    xs_pertubed,
    hidden_model=False,
    projection_matrix=None,
):
    if hidden_model:
        if projection_matrix is None:
            raise ValueError("Hidden model requires projection matrix")
        return mean(sign(xs @ projection_matrix @ w) != sign(xs_pertubed @ projection_matrix @ w))

    return mean(sign(xs @ w) != sign(xs_pertubed @ w))


# Overlaps Estimation


def m_real_overlaps(ys, xs, w, wstar):
    d = xs.shape[1]
    m = dot(w, wstar) / d
    return m


def q_real_overlaps(ys, xs, w, wstar):
    d = xs.shape[1]
    q = sum(square(w)) / d
    return q
