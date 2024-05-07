from numpy import around, empty, sum, mean, std, square, divide, sqrt, dot, sign
from math import acos, pi


def estimation_error_data(ys, xs, estimated_theta, ground_truth_theta):
    _, d = xs.shape
    return sum((ground_truth_theta - estimated_theta) ** 2) / d


def train_error_data(
    ys, xs, estimated_theta, ground_truth_theta, loss_function, loss_function_args
):
    n, d = xs.shape
    xs_norm = xs / sqrt(d)
    tmp = loss_function(ys, xs_norm @ estimated_theta, *loss_function_args)
    return sum(tmp) / n


def angle_teacher_student_data(ys, xs, estimated_theta, ground_truth_theta):
    tmp = dot(estimated_theta, ground_truth_theta) / sqrt(
        dot(estimated_theta, estimated_theta)
        * dot(ground_truth_theta, ground_truth_theta)
    )
    return acos(tmp) / pi


# Adversarial Errors

# def adversarial_error_data(ys, xs, estimated_theta, ground_truth_theta):
#     n, d = xs.shape
#     xs_norm = xs / sqrt(d)
#     tmp = ys - xs_norm @ estimated_theta
#     return sum(square(tmp)) / n


# def percentage_flipped_labels(
#     ys, xs, estimated_theta, ground_truth_theta, xs_perturbation
# ):
#     return mean(
#         sign(xs @ estimated_theta) != sign((xs + xs_perturbation) @ estimated_theta)
#     )


def percentage_flipped_labels(ys, xs, estimated_theta, ground_truth_theta, xs_pertubed, hidden_model=False, projection_matrix=None):
    if hidden_model:
        if projection_matrix is None:
            raise ValueError("Hidden model requires projection matrix")
        return mean(sign(xs @ projection_matrix @ estimated_theta) != sign(xs_pertubed @ projection_matrix @ estimated_theta))
    
    return mean(sign(xs @ estimated_theta) != sign(xs_pertubed @ estimated_theta))


# Overlaps Estimation


def m_real_overlaps(ys, xs, estimated_theta, ground_truth_theta):
    d = xs.shape[1]
    m = dot(estimated_theta, ground_truth_theta) / d
    return m


def q_real_overlaps(ys, xs, estimated_theta, ground_truth_theta):
    d = xs.shape[1]
    q = sum(square(estimated_theta)) / d
    return q
