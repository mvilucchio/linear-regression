from typing import Tuple, List
import numpy as np
from numpy import sqrt, sum, dot, empty_like
from numpy.linalg import norm


def adversarial_direction_generation(
    xs,
    function_grad: callable,
    function_grad_args: List[Tuple],
    teacher_vector,
    orthogonal_projetion: bool = True,
    ord: float = 2,
    hidden_model: bool = False,
    ratio_hidden: float = 1.0,
    hidden_fun: callable = None,
    proj_mat_hidden=None,
):
    if hidden_model and hidden_fun is None:
        hidden_fun = lambda x: x

    if hidden_model and proj_mat_hidden is None:
        raise ValueError("Hidden model requires a projection matrix")

    n, d = xs.shape
    adv_perturbation = empty_like(xs)
    for i, (x, f_args) in enumerate(zip(xs, function_grad_args)):
        grad_dir = function_grad(x, *f_args)

        if orthogonal_projetion:
            grad_dir -= (
                dot(grad_dir, teacher_vector) / sum(teacher_vector**2) * teacher_vector
            )

        adv_perturbation[i] = grad_dir / norm(grad_dir, ord=ord)

    if hidden_model:
        vs_adv = hidden_fun(proj_mat_hidden @ (xs + adv_perturbation) / sqrt(d)) / sqrt(
            ratio_hidden * d
        )
        return vs_adv, xs + adv_perturbation, adv_perturbation
    else:
        return xs + adv_perturbation, adv_perturbation
