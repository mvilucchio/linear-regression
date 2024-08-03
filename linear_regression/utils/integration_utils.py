import numpy as np
from scipy.integrate import romb
from math import sqrt
from numba import njit
from sklearn.metrics import max_error
from typing import Callable, Optional, Iterable, List

MULT_INTEGRAL = 10
TOL_INT = 1e-6

BIG_NUMBER = 10

N_TEST_POINTS = 200

N_GAUSS_HERMITE = 95


x_ge, w_ge = np.polynomial.hermite.hermgauss(N_GAUSS_HERMITE)


@njit(error_model="numpy", fastmath=True)
def gauss_hermite_quadrature(fun: Callable[[float], float], mean: float, std: float) -> float:
    # x, w = np.polynomial.hermite.hermgauss(N_GAUSS_HERMITE)
    y = np.sqrt(2.0) * std * x_ge + mean
    jacobian = np.sqrt(2.0) * std
    return np.sum(w_ge * jacobian * fun(y))


def _check_nested_list(nested_list):
    if isinstance(nested_list, list):
        if isinstance(nested_list[0], list):
            if isinstance(nested_list[0][0], (int, float)):
                return True
    return False


def find_integration_borders_square(
    fun: Callable[[float, float], float],
    scale1: float,
    scale2: float,
    mult: float = MULT_INTEGRAL,
    tol: float = TOL_INT,
    n_points: int = N_TEST_POINTS,
    args: Optional[Iterable[float]] = None,
) -> List[List[float]]:
    if args is None:
        args = tuple()

    test_pts = np.empty(n_points)
    max_vals = np.empty(4)

    n_test = 0
    while True:
        test_pts = np.linspace(-mult * scale1, mult * scale1, n_points)
        max_vals[0] = np.max(fun(test_pts, mult * scale2, *args))
        max_vals[2] = np.max(fun(test_pts, -mult * scale2, *args))

        test_pts = np.linspace(-mult * scale2, mult * scale2, n_points)
        max_vals[1] = np.max(fun(mult * scale1, test_pts, *args))
        max_vals[3] = np.max(fun(-mult * scale1, test_pts, *args))

        if np.max(max_vals) < tol:
            max_scale = max(scale1, scale2)
            return [[-mult * max_scale, mult * max_scale], [-mult * max_scale, mult * max_scale]]
        else:
            n_test += 1
            mult += 1

        if n_test > 10:
            raise ValueError(
                "Cannot find the integration borders. The function is probably not bounded."
            )


def divide_integration_borders_multiple_grid(square_borders, N=10):
    if N < 1:
        raise ValueError("N must be greater than 1")
    if N == 1:
        return square_borders[0], square_borders[1]

    max_range = square_borders[0][1]
    step = 2 * max_range / N

    domain_x = []
    domain_y = []
    for idx in range(N):
        for jdx in range(N):
            domain_x.append([-max_range + idx * step, -max_range + (idx + 1) * step])
            domain_y.append([-max_range + jdx * step, -max_range + (jdx + 1) * step])

    return domain_x, domain_y


def domains_line_constraint(square_borders, y_fun, x_fun, args_y, args_x):
    max_range = square_borders[0][1]

    x_test_val = x_fun(max_range, **args_x)

    if x_test_val > max_range:
        domain_x = [
            [-max_range, -1.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, max_range],
            [-max_range, -1.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, max_range],
        ]
        domain_y = [
            [lambda x: y_fun(x, **args_y), lambda x: max_range],
            [lambda x: y_fun(x, **args_y), lambda x: max_range],
            [lambda x: y_fun(x, **args_y), lambda x: max_range],
            [lambda x: y_fun(x, **args_y), lambda x: max_range],
            [lambda x: -max_range, lambda x: y_fun(x, **args_y)],
            [lambda x: -max_range, lambda x: y_fun(x, **args_y)],
            [lambda x: -max_range, lambda x: y_fun(x, **args_y)],
            [lambda x: -max_range, lambda x: y_fun(x, **args_y)],
        ]
    else:
        domain_x = [
            [x_test_val, max_range],
            [-x_test_val, 0.0],
            [0.0, x_test_val],
            [-x_test_val, 0.0],
            [0.0, x_test_val],
            [-max_range, -x_test_val],
        ]
        domain_y = [
            [-max_range, max_range],
            [lambda x: y_fun(x, **args_y), max_range],
            [lambda x: y_fun(x, **args_y), max_range],
            [-max_range, lambda x: y_fun(x, **args_y)],
            [-max_range, lambda x: y_fun(x, **args_y)],
            [-max_range, max_range],
        ]

    return domain_x, domain_y


def domains_double_line_constraint(
    square_borders, y_fun_upper, y_fun_lower, x_fun_upper, args1, args2, args3
):
    max_range = square_borders[0][1]

    x_test_val = x_fun_upper(max_range, **args3)
    x_test_val_2 = x_fun_upper(-max_range, **args3)  # attention

    if x_test_val > max_range:
        domain_x = [[-max_range, max_range]] * 3
        domain_y = [
            [lambda x: y_fun_upper(x, **args1), lambda x: max_range],
            [
                lambda x: y_fun_lower(x, **args2),
                lambda x: y_fun_upper(x, **args1),
            ],
            [lambda x: -max_range, lambda x: y_fun_lower(x, **args2)],
        ]
    elif x_test_val >= 0:
        if x_test_val_2 < -max_range:
            domain_x = [
                [-max_range, x_test_val],
                [-x_test_val, max_range],
                [-max_range, -x_test_val],
                [x_test_val, max_range],
                [-x_test_val, x_test_val],
            ]
            domain_y = [
                [lambda x: y_fun_upper(x, **args1), lambda x: max_range],
                [lambda x: -max_range, lambda x: y_fun_lower(x, **args2)],
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: y_fun_lower(x, **args2), lambda x: y_fun_upper(x, **args1)],
            ]
        else:
            x_test_val_2 = x_fun_upper(-max_range, **args3)
            domain_x = [
                [x_test_val_2, x_test_val],
                [-x_test_val, -x_test_val_2],
                [x_test_val_2, -x_test_val],
                [x_test_val, -x_test_val_2],
                [-x_test_val, x_test_val],
                [-max_range, x_test_val_2],
                [-x_test_val_2, max_range],
            ]
            domain_y = [
                [lambda x: y_fun_upper(x, **args1), lambda x: max_range],
                [lambda x: -max_range, lambda x: y_fun_lower(x, **args2)],
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: y_fun_lower(x, **args2), lambda x: y_fun_upper(x, **args1)],
                [lambda x: -max_range, lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
            ]
    elif x_test_val > -max_range:
        if x_test_val_2 < -max_range:
            domain_x = [
                [-max_range, x_test_val],
                [-x_test_val, max_range],
                [-max_range, x_test_val],
                [-x_test_val, max_range],
                [x_test_val, -x_test_val],
            ]
            domain_y = [
                [lambda x: y_fun_upper(x, **args1), lambda x: max_range],
                [lambda x: -max_range, lambda x: y_fun_lower(x, **args2)],
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
            ]
        else:
            x_test_val_2 = x_fun_upper(-max_range, **args3)
            domain_x = [
                [x_test_val_2, x_test_val],
                [-x_test_val, -x_test_val_2],
                [x_test_val_2, x_test_val],
                [-x_test_val, -x_test_val_2],
                [x_test_val, -x_test_val],
                [-max_range, x_test_val_2],
                [-x_test_val_2, max_range],
            ]
            domain_y = [
                [lambda x: y_fun_upper(x, **args1), lambda x: max_range],
                [lambda x: -max_range, lambda x: y_fun_lower(x, **args2)],
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
            ]
    else:
        domain_x = [[-max_range, max_range]]
        domain_y = [[lambda x: -max_range, lambda x: max_range]]

    return domain_x, domain_y


def domains_double_line_constraint_only_inside(
    square_borders, y_fun_upper, y_fun_lower, x_fun_upper, args1, args2, args3
):
    max_range = square_borders[0][1]

    x_test_val = x_fun_upper(max_range, **args3)
    x_test_val_2 = x_fun_upper(-max_range, **args3)  # attention

    if x_test_val > max_range:
        # # print("Case 1")
        domain_x = [[-max_range, max_range]]
        domain_y = [
            [
                lambda x: y_fun_lower(x, **args2),
                lambda x: y_fun_upper(x, **args1),
            ],
        ]
    elif x_test_val >= 0:
        if x_test_val_2 < -max_range:
            # # print("Case 2.A")
            domain_x = [
                [-max_range, -x_test_val],
                [x_test_val, max_range],
                [-x_test_val, x_test_val],
            ]
            domain_y = [
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: y_fun_lower(x, **args2), lambda x: y_fun_upper(x, **args1)],
            ]
        else:
            # # print("Case 2.B")
            x_test_val_2 = x_fun_upper(-max_range, **args3)
            domain_x = [
                [x_test_val_2, -x_test_val],
                [x_test_val, -x_test_val_2],
                [-x_test_val, x_test_val],
            ]
            domain_y = [
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: y_fun_lower(x, **args2), lambda x: y_fun_upper(x, **args1)],
            ]
    elif x_test_val > -max_range:
        if x_test_val_2 < -max_range:
            # # print("Case 3.A")
            domain_x = [
                [-max_range, x_test_val],
                [-x_test_val, max_range],
                [x_test_val, -x_test_val],
            ]
            domain_y = [
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
            ]
        else:
            # # print("Case 3.B")
            x_test_val_2 = x_fun_upper(-max_range, **args3)
            domain_x = [
                [x_test_val_2, x_test_val],
                [-x_test_val, -x_test_val_2],
                [x_test_val, -x_test_val],
            ]
            domain_y = [
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
            ]
    else:
        # # print("Case 4")
        domain_x = [[-max_range, max_range]]
        domain_y = [[lambda x: -max_range, lambda x: max_range]]

    return domain_x, domain_y


def domains_sep_hyperboles_inside(square_borders, hyp_min, hyp_max, arg_hyp_min, arg_hyp_max):
    max_range = square_borders[0][1]

    x_test_val_min = hyp_min(max_range, **arg_hyp_min)
    x_test_val_max = hyp_max(max_range, **arg_hyp_max)

    if x_test_val_max < 0.0:
        raise ValueError("The max hyperbole should be in the first and third quadrant")

    if x_test_val_min < 0.0:
        # print("Case 1")
        x_test_val_min_abs = abs(x_test_val_min)
        if x_test_val_max > x_test_val_min_abs:
            # print("Case 1.A")
            domain_x = [
                [-max_range, -x_test_val_max],
                [-x_test_val_max, -x_test_val_min_abs],
                [-x_test_val_min_abs, x_test_val_min_abs],
                [x_test_val_min_abs, x_test_val_max],
                [x_test_val_max, max_range],
            ]
            domain_y = [
                [lambda x: hyp_max(x, **arg_hyp_max), lambda x: hyp_min(x, **arg_hyp_min)],
                [lambda x: -max_range, lambda x: hyp_min(x, **arg_hyp_min)],
                [lambda x: -max_range, lambda x: max_range],
                [lambda x: hyp_min(x, **arg_hyp_min), lambda x: max_range],
                [lambda x: hyp_min(x, **arg_hyp_min), lambda x: hyp_max(x, **arg_hyp_max)],
            ]
        elif x_test_val_min_abs < max_range:
            # print("Case 1.B")
            domain_x = [
                [-max_range, -x_test_val_min_abs],
                [-x_test_val_min_abs, -x_test_val_max],
                [-x_test_val_max, x_test_val_max],
                [x_test_val_max, x_test_val_min_abs],
                [x_test_val_min_abs, max_range],
            ]
            domain_y = [
                [lambda x: hyp_max(x, **arg_hyp_max), lambda x: hyp_min(x, **arg_hyp_min)],
                [lambda x: hyp_max(x, **arg_hyp_max), lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
                [lambda x: -max_range, lambda x: hyp_max(x, **arg_hyp_max)],
                [lambda x: hyp_min(x, **arg_hyp_min), lambda x: hyp_max(x, **arg_hyp_max)],
            ]
        else:
            # print("Case 1.C")
            domain_x = [
                [-max_range, -x_test_val_max],
                [-x_test_val_max, x_test_val_max],
                [x_test_val_max, max_range],
            ]
            domain_y = [
                [lambda x: hyp_max(x, **arg_hyp_max), lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
                [lambda x: -max_range, lambda x: hyp_max(x, **arg_hyp_max)],
            ]
    else:
        # print("Case 2")
        if x_test_val_min > max_range:
            # print("Case 2.A")
            domain_x = []
            domain_y = []
        else:
            # print("Case 2.B")
            if x_test_val_max > max_range:
                # print("Case 2.B.1")
                domain_x = [[x_test_val_min, max_range], [-max_range, -x_test_val_min]]
                domain_y = [
                    [lambda x: hyp_min(x, **arg_hyp_min), lambda x: max_range],
                    [lambda x: -max_range, lambda x: hyp_min(x, **arg_hyp_min)],
                ]
            else:
                # print("Case 2.B.2")
                domain_x = [
                    [-max_range, -x_test_val_max],
                    [-x_test_val_max, -x_test_val_min],
                    [x_test_val_min, x_test_val_max],
                    [x_test_val_max, max_range],
                ]
                domain_y = [
                    [lambda x: hyp_max(x, **arg_hyp_max), lambda x: hyp_min(x, **arg_hyp_min)],
                    [lambda x: -max_range, lambda x: hyp_min(x, **arg_hyp_min)],
                    [lambda x: hyp_min(x, **arg_hyp_min), lambda x: max_range],
                    [lambda x: hyp_min(x, **arg_hyp_min), lambda x: hyp_max(x, **arg_hyp_max)],
                ]

    return domain_x, domain_y


def domains_sep_hyperboles_above(square_borders, hyp, arg_hyp):
    max_range = square_borders[0][1]

    x_test_val = hyp(max_range, **arg_hyp)

    if x_test_val < 0.0:
        x_test_val_abs = abs(x_test_val)
        if x_test_val_abs > max_range:
            domain_x = []
            domain_y = []
        else:
            domain_x = [
                [-max_range, -x_test_val_abs],
                [x_test_val_abs, max_range],
            ]
            domain_y = [
                [lambda x: hyp(x, **arg_hyp), lambda x: max_range],
                [lambda x: -max_range, lambda x: hyp(x, **arg_hyp)],
            ]
    else:
        if x_test_val > max_range:
            domain_x = [[-max_range, max_range]]
            domain_y = [[-max_range, max_range]]
        else:
            domain_x = [
                [-max_range, -x_test_val],
                [-x_test_val, x_test_val],
                [x_test_val, max_range],
            ]
            domain_y = [
                [lambda x: hyp(x, **arg_hyp), lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
                [lambda x: -max_range, lambda x: hyp(x, **arg_hyp)],
            ]

    return domain_x, domain_y


def line_borders_hinge_inside(m, q, V):
    test_value_1 = (1 - V) / sqrt(q)

    if test_value_1 > -BIG_NUMBER:
        return [(1, [test_value_1, 1 / sqrt(q)]), (-1, [-1 / sqrt(q), -test_value_1])]
    else:
        return [(1, [-BIG_NUMBER, 1 / sqrt(q)]), (-1, [-1 / sqrt(q), BIG_NUMBER])]


def line_borders_hinge_above(m, q, V):
    test_value_1 = (1 - V) / sqrt(q)

    if test_value_1 > -BIG_NUMBER:
        return [(1, [-BIG_NUMBER, test_value_1]), (-1, [-test_value_1, BIG_NUMBER])]
    else:
        return [(1, [-BIG_NUMBER, -BIG_NUMBER]), (-1, [BIG_NUMBER, BIG_NUMBER])]


def stability_integration_domains():
    domains_z = [[-BIG_NUMBER, 0.0], [0.0, BIG_NUMBER]]
    domains_ω = [
        [lambda z: -BIG_NUMBER, lambda z: BIG_NUMBER],
        [lambda z: -BIG_NUMBER, lambda z: BIG_NUMBER],
    ]

    return domains_z, domains_ω


def stability_integration_domains_triple():
    domains_z = [[-BIG_NUMBER, BIG_NUMBER], [-BIG_NUMBER, BIG_NUMBER]]
    domains_ω = [
        [lambda z: -BIG_NUMBER, lambda z: BIG_NUMBER],
        [lambda z: -BIG_NUMBER, lambda z: BIG_NUMBER],
    ]
    domains_w = [
        [lambda z, ω: -BIG_NUMBER, lambda z, ω: -z],
        [lambda z, ω: -z, lambda z, ω: BIG_NUMBER],
    ]

    return domains_z, domains_ω, domains_w
