import numpy as np
from scipy.integrate import romb
from numba import njit
from sklearn.metrics import max_error

MULT_INTEGRAL = 10
TOL_INT = 1e-6

N_TEST_POINTS = 200

N_GAUSS_HERMITE = 95


x_ge, w_ge = np.polynomial.hermite.hermgauss(N_GAUSS_HERMITE)


@njit(error_model="numpy", fastmath=True)
def gauss_hermite_quadrature(fun, mean, std):
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
    fun, scale1, scale2, mult=MULT_INTEGRAL, tol=TOL_INT, n_points=N_TEST_POINTS, args=[]
):
    borders = [[-mult * scale1, mult * scale1], [-mult * scale2, mult * scale2]]

    for idx, ax in enumerate(borders):
        for jdx, border in enumerate(ax):
            while True:
                if idx == 0:
                    max_val = np.max(
                        [
                            fun(borders[idx][jdx], pt, *args)
                            for pt in np.linspace(
                                borders[1 if idx == 0 else 0][0],
                                borders[1 if idx == 0 else 0][1],
                                n_points,
                            )
                        ]
                    )
                else:
                    max_val = np.max(
                        [
                            fun(pt, borders[idx][jdx], *args)
                            for pt in np.linspace(
                                borders[1 if idx == 0 else 0][0],
                                borders[1 if idx == 0 else 0][1],
                                n_points,
                            )
                        ]
                    )
                if max_val > tol:
                    borders[idx][jdx] = borders[idx][jdx] + (-1.0 if jdx == 0 else 1.0) * (
                        scale1 if idx == 0 else scale2
                    )
                else:
                    break

    for ax in borders:
        ax[0] = -np.max(np.abs(ax))
        ax[1] = np.max(np.abs(ax))

    max_val = np.max([borders[0][1], borders[1][1]])

    borders = [[-max_val, max_val], [-max_val, max_val]]

    return borders


def divide_integration_borders_multiple_grid(square_borders, N=10):
    if N < 1:
        raise ValueError("N must be greater than 1")

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


def domains_double_line_constraint(square_borders, y_fun_upper, y_fun_lower, x_fun_upper, args1, args2, args3):
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
            domain_x = [[-max_range, -x_test_val], [-x_test_val, x_test_val], [x_test_val, max_range]]
            domain_y = [
                [lambda x: hyp(x, **arg_hyp), lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
                [lambda x: -max_range, lambda x: hyp(x, **arg_hyp)],
            ]

    return domain_x, domain_y
