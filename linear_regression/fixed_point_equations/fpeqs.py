from ..fixed_point_equations import (
    BLEND_FPE,
    TOL_FPE,
    REL_TOL_FPE,
    MIN_ITER_FPE,
    MAX_ITER_FPE,
    PRINT_EVERY,
    ANDERSON_PREVIOUS_POINTS,
)
from ..utils.errors import ConvergenceError
from ..aux_functions.misc import damped_update, max_difference
from numpy import array, roll, zeros
from numpy.linalg import lstsq, LinAlgError


# ---------------------------------------------------------------------------- #
def print_status_fixed_point(iter_nb, x, err):
    print(f"iter {iter_nb}: x = ", end="")
    for val in x:
        print(f"{val:.6f}", end=", ")
    print(f"err = {err:.6e}")


# ----------------------------- finder functions ----------------------------- #


def fixed_point_finder(
    f_func: callable,
    f_hat_func: callable,
    initial_condition: tuple[float, ...],
    f_kwargs: dict,
    f_hat_kwargs: dict,
    abs_tol: float = TOL_FPE,
    min_iter: int = MIN_ITER_FPE,
    max_iter: int = MAX_ITER_FPE,
    update_function: callable = damped_update,
    args_update_function: tuple = (BLEND_FPE,),
    error_function: callable = max_difference,
    verbose: bool = False,
    print_every: int = PRINT_EVERY,
):
    x = initial_condition
    err = 1.0
    iter_nb = 0

    if verbose:
        print_status_fixed_point(iter_nb, x, err)

    while err > abs_tol or iter_nb < min_iter:
        y = f_hat_func(*x, **f_hat_kwargs)
        new_x = f_func(*y, **f_kwargs)

        err = error_function(new_x, x)

        x = tuple(update_function(new_x, x, *args_update_function))

        iter_nb += 1

        if verbose and iter_nb % print_every == 0 and iter_nb > 0:
            print_status_fixed_point(iter_nb, x, err)

        if iter_nb > max_iter:
            raise ConvergenceError(
                f"fixed_point_finder with {f_kwargs} and {f_hat_kwargs}", iter_nb
            )

    if verbose:
        print("Final result fixed_point_finder: ", end="")
        print_status_fixed_point(iter_nb, x, err)

    return tuple(x)


def fixed_point_finder_anderson(
    f_func: callable,
    f_hat_func: callable,
    initial_condition: tuple[float, ...],
    f_kwargs: dict,
    f_hat_kwargs: dict,
    abs_tol: float = TOL_FPE,
    min_iter: int = MIN_ITER_FPE,
    max_iter: int = MAX_ITER_FPE,
    m: int = ANDERSON_PREVIOUS_POINTS,
    verbose: bool = False,
    update_function: callable = damped_update,
    args_update_function: tuple = (BLEND_FPE,),
    error_function: callable = max_difference,
    print_every: int = PRINT_EVERY,
):
    x = array(initial_condition)
    n_params = len(x)
    iter_nb = 0
    err = 1.0

    X = zeros((n_params, m + 1))
    F = zeros((n_params, m + 1))

    if verbose:
        print_status_fixed_point(iter_nb, x, err)

    while iter_nb < min_iter or err > abs_tol:

        y = f_hat_func(*x, **f_hat_kwargs)
        fx = array(f_func(*y, **f_kwargs))

        res = fx - x

        if iter_nb < m:
            X[:, iter_nb] = x
            F[:, iter_nb] = fx
        else:
            X = roll(X, -1, axis=1)
            F = roll(F, -1, axis=1)
            X[:, -1] = x
            F[:, -1] = fx

        if iter_nb > 1:
            m_k = min(m, iter_nb)

            G = F - X
            dG = G[:, 1 : m_k + 1] - G[:, :m_k]
            dX = X[:, 1 : m_k + 1] - X[:, :m_k]

            try:
                gammas = lstsq(dG, res, rcond=None)[0]
                x_new = x + res - (dX + dG) @ gammas
            except LinAlgError:
                x_new = fx
        else:
            x_new = fx

        err = error_function(x_new, x)

        if verbose and iter_nb % print_every == 0 and iter_nb > 0:
            print_status_fixed_point(iter_nb, x, err)

        x = update_function(x_new, x, *args_update_function)

        iter_nb += 1

        if iter_nb >= max_iter:
            raise ConvergenceError("fixed_point_finder_anderson", iter_nb)

    if verbose:
        print("Final result fixed_point_finder_anderson: ", end="")
        print_status_fixed_point(iter_nb, x, err)

    return tuple(x)


# ------------------------------- old functions ------------------------------ #


def fixed_point_finder_old(
    f_func,
    f_hat_func,
    initial_condition: tuple[float, float, float],
    f_kwargs: dict,
    f_hat_kwargs: dict,
    abs_tol: float = TOL_FPE,
    # rel_tol: float = REL_TOL_FPE,
    min_iter: int = MIN_ITER_FPE,
    max_iter: int = MAX_ITER_FPE,
):
    m, q, V = initial_condition[0], initial_condition[1], initial_condition[2]
    err = 1.0
    iter_nb = 0
    while err > abs_tol or iter_nb < min_iter:
        m_hat, q_hat, V_hat = f_hat_func(m, q, V, **f_hat_kwargs)
        new_m, new_q, new_V = f_func(m_hat, q_hat, V_hat, **f_kwargs)

        err = max([abs(new_m - m), abs(new_q - q), abs(new_V - V)])

        m = damped_update(new_m, m, BLEND_FPE)
        q = damped_update(new_q, q, BLEND_FPE)
        V = damped_update(new_V, V, BLEND_FPE)

        iter_nb += 1
        if iter_nb > max_iter:
            raise ConvergenceError("fixed_point_finder", iter_nb)

    return m, q, V


def fixed_point_finder_adversiaral(
    f_func,
    f_hat_func,
    initial_condition: tuple[float, float, float],
    f_kwargs: dict,
    f_hat_kwargs: dict,
    abs_tol: float = TOL_FPE,
    min_iter: int = MIN_ITER_FPE,
    max_iter: int = MAX_ITER_FPE,
    verbose: bool = False,
):
    m, q, V, P = (
        initial_condition[0],
        initial_condition[1],
        initial_condition[2],
        initial_condition[3],
    )
    err = 1.0
    iter_nb = 0

    if verbose:
        print(f"Fixed point Finder Adversarial called with:")
        print(f"Initial conditions m = {m:.3e}, q = {q:.3e}, V = {V:.3e}, P = {P:.3e}")
        print(f"Parameters f_func = {f_func}, f_hat_func = {f_hat_func}")

    while err > abs_tol or iter_nb < min_iter:
        m_hat, q_hat, V_hat, P_hat = f_hat_func(m, q, V, P, **f_hat_kwargs)

        if verbose and iter_nb % PRINT_EVERY == 0:
            print(
                f"m_hat = {m_hat:.3e}, q_hat = {q_hat:.3e}, V_hat = {V_hat:.3e}, P_hat = {P_hat:.3e}"
            )

        new_m, new_q, new_V, new_P = f_func(m_hat, q_hat, V_hat, P_hat, **f_kwargs)

        err = max(
            [
                abs((new_m - m)),
                abs((new_q - q)),
                abs((new_V - V)),
                abs((new_P - P)),
            ]
        )

        m = damped_update(new_m, m, BLEND_FPE)
        q = damped_update(new_q, q, BLEND_FPE)
        V = damped_update(new_V, V, BLEND_FPE)
        P = damped_update(new_P, P, BLEND_FPE)

        if verbose and iter_nb % PRINT_EVERY == 0:
            print(f"m = {m:.3e}, q = {q:.3e}, V = {V:.3e}, P = {P:.3e}")
            print(f"err = {err:.3e}")

        iter_nb += 1
        if iter_nb > max_iter:
            raise ConvergenceError("fixed_point_finder", iter_nb)

    if verbose:
        print(f"Fixed point Finder Adversarial finished with:")
        print(f"Final conditions m = {m:.3e}, q = {q:.3e}, V = {V:.3e}, P = {P:.3e}")

    return m, q, V, P


def fixed_point_finder_loser(
    f_func,
    f_hat_func,
    initial_condition: tuple[float, float, float],
    f_kwargs: dict,
    f_hat_kwargs: dict,
    abs_tol: float = TOL_FPE,
    rel_tol: float = REL_TOL_FPE,
    min_iter: int = MIN_ITER_FPE,
    max_iter: int = MAX_ITER_FPE,
    control_variate: tuple[bool, bool, bool] = (True, True, True),
):
    m, q, V = initial_condition[0], initial_condition[1], initial_condition[2]
    err = 1.0
    iter_nb = 0
    while err > abs_tol or iter_nb < min_iter:
        m_hat, q_hat, V_hat = f_hat_func(m, q, V, **f_hat_kwargs)
        new_m, new_q, new_V = f_func(m_hat, q_hat, V_hat, **f_kwargs)

        errs = list()
        if control_variate[0]:
            errs.append(abs(new_m - m))
        if control_variate[1]:
            errs.append(abs(new_q - q))
        if control_variate[2]:
            errs.append(abs(new_V - V))

        err = max(errs)

        # m = damped_update(new_m, m, BLEND_FPE)
        # q = damped_update(new_q, q, BLEND_FPE)
        # V = damped_update(new_V, V, BLEND_FPE)

        m = BLEND_FPE * new_m + (1 - BLEND_FPE) * m
        q = BLEND_FPE * new_q + (1 - BLEND_FPE) * q
        V = BLEND_FPE * new_V + (1 - BLEND_FPE) * V

        iter_nb += 1
        if iter_nb > max_iter:
            raise ConvergenceError("fixed_point_finder", iter_nb)

    # print(
    #     "\t\t\tm = {:.1e} Δm = {:.1e} q = {:.1e} Δq = {:.1e} V = {:.1e} ΔV = {:.1e} ".format(
    #         m, abs(new_m - m), q, abs(new_q - q), V, abs(new_V - V)
    #     )
    # )

    return m, q, V


# implementation of the same as fixed_point_finder but using Anderson acceleration for the fixed point equation
# def anderson_fixed_point_finder(
#     f_func,
#     f_hat_func,
#     initial_condition: tuple[float, float, float],
#     f_kwargs: dict,
#     f_hat_kwargs: dict,
#     abs_tol: float = TOL_FPE,
#     min_iter: int = MIN_ITER_FPE,
#     max_iter: int = MAX_ITER_FPE,
# ):
#     m, q, V = initial_condition[0], initial_condition[1], initial_condition[2]
#     err = 1.0
#     iter_nb = 0
#     for idx in range(max_iter):


#         if err < abs_tol:
#             break
#         if iter_nb > min_iter:
#             raise ConvergenceError("anderson_fixed_point_finder", iter_nb)

#     return m, q, V


def plateau_fixed_point_finder(
    x_next_func,
    inital_condition: float,
    x_next_func_kwargs: dict,
    abs_tol: float = TOL_FPE,
    min_iter: int = MIN_ITER_FPE,
    max_iter: int = MAX_ITER_FPE,
):
    x = inital_condition
    err = 1.0
    iter_nb = 0
    while err > abs_tol or iter_nb < min_iter:
        x_next = x_next_func(x, **x_next_func_kwargs)
        err = abs(x_next - x)
        x = x_next
        if iter_nb > max_iter:
            raise ConvergenceError("plateau_fixed_point_finder", iter_nb)
    return x
