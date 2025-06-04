from numpy import empty
from numba import njit
from scipy.optimize import minimize
from ..utils.errors import MinimizationError
from .fpeqs import fixed_point_finder
from ..fixed_point_equations import SMALLEST_REG_PARAM, SMALLEST_HUBER_PARAM, XATOL, FATOL
from ..aux_functions.misc import estimation_error
import numpy as np

# --------------------------------


def find_optimal_reg_param_function(
    f_func,
    f_hat_func,
    f_kwargs: dict,
    f_hat_kwargs: dict,
    initial_guess_reg_param: float,
    initial_cond_fpe: tuple[float, float, float],
    funs=[estimation_error],
    funs_args=[list()],
    f_min=estimation_error,
    f_min_args={},
    min_reg_param=SMALLEST_REG_PARAM,
):
    if len(funs) != len(funs_args):
        raise ValueError(
            "The length of funs and funs_args should be the same, in this case is {:d} and {:d}".format(
                len(funs), len(funs_args)
            )
        )
    n_observables = len(funs)
    copy_f_kwargs = f_kwargs.copy()

    def minimize_fun(reg_param):
        copy_f_kwargs.update({"reg_param": float(reg_param)})
        print("\t\tλ = {:.5f}".format(float(reg_param)))
        m, q, V = fixed_point_finder(
            f_func,
            f_hat_func,
            initial_condition=initial_cond_fpe,
            f_kwargs=copy_f_kwargs,
            f_hat_kwargs=f_hat_kwargs,
        )
        return f_min(m, q, V, **f_min_args)

    bnds = [(min_reg_param, None)]
    obj = minimize(
        minimize_fun,
        x0=initial_guess_reg_param,
        method="Nelder-Mead",
        bounds=bnds,
        options={"xatol": XATOL, "fatol": FATOL},
    )

    if obj.success:
        fun_min_val = obj.fun
        reg_param_opt = obj.x

        copy_f_kwargs.update({"reg_param": float(reg_param_opt)})
        out_values = empty(n_observables)
        m, q, V = fixed_point_finder(
            f_func,
            f_hat_func,
            initial_condition=initial_cond_fpe,
            f_kwargs=copy_f_kwargs,
            f_hat_kwargs=f_hat_kwargs,
        )

        for idx, (f, f_args) in enumerate(zip(funs, funs_args)):
            out_values[idx] = f(m, q, V, **f_args)                   # Beware of the ** instead of * in the f_args

        return fun_min_val, reg_param_opt, (m, q, V), out_values
    else:
        raise MinimizationError("find_optimal_reg_param_function", initial_guess_reg_param)


def find_optimal_reg_and_huber_parameter_function(
    f_func,
    f_hat_func,
    f_kwargs: dict,
    f_hat_kwargs: dict,
    initial_guess_reg_and_huber_param: tuple[float, float],
    initial_cond_fpe: tuple[float, float, float],
    funs=[estimation_error],
    funs_args=[list()],
    f_min=estimation_error,
    f_min_args={},
    min_reg_param=SMALLEST_REG_PARAM,
    min_huber_param=SMALLEST_HUBER_PARAM,
):
    if len(funs) != len(funs_args):
        raise ValueError(
            "The length of funs and funs_args should be the same, in this case is {:d} and {:d}".format(
                len(funs), len(funs_args)
            )
        )
    n_observables = len(funs)
    copy_f_kwargs = f_kwargs.copy()
    copy_f_hat_kwargs = f_hat_kwargs.copy()

    def minimize_fun(x):
        print("\t\tλ = {:.5f}, τ = {:.5f}".format(x[0], x[1]))
        copy_f_kwargs.update({"reg_param": x[0]})
        copy_f_hat_kwargs.update({"tau": x[1]})

        m, q, V = fixed_point_finder(
            f_func,
            f_hat_func,
            initial_condition=initial_cond_fpe,
            f_kwargs=copy_f_kwargs,
            f_hat_kwargs=copy_f_hat_kwargs,
        )
        return f_min(m, q, V, **f_min_args)

    bnds = [(min_reg_param, None), (min_huber_param, None)]
    obj = minimize(
        minimize_fun,
        x0=list(initial_guess_reg_and_huber_param),
        method="Nelder-Mead",
        bounds=bnds,
        options={
            "xatol": XATOL,
            "fatol": FATOL,
            "adaptive": True,
        },
    )
    if obj.success:
        fun_min_val = obj.fun
        reg_param_opt, a_opt = obj.x

        copy_f_kwargs.update({"reg_param": reg_param_opt})
        copy_f_hat_kwargs.update({"tau": a_opt})
        out_values = empty(n_observables)
        m, q, V = fixed_point_finder(
            f_func,
            f_hat_func,
            initial_condition=initial_cond_fpe,
            f_kwargs=copy_f_kwargs,
            f_hat_kwargs=copy_f_hat_kwargs,
        )

        for idx, (f, f_args) in enumerate(zip(funs, funs_args)):
            out_values[idx] = f(m, q, V, **f_args)                 # Beware of the ** instead of * in the f_args

        return fun_min_val, (reg_param_opt, a_opt), (m, q, V), out_values
    else:
        raise MinimizationError(
            "find_optimal_reg_and_huber_parameter_function", initial_guess_reg_and_huber_param
        )

# --- Matéo begins here ---

# =============================================================================
#  find_optimal_reg_param_function (with optional barrier constraints & generic loss parameter)
# =============================================================================

def find_optimal_reg_param_function_(
    f_func,
    f_hat_func,
    f_kwargs: dict,
    f_hat_kwargs: dict,
    initial_guess_reg_param: float,
    initial_cond_fpe: tuple[float, float, float],
    alpha_key: str ='alpha',
    funs: list = None,
    funs_args: list = None,
    f_min=None,
    f_min_args: dict = None,
    barrier_V_threshold: float = None,
    barrier_RS_threshold: float = 1.0,
    barrier_penalty: float = 1e10,
    RS_func=None,
    min_reg_param=1e-8,
    verbose = False
):
    """
    Find the optimal regularization parameter (lambda) for a fixed alpha (passed in f_hat_kwargs[alpha_key]),
    enforcing optional barrier constraints on V and RS.

    Arguments:
    -----------
    f_func              : function(m_hat, q_hat, V_hat, **f_kwargs) -> (m, q, V)
    f_hat_func          : function(m, q, V, **f_hat_kwargs) -> (m_hat, q_hat, V_hat)
    f_kwargs            : dict of keyword arguments for f_func, excluding 'reg_param'
    f_hat_kwargs        : dict of keyword arguments for f_hat_func, must include {alpha_key: alpha_value, loss_param_name: value (if any), ...}
    initial_guess_reg_param : starting guess for lambda
    initial_cond_fpe    : tuple (m0, q0, V0) initial condition for fixed_point_finder
    alpha_key           : string key in f_hat_kwargs that stores the current alpha
    funs                : list of callable observables, each with signature f(m, q, V, **args)
    funs_args           : list of dicts, each matching a callable in funs to pass keyword args, not counting 'm', 'q', 'V'
    f_min               : callable to minimize (e.g., excess_gen_error), signature f_min(m, q, V, **f_min_args)
    f_min_args          : dict of keyword args for f_min
    barrier_V_threshold : float threshold on V; if V >= threshold, return barrier_penalty
    barrier_RS_threshold: float threshold on RS; if RS >= threshold, return barrier_penalty
    barrier_penalty     : float penalty to return when barriers are violated
    RS_func             : callable to compute RS_func(m, q, V, **rs_args)
    min_reg_param       : float minimum allowed lambda
    verbose             : bool, if True, print progress messages

    Returns:
    --------
    (reg_param_opt, (m_opt, q_opt, V_opt, m_hat_opt, q_hat_opt, V_hat_opt), observables_out)

    - reg_param_opt : optimal lambda (scalar)
    - (m_opt, q_opt, V_opt, m_hat_opt, q_hat_opt, V_hat_opt) : tuple of fixed-point solutions
    - observables_out : numpy array of length len(funs), each f_i(m_opt, q_opt, V_opt, **funs_args[i])
    """
    # Validate funs / funs_args lengths
    if funs is None:
        funs = []
    if funs_args is None:
        funs_args = [{}] * len(funs)
    if len(funs) != len(funs_args):
        raise ValueError(f"Length mismatch: len(funs)={len(funs)}, len(funs_args)={len(funs_args)}")

    # f_min and its args
    if f_min is None:
        raise ValueError("A minimization function f_min must be provided")
    if f_min_args is None:
        f_min_args = {}

    # Make local copies to avoid side-effects
    base_f_kwargs = f_kwargs.copy()
    base_f_hat_kwargs = f_hat_kwargs.copy()

    # Internal function to evaluate the barrier-constrained objective at a given lambda
    def minimize_fun(reg_param):
        reg_param = float(reg_param)
        if reg_param < min_reg_param:
            reg_param = min_reg_param
        if verbose:
            print(f"\t\tλ = {reg_param:.5f}")
        # Update f_kwargs with current lambda
        local_f_kwargs = base_f_kwargs.copy()
        local_f_kwargs["reg_param"] = reg_param

        local_f_hat_kwargs = base_f_hat_kwargs.copy()

        # 1) Solve fixed-point equations for (m, q, V)
        m, q, V = fixed_point_finder(
            f_func=f_func,
            f_hat_func=f_hat_func,
            initial_condition=initial_cond_fpe,
            f_kwargs=local_f_kwargs,
            f_hat_kwargs=local_f_hat_kwargs,
            # abs_tol=None,    
            # min_iter=None,
            # max_iter=None,
            # etc,
            verbose=verbose
        )
        
        # 2) Barrier on V
        if barrier_V_threshold is not None and (V >= barrier_V_threshold or V < 0):
            if verbose:
                print(f"\t\tBarrier on V violated: V = {V}, threshold = {barrier_V_threshold} at λ = {reg_param:.5f}, alpha = {local_f_hat_kwargs.get(alpha_key, 1.0)}")
            return barrier_penalty
        
        # 3) Barrier on RS stability if RS_func provided
        if RS_func is not None and barrier_RS_threshold is not None:
            # Build args for RS_func: it typically requires (m, q, V, alpha, any noise/loss params)
            rs_kwargs = {}
            rs_kwargs.update(local_f_hat_kwargs)
            rs_kwargs["reg_param"] = reg_param
            # Extract alpha
            alpha_val = float(local_f_hat_kwargs.get(alpha_key, 1.0))
            rs_kwargs[alpha_key] = alpha_val
            try:
                RS_val = RS_func(m, q, V, **rs_kwargs)
                if RS_val >= barrier_RS_threshold:
                    if verbose:
                        print(f"\t\tBarrier on RS violated: RS = {RS_val}, threshold = {barrier_RS_threshold} at λ = {reg_param:.5f}, alpha = {alpha_val}")
                    return barrier_penalty
            except Exception as e:
                print(f"\t\tRS_func raised an exception: {e}")
                return barrier_penalty
        # 4) If all barriers passed, compute the objective f_min
        return f_min(m, q, V, **f_min_args)

    # Perform Nelder-Mead minimization over lambda
    bounds = [(min_reg_param, None)]
    result = minimize(
        minimize_fun,
        x0=[float(initial_guess_reg_param)],
        method="Nelder-Mead",
        bounds=bounds,
        options={"xatol": 1e-8, "fatol": 1e-8},
    )

    if not result.success:
        raise MinimizationError("find_optimal_reg_param_function", initial_guess_reg_param)

    reg_param_opt = float(result.x[0])

    # Recompute the fixed-point at optimal lambda to extract observables
    final_f_kwargs = base_f_kwargs.copy()
    final_f_kwargs["reg_param"] = reg_param_opt
    final_f_hat_kwargs = base_f_hat_kwargs.copy()

    m_opt, q_opt, V_opt = fixed_point_finder(
        f_func=f_func,
        f_hat_func=f_hat_func,
        initial_condition=initial_cond_fpe,
        f_kwargs=final_f_kwargs,
        f_hat_kwargs=final_f_hat_kwargs,
        # etc
    )

    # Compute the requested observables
    m_hat_opt, q_hat_opt, V_hat_opt = f_hat_func(m_opt, q_opt, V_opt, **final_f_hat_kwargs)
    n_obs = len(funs)
    observables_out = np.empty(n_obs, dtype=float)
    for idx, (fun, args) in enumerate(zip(funs, funs_args)):
        observables_out[idx] = fun(m_opt, q_opt, V_opt, **args)

    return reg_param_opt, (m_opt, q_opt, V_opt, m_hat_opt, q_hat_opt, V_hat_opt), observables_out


# =============================================================================
#  find_optimal_reg_and_loss_param_function (for jointly optimizing reg_param & a generic loss parameter) # NOT TESTED YET
# =============================================================================

def find_optimal_reg_and_loss_param_function(
    f_func,
    f_hat_func,
    f_kwargs: dict,
    f_hat_kwargs: dict,
    initial_guess_reg_and_loss: tuple[float, float],
    initial_cond_fpe: tuple[float, float, float],
    alpha_key: str,
    loss_param_name: str,
    funs: list = None,
    funs_args: list = None,
    f_min=None,
    f_min_args: dict = None,
    barrier_V_threshold: float = None,
    barrier_RS_threshold: float = None,
    barrier_penalty: float = 1e10,
    RS_func=None,
    min_reg_param=1e-8,
    min_loss_param=1e-8,
):
    """
    Jointly optimize regularization parameter (lambda) AND a generic loss parameter (e.g., 'tau' or 'c'),
    with optional barrier constraints on V and RS. The current alpha is read from f_hat_kwargs[alpha_key].

    Arguments:
    -----------
    f_func                  : function(...) -> ... for fixed-point
    f_hat_func              : function(...) -> ... for fixed-point
    f_kwargs                : dict of kwargs for f_func (without 'reg_param')
    f_hat_kwargs            : dict of kwargs for f_hat_func (must include {alpha_key: alpha_value, loss_param_name: initial_loss})
    initial_guess_reg_and_loss : tuple (lambda_guess, loss_param_guess)
    initial_cond_fpe        : (m0, q0, V0)
    alpha_key               : string key in f_hat_kwargs for alpha
    loss_param_name         : string key in f_hat_kwargs for the loss parameter
    funs                    : list of callables to compute observables after finding optimum
    funs_args               : list of dicts of kwargs for each callable in funs
    f_min                   : callable to minimize (e.g., excess_gen_error with barriers)
    f_min_args              : dict of kwargs for f_min
    barrier_V_threshold     : float threshold on V, or None
    barrier_RS_threshold    : float threshold on RS, or None
    barrier_penalty         : float penalty value when barrier is violated
    RS_func                 : callable to compute E2 = RS_func(m, q, V, **rs_kwargs)
    min_reg_param           : float lower bound for lambda
    min_loss_param          : float lower bound for loss parameter

    Returns:
    --------
    (fun_min_val, (reg_param_opt, loss_param_opt), (m_opt, q_opt, V_opt), observables_out)
    """
    # Validate funs / funs_args lengths
    if funs is None:
        funs = []
    if funs_args is None:
        funs_args = [{}] * len(funs)
    if len(funs) != len(funs_args):
        raise ValueError(f"Length mismatch: len(funs)={len(funs)}, len(funs_args)={len(funs_args)}")

    if f_min is None:
        raise ValueError("A minimization function f_min must be provided")
    if f_min_args is None:
        f_min_args = {}

    base_f_kwargs = f_kwargs.copy()
    base_f_hat_kwargs = f_hat_kwargs.copy()

    # Internal function to evaluate barrier-constrained objective at (lambda, loss_param)
    def minimize_fun(x):
        reg_param = float(x[0])
        loss_param = float(x[1])

        # Enforce minimum bounds
        if reg_param < min_reg_param:
            reg_param = min_reg_param
        if loss_param < min_loss_param:
            loss_param = min_loss_param

        # Update f_kwargs and f_hat_kwargs
        local_f_kwargs = base_f_kwargs.copy()
        local_f_kwargs["reg_param"] = reg_param

        local_f_hat_kwargs = base_f_hat_kwargs.copy()
        local_f_hat_kwargs[loss_param_name] = loss_param

        # Solve fixed-point for (m, q, V)
        m, q, V = fixed_point_finder(
            f_func=f_func,
            f_hat_func=f_hat_func,
            initial_condition=initial_cond_fpe,
            f_kwargs=local_f_kwargs,
            f_hat_kwargs=local_f_hat_kwargs,
            abs_tol=None,
            min_iter=None,
            max_iter=None
        )

        # Barrier on V
        if barrier_V_threshold is not None and V >= barrier_V_threshold:
            return barrier_penalty

        # Barrier on RS
        if RS_func is not None and barrier_RS_threshold is not None:
            rs_kwargs = {}
            rs_kwargs.update(local_f_hat_kwargs)
            rs_kwargs["m"] = m
            rs_kwargs["q"] = q
            rs_kwargs["V"] = V
            alpha_val = float(local_f_hat_kwargs.get(alpha_key, 1.0))
            try:
                E2_val = RS_func(m, q, V, **rs_kwargs)
                if not np.isfinite(E2_val):
                    return barrier_penalty
                RS_val = alpha_val * (V ** 2) * E2_val
                if RS_val >= barrier_RS_threshold:
                    return barrier_penalty
            except Exception:
                return barrier_penalty

        # If barriers passed, return f_min
        return f_min(m, q, V, **f_min_args)

    # Set bounds: [(min_reg, None), (min_loss, None)]
    bounds = [(min_reg_param, None), (min_loss_param, None)]
    x0 = [float(initial_guess_reg_and_loss[0]), float(initial_guess_reg_and_loss[1])]

    result = minimize(
        minimize_fun,
        x0=x0,
        method="Nelder-Mead",
        bounds=bounds,
        options={"xatol": 1e-8, "fatol": 1e-8, "adaptive": True},
    )

    if not result.success:
        raise MinimizationError("find_optimal_reg_and_loss_param_function", initial_guess_reg_and_loss)

    fun_min_val = result.fun
    reg_param_opt = float(result.x[0])
    loss_param_opt = float(result.x[1])

    # Recompute the fixed-point solution at optimal (reg_param_opt, loss_param_opt)
    final_f_kwargs = base_f_kwargs.copy()
    final_f_kwargs["reg_param"] = reg_param_opt

    final_f_hat_kwargs = base_f_hat_kwargs.copy()
    final_f_hat_kwargs[loss_param_name] = loss_param_opt

    m_opt, q_opt, V_opt = fixed_point_finder(
        f_func=f_func,
        f_hat_func=f_hat_func,
        initial_condition=initial_cond_fpe,
        f_kwargs=final_f_kwargs,
        f_hat_kwargs=final_f_hat_kwargs,
    )

    # Compute observables
    n_obs = len(funs)
    observables_out = np.empty(n_obs, dtype=float)
    for idx, (fun, args) in enumerate(zip(funs, funs_args)):
        observables_out[idx] = fun(m_opt, q_opt, V_opt, **args)

    return (fun_min_val, (reg_param_opt, loss_param_opt), (m_opt, q_opt, V_opt), observables_out)
