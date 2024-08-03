from numpy import array, ones, log, vstack, logspace
from numpy.linalg import lstsq


def log_log_linear_fit(x, y, base=10, return_points=False, extend_percent=0.1):
    """
    Perform a log-log linear fit on input vectors x and y.

    Parameters:
    x (array-like): Independent variable values.
    y (array-like): Dependent variable values.
    base (float, optional): Base of the logarithm. Defaults to 10.
    return_points (bool, optional): If True, return points for plotting. Defaults to False.
    extend_percent (float, optional): Percentage to extend the fit range. Defaults to 0.1 (10%).

    Returns:
    If return_points is False:
        tuple: (m, coefficient) where m is the slope and coefficient is the y-intercept in original scale.
    If return_points is True:
        tuple: (m, coefficient, (x_fit, y_fit)) where (x_fit, y_fit) are points for plotting the fit.

    The function performs a linear regression on log-transformed data and returns the power-law
    fit parameters. If return_points is True, it also provides points for plotting the fit,
    extended by extend_percent beyond the input data range.
    """
    x, y = array(x), array(y)
    log_x = log(x) / log(base)
    log_y = log(y) / log(base)

    A = vstack([log_x, ones(len(log_x))]).T
    m, c = lstsq(A, log_y, rcond=None)[0]

    coefficient = base**c

    if return_points:
        log_x_min, log_x_max = log(min(x)) / log(base), log(max(x)) / log(base)
        log_x_range = log_x_max - log_x_min
        extended_log_x_min = log_x_min - extend_percent * log_x_range
        extended_log_x_max = log_x_max + extend_percent * log_x_range

        x_fit = logspace(extended_log_x_min, extended_log_x_max, 100, base=base)
        y_fit = coefficient * x_fit**m

        return m, coefficient, (x_fit, y_fit)
    else:
        return m, coefficient
