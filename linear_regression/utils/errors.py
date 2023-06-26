class ConvergenceError(RuntimeError):
    def __init__(self, fname, n_iteration, *args, **kwargs):
        super().__init__(
            "The function {} didn't converge after {:d} iterations".format(fname, n_iteration),
            *args,
            **kwargs
        )


class MinimizationError(RuntimeError):
    def __init__(self, fname, initial_point, *args, **kwargs):
        super().__init__(
            "The function {} could't find a minima starting from {}.".format(fname, initial_point),
            *args,
            **kwargs
        )
