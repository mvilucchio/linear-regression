from unittest import TestCase
import numpy as np
import random
import inspect
import numba
import statistics


def stencil_derivative_1d(fun, h, x):
    return (
        -1259728571316 * fun(x - 2.5 * h)
        + 17321267855599 * fun(x - 2 * h)
        - 115475119037326 * fun(x - 1.5 * h)
        + 519638035668003 * fun(x - 1 * h)
        - 2078552142671768 * fun(x - 0.5 * h)
        - 484995499956000 * fun(x + 0 * h)
        + 2909972999739752 * fun(x + 0.5 * h)
        - 1039276071335648 * fun(x + 1 * h)
        + 346425357111896 * fun(x + 1.5 * h)
        - 86606339277977 * fun(x + 2 * h)
        + 13857014284477 * fun(x + 2.5 * h)
        - 1049773809430 * fun(x + 3 * h)
    ) / (1454986499869980 * 1.0 * h**1)


class TestFunctionGeneral(TestCase):
    def get_args(self, func):
        if isinstance(func, numba.core.registry.CPUDispatcher):
            py_func = func.py_func
            return len(inspect.signature(py_func).parameters)
        elif isinstance(func, numba.np.ufunc.dufunc.DUFunc):
            return func.nin
        else:
            return len(inspect.signature(func).parameters)

    def sample_arg(self, signature):
        if signature == "+":
            return random.uniform(1e-3, 50)
        elif signature == "++":
            return random.uniform(1e-6, 100)
        elif signature == "u":
            return random.uniform(-50, 50)
        elif signature == "u+":
            return random.uniform(-100, 100)
        elif signature == "0-1":
            return random.uniform(0, 1)
        elif signature == "b":
            return random.choice([0, 1])
        elif signature == "+/-":
            return random.choice([1, -1])
        elif signature == "n":
            return random.gauss(0, 1)
        elif isinstance(signature, tuple):
            return random.choice(signature)
        else:
            raise ValueError("Invalid argument signature")


class TestFunctionComparison(TestFunctionGeneral):
    def compare_two_functions(
        self, func1, func2, num_points=100, tolerance=1e-4, arg_signatures=None
    ):
        num_args_func1 = self.get_args(func1)
        num_args_func2 = self.get_args(func2)

        if num_args_func1 != num_args_func2:
            raise ValueError("Functions have different number of arguments")

        if arg_signatures is not None and len(arg_signatures) != num_args_func1:
            raise ValueError("Length of arg_signatures doesn't match the number of arguments")

        for _ in range(num_points):
            if arg_signatures is None:
                args = [np.array([random.uniform(-100, 100)]) for _ in range(num_args_func1)]
            else:
                # args = [np.array([self.sample_arg(sig)]) for sig in arg_signatures]
                args = [float(self.sample_arg(sig)) for sig in arg_signatures]

            result1 = func1(*args)
            result2 = func2(*args)

            # Convert results to scalar values for comparison
            result1_scalar = result1.item() if isinstance(result1, np.ndarray) else result1
            result2_scalar = result2.item() if isinstance(result2, np.ndarray) else result2

            self.assertAlmostEqual(
                result1_scalar,
                result2_scalar,
                delta=tolerance,
                msg=f"Functions outputs don't match at args = {args}: {result1_scalar} != {result2_scalar}",
            )


class TestComparisonTheoryExperiment(TestFunctionGeneral):
    def test_comparison_thoery_experiment(
        self, theory_func, exp_func, num_points=10, repetitions=10, arg_signatures=None
    ):
        num_args_theory_func = self.get_args(theory_func)
        num_args_exp_func = self.get_args(exp_func)

        if num_args_theory_func != num_args_exp_func:
            raise ValueError("Functions have different number of arguments")

        if arg_signatures is not None and len(arg_signatures) != num_args_theory_func:
            raise ValueError("Length of arg_signatures doesn't match the number of arguments")

        for _ in range(num_points):
            if arg_signatures is None:
                args = [np.array([random.uniform(-100, 100)]) for _ in range(num_args_theory_func)]
            else:
                args = [np.array([self.sample_arg(sig)]) for sig in arg_signatures]

            theory_result = theory_func(*args)
            theory_result_scalar = (
                theory_result.item() if isinstance(theory_result, np.ndarray) else theory_result
            )

            all_exp_result_scalar = list()
            for _ in range(repetitions):
                exp_result = exp_func(*args)
                all_exp_result_scalar.append(
                    exp_result.item() if isinstance(exp_result, np.ndarray) else exp_result
                )

            exp_mean = statistics.mean(all_exp_result_scalar)
            exp_stdev = statistics.stdev(all_exp_result_scalar)

            self.assertAlmostEqual(
                theory_result_scalar,
                exp_mean,
                delta=exp_stdev,
                msg=f"Functions outputs don't match at args = {args}: {theory_result_scalar} != {exp_stdev} +/- {exp_stdev}",
            )
