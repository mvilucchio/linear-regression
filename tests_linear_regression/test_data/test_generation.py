from unittest import TestCase, main
from numpy.random import normal
from numpy import mean
import linear_regression.data.generation as dg

data_generation_functions = [
    dg.measure_gen_no_noise_clasif,
    dg.measure_gen_probit_clasif,
    dg.measure_gen_single_noise_clasif,
    dg.measure_gen_single,
    dg.measure_gen_double,
    dg.measure_gen_decorrelated
]

template_args_functions = [
    (0.1,),
    (0.1,),
    (0.1,),
    (0.1,),
    (0.1, 0.2, 0.3),
    (0.1, 0.2, 0.3, 0.1),
]

class TestGeneralAllFunctions(TestCase):
    def test_out_dimensions(self):
        d = 100
        n = 200
        teacher_vector = normal(0.0, 1.0, d)
        for f, args in zip(data_generation_functions, template_args_functions):
            self.assertEqual(f(True, teacher_vector, normal(0.0, 1.0, (n, d)), *args).shape, (n,))
            self.assertEqual(f(False, teacher_vector, normal(0.0, 1.0, (n, d)), *args).shape, (n,))
    
    def test_mean(self):
        self.assertEqual(0.0,0.0)


class TestDataGeneration(TestCase):
    def test_out_dimensions(self):
        return

if __name__ == '__main__':
    main()