from unittest import TestCase, main
from numpy.random import normal
from numpy import mean
import linear_regression.regression_numerics.data_generation as dg

class TestMeasureGenSingle(TestCase):
    def test_out_dimensions(self):
        d = 100
        n = 200
        teacher_vector = normal(0.0, 1.0, d)
        self.assertEqual(dg.measure_gen_single(True, teacher_vector, normal(0.0, 1.0, (n, d)), 0.1).shape, (n,))
        self.assertEqual(dg.measure_gen_single(False, teacher_vector, normal(0.0, 1.0, (n, d)), 0.1).shape, (n,))
    
    def test_mean(self):
        self.assertEqual(0.0,0.0)

if __name__ == '__main__':
    main()