from unittest import TestCase, main
import linear_regression.erm.aux_functions as af
from numpy import array
from numpy.random import randn
from jax.nn import sigmoid
from jax import grad


class TestSigmoid(TestCase):
    def setUp(self):
        self.x = array([-1.0, -0.5, 0.0, 0.5, 1.0])
        self.F = array([[-1.0, -0.5, 0.0, 0.5, 1.0], [-3.0, -1.5, 0.0, 1.5, 3.0]])

    def test_output(self):
        self.assertEqual(af.sigmoid(0), 0.5)
        self.assertAlmostEqual(af.sigmoid(100), 1.0)
        self.assertAlmostEqual(af.sigmoid(-100), 0.0)

        rand_vals = randn(30)
        for a in rand_vals:
            self.assertAlmostEqual(af.sigmoid(a), sigmoid(a), places=6)

    def test_output_shape(self):
        self.assertEqual(af.sigmoid(0).shape, ())
        self.assertEqual(af.sigmoid(self.x).shape, (5,))
        self.assertEqual(af.sigmoid(self.F).shape, (2, 5))


class TestDSigmoid(TestCase):
    def setUp(self):
        self.x = array([-1.0, -0.5, 0.0, 0.5, 1.0])
        self.F = array([[-1.0, -0.5, 0.0, 0.5, 1.0], [-3.0, -1.5, 0.0, 1.5, 3.0]])

    def test_output(self):
        self.assertEqual(af.D_sigmoid(0), 0.25)
        self.assertAlmostEqual(af.D_sigmoid(100), 0.0)
        self.assertAlmostEqual(af.D_sigmoid(-100), 0.0)

        rand_vals = randn(30)
        jax_autograd = grad(sigmoid)
        for a in rand_vals:
            self.assertAlmostEqual(af.D_sigmoid(a), jax_autograd(a), places=6)

    def test_output_shape(self):
        self.assertEqual(af.D_sigmoid(0).shape, ())
        self.assertEqual(af.D_sigmoid(self.x).shape, (5,))
        self.assertEqual(af.D_sigmoid(self.F).shape, (2, 5))


if __name__ == "__main__":
    main()
