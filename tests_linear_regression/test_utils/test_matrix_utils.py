from unittest import TestCase, main
import linear_regression.utils.matrix_utils as mu
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal


class TestAxis0PosNegMask(TestCase):
    def setUp(self):
        self.arr_valid = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        self.mask_valid = np.array([True, False, True, True, True, False])
        self.mask_length_valid = 4
        self.expected_out_pos = np.array([[1, 2], [5, 6], [7, 8], [9, 10]])
        self.expected_out_neg = np.array([[3, 4], [11, 12]])

    def test_valid_input(self):
        out_pos, out_neg = mu.axis0_pos_neg_mask(
            self.arr_valid, self.mask_valid, self.mask_length_valid
        )
        self.assertEqual(out_pos.shape, self.expected_out_pos.shape)
        self.assertEqual(out_neg.shape, self.expected_out_neg.shape)
        assert_array_equal(out_pos, self.expected_out_pos)
        assert_array_equal(out_neg, self.expected_out_neg)

    def test_invalid_input(self):
        # Test that assertion is raised when self.arr_valid is not self.a 2D array
        with self.assertRaises(AssertionError):
            mu.axis0_pos_neg_mask([1, 2, 3], self.mask_valid, self.mask_length_valid)

        # Test that assertion is raised when self.mask_valid is not self.a 1D array or does not have the same shape as self.arr_valid's first axis
        with self.assertRaises(AssertionError):
            mu.axis0_pos_neg_mask(
                self.arr_valid, np.array([[True, False], [False, True]]), self.mask_length_valid
            )

        # Test that assertion is raised when self.mask_length_valid is negative or greater than self.mask_valid's sum
        with self.assertRaises(AssertionError):
            mu.axis0_pos_neg_mask(self.arr_valid, self.mask_valid, -1)

        with self.assertRaises(AssertionError):
            mu.axis0_pos_neg_mask(self.arr_valid, self.mask_valid, 9)

    def test_output_shape(self):
        out_pos, out_neg = mu.axis0_pos_neg_mask(
            self.arr_valid, self.mask_valid, self.mask_length_valid
        )

        expected_shape_pos = (2, 2)
        expected_shape_neg = (1, 2)
        self.assertEqual(out_pos.shape, expected_shape_pos)
        self.assertEqual(out_neg.shape, expected_shape_neg)

    def test_output_dtype(self):
        out_pos, out_neg = mu.axis0_pos_neg_mask(
            self.arr_valid, self.mask_valid, self.mask_length_valid
        )

        self.assertEqual(out_pos.dtype, self.arr_valid.dtype)
        self.assertEqual(out_neg.dtype, self.arr_valid.dtype)


class TestSafeSparseDot(TestCase):
    def setUp(self):
        self.a = np.array([[1, 2], [3, 4], [5, 6]])
        self.b = np.array([[1, 2, 3], [4, 5, 6]])

    def test_valid_input(self):
        # Test dot product of 2D arrays
        out = mu.safe_sparse_dot(self.a, self.b.T)
        expected_out = np.array([[7, 16], [19, 46], [31, 76]])
        assert_array_almost_equal(out, expected_out)

        # Test matrix multiplication of 2D arrays
        out = mu.safe_sparse_dot(self.a.T, self.b)
        expected_out = np.array([[22, 28, 34], [28, 37, 46]])
        assert_array_almost_equal(out, expected_out)

        # Test dot product of 1D arrays
        out = mu.safe_sparse_dot(self.a[0], self.b[1])
        expected_out = 14
        self.assertEqual(out, expected_out)

    def test_invalid_input(self):
        # Test that assertion is raised when both self.a and self.b have more than 2 dimensions
        with self.assertRaises(AssertionError):
            mu.safe_sparse_dot(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))

    def test_output_shape(self):
        self.a, self.b = self.get_test_data()

        # Test dot product of 2D arrays
        out = mu.safe_sparse_dot(self.a, self.b.T)
        expected_shape = (3, 2)
        self.assertEqual(out.shape, expected_shape)

        # Test matrix multiplication of 2D arrays
        out = mu.safe_sparse_dot(self.a.T, self.b)
        expected_shape = (2, 3)
        self.assertEqual(out.shape, expected_shape)

        # Test dot product of 1D arrays
        out = mu.safe_sparse_dot(self.a[0], self.b[1])
        expected_shape = ()
        self.assertEqual(out.shape, expected_shape)

    def test_output_dtype(self):
        self.a, self.b = self.get_test_data()

        out = mu.safe_sparse_dot(self.a, self.b.T)
        self.assertEqual(out.dtype, self.a.dtype)

        out = mu.safe_sparse_dot(self.a.T, self.b)
        self.assertEqual(out.dtype, self.a.dtype)

        out = mu.safe_sparse_dot(self.a[0], self.b[1])
        self.assertEqual(out.dtype, self.a.dtype)


if __name__ == "__main__":
    main()
