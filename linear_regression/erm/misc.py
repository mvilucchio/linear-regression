from numba import njit


@njit
def weight_update(w, gradient, learning_rate):
    return w - learning_rate * gradient

