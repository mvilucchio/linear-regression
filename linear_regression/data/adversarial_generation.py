from autograd import grad, elementwise_grad
import matplotlib.pyplot as plt 
import autograd.numpy as np

def func(x):
    return np.sin(x**2)

if __name__ == "__main__":

    # Create a function that computes the gradient of func
    grad_func = elementwise_grad(func)

    xs = np.linspace(-3, 5, 1000)

    # Compute the gradient at each point
    gradients = grad_func(xs)
    func_values = func(xs)

    plt.plot(xs, func_values, label="f(x) = x^2")
    plt.plot(xs, gradients, label="f'(x) = 2x")

    plt.legend()
    plt.grid()
    plt.show()