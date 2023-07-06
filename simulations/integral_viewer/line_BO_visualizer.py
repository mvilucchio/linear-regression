import numpy as np
from math import sqrt
from scipy.integrate import dblquad, quad
import matplotlib.pyplot as plt
from linear_regression.fixed_point_equations.classification.BO import q_int_BO_no_noise_classif

BIG_NUMBER = 5


m = 0.41562244834593775
q = 0.6
sigma = 40.5354931610250615

m = 0.88787 
q = 0.88787 
sigma = 0.11213


domains = [(1, [-BIG_NUMBER, BIG_NUMBER]), (-1, [-BIG_NUMBER, BIG_NUMBER])]

plt.figure(figsize=(10, 10))
for y, d in domains:
    print(y, d)
    xs = np.linspace(d[0], d[1], 1000)

    if y == 1:
        plt.axvline(d[0], color="k", linestyle="--")
        plt.axvline(d[1], color="k", linestyle="--")
        plt.plot(xs, [q_int_BO_no_noise_classif(x, y, q, m, sigma) for x in xs], "g", label="q")

    print("y = ", y)
    print("integral value = ", quad(q_int_BO_no_noise_classif, d[0], d[1], args=(y, q, m, sigma))[0])

plt.xlabel(r"$\xi$")
plt.legend()
plt.grid()

plt.show()
