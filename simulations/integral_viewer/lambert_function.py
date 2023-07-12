from scipy.special import lambertw
import numpy as np
import matplotlib.pyplot as plt


xs = np.linspace(-1, 10000 , 1000)

plt.figure(figsize=(10, 7.5))
plt.plot(xs, lambertw(np.exp(xs)), label="W(x)")

plt.xlabel("x")
plt.ylabel("W(x)")
plt.grid()
plt.show()