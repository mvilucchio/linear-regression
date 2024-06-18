import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from scipy.optimize import curve_fit

rng = np.random.default_rng()


def linear_fit(log_x, a, b):
    return a * log_x + b


alpha = 0.2
reps = 20

ds = np.logspace(1, 4, 5)
min_val_mean = np.empty_like(ds)
min_val_std = np.empty_like(ds)

for i, d in enumerate(tqdm(ds)):
    min_val_list = []
    n = int(alpha * d)
    d = int(d)

    for _ in tqdm(range(reps), leave=False):
        wstar = rng.standard_normal((d,), dtype=np.float32)
        xs = rng.standard_normal((n, d), dtype=np.float32)

        m = np.min(np.abs(xs @ wstar) / np.linalg.norm(wstar, ord=2))

        min_val_list.append(m)

        del wstar
        del xs

    min_val_mean[i] = np.mean(min_val_list)
    min_val_std[i] = np.std(min_val_list)

# fitting the data
log_x = np.log10(ds)
log_y = np.log10(min_val_mean)
log_y_err = min_val_std / (min_val_mean * np.log(10))

popt, pcov = curve_fit(linear_fit, log_x, log_y, sigma=log_y_err, absolute_sigma=True)
a, b = popt
a_err, b_err = np.sqrt(np.diag(pcov))

x_fit = np.linspace(min(log_x), max(log_x), 100)
y_fit = linear_fit(x_fit, a, b)

# plotting
plt.plot(10**x_fit, 10**y_fit, "--", label="Fit: b x^a ")
plt.text(0.1, 0.1, "a = {:.3f} +/- {:.3f}".format(a, a_err), transform=plt.gca().transAxes)
plt.text(0.1, 0.2, "b = {:.3f} +/- {:.3f}".format(b, b_err), transform=plt.gca().transAxes)
plt.errorbar(ds, min_val_mean, yerr=min_val_std, fmt=".-")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("d")
plt.ylabel(r"$| \langle w, x \rangle | / \|w\|_2$")
plt.title(r"$\alpha$ = {:.3f} reps = {:d}".format(alpha, reps))
plt.legend()
plt.grid(True)

plt.show()
