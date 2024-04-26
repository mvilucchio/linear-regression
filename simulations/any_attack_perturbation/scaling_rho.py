import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np

rng = np.random.default_rng()


alpha = 0.2
reps = 10

ds = np.logspace(1, 5, 5)
min_val_mean = np.empty_like(ds)
min_val_std = np.empty_like(ds)

for i, d in enumerate(tqdm(ds)):
    min_val_list = []
    n = int(0.8 * d)
    d = int(d)

    for _ in tqdm(range(reps), leave=False):
        # wstar = np.random.normal(0, 1., (d,),)
        # xs = np.random.normal(0, 1., (n, d))

        wstar = rng.standard_normal((d,), dtype=np.float32)
        xs = rng.standard_normal((n, d), dtype=np.float32)

        m = np.min(np.abs(xs @ wstar) / np.linalg.norm(wstar, ord=2))

        min_val_list.append(m)

        del wstar
        del xs

    min_val_mean[i] = np.mean(min_val_list)
    min_val_std[i] = np.std(min_val_list)

plt.errorbar(ds, min_val_mean, yerr=min_val_std, fmt=".-")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("d")
plt.ylabel("Min Margin of Random Gaussians")
plt.title("$\\alpha$ = {:.3f}".format(alpha))
plt.grid(True)

plt.show()
