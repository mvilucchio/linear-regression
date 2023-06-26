import linear_regression.regression_numerics.data_generation as dg
import linear_regression.regression_numerics.erm_solvers as erm
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def main():
    delta_in, delta_out, percentage, beta, a = 1.0, 5.0, 0.1, 0.0, 1.0
    d = 1000
    alphas = [10.0, 30.0, 50.0, 100.0]

    for alpha in alphas:
        reg_params = np.linspace(1.0, 10.0, 40)
        # reg_params = np.logspace(0, 1, 50)
        mse_errors = np.zeros_like(reg_params)

        xs, ys, _, _, _ = dg.data_generation(
            dg.measure_gen_decorrelated, d, max(int(np.around(d * alpha)), 1), 1, (delta_in, delta_out, percentage, beta)
        )

        xs_train, ys_train, _, _, _ = dg.data_generation(
            dg.measure_gen_decorrelated, d, 100, 1, (delta_in, delta_out, percentage, beta)
        )

        # this part shifts and normalises the data
        # xs = xs - np.mean(xs, axis=0)
        # xs = xs / np.sqrt(np.sum(np.square(xs), axis=0))

        # xs_train = xs_train - np.mean(xs_train, axis=0)
        # xs_train = xs_train / np.sqrt(np.sum(np.square(xs_train), axis=0))

        for idx, rp in enumerate(tqdm(reg_params, desc=f"alpha={alpha}")):
            w_hat = erm.find_coefficients_Huber(ys, xs, rp, a)
            mse_errors[idx] = 0.5 * np.mean(np.square(ys_train - (xs_train @ w_hat) / np.sqrt(d)))

        plt.plot(reg_params, mse_errors, marker='.', label="$\\alpha$ = {:.2f}".format(alpha))
        min_idx = np.argmin(mse_errors)
        plt.scatter(reg_params[min_idx], mse_errors[min_idx], marker='x', color='red')
        
    plt.legend()
    plt.xlabel("$\\lambda$")
    plt.ylabel("MSE Error")
    # plt.xscale("log")
    plt.show()

if __name__ == "__main__":
    main()
