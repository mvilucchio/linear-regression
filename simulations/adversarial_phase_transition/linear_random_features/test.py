import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def elastic_net(x, lambda1, lambda2):
    """Elastic net function: lambda2 * x^2 + lambda1 * |x|"""
    return lambda2 * x**2 + lambda1 * np.abs(x)


def prox_elastic_net(v, V, lambda1, lambda2):
    """Proximal operator of the elastic net function"""
    if v > V * lambda1:
        return (v - V * lambda1) / (2 * lambda2 * V + 1)
    elif v < -V * lambda1:
        return (v + V * lambda1) / (2 * lambda2 * V + 1)
    else:
        return 0


def moreau_envelope(v, V, lambda1, lambda2):
    """Moreau envelope of the elastic net function"""
    # Use the proximal operator to get x*
    x_star = prox_elastic_net(v, V, lambda1, lambda2)

    # Compute the Moreau envelope directly from the definition
    return elastic_net(x_star, lambda1, lambda2) + (1 / (2 * V)) * (x_star - v) ** 2


def verify_moreau_envelope(v, V, lambda1, lambda2):
    """Verify Moreau envelope by direct minimization"""

    # Define the objective function to minimize
    def objective(x):
        return elastic_net(x, lambda1, lambda2) + (1 / (2 * V)) * (x - v) ** 2

    # Find minimum using minimize_scalar
    result = minimize_scalar(objective)
    min_value = result.fun
    min_x = result.x

    return min_value, min_x


def verify_optimality_condition(v, V, lambda1, lambda2):
    """Verify the optimality condition for the proximal operator"""
    prox = prox_elastic_net(v, V, lambda1, lambda2)
    subgradient = (v - prox) / V

    if prox > 0:
        expected = 2 * lambda2 * prox + lambda1
        return abs(subgradient - expected) < 1e-10, subgradient, expected
    elif prox < 0:
        expected = 2 * lambda2 * prox - lambda1
        return abs(subgradient - expected) < 1e-10, subgradient, expected
    else:  # prox = 0
        return -lambda1 <= subgradient <= lambda1, subgradient, f"in [{-lambda1}, {lambda1}]"


def main():
    # Parameters for testing
    lambda1 = 1.0
    lambda2 = 0.5
    V = 3.0

    # Test values
    v_values = np.linspace(-10, 10, 300)

    # Calculate proximal operator and Moreau envelope
    prox_values = [prox_elastic_net(v, V, lambda1, lambda2) for v in v_values]
    moreau_values = [moreau_envelope(v, V, lambda1, lambda2) for v in v_values]

    # Verify by direct minimization
    verified_results = [verify_moreau_envelope(v, V, lambda1, lambda2) for v in v_values]
    verified_moreau = [result[0] for result in verified_results]
    verified_prox = [result[1] for result in verified_results]

    # Create plots
    plt.figure(figsize=(12, 8))

    # Plot proximal operator
    plt.subplot(2, 1, 1)
    plt.plot(v_values, prox_values, "b-", label="Formula")
    plt.plot(v_values, verified_prox, "r--", label="Verified")
    plt.grid(True)
    plt.legend()
    plt.title("Proximal Operator of Elastic Net")
    plt.xlabel("v")
    plt.ylabel("prox_Vf(v)")

    # Plot Moreau envelope
    plt.subplot(2, 1, 2)
    plt.plot(v_values, moreau_values, "b-", label="Formula")
    plt.plot(v_values, verified_moreau, "r--", label="Verified")
    plt.grid(True)
    plt.legend()
    plt.title("Moreau Envelope of Elastic Net")
    plt.xlabel("v")
    plt.ylabel("M_Vf(v)")

    plt.tight_layout()

    # Print some example values for verification
    print("Testing with v = 3:")
    v_test = 3
    prox_test = prox_elastic_net(v_test, V, lambda1, lambda2)
    moreau_test = moreau_envelope(v_test, V, lambda1, lambda2)
    verified_moreau_test, verified_prox_test = verify_moreau_envelope(v_test, V, lambda1, lambda2)

    print(f"Proximal Operator (Formula): {prox_test}")
    print(f"Proximal Operator (Verified): {verified_prox_test}")
    print(f"Moreau Envelope (Formula): {moreau_test}")
    print(f"Moreau Envelope (Verified): {verified_moreau_test}")

    # Verify the optimality conditions
    print("\nVerifying optimality conditions:")
    for v in [-4, -2, 0, 2, 4]:
        valid, subgradient, expected = verify_optimality_condition(v, V, lambda1, lambda2)
        prox = prox_elastic_net(v, V, lambda1, lambda2)
        status = "VALID" if valid else "INVALID"
        print(
            f"v = {v}, prox = {prox}, subgradient = {subgradient}, expected = {expected}, status = {status}"
        )

    plt.show()


if __name__ == "__main__":
    main()
