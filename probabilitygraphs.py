import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

def radial_wave_function(n, l, r):
    a0 = 5.29177e-11  # Bohr radius in meters

    if n < 1 or l < 0 or l >= n:
        raise ValueError("Invalid quantum numbers: n must be >= 1 and 0 <= l < n.")

    r_abs = np.abs(r)

    if n == 1 and l == 0:
        R = (1 / np.sqrt(np.pi * a0**3)) * np.exp(-r_abs / a0)
    elif n == 2 and l == 0:
        R = (1 / (4 * np.sqrt(2 * np.pi) * a0**(3/2))) * (2 - r_abs / a0) * np.exp(-r_abs / (2 * a0))
    elif n == 2 and l == 1:
        R = (1 / np.sqrt(24 * np.pi * a0**3)) * (r_abs / a0) * np.exp(-r_abs / (2 * a0))
    elif n == 3 and l == 0:
        R = (1 / (27 * np.sqrt(3 * np.pi) * a0**(3/2))) * (27 - 18 * (r_abs / a0) + 2 * (r_abs**2 / a0**2)) * np.exp(-r_abs / (3 * a0))
    elif n == 3 and l == 1:
        R = (1 / (27 * np.sqrt(6 * np.pi) * a0**(3/2))) * (r_abs / a0) * (3 - r_abs / (3 * a0)) * np.exp(-r_abs / (3 * a0))
    elif n == 3 and l == 2:
        R = (1 / (6 * np.sqrt(6 * np.pi) * a0**(3/2))) * (r_abs**2 / (2 * a0**2) - r_abs / (3 * a0)) * np.exp(-r_abs / (3 * a0))
    else:
        raise ValueError("Currently only supports (n=1, l=0), (n=2, l=0 or 1), and (n=3, l=0, 1, or 2) orbitals.")

    return R

def probability_density(n, l, r):
    R = radial_wave_function(n, l, r)
    P = (R**2) * r**2
    return P

def calculate_expectation_value(n, l):
    r_values = np.linspace(0, 25 * 5.29177e-11, 1000)
    P = probability_density(n, l, r_values)
    normalization_factor = simps(P, r_values)
    P_normalized = P / normalization_factor
    
    # Calculate expected value of r
    expected_value = simps(r_values * P_normalized, r_values)
    
    return expected_value

def calculate_most_probable_value(n, l):
    r_values = np.linspace(0, 25 * 5.29177e-11, 1000)
    P = probability_density(n, l, r_values)
    normalization_factor = simps(P, r_values)
    P_normalized = P / normalization_factor
    
    # Find the most probable value of r
    most_probable_index = np.argmax(P_normalized)
    most_probable_value = r_values[most_probable_index]
    
    return most_probable_value

def plot_probability_density(n, l):
    r = np.linspace(0, 25 * 5.29177e-11, 1000)  # Radius from 0 to 25 Bohr radii
    P = probability_density(n, l, r)

    normalization_factor = simps(P, r)
    P_normalized = P / normalization_factor  # Normalize the probability density

    expected_value = calculate_expectation_value(n, l)
    most_probable_value = calculate_most_probable_value (n, l)

    plt.figure(figsize=(10, 6))
    plt.plot(r, P_normalized, label=f"P(r) for n={n}, l={l}", color='red')
    plt.axvline(0, color='black', linestyle='--', label='r = 0')  # Mark the zero point
    plt.axvline(expected_value, color='blue', linestyle='--', label=f'Expected Value of r: {expected_value:.2e} m')
    plt.axvline(most_probable_value, color='green', linestyle='--', label=f'Most Probable Value of r: {most_probable_value:.2e} m')
    
    plt.title("Normalized Radial Probability Density P(r)")
    plt.xlabel("Radial Distance (m)")
    plt.ylabel("Normalized Probability Density P(r)")
    plt.xscale('linear')  # Use linear scale for better visibility in this range
    plt.yscale('linear')  # Use linear scale for better visibility in this range
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    for n in range(1, 4):  # Loop over n values (1 to 3)
        for l in range(n):  # Loop over l values (0 to n-1)
            plot_probability_density(n, l)

if __name__ == "__main__":
    main()