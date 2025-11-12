import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox

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
    normalization_factor = np.trapz(P, r_values)  # Use trapezoidal rule for normalization
    P_normalized = P / normalization_factor
    
    # Calculate expected value of r
    expected_value = np.trapz(r_values * P_normalized, r_values)  # Use trapezoidal rule for expected value
    
    return expected_value

def calculate_most_probable_value(n, l):
    r_values = np.linspace(0, 25 * 5.29177e-11, 1000)
    P = probability_density(n, l, r_values)
    normalization_factor = np.trapz(P, r_values)  # Use trapezoidal rule for normalization
    P_normalized = P / normalization_factor
    
    # Find the most probable value of r
    most_probable_index = np.argmax(P_normalized)
    most_probable_value = r_values[most_probable_index]
    
    return most_probable_value

def plot_probability_density(n, l, canvas):
    r = np.linspace(0, 25 * 5.29177e-11, 1000)  # Radius from 0 to 25 Bohr radii
    P = probability_density(n, l, r)

    normalization_factor = np.trapz(P, r)  # Use trapezoidal rule for normalization
    P_normalized = P / normalization_factor  # Normalize the probability density

    expected_value = calculate_expectation_value(n, l)
    most_probable_value = calculate_most_probable_value(n, l)

    # Clear the previous plot
    plt .clf()  
    plt.plot(r * 1e10, P_normalized)  # Convert to nanometers
    plt.title('Radial Probability Density')
    plt.xlabel('Radius (nm)')
    plt.ylabel('Probability Density')
    plt.axvline(expected_value * 1e10, color='r', linestyle='--', label='Expected Value')
    plt.axvline(most_probable_value * 1e10, color='g', linestyle='--', label='Most Probable Value')
    plt.legend()
    plt.grid()

    canvas.draw()  # Refresh the canvas to show the new plot

def main():
    root = tk.Tk()
    root.title("Radial Probability Density Plot")

    # Create a Matplotlib figure
    fig = plt.Figure(figsize=(6, 4), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Example parameters for plotting
    n = 2  # Principal quantum number
    l = 1  # Azimuthal quantum number

    # Plot the graph
    plot_probability_density(n, l, canvas)

    # Start the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    main()