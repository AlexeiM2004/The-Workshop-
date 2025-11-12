import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import simps
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

def plot_probability_density(n, l, canvas):
    r = np.linspace(0, 25 * 5.29177e-11, 1000)  # Radius from 0 to 25 Bohr radii
    P = probability_density(n, l, r)

    normalization_factor = simps(P, r)
    P_normalized = P / normalization_factor  # Normalize the probability density

    # Clear the previous plot
    plt.clf()
    plt.plot(r, P_normalized, label=f"P(r) for n={n}, l={l}", color='red')
    plt.axvline(0, color='black', linestyle='--', label='r = 0')  # Mark the zero point
    plt.title("Normalized Radial Probability Density P(r)")
    plt.xlabel("Radial Distance (m)")
    plt.ylabel("Normalized Probability Density P(r)")
    plt.xscale('linear')
    plt.yscale('linear')
    plt.grid(True)
    plt.legend()

    # Draw the new plot on the canvas
    canvas.draw()

def on_submit(canvas):
    try:
        n = int(entry_n.get())
        l = int(entry_l.get())
        
        if n < 1 or l < 0 or l >= n:
            raise ValueError("Invalid quantum numbers.")
        if n > 3:
            messagebox.showwarning("Invalid Input", "Currently, this program does not support orbitals with n > 3.")
            return
        
        plot_probability_density(n, l, canvas)
    except ValueError as e:
        messagebox.showerror("Input Error", str(e))

# Create the main Tkinter window
root = tk.Tk()
root.title("Radial Probability Density Plotter")

# Create and place labels and entry fields
label_n = tk.Label(root, text="Enter principal quantum number (n, must be >= 1):")
label_n.pack()

entry_n = tk.Entry(root)
entry_n.pack()

label_l = tk.Label(root, text="Enter azimuthal quantum number (l, 0 for s, 1 for p, 2 for d):")
label_l.pack()

entry_l = tk.Entry(root)
entry_l.pack()

# Create a Matplotlib figure and axis
fig, ax = plt.subplots(figsize=(6, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

# Create and place the submit button
submit_button = tk.Button(root, text="Submit", command=lambda: on_submit(canvas))
submit_button.pack()

# Start the Tkinter event loop
root.mainloop()