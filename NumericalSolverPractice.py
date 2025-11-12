import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from scipy.special import jv
import matplotlib.pyplot as plt
import time

# Parameters
L = 10.0           # Domain length [0, L]
N = 400            # Reduced number of points for stability
n = 0              # Bessel function order
order = 2          # Finite difference order

# Grid setup
x = np.linspace(0, L, N)
h = x[1] - x[0]    # Step size

def build_matrix(x, h, n, order):
    interior_points = x[1:-1]
    size = len(interior_points)
    
    if order == 2:
        # 2nd order finite differences
        main_diag = -2/h**2 + (1 - n**2/interior_points**2)
        lower_diag = [1/h**2 - 1/(2*h*x_i) for x_i in interior_points[1:]]
        upper_diag = [1/h**2 + 1/(2*h*x_i) for x_i in interior_points[:-1]]
        diagonals = [lower_diag, main_diag, upper_diag]
        offsets = [-1, 0, 1]
    
    A = diags(diagonals, offsets, shape=(size, size), format='csr')
    A_full = eye(N, format='csr')
    A_full[1:-1, 1:-1] = A
    
    # Boundary conditions
    A_full[0,0] = 1
    A_full[-1,-1] = 1
    
    return A_full

# Build and solve
A = build_matrix(x, h, n, order)
b = np.zeros(N)
b[0] = 1.0 if n == 0 else 0.0  # Jₙ(0)
b[-1] = 0.0                     # Jₙ(L) = 0

# Solve system
y = spsolve(A, b)
y_exact = jv(n, x)

# Normalize and plot
if n == 0:
    y /= y[0]

plt.figure(figsize=(10,6))
plt.plot(x, y, 'b-', linewidth=2, label='FDM Solution')
plt.plot(x, y_exact, 'r--', linewidth=1, label='Exact Solution')
plt.xlabel('x')
plt.ylabel(f'J_{n}(x)')
plt.title('Bessel Function Solution')
plt.legend()
plt.grid(True)
plt.show()
