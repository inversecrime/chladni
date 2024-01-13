import matplotlib.pyplot as plt
import numpy as np

import chladni

grid_size = 30
mu = 0.2
n_patterns = 10

(eigenvalues, eigenfunctions) = chladni.calculate_patterns(grid_size, mu, n_patterns)

for (eigenvalue, eigenfunction) in zip(eigenvalues, eigenfunctions):
    plt.figure()
    plt.title(f"Eigenvalue = {eigenvalue}")
    plt.contour(np.reshape(eigenfunction, newshape=(grid_size, grid_size)), (-1e-8, +1e-8))
    plt.colorbar()

plt.show()
