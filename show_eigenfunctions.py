from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

import chladni

# TODO error if it takes more than 10sec
grid_size = 50
ratio = 1
mu = 0.2
n_patterns = 25

(eigenvalues, eigenfunctions) = chladni.calculate_patterns(grid_size, mu, n_patterns)

for index in range(n_patterns):
    plt.subplot(round(sqrt(n_patterns)) + 1, round(sqrt(n_patterns)) + 1, index + 1)
    plt.title(f"Eigenfrequenz = {ratio * sqrt(eigenvalues[index]):.2f}")
    plt.contour(np.reshape(eigenfunctions[index], shape=(grid_size, grid_size)), (-1e-10, +1e-10))
    # plt.imshow(np.reshape(eigenfunctions[index], shape=(grid_size, grid_size)))

plt.show()
