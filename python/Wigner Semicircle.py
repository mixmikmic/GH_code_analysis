#NAME: Wigner Semicircle
#DESCRIPTION: Investigating the distribution of eigenvalues for real symmetric matrices.

# coding: utf-8
get_ipython().magic('matplotlib notebook')
import numpy as np
import math
import matplotlib.pyplot as plt
from __future__ import print_function, division
from scipy.optimize import *
from scipy.ndimage.filters import *

num_matrices = 50
matrix_dimensions = 500
eigenvalues = []

for __ in range(0, num_matrices):
    matrix = np.random.rand(matrix_dimensions, matrix_dimensions)
    sym_matrix = matrix + matrix.T
    eigenvalues.extend(np.linalg.eig(sym_matrix)[0])

wigner = (lambda x, R: (2/(math.pi * R**2)) * np.sqrt(R**2 - x**2))
y, x, __ = plt.hist(eigenvalues, 1000, normed = True, facecolor='white', label = "Eigenvalues of symmetric matrix")
x = uniform_filter1d(x, size = 2)
x = np.delete(x, 0)

y_new = y[np.where(y != 0.0)]
x = x[np.where(y != 0.0)]

y_new = y_new[np.where(x <= matrix_dimensions/2)]
x_new = x[np.where(x <= matrix_dimensions/2)]

ax = plt.subplot()
params, __ = curve_fit(wigner, x_new, y_new, p0 = [100.])
wigner_plot = plt.plot(x, wigner(x, params[0]), color = 'red', label = "Wigner Semicircle distribution")
ax.set_xlim([-abs(params[0]), abs(params[0])])
# Comment the above line to see not only the Wigner Semicircle part of the distribution, but also the peak at the higher value.
plt.ylabel('Probability density')
plt.xlabel('Eigenvalues')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=1, fancybox=True, shadow=True)
print("R = " + str(abs(params[0])))
plt.show()





