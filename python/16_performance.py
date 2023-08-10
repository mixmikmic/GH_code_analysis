# Plotting code for each of the methods above
import os

import numpy
import matplotlib.pyplot as plt

u_true = lambda x: (4.0 - numpy.exp(1.0)) * x - 1.0 + numpy.exp(x)
x_fine = numpy.linspace(0.0, 1.0, 100)

base_path = "./src"
file_names = ["jacobi_omp1.txt", "jacobi_omp2.txt", "jacobi_mpi.txt"]
titles = ["OpenMP - Fine-Grained", "OpenMP - Coarse-Grained", "MPI"]
fig = plt.figure()
fig.set_figwidth(fig.get_figwidth() * 3)
for (i, title) in enumerate(titles):
    path = os.path.join(base_path, file_names[i])
    data = numpy.loadtxt(path)
    x = data[:, 0]
    U = data[:, 1]
    
    axes = fig.add_subplot(1, 3, i + 1)
    axes.plot(x, U, 'ro', label='computed')
    axes.plot(x_fine, u_true(x_fine), 'k', label="exact")
    axes.set_title(title)
    axes.set_xlabel('x')
    axes.set_ylabel('u(x)')

plt.show()

