get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

M = 100
c = 1

ρ0 = np.ones(M) / (2 * np.pi) # Initial guess

a = np.linspace(-1, 1, M)
b = np.linspace(-1, 1, M)
A, B = np.meshgrid(a, b)

K = - (2 / M) * 2 * c / ((A - B)**2 + c**2) # Kernel in the integral equation

def Bethe(ρ):
    return 2 * np.pi * ρ + np.dot(K,ρ) - 1

# Now find the solution
ρ_out = root(Bethe, ρ0).x 

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

plt.plot(a, ρ_out)
plt.xlabel(r'$k/q$', fontsize = 30)
plt.ylabel(r'$\rho(k)$', fontsize = 30, rotation = 'horizontal')
plt.axes().set_aspect(30)
plt.show()





