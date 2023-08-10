from numpy import sqrt,linspace
from qutip import *
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

a = destroy(20)

n = a.dag()*a

psi = coherent(20,2)

psi.dag()*n*psi

psi_dm = ket2dm(psi)

plot_fock_distribution(psi_dm)

plot_fock_distribution(coherent_dm(40,4))

plot_fock_distribution(fock_dm(20,2))

plot_fock_distribution(thermal_dm(20,2))

xvec = linspace(-10,10,200) # Create an array for the phase space coordinates.
plt.contourf(xvec, xvec, wigner(coherent_dm(20,2),xvec,xvec),20) # contour plot of Wigner function



