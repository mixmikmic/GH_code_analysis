get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaincc, gamma

N = 100
x = np.linspace(0,20,100) 
rho = gammaincc(N, x**2/2) / (2 * np.pi)

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

plt.plot(x, rho)
plt.xlabel(r'$|z|$', fontsize = 30)
plt.ylabel(r'$\rho$', fontsize = 30, rotation = 'horizontal')
plt.axes().set_aspect(80)

