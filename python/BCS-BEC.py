get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root

def Integrand1(x, Δ, μ):
    return 1 - x**2 / np.sqrt((x**2-μ)**2+Δ**2) 

def Integrand2(x, Δ, μ):
    return x**2 * (1 - (x**2 - μ) / np.sqrt((x**2-μ)**2+Δ**2))

def Pair(x,inverse_kFa):
    Δ = x[0]
    μ = x[1]
    return quad(Integrand1, 0, np.inf, args=(Δ, μ))[0] - np.pi / 2 * inverse_kFa, quad(Integrand2, 0, 20, args=(Δ, μ))[0] - 2/3

def Solutions(inverse_kFa):
    return root(Pair,[0.01,1], args = (inverse_kFa))

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

inverse_kFa = np.linspace(-1.5,1.5,100)
plt.plot(inverse_kFa, [Solutions(inverse_kFa_val).x[0] for inverse_kFa_val in inverse_kFa])
plt.plot(inverse_kFa, [Solutions(inverse_kFa_val).x[1] for inverse_kFa_val in inverse_kFa])
plt.ylabel(r'$\Delta, \mu$', fontsize = 30)
plt.xlabel(r'$1/k_\text{F}a$', fontsize = 30, rotation = 'horizontal')
plt.show()





