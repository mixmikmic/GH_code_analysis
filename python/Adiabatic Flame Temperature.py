import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

n = dict()
n['CO2'] = 1
n['H2O'] = 2
n['N2'] = 3*(.79/0.21)

for k in n.keys(): 
    print("n[{:3s}] =  {:5.2f}".format(k,n[k]))

DeltaH_Rxn = -890400

Cp = dict()
Cp['CO2','g'] = lambda T: 36.11 + 4.233e-2*T - 2.887e-5*T**2 + 7.464e-9*T**3
Cp['H2O','g'] = lambda T: 33.46 + 0.688e-2*T + 0.7604e-5*T**2 - 3.593e-9*T**3
Cp['N2','g']  = lambda T: 29.00 + 0.2199e-2*T + 0.5723e-5*T**2 - 2.871e-9*T**3

from scipy import integrate

def DeltaH_Prod(T):
    h1,err = integrate.quad(Cp['CO2','g'],25,T)
    h2,err = integrate.quad(Cp['H2O','g'],25,T)
    h3,err = integrate.quad(Cp['N2','g'],25,T)
    return n['CO2']*h1 + n['H2O']*h2 + n['N2']*h3

f = lambda T: DeltaH_Rxn + DeltaH_Prod(T)

T = np.linspace(25,3000,200)
plt.plot(T,[f(T) for T in T],[25,3000],[0,0])
plt.xlabel('Temperature [degrees C]')
plt.ylabel('Net Enthalpy [Joules/gmol]');

from scipy.optimize import brentq

T = brentq(f,25,3000)
print('The adiabatic flame temperature is {:6.1f} degrees C.'.format(T))



