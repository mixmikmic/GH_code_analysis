import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')

gas1 = ct.Solution('gri30.xml')

gas1.TP = 300, 101325   # [K], [Pa]
phi = 1
gas1.set_equivalence_ratio(phi,'CH4','O2:1,N2:3.76')
gas1()

gas1.equilibrate('HP')
print "The adiabatic Flame Temperature {0:.2f} [K]".format(gas1.T)

gas1 = ct.Solution('gri30.xml')
phis = np.linspace(0,4,1000)
T_adiabatic = np.zeros_like(phis)
for l, phi in enumerate(phis):
    gas1.TP = 300, 101325
    gas1.set_equivalence_ratio(phi,'CH4','O2:2,N2:8')
    gas1.equilibrate('HP')
    T_adiabatic[l] = gas1.T

fig, ax = plt.subplots()
ax.set_title("Adiabatic Flame Temperature of $CH_4$/Air Mixtures")
ax.plot(phis,T_adiabatic)
ax.set_xlabel('$\phi$, Equivalence Ratio [-]')
ax.set_ylabel('$T_{adiabatic}$ [K]')
plt.show()



