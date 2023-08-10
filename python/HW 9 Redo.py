from qutip import *
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

Γ1 = 0.5
Γ2 = 0.25
α = 0.1 + 0.2j
H = Qobj([[Γ1, α, 0], [np.conj(α), Γ1, 0], [0, 0, Γ2]])
H

# Make sure this is Hermitian:
H.isherm

energies, [E1, E2, E3] = H.eigenstates()

energies

print(Γ2)
print(Γ1 + np.abs(α))
print(Γ1 - np.abs(α))

E1

E2

E3

a = Qobj([[1],[0],[0]])
a

a.dag()*E1

a.dag()*E2

a.dag()*E3

b = Qobj([[0],[1],[0]])

b.dag()*E1

b.dag()*E2

b.dag()*E3

N = 100
T = 2*np.pi/np.abs(α)   # period of oscillation
tlist = np.linspace(0,T,N)
proj_b = b.dag()*b
solution = sesolve(H, a, tlist, proj_b)

plt.plot(tlist/T,solution.expect[0],".")



