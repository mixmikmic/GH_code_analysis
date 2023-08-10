import matplotlib.pyplot as plt
from numpy import sqrt,cos,sin,arange,pi
from qutip import *
get_ipython().magic('matplotlib inline')

H = Qobj([[1],[0]])
V = Qobj([[0],[1]])
P45 = Qobj([[1/sqrt(2)],[1/sqrt(2)]])
M45 = Qobj([[1/sqrt(2)],[-1/sqrt(2)]])
R = Qobj([[1/sqrt(2)],[-1j/sqrt(2)]])
L = Qobj([[1/sqrt(2)],[1j/sqrt(2)]])

H

Ph = H*H.dag()
Ph

Pv = V*V.dag()
Pv

Ph + Pv

P45*P45.dag() + M45*M45.dag()

L*L.dag() + R*R.dag()

def Rp(theta):
    return Qobj([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]]).tidyup()

Rp(pi/2)

V == Rp(pi/2)*H

# Solution Goes Here

# Solution Goes Here

def sim_transform(o_basis1, o_basis2, n_basis1, n_basis2):
    a = n_basis1.dag()*o_basis1
    b = n_basis1.dag()*o_basis2
    c = n_basis2.dag()*o_basis1
    d = n_basis2.dag()*o_basis2
    return Qobj([[a.data[0,0],b.data[0,0]],[c.data[0,0],d.data[0,0]]])

Shv45 = sim_transform(H,V,P45,M45)  # as found in Example 4.A.1, Eq. 4.A.10.
Shv45

Shv45 * H  # compare to Eq. 4.A.12







