# Python dependencies
# from __future__ import division, print_function
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.constants import codata

# change the default font set (matplotlib.rc)
mpl.rc('mathtext', fontset='stixsans', default='regular')

# increase text size somewhat
mpl.rcParams.update({'axes.labelsize':12, 'font.size': 12})

# set up notebook for inline plotting
get_ipython().magic('matplotlib inline')

# get constants from CODATA 2010
F = codata.physical_constants['Faraday constant'][0]
R = codata.physical_constants['molar gas constant'][0]
EPSILON_0 = codata.physical_constants['electric constant'][0]
# check the constants
F,R,EPSILON_0

def cap_helmholtz(epsilon_H=6.8, x=0.2e-9, A=1e-4):
    """
    *input*
    epsilon_H - the relative permittivity of the Helmholtz layer, default is 6.8
    x, the thickness of the Helmholtz layer [m], default is 0.2 nm, i.e. 0.2e-9 cm.
    A, electrode area [m²], default is 1 cm², i.e. 1e-4
    *output*
    C,  [F]
    """
    C_H = (epsilon_H*EPSILON_0*A)/x
    return C_H

#For a Helmholtz layer thickness of 0.2 nm (roughly half the diameter of a hydrated cation), and a relative permittivity of 5:
print(round(cap_helmholtz(5,0.2e-9,100e-6),7), 'F')

def cap_gouychapman(E,Epz=0,c=10,T=298.15,epsilon=78.54,A=1e-4):
    """
    *input*
    E, electrode potential [V]
    Epz, potential of zero charge [V], default is 0 V
    c [mol/m³], default is 10 mM, i.e. 10e-3 mol/dm³ = 10 mol/m³ 
    T [K], default is 298.15 K
    epsilon [dimensionless]
    A, the electrode area in m²    
    *output*
    C_GC_spec, the capacitance (C_GC) of  
    the diffuse double layer as predicted by the Gouy-Chapman model for  
    a 1:1 electrolyte with monovalent ions.
    """
    C_GC = A*np.sqrt((2*(F**2)*epsilon*EPSILON_0*c)/(R*T))*np.cosh(F*(E-Epz)/(2*R*T))
    return C_GC

def cap_stern(E,Epz=0,epsilon_H=6.8,x=0.2e-9,c=10,T=298.15,epsilon=78.54,A=1e-4):
    """Returns the total double-layer capacitance as predicted by the
       Gouy-Chapman-Stern model"""
    C_h = cap_helmholtz(epsilon_H=epsilon_H,x=x,A=A)
    C_gc = cap_gouychapman(E,Epz=Epz,c=c,T=T,epsilon=epsilon,A=A)
    recip_C_S = 1/C_h + 1/C_gc
    return 1/recip_C_S

#generate a range of potentials
E = np.linspace(-0.2,0.2, num=401)
E[:5] #check the first five values in the array

#now plot the component capacitances and the total as well
fig, ax = plt.subplots(nrows=1, ncols=1)

#plot the data,multiply by 1e6 to get µF
ax.plot(E, 1e6*cap_helmholtz()*np.ones(len(E)), 'b-', label='Helmholtz')
ax.plot(E, 1e6*cap_gouychapman(E), 'g-', label='Gouy-Chapman')
ax.plot(E, 1e6*cap_stern(E), 'r-', label='Stern')

#set axis labels
ax.set_xlabel('$E-E_{pz}$ [V]')
ax.set_ylabel('C [$\mu F \cdot cm^{-2}$]')

#set axis limits
ax.set_ylim(0,100)
ax.set_xlim(-0.2,0.2)

#figure legend
ax.legend(loc='best', ncol=1, frameon=False, fontsize=10)

#savefig
#plt.savefig('double-layer-cap_vs_potential.png', dpi=200)

plt.show()

#electrolyte concentrations (in M)

c_electrolyte_M = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5] 

#electrolyte concentration (convert to array and to units of mol/m³ and then back again to a list)

c_electrolyte = list(np.array(c_electrolyte_M)*1e3)

#now plot the component capacitances and the total as well
fig, ax = plt.subplots(nrows=1, ncols=1)

cmap = plt.cm.Greens(range(50,255,20))

#plot the data,multiply by 1e6 to get µF
for i,conc in enumerate(c_electrolyte):
    ax.plot(E, 1e6*cap_stern(E, c=conc), ls='-', color=cmap[i], label=str(conc/1e3)+' M')
    
ax.plot(E, 1e6*cap_helmholtz()*np.ones(len(E)), 'k--', label='Helmholtz')

#set axis labels
ax.set_xlabel('$E-E_{pz}$ [V]')
ax.set_ylabel('C [$\mu F \cdot cm^{-2}]$')

#figure legend
ax.legend(loc='best', ncol=3, frameon=False, fontsize=10)

#set axis limits
ax.set_ylim(0,32)
ax.set_xlim(-0.2,0.2)

#save figure
#plt.savefig('double-layer-cap_vs_conc.png', dpi=200)

plt.show()





