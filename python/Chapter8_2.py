#importing modules
import math
from __future__ import division

#Variable declaration
k=1.000074
E=100
E0=8.854*10**-12
n=0.268*10**26

#Calculations
p=(k-1)*E0*E
P=(p/n)*10**38

#Result
print"The Dipole Moment induced in each Helium atom is %1.3f"%P,"*10**-38 Coul-m"

#importing modules
import math
from __future__ import division

#Variable declaration
k=1.000074
#Calculations
X=(k-1)

#Result
print"The Electrical Susceptibility is %0.6f"%X

#importing modules
import math
from __future__ import division

#Variable declaration
E=1*10**-4
D=5*10**-4
V=0.5
P=4*10**-4

#Calculations
Er=(D/E)
NDM=P*V

#Result
print"(a) The Value of Er is %i"%Er
print"(b) The Net Dipole Moment is ",NDM,"coul-m or 2*10**-4 coul-m"

#importing modules
import math
from __future__ import division

#Variable declaration
k=3
E0=8.854*10**-12
E=10**6

#Calculations
P=(E0*(k-1)*E)*10**6
D=(E0*k*E)*10**6
Ed=0.5*E0*k*(E**2)

#Result
print"(a) The Polarization in the Dielectric is %2.2f"%P,"*10**-6 coul/m**2"
print"(b) The Displacement Current Density is %2.2f"%D,"*10**-6 coul/m**2"
print"(c) The Energy Density is",Ed,"J/m**3"

