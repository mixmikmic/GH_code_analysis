#importing modules
import math
from __future__ import division

#Variable declaration
h=6.63*10**-34;      #planck's constant
c=3*10**8;      #velocity of light(m/s)
lamda=1.55*10**-6;     #wavelength(m)
e=1.6*10**-19;   

#Calculation
Eg=h*c/(lamda*e);     #energy gap(eV)

#Result
print "energy gap is",round(Eg,1),"eV"

#importing modules
import math
from __future__ import division

#Variable declaration
h=6.63*10**-34;      #planck's constant
c=3*10**8;      #velocity of light(m/s)
Eg=1.44;     #energy gap(eV)
e=1.6*10**-19; 

#Calculation
lamda=h*c/(Eg*e);    #wavelength(m)

#Result
print "wavelength is",round(lamda*10**10),"angstrom"

