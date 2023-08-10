#importing modules
import math
from __future__ import division

#Variable declaration
C=20/(9*10^11)#converting cms to farads

#Calculation
F=154-100#fall in potential
R=F/60#rate of fall in potential
I=C*R#ionization current

#Result
print"The  width = ",round(I*10,4),"*10**-11 Amp"

