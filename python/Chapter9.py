#importing module
import math
from __future__ import division

#Variable declaration
P=60                   #Power
r=2                    #distance from source
epsilon0=8.85*10**-12
C=3*10**2

#Calculations
E0=math.sqrt((P*2)/(4*math.pi*r**2*C*epsilon0))/1000

#Result
print"Amplitude of field E= ",round(E0),"V/m"

#importing module
import math
from __future__ import division

#Variable declaration
P=8*10**-4              #Power
A=2*10**-6              #cross-sectional Area
epsilon0=8.85*10**-12
C=3*10**2

#Calculations
I=P/A/100
E0=math.sqrt((2*I)/(C*epsilon0))/100
B0=E0/C

#Result
print"Intensity of Beam= %i*10**2 W" %I
print"E0= %i" %round(E0),"V/m"
print"B0= %1.2f" %B0,"myu-T"

#importing module
import math
from __future__ import division

#Variable declaration
E0=9*10**-12
myu0=4*math.pi*10**-7
r=10**4                  #radius of Hemisphere
epsilon0=8.85*10**-12
C=3*10**2
P=10**5                   

#Calculations
S=P/(2*math.pi*r**2)/10**-4
E0=math.sqrt((2*S)/(C*epsilon0))/10**5
B0=E0/C/10**-4

#Result
print"Poynting vector S= %1.2f*10**-4 W/m**2" %S
print"E0= %0.3f V/m" %E0
print"B0= %2.1f *10**-10 T" %B0

#importing module
import math
from __future__ import division

#Variable declaration
myu0=4*math.pi*10**-7
r=2                         #radius of Hemisphere
epsilon0=8.85*10**-12
P0=1000                     #Power 

#Calculations
E=((P0*math.sqrt(myu0/epsilon0))/(16*math.pi))**(1/2)
H=P0/(16*math.pi*E)

#Result
print"Intensity of Electric field E= %2.2f" %E,"V/m"
print"Intensity of Magnetic field H= %1.2f" %H,"amp. turn/m"

#importing module
import math
from __future__ import division

#Variable declaration
E=81
c=3*10**8             #speed of ligth

#Calculations
n=math.sqrt(E)
V=c/n/10**7

#Result
print"Refractive index n= %i" %n
print"Velocity of light= %1.2f*10**7 m/sec"  %V

