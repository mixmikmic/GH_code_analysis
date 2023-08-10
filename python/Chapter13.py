#importing modules
import math
from __future__ import division

#Variable declaration
l=10*10**-6 #length in m
f=10*10**9 #frequency in Hz
n=2*10**14 # n type doping concentration in cm**-3
e=1.6*10**-19 #in J
E=3200 #electric field in V/cm

#Calculatiions
vd=l*f*10**2 #converting from m**2 to cm**2
J=e*n*vd
myu=-vd/E

#Result
print("Drift velocity= %.0f*10**7 cms**-1" %round(vd/10**7,0))
print("current density= %f A/cm**2" %J) 
print("negative electron mobility= %d cm**2/Vs" %myu) #The answer provided in the textbook is wrong

#importing modules
import math
from __future__ import division

#Variable declaration
drift_length=2*10**-4 #in cm
drift_velocity=2*10**7 #in cm/s

#Calculatiions
d=drift_length/drift_velocity
f=(drift_velocity*10**-2)/(2*drift_length*10**-2)

#Result
print("Drift time= %.0f*10**-11 s" %round(d*10**11,0))
print("Operating frequency= %.0f GHz" %round(f/10**9,0))

#importing modules
import math
from __future__ import division

#Variable declaration
J=20*10**3 #in kA/cm**2
e=1.6*10**-19 #in C
Nd=2*10**15 #in cm**-3

#Calculatiions
vz=J/(e*Nd)

#Result
print("avalanche-zone velocity is= %.2f*10**7 cm/s" %(vz/10**7))

#importing modules
import math
from __future__ import division

#Variable declaration
e=1.6*10**-19 #in eV
Nd=2.8*10**21 # donor doping concentration in m**-3
L=6*10**-6 #length in m
epsilon_s=8.854*10**-12*11.8 # in F/m

#Calculatiions
Vbd=(e*Nd*L**2)/epsilon_s
Ebd=Vbd/L

#Result#importing modules
import math
from __future__ import division

#Variable declaration
e=1.6*10**-19 #in eV
Nd=2.8*10**21 # donor doping concentration in m**-3
L=6*10**-6 #length in m
epsilon_s=8.854*10**-12*11.8 # in F/m

#Calculatiions
Vbd=(e*Nd*L**2)/epsilon_s
Ebd=Vbd/L

#Result
print("Breakdown voltage is= %.2f V" %Vbd)#The answers vary due to round off error
print("Breakdown electric field is=%.2f*10**5 V/m" %round(Ebd/10**7,2))
print("Breakdown voltage is= %.2f V" %Vbd)#The answers vary due to round off error
print("Breakdown electric field is=%.2f*10**5 V/m" %round(Ebd/10**7,2))



