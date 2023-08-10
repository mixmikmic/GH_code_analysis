#importing modules
import math
from __future__ import division

#Variable declaration
KE=10             #Kinetic Energy of neutron in keV
m=1.675*10**-27
h=6.625*10**-34
#Calculations
KE=10**4*1.6*10**-19      #in joule
v=((2*KE)/m)**(1/2)       #derived from KE=1/2*m*v**2
lamda=h/(m*v)
#Results
print"Velocity =",round(v/10**6,2),"*10**6 m/s"
print"Wavelength =",round(lamda*10**10,5),"Angstorm"
                        

#importing modules
import math
from __future__ import division

#Variable declaration
E=2*1000*1.6*10**-19       #in joules
m=9.1*10**-31
h=6.6*10*10**-34

#Calculations
p=math.sqrt(2*m*E)
lamda= h/p

#Result
print"Momentum",round(p*10**23,4)
print"de Brolie wavelength =",round(lamda*10**10,2),"*10**-11 m"

#importing modules
import math
from __future__ import division

#Variable declaration
M=1.676*10**-27             #Mass of neutron
m=0.025
v=1.602*10**-19
h=6.62*10**-34

#Calculations
mv=(2*m*v)**(1/2)
lamda=h/(mv*M**(1/2))

#Result
print"wavelength =",round(lamda*10**10,3),"Angstorm"

#importing modules
import math
from __future__ import division

#Variable declaration
V=10000

#Calculation
lamda=12.26/math.sqrt(V)

#Result
print"Wavelength =",lamda,"Angstorm"

#importing modules
import math
from __future__ import division


#Variable declaration
e=1.6*10**-19;   #charge of electron(coulomb)
L=10**-10            #1Angstrom=10**-10 m
n1=1;
n2=2;
n3=3;
h=6.626*10**-34
m=9.1*10**-31
L=10**-10

#Calculations
E1=(h**2)/(8*m*L**2*e)
E2=4*E1
E3=9*E1
#Result
print"The permitted electron energies =",round(E1),"*n**2 eV"
print"E1=",round(E1),"eV"
print"E2=",round(E2),"eV"
print"E3=",round(E3),"eV"
print"#Answer varies due to rounding of numbers"

#importing modules
import math
from __future__ import division

#Variable declaration
i=1*10**-10;    #interval
L=10*10**-10;   #width

#Calculations
si2=2*i/L;

#Result
print"si**2 delta(x)=",si2

#importing modules
import math
from __future__ import division

#Variable declaration
nx=1
ny=1
nz=1
a=1
h=6.63*10**-34
m=9.1*10**-31

#Calculations
E1=h**2*(nx**2+ny**2+nz**2)/(8*m*a**2)
E2=(h**2*6)/(8*m*a**2)             #nx**2+ny**2+nz**2=6
diff=E2-E1
#Result
print"E1 =",round(E1*10**37,2),"*10**-37 Joule"
print"E2 =",round(E2*10**37,2),"*10**-37 Joule"
print"E2-E1 =",round(diff*10**37,2),"*10**-37 J"

#importing modules
import math
from __future__ import division

#Variable declaration
m=1.67*10**-27
a=10**-14
h=1.054*10**-34

#Calculations
E1=(1*math.pi*h)**2/(2*m*a**2)

#Result
print"E1 =",round(E1*10**13,2),"*10**-13 J"

#importing modules
import math
from __future__ import division
from scipy.integrate import quad
#Variable declarations
k=1;

#Calculations
def zintg(x):
    return 2*k*math.exp(-2*k*x)
a=quad(zintg,2/k,3/k)[0]

#Result
print "a=",round(a,4)

