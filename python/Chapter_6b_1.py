#importing modules
import math
from __future__ import division

#Variable declaration
H0=64*10**3;    #initial field(ampere/m)
T=5;    #temperature(K)
Tc=7.26;   #transition temperature(K)

#Calculation
H=H0*(1-(T/Tc)**2);     #critical field(ampere/m)

#Result
print "critical field is",round(H/10**3,2),"*10**3 ampere/m"

#importing modules
import math
from __future__ import division

#Variable declaration
e=1.6*10**-19
V=1*10
h=6.625*10**-34

#Calculations
v=(2*e*V**-3)/h 

#Result
print"Frequency of generated microwaves=",round(v/10**9),"*10**9 Hz"

#importing modules
import math
from __future__ import division

#Variable declaration
d=7300                  #density in (kg/m**3)
N=6.02*10**26           #Avagadro Number
A=118.7                 #Atomic Weight
E=1.9                 #Effective mass
e=1.6*10**-19

#Calculations
n=(d*N)/A
m=E*9.1*10**-31
x=4*math.pi*10**-7*n*e**2
lamda_L=math.sqrt(m/x)
      
#Result
print "Number of electrons per unit volume =",round(n/10**28,1),"*10**28/m**3"
print"Effective mass of electron 'm*' =",round(m*10**31,1),"*10*-31 kg"
print"Penetration depth =",lamda_L*10**8,"Angstroms"
print"#The answer given in the text book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
lamda_L1=39.6*10**-9
lamda_L2=173*10**-9
T1=7.1
T2=3

#Calculations
x=(lamda_L1/lamda_L2)**2
Tc4=(T1**4)-((T2**4)*x)/(1-x)
Tc=(Tc4)**(1/4)
print"Tc =",round(Tc,4),"K"
print"lamda0=",round((math.sqrt(1-(T2/Tc)**4)*lamda_L1)*10**9),"nm"

#importing modules
import math
from __future__ import division

#Variable declaration
H0=6.5*10**4           #(ampere/metre)
T=4.2                  #K
Tc=7.18                #K
r=0.5*10**-3

#Calculations
Hc=H0*(1-(T/Tc)**2)
Ic=(2*math.pi*r)*Hc
A=math.pi*r**2
Jc=Ic/A                #Critical current density

#Result
print"Hc =",round(Hc/10**4,4),"*10**4"
print "Critical current density,Jc =",round(Jc/10**8,2),"*10**8 ampere/metre**2"

#importing modules
import math
from __future__ import division

#Variable declaration
Tc1=4.185
M1=199.5
M2=203.4

#Calculations
Tc2=Tc1*(M1/M2)**(1/2)

#Result
print"New critical temperature for mercury =",round(Tc2,3),"K"

