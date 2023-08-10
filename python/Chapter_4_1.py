#importing modules
import math
from __future__ import division

#Variable declaration
m=9.1*10**-31;     #mass(kg)
nx=ny=nz=1;
n=6;
a=1;     #edge(m)
h=6.63*10**-34;     #planck's constant
k=1.38
#Calculation
E1=h**2*(nx**2+ny**2+nz**2)/(8*m*a**2);
E2=h**2*n/(8*m*a**2);
E=E2-E1;               #energy difference(J)
T=(2*E2*10**37)/(3*k*10**-23)
#Result
print "energy difference is",round(E*10**37,2),"*10**-37 J"
print "3/2*k*T = E2 =",round(E2*10**37,2),"*10**-37 J"
print "T =",round(T/10**23,2),"*10**-14 K"

#importing modules
import math
from __future__ import division

#Variable declaration
y=1/100;    #percentage of probability
x=0.5*1.6*10**-19;     #energy(J)
k=1.38*10**-23;       #boltzmann constant

#Calculation
xbykT=math.log((1/y)-1);
T=x/(k*xbykT);       #temperature(K)

#Result
print "temperature is",int(T),"K"
print "answer varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
d=970;    #density(kg/m**3)
Na=6.02*10**26;    #avagadro number
w=23;    #atomic weight
m=9.1*10**-31;     #mass(kg)
h=6.62*10**-34;     #planck's constant

#Calculation
N=d*Na/w;    #number of atoms/m**3
x=h**2/(8*m);
y=(3*N/math.pi)**(2/3);
EF=x*y;     #fermi energy(J)

#Result
print "fermi energy is",round(EF/(1.6*10**-19),1),"eV"

#importing modules
import math
from __future__ import division

#Variable declaration
kT=1;
E_EF=1;

#Calculations
p_E=1/(1+math.exp(E_EF/kT)) 
        
#Result        
print "p(E) =",round(p_E,3)

#importing modules
import math
from __future__ import division
from scipy.integrate import quad 

#Variable declarations
m=9.1*10**-31
h=6.626*10**-34
Ef=3.1
Ef1=Ef+0.02
e=1.6*10**-19
#Calculations
def zintg(E):
    return math.pi*((8*m)**(3/2))*(E**(1/2)*e**(3/2))/(2*(h**3))
N=quad(zintg,Ef,Ef1)[0]

#Result
print"N =",round(N*10**-26,1),"*10**26 states"

#importing modules
import math
from __future__ import division

#Variable declaration
N=6.023*10**26                   #Avagadro number
D=8960                           #density  
F_e=1                            #no.of free electrons per atom     
W=63.54                          #Atomic weight
i=10
e=1.602*10**-19
m=9.1*10**-31
rho=2*10**-8
Cbar=1.6*10**6                   #mean thermal velocity(m/s)

#Calculations
n=(N*D*F_e)/W
A=math.pi*0.08**2*10**-4
Vd=i/(A*n*e)                      #Drift speed
Tc=m/(n*(e**2)*rho)
lamda=Tc*Cbar

#Result
print"n =",round(n/10**28,1),"*10**28 /m**3"
print"The drift speed Vd =",round(Vd*10**5,1),"*10**-5 m/s"
print"The mean free collision time Tc =",round(Tc*10**14,3),"*10**-14 seconds"
print"Mean free path =",round(lamda*10**8,2),"*10**-8 m""(answer varies due to rounding off errors)" 

#importing modules
import math
from __future__ import division

##Variable declaration
n=8.5*10**28
e=1.602*10**-19
t=2*10**-14
m=9.1*10**-31

#Calculations
Tc=n*(e**2)*t/m

#Result
print "The mean free collision time =",round(Tc/10**7,1),"*10**7 ohm**-1 m**-1"

#importing modules
import math
from __future__ import division


#Variable declaration
e=1.6*10**-19
E=1                    #(V/m)
rho=1.54*10**-8
n=5.8*10**28

#Calculations
T=m/(rho*n*e**2)
Me=(e*T)/m
Vd=Me*E

#Result 
print"Relaxation time =",round(T*10**14),"*10**-14 second"
print"Mobility =",round(Me*10**3),"*10**-3 m**2/volt-s"
print"Drift Velocity=",round(Vd*100,1),"m/s"

#importing modules
import math
from __future__ import division


##Variable declaration
rho_r=0
T=300
rho=1.7*10**-18

#Calculations 
a=rho/T
rho_973=a*973

#Results
print"Temperature coefficient of resistivity,a =",round(a*10**21,1)
print"rho_973 =",round(rho_973*10**18,2),"*10**-8 ohm-m"

#importing modules
import math
from __future__ import division


##Variable declaration
rho1=1.2*10**-8
p1=0.4
rho2=0.12*10**-8
p2=0.5
rho3=1.5*10**-8
#Calculations
R=(rho1*p1)+(rho2*p2)
R_c=R+rho3

#Results
print"Increase in resistivity in copper =",round(R*10**8,2),"*10**-8 ohm m"
print"Total resistivity of copper alloy =",round(R_c*10**8,2),"*10**-8 ohm m"
print"The resistivity of alloy at 3K =",round(R*10**8,2),"*10**-8 ohm m"

