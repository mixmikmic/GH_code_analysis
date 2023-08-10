import math
from math import pi
#given
Hc=670                                                           #magnetic field intensity in the core material in At/m
G=5
D=10
A=5
B=10
Bg=1
Z=0.00000125663         
N=250
Area=700
Lc=2*(A+B)+2*(G+D)                                               #length of the flux path in the core
#calculations
Hg=Bg/Z
Lc=0.6
Hg=Bg/Z                                                          #magnetic intensity in the air gap
Ni=(Hc*Lc)+(Hg*10**-2)
I=Ni/N
Vdc=I*G
Wfc=Area/2                                                       #energy density in the core
Vc=2*(G*10**-2*D*10**-2*0.20)+2*(A*10**-2*B*10**-2*0.10)
Wfc=Wfc*Vc                                                       #stored energy in the core
Wfg=1.0/(2*Z)                                                    #energy density in the airgap
Vg=2*(G*10**-2*10*10**-2*0.005)
Wfg=(Wfg*G*10**-2*10**-3)                                        #stored energy in air gap
Wf=Wfc+Wfg
Vdc=round(Vdc,2)
Wf=round(Wf,2)
print 'Voltage of the dc source=',Vdc,'volts'
print 'Total field energy=',Wf,'joules'

import math
#given
I=3                             #current
G=0.05                          #air-gap length
Lam=(0.09*I**(.5)/G)            #Lambda
Wf=((0.09*2)/(G*I))*I**(1.5)
Fm=-0.09*(2/3)*I**(1.5)*(1/G**2)
Wf1=(G**2*Lam**3)/(0.09**2*I)  #The co-energy of the system
Lam1=(0.09*I**(.5)/G)
Fm=-((Lam1**3)*2*G)/(I*0.09**2)
Fm=round(Fm,2)
print 'mechanical force=',Fm,'N-m'

import math
from math import pi
#given
N=500.0
i=2.0            #current
W=2.0            #width of airgap in cm
D=2.0            #depth of airgap in cm
L=1.0            #length of airgap in mm
A=4.0*pi*10.0**-7.0
fm=A*(((N*i)**2)/(2*L**2*10**-6))*W*D*10**-4
W=fm*10**-3
W=round(W,3)
print 'force of attraction=',fm,'N'
print 'energy stored in the air gap=',W,'joules'

import math
from math import pi,sqrt
#given
A=0.00000125663
N=300
V=120                   #voltage
R=6                     #resistance
G=5*10**-3
Ag=6*6*10**-4
Lg=2*5*10**-3           #inductance of the coil
Vo=2*6*6*5*10**-7
I=V/R                   #ohm's law 
Bg=(A*N*I)/(2*G)        #RMS value of the flux density
Wf=(Bg**2)/(2*A)*(Vo)   #field energy
Fm=(Bg**2)/(2*A)*(2*Ag)
L=(N**2*A*Ag)/(Lg)
Irms=V/(sqrt(6**2+15.34**2))
Brms=(A*N*Irms)/(2*G)
Fm=(Brms**2)/(2*A)*(2*Ag)
Fm=round(Fm,2)
Wf=round(Wf,3)
print 'stored field energy=',Wf,'N'
print 'lifting force=',Fm,'N'

