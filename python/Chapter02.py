#importing modules
import math
from __future__ import division

#Variable declaration
I=1/2

#Calculation
theta1=math.acos(1/math.sqrt(2))*(180/math.pi)
theta2=math.acos(-1/math.sqrt(2))*(180/math.pi)
#Result
print"theta=",theta1,"degrees"
print"theta=",theta2,"degrees"
print"#The value of theta can be +(or)- 45 degrees and +(or)-135 degrees."

#importing modules
import math
from __future__ import division

#Calculation
ip=math.atan(1.732)*(180/math.pi)

#Result
print"ip=",round(ip),"degrees"

#importing modules
import math
from __future__ import division

#Variable declaration
d=1*10**-3
lamda=6000*10**-10
nd=0.01                  #difference between the refractive indices(n1 - n2)

#Calculation
phi=(2*math.pi*d*nd)/lamda

#Result
print"phi=",round(phi,1),"rad."
print"Since the phase difference should be with in 2pi radius, we get phi=4.169 rad."

#importing modules
import math
from __future__ import division

#Variable declaration
lamda=5000*10**-10
mu_0=1.5533
mu_1=1.5442

#Calculations
t=lamda/(2*(mu_0 - mu_1))
         
#Result
print"Thickness,t=",round(t*10**6,2),"micro m." 

#importing modules
import math
from __future__ import division

#Variable declaration
lamda=6000*10**-10
t=0.003*10**-2

#Calculations
delta_mu=lamda/(4*t)

#Result
print"Birefringence of the crystal delta/mu=",delta_mu

#importing modules
import math
from __future__ import division

#Variable declaration
theta=60*(math.pi/180)        #When the angle of refraction is 30degrees, angle of reflection will be 60degrees

#Calculation
mu=math.tan(theta)

#Result
print"Refractive index of medium=",round(mu,3)

#importing modules
import math
from __future__ import division

#Variable declaration
m=1
lamda_l=6000*10**-10
theta=0.046*(math.pi/180)
n=2*10**6

#Calculation
lamda_s=(m*lamda_l)/(math.sin(theta))
v=n*lamda_s

#Result
print"Ultrasonic wavelength,lamda s =",round(lamda_s*10**4,2),"*10**-4 m"
print"Velocity of ultrasonic waves in liquid =",round(v),"ms**-1"
print"#Answer varies due to rounding of numbers"

#importing modules
import math
from __future__ import division

#Variable declaration
C=1500
Df=267
f=2*10**6
theta=0*math.pi/180          #degrees

#Calculation
V=(C*Df)/(2*f*math.cos(theta))

#Result
print"Velocity of blood flow =",round(V,4),"m s**-1"

#importing modules
import math
from __future__ import division

#Variable declaration
t=0.7*10**-3
E=8.8*10**10
rho=2800

#Calculation
f=(1/(2*t))*math.sqrt(E/rho)       #Fundamental frequency

#Result
print"Fundamental frequency,f =",round(f*10**-6),"*10**6 Hz."

#importing modules
import math
from __future__ import division

#Variable declaration
v=1500
t=1.33

#Calculation
d=(v*t)/2

#Result
print"The depth of the sea =",d,"m."

