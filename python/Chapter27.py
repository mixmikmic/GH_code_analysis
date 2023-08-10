m=9.1*10**-31 #mass of electron in kg
g=9.8 #acceleration due to gravity in m/s
q=1.6*10**-19 #charge of electron in coul
print("Electric field strength E=F/q where F=mg")
E=m*g/q
print("electric field strength in nt/coul is %.3e"%E)

import math

q1=1.0*10**-6 #in coul
q2=2.0*10**-6 #in coul
l=10 #sepearation b/w q1 and q2 in cm
print("For the electric field strength to be zero the point should lie between the charges where E1=E2")
#"Refer to the fig 27.9"
#E1=electric fied strength due to q1
#E2=electric fied strength due to q2
print("E1=E2 which implies q1/4πϵx2 = q2/4πϵ(l-x)2")
x=l/(1+math.sqrt(q2/q1))
print("Electric field strength is zero at x=%.3f cm"%x)

e=1.6*10**-19 #charge in coul
E=1.2*10**4 #electric field in nt/coul
x=1.5*10**-2 #length of deflecting assembly in m
K0=3.2*10**-16 #kinetic energy of electron in joule
#calculation
y=e*E*x**2/(4*K0)
print("Corresponding deflection in meters is %.6f"%y)

import math
q=1.0*10**-6 #magnitude of two opposite charges of a electric dipole in coul
d=2.0*10**-2 #seperation b/w charges in m
E=1.0*10**5 #external field in nt/coul
#calculations
#(a)Max torque if found when theta=90 degrees
#Torque =pEsin(theta)
p=q*d #electric dipole moment
T=p*E*math.sin(math.pi/2)
print("(a)Maximum torque exerted by the fied in nt-m is")
print(T)
#(b)work done by the external agent is the potential energy b/w the positions theta=180 and 0 degree
W=(-p*E*math.cos(math.pi))-(-p*E*math.cos(0))
print("(b) Work done by the external agent to turn dipole end for end in joule is ")
print(W)

