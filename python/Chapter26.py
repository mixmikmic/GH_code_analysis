#Example 1.1

m =3.1 #mass of copper penny in grams
e =4.6*10** -18 #charge in coulombs
N0 =6*10**23 #avogadro’s number atoms / mole
M =64  #molecular weight of copper in gm/ mole

#Calculation
N =( N0 * m ) / M  #No. of copper atoms in penny
q = N * e  # magnitude of the charges in coulombs
print (" Magnitude of the charges in coulomb is ",q )

#Example 2

import math

F =4.5 #Force of attraction in nt
q =1.3*10**5 #total charge in coulomb
r = q * math.sqrt ((9*10**9) / F ) ;
print(" Separation between total positive and negative charges in meters is ",r )

#Example 3

import math

#given three charges q1,q2,q3
q1=-1.0*10**-6 #charge in coul
q2=+3.0*10**-6 #charge in coul
q3=-2.0*10**-6 #charge in coul
r12=15*10**-2 #separation between q1 and q2 in m
r13=10*10**-2 # separation between q1 and q3 in m
angle=math.pi/6 #in degrees
F12=(9.0*10**9)*q1*q2/(r12**2) #in nt
F13=(9.0*10**9)*q1*q3/(r13**2) #in nt
F12x=-F12  #ignoring signs of charges
F13x=F13*math.sin(angle);
F1x=F12x+F13x
F12y=0 #from fig.263
F13y=-F13*math.cos(angle);
F1y=F12y+F13y #in nt
print("X component of resultant force acting on q1 in nt is",F1x)
print("Y component of resultant force acting on q1 in nt is",F1y)

#Example 4

r=5.3*10**-11 #distance between electron and proton in the hydrogen atom in meter
e=1.6*10**-19 #charge in coul
G=6.7*10**-11 #gravitatinal constant in nt-m2/kg2
m1=9.1*10**-31 #mass of electron in kg
m2=1.7*10**-27 #mass of proton in kg
F1=(9*10**9)*e*e/(r**2) #coulomb's law
F2=G*m1*m2/(r**2) #gravitational force
print("Coulomb force in nt is",F1)
print("Gravitational force in nt is",F2)

#Example 5

r=4*10**-15 #separation between proton annd nucleus in iron in meters
q=1.6*10**-19 #charge in coul
F=(9*10**9)*(q**2)/(r**2) #coulomb's law
print("Repulsive coulomb force F ",F,'nt')

