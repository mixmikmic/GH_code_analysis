# Ex 18.1
from __future__ import division

# Variable Declaration 
mB = 6  #[kilogram]
mD = 10  #[kilogram]
mC = 12  #[kilogram]

# Calculation
# Block
TB = (1/2)*mB*0.8**(2)  #[Joules]
# Disk
TD = (1/2)*(1/2)*mD*0.1**(2)*8**(2)  #[Joules]
# Cylinder
TC = (1/2)*12*0.4**(2)+(1/2)*(1/2)*mC*0.1**(2)*4**(2)  #[Joules]
# Let T be total kinetic energy of system
T = TB+TD+TC  #[Joules]

# Result
print"T = ",(T),"J"

# Ex 18.2
import math
from __future__ import division

# Variable Declaration
M = 50  #[Newton meter]
P = 80  #[Newton]

# Calculation
# Weight W
UW = 98.1*1.5  #[Joules]
# Couple Moment
UM = M*math.pi/2  #[Joules]
# Spring Force Fs
Us = -((1/2)*30*2.25**(2)-(1/2)*30*0.25**(2))  #[Joules]
# Force P
UP = P*4.712  #[Joules]
# let U be total work
U = round(UW+UM+Us+UP,1)  #[Joules]

# Result
print"U = ",(U),"J"

# Ex 18.3
from __future__ import division
import math

# Variable Declaration
F = 10  #[Newton]
w = 20  #[radians per second]
M = 5  #[Newton meter]

# Calculation
# Kinetic Energy
T1 = 0  #[Joules]
T2 = (1/2)*(1/2)*30*(0.2**(2))*(w**(2))  #[Joules]
# Using principle of Work and Energy
theta = (T2-T1)/(M+F*0.2)  #[radians]
theta = round((theta*1)/(2*math.pi),2)  #[rev]

# Result
print"theta = ",(theta),"rev"

# Ex 18.4
import math

# Calculation
w2 = math.sqrt((700*9.81*0.05359)/63.875)  #[radians per second]
# Using +ΣFn(upward) = m(aG)n
NT = 700*9.81+700*2.40**(2)*0.4  #[kilo Newton]
# Using +ΣMO = IO*alpha
alpha = 0/(700*0.15**(2)+700*0.4**(2))  #[radians per second square]
# Using +ΣFt(left) = m(aG)t and (aG)t = 0.4*alpha
FT = 700*0.4*alpha  #[kilo Newton]
# there are two tines to support the load
FdashT = 0  #[Newton]
NdashT = NT/2  #[Newton]

# Result
print"FdashT = ",(FdashT/1000),"kN"
print"NdashT = ",round((NdashT/1000),2),"kN"

# Ex 18.5
import math

# Variable Declaration
M = 75  #[Newton meter]
k = 50  #[Newton per meter]
W = 20  #[kilogram]

# Calculation
# Using Principle of work and energy
w2 = round(math.sqrt((M*0.625-(1/2)*k*1**(2))/10),2)  #[radians per second]

# Result
print"w2 = ",(w2),"rad/s"

# Ex 18.6
import math

# Variable Declaration
P = 50  #[Newton]

# Calculation
# Using Principle of work and energy
w2 = round(math.sqrt((98.1*(0.4-0.4*math.cos(math.pi*45/180))+50*0.8*math.sin(math.pi*45/180))/(1.067)),2)  #[radians per second]

# Result
print"w2 = ",(w2),"rad/s"

# Ex 18.7
import math
from __future__ import division

# Variable Declaration
m = 10  #[kilogram]
k = 800  #[Newton per meter]

# Calculation
# Potential Energy
V1 = round(-98.1*(0.2*math.sin(math.pi*30/180))+(1/2)*k*(0.4*math.sin(math.pi*30/180))**(2),2)  #[Joules]
# CG is located at datum
V2 = 0  #[Joules]
# Kinetic Energy
# Since the rod is released from rest position 1
T1 = 0  #[Joules]
# Using principle of conservation of energy
w2 = round(math.sqrt((T1+V1-V2)/0.267),2)  #[radians per second]

# Result
print"w2 = ",(w2),"rad/s"

# Ex 18.8
from __future__ import division
import math

# Variable Declaration
k = 30  #[Newton per meter]

# Calculation
V1 = (1/2)*k*(math.sqrt(0.9**(2)+1.2**(2))-0.3)**(2)  #[Joules]
V2 = (1/2)*k*(1.2-0.3)**(2)  #[Joules]
# The disk is releaseg from rest  
T1 = 0  #[Joules]
# Using principle of conservation of energy
w2 = round(math.sqrt((T1+V1-V2)/0.6227),2)  #[radians per second]

# Result
print"w2 = ",(w2),"rad/s"

# Ex 18.9
import math

# Calculation
# Potential energy
V1 = 49.05*0.3*math.sin(math.pi*60/180)  #[Joules]
# At position 2 weight of rod and disk have zero potential energy
V2 = 0  #[Joules]
# Kinetic energy
# Since the entire system is at rest
T1 = 0  #[Joules]
# Using conservation of energy
wR2 = round(math.sqrt((T1+V1-V2)/0.3),2)  #[radians per second]

# Result
print"wR2 = ",(wR2),"rad/s"



