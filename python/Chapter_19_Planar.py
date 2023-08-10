# Ex 19.1
from __future__ import division 

# Variable Declaration
md = 10  #[kilogram]
mb = 5  #[kilogram]

# Calculation Disk
# Using +HG(clockwise) = IG*w
HG = (1/2)*md*0.25**(2)*8  #[meters square per second]
# Using +HB(clockwise) = IG*w +m(vG)rG
HB = HG+md*2*0.25  #[meters square per second]

# Result Disk
print"HG = ",(HG),"m**(2)/s"
print"HB = ",(HB),"m**(2)/s"


# Calculation Bar
# Using +HG(clockwise) = IG*w
HG = round((1/12)*mb*4**(2)*0.5774,2)  #[meters square per second]
HIC = round(HG+2*mb*1.155,2)  #[meters square per second]

# Result Bar
print"HG = ",(HG),"m**(2)/s"
print"HIC = ",(HIC),"m**(2)/s"

# Ex 19.2
import numpy as np
from __future__ import division 

# Calculation
IA = round((1/2)*(100/9.81)*0.3**(2),3)  #[kilogram meter square]
# Using principle of impulse and momentum
a = np.array([[2,0,0],[0,2,0],[0,0,IA]])
b = np.array([0,100*2+40*2,4*2+40*2*0.3])
x = np.linalg.solve(a, b)
Ax = round(x[0],1)  #[Newton]
Ay = round(x[1],1)  #[Newton]
w2 = round(x[2],1)  #[radians per second]

# Result
print"Ax = ",(Ax),"N"
print"Ay = ",(Ay),"N"
print"w2 = ",(w2),"rad/s"

# Ex 19.3
import numpy as np
from __future__ import division 

# Calculation
IG = round(100*0.35**(2),3)  #[kilogram meter square]
# Using principle of impulse and momentum
a = np.array([[1,100*0.75],[0.75,-12.25]])
b = np.array([62.5,-25])
x = np.linalg.solve(a, b)
w2 = round(x[1],2)  #[radians per second]

# Result

print"w2 = ",(w2),"rad/s"

# Ex 19.4
import numpy as np
from __future__ import division 

# Variable Declaration
IA = 0.40  #[kilogram meter square]

# Calculation Solution 1
# Using principle of impulse and momentum
a = np.array([[3*0.2,-IA/0.2],[3,6]])
b = np.array([-IA*10,6*2+58.86*3])
x = np.linalg.solve(a, b)
T = round(x[0],1)  #[Newton]
vB2 = round(x[1],1)  #[meters per second]

# Result Solution 1
print"Solution 1"
print"vB2 = ",(vB2),"m/s\n"

# Calculation Solution 2
# Using principle of angular impulse and momentum
vB2 = (6*2*0.2+0.4*10+58.86*3*0.2)/(6*0.2+0.4*5*0.2)   #[meters per second]


# Result Solution 2
print"Solution 2"
print"vB2 = ",(vB2),"m/s"

# Ex 19.5
from __future__ import division
import math

# Variable Declaration
IG = 0.156  #[kilogram meter square]

# Calculation
# Using principle of conservation of energy
vG2 = math.sqrt((98.1*0.03)/((1/2)*10+(1/2)*IG*25))  #[meters per second]
# vG2 = 0.892*vG1
vG1 = round(vG2/0.892,3)   #[meters per second]

# Result
print"vG1 = ",(vG1),"m/s"

# Ex 19.7
from __future__ import division

# Calculation
#  Σ(HO)1 =  Σ(HO)2, vG2 = 0.5*w2 and vB2 = 0.75*w2
w2 = round((1.039)/(0.003*0.75+2.5*0.5+0.417),3)  #[radians per second]

# Result
print"w2 = ",(w2),"rad/s"

# Ex 19.8
import numpy as np

# Calculation
# Using principle of conservation of angular momentum and coefficient of restitution
a = np.array([[1.67,0.5],[0.5,-1]])
b = np.array([5,4])
x = np.linalg.solve(a, b)
w2 = round(x[0],2)  #[radians per second]
vB2 = round(x[1],2)  #[meters per second]

# Result
print"w2 = ",(w2),"rad/s"



