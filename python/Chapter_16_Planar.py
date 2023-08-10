# Ex 16.2
import math

# Variable Declaration
alphaA = 2  #[radians per second square]

# Calculation
thetaB = round(1*(2*math.pi)/1,3)  #[radians]
thetaA = round(thetaB*0.4/0.15,2)  #[radians]
wA = round(math.sqrt(0+2*2*(thetaA-0)),3)  #[radians per second]
wB = wA*0.15/0.4  #[radians per second]
alphaB = alphaA*0.15/0.4  #[radians per second square]
vP = round(wB*0.4,2)  #[meters per second]
alphaPt = alphaB*0.4  #[radians per second square]
alphaPn = wB**(2)*0.4  #[radians per second square]
alphaP = round(math.sqrt(alphaPt**(2)+alphaPn**(2)),2)  #[radians per second square]

# Result
print"vP = ",(vP),"m/s"
print"alphaP = ",(alphaP),"m/s**(2)"

# Ex 16.5
import math
from __future__ import division

# Variable Declaration
theta = 30  #[degrees]
vC = 0.5  #[meter per second]

# Calculation
s = round(math.sqrt(5-4*math.cos(math.pi*theta/180)),3)  #[meters]
w = round((s*0.5)/(2*math.sin(math.pi*theta/180)),3)  #[radians per second]
alpha = round((0.5**(2)-2*math.cos(math.pi*theta/180)*w**(2))/(2*math.sin(math.pi*theta/180)),3)  #[Degrees]

# Result
print"w = ",(w),"rad/s"
print"alpha = ",(alpha),"rad/s**(2)"

# Ex 16.6
import math

# Variable Declaration
vA = 2  #[meters per second]
theta = 45  #[Degrees]

# Calculation
# Equating j component
w = round(vA/(0.2*math.sin(math.pi*theta/180)),1)  #[radians per second]
# Equating i component
vB = round(0.2*w*math.sin(math.pi*theta/180),1)  #[meters per second]

# Result
print"w = ",(w),"rad/s"
print"vB = ",(vB),"m/s"

# Ex 16.7
import math

# Calculation
# Solution 1 Vector Analysis
vA_x = 1+3.0  #[meters per second]
vA_y = 3   #[meters per second]
vA = round(math.sqrt(vA_x**(2)+vA_y**(2)),1)   #[meters per second]
theta = round(math.degrees(math.atan(vA_y/vA_x)),1)  #[Degrees]

# Result 1 Vector Analysis
print"Solution 1 Vector Analysis"
print"vA = ",(vA),"m/s"
print"theta = ",(theta),"degrees\n"

# Solution 2  Scalar Analysis
vA_x = 1+4.24*math.cos(math.pi*45/180)   #[meters per second]
vA_y = 0+4.24*math.sin(math.pi*45/180)   #[meters per second]
vA = round(math.sqrt(vA_x**(2)+vA_y**(2)),1)   #[meters per second]
theta = round(math.degrees(math.atan(vA_y/vA_x)),1)  #[Degrees]

# Result 2 Scalar Analysis
print"Solution 2 Scalar Analysis"
print"vA = ",(vA),"m/s"
print"theta = ",(theta),"degrees"

# Ex 16.8

# Calculation
# Link CB
wCB = 2/0.2  #[radians per second]
vB = 0.2*wCB  #[meters per second]
# Link AB
wAB = 2/0.2  #[radians per second]

# Result
print"vB = ",(vB),"m/s"
print"wAB = ",(wAB),"rad/s"

# Ex 16.9

# Calculation
# Link BC
vC = 5.20  #[meters per second]
wBC = 3.0/0.2  #[radians per second]
# Wheel
wD = 5.20/0.1  #[radians per second]

# Result
print"wBC = ",(wBC),"rad/s"
print"wD = ",(wD),"rad/s"

# Ex 16.10
import math

# Variable Declaration
vD = 3  #[meters per second]

# Calculation
rBIC = round(0.4*math.tan(math.pi*45/180),1)   #[meters]
rDIC = round(0.4/math.cos(math.pi*45/180),3)   #[meters]
wBD = round(vD/rDIC,2)  #[radians per second]
vB = wBD*rBIC  #[meters per second]
wAB = vB/0.4  #[radians per second]

# Result
print"wBD = ",(wBD),"rad/s"
print"wAB = ",(wAB),"rad/s"

# Ex 16.12

# Calculation
x = 0.1/0.65   #[meters]
w = 0.4/x  #[radians per second]
vC = w*(x-0.125)  #[meters per second]

# Result
print"w = ",(w),"rad/s"
print"vC = ",(vC),"m/s"

# Ex 16.13
import numpy as np
import math

# Calculation
a = np.array([[math.cos(math.pi*45/180),0],[math.sin(math.pi*45/180),-10]])
b = np.array([3*math.cos(math.pi*45/180)-0.283**(2)*10,-3*math.sin(math.pi*45/180)])
x = np.linalg.solve(a, b)
aB = x[0]  #[meters per second square]
alpha = round(x[1],3)  #[radians per second square]
              
# Result
print"alpha = ",(alpha),"rad/s**(2)"              

# Ex 16.15

# Calculation
# For point B
aB_x = -2-6**(2)*0.5  #[meters per second square]
aB_y = 4*0.5  #[meters per second square]
# For point A
aA_x = -2-4*0.5  #[meters per second square]
aA_y = -6**(2)*0.5  #[meters per second square]

# Result
print"aB_x  = ",(aB_x),"m/s**(2)"
print"aB_y  = ",(aB_y),"m/s**(2)"
print"aA_x  = ",(aA_x),"m/s**(2)"
print"aA_y  = ",(aA_y),"m/s**(2)"

# Ex 16.16
import math

# Variable Declaration
w = 3  #[radians per second]
alpha = 4  #[radians per second square]

# Calculation
aB_x = alpha*0.75  #[meters per second square]
aB_y = -2-w**(2)*0.75  #[meters per second square]
aB = round(math.sqrt(aB_x**(2)+aB_y**(2)),2)  #[meters per second square]
theta = round(math.degrees(math.atan(-aB_y/aB_x)),1)  #[Degrees]

# Result
print"aB = ",(aB),"m/s**(2)"
print"theta = ",(theta),"degrees"

# Ex 16.17
import numpy as np

# Calculation
a = np.array([[0.2,-0.2],[0,0.2]])
b = np.array([-20,1])
x = np.linalg.solve(a, b)
alphaAB = round(x[0],1)  #[meters per second square]
alphaCB = round(x[1],1)  #[radians per second square]
                
# Result
print"alphaCB = ",(alphaCB),"rad/s**(2)"   
print"alphaAB = ",(alphaAB),"rad/s**(2)"                

# Ex 16.18
import numpy as np
import math

# Calculation
rB_x = -0.25*math.sin(math.pi*45/180)  #[meters]
rB_y = 0.25*math.cos(math.pi*45/180)  #[meters]
rCB_x = 0.75*math.sin(math.pi*13.6/180)  #[meters]
rCB_y = 0.75*math.cos(math.pi*13.6/180)  #[meters]
aB_x = np.cross([0,0,-20],[-0.177,0.177,0])[0]-10**(2)*-0.177  #[meters per second square]
aB_y = np.cross([0,0,-20],[-0.177,0.177,0])[1]-10**(2)*0.177   #[meters per second square]
alphaBC = round(20.17/0.729,1)   #[radians per second square]
aC = round(0.176*alphaBC-18.45,1)   #[meters per second square]

# Result
print"alphaBC = ",(alphaBC),"rad/s**(2)"
print"aC = ",(aC),"m/s**(2)"

# Ex 16.19
import numpy as np

# Calculation
aCor_x = np.cross([0,0,2*-3],[2,0,0])[0]  #[meters per second square]
aCor_y = np.cross([0,0,2*-3],[2,0,0])[1]  #[meters per second square]
vC_x = 2  #[meters per second]
vC_y = -0.6  #[meters per second]
aC_x = 3-1.80  #[meters per second square]
aC_y = -0.4-12  #[meters per second square]

# Result
print"aCor_x = ",(aCor_x),"m/s**(2)"
print"aCor_y = ",(aCor_y),"m/s**(2)"
print"vC_x = ",(vC_x),"m/s"
print"vC_y = ",(vC_y),"m/s"
print"aC_x = ",(aC_x),"m/s"
print"aC_y = ",(aC_y),"m/s"

# Ex 16.20

# Calculation
vCDxyz = 1.2  #[meters per second]
wDE = 3  #[radians per second]
aCDxyz = 3.6-2  #[meters per second square]
alphaDE = (7.2-5.2)/-0.4  #[radians per second square]

# Result
print"wDE = ",(wDE),"rad/s"
print"alphaDE = ",(alphaDE),"rad/s**(2)"

# Ex 16.21

# Calculation
vABxyz_x = 0  #[kilometer per hour]
vABxyz_y = 700-600-(-1.5*-4)   #[kilometer per hour]
aABxyz_x = -900+9+282  #[kilometer per hour square]
aABxyz_y = 50+100+2  #[kilometer per hour square]

# Result
print"vABxyz_x = ",(vABxyz_x),"km/h"
print"vABxyz_y = ",(vABxyz_y),"km/h"
print"aABxyz_x = ",(aABxyz_x),"km/h**(2)"
print"aABxyz_y = ",(aABxyz_y),"km/h**(2)"



