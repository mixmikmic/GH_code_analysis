import numpy as np

#variable Declaration
A = np.array([1,2,3]) # A is a vector

#calculations
l=np.linalg.norm(A)   # magnitude or length of vector A
a=A/l                 # direction of vector A

#results
print "magnitude of vector:",round(l,2)
print "direction of vector",np.around(a,3)

import numpy as np

#variable Declaration
A=np.array([2,5,6])  # vector A
B=np.array([1,-3,6]) # vector B

#calculations
Sum = A+B  # summation of two vectors
Sub = A-B  # subtraction of two vectors

#results
print "summation of two vectors:",Sum
print "subtraction of two vectors:",Sub

import numpy as np

#Variable Declaration

A = np.array([1,1,2]) # vector A
B = np.array([2,1,1]) # vector B

#Calculations
k = np.dot(A,B) # dot product of vector A and B

#Results

print "dot product of vector A and B:",k

import numpy as np

#Variable Declaration

A = np.array([2,1,2]) # vector A
B = np.array([1,2,1]) # vector B

#Calculations
Cross = np.cross(A,B) # dot product of vector A and B

#Results

print "cross product of vector A and B:",Cross

import numpy as np

#Variable Declaration

A = np.array([1,3,4]) # vector A
B = np.array([1,0,2]) # vector B

#Calculations
k = np.dot(A,B) # dot product of vector A and B

#Results

print "dot product of vector A and B:",k

from __future__ import division
import math

#variable declaration
p = [1,2,3] # coordinates of point p
x = 1       # x coordinate of P
y = 2       # y coordinate of P
z = 3       # z coordinate of P

#Calculations
rho = math.sqrt(x**2+y**2) #radius of cylinder in m
phi = (math.atan(y/x))*(180/math.pi) # azimuthal angle in degrees
z = 3 # in m


#results
print "radius of cylinder is:",round(rho,2),"m"
print "azimuthal angle is:",round(phi,2),"degrees"
print "z coordinate is:",z,"m"

from __future__ import division
from math import cos,sin,pi,atan
import numpy as np

#Variable Declaration

A = np.array([4,2,1])  # vector A
A_x = 4  # x coordinate of P
A_y = 2  # y coordinate of P
A_z = 1  # z coordinate of P


#calculations
phi = atan(A_y/A_x)  # azimuthal in radians
A_rho = (A_x*cos((phi)))+(A_y*sin((phi)))   # x coordinate of cylinder
A_phi = (-A_x*sin(phi))+(A_y*cos(phi))      # y coordinate of cylinder
A_z = 1  # z coordinate of cylinder
A = [A_rho,A_phi,A_z]  # cylindrical coordinates if vector A

#Result
print "cylindrical coordinates of vector A:",np.around(A,3)

from __future__ import division
from math import sqrt,acos,atan
import numpy as np

#Variable Declaration
P = np.array([1,2,3])  # coordinates of point P in cartezian system
x = 1 # x coordinate of point P in cartezian system
y = 2 # y coordinate of point P in cartezian system
z = 3 # z coordinate of point P in cartezian system

#calculations
r = sqrt(x**2+y**2+z**2)  # radius of sphere in m
theta = acos(z/r)  # angle of elevation in degrees
phi = atan(x/y)  # azimuthal angle in degrees

#results
print "radius of sphere is:",round(r,3),"m"
print "angle of elevation is:",round(theta,3),"radians"
print "azimuthal angle is:",round(phi,3),"radians"


# note : answer in the book is incomplete they find only one coordinate but there are three

import numpy as np

#variable declaration
A_p=22  # power gain

#calulation
A_p_dB=10*(np.log10(A_p)) # power gain in dB

#result
print "power gain is:",round(A_p_dB,3),"dB"

import numpy as np

#variable declaration
A_v=95  # voltage gain

#calculation
A_v_dB=20*(np.log10(A_v)) # voltage gain in dB

#result
print "voltage gain is:",round(A_v_dB,3),"dB"

import numpy as np
from math import sqrt

#variable declaration
A_p = 16  # power gain

#calculations
A_p_Np = np.log(sqrt(A_p)) # power gain in Np

#results
print "power gain is:",round(A_p_Np,3),"Np"

import numpy as np

#variable declaration
A_i = 34  # current gain

#calculations
A_i_Np = np.log(A_i) # current gain in Nepers

#result
print "power gain is:",round(A_i_Np,3),"Np"

from __future__ import division
import cmath 
from math import sqrt,pi


#variable declaration
A=2+4j  # complex number A

#calculations
magnitude = abs(A)  # magnitude of complex number A
phi = cmath.phase(A)*(180/pi)  # phase of complex number A in degrees

#results
print "magnitude of complex number A is:",round(magnitude,3)
print "phase of complex number A is:",round(phi,3),"degrees"

from __future__ import division
from math import pi
import cmath

#variable declaration
A = 1+3j  # complex no. A

#calculations
c = A.conjugate()  # conjugate of complex no. A
magnitude = abs(A)  # magnitude of complex number A
phi = cmath.phase(A)*(180/pi)  # phase of complex number A in degrees


#results
print "magnitude of complex number A is:",round(magnitude,3)
print "phase of complex number A in degrees:",round(phi,3)
print "conjugate of complex no. A:",c

from __future__ import division
from math import cos,sin,radians
import numpy as np

#variable declaration
rho = 5   # magnitude of the complex number A
phi = 45  # phase of a complex number A in Degrees

#calculations
x = rho*cos(radians(phi))  # real part of complex number A
y = rho*sin(radians(phi))  # imaginary part of complex number A
A = complex(x,y)  # complex number A

#results
print "real part of complex number A:",round(x,3)
print "imaginary part of complex number A:",round(y,3)
print "complex number A:",np.around(A,3)

#Variable Declaration

A_1 = 2+3j # complex number A_1
A_2 = 4+5j # complex number A_2


#calculation
A = A_1 + A_2

#Result
print "sum of complex numbers A_1 and A_2 is:",A

#Variable Declaration

A_1 = 6j # complex number A_1
A_2 = 1-2j # complex number A_2


#calculation
A = A_1 - A_2

#Result
print "Difference of complex numbers A_1 and A_2 is:",A

#Variable Declaration

A = 0.4 + 5j # complex number A
B = 2+3j     # complex number B


#calculation
P = A*B

#Result
print "Product of complex numbers A and B is:",P

import numpy as np

#Variable Declaration

A = 10+6j # complex number A
B = 2-3j  # complex number B 

#calculation
D = A/B

#Result
print "Division of complex numbers A and B is:",np.around(D,3)

from sympy import *

#variable Declaration

x = Symbol('x')
p = (x)**2 + 2*x + 4

#calculations
Roots = solve(p,x)


#result
print "The roots of the given quadratic equation are:",Roots

for i in range(len(Roots)):
    print "Root %i = %s" % (i + 1, str(Roots[i].n(5)))

from math import factorial

f1 = factorial(4)   # factorial of 4
f2 = factorial(6)   # factorial of 6
print "factorial of 4 is:",f1
print "factorial of 6 is:",f2

