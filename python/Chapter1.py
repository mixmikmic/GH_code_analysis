#importing modules
import math
from __future__ import division
from scipy.integrate import tplquad

#Calculation
func = lambda x,y,z: 2*(x+y+z)
x1,x2 = 0,1
y1,y2 = lambda x: 0, lambda x: 1
z1,z2 = lambda x,y: 0, lambda x,y: 1
r1,r2=tplquad(func,x1,x2,y1,y2,z1,z2)      
r=r1-r2;          #divergence of force vector

#Result
print "divergence of force vector is",int(round(r))

#importing modules
import math
from __future__ import division
from scipy.integrate import quad

#Calculation
def zintg(x):
    return x**3
r1=quad(zintg,0,1)[0]
def zintg(x):
    return 2*x**4
r2=quad(zintg,0,1)[0]
def zintg(x):
    return 3*x**8
r3=quad(zintg,0,1)[0]
r=r1+r2+r3;             #result

#Result
print "the result is",int(r*60),"/60"

#importing modules
import math
from __future__ import division
from scipy.integrate import quad
from fractions import Fraction

#Calculation
def zintg(x):
    return (x-(x**2))
r1=quad(zintg,0,1)[0]
def zintg(x):
    return 2*((x**2)+(x**3))
r2=quad(zintg,0,1)[0]
def zintg(y):
    return 2*((y**3)-(y**2))
r3=quad(zintg,1,0)[0]
def zintg(y):
    return (y**2)+y
r4=quad(zintg,1,0)[0]
r=r1+r2+r3+r4;             #result

#Result
print "the result is",int(r*3),"/3"

