# Ex 14.1
import math
from __future__ import division

# Variable Declaration
P = 400  #[Newtons]
s = 2  #[meters]

# Calculation
# Horizontal Force P
UP = round(P*s*math.cos(math.pi*30/180),1)  #[Joules]
# Spring force Fs
Us = round(-((1/2)*30*2.5**(2)-(1/2)*30*0.5**(2)),1)  #[Joules]
# Weight W
UW = round(-98.1*(2*math.sin(math.pi*30/180)),1)  #[Joules]
# Total Work
UT = UP+Us+UW  #[Joules]

# Result
print"UT = ",(UT),"J"

# Ex 14.2
from __future__ import division
import math

# Variable Declaration
uk = 0.5
m = 20  #[kilo Newton]

# Calculation
# Using +ΣFn = 0
NA = round(m*math.cos(math.pi*10/180),2)  #[kilo Newtons]
FA = uk*NA  #[kilo Newtons]
# Principle of Work and Energys
s = round((-(1/2)*(m/9.81)*(5**(2)))/(m*math.sin(math.pi*10/180)-9.85),1)  #[meters]
           
# Result
print"s = ",(s),"m"

# Ex 14.3
from __future__ import division
from scipy import integrate

# Calculation
v = round((2.78*3+0.8*3**(3))**(1/2),2)  #[meters per second]
x = lambda s : 1/((2.78*s+0.8*s**(3))**(1/2))
t = round(integrate.quad(x,0,3)[0],2)  #[seconds]
 
# Result
print"v = ",(v),"m/s"
print"t = ",(t),"s"

# Ex 14.4
from __future__ import division

# Calculation
# Using Principle of Work and Energy
h = round(((-(1/2)*200*(0.6**(2))+(1/2)*200*(0.7**(2)))/(19.62))+0.3,3)  #[meters]

# Result
print"h = ",(h),"m"

# Ex 14.5
import math

# Calculation
thetamax = round(math.degrees(math.acos((9.81+1)/(4.905+9.81))),1)  #[Degrees]

# Result
print"thetamax = ",(thetamax),"degrees"

# EX 14.6
from __future__ import division

# Calculation
vA = -4*2  #[meters per second]
# Substituting delta_sA = -4*delta_sB
delta_sB = round(((1/2)*10*(vA**(2))+(1/2)*100*(2**(2)))/(-4*98.1+981),3)  #[meters]

# Result
print"delta_sB = ",(delta_sB),"m"

# Ex 14.8
import math
from __future__ import division

# Variable Declaration
uk = 0.35

# Calculation
# Using +ΣFy(upward) = 0
NC = 19.62  #[kilo Newtons]
FC = uk*NC  #[kilo Newtons]
v = round(math.sqrt(((1/2)*2000*(25**(2))-6.867*(10**(3))*10)/((1/2)*2000)),2)  #[meters per second]
P = round(FC*v,1)  #[kilo Watts]

# Result
print"P = ",(P),"kW"  # Correction in the answer

# Ex 14.9
import math

# Calculation
# Using Principle of Conservation of Energy
vB = round(math.sqrt((8000*9.81*20*math.cos(math.pi*15/180)-8000*9.81*20*math.cos(math.pi*60/180))/((1/2)*8000)),1)  #[meters per second]
# Using ΣFn = m*an
T = 8000*9.81*math.cos(math.pi*15/180)+8000*(13.5**(2))/20  #[Newtons]

# Result
print"T = ",round((T/1000),1),"kN"

# Ex 14.10
import numpy as np

# Calculation
coeff = [13500, -2481, -660.75]
# Taking positive root
sA = round(np.roots(coeff)[0],3)  #[meters]

# Result
print"sA = ",(sA),"m"

# Ex 14.11
import math
from __future__ import division

# Calculation
# Part(a) Potential Energy
vC = round(math.sqrt((-(1/2)*3*(0.5**(2))+2*9.81*1)/((1/2)*2)),2)  #[meters per second]

# Result Part(a)
print"Part(a)"
print"vC = ",(vC),"m/s\n"

# Part(b) Conservation of Energy
vC = round(math.sqrt(((1/2)*2*(2**(2))-(1/2)*3*(0.5**(2))+2*9.81*1)/((1/2)*2)),2)   #[meters per second]

# Result Part(b)
print"Part(b)"
print"vC = ",(vC),"m/s"



