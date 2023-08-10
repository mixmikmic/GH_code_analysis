# Ex 15.1
import math
from __future__ import division

# Variable Declaration
ws = 100  #[kilogram]
F = 200  #[Newton]
theta = 45  #[degrees]

# Calculation
v2 = round(F*10*math.cos(math.pi*theta/180)/100,1)  #[meters per second]
NC = round((9.81*ws*10-F*10*math.sin(math.pi*theta/180))/10,1)  #[Newtons]

# Result
print"v2 = ",(v2),"m/s"
print"NC = ",(NC),"N"

# Ex 15.2
import math
from __future__ import division

# Calculation
# Using +ΣFy = 0
NC = round(500*math.cos(math.pi*30/180),1)  #[Newtons]
v2 = round((50.97+100-0.6*NC+500)/50.97,2)  #[meters per second]

# Result
print"v2 = ",(v2),"m/s"
print"NC = ",(NC),"N"

# Ex 15.3
import numpy as np
from __future__ import division

# Calculation
a = np.array([[-(1/2)*3,2*6], [5,6]])
b = np.array([3*9.81*6,5*9.81*6])
x = np.linalg.solve(a, b)
vB2 = round(x[0],1)  #[meters per second]
TB = round(x[1],1)  #[Newtons]

# Result
print"vB2 = ",(vB2),"m/s"
print"TB = ",(TB),"N"

# Ex 15.4
from __future__ import division

# Calculation
# Part(a)
v2 = (15000*1.5-12000*0.75)/27000   #[meters per second]
# Part(b)
Favg = (15000*1.5-15000*0.5)/0.8  #[Newtons]

# Result
print"v2 = ",(v2),"m/s"
print"Favg = ",(Favg/1000),"kN"

# Ex 15.5
from __future__ import division

# Calculation
# Part(a)
vC2 = 4*500/500  #[meters per second]
# Part(b)
Favg = 4*500/0.03  #[Newtons]

# Result
print"vC2 = ",(vC2),"m/s"
print"Favg = ",round((Favg/1000),2),"kN"

# Ex 15.6
from __future__ import division

# Calculation
vT2 = round((350*10**(3)*3)/(350*10**(3)+50*10**(3)),2)  #[meters per second]

# Result
print"vT2 = ",(vT2),"m/s"

# Ex 15.7
import math
from __future__ import division

# Variable Declaration
mH = 300  #[kilogram]
mP = 800  #[kilogram]

# Calculation
# Using conservation of energy
vH1 = round(math.sqrt((mH*9.81*0.5)/((1/2)*mH)),2)  #[meters per second]
# Using conservation of momentum
v2 = (mH*3.13)/(mH+mP)  #[meters per second]
# Using Principle of Impulse and Momentum
Impulse = round(300*vH1-300*v2,1)  #[Newtons second]

# Result
print"Impulse = ",(Impulse),"N.s"

# Ex 15.9
import numpy as np
from __future__ import division
import math

# Calculation
# Using conservation of energy
vA1 = round(math.sqrt((6*9.81*1)/((1/2)*6)),2)
# Using Conservation of Momentum and formula for coefficient of restitution
a = np.array([[1,3], [1,-1]])
b = np.array([4.43,-2.215])
x = np.linalg.solve(a, b)
vA2 = round(x[0],3)  #[meters per second]
vB2 = round(x[1],2) #[meters per second]
Energy_loss = round((1/2)*18*vB2**(2)+(1/2)*6*vA2**(2)-(1/2)*6*vA1**(2),2)  #[Joules]

# Result
print"Energy_loss = ",(Energy_loss),"J"

# Ex 15.10
from __future__ import division
import numpy as np
import math

# Variable Declaration
wB = 1.5  #[kilogram]
k = 800  #[Newton meter]

# Calculation
# Using Principle of conservation of energy
vB1 = round(math.sqrt((-wB*9.81*1.25+(1/2)*k*0.25**(2))/((1/2)*1.5)),2)  #[meters per second]
# Using Principle of coefficient of restitution
vB2 = 0.8*(0-2.97)+0  #[meters per second]
# Using Principle of conservation of energy
coeff = [400,-14.72,-18.94]
# Taking positive root
s3 = round(np.roots(coeff)[0],3)  #[meters]

# Result
print"s3 = ",(s3*1000),"mm"

# Ex 15.11
import numpy as np
import math
from __future__ import division
# Calculation
vAx1 = round(3*math.cos(math.pi*30/180),2)
vAy1 = round(3*math.sin(math.pi*30/180),2)
vBx1 = round(-1*math.cos(math.pi*45/180),2)
vBy1 = round(-1*math.sin(math.pi*45/180),2)
# Using Conservation of "x" Momentum and Coefficient of restitution
a = np.array([[1,2], [-1,1]])
b = np.array([1.18,2.48])
x = np.linalg.solve(a, b)
vAx2 = round(x[0],3)  #[meters per second]
vBx2 = round(x[1],2) #[meters per second]
# Using Conservation of "x" Momentum
vAy2 = vAy1
vBy2 = vBy1

# Result
print"vAx2 = ",(vAx2),"m/s"
print"vBx2 = ",(vBx2),"m/s"
print"vAy2 = ",(vAy2),"m/s"
print"vBy2 = ",(vBy2),"m/s"

# Ex 15.13
from __future__ import division

# Variable Declaration
P = 10  #[Newton]
wB = 5  #[kilogram]

# Calculation
vA2 = ((3/2)*(4**(2)-0**(2))+0.4*P*4)/(wB*0.4)  #[meters per second]

# Result
print"vA2 = ",(vA2),"m/s"

# Ex 15.14
from __future__ import division
import math

# Variable Declaration
v1 = 1  #[meters per second]
r1 = 0.5  #[meters]
r2 = 0.2  #[meters]
vC = 2  #[meters per second]

# Calculation
# Part(a)
# Using principle of Conservation of Angular Momentum
v2dash = (r1*0.5*v1)/(r2*0.5)  #[meters per second]
v2 = round(math.sqrt(2.5**(2)+2**(2)),2)  #[meters per second]
# Part(b)
UF = (1/2)*0.5*v2**(2)-(1/2)*0.5*v1**(2)  #[Joules]

# Result
print"UF = ",(UF),"J"

# Ex 15.15
import math
from __future__ import division

# Variable Declaration
vD1 = 1.5  #[meters per second]
kc = 20  #[Newtons per meter]

# Calculation
# Using principle of Conservation of Angular Momentum
vD2dash = (0.5*2*1.5)/(0.7*2)  #[meters per second]
# Using Conservation of Energy
vD2 = round(math.sqrt(((1/2)*2*vD1**(2)-(1/2)*kc*0.2**(2))/((1/2)*2)),2)  #[meters per second]
vD2doubledash = round(math.sqrt(vD2**(2)-vD2dash**(2)),3)  #[meters per second]

# Result
print"vD2doubledash = ",(vD2doubledash),"m/s"



