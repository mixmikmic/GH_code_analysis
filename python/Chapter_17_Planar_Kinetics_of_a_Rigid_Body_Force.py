# Ex 17.2
from scipy import integrate
import math

# Calculation
y = lambda y: ((math.pi*2)/2)*y**(8)
Iy = round(integrate.quad(y, 0, 1)[0],3)  #[milligram meter square]

# Result
print"Iy = ",(Iy),"Mg.m**(2)"

# Ex 17.3
from __future__ import division
import math

# Variable Declaration
d = 8000  #[kilogram per meter cube]
t = 0.01  #[meter]

# Calculation
md = round(d*math.pi*0.25**(2)*t,2)
# Disk
IdO = (1/2)*md*0.25**(2)+md*0.25**(2)  #[kilogram meter square]

# Hole
IhO = (1/2)*3.93*0.125**(2)+3.93*0.25**(2)  #[kilogram meter square]

# Let IO be moment of inertia about point O
IO = round(IdO-IhO,2)  #[kilogram meter square]

# Result
print"IO = ",(IO),"kg m**(2)"

# Ex 17.4
from __future__ import division

# Variable Declaration
m = 10  #[kilogram]

# Calculation
# Part(a)
IOAO = (1/3)*m*1**(2)  #[kilogram meter square]
IBCO = (1/12)*m*1**(2)+m*1**(2)  #[kilogram meter square]
# Let IO be moment of inertia about O
IO = round(IOAO+IBCO,3)  #[kilogram meter square]

# Part(b)
ybar = (0.5*10+1*10)/(10+10)  #[meter]
IG = round(IO-2*m*ybar**(2),3)  #[kilogram meter square]

# Result
print"IO = ",(IO),"kg.m**(2)"
print"IG = ",(IG),"kg.m**(2)"

# Ex 17.5
import numpy as np

# Calculation
# Using +ΣF_x = m(aG)_x, +ΣF_y = m(aG)_y and +ΣMG(counter clockwise)=0 
a = np.array([[0,-0.25,2000],[1,1,0],[-1.25,-0.25*0.3+0.75,0]])
b = np.array([0,2000*9.81,0])
x = np.linalg.solve(a, b)
NA = round(x[0]/1000,2)  #[kilo Newton]
NB = round(x[1]/1000,2)  #[kilo Newton]
aG = round(x[2],2)  #[meters per second square]

# Result
print"NA = ",(NA),"kN"
print"NB = ",(NB),"kN"
print"aG = ",(aG),"m/s**(2)"

# Ex 17.6
import numpy as np

# Variable Declaration
mm = 125  #[kilogram]
mr = 75  #[kilogram]

# Calculation
# Using +ΣF_x = m(aG)_x, +ΣF_y = m(aG)_y and +ΣMG(counter clockwise)=0 
a = np.array([[1,0,-(mm+mr)],[0,1,0],[0,0,mr*0.9+mm*0.6]])
b = np.array([0,735.75+1226.25,-735.75*0.4-1226.25*0.8])
x = np.linalg.solve(a, b)
FB = round(x[0],1)  #[kilo Newton]
NB = round(x[1],1)  #[kilo Newton]
aG = round(x[2],2)  #[meters per second square]
# Let usmin be minimum coefficient of static friction
usmin = round(-FB/NB,3)

# Result
print"usmin = ",(usmin)

# Ex 17.7

# Variable Declaration
P = 600  #[Newton]
uk = 0.2

# Calculation
# Using +ΣF_y = m(aG)_y
NC = 490.5  #[Newton]

# Using +ΣF_x = m(aG)_x
aG = round((600-uk*NC)/50,1)  #[meters per second square]

# Using +ΣMG(counter clockwise)=0 
x = round((uk*NC*0.5+600*0.3)/NC,3)  #[meter]

# Result
print"NC = ",(NC),"N"
print"x = ",(x),"m"
print"aG = ",(aG),"m/s**(2)"

# Ex 17.8
import numpy as np
import math

# Variable Declaration
theta = 30  #[Degrees]

# Calculation
# Using +ΣFn = 0, +ΣFt = 0 and +ΣMG(counterclockwise) = 0
a = np.array([[1,1,0],[0,0,100],[-math.cos(math.pi*theta/180)*0.4,0.4*math.cos(math.pi*theta/180),0]])
b = np.array([981*math.cos(math.pi*theta/180)+100*18,981*math.sin(math.pi*theta/180),0])
x = np.linalg.solve(a, b)
TB = round(x[0]/1000,2)  #[kilo Newton]
TD = round(x[1]/1000,2)  #[kilo Newton]
aG = round(x[2],2)  #[meters per second square]

# Result
print"TB = ",(TB),"kN"
print"TD = ",(TD),"kN"
print"aG = ",(aG),"m/s**(2)"

# Ex 17.9

# Calculation
# Using +ΣFx(right) = m(aG)x
Ox = 0  #[Newton]

# Using +ΣFy(upward) = m(aG)y
Oy = 294.3+10  #[Newton]

# Using +ΣMO(counterclockwise) = IO*alpha
alpha = (-10*0.2-5)/-0.6
theta = (-20**(2))/(2*-11.7)
theta = round(theta*(1/(2*math.pi)),2)

# Result
print"Ox = ",(Ox),"N"
print"Oy = ",(Oy),"N"
print"theta = ",(theta),"rev" 

# Ex 17.10
import numpy as np
from __future__ import division

# Calculation Solution 1
# Using +ΣFn(left) = mw**(2)rG, +ΣFt(downwards) = malpharG and +ΣMG(clockwise) = IGalpha
a = np.array([[1,0,0],[0,1,20*1.5],[0,1.5,-(1/12)*20*3**(2)]])
b = np.array([20*5**(2)*1.5,20*9.81,-60])
x = np.linalg.solve(a, b)
On = round(x[0],2)  #[Newton]
Ot = round(x[1],2)  #[Newton]
alpha = round(x[2],2)  #[radians per second square]


# Result Solution 1
print"Solution 1"
print"On = ",(On),"N"
print"Ot = ",(Ot),"N"
print"alpha = ",(alpha),"rad/s**(2)\n"

# Calculation Solution 2
# Using +ΣMO(clockwise) = Σ(Mk)O
alpha = round((60+20*9.81*1.5)/((1/12)*20*3**(2)+20*1.5*1.5),2)  #[radians per second square]

# Result Solution 2
print"Solution 2"
print"alpha = ",(alpha),"rad/s**(2)\n"

# Calculation Solution 3
# Using +ΣMO(clockwise) = IOalpha
alpha = round((60+20*9.81*1.5)/((1/3)*20*3**(2)),2)  #[radians per second square]

# Result Solution 3
print"Solution 3"
print"alpha = ",(alpha),"rad/s**(2)\n"

# Ex 17.11
import numpy as np

# Variable Declaration
m = 60  #[kilogram]
k = 0.25  #[meters]

# Calculation Solution 1
IO = m*k**(2)
# Using +ΣMO(counterclockwise) = IOalpha, +ΣFy(upward) = m(aG)y and +a(counterclockwise) = alpha*r
a = np.array([[0.4,0,-IO],[1,20,0],[0,1,-0.4]])
b = np.array([0,20*9.81,0])
x = np.linalg.solve(a, b)
T = round(x[0],2)  #[Newton]
a = round(x[1],2)  #[meters per second square]
alpha = round(x[2],1)  #[radians per second square]

# Result Solution 1
print"Solution 1"
print"alpha = ",(alpha),"rad/s**(2)\n"

# Calculation Solution 2
# Using +ΣMO(clockwise) = Σ(Mk)O
alpha = round((20*9.81*0.4)/(3.75+20*0.4*0.4),1)   #[radians per second square]


# Result Solution 2
print"Solution 2"
print"alpha = ",(alpha),"rad/s**(2)"

# Ex 17.12
import numpy as np

# Calculation
# Using +ΣFn(left) = mw**(2)rG, +ΣFt(upward) = m*alpha*rG and +ΣMG(clockwise) = IG*alpha
a = np.array([[1,0,0],[0,1,50*0.5],[0,0.5,-18]])
b = np.array([50*8**(2)*0.5,50*9.81,-80])
x = np.linalg.solve(a, b)
On = round(x[0],2)  #[Newton]
Ot = round(x[1],2)  #[Newton]
alpha = round(x[2],2)  #[radians per second square]


# Result
print"On = ",(On),"N"
print"Ot = ",(Ot),"N"  # Correction in the textbook
print"alpha = ",(alpha),"rad/s**(2)"

# Ex 17.14
import numpy as np

# Calculation Solution 1
IG = 8*0.35**(2)  #[kilogram meter square]
# Using +ΣFy(upward) = m(aG)y, +ΣMG(clockwise) = IG*alpha and +aG(clockwise) = alpha*r
a = np.array([[1,-8,0],[0.5,0,IG],[0,1,-0.5]])
b = np.array([-100+78.48,100*0.2,0])
x = np.linalg.solve(a, b)
T = round(x[0],2)  #[Newton]
aG = round(x[1],2)  #[meters per second square]
alpha = round(x[2],1)  #[radians per second square]

# Result Solution 1
print"Solution 1"
print"alpha = ",(alpha),"rad/s**(2)\n"

# Calculation Solution 2
# Using +ΣMA(clockwise) = Σ(Mk)A and aG = 0.5*alpha
alpha = round((100*0.7-78.48*0.5)/(0.980+8*0.5*0.5),1)  #[radians per second square]

# Result Solution 2
print"Solution 2"
print"alpha = ",(alpha),"rad/s**(2)"

# Ex 17.15
import numpy as np

# Calculation
# Slipping
# Using +ΣFx(right) = m(aG)x, +ΣFy(upward) = m(aG)y, +ΣMG(clockwise) = IG*alpha and FA = 0.25*NA
a = np.array([[1,0,-5,0],[0,1,0,0],[1.25,0,0,2.45],[1,-0.25,0,0]])
b = np.array([0,5*9.81,35,0])
x = np.linalg.solve(a, b)
FA = round(x[0],2)  #[Newton]
NA = round(x[1],2)  #[Newton]
aG = round(x[2],3)  #[meters per second square]
alpha = round(x[3],2)  #[radians per second square]

# Result
print"aG = ",(aG),"m/s**(2)"

# Ex 17.16
import numpy as np

# Calculation
# Slipping
# Using +ΣFx(right) = m(aG)x, +ΣFy(upward) = m(aG)y, +ΣMG(clockwise) = IG*alpha and FA = 0.25*NA
a = np.array([[1,0,100,0],[0,1,0,0],[1.5,0,0,-75],[1,-0.25,0,0]])
b = np.array([400,981,400,0])
x = np.linalg.solve(a, b)
FA = round(x[0],2)  #[Newton]
NA = round(x[1],2)  #[Newton]
aG = round(x[2],3)  #[meters per second square]
alpha = round(x[3],3)  #[radians per second square]

# Result
print"alpha = ",(alpha),"rad/s**(2)"

# Ex 17.17
import numpy as np

# Calculation
# Using +ΣMA(counterclockwise) = Σ(Mk)A, aGx = alpha*0.25 and aGy = alpha*0.1
a = np.array([[0.675,30*0.25,30*0.1],[0.25,-1,0],[0.1,0,-1]])
b = np.array([30*9.81*0.1,0,0])
x = np.linalg.solve(a, b)
alpha = round(x[0],1)  #[radians per second square]
aGx = round(x[1],2)  #[meters per second square]
aGy = round(x[2],2)  #[meters per second square]

# Result
print"alpha = ",(alpha),"rad/s**(2)"



