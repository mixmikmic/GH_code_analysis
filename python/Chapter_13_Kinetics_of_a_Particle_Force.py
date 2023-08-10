# Ex 13.1
import numpy as np
import math

# Variable Declaration
uk = 0.3
F = 400  #[Newton]

# Calculation
# Using  ΣF_x(right) = m*a_x and ΣF_y(upward) = m*a_y
a = np.array([[uk,50],[1,0] ])
b = np.array([F*math.cos(math.pi*30/180),490.5-F*math.sin(math.pi*30/180)])
x = np.linalg.solve(a, b)
NC = round(x[0],1)  #[Newton]
a =  round(x[1],1)  #[meters per second square]
v = 0+a*3  #[meter per second]

# Result
print"v = ",(v),"m/s"

# Ex 13.2
from __future__ import division
from scipy import integrate

# Calculation Part(a)
h = round(-(50**(2))/(2*-9.81),1)  #[meters]

# Result Part(a)
print"Part(a)"
print"h = ",(h),"m\n"

# Calculation Part(b)
v = lambda v: -v/(0.001*v**(2)+9.81)   
h = round(integrate.quad(v, 50, 0)[0],1)  #[meters]

# Result Part(b)
print"Part(b)"
print"h = ",(h),"m"

# Ex 13.3
from scipy import integrate
from __future__ import division

# Calculation
V = lambda t: 0.221*t
v = round(integrate.quad(V, 0, 2)[0],2)  #[meters per second]
# Using  ΣF_x(left) = m*a_x
T = round(40*2-(900/9.81)*0.221*2,1)  #[Newton]

# Result
print"v = ",(v),"m/s"
print"T = ",(T),"N"

# Ex 13.4
import numpy as np
import math

# Variable Declaration
k = 3  #[Newtons per meter]
ul = 0.75  #[meter] (ul is the unstretched length)
y = 1  #[meter]

# Calculation
Fs = round(k*(math.sqrt(y**(2)+0.75**(2))-ul),2)  #[Newton]
theta = round(math.degrees(math.atan(y/0.75)),1)  #[Degrees]
# Using  ΣF_x(right) = m*a_x and ΣF_y(downward) = m*a_y
a = np.array([[1,0],[0,2] ])
b = np.array([Fs*math.cos(math.pi*theta/180),19.62-Fs*math.sin(math.pi*theta/180)])
x = np.linalg.solve(a, b)
NC = round(x[0],3)  #[Newton]
a = round(x[1],2)  #[meters per second square]

# Result
print"NC = ",(NC),"N"
print"a = ",(a),"m/s**(2)"

# Ex 13.5
import numpy as np

# Calculation
# Using  ΣF_y(downward) = m*a_y (Block A), ΣF_y(downward) = m*a_y (Block B) and 2*aA = -aB
a = np.array([[2,100],[1,20*-2] ])
b = np.array([981,196.2])
x = np.linalg.solve(a, b)
T = round(x[0],1)  #[Newton]
aA = round(x[1],2)  #[meters per second square]
aB = -2*aA  #[meters per second square]
v = round(0+aB*2,1)  #[meters per second]

# Result
print"v = ",(v),"m/s"

# Ex 13.7
from __future__ import division

# Variable Declaration
uk = 0.1

# Calculation
# Using  ΣFb = 0
ND = 29.43  #[Newton]
# Using ΣFt = m*at
at = (0.1*ND)/3  #[meters per second square]
# Using ΣFn = m*an
vcr = round(math.sqrt(100/3),2)  #[meters per second]
t = round((vcr-0)/at,2)  #[seconds]

# Result
print"t = ",(t),"s"

# Ex 13.8
from __future__ import division

# Variable Declaration
v = 20  #[meters per second]

# Calculation
rho = ((1+0**(2))**(3/2))/(1/30)  #[meters]
# Using ΣFn = m*an
NA = round(700+(700/9.81)*(20**(2)/rho),1)  #[Newtons]
an = v**(2)/rho  #[meters per second square]
aA = round(an,2)   #[meters per second square]

# Result
print"NA = ",(NA),"N"
print"aA = ",(aA),"m/s**(2)"

# Ex 13.9

# Calculation
thetamax = round(math.degrees((9.81+1)/((19.62*0.5/2)+9.81)),1)  #[Degrees]

# Result
print"thetamax = ",(thetamax),"degrees"

# Ex 13.10
import numpy as np
import math
from __future__ import division

# Calculation
a = np.array([[math.cos(math.pi*14.04/180),-math.sin(math.pi*14.04/180)],[math.sin(math.pi*14.04/180),math.cos(math.pi*14.04/180)] ])
b = np.array([(2/32.2)*(6-3*0.5**(2)),(2/32.2)*(3*0+2*6*0.5)])
x = np.linalg.solve(a, b)
F = round(x[0],2)  #[Newton]
N = round(x[1],2)  #[Newton]

# Result
print"F = ",(F),"N"  #[Correction in the answer]
print"N = ",(N),"N"  #[Correction in the answer]

# Ex 13.11
import numpy as np
import math


# Variable Declaration
theta = 60  #[Degrees]

# Calculation
a = np.array([[0,-math.sin(math.pi*theta/180)],[1,-math.cos(math.pi*theta/180)] ])
b = np.array([2*(0.192-0.462*(0.5**(2)))-19.62*math.sin(math.pi*theta/180),2*(0+2*-0.133*0.5)-19.62*math.cos(math.pi*theta/180)])
x = np.linalg.solve(a, b)
FP = round(x[0],3)  #[Newton]
NC = round(x[1],1)  #[Newton]

# Result
print"FP = ",(FP),"N"  
print"NC = ",(NC),"N"  

# Ex 13.12
import numpy as np
import math
from __future__ import division

# Calculation
a = np.array([[0,math.cos(math.pi*17.7/180)],[1,-math.sin(math.pi*17.7/180)] ])
b = np.array([0.5*(0-0.1*math.pi*4**(2)),0.5*(0+2*0.4*4)])
x = np.linalg.solve(a, b)
FC = round(x[0],1)  #[Newton]
NC = round(x[1],2)  #[Newton]

# Result
print"FC = ",(FC),"N"  
print"NC = ",(NC),"N"  



