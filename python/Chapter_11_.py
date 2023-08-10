# Ex 11.1
import math
from __future__ import division

# Calculation
theta = round(math.degrees(math.atan(98.1/50)),1)  #[Degrees]

# Result
print"theta = ",(theta),"degrees"

# Ex 11.3
import math
import numpy as np

# Variable Declaration
theta = 45  #[Degrees]

# Calculation
coeff = [1, -1.2*math.cos(math.pi*theta/180), -0.13]
# Taking only the positive root
xc = round(np.roots(coeff)[0],3)  #[meter]
Cx = round(((-120*math.cos(math.pi*theta/180))*(1.2*math.cos(math.pi*theta/180)-2*xc))/(1.2*xc*math.sin(math.pi*theta/180)),0)  #[Newton]

# Result
print"xc = ",(xc),"m"
print"Cx = ",(Cx),"N"

# Ex 11.5
import math
from scipy.misc import derivative
from __future__ import division

# Variable Declaration
W = 10*9.81   #[Newton]
k = 200  #[Newton per meter]
l = 0.6  #[meter]

# Calculation

# Let V1 = d**(2)V/dtheta**(2) at theta = 0 
def f1(x):
    return (1/2)*k*(l**(2))*((1-math.cos(x))**(2))-(W*l/2)*(2-math.cos(x))
V1 = round(derivative(f1, 0*math.pi/180,dx=1e-6, n=2),1)

# Let V2 = d**(2)V/dtheta**(2) at theta = 53.8 
def f2(x):
    return (1/2)*k*(l**(2))*((1-math.cos(x))**(2))-(W*l/2)*(2-math.cos(x))
V2 = round(derivative(f2, 53.8*math.pi/180,dx=1e-6, n=2),1)
                     
# Result
print"theta = ",(theta),"degrees"
print"V1 = ",(V1),"(unstable equilibrium at theta = 0 degrees)"
print"V2 = ",round(V2,1),"(stable equilibrium at theta = 53.8 degrees)"                     

# Ex 11.6
import math
from __future__ import division
from scipy.misc import derivative

# Calculation
# From dV/dtheta at theta = 20 degrees
m = round(69.14/10.58,2)  #[kilogram]
def f(x):
    return 98.1*(1.5*math.sin(x)/2)-m*9.81*(1.92-math.sqrt(3.69-3.6*math.sin(x)))

# Let V = d**(2)V/dtheta**(2) at theta = 20 degrees
V = round(derivative(f, 20*math.pi/180,dx=1e-6, n=2),1)
# Result
print"m = ",(m),"kg"
print"V = ",(V)," (unstable equilibrium at theta = 20 degrees)"



