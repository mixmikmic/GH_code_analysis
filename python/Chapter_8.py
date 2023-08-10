# Example 8.1
import math

# Variable Declaration
P = 80  #[Newton]

# Calculation
# Using +ΣF_x(right) = 0+ΣMO(counterclockwise) = 0
F = round(80*math.cos(math.pi*30/180),1)  #[Newton]

# Using +ΣF_y(upward) = 0
NC = round(80*math.sin(math.pi*30/180)+196.2,1)  #[Newton]

# Using +ΣMO(counterclockwise) = 0
x = (80*math.cos(math.pi*30/180)*0.2-80*math.sin(math.pi*30/180)*0.4)/NC  #[meter]

# Result
print"F = ",(F),"N"
print"NC = ",(NC),"N"
print"x = ",(round(x*1000,2)),"mm"

# Ex 8.2
import math

# Calculation
# W*sin25 = us(W*cos25)
us = round(math.tan(math.pi*25/180),3)

# Result
print"us = ",(us)

# Ex 8.3
import numpy as np

# Calculation
coeff = [1, -4.619, 1]
us = np.roots(coeff)

# Result
# Finding the smallest root
print"us = ",(round(min(us),3))

# Ex 8.4
import math

# Calculation
# Using +ΣF_x(right) = 0,FA = F and NA = N for bottom pipe
# us_min = F/N
us_min = round(math.sin(math.pi*30/180)/(1+math.cos(math.pi*30/180)),3)
# Let smallest required coefficient of static friction be ug_min
# ug_min = F/NC
ug_min = round(0.2679*0.5/1.5,4)

# Result
print"us_min = ",(us_min)
print"ug_min = ",(ug_min)

# Ex 8.5
import numpy as np

# Variable Declaration
uB = 0.2
uC = 0.5

# Calculation
# Post slips only at B
# FB = uB*NB
FB = uB*400  #[Newton]
# Using +ΣMC(counterclockwise) = 0
P = FB/0.25  #[Newton]
# Using +ΣF_y(upward) = 0
NC = 400  #[Newton]
# Using +ΣF_x(right) = 0
FC = P-FB  #[Newton]

# Post slips only at C
FC = uC*NC  #[Newton]
# Using +ΣF_x(right) = 0 and # Using +ΣMC(counterclockwise) = 0
a = np.array([[1,-1],[-0.25,1]])
b = np.array([200,0])
x = np.linalg.solve(a, b)
P = round(x[0],1)  #[Newton]
FB = round(x[1],1)  #[Newton]

# Result
print"P = ",(P),"N"
print"NC = ",(NC),"N"
print"FC = ",(FC),"N"
print"FB = ",(FB),"N"

# Ex 8.6
import math
import numpy as np

# Variable Declaration
usA = 0.15
usB = 0.4

# Calculation

# Pipe rolls up incline
# Using +ΣF_x(right) = 0, +ΣMO(counterclockwise) = 0 and FB = 0.4*P
a = np.array([[-1,1],[-400,0.4*400]])
b = np.array([981*math.sin(math.pi*20/180),0])
x = np.linalg.solve(a, b)
FA = round(x[0],1)  #[Newton]
P = round(x[1],1)  #[Newton]
FB = FA  #[Newton]
NA = round(FB+981*math.cos(math.pi*20/180),1)  #[Newton]
P = round(981*math.sin(math.pi*20/180)+FA,1)  #[Newton]

# Pipe slides up incline
# Using +ΣMO(counterclockwise) = 0 and FA = 0.15*NA
a = np.array([[-0.15*400,400],[1,-1]])
b = np.array([0,981*math.cos(math.pi*20/180)])
x = np.linalg.solve(a, b)
NA = round(x[0],1)  #[Newton]
FB = round(x[1],1)  #[Newton]
FA = FB  #[Newton]
P = round(FA+981*math.sin(math.pi*20/180),1)  #[Newton]

# Result
print"NA = ",(NA),"N"
print"FA = ",(FA),"N"
print"FB = ",(FB),"N"
print"P = ",(P),"N"

# Ex 8.8
import math
from __future__ import division

# Variable Declaration
us = 0.25
W = 2000  #[Newton]
r = 5  #[millimeter]

# Calculation
phi_s = round(math.degrees(math.atan(us)),2)
theta = round(math.degrees(math.atan(2/(2*math.pi*5))),2)
M = 2*W*r*math.tan(math.pi*(phi_s+theta)/180)  #[Newton millimeter]

# Result
print"M = ",round(M/1000,2),"N.m"

# Ex 8.9
import math
from __future__ import division

# Variable Declaration
us = 0.25

# Calculation
T1 = round(500/(math.exp(us*(3/4)*math.pi)),1)
W = round(T1/(math.exp(us*(3/4)*math.pi)),1)
m = round(W/9.81,1)

# Result
print"m = ",(m),"kg"



