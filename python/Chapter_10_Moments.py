# Ex 10.2
from scipy import integrate
from __future__ import division

# Calculation Solution 1
a = lambda y: (y**2)*(100-(y**2)/400)
I_x = round(integrate.quad(a, 0, 200)[0],0)   #[millimeter**(4)]

# Result Solution 1
print"Solution 1"
print"I_x = ",(I_x),"mm**(4)\n"

# Calculation Solution 2
a = lambda x: (1/3)*(400*x)**(3/2)
I_x = round(integrate.quad(a, 0, 100)[0],0)  #[millimeter**(4)]

# Result Solution 2
print"Solution 2"
print"I_x = ",(I_x),"mm**(4)"

# Ex 10.4

from scipy import integrate
from __future__ import division
import math

# Calculation Solution 1
a = lambda y: y**(2)*(math.sqrt(y)-y)
I_x = round(integrate.quad(a, 0, 1)[0],4)  #[meter**(4)]

# Result Solution 1
print"Solution 1"
print"I_x = ",(I_x),"m**(4)\n"

# Calculation Solution 2
a = lambda x: (1/3)*(x**(3)-x**(6))
I_x = round(integrate.quad(a, 0, 1)[0],4)  #[meter**(4)]

# Result Solution 2
print"Solution 2"
print"I_x = ",(I_x),"m**(4)"

# Ex 10.5
import math
from __future__ import division

# Calculation
# Using Parallel Axis Theorem
# Circle
I_xc = (1/4)*math.pi*25**(4)+math.pi*25**(2)*75**(2)  #[millimeter**(4)]
# Rectangle
I_xr = (1/12)*100*150**(3)+100*150*75**(2)  #[millimeter**(4)]
# Let I_x be moment of inertia for composite area
I_x = -I_xc+I_xr  #[millimeter**(4)]

# Result
print"I_x = ",(I_x),"mm**(4)"

# Ex 10.6
import math
from __future__ import division

# Calculation
# Using Parallel axis theorem
# Rectangle A
I_xA = (1/12)*100*300**(3)+100*300*200**(2)  #[millimeter**(4)]
I_yA = (1/12)*300*100**(3)+100*300*250**(2)  #[millimeter**(4)]
# Rectangle B
I_xB = (1/12)*600*100**(3)  #[millimeter**(4)]
I_yB = (1/12)*100*600**(3)  #[millimeter**(4)]
# Rectangle D
I_xD = (1/12)*100*300**(3)+100*300*200**(2)  #[millimeter**(4)]
I_yD = (1/12)*300*100**(3)+100*300*250**(2)  #[millimeter**(4)]
I_x = I_xA+I_xB+I_xD  #[millimeter**(4)]
I_y = I_yA+I_yB+I_yD  #[millimeter**(4)]

# Result
print"I_x = ",(I_x),"mm**(4)"
print"I_y = ",(I_y),"mm**(4)"

# Ex 10.8

# Calculation
# Rectangle A
I_xyA = 0+300*100*(-250)*(200)
# Rectangle B
I_xyB = 0+0
# Rectangle D
I_xyD = 0+300*100*(250)*(-200)
# Let I_xy be product of inertia for entire cross section
I_xy = I_xyA+I_xyB+I_xyD

# Result
print"I_xy = ",(I_xy),"mm**(4)"

# Ex 10.10
from scipy import integrate
from __future__ import division
import math

# Variable Declaration
rho = 5  #[milligram per meter cube]

# Calculation 
a = lambda y: ((5*math.pi)/2)*y**(8)
I_y = round(integrate.quad(a, 0, 1)[0],3)  #[kg meter square]

# Result
print"I_y = ",(I_y),"kg.m**(2)"



# Ex 10.11
import math

# Variable Declaration
rho = 8000  #[kg meter**(2)]
t = 0.01  #[meter]

# Calculation
# Disk
IOd = (1/2)*15.71*0.25**(2)+15.71*0.25**(2)  #[kg meter**(2)]
# Hole
IOh = (1/2)*3.93*0.125**(2)+3.93*0.25**(2)  #[kg meter**(2)]
IO = round(IOd-IOh,2)  #[kg meter**(2)]
# Result
print"IO = ",(IO),"kg.m**(2)"

# Ex 10.12
import math
from __future__ import division

# Variable Declaration
W = 3  #[kilogram]

# Calculation
# Rod OA
IOAO = (1/3)*W*(2**(2))  #[kg meter**(2)]
# Rod BC
IBCO = (1/12)*W*(2**(2))+W*(2**(2))  #[kg meter**(2)]
# Let IO be moment of inertia of pendulum about O
IO = round(IOAO+IBCO,1)  #[kg meter**(2)]
ybar = ((1*W+2*W)/(3+3))  #[meter]
IG = IO-2*W*ybar  #[kg meter**(2)]

# Result
print"IO = ",(IO),"kg.m**(2)"
print"IG = ",(IG),"kg.m**(2)"



