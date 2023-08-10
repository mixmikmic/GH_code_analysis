# Example 3.2
import math
import numpy as np

# Variable Declaration
theta = 30  #[Degrees]

# Calculation
# The two unknown magnitudes TB and TD can be obtained from two scalar equations of equilibrium ΣF_x = 0 and ΣF_y = 0
a = np.array([[math.cos(math.pi*theta/180),-1], [math.sin(math.pi*theta/180),0]])
b = np.array([0,2.452])
x = np.linalg.solve(a, b)
TB = round(x[0],2)  #[kilo Newton]
TD = round(x[1],2)  #[kilo Newton]

# Result
print"TB = ",(TB),"kN"
print"TD = ",(TD),"kN"

# Example 3.3
import math
import numpy as np
from __future__ import division

# Variable Declaration
WA = 20   #[Newton]

# Calculation
# The two unknown magnitudes TEG and TEC can be obtained from two scalar equations of equilibrium ΣF_x = 0 and ΣF_y = 0 at point E
a = np.array([[math.sin(math.pi*30/180),-math.cos(math.pi*45/180)], [math.cos(math.pi*30/180),-math.sin(math.pi*45/180)]])
b = np.array([0,20])
x = np.linalg.solve(a, b)
TEG = round(x[0],1)  #[Newton]
TEC = round(x[1],1)  #[Newton]

# The two unknown magnitudes TCD and WB can be obtained from two scalar equations of equilibrium ΣF_x = 0 and ΣF_y = 0 at point C
a = np.array([[-(4/5),0], [3/5,-1]])
b = np.array([-38.6*math.cos(math.pi*45/180), -38.6*math.sin(math.pi*45/180)])
x = np.linalg.solve(a, b)
TCD = round(x[0],1)  #[Newton]
WB = round(x[1],1)  #[Newton]

# Result
print"TB = ",(TEG),"N"
print"TD = ",(TEC),"N"
print"TCD = ",(TCD),"N"
print"WB = ",(WB),"N"

# Example 3.4
import math
import numpy as np
from __future__ import division

# Variable Declaration
lAB_dash = 0.4  #[meter]
kAB = 300 #[Newton/meter]

# Calculation
# Let W be weight of lamp
W = 8*9.81  #[Newton]

# The two unknown magnitudes TAB and TAC can be obtained from two scalar equations of equilibrium ΣF_x = 0 and ΣF_y = 0 at point A
a = np.array([[1,-math.cos(math.pi*30/180)], [0,math.sin(math.pi*30/180)]])
b = np.array([0, 78.5])
x = np.linalg.solve(a, b)
TAB = round(x[0],1)  #[Newton]
TAC = round(x[1],1)  #[Newton]

# Let sAB denote the stretch of string
sAB = TAB/kAB

# Let lAB denote the stretch of string
lAB = lAB_dash + sAB
lAC = round((2 - lAB)/math.cos(math.pi*30/180),2)

# Result
print"lAC = ",(lAC),"m"

# Example 3.5
import math
import numpy as np
from __future__ import division

# Variable Declaration
L = 90  #[Newton]
k = 5000  #[Newton/meter]

# Calculation
# The three unknown magnitudes FB, FC and FD can be obtained from three scalar equations of equilibrium ΣF_x = 0,ΣF_y = 0 and ΣF_z = 0 at point A
a = np.array([[0,-4/5,math.sin(math.pi*30/180)], [1,0,-math.cos(math.pi*30/180)], [0,3/5,0]])
b = np.array([0, 0, 90])
x = np.linalg.solve(a, b)
FB = round(x[0],1)  #[Newton]
FC = round(x[1],1)  #[Newton]
FD = round(x[2],1)  #[Newton]

# Let sAB denote stretch of a string
sAB= round((FB*1000)/k,1)  #[millimeter]

# Result
print"FB = ",(FB),"N"
print"FC = ",(FC),"N"
print"FD = ",(FD),"N"
print"sAB = ",(sAB),"mm"

# Example 3.6
import math
from __future__ import division

# Variable Declaration
B_x = -2  #[meter]
B_y = -3  #[meter]
B_z = 6  #[meter]
F1_x = 0 #[Newton]
F1_y = 400  #[Newton]
F1_z = 0  #[Newton]
F2_x = 0  #[Newton]
F2_y = 0  #[Newton]
F2_z = -800  #[Newton]
F3 = 700  #[Newton]

# Calculation
# Let rB be unit vector along OB
rB_x = -2
rB_y = -3
rB_z = 6
rB = math.sqrt(rB_x**(2)+rB_y**(2)+rB_z**(2))
F3_x = F3*(rB_x/rB)  #[Newton]
F3_y = F3*(rB_y/rB)  #[Newton]
F3_z = F3*(rB_z/rB)  #[Newton]

# For equilibrium ΣF = 0 and F1+F2+F3+F = 0
F_x = 0 - (F1_x +F2_x +F3_x)  #[Newton]
F_y = 0 - (F1_y +F2_y +F3_y)  #[Newton]
F_z = 0 - (F1_z +F2_z +F3_z)  #[Newton]
F = round(math.sqrt(F_x**(2)+F_y**(2)+F_z**(2)),1)  #[Newton]
alpha = round(math.degrees(math.acos(F_x/F)),1)  #[Degrees]
beta = round(math.degrees(math.acos(F_y/F)),1)  #[Degrees]
gamma = round(math.degrees(math.acos(F_z/F)),1)  #[Degrees]


# Result
print"F = ",(F),"N"
print"alpha = ",(alpha),"degrees"
print"beta = ",(beta),"degrees"
print"gamma = ",(gamma),"degrees"

# Example 3.7
import math
import numpy as np
from __future__ import division

# Variable Declaration
B_x = -3  #[meter]
B_y = -4  #[meter]
B_z = 8  #[meter]
C_x = -3  #[meter]
C_y = 4  #[meter]
C_z = 8  #[meter]

# Calculation
# let FB_x/FB = b_x, FB_y/FB = b_y, FB_z/FB = b_z 
b_x = round(B_x/(math.sqrt(B_x**(2)+B_y**(2)+B_z**(2))),3)  #[Newton]
b_y = round(B_y/(math.sqrt(B_x**(2)+B_y**(2)+B_z**(2))),3)  #[Newton]
b_z = round(B_z/(math.sqrt(B_x**(2)+B_y**(2)+B_z**(2))),3)  #[Newton]

# let FC_x/FC = c_x, FC_y/FC = c_y, FC_z/FC = c_z 
c_x = round(C_x/(math.sqrt(C_x**(2)+C_y**(2)+C_z**(2))),3)  #[Newton]
c_y = round(C_y/(math.sqrt(C_x**(2)+C_y**(2)+C_z**(2))),3)  #[Newton]
c_z = round(C_z/(math.sqrt(C_x**(2)+C_y**(2)+C_z**(2))),3)  #[Newton]

# let FD_x/FD = d_x, FD_y/FD = d_y, FD_z/FD = d_z 
d_x = 1
d_y = 0
d_z = 0

W_x = 0
W_y = 0
W_z = -40

# The three unknown magnitudes FB, FC and FD can be obtained from three scalar equations of equilibrium ΣF_x = 0,ΣF_y = 0 and ΣF_z = 0 at point A
a = np.array([[b_x,c_x,d_x], [b_y,c_y,d_y], [b_z,c_z,d_z]])
b = np.array([0, 0, -W_z])
x = np.linalg.solve(a, b)
FB = round(x[0],1)  #[Newton]
FC = round(x[1],1)  #[Newton]
FD = round(x[2],1)  #[Newton]

# Result
print"FB = ",(FB),"N"
print"FC = ",(FC),"N"
print"FD = ",(FD),"N"

# Ex 3.8
import math
import numpy as np
from __future__ import division

# Variable Declaration
W = 981  #[Newton]
D_x = -1   #[meter]
D_y = 2   #[meter]
D_z = 2   #[meter]
k = 1500   #[Newton meter]

# Calculation
# let FB_x/FB = b_x, FB_y/FB = b_y, FB_z/FB = b_z 
b_x = 1  
b_y = 0
b_z = 0

# let FC_x/FC = c_x, FC_y/FC = c_y, FC_z/FC = c_z 
c_x = math.cos(120*math.pi/180)  #[Newton]
c_y = math.cos(135*math.pi/180)  #[Newton]
c_z = math.cos(60*math.pi/180)  #[Newton]

# let FD_x/FD = d_x, FD_y/FD = d_y, FD_z/FD = d_z 
d_x = round(D_x/(math.sqrt(D_x**(2)+D_y**(2)+D_z**(2))),3)  #[Newton]
d_y = round(D_y/(math.sqrt(D_x**(2)+D_y**(2)+D_z**(2))),3)  #[Newton]
d_z = round(D_z/(math.sqrt(D_x**(2)+D_y**(2)+D_z**(2))),3)  #[Newton]

W_x = 0  #[Newton]
W_y = 0  #[Newton]
W_z = -981  #[Newton]

# The three unknown magnitudes FB, FC and FD can be obtained from three scalar equations of equilibrium ΣF_x = 0,ΣF_y = 0 and ΣF_z = 0 at point A
a = np.array([[b_x,c_x,d_x], [b_y,c_y,d_y], [b_z,c_z,d_z]])
b = np.array([0, 0, -W_z])
x = np.linalg.solve(a, b)
FB = round(x[0],1)  #[Newton]
FC = round(x[1],1)  #[Newton]
FD = round(x[2],1)  #[Newton]

# Let stretch of spring be denoted by s
s = round(FB/k,3)

# Result
print"FB = ",(FB),"N"
print"FC = ",(FC),"N"
print"FD = ",(FD),"N"
print"s = ",(s),"m"



