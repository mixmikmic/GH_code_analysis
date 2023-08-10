# Example 5.6
import math

# Calculation
# Summing forces in the x direction +ΣF_x(right) = 0
B_x = round(600*math.cos(math.pi*45/180),1)  #[Newton]

# Refer Fig 5-14b we have +ΣMB(counterclockwise) = 0
A_y = round((100*2+600*math.sin(math.pi*45/180)*5-600*math.cos(math.pi*45/180)*0.2)/7,1)   #[Newton]

# Summing forces in y direction +ΣF_y(upward) = 0
B_y = round(-319+600*math.sin(math.pi*45/180)+100+200,1)   #[Newton]

# Result
print"B_x = ",(B_x),"N"
print"A_y = ",(A_y),"N"
print"B_y = ",(B_y),"N"

# Example 5.7
import math
from __future__ import division

# Variable Declaration
F = 100  #[Newton]

# Calculation
# Summing moments about point A to eliminate A_x and A_y Refer Fig 5-15c
# +ΣMA(counterclockwise) = 0
T = round((F*0.5)/0.5,1)   #[Newton]

# Using result of T a force summation is applied to determine the components of reaction at pin A
# +ΣF_x(right) = 0
A_x = round(F*math.sin(math.pi*30/180),1)   #[Newton]

# +ΣF_y(upward) = 0
A_y = round(F+F*math.cos(math.pi*30/180),1)   #[Newton]
                      
# Result
print"T = ",(T),"N"
print"A_x = ",(A_x),"N"
print"A_y = ",(A_y),"N"
                       

# Example 5.8
import math

# Calculation
# Summing moments about A, we obtain direct solution for NB
# +ΣMA(counterclockwise) = 0
NB = (90+60)/0.75   #[Newton]

# +ΣF_x(right) = 0
A_x = round(NB*math.sin(math.pi*30/180),1)   #[Newton]

# +ΣF_y(upward) = 0
A_y = round(NB*math.cos(math.pi*30/180)+60,1)   #[Newton]

# Result
print"A_x = ",(A_x),"N"
print"A_y = ",(A_y),"N"

# Example 5.9
import math
from __future__ import division

# Calculation
# +ΣF_x(right) = 0
A_x = round(52*(5/13)-30*math.cos(math.pi*60/180),1)  #[Newton]
# +ΣF_y(upward) = 0
A_y = round(52*(12/13)+30*math.sin(math.pi*60/180),1)  #[Newton]
# +ΣMA(counterclockwise) = 0
MA = round(52*(12/13)*0.3+30*math.sin(math.pi*60/180)*0.7,1)  #[Newton meter]
FA = round(math.sqrt(A_x**(2)+A_y**(2)),1)  #[Newton]
theta = math.degrees(math.atan(A_y/A_x))  #[degrees]

# Result
print"A_x = ",(A_x),"N"
print"A_y = ",(A_y),"N"
print"MA = ",(MA),"N.m"
print"FA = ",(FA),"N"

# Example 5.10
import math
import numpy as np

# Calculation
# Refer Fig 5-18b
# Using +ΣF_y(upward) = 0 and +ΣMA(counterclockwise) = 0
a = np.array([[math.cos(math.pi*30/180),math.cos(math.pi*30/180)], [6,2]])
b = np.array([300,4000+300*math.cos(math.pi*30/180)*8])
x = np.linalg.solve(a, b)
C_y_dash = round(x[0],1)  #[Newton]
B_y_dash = round(x[1],1)  #[Newton]

# Using +ΣF_x(right) = 0
A_x = round(C_y_dash*math.sin(math.pi*30/180)+B_y_dash*math.sin(math.pi*30/180),1)  #[Newton]

# Result
print"C_y_dash = ",(C_y_dash/1000),"kN"
print"B_y_dash = ",(B_y_dash/1000),"kN"
print"A_x = ",(A_x),"N"

# Example 5.11
import math
import numpy as np

# Calculation
# Since ΣMO = 0 angle theta which defines the line of action of FA can be determined by trigonometry
theta = round(math.degrees(math.atan(0.7/0.4)),1)  #[Degrees]

# Using +ΣF_x(right) = 0 and +ΣF_y(upward) = 0
a = np.array([[math.cos(math.pi*theta/180),-math.cos(math.pi*45/180)], [math.sin(math.pi*theta/180),-math.sin(math.pi*45/180)]])
b = np.array([-400,0])
x = np.linalg.solve(a, b)
FA = round(x[0]/1000,2)  #[Newton]
F = round(x[1]/1000,2)  #[Newton]

# Result
print"theta = ",(theta),"degrees"
print"FA = ",(FA),"kN"
print"F = ",(F),"kN"

# Example 5.13

# Calculation
# Using ΣF_x = 0ΣF_z = 0
B_x = 0  #[Newton]
# Using ΣF_y= 0
B_y = 0  #[Newton]
# Using ΣF_z = 0, A_z + B_z + TC = 300 + 981(1)
# Using ΣM_x = 0, 2TC + 2B_z = 981(2)
# Using ΣM_y = 0, 3B_z + 3A_z = 300(1.5) + 981(1.5) - 200(3)
# Solving (1),(2) and (3)
a = np.array([[1,1,1], [0,2,2],[3,3,0] ])
b = np.array([300+981,981,300*1.5+981*1.5-200])
x = np.linalg.solve(a, b)
A_z = round(x[0],1)  #[Newton]
B_z = round(x[1],1)  #[Newton]
TC = round(x[2],1)  #[Newton]

# Result
print"B_x = ",(B_x),"N"
print"B_y = ",(B_y),"N"
print"A_z = ",(A_z),"N"
print"B_z = ",(B_z),"N"
print"TC = ",(TC),"N"

# Example 5.14
import math 

# Calculation
# Using right hand rule and assuming positive moments act in +i direction, ΣM_x = 0
P = round((981*0.1)/(0.3*math.cos(math.pi*30/180)),1)   #[Newton]

# Using this result for P and summing moments about y and z axis, ΣM_y = 0 and  ΣM_z = 0
A_z = round((981*0.5-P*0.4)/0.8,1)   #[Newton]
A_y = -0/0.8   #[Newton]

# The reactions at B are determined by using ΣF_x = 0, ΣF_y = 0 and ΣF_z = 0
A_x = 0   #[Newton]
B_y = 0   #[Newton]
B_z = round(P+981-A_z,1)   #[Newton]

# Result
print"P = ",(P),"N"
print"A_z = ",(A_z),"N"
print"B_z = ",(B_z),"N"

# Example 5.15
import numpy as np
from __future__ import division

# Calculation
# Summing moments about point A ΣMA = 0, rB X (F+TC+TD)
# Evaluating cross product
a = np.array([[0,-4], [4.24,-2]])
b = np.array([-6000,0])
x = np.linalg.solve(a, b)
TC = round(x[0],1)   #[Newton]
TD = round(x[1],1)   #[Newton]

# Using ΣF_x = 0
A_x = round(-0.707*TC+(3/9)*TD,0)   #[Newton]

# Using ΣF_y = 0   
A_y = round(1000-(6/9)*TD,1)   #[Newton]

# Using ΣF_z= 0
A_z = round(0.707*TC+(6/9)*TD,1)   #[Newton]
              
# Result
print"TC = ",(TC),"N"  
print"TD = ",(TD),"N" 
print"A_x = ",(A_x),"N"  
print"A_y = ",(A_y),"N"                
print"A_z = ",(A_z),"N"                

# Example 5.16
from __future__ import division

# Calculation
# Summing moments about point A, ΣMA = 0 and rC X F + rB X (TE + TD) = 0
# Using ΣM_x = 0 and ΣM_y = 0
TD = 200/2   #[Newton]
TE = 100/2   #[Newton]

# Using ΣF_x = 0, ΣF_y = 0 and ΣF_z = 0
A_x = -TE   #[Newton]
A_y = -TD   #[Newton]
A_z = 200   #[Newton]

# Result
print"TD = ",(TD),"N"
print"TE = ",(TD),"N"
print"A_x = ",(A_x),"N"
print"A_y = ",(A_y),"N"
print"A_z = ",(A_z),"N"

# Example 5.17

# Calculation
# Using u.(rB X TB + rE X W)
TB = round(490.5/0.857,1)   #[Newton]

print"TB = ",(TB),"N"



