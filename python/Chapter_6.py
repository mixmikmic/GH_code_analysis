# Example 6.1
import math

# Calculation

# Joint B
# Using  +ΣF_x(right) = 0
FBC = round(500/math.sin(math.pi*45/180),1)  #[Newton]

# Using  +ΣF_y(upward) = 0
FBA = round(FBC*math.cos(math.pi*45/180),1)  #[Newton]

# Joint C
# Using  +ΣF_x(right) = 0
FCA = round(FBC*math.cos(math.pi*45/180),1)  #[Newton]

# Using  +ΣF_y(upward) = 0
C_y = round(FBC*math.sin(math.pi*45/180),1)  #[Newton]

# Joint A
# # Using  +ΣF_x(right) = 0
A_x = 500  #[Newton]
A_y = 500  #[Newton]

# Result
print"FBC = ",(FBC),"N"
print"FBA = ",(FBA),"N"
print"FCA = ",(FCA),"N"
print"C_y = ",(C_y),"N"
print"A_x = ",(A_x),"N"
print"A_y = ",(A_y),"N"

# Example 6.2
import math
import numpy as np

# Calculation
# Joint C
# Using  +ΣF_x(right) = 0 and  +ΣF_y(upward) = 0
a = np.array([[-math.cos(math.pi*30/180),math.sin(math.pi*45/180)],[math.sin(math.pi*30/180),-math.cos(math.pi*45/180)]])
b = np.array([0,-1.5])
x = np.linalg.solve(a, b)
FCD = round(x[0],2)  #[kilo Newton]
FCB = round(x[1],2)  #[kilo Newton]

# Calculation
# Joint D
# Using  +ΣF_x(right) = 0
FDA = round(FCD*math.cos(math.pi*30/180)/math.cos(math.pi*30/180),2)  #[kilo Newton]
# Using  +ΣF_y(upward) = 0
FDB = round(2*FCD*math.sin(math.pi*30/180),2)  #[kilo Newton]

# Result
print"FCB = ",(FCB),"kN"
print"FCD = ",(FCD),"kN"
print"FDA = ",(FDA),"kN"
print"FDB = ",(FDB),"kN"

# Example 6.3
from __future__ import division

# Calculation
# Using  +ΣF_x(right) = 0
C_x = 600  #[Newton]
# Using  +ΣMC(counterclockwise) = 0
A_y = (400*3+600*4)/6  #[Newton]
# Using  +ΣF_y(upward) = 0
C_y = A_y - 400  #[Newton]

# Joint A
# Using  +ΣF_y(upward) = 0
FAB = 600*(5/4)  #[Newton]
# Using  +ΣF_x(right) = 0
FAD = (3/5)*FAB  #[Newton]

# Joint D
# Using  +ΣF_x(right) = 0
FDB = (450-600)*(5/3)  #[Newton]
# Using  +ΣF_y(upward) = 0
FDC = (-4/5)*(FDB)  #[Newton]

# Joint C
# Using  +ΣF_x(right) = 0
FCB = 600  #[Newton]

# Result
print"FAB = ",(FAB),"N"
print"FAD = ",(FAD),"N"
print"FDB = ",(FDB),"N"
print"FDC = ",(FDC),"N"
print"FCB = ",(FCB),"N"

# Example 6.4

# Calculation
# Using  +ΣF_y(upward) = 0 at joint G
FGC = 0  #[Newton]

# GC is a zero force menber means that 5-kN load at C must be supported by members CB,CH,CF and CD
# Using  +ΣF_y(upward) = 0 at joint F
FDF = 0  #[Newton]

# Result
print"FGC = ",(FGC),"N"
print"FDF= ",(FDF),"N"

# Example 6.5
from __future__ import division

# Calculation
# Applying equations of equilibrium
# Using  +ΣF_x(right) = 0
A_x = 400  #[Newton]

# Using  +ΣMA(counterclockwise) = 0
D_y = (400*3+1200*8)/12  #[Newton]

# Using  +ΣF_y(upward) = 0
A_y = 1200-900  #[Newton]

# Using  +ΣMG(counterclockwise) = 0
FBC = (400*3+300*4)/3  #[Newton]

# Using  +ΣMC(counterclockwise) = 0
FGE = (300*8)/3  #[Newton]

# Using  +ΣF_y(upward) = 0
FGC = (300*5)/3  #[Newton]

# Result
print"FBC = ",(FBC),"N"
print"FGE = ",(FGE),"N"
print"FGC = ",(FGC),"N"

# Example 6.6
import math
from __future__ import division

# Calculation
# Using  +ΣMO(counterclockwise) = 0
FCF = round((3*8-4.75*4)/(12*math.sin(math.pi*45/180)),3)  #[kilo Newton]

# Result
print"FCF = ",(FCF),"kN"

# Example 6.7
import math

# Calculation
# Using  +ΣMB(counterclockwise) = 0
FED = (-1000*4-3000*2+4000*4)/(math.sin(math.pi*30/180)*4)  #[Newton]

# Using  +ΣF_x(right) = 0 for section bb Fig 6-18c
FEF = 3000*math.cos(math.pi*30/180)/math.cos(math.pi*30/180)  #[Newton]

# Using  +ΣF_y(upward) = 0
FEB = 2*3000*math.sin(math.pi*30/180)-1000  #[Newton]

# Result
print"FEB = ",(FEB),"N"

# Example 6.8
import numpy as np
import math

# Calculation
# At joint A, ΣF_x = 0, ΣF_y = 0, ΣF_z = 0
a = np.array([[0.577,0,0],[0.577,1,0],[-0.577,0,-1]])
b = np.array([0,4,0])
x = np.linalg.solve(a, b)
FAE = round(x[0],2)  #[kilo Newton]
FAB = round(x[1],2)  #[kilo Newton]
FAC = round(x[2],2)  #[kilo Newton]

# At joint B, ΣF_x = 0, ΣF_y = 0, ΣF_z = 0
a = np.array([[-math.cos(math.pi*45/180),0.707,0],[math.sin(math.pi*45/180),0,0],[0,-0.707,1]])
b = np.array([0,4,-2])
x = np.linalg.solve(a, b)
RB = round(x[0],2)  #[kilo Newton]
FBE = round(x[1],2)  #[kilo Newton]
FBD = round(x[2],2)  #[kilo Newton]

# The scalar equation of equilibrium can be applied at joints D and C
FDE = FDC = FCE = 0  #[kilo Newton]

# Result
print"FAE = ",(FAE),"kN"
print"FAB = ",(FAB),"kN"
print"FAC = ",(-FAC),"kN"
print"RB = ",(RB),"kN"
print"FBE = ",(FBE),"kN"
print"FBD = ",(FBD),"kN"
print"FDE = ",(FDE),"kN"
print"FDC = ",(FDC),"kN"
print"FCE = ",(FCE),"kN"

# Example 6.14
import math
from __future__ import division

# Calculation

# Solution 1
# Applying equations of equilibrium to member CB
# Using  +ΣMC(counterclockwise) = 0
FAB = round((2000*2)/(math.sin(math.pi*60/180)*4),1)  #[Newton meter]
# Using  +ΣF_x(right) = 0
C_x = round(FAB*math.cos(math.pi*60/180),1)   #[Newton]
# Using  +ΣF_y(upward) = 0
C_y = round(-FAB*math.sin(math.pi*60/180)+2000,1)   #[Newton]

# Solution 2 
# Using  +ΣMC(counterclockwise) = 0 at Member BC
B_y = (2000*2)/4   #[Newton]
# Using  +ΣMA(counterclockwise) = 0 at Member AB
B_x = round(B_y*3*math.cos(math.pi*60/180)/(3*math.sin(math.pi*60/180)),1)   #[Newton]
# Using  +ΣF_y(upward) = 0 at member BC
C_x = B_x   #[Newton]
C_y = 2000-B_y   #[Newton]
# Result
print"C_x = ",(C_x),"N"
print"C_y = ",(C_y),"N"

# Example 6.15
from __future__ import division

# Calculation
# Using  +ΣF_x(right) = 0 at member BC
B_x = 0  #[kilo Newton]
# Using  +ΣMB(counterclockwise) = 0 at member BC
C_y = (8*1)/2  #[kilo Newton]
# Using  +ΣF_y(upward) = 0 at member BC
B_y = 8-C_y   #[kilo Newton]
# Using  +ΣF_x(right) = 0 at member AB
A_x = 10*(3/5)-B_x  #[kilo Newton]
# Using  +ΣMA(counterclockwise) = 0 at member AB
MA = 10*(4/5)*2+B_y*4  #[kilo Newton meter]
# Using  +ΣF_y(upward) = 0 at member AB
A_y = 10*(4/5)+B_y  #[kilo Newton]

# Result
print"A_x = ",(A_x),"kN"
print"A_y = ",(A_y),"kN"
print"MA = ",(MA),"kN.m"
print"B_x = ",(B_x),"kN"
print"B_y = ",(B_y),"kN"
print"C_y = ",(C_y),"kN"

# Example 6.16
import math

# Calculation
# Using  +ΣMA(counterclockwise) = 0
D_x = (981*2)/2.8   #[Newton]
# Using  +ΣF_x(right) = 0
A_x = D_x   #[Newton]
# Using  +ΣF_y(upward) = 0
A_y = 981   #[Newton]

# Consider member CEF
# Using  +ΣMC(counterclockwise) = 0
FB = round((-981*2)/(math.sin(math.pi*45/180)*1.6),1)   #[Newton]
# Using  +ΣF_x(right) = 0
C_x = round(-FB*math.cos(math.pi*45/180),1)   #[Newton]
# Using  +ΣF_y(upward) = 0
C_y = round(FB*math.sin(math.pi*45/180)+981,1)   #[Newton]

# Result
print"C_x = ",(C_x),"N"
print"C_y = ",(C_y),"N"

# Example 6.17

# Calculation
# Consider entire frame
# Using  +ΣMA(counterclockwise) = 0
C_x = (20*1)/1.1   #[Newton]
# Using  +ΣF_x(right) = 0
A_x = 18.2   #[Newton]
A_y = 20   #[Newton]

# Consider member AB
# Using  +ΣF_x(right) = 0
B_x= 18.2   #[Newton]
# Using  +ΣMB(counterclockwise) = 0
ND = (20*2)/1   #[Newton]
# Using  +ΣF_y(upward) = 0
B_y = 40-20   #[Newton]

# Consider Disk
# Using  +ΣF_x(right) = 0
D_x = 0   #[Newton]
# Using  +ΣF_y(upward) = 0
D_y = 40-20   #[Newton]

# Result
print"B_x = ",(B_x),"N"
print"B_y = ",(B_y),"N"
print"D_x = ",(D_x),"N"
print"D_y = ",(D_y),"N"

# Example 6.18

# Calculation
# Using equations of equilibrium

# Pulley A
# Using  +ΣF_y(upward) = 0
P = 600/3   #[Newton]

# Pulley B
# Using  +ΣF_y(upward) = 0
T = 2*P   #[Newton]

# Pulley C
# Using  +ΣF_y(upward) = 0
R = 2*P+T   #[Newton]

# Result
print"P = ",(P),"N"
print"T = ",(T),"N"
print"R = ",(R),"N"

# Example 6.19
import math
from __future__ import division

# Calculation
# Applying equations of equilibrium to pulley B
# Using  +ΣF_x(right) = 0
B_x = round(490.5*math.cos(math.pi*45/180),1)   #[Newton]
# Using  +ΣF_y(upward) = 0
B_y = round(490.5*math.sin(math.pi*45/180)+490.5,1)   #[Newton]

# Applying equations of equilibrium to pin
# Using  +ΣF_y(upward) = 0
FCB = round((B_y+490.5)*(5/4),1)   #[Newton]
# Using  +ΣF_x(right) = 0
FAB = round((3/5)*FCB+B_x,1)   #[Newton]

# Result
print"B_x = ",(B_x),"N"
print"B_y = ",(B_y),"N"
print"FCB = ",(FCB),"N"
print"FAB = ",(FAB),"N"



