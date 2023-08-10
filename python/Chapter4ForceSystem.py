# Ex 4.1
import math

# Calculation Fig 4-4a
MO = 100*2  #[Newton meter]

# Result Fig 4-4a
print"Fig 4-4a MO = ",(MO),"N.m(clockwise)"

# Calculation Fig 4-4b
MO = 50*0.75  #[Newton meter]

# Result Fig 4-4b
print"Fig 4-4b MO = ",(MO),"N.m(clockwise)"

# Calculation Fig 4-4c
MO = round(40*(4+2*math.cos(math.pi*30/180)),1)  #[Newton meter]

# Result Fig 4-4b
print"Fig 4-4c MO = ",(MO),"N.m(clockwise)"

# Calculation Fig 4-4d
MO = round(60*1*math.sin(math.pi*45/180),1)  #[Newton meter]

# Result Fig 4-4d
print"Fig 4-4d MO = ",(MO),"N.m(anti clockwise)"

# Calculation Fig 4-4e
MO = round(7*(4-1),1)  #[Newton meter]

# Result Fig 4-4e
print"Fig 4-4e MO = ",(MO),"kN.m(anti clockwise)"

# Example 4.2

# Variable Declaration
F = 800  #[Newton]

# Calculation
MA = F*2.5  #[Newton meter]
MB = F*1.5  #[Newton meter]
MC = F*0  #[Newton meter]
MD = F*0.5  #[Newton meter]

# Result
print"MA = ",(MA),"N.m(clockwise)"
print"MB = ",(MB),"N.m(clockwise)"
print"MC = ",(MC)
print"MD = ",(MD),"N.m(anti clockwise)"

# Ex 4.3
import math

# Calculation
# Assuming positive moments act in +k direction i.e counterclockwise
# +MRO(counterclockwise) = ΣFd
MRO = round(-50*(2)+60*(0)+20*(3*math.sin(30*math.pi/180))-40*(4+3*math.cos(30*math.pi/180)),0)  #[Newton meter]

# Result
print"MRO = ",(MRO),"N.m = ",(abs(MRO)),"N.m(clockwise)"

# Ex 4.4
import math
import numpy as np

# Variable Declaration
F = 60  #[Newton]
B_x = 1  #[meter]
B_y = 3  #[meter]
B_z = 2  #[meter]
C_x = 3  #[meter]
C_y = 4  #[meter]
C_z = 0  #[meter]

# Calculation
rB_x = B_x  #[meter]
rB_y = B_y  #[meter]
rB_z = B_z  #[meter]
F_x = round(F*((C_x-B_x)/math.sqrt((C_x-B_x)**(2)+(C_y-B_y)**(2)+(C_z-B_z)**(2))),1)  #[Newton]
F_y = round(F*((C_y-B_y)/math.sqrt((C_x-B_x)**(2)+(C_y-B_y)**(2)+(C_z-B_z)**(2))),1)  #[Newton]
F_z = round(F*((C_z-B_z)/math.sqrt((C_x-B_x)**(2)+(C_y-B_y)**(2)+(C_z-B_z)**(2))),1)  #[Newton]

# Let a = rB X F
a = np.cross([rB_x,rB_y,rB_z],[F_x,F_y,F_z])
MA_x = a[0]  #[Newton meter]
MA_y = a[1]  #[Newton meter]
MA_z = a[2]  #[Newton meter]
MA = round(math.sqrt(MA_x**(2)+MA_y**(2)+MA_z**(2)),0)  #[Newton meter]

# Result
print"MA = ",(MA),"N.m"

# Example 4.5
import math
from __future__ import division
import numpy as np

# Variable Declaration
F1_x = -60   #[Newton]
F1_y = 40   #[Newton]
F1_z = 20   #[Newton]
F2_x = 0   #[Newton]
F2_y = 50   #[Newton]
F2_z = 0   #[Newton]
F3_x = 80   #[Newton]
F3_y = 40   #[Newton]
F3_z = -30   #[Newton]
rA_x = 0  #[meter]
rA_y = 5   #[meter]
rA_z = 0   #[meter]
rB_x = 4   #[meter]
rB_y = 5   #[meter]
rB_z = -2   #[meter]

# Calculation
# Let MRO be resultant moment about O
# Let a = rA X F1, b = rA X F2, c = rB X F3
a = np.cross([rA_x,rA_y,rA_z],[F1_x,F1_y,F1_z])
b = np.cross([rA_x,rA_y,rA_z],[F2_x,F2_y,F2_z])
c = np.cross([rB_x,rB_y,rB_z],[F3_x,F3_y,F3_z])
MRO_x = a[0]+b[0]+c[0]  #[Newton meter]
MRO_y = a[1]+b[1]+c[1]  #[Newton meter]
MRO_z = a[2]+b[2]+c[2]  #[Newton meter]
MRO = round(math.sqrt(MRO_x**(2)+MRO_y**(2)+MRO_z**(2)),2)  #[Newton meter]

# Let u be unit vector which defines the direction of moment axis
u_x = MRO_x/MRO
u_y = MRO_y/MRO
u_z = MRO_z/MRO

# Let alpha,beta and gamma be coordinate direction angles of moment axis 
alpha = round(math.degrees(math.acos(u_x)),1)  #[degrees]
beta = round(math.degrees(math.acos(u_y)),1)  #[degrees]
gamma = round(math.degrees(math.acos(u_z)),1)  #[degrees]

# Result
print"MRO_x = ",(MRO_x),"N.m"
print"MRO_y = ",(MRO_y),"N.m"
print"MRO_z = ",(MRO_z),"N.m"
print"alpha = ",(alpha),"degrees"
print"beta = ",(beta),"degrees"
print"gamma = ",(gamma),"degrees"

# Ex 4.6
import math
from __future__ import division

# Variable Declaration
F = 200   #[Newton]

# Calculation Solution 1
# Moment arm d can be found by trigonometry Refer Fig 4-19b
d = (100*math.cos(math.pi*45/180))/1000  #[meter]
# MA = Fd
MA = round(F*d,1)  #[Newton meter]

# Result Solution 1
# According to right hand thumb rule MA is directed in +k direction
print"Solution 1"
print"MA_x = 0 N.m"
print"MA_y = 0 N.m"
print"MA_z = ",(MA),"N.m\n"

# Calculation Solution 2
# F is resolved into x and y components Refer Fig 4-19c
# MA = ΣFd
MA = round(F*math.sin(math.pi*45/180)*(0.20)-200*math.cos(math.pi*45/180)*(0.10),1)  #[Newton meter]

# Result Solution 2
# According to right hand thumb rule MA is directed in +k direction
print"Solution 2"
print"MA_x = 0 N.m"
print"MA_y = 0 N.m"
print"MA_z = ",(MA),"N.m"

# Example 4.7
import math
import numpy as np 

# Variable Declaration
F = 400  #[Newton]

# Calculation(Scalar Analysis)
# F is resolved into x and y components Refer 4-20b
# Taking +ve moments about O in +k direction
MO = round(F*math.sin(math.pi*30/180)*(0.2)-F*math.cos(math.pi*30/180)*(0.4),1)  #[Newton meter]

# Result Solution 1
print"Solution 1(Scalar Analysis)"
print"MO_x = 0 N.m"
print"MO_y = 0 N.m"
print"MO_z = ",(MO),"N.m\n"

# Calculation(Vector Analysis)
# let r be positon vector and F be force vector
r_x = 0.4  #[meter]
r_y = -0.2  #[meter]
r_z = 0  #[meter]
F_x = round(F*math.sin(math.pi*30/180),1)  #[Newton]
F_y = round(-F*math.cos(math.pi*30/180),1)  #[Newton]
F_z = 0  #[Newton]

# Let MO be the moment given by MO = r X F
a = np.cross([r_x,r_y,r_z],[F_x,F_y,F_z])
MO_x = a[0]   #[Newton meter]
MO_y = a[1]  #[Newton meter]
MO_z=  a[2] #[Newton meter]

# Result Solution 2
print"Solution 2(Vector Analysis)"
print"MO_x = ",(MO_x),"N.m"
print"MO_y = ",(-MO_y),"N.m"
print"MO_z = ",(MO_z),"N.m\n"


# Example 4.8
import math
from __future__ import division
import numpy as np

# Variable Declaration
F_x = -40  #[Newton]
F_y = 20  #[Newton]
F_z = 10  #[Newton]
rA_x = -3  #[meter]
rA_y = 4  #[meter]
rA_z = 6  #[meter]
ua_x = -3/5
ua_y = 4/5
ua_z = 0

# Calculation Solution 1(Vector Analysis)
# Mx = i.(rA X F)
Mx = np.dot([1,0,0],np.cross([rA_x,rA_y,rA_z],[F_x,F_y,F_z]))  #[Newton meter]
# Ma = ua.(rA X F)
Ma = np.dot([ua_x,ua_y,ua_z],np.cross([rA_x,rA_y,rA_z],[F_x,F_y,F_z]))  #[Newton meter]

# Result Solution 1
print"Solution 1(Vector Analysis)"
print"Mx = ",(Mx),"N.m"
print"Ma = ",(Ma),"N.m\n"

# Calculation Solution 2(Scalar Analysis)
# Refer Fig 4-23c
Mx = 10*4 - 20*6  #[Newton meter]
My = 10*3 - 40*6  #[Newton meter]
Mz = 40*4 - 20*3  #[Newton meter]

# Result Solution 2
print"Solution 2(Scalar Analysis)"
print"Mx = ",(Mx),"N.m"

# Example 4.9
import math
from __future__ import division
import numpy as np

# Variable Declaration
F_x = -600  #[Newton]
F_y = 200  #[Newton]
F_z = -300  #[Newton]
rD_x = 0  #[meter]
rD_y = 0.2  #[meter]
rD_z = 0  #[meter]

# Calculation
# Unit vector uB defines the direction of AB axis of the rod
uB_x = 0.4/math.sqrt(0.4**(2)+0.2**(2))
uB_y = 0.2/math.sqrt(0.4**(2)+0.2**(2)) 
uB_z = 0
# MAB = uB.(rD X F)
MAB = np.dot([uB_x,uB_y,uB_z],np.cross([rD_x,rD_y,rD_z],[F_x,F_y,F_z])) #[Newton meter]
MAB_x = MAB*uB_x  #[Newton meter]
MAB_y = MAB*uB_y  #[Newton meter]

# Result
print"MAB_x = ",(MAB_x),"N.m"
print"MAB_y = ",(MAB_y),"N.m"


# Example 4.10
from __future__ import division

# Calculation
# Let the couple has magnitude of M having direction out of page
M = 40*0.6  #[Newton meter]
# To preserve counterclockwise rotation of M vertical forces acting through points A and B must be directed as shown in Fig 4-29c
F = M/0.2   #[Newton]

# Result
print"F = ",(F),"N"

# Example 4.11
from __future__ import division

# Variable Declaration
F = 150   #[Newton]

# Calculation
F_x = (4/5)*F   #[Newton]
F_y = (3/5)*F   #[Newton]

#  Let the couple moment is calculated about D
MD = F_x*0 - F_y*1 + F_y*2.5 + F_x*0.5  #[Newton meter]

# Let the couple moment is calculated about A
MA = F_y*1.5 + F_x*0.5  #[Newton meter]

# Result
print"Moment calculated about D"
print"MD = ",(MD),"N.m(counterclockwise)"
print"Moment calculated about A"
print"MA = ",(MA),"N.m(counterclockwise)"

# Example 4.12
import math

# Variable Declaration
rA_x = 0   #[meter]
rA_y = 0.8   #[meter]
rA_z = 0   #[meter]
rB_x = 0.6*math.cos(math.pi*30/180)   #[meter]
rB_y = 0.8   #[meter]
rB_z = -0.6*math.sin(math.pi*30/180)   #[meter]
rAB_x = 0.6*math.cos(math.pi*30/180)   #[meter]
rAB_y = 0   #[meter]
rAB_z = -0.6*math.sin(math.pi*30/180)   #[meter]
# Let force acting at B be FB
FB_x = 0   #[Newton]
FB_y = 0   #[Newton]
FB_z = -25   #[Newton]
# Let force acting at A be FA
FA_x = 0   #[Newton]
FA_y = 0   #[Newton]
FA_z = 25   #[Newton]

# Calculation Solution 1(Vector Analysis)
# Let MO be moment about about O Refer Fig 4-31b
# Let a = rA X FB, b = rB X FA
a = np.cross([rA_x,rA_y,rA_z],[FB_x,FB_y,FB_z])
b = np.cross([rB_x,rB_y,rB_z],[FA_x,FA_y,FA_z])
MO_x = round(a[0]+b[0],1) #[Newton meter]
MO_y = round(a[1]+b[1],1)  #[Newton meter]
MO_z = round(a[2]+b[2],1)  #[Newton meter]

# Let MA be moment about about A Refer Fig 4-31c
# MA = rAB X FA
a = np.cross([rAB_x,rAB_y,rAB_z],[FA_x,FA_y,FA_z])
MA_x = round(a[0],1)  #[Newton meter]
MA_y = round(a[1],1)  #[Newton meter]
MA_z = round(a[2],1)  #[Newton meter]

# Calculation Solution 2(Vector Analysis)
M = 25*0.52  #[Newton meter]
# M acts in -j direction
M_x = 0  #[Newton meter]
M_y = -M  #[Newton meter]
M_z = 0  #[Newton meter]

# Result 
print"Solution 1 (Vector Analysis)"
print"MO_x = ",(MO_x),"N.m"
print"MO_y = ",(MO_y),"N.m"
print"MO_z = ",(MO_z),"N.m"
print"MA_x = ",(MA_x),"N.m"
print"MA_y = ",(MA_y),"N.m"
print"MA_z = ",(MA_z),"N.m\n"
print"Solution 2 (Scalar Analysis)"
print"M_x = ",(M_x),"N.m"
print"M_y = ",(M_y),"N.m"
print"M_z = ",(M_z),"N.m"

# Example 4.13
from __future__ import division
import numpy as np

# Variable Declaration
rDC_x = 0.3  #[meter]
rDC_y = 0  #[meter]
rDC_z = 0  #[meter]
FC_x = 0  #[Newton]
FC_y = 125*(4/5)  #[Newton]
FC_z = -125*(3/5)  #[Newton]

# Calculation
M1 = 150*0.4   #[Newton meter]

# By right hand rule M1 acts in +i direction
M1_x = M1   #[Newton meter]
M1_y = 0   #[Newton meter]
M1_z = 0   #[Newton meter]

# M2 = rDC X FC
a = np.cross([rDC_x,rDC_y,rDC_z],[FC_x,FC_y,FC_z]) 
M2_x = a[0]  #[Newton meter]
M2_y = a[1]   #[Newton meter]
M2_z = a[2] #[Newton meter]

# M1 and M2 are free vectors.So they may be moved to some arbitrary point P and added vectorially.
# The resultant couple moment becomes MR = M1 + M2
MR_x = M1_x + M2_x   #[Newton meter]
MR_y = M1_y + M2_y   #[Newton meter]
MR_z = M1_z + M2_z   #[Newton meter]

# Result
print"MR_x = ",(MR_x),"N.m"
print"MR_y = ",(MR_y),"N.m"
print"MR_z= ",(MR_z),"N.m"

# Example 4.14
import math 

# Calculation
# Let resultant force be FR
FR_x = -100-400*math.cos(math.pi*45/180)  #[Newton]
FR_y = -600-400*math.sin(math.pi*45/180)  #[Newton]
FR = round(math.sqrt(FR_x**(2)+FR_y**(2)),1)  #[Newton]
theta = round(math.degrees(math.atan(FR_y/FR_x)),1)  #[degrees]

# Let resultant couple moment be MRA which is calculated by summing moments of forces about A
MRA = round(100*0-600*0.4-400*math.sin(math.pi*45/180)*0.8-400*math.cos(math.pi*45/180)*0.3,1)   #[Newton meter]

# Result
print"FR = ",(FR),"N"
print"theta = ",(theta),"degrees"
print"MRA = ",(MRA),"N.m = ",(-MRA),"N.m(clockwise)"


# Example 4.15
import math
from __future__ import division
import numpy as np

# Variable Declaration
F1_x = 0  #[Newton]
F1_y = 0  #[Newton]
F1_z = -800  #[Newton]
F2 = 300  #[Newton]
rCB_x = -0.15   #[meter]
rCB_y = 0.1   #[meter]
rCB_z = 0   #[meter]
rC_x =  0  #[meter]
rC_y =  0   #[meter]
rC_z = 1   #[meter]
rB_x = -0.15   #[meter]
rB_y = 0.1   #[meter]
rB_z = 1   #[meter]
# Calculation
# Let uCB be unit vector along rCB
uCB_x = rCB_x/math.sqrt(rCB_x**(2)+rCB_y**(2)+rCB_z**(2))
uCB_y = rCB_y/math.sqrt(rCB_x**(2)+rCB_y**(2)+rCB_z**(2))
uCB_z = rCB_z/math.sqrt(rCB_x**(2)+rCB_y**(2)+rCB_z**(2))
F2_x = round(300*uCB_x,1)  #[Newton]
F2_y = round(300*uCB_y,1)  #[Newton]
F2_z = round(300*uCB_z,1)  #[Newton]
M_x = 0  #[Newton meter]
M_y = -500*(4/5)  #[Newton meter]
M_z = 500*(3/5)  #[Newton meter]
# FR = F1 + F2
FR_x = F1_x + F2_x  #[Newton]
FR_y = F1_y + F2_y  #[Newton]
FR_z = F1_z + F2_z  #[Newton]

# MRO = M + rC X F1 + rB X F2
# Let a = rC X F1 and b = rB X F2
a = np.cross([rC_x,rC_y,rC_z], [F1_x,F1_y,F1_z])
b = np.cross([rB_x,rB_y,rB_z], [F2_x,F2_y,F2_z])
MRO_x = M_x + a[0] + b[0]  #[Newton meter]
MRO_y = M_y + a[1] + b[1]  #[Newton meter]
MRO_z = M_z + a[2] + b[2]  #[Newton meter]

# Result
print"FR_x = ",(FR_x),"N"
print"FR_y = ",(FR_y),"N"
print"FR_z = ",(FR_z),"N"
print"MRO_x = ",(MRO_x),"N"
print"MRO_y = ",(MRO_y),"N"
print"MRO_z = ",(MRO_z),"N"

# Example 4.16
import math
from __future__ import division
# Calculation
FR_x = 500*math.cos(math.pi*60/180)+100  #[Newton]
FR_y = -500*math.sin(math.pi*60/180)+200  #[Newton]
FR = round(math.sqrt(FR_x**(2)+FR_y**(2)),1)  #[Newton]
theta = round(math.degrees(math.atan(FR_y/FR_x)),1)  #[degrees]

# +MRE(counterclockwise) = ΣME
d = round((500*math.sin(math.pi*60/180)*(4) + 500*math.cos(math.pi*60/180)*(0) - 100*0.5 - 200*2.5 - 350)/233.0,2)   #[meter] 

# Result
print"FR = ",(FR),"N"
print"theta = ",(-theta),"degrees"
print"d = ",(d),"m"  # Correction in the answer

# Example 4.17
import math
from __future__ import division

# Calculation
# Let FR be resultant force
FR_x = -250*(3/5)-175  #[Newton]
FR_y = -250*(4/5)-60  #[Newton]
FR = round(math.sqrt(FR_x**(2)+FR_y**(2)),1)  #[Newton]
theta = round(math.degrees(math.atan(FR_y/FR_x)),1)  #[degrees]

# +MRA(counterclockwise) = ΣMA
y = round((175*2.5 - 60*1.5 + 250*(3/5)*5.5 - 250*(4/5)*4 - 260*(0))/325,3)   #[meter]

# By principle of transmissibility FR can be treated as intersecting BC Refer Fig 4-44b
# +MRA(counterclockwise) = ΣMA
x = round((325*5.5 - 175*2.5 + 60*1.5 - 250*(3/5)*5.5 + 250*(4/5)*4)/260,3)   #[meter]

# Result
print"y = ",(y),"m"
print"x = ",(x),"m"

# Example 4.18

# Calculation
# +FR = ΣF Refer Fig 4-45a
FR = -600+100-400-500  #[Newton]
# Using right hand thumb rule where ppsitive moments act in +i direction we have MR_x = ΣM_x
y = round(-(600*(0)+100*5-400*10+500*0)/1400,2)   #[meter]
# Assuming positive moments act in +j direction
# MR_y = ΣM_y
x = round((600*8-100*6+400*0+500*0)/1400,2)   #[meter]

# Result
print"y = ",(y),"m"
print"x = ",(x),"m"

# Example 4.19
import math
import numpy as np

# Variable Declaration
FR_x = 0  #[Newton]
FR_y = 0  #[Newton]
FR_z = -300-200-150  #[Newton]
rA_x = 2   #[meter]
rA_y = 0    #[meter]
rA_z = 0   #[meter]  
rB_x = 0    #[meter]
rB_y = -2    #[meter]
rB_z = 0   #[meter]
rC_x = -2*math.sin(math.pi*45/180)   #[meter] 
rC_y = 2*math.cos(math.pi*45/180)   #[meter]
rC_z = 0   #[meter]

# Let a = rA X (-300k), b = rB X (-200k), c = rC X (-150k)
a = np.cross([rA_x,rA_y,rA_z], [0,0,-300])
b = np.cross([rB_x,rB_y,rB_z], [0,0,-200])
c = np.cross([rC_x,rC_y,rC_z], [0,0,-150])
x = round((a[1]+b[1]+c[1])/650,2)   #[meter]
y = round(-((a[0]+b[0]+c[0]))/650,2)   #[meter]

# Result
print"FR_x = ",(FR_x),"N"
print"FR_y = ",(FR_y),"N"
print"FR_z = ",(FR_z),"N"
print"x = ",(x),"m"
print"y = ",(y),"m"

# Example 4.20
from scipy import integrate
import numpy as np

# Calculation
# The coloured differential area element dA = wdx = 60x**(2)
# Summing these elements from x = 0 to x = 2m we obtain force FR
# FR = ΣF
var = lambda x: 60*x**(2)
a = integrate.quad(var, 0, 2)
FR = a[0]  #[Newton]

# Since the element of area dA is located at distance x from O, the location x_bar of FR is measured from O
var = lambda x: x*60*x**(2)
a = integrate.quad(var, 0, 2)
x_bar = a[0]/FR   #[meter]

# Result
print"FR = ",(FR),"N"
print"x_bar = ",(x_bar),"m"

# Example 4.21
from __future__ import division

# Calculation
# The magnitude of resultant force is equal to the area under the triangle
FR = (0.5*9*1440)/1000  #[Newton]

# The line of action of FR passes through the centroid C of the triangle
x_bar = 9-(1/3)*9   #[meter]

# Result
print"FR = ",(FR),"kN"
print"x_bar =",(x_bar),"m"

# Example 4.22

# Calculation
# Refer Fig 4-50b
# The magnitude of the force represented by each of these loadings is equal to its associated area
F1 = 0.5*3*50  #[Newton]
F2 = 3*50  #[Newton]

# The lines of action of these parallel forces act through the centroid of their associated areas
x1_bar = (1/3)*3   #[meter]
x2_bar = (1/2)*3   #[meter]

# +FR(downward) = ΣF
FR = F1 + F2  #[Newton]

# +MRA(clockwise) = ΣMA
x_bar = (1*75+1.5*150)/225   #[meter]

# Result
print"FR = ",(FR),"N"
print"x_bar = ",(x_bar),"m"



