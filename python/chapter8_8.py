#Given 
import math 
from math import radians
o = 22.5     #degrees , The angle of infetisimal wedge 
A = 1        #mm2 The area of the element 
A_ab = 1*(math.cos(radians(o))) #mm2 - The area corresponds to AB
A_bc = 1*(math.sin(radians(o))) #mm2 - The area corresponds to BC
S_1 = 3 #MN The stresses applying on the element 
S_2 = 2 #MN
S_3 = 2 #MN
S_4 = 1 #MN 
F_1 = S_1*A_ab # The Forces obtained by multiplying stress by their areas 
F_2 = S_2*A_ab
F_3 = S_3*A_bc
F_4 = S_4*A_bc
#sum of F_N = 0 equilibrim in normal direction 
N = (F_1-F_3)*(math.cos(radians(o))) + (F_4 - F_2)*(math.sin(radians(o)))

#sum of F_s = 0 equilibrim in tangential direction 

S = (F_2-F_4)*(math.cos(radians(o))) + (F_1 - F_3)*(math.sin(radians(o)))

Stress_Normal = N/A #Mpa - The stress action in normal direction on AB
Stress_tan = S/A    #Mpa - The stress action in tangential direction on AB
print "The stress action in normal direction on AB",round(Stress_Normal,2),"Mpa"
print "The stress action in tangential direction on AB",round(Stress_tan,2),"Mpa"

#Given
o = -22.5 #degrees , The angle of infetisimal wedge 
A = 1     #mm2 The area of the element 
import math 
from math import radians
from numpy import array
A_ab = 1*(math.cos(radians(o))) #mm2 - The area corresponds to AB
A_bc = 1*(math.sin(radians(o))) #mm2 - The area corresponds to BC
S_1 = 3.0 #MN The stresses applying on the element 
S_2 = 2.0 #MN
S_3 = 2.0 #MN
S_4 = 1.0 #MN
#Caliculations 

F_1 = S_1*A_ab # The Forces obtained by multiplying stress by their areas 
F_2 = S_2*A_ab
F_3 = S_3*A_bc
F_4 = S_4*A_bc
#sum of F_N = 0 equilibrim in normal direction 
N = (F_1-F_3)*(math.cos(radians(o))) + (F_4 - F_2)*(math.sin(radians(o)))

#sum of F_s = 0 equilibrim in tangential direction 

S = (F_2-F_4)*(math.cos(radians(o))) + (F_1 - F_3)*(math.sin(radians(o)))

Stress_Normal = N/A #Mpa - The stress action in normal direction on AB
Stress_tan = S/A    #Mpa - The stress action in tangential direction on AB
print "a) The stress action in normal direction on AB",round(Stress_Normal,2),"Mpa"
print "a) The stress action in tangential direction on AB",round(Stress_tan,2),"Mpa"

#Part- b

S_max = (S_4+S_1)/2 + (((((S_4-S_1)/2)**2) + S_3**2)**0.5)   #Mpa - The maximum stress
S_min = (S_4+S_1)/2.0 - (((((S_4-S_1/2))**2) + S_3**2)**0.5) #Mpa - The minumum stress
k = 0.5*math.atan(S_3/((S_1-S_4)/2))                         #radians The angle of principle axis
k_1 = math.degrees(k)
k_2 = k_1+90 #The principle plane angles
print "b) The principle stress ",round(S_max,1),"Mpa tension"
print "b) The principle stress ",round(S_min,2),"Mpa compression"
print "b) The principle plane angles are",round(k_1,0),",",round(k_2,0),"degrees"

#part-c
#The maximum shear stress case
t_xy = (((((S_4-S_1)/2)**2) + S_3**2)**0.5) #Mpa - The maximum shear stress case
K = 0.5*math.atan((-(S_1-S_4)/(2*S_3)))     #radians The angle of principle axis
K_0 = math.degrees(K)
if K_0<0:
    K_1 = K_0+90
else:
    K_1 = K_0
K_2 = K_1+90 #PRinciple plain angles
T_xy = -((S_1-S_4)/2)*(math.sin(radians(2*K_1))) + ((S_4+S_1)/2)*(math.cos(radians(2*K_1))) # Shear stress
print "c) The maximum shear is ",round(T_xy,2),"Mpa" 
S_mat_a = array([round(S_max,1),round(S_min,1),0])                       #MPa maximum stress matrix
S_mat_b = array([(S_4+S_1)/2,round(T_xy,2),round(T_xy,2),(S_4+S_1)/2])   #MPa maximum stress matrix at maximum shear
print "a)",S_mat_a,"Mpa"
print "b)",S_mat_b,"Mpa"

#Given 
import math 
from math import radians 
S_x = -2 #Mpa _ the noraml stress in x direction
S_y = 4 #Mpa _ the noraml stress in Y direction
c = (S_x + S_y)/2 #Mpa - The centre of the mohr circle 
point_x = -2 #The x coordinate of a point on mohr circle
point_y = 4  #The y coordinate of a point on mohr circle
Radius = pow((point_x-c)**2 + point_y**2,0.5) # The radius of the mohr circle
S_1  = Radius +1#MPa The principle stress
S_2 = -Radius +1 #Mpa The principle stress
S_xy_max = Radius #Mpa The maximum shear stress
print "The principle stresses are",S_1 ,"Mpa",S_2,"Mpa"
print "The maximum shear stress",S_xy_max,"Mpa"

#Given
import math 
S_x = 3.0 #Mpa _ the noraml stress in x direction
S_y = 1.0 #Mpa _ the noraml stress in Y direction
c = (S_x + S_y)/2 #Mpa - The centre of the mohr circle 
point_x = 1 #The x coordinate of a point on mohr circle
point_y = 3  #The y coordinate of a point on mohr circle
#Caliculations 

Radius = pow((point_x-c)**2 + point_y**2,0.5) # The radius of the mohr circle
#22.5 degrees line is drawn 
o = 22.5 #degrees 
a = 71.5 - 2*o #Degrees, from diagram 
stress_n = c + Radius*math.sin(math.degrees(o)) #Mpa The normal stress on the plane 
stress_t =  Radius*math.cos(math.degrees(o)) #Mpa The tangential stress on the plane
print "The normal stress on the 221/2 plane ",round(stress_n,2),"Mpa"
print "The tangential stress on the 221/2 plane ",round(stress_t,2),"Mpa"

import math
e_x = -500   #10-6 m/m The contraction in X direction
e_y = 300   #10-6 m/m The contraction in Y direction
e_xy = -600 #10-6 m/m discorted angle
centre = (e_x + e_y)/2  #10-6 m/m 
point_x = -500 #The x coordinate of a point on mohr circle
point_y = 300  #The y coordinate of a point on mohr circle
Radius = 500   #10-6 m/m - from mohr circle
e_1  = Radius +centre    #MPa The principle strain
e_2 = -Radius +centre    #Mpa The principle strain
k = math.atan(300.0/900) # from geometry
k_1 = math.degrees(k)
print "The principle strains are",e_1,"um/m",e_2,"um/m"
print "The angle of principle plane",round(k_1,2) ,"degrees"

#Given
e_0 = -500 #10-6 m/m 
e_45 = 200 #10-6 m/m 
e_90 = 300 #10-6 m/m
E = 200    #Gpa - youngs modulus of steel 
v = 0.3    # poissions ratio 
#Caliculations 

e_xy = 2*e_45 - (e_0 +e_90 ) #10-6 m/m from equation 8-40 in text
# from example 8.7
e_x = -500        #10-6 m/m The contraction in X direction
e_y = 300         #10-6 m/m The contraction in Y direction
e_xy = -600       #10-6 m/m discorted angle
centre = (e_x + e_y)/2  #10-6 m/m 
point_x = -500          #The x coordinate of a point on mohr circle
point_y = 300           #The y coordinate of a point on mohr circle
Radius = 500            #10-6 m/m - from mohr circle
e_1  = Radius +centre #MPa The principle strain
e_2 = -Radius +centre #Mpa The principle strain

stress_1 = E*(10**-3)*(e_1+v*e_2)/(1-v**2) #Mpa the stress in this direction 
stress_2 = E*(10**-3)*(e_2+v*e_1)/(1-v**2) #Mpa the stress in this direction 
print"The principle stresses are ",round(stress_1,2),"Mpa",round(stress_2,2),"MPa" 

