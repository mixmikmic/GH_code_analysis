#Given
import math
d_bolt = 20.0       #mm,diameter,This is not the minimum area
d_bolt_min = 16.0   #mm This is at the roots of the thread 
#This yealds maximum stress 
A_crossection = (math.pi)*(d_bolt**2)/4         #mm*2
A_crossection_min = (math.pi)*(d_bolt_min**2)/4 #mm*2 ,This is minimum area which yeilds maximum stress
load = 10.0 #KN
BC = 1.0    #m
CF = 2.5    #m
contact_area = 200*200  # mm*2 , The contact area at c

#caliculations 
#Balancing forces in the x direction:
# Balncing the moments about C and B:
Fx = 0 
R_cy = load*(BC+CF)   #KN , Reaction at C in y-direction
R_by = load*(CF)      #KN , Reaction at B in y-direction
#Because of 2 bolts
stress_max = (R_by/(2*A_crossection_min))*(10**3)  # MPA,maximum stess records at minimum area
stress_shank = (R_by/(2*A_crossection))*(10**3)    # MPA
Bearing_stress_c = (R_cy/contact_area)*(10**3)     #MPA, Bearing stress at C

print"The bearing stress at C  is  ",(Bearing_stress_c) ,"MPA"
print"The maximum normal stress in BD bolt is: ",round(stress_max),"MPA"
print"The tensile strss at shank of the bolt is: ",round(stress_shank),"MPA"



#Given 
load_distributed = 20 #KN/m*2, This is the load distributed over the pier
H = 2          # m, Total height 
h = 1          #m , point of investigation 
base = 1.5     #m The length of crossection in side veiw 
top = 0.5      #m ,The length where load is distributed on top
base_inv = 1   #m , the length at the point of investigation 
area = 0.5*1   #m ,The length at a-a crossection 
density_conc = 25 #KN/m*2
#caliculation of total weight 

v_total = ((top+base)/2)*top*H       #m*2 ,The total volume 
w_total = v_total* density_conc  #KN , The total weight
R_top = (top**2)*load_distributed    #KN , THe reaction force due to load distribution 
reaction_net = w_total + R_top

#caliculation of State of stress at 1m 
v_inv = ((top+base_inv)/2)*top*h    #m*2 ,The total volume from 1m to top
w_inv = v_inv*density_conc          #KN , The total weight from 1m to top
reaction_net = w_inv + R_top        #KN
Stress = reaction_net/area           #KN/m*2
print"The total weight of pier is",w_total,"KN"
print"The stress at 1 m above is",Stress,"MPA"

#Given
from math import pow
d_pins = 0.375 #inch
load = 3      #Kips
AB_x = 6      #inch,X-component
AB_y = 3      #inch,Y-component  
BC_y = 6      #inch,Y-component
BC_x = 6      #inch,X-component
area_AB = 0.25*0.5                #inch*2 
area_net = 0.20*2*(0.875-0.375) #inch*2 
area_BC = 0.875*0.25              #inch*2 
area_pin = d_pins*2*0.20           #inch*2 
area_pin_crossection = 3.14*((d_pins/2)**2)
#caliculations

slope = AB_y/ AB_x   #For AB
slope =  BC_y/ BC_x  #For BC

#momentum at point C:
F_A_x = (load*AB_x )/(BC_y + AB_y ) #Kips, F_A_x X-component of F_A

#momentum at point A:
F_C_x = -(load*BC_x)/(BC_y + AB_y ) #Kips, F_C_x X-component of F_c

#X,Y components of F_A
F_A= (pow(5,0.5)/2)*F_A_x  #Kips
F_A_y = 0.5*F_A_x          #Kips

#X,Y components of F_C  
F_C= pow(2,0.5)*F_C_x    #Kips
F_C_y = F_C_x            #Kips

T_stress_AB = F_A/area_AB                 #Ksi , Tensile stress in main bar AB
stress_clevis = F_A/area_net              #Ksi ,Tensile stress in clevis of main bar AB
c_strees_BC = F_C/area_BC                 #Ksi , Comprensive stress in main bar BC
B_stress_pin = F_C/area_pin               #Ksi , Bearing stress in pin at C
To_stress_pin =  F_C/area_pin_crossection #Ksi , torsion stress in pin at C

print"Tensile stress in main bar AB:",round(T_stress_AB,2),"Ksi"
print"Tensile stress in clevis of main bar AB:",round(stress_clevis,2),"Ksi"
print"Comprensive stress in main bar BC:",round(-c_strees_BC,2),"Ksi"
print"Bearing stress in pin at C:",round(-B_stress_pin,2),"Ksi"
print"torsion stress in pin at C:",round(To_stress_pin,2),"Ksi"

#Given
strength_steel = 120 #Ksi
factor = 2.5
F_C =   2.23 #Ksi

#caliculations

stress_allow = strength_steel/factor #Ksi
A_net = F_C/strength_steel           #in*2 , 
#lets adopt 0.20x0.25 in*2 and check wether we are correct or not? 

A_net_assumption = 0.25*0.20            #in*2 , this is assumed area which is near to A_net
stress = 2.23/A_net_assumption          #Ksi
factor_assumed = strength_steel/stress 

if factor_assumed > factor :
    print "The factor",factor,"is less than assumed factor",round(factor_assumed,1),"so this can be considered"
else:
    print "The assumed factor",factor, "is more than assumed factor",factor_assumed,"factor_assumed"
 
    

#Given
mass = 5       #Kg
frequency = 10 #Hz
stress_allow = 200 #MPa
R = 0.5        #m

#caliculations 
from math import pi
w = 2*pi*frequency #rad/sec
a = (w**2)*R       #m*2/sec
F = mass*a         #N
A_req = F/stress_allow  #m*2 , The required area for aloowing stress
print"The required size of rod is:",round(A_req,2),"m*2"

#Given
D_n = 5.0           #kips, dead load
L_n_1 = 1.0         #kips ,live load 1
L_n_2 = 15          #kips ,live load 2
stress_allow = 22   #ksi
phi = 0.9           #probalistic coefficients
y_stress = 36       #ksi,Yeild strength
#According to AISR 

#a
p_1 = D_n + L_n_1 #kips since the total load is sum of dead load and live load
p_2 = D_n + L_n_2 #kips, For second live load

Area_1 = p_1/stress_allow  #in*2 ,the allowable area for the allowed stress
Area_2 = p_2/stress_allow  #in*2
print "the allowable area for live load",L_n_1,"is",round(Area_1,3),"in*2"
print "the allowable area for live load",L_n_2,"is",round(Area_2,3),"in*2"

#b
#area_crossection= (1.2*D_n +1.6L_n)/(phi*y_stress)

area_crossection_1= (1.2*D_n +1.6*L_n_1)/(phi*y_stress) #in*2,crossection area for first live load
area_crossection_2= (1.2*D_n +1.6*L_n_2)/(phi*y_stress) #in*2,crossection area for second live load
print "the crossection area for live load",L_n_1,"is",round(area_crossection_1,3),"in*2"
print "the crossection area for live load",L_n_2,"is",round(area_crossection_2,3),"in*2"

#Given
A_angle = 2              #in*2 
stress_allow = 20        #ksi, The maximum alowable stress
F = stress_allow*A_angle #K, The maximum force
AD = 3                   #in, from the figure
DC = 1.06                #in, from the figure
strength_AWS = 5.56 # kips/in,Allowable strength according to AWS

#caliculations 
#momentum at point "d" is equal to 0
R_1 = (F*DC)/AD      #k,Resultant force developed by the weld
R_2 = (F*(AD-DC))/AD #k,Resultant force developed by the weld

l_1 = R_1/strength_AWS #in,Length of the Weld 1
l_2 = R_2/strength_AWS #in,Length of the Weld 2
       
print "Length of the Weld 1:",round(l_1,2),"in"
print "Length of the Weld 2:",round(l_2,2),"in"      
       



