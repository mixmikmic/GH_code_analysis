#Given
from numpy.random import randn
th = 8     #mm - thickness 
G =  0.64  #N/mm2 -shear modulus
 
a = 40     #mm length
b = 20     #mm width \
Area = a*b #mm2
e =  randn()   # lets say any random vale
strain =  e/th # strain in shearing case
#caliculations

stress =G*strain
F = stress*Area
#stiffness =  froce/displacement
k = F/e #N/mm
print "the stiffness constant is ",k,"N/mm"


#Given
a = 50          #mm - length of a cube
E =  200        # Gpa - the youngs modulus 
v =  0.25       # no units- poissions ratio 
pressure =  200 # Mpa - pressure acting on all sides 
#pressure is a compressive stress 
S_x = -200 # Gpa - The stress in X direction 
S_y = -200 # Gpa - The stress in Y direction
S_z = -200 # Gpa - The stress in Z direction
#Caliculations

e = S_x*pow(10,-3)/E - v*S_y*pow(10,-3)/E-v*S_z*pow(10,-3)/E#mm - considering all three directions 
x = e*a #mmThe change in the dimension between parallel faces
print "The change in the dimension between parallel faces is ",x,"mm"

#Given
R = 1000    # mm - radius of the cylinder 
t =  10     #mm - thickness of the cylinder
p_in = 0.80 #Mpa- Internal pressure 
E = 200     #Mpa- youngs modulus 
v = 0.25    # poission ratio
#caliculations

Stress_1 = p_in*R/t     #Mpa -Hoop stress                    #From derived expressions 
Stress_2 = p_in*R/(2*t) #Mpa- Longitudinal stress 
e = Stress_1*pow(10,-3)/E-v*Stress_2*pow(10,-3)/E
dia_change = e*R        #mm- The change in daimeter of the cylinder 
print "The Hoop stress is ",Stress_1,"mm"
print "The longitudinal stress is ",Stress_2, "mm"
print "The change in daimeter of the cylinder is",dia_change,"mm"

#Given
R = 1000             #mm - radius of the cylinder
th = 10              #mm - thickness of the cylinder
E = 200              #Mpa- youngs modulus 
v = 0.25             # poission ratio
p_in = 0.80          #Mpa- Internal pressure
t =  10               #mm - thickness of the cylinder
#caliculations

Stress_1 = p_in*R/(2*t) #Mpa -Hoop stress                    #From derived expressions 
Stress_2 = p_in*R/(2*t) #Mpa- Longitudinal stress
# Hoop stress and Longitudinal stress are same in this case
e = Stress_1*pow(10,-3)/E-v*Stress_2*pow(10,-3)/E
dia_change = e*R        #mm- The change in daimeter of the cylinder
print "The maximum stress is ",Stress_2, "mm"
print "The change in daimeter of the cylinder is",dia_change,"mm"

#Given
p_in = 0.7   #Mpa - internal pressure 
n_bolts = 20 # number of bolts 
dia = 650    #mm - bolt circle diameter 
stress_allow = 125 #mm Maximum alowable stress
Stress_conc = 2    #stress concentration
d = 25             #mm 
#caliculations

F = p_in*3.14*pow(((dia-2*d)/2),2)*pow(10,6) #N
F_each = F/n_bolts #N- force per each Bolt
A = Stress_conc*F_each/(stress_allow*pow(10,-6)) #mm2 The bolt area 
Bolt_dia = 2*pow((round(A,3)/3.14),0.5)*pow(10,-7) #mm the bolt daimeter
print "The diameter of each bolt is",Bolt_dia,"mm"

#Given
from numpy.random import randn
r_1 = randn()# let r_1 is a random number 
t =  0.1*r_1 #Thickness
r_2 = r_1+t  #Outer Radius
#caliculations

stress_1_a=  ((r_1**2)/((r_2**2)-(r_1**2)))*(1+((r_2**2)/(r_1**2)))#p -r =r_2
stress_2_a=  ((r_1**2)/((r_2**2)-(r_1**2)))*(1+((r_2**2)/(r_2**2)))#p -r =r_1
stress_avg_a=(stress_1_a+stress_2_a)/2 #p
print "The average stress in case a is",stress_avg_a,"p"

#Case-B
#Given
r_1 = randn()# let r_1 is a random number 
r_2_b= 4*r_1 
#caliculations

stress_1_b =  ((r_1**2)/((r_2_b**2)-(r_1**2)))*(1+((r_2_b**2)/(r_1**2)))#p -r =r_2
stress_2_b=  ((r_1**2)/((r_2_b**2)-(r_1**2)))*(1+((r_2_b**2)/(r_2_b**2)))#p -r =r_1
stress_avg_b=(stress_1_b+stress_2_b)/2 #p
print "The average stress in case b is",stress_2_b,"p"

