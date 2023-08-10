#Given 
import math 
b = 40.0    #mm - The width of the beam crossection
h = 300.0   #mm - The length of the beam crossection 
V = 40.0    #KN - The shear stress in teh crossection
M = 10.0    #KN-m - The bending moment on K----K crossection 
c = h/2     #mm -The position at which maximum stress occurs on the crossection
I = b*(h**3)/12                #mmm4 - the moment of inertia 
#Caliculations 

stress_max_1 = M*c*(10**6)/I   #The maximum stress occurs at the end
stress_max_2 = -M*c*(10**6)/I  #The maximum stress occurs at the end
y = 140                        #mm The point of interest, the distance of element from com
n = y/(c)                      # The ratio of the distances from nuetral axis to the elements
stress_L_1 = n*stress_max_1    #The normal stress on elements L--L
stress_L_2 = -n*stress_max_1   #The normal stress on elements L--L
x = 10       #mm The length of the element
A = b*x      #mm3 The area of the element 
y_1 = y+x/2  # the com of element from com of whole system
stress_xy = V*A*y_1*(10**3)/(I*b) #Mpa - The shear stress on the element 
#stresses acting in plane 30 degrees 
o = 60 #degrees - the plane angle
stress_theta = stress_L_1/2 +  stress_L_1*(math.cos(math.radians(o)))/2 - stress_xy*(math.sin(math.radians(o))) #Mpa by direct application of equations
stress_shear =  -stress_L_1*(math.sin(math.radians(o)))/2 - stress_xy*(math.cos(math.radians(o)))               #Mpa Shear stress
  
print "a)The principle stresses are ",round(stress_max_1,2),"MPa,",round(stress_max_2,2),"Mpa"
print "b)The stresses on inclines plane ",round(stress_theta,2),"Mpa noraml, ",round(stress_shear,2),"Mpa shear "

#Given
M = 10           #KN-m moment
v = 8.0            #KN - shear Stress 
stress_allow = 8 #MPa - The maximum allowable stress
shear_allow_per = 1.4      #Mpa - The allowable stress perpendicular to grain
stress_allow_shear = 0.7   #MPa - The maximum allowable shear stress
#Caliculations 

S = M*(10**6)/stress_allow #mm3 
#lets arbitarly assume h = 2b
#S = b*(h**2)/6
h = pow(12*S,0.333)              #The depth of the beam
b = h/2                          #mm The width of the beam
A = h*b                          #mm2 The area of the crossection , assumption
stress_shear = 3*v*(10**3)/(2*A) #Mpa The strear stress 
if stress_shear<stress_allow_shear:
    print "The stress developed ",round(stress_shear,2)," is in allowable ranges for ",round(A,2),"mm2 area"
else:
    print "The stress developed",stress_shear," is in non allowable ranges",A,"area"
Area_allow = v*(10**3)/shear_allow_per  #mm - the allowable area
print "The minimum area is ",Area_allow ,"mm2"

#Given
stress_allow = 24 #ksi - The maximum allowable stress
stress_allow_shear = 14.5   #ksi- The maximum allowable shear stress
M_max  = 36 #k-ft The maximum moment
l = 16 #in-The length of the rod
w = 2 #k/ft - The force distribution on the rod
A = l*w
R_A = 6.4 #k - The reaction at A
R_B = 25.6 #k - the reaction at B
v_max = R_B-l*w #kips the maximum stress, from diagram
#W8x24 is used from the appendix table 3 and 4 
l =0.245 #in - W8x24 crossesction length
#Caliculations 

stress_xy = v_max/A        #ksi the approximate shear stress 
if stress_xy < stress_allow_shear:
    print "W8x24 gives the allowable ranges of shear stress"
else:
     print "W8x24 doesnot gives the allowable ranges of shear stress"
k = 7.0/8 #in the distance from the outer face of the flange to the webfillet
#at+kt should not exceed 0.75 of yeild stress
#a1t+2kt should not exceed 0.75 of yeild stress
Stress_yp = 36 #Ksi - The yeild stress
t = 0.245 #in thickness of the web
#support a 
a = R_A/(0.75*Stress_yp*t)-k #in lengths of the bearings
#support b
a_1 = R_B/(0.75*Stress_yp*t)-2*k #in lengths of the bearings
print "lengths of the bearing at A ",round(a,3),"in"
print "lengths of the bearing at B",round(a_1,3),"in"
    



#given 
hp = 63000              #horse power
T = hp*20*(10**-3)/63   #k-in the torsion implies due to horse power
stress_allow_shear = 6  #ksi- The maximum allowable shear stress
M_ver = 6.72/2          #k-in the vertical component of the moment 
M_hor = 9.10            #k-in the horizantal component of the moment 
#Caliculations 

M = pow(((M_ver**2)+(M_hor**2)),0.5)                                 #K-in The resultant 
d = pow((16*(((M**2)+(T**2))**0.5)/(stress_allow_shear*3.14)),0.333) #in, The suggested diameter from derivation
print "The suggested diameter is",round(d,2),"in"

