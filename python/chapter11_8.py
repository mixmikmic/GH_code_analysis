#Given 
import math
h =  60             #mm - the length of the crossection 
b = 100             #mm - the width of hte crossection 
E = 200             #Gpa - The youngs modulus
stress_cr  = 250    #Mpa - The proportionality limit
#Caliculations 

I = b*(h**3)/12     #mm3 The momentof inertia of the crossection
A = h*b             #mm2 - The area of teh crossection 
#From Eulier formula
r_min  =  pow((I/A),0.5)                            #mm - The radius of the gyration 
#(l/r)**2= (pi**2)*E/stress_cr                      #From Eulier formula
l = (((math.pi**2)*E*(10**3)/stress_cr)**0.5)*r_min #mm - the length after which the beam starts buckling
print "The length after which the beam starts buckling is ",round(l,0),"mm"

#Given
import math
L = 15                                    #ft - The length of the each rod
A = 46.7                                  #in2 - The length of the crossection 
r_min = 4                                 #in - The radius of gyration
stress_yp = 36                            #Ksi - the yielding point stress
E = 29*(10**3)                            #ksi - The youngs modulus
C_c = ((2*(math.pi**2)*E/stress_yp)**0.5) #Slenderness ratio L/R
C_s = L*12/r_min                          # Slenderness ratio L/R of the present situation 
#According to AISC formulas 
if C_s <C_c :
    print "a)The following approch is solvable"
else: 
    print "The caliculation is not possible"
F_S = 5.0/3 +3*C_s/(8*C_c) -(3*C_s**3)/(8*C_c**3)        #Safety factor 
Stress_all = (1 - (C_s**2)/(2*C_c**2))*stress_yp/F_S    #The allowable strees 
print "a) The allowable stress in this case is",round(Stress_all,2),"Kips" 
#Part - B
#Given
L = 40                                   #ft - The length of the each rod
A = 46.7                                  #in2 - The length of the crossection 
r_min = 4                                 #in - The radius of gyration
stress_yp = 36                            #Ksi - the yielding point stress
E = 29*(10**3)                            #ksi - The youngs modulus
C_c = ((2*(math.pi**2)*E/stress_yp)**0.5) #Slenderness ratio L/R
C_s = L*12/r_min                          # Slenderness ratio L/R of the present situation 
#According to AISC formulas 
if C_s <C_c :
    print "b) The following approch is solvable"
else: 
    print "The caliculation is not possible"
F_S = 5.0/3 +3*C_s/(8*C_c) -(3*C_s**3)/(8*C_c**3)        #Safety factor 
Stress_all = (1 - (C_s**2)/(2*C_c**2))*stress_yp/F_S    #The allowable strees 
print "b) The allowable stress in this case is",round(Stress_all,2),"Kips"

#Given
import math
L = 15                                    #ft - The length of the each rod      
p  = 200                                  #Kips The concentric load applied 
r_min = 2.10                              #in - The radius of gyration
stress_yp =   50                          #Ksi - the yielding point stress
E = 29*(10**3)                            #ksi - The youngs modulus
C_c = ((2*(math.pi**2)*E/stress_yp)**0.5) #Slenderness ratio L/R
C_s = L*12/r_min  #Slenderness ratio L/R present situation
if C_s <C_c :
    print "a)The following approch is solvable"
else: 
    print "The caliculation is not possible"
F_S = 5.0/3 +3*C_s/(8*C_c) -(3*C_s**3)/(8*C_c**3)        #Safety factor 
Stress_all = (1 - (C_s**2)/(2*C_c**2))*stress_yp/F_S    #The allowable strees
a = p/Stress_all #in2 the alloawble area of the beam 
print "The allowable stress in this case is",round(Stress_all,2),"Kips"
print "This stress requires ",round(a,2),"in2"
if a <11.5:
    print "This case is satisfying W8x24 section" #From AISC Manual 
else:
    print "This case is not satisfying W8x24 section"
                                #The ans are quiet varying because of rounding

#Given
import math
L = 15.0                                        #ft - The length of the each rod
A = 46.7                                        #in2 - The length of the crossection 
r_min = 4                                       #in - The radius of gyration
stress_yp = 36.0                                #Ksi - the yielding point stress
E = 29*(10**3)                                  #ksi - The youngs modulus
lamda = L*12*((stress_yp/E)**0.5)/(4*(math.pi)) #column slenderness ratio
if lamda<1.5:
    print "The following approach is right"
else:
    print "The following approach is wrong"
stress_cr = (0.658**(lamda**2))*stress_yp    #Ksi - The critical stress  
P_n = stress_cr*A                            #Kips #Nominal compressive strength 
o = 0.85                                     #Resistance factor
p_u = o*P_n                                  #Kips ,column design compressive strength 
print "column design compressive strength ",p_u,"Kips"

#Given 
#FOR FLANGS
l = 5 #in - The length of the flang
b = 5 #in - Teh width of the flang
t  = 0.312 #in - the thickness of the flang
L = 20 #in - Length of the beam, Extracted from AISC manuals
A = 4.563 #in2 - The area of crossection of the beam
r = 1.188 #in - radius of the gyration, Extracted from AISC manuals 
#b/t- value of the flang 
k = (5 -t)/(2*t) #b/t ratio 
#AISC, lets check maximum allowable stress for slang
Stressf_all = 23.1 - 0.79*k #ksi The maximum allowable stress in case of flang,AISC

#web width thickness ratio
k_2 = (5 -2*t)/(t)
if k_2<16:
    Stressw_all = 19 #ksi - The allowable stress in case of web width
    
#a) Overall buckling investment 
k_3 = L/r #slenderness ratio
Stressb_all = 20.2 - 0.216*k_3#ksi The maximum allowable stress in case of Buckling,AISC
p_allow = A*Stressb_all #Kips The allowable concentric load 

#b) Overall buckling investment
L_2 = 60 #in 
k_3 = L_2/r #slenderness ratio
Stressb_all_2 = 20.2 - 0.126*k_3#ksi The maximum allowable stress in case of Buckling,AISC
p_allow_2 = A*Stressb_all_2 #Kips The allowable concentric load 

print "The maximum allowable stress in case of web width",round(Stressw_all,2),"Ksi"
print "The maximum allowable stress in case of flang",round(Stressf_all,2),"Ksi"
print "a) The maximum allowable load in case of Buckling",round(p_allow,2),"Kips"
print "b) The maximum allowable load in case of Buckling",round(p_allow_2,2),"Kips"


     

import math
P = 200.0       #K The force on the beam 
L = 15          #ft - The length of the rod
F_y = 50.0 #Ksi 
F_a = F_y/(5.0/3) #Ksi -AISC MANUAL ,allowable axial stress if axial force is alone
F_b = F_a         #Allowable compressive bending stress
M_1 = 600.0       #k-in - The moment acting on the ends of the rod
M_2 = 800.0       #k-in - the moment acting on the other end of teh rod
B_x = 0.264       #in - Extracted from AISC manual 
E = 29*(10**3) 
A = P/F_a + M_2*B_x/F_b #in2- The minimum area 
print "The minimum area is ",round(A,2),"in2"
#we will select W10x49 section 
A_s = 14.4         #in2 - The area of the section 
r_min  = 2.54      #in The minimum radius 
r_x = 4.35         #in 
f_a = P/A_s        #Ksi- The computed axial stress
f_b =  M_2*B_x/A_s #Computed bending stess
C_c = ((2*(math.pi**2)*E/F_y)**0.5) #Slenderness ratio L/R
C_s = L*12/r_min                    # Slenderness ratio L/R of the present situation
if C_s <C_c :
    print "The following approch is solvable"
else: 
    print "The caliculation is not possible"
F_a_1 = 19.3                              #Ksi - AISC lets try this
c_m = 0.6 - 0.4*(-M_1/M_2) 
F_e = (12*(math.pi**2)*E)/(23*(L*12/r_x)**2) 
k = f_a/F_a_1 + c_m*f_b*(1-(f_a/F_e))/F_b #Condition mentioned in AISC
if k>1:
    print "The following W10x49 section is not satisfying our constraints since f_a/F_a_1 + c_m*f_b*(1-(f_a/F_e))/F_b",round(k,3),">1"
else:
    print "The following W10x49 section is satisfying our constraints since f_a/F_a_1 + c_m*f_b*(1-(f_a/F_e))/F_b",k,"<1"
 
#trail - 2
#Lets take  W10 x 60
A_s = 17.6                         #in2 - The area of the section 
r_min  = 2.57                      #in The minimum radius 
r_x = 4.39                         #in 
f_a = P/A_s                        #Ksi- The computed axial stress
f_b =  M_2*B_x/A_s                 #Computed bending stess
C_c = ((2*(math.pi**2)*E/F_y)**0.5) #Slenderness ratio L/R
C_s = L*12/r_min                    # Slenderness ratio L/R of the present situation
if C_s <C_c :
    print "The following approch is solvable"
else: 
    print "The caliculation is not possible"
F_a_1 = 19.3                              #Ksi - AISC lets try this
c_m = 0.6 - 0.4*(-M_1/M_2) 
F_e = (12*(math.pi**2)*E)/(23*(L*12/r_x)**2) 
k = f_a/F_a_1 + c_m*f_b*(1-(f_a/F_e))/F_b #Condition mentioned in AISC
if k>1:
    print "The following W10x49 section is not satisfying our constraints since f_a/F_a_1 + c_m*f_b*(1-(f_a/F_e))/F_b",round(k,3),">1"
else:
    print "The following W10x49 section is satisfying our constraints since f_a/F_a_1 + c_m*f_b*(1-(f_a/F_e))/F_b",round(k,2),"<1"
 

