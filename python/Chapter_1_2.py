import math

#Variable declaration
Fac = 750                             #Force on rod AC(lb)
D = 0.375                             #Diameter at the upper junction of rod ABC(in)


#Calculation         
#Case(a)
A=(1/4.0)*((math.pi)*math.pow(D,2))   #Area at the upper junction of rod ABC(in^2) 
tA=(Fac/A)                            #Shearing Stress in Pin A(psi)   
#Case(b) 
Ab=(1/4.0)*((math.pi)*pow(0.25,2))    #Area at the lower junction of rod ABC(in^2)
tC=(((1/2.0)*Fac)/Ab)                 #Shearing Stress in Pin C(psi)
#Case(c)
Anet=(3/8.0)*(1.25-0.375)             #Area of cross section at A(in^2)
sA=(Fac/Anet)                         #Largest Normal Stress in Link ABC(psi)
#Case(d)
F1=750/2                              #Force on each side(lb)
Ad=(1.25*1.75)                        #Area at junction B(in^2)
tB=(F1/Ad)                            #Average Shearing Stress at B
#Case(e)
Ae=0.25*0.25                          #Area at point C(in^2)
sB=(F1/Ae)                            #Bearing Stress in Link at C


#Result
print('Case(a): Shearing Stress in Pin A = %.1f psi' %tA)
print('Case(b): Shearing Stress in Pin C = %.f psi' %tC)
print('Case(c): Largest Normal Stress in Link ABC = %.f psi' %sA)
print('Case(d): Average Shearing Stress at B = %.f psi' %tB)
print('Case(e): Bearing Stress in Link at C = %.f psi' %sB)

import math

#Variable declaration
P = 120                                             #Maximum allowable tension force 
s = 175                                             #Maximum allowable stress
t = 100                                             #Maximum allowable stress
Sb = 350                                            #Maximum allowable stress


#Calculation
#Case(a)
F1=P/2         #Current(A)
d=math.sqrt(((P/2.0)*1000)/((22/(4*7.0))*(100000000)))  #Diameter of bolt(m)
d=d*1000                                            #Diameter of bolt(mm)
d=round(d,0)                                        #Rounding of the value of diameter of bolt(mm)
Ad=(0.020*0.028)                                    #Area of cross section of plate                                    
tb=((P*1000)/Ad)/(1000000)                          #Stress between between the 20-mm-thick plate and the 28-mm-diameter bolt
tb=round(tb,0)                                      #Rounding of the above calculated stress to check if it is less than 350
a=(P/2)/((0.02)*(175))                              #Dimension of cross section of ring 
a=round(a,2)                                        #Rounding dimension of cross section of ring to two decimal places
b=28 + (2*(a))                                      #Dimension b at Each End of the Bar
b=round(b,2)                                        #Rounding the dimension b to two decimal places 
h=(P)/((0.020)*(175))                               #Dimension h of the Bar
h=round(h,1)                                        #Rounding dimension h of bar to 1 decimal place


#Result
print ('Case(a): Diameter of the bolt = %.f mm' %d)
print ('Case(b): Dimension b at Each End of the Bar = %.f mm' %b)
print ('Case(c): Dimension h of the Bar = %f mm' %h)

import math

#Variable declaration
Su = 600    #ultimate normal stress(MPa)   
FS = 3.3    #Factor of safety with respect to failure
tU=350      #Ultimate shearing stress(MPa)
Cx=40       #X Component of reaction at C(kN)
Cy=65       #Y Component of reaction at C(kN)
Smax=300    #Allowable bearing stress of the steel 

#Calculation
C=math.sqrt((math.pow(40,2))+(math.pow(65,2)))

#Case(a)
P=(15*0.6 + 50*0.3)/(0.6)  #Allowable bearing stress of the steel(MPa)
Sall=(Su/FS)               #Allowable Stress(MPa)
Sall=round(Sall,1)         #Rounding Allowable stress to 1 decimal place(MPa)
Areqa=(P/(Sall*(1000)))    #Cross Sectional area(m^2)
Areqa=round(Areqa,5)       #Rounding cross sectional area to 5 decimal places(m^2)
dAB=math.sqrt(((Areqa)*(4))/(22/7)) #Diameter of AB(m)
dAB=dAB*1000                        #Diameter of AB(mm)
dAB=round(dAB,2)                    #Rounding Diameter of AB(mm)

#Case(b)
tALL=tU/FS                          #Stress(MPa)
tALL=round(tALL,1)                  #Rounding of Stress
AreqC=((C/2)/tALL)                  #Cross sectional area(m^2)
AreqC=AreqC*1000                    
AreqC=round(AreqC,0)                #Rounding the cross sectional area
dC=math.sqrt((4*AreqC)/(22/7))      #Diameter at point C
dC=round((dC+1),0)                  #Rounding of the diameter at C

#Case(c)

Areq=((C/2)/Smax)                   
Areq=Areq*1000                      #Cross sectional area(mm^2)
t=(Areq/22)                         #Thickness of the bracket
t=round(t,0)

#Result
print ('Case(a): Diameter of the bolt = % f mm' %dAB)
print ('Case(a): Dimension b at Each End of the Bar = % f mm' %dC)
print ('Case(a): Dimension h of the Bar = % f mm' %t)

import math

#Variable declaration
tU=40                                     #ultimate tensile stress
sU=60                                     #ultimate shearing stress
FS=3                                      #Mimnimum factor of safety
dA=(7/16)                                 #Diameter of bolt at A(in)
dB=3/8                                    #Diameter of bolt at B(in) 
dD=3/8                                    #Diameter of bolt at D(in)
dC=1/2                                    #Diameter of bolt at C(in)


#Calculation
Sall=(sU/FS)                              #Total tensile stress(kips)
B=Sall*((1/4)*(22/7)*(pow((7/16),2)))     #Allowable force in the control rod(kips)
C1=1.75*(B)                               #Control Rod(kips)
tall=(tU/FS)                              #Total shearing stress
B=2*(tall*(1/4)*(22/7)*(3/8)*(3/8))       #Allowable magnitude of the force B exerted on the bolt
C2=1.75*B                                 #Bolt at B(kips)                       
D=B                                       #Bolt at D. Since this bolt is the same as bolt B, the allowable force is same(kips) 
C3=2.33*D                                 #Bolt at D(kips)
C4=2*(tall*(1/4)*(22/7)*(1/2)*(1/2))      #Bolt at C(kips)      
list1=[C1,C2,C3,C4]                       #Adding all the maximum allowable forces on C(kips) 


#Result
print ('Case(a): Control Rod  = % f kips' %C1)
print ('Case(b): Bolt at B = % f kips' %C2)
print ('Case(c): Bolt at D = % f kips' %C3)
print ('Case(d): Bolt at C = % f kips' %C4)
print ('Summary. We have found separately four maximum allowable values of the force C. In order to satisfy all these criteria we must choose the smallest value, namely:= % f kips' %min(list1));



