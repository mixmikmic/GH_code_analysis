##Example 5.2 Equations of equilibrium

import math

#Initialization of Variables
W=2000 #N
Lab=2 #m #length of the member from the vertical to the 1st load of 2000 N
Lac=5 #m #length of the member from the vertical to the 2nd load of 2000 N
Lpq=3.5 #m

#Calculations
Rq=((W*Lab)+(W*Lac))/Lpq #N #take moment abt. pt P
Xp=Rq #N #sum Fx=0
Yp=2*W #N #sum Fy=0
Rp=math.sqrt(Xp**2+Yp**2) #N

#Resuts
print('The reaction at P is' ,Rp ,'N')
print('The reaction at Q is ',Rq ,'N')

##Example 5.3 Equations of equilibrium

import math,numpy
#Initilization of vaiables
W=25 #N # self weight of the ladder
M=75 #N # weight of the man standing o the ladder
theta=63.43 #degree # angle which the ladder makes with the horizontal
alpha=30 #degree # angle made by the string with the horizontal
Loa=2 #m # spacing between the wall and the ladder
Lob=4 #m #length from the horizontal to the top of the ladder touching the wall(vertical)

#Calculations
#Using matrix to solve the simultaneous eqn's 3 & 4
A=numpy.matrix('2 -4; 1 -0.577')
B=numpy.matrix('100;100')
C=numpy.linalg.inv(A)*B

#Results
print('The reaction at A i.e Ra is ',C[0] ,'N')
print('The reaction at B i.e Rb is ',C[1] ,'N')

#Calculations
T=C[1]/math.cos(math.radians(alpha)) #N # from (eqn 1)

#Results
print('The required tension in the string is ',T, 'N')

##Example 5.4 Equations of Equilibrium

import math
#Initilization of variables
W=100 #N
theta=60 #degree angle made by the ladder with the horizontal
alpha=30 #degree angle made by the ladder with the vertical wall
Lob=4 #m  length from the horizontal to the top of the ladder touching the wall(vertical)
Lcd=2 #m  length from the horizontal to the centre of the ladder where the man stands

#Calculations
Lab=Lob*(1/math.cos(math.radians(alpha))) #m length of the ladder
Lad=Lcd*math.tan(math.radians(alpha)) #m
Rb=(W*Lad)/Lab #N take moment at A
Xa=Rb*math.sin(math.radians(theta)) #N  From eq'n 1
Ya=W+Rb*math.cos(math.radians(theta)) #N From eq'n 2

#Results
print('The reaction at B i.e Rb is ',Rb, 'N')
print('The horizontal reaction at A i.e Xa is ',Xa, 'N')
print('The vertical reaction at A i.e Ya is ',Ya,'N')

##Example 5.5 Equations of Equilibrium

import math
#Initilization of variables
W=100 #N self weight of the man
alpha=30 #degree angle made by the ladder with the wall
Lob=4 #m  length from the horizontal to the top of the ladder touching the wall(vertical)
Lcd=2 #m

#Calculations
# using the equiblirium equations
Ya=W #N  From eq'n 2
Lad=Lcd*math.tan(math.radians(alpha)) #m Lad is the distance fom pt A to the point where the line from the cg intersects the horizontal
Rb=(W*Lad)/Lob #N  Taking sum of moment abt A
Xa=Rb #N From eq'n 1

#Results
print('The horizontal reaction at A i.e Xa is ',Xa, 'N')
print('The vertical reaction at A i.e Ya is ',Ya,'N' )
print('The reaction at B i.e Rb is ',Rb ,'N')

##Example 5.6 Equations of Equilibrium

import math
#Initilization of variables
d=0.09 #m diametre of the right circular cylinder
h=0.12 #m height of the cyinder
W=10 #N  self weight of the bar
l=0.24 #m length of the bar

#Calculations
theta=math.degrees(math.atan(h/d))  #angle which the bar makes with the horizontal
Lad=math.sqrt(d**2+h**2) #m Lad is the length of the bar from point A to point B
Rd=(W*h*(math.cos(theta*math.pi/180)))/Lad #N  Taking moment at A
Xa=Rd*(math.sin(theta*math.pi/180)) #N  sum Fx=0.... From eq'n 1
Ya=W-(Rd*(math.cos(theta*math.pi/180))) #N  sum Fy=0..... From eq'n 2
Ra=math.sqrt(Xa**2+Ya**2) #resultant of Xa & Ya

#Results
print('The horizontal reaction at A i.e Xa is ',Xa, 'N')
print('The vertical reaction at A i.e Ya is ',Ya, 'N')
print('Therefore the reaction at A i.e Ra is ',Ra,'N')
print('The reaction at D i.e Rd is ',Rd,'N')

