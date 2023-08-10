import math
#Initilization of variables
W=1000 #N
Lab=1 #m
Lac=0.6 #m
theta=60 #degree #angle made by the beam with the horizontal
#Calculations
Q=(W*Lac*math.cos(theta*180/math.pi))/(Lab*math.cos(theta*180/math.pi)) #N # from eq'n 2
P=W-Q #N # from eq'n 1
#Results
print('The load taken by man P is %d N'%P)
print('The load taken by man Q is %d N'%Q)

#Initilization of variables
F=1000 #N
Lab=1 #m
Lbc=0.25 #m
Lac=1.25 #m
#Calculations
Rb=(F*Lac)/Lab #N # from eq'n 2
Ra=Rb-F #N # fom eq'n 1
#Results
print('The reaction (downwards)at support A is %d N'%Ra)
print('The reaction (upwards)at support B is %d N'%Rb)

#Inilitization of variables
Lab=12 #m
Mc=40 #kN-m 
Md=10 #kN-m
Me=20 #kN-m
Fe=20 #kN #force acting at point E
#Calculations
Xa=-(Fe) #kN #take sum Fx=0
Rb=(Md+Me-Mc)/Lab #N #take moment at A
Ya=-Rb #N #take sum Fy=0
#Results
print('The vertical reaction (upwards) at A is %f kN'%Ya)
print('The horizontal reaction (towards A) is %d kN'%Xa)
print('The reaction (downwards) at B is %f kN'%Rb)

import numpy
#Initilization of variables
W=1000 #N
Lad=7.5 #m
Lae=1.5 #m
La1=3.75 #m #distance of 1st 1000N load from pt A
La2=5 #m #distance of 2nd 1000N load from pt A
La3=6 #m # distance of 3rd 1000N load from pt A
# Calculations (part1)
#using matrix to solve the given eqn's 1 & 2
A=numpy.matrix('1 -2.5;3.5 -5')
B=numpy.matrix('1000;7250')
C=numpy.linalg.inv(A)*B
#Calculations (part 2)
#Consider combined F.B.D of beams AB,BC &CD. Take moment at A
Re=((W*La1)+(W*La2)+(W*La3)+(C[1]*Lad)-(C[0]*La3))/Lae #N
Ra=C[1]-Re-C[0]+(3*W) #N #Taking sum of forces in Y direction
#Results
print('The reaction at F i.e Rf is %f N'%C[0])
print('The reaction at D i.e Rd is %f N'%C[1])
print('The reaction at pt E i.e Re is %f N'%Re)
print('The reaction at pt A i.e Ra is %f N'%Ra) 

# Initilization of variables
W=100 # N #force acting at D
AB=50 # N # weight of bar ab
CD=50 # N # weight of bar cd
# Calculations
# From  the derived expression the value of the angle is given as,
theta=math.degrees(math.atan(5/17.5)) #degrees
# Results
print('The angle theta is %f degrees'%theta)

#Initilization of variables
Ws=2 #kN #weight of scooter
Wd=0.5 #kN #weight of driver
Lab=1 #m
Led=0.8 #m
Leg=0.1 #m
#Calculations
Rc=((2*Leg)+(Wd*Led))/Lab #kN #take moment at E
Ra=(2+Wd-Rc)/2 # kN # as Ra=Rb,(Ra+Rb=2*Ra)
Rb=Ra # kN
#Results
print('The reaction at wheel A is %f kN'%Ra)
print('The reaction at wheel B is %f kN'%Rb)
print('The reaction at wheel C is %f kN'%Rc)

#Initilization of variables
W1=15 #N #up
W2=60 #N #down
W3=10 #N #up
W4=25 #N #down
Lab=1.2 #m
Lac=0.4 #m
Lcd=0.3 #m
Ldb=0.5 #m
Lad=0.7 #m
Leb=0.417 #m #Leb=Lab-x
#Calculations
#(a) A single force
Ry=W1-W2+W3-W4 #N #take sum Fy=0
x=((-W2*Lac)+(W3*Lad)-(W4*Lab))/(Ry) #m
# (b) Single force moment at A
Ma=(Ry*x) #N-m
# Single force moment at B
Mb=W2*Leb #N-m
#Results
print('The reaction for single force is %d N'%Ry)
print('The distance of Ry from A is %f m'%x)
print('The moment at A is %d N-m'%Ma)
print('The moment at B is %f N-m'%Mb)

#Initilization of variables
Ra=5000 #N
Ma=10000 #Nm
alpha=60 #degree #angle made by T1 with the pole
beta=45 #degree #angle made by T2 with the pole
theta=30 #degree #angle made by T3 with the pole
Lab=6 #m
Lac=1.5 #m
Lcb=4.5 #m
#Calculations
T3=Ma/(4.5*math.sin(theta*math.pi/180)) #N #take moment at B
# Now we use matrix to solve eqn's 1 & 2 simultaneously,
A=numpy.matrix('-0.707 0.8666;0.707 0.5')
B=numpy.matrix('2222.2;8848.8')
C=numpy.linalg.inv(A)*B
#Results
print('Tension in wire 1 i.e T1 is %f N \n'%C[1])
print('Tension in wire 2 i.e T2 is %f N \n'%C[0])
print('Tension in wire 3 i.e T3 is %f N \n'%T3)

#Initilization of variables
w=2000 #N/m
Lab=3 #m
#Calculations
W=w*Lab/2 #N# Area under the curve
Lac=(2/3)*Lab #m#centroid of the triangular load system
Rb=(W*Lac)/Lab #N #sum of moment at A
Ra=W-Rb #N
#Results
print('The resultant of the distibuted load lies at %d m'%Lac)
print('The reaction at support A is %d N'%Ra)
print('The reaction at support B is %d N'%Rb)

#Initiization of variables
w=1500 #N/m
x=4 #m
L=4 #m
#Calculations
k=x**2/w #m**3/N
#Solving the intergral we get
W=L**3/(3*k) #N
x_bar=L**4/(4*k*W) #m
#Result
print("The resultant is %d N and the line of action of the force is %d m"%(W,x_bar))

# Initilization of variables
w1=1.5 #kN/m # intensity of varying load at the starting point of the beam
w2=4.5 #kN/m # intensity of varying load at the end of the beam
l=6 #m # ength of the beam
# Calculations
# The varying load distribution is divided into a rectangle and a right angled triangle
W1=w1*l #kN # where W1 is the area of the load diagram(rectangle ABED)
x1=l/2 #m # centroid of the rectangular load system
W2=(w2-w1)*l/2 #kN # where W1 is the area of the load diagram(triangle DCE)
x2=2*l/3 #m # centroid of the triangular load system
W=W1+W2 #kN # W is the resultant
x=((W1*x1)+(W2*x2))/W #m # where x is the distance where the resultant lies
#Results
print('The resultant of the distributed load system is %d kN'%W)
print('The line of action of the resulting load is %f m'%x)

# Initiization of variables
W1=10 #kN #point load acting at D
W2=20 #kN # point load acting at C at an angle of 30 degree
W3=5 #kN/m # intensity of udl acting on span EB of 4m
W4=10 #kN/m # intensity of varying load acting on span BC of 3m
M=25 #kN-m # moment acting at E
theta=30 #degree # angle made by 20 kN load with the beam
Lad=2 #m
Leb=4 #m
Laf=6 #m #distance between the resultant of W3 & point A
Lac=11 #m
Lag=9 #m #distance between the resultant of W4 and point A
Lbc=3 #m
Lab=8 #m
# Calculations
Xa=20*math.cos(theta*math.pi/180) #kN # sum Fx=0
Rb=((W1*Lad)+(-M)+(W3*Leb*Laf)+(W2*math.sin(theta*math.pi/180)*Lac)+((W4*Lbc*Lag)/2))/Lab #kN # taking moment at A
Ya=W1+(W2*math.sin(theta*math.pi/180))+(W3*Leb)+(W4*Lbc/2)-Rb #kN # sum Fy=0
Ra=math.sqrt(Xa**2+Ya**2) #kN # resultant at A
#Results
print('The horizontal reaction at A i.e Xa is %f kN'%Xa)
print('The vertical reaction at A i.e Ya is %d kN'%Ya)
print('The reaction at A i.e Ra is %d kN'%Ra)
print('The reaction at B i.e Rb is %d kN'%Rb)

# Initilization of variables
h=4 #m #height of the dam wall
rho_w=1000 # kg/m**3 # density of water
rho_c=2400 # kg/m**3 # density of concrete
g=9.81 # m/s**2
# Calculations
P=(rho_w*g*h**2)/2 # The resultant force due to water pressure per unit length of the dam
x=(2/3)*h #m # distance at which the resutant of the triangular load acts 
b=math.sqrt((2*P*h)/(3*h*rho_c*g)) # m # eq'n required to find the minimum width of the dam
# Results
print('The minimum width which is to be provided to the dam to prevent overturning about point B is %f m'%b)

