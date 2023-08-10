import math
#Initilization of variables
N=1800 # r.p.m # Speed of the shaft
t=5 # seconds # time taken to attain the rated speed # case (a)
T=90 # seconds # time taken by the unit to come to rest # case (b)
# Calculations
omega=(2*math.pi*N)/(60)
# (a)
# we take alpha_1,theta_1 & n_1 for case (a)
alpha_1=omega/t # rad/s**2 #
theta_1=(omega**2)/(2*alpha_1) # radian
# Let n_1 be the number of revolutions turned,
n_1=theta_1*(1/(2*math.pi))
# (b)
# similarly we take alpha_1,theta_1 & n_1 for case (b)
alpha_2=(omega/T) # rad/s**2 # However here alpha_2 is -ve
theta_2=(omega**2)/(2*alpha_2) # radians
# Let n_2 be the number of revolutions turned,
n_2=theta_2*(1/(2*math.pi))
# Results
print('(a) The no of revolutions the unit turns to attain the rated speed is %f'%n_1)
print('(b) The no of revolutions the unit turns to come to rest is %f'%n_2)

# Initilization of variables
r=1 # m # radius of the cylinder
v_c=20 # m/s # velocity of the cylinder at its centre
# Calculations
# The velocity of point E is given by using the triangle law as,
v_e=math.sqrt(2)*v_c # m/s 
# Similarly the velocity at point F is given as,
v_f=2*v_c # m/s 
# Results
print('The velocity of point E is %f m/s'%v_e)
print('The velocity of point F is %f m/s'%v_f)

import numpy
# Initilization of Variables
v_1=3 # m/s # uniform speed of the belt at top
v_2=2 # m/s # uniform speed of the belt at the bottom
r=0.4 # m # radius of the roller
# Calculations
# equating eq'ns 2 & 4 and solving for v_c & theta' (angular velocity). We use matrix to solve the eqn's
A=numpy.matrix([[1,r],[1,-r]])
B=numpy.matrix([[v_1],[v_2]])
C=numpy.linalg.inv(A)*B
# Results
print('The linear velocity (v_c) at point C is %f m/s'%C[0])
print('The angular velocity at point C is %f radian/seconds'%C[1])
# The answer of angular velocity is incorrect in the book

# Initilization of Variables
l=1 # m # length of bar AB
v_a=5 # m/s # velocity of A
theta=30 # degree # angle made by the bar with the horizontal
# Calculations
# From the vector diagram linear velocity of end B is given as,
v_b=v_a/math.tan(theta*math.pi/180) # m/s 
# Now let the relative velocity be v_ba which is given as,
v_ba=v_a/math.sin(theta*math.pi/180) # m/s
# Now let the angular velocity of the bar be theta_a which is given as,
theta_a=(v_ba)/l # radian/second
# Velocity of point A
v_a=(l/2)*theta_a # m/s
# Magnitude of velocity at point C is,
v_c=v_a # m/s # from the vector diagram
# Results
print('(a) The angular velocity of the bar is %f radian/second'%theta_a)
print('(b) The velocity of end B is %f m/s'%v_b)
print('(c) The velocity of mid point C is %f m/s'%v_c)

# Initilization of Variables
r=0.12 # m # length of the crank
l=0.6 # m # length of the connecting rod
N=300 # r.p.m # angular velocity of the crank
theta=30 # degree # angle made by the crank with the horizontal
# Calculations
# Now let the angle between the connecting rod and the horizontal rod be phi
phi=math.asin(((r*math.sin(theta*math.pi/180))/(l))*math.pi/180) # degree
# Now let the angular velocity of crank OA be omega_oa, which is given by eq'n
omega_oa=(2*math.pi*N)/(60) # radian/second
# Linear velocity at A is given as,
v_a=r*omega_oa # m/s
# Now using the sine rule linear velocity at B can be given as,
v_b=(v_a*math.sin(35.7*math.pi/180))/(math.sin(84.3*math.pi/180)) # m/s
# Similarly the relative velocity (assume v_ba) is given as,
v_ba=(v_a*math.sin(60*math.pi/180))/(math.sin(84.3*math.pi/180))
# Angular velocity (omega_ab) is given as,
omega_ab=v_ba/l # radian/second
# Results
print('(a) The angular velocity of the connecting rod is %f radian/second'%omega_ab)
print('(b) The velocity of the piston when the crank makes an angle of 30 degree is %f m/s'%v_b)

# Initiization of variables
r=1 # m # radius of the cylinder
v_c=20 # m/s # velocity at the centre
# Calculations
# Angular velocity is given as,
omega=v_c/r # radian/second
# Velocity at point D is
v_d=omega*math.sqrt(2)*r # m/s # from eq'n 1
# Now, the velocity at point E is,
v_e=omega*2*r # m/s 
# Results
print('The velocity at point D is %f m/s'%v_d)
print('The velocity at point E is %f m/s'%v_e)

import numpy
# Initilization of Variables
r=5 # cm # radius of the roller
AB=0.1 # m
v_a=3 # m/s # velocity at A
v_b=2 # m/s # velocity at B
# Calculations
# Solving eqn's 1 & 2 using matrix for IA & IB we get,
A=([[-2,3],[1,1]])
B=numpy.matrix([[0],[AB]])
C=numpy.linalg.inv(A)*B
d1=C[1]*10**2 # cm # assume d1 for case 1
# Similary solving eqn's 3 & 4 again for IA & IB we get,
P=numpy.matrix([[-v_b,v_a],[1,-1]])
Q=numpy.matrix([[0],[AB]])
R=numpy.linalg.inv(P)*Q
d2=R[1]*10**2 # cm # assume d2 for case 2
# Results
print('The distance d when the bars move in the opposite directions are %f cm'%d1)
print('The distance d when the bars move in the same directions are %f cm'%d2)

# Initilization of Variables
v_c=1 # m/s # velocity t the centre
r1=0.1 # m 
r2=0.20 # m
EB=0.1 # m
EA=0.3 # m
ED=math.sqrt(r1**2+r2**2) # m
# Calculations
# angular velocity is given as,
omega=v_c/r1 # radian/seconds
# Velocit at point B
v_b=omega*EB # m/s 
# Velocity at point A
v_a=omega*EA # m/s
# Velocity at point D
v_d=omega*ED # m/s
# Results
print('The velocity at point A is %f m/s'%v_a)
print('The velocity at point B is %f m/s'%v_b)
print('The velocity at point D is %f m/s'%v_d)

# Initilization of variables
l=1 # m # length of bar AB
v_a=5 # m/s # velocity at A
theta=30 # degree # angle made by the bar with the horizontal
# Calculations
IA=l*math.sin(theta*math.pi/180) # m
IB=l*math.cos(theta*math.pi/180) # m
IC=0.5 # m # from triangle IAC
# Angular veocity is given as,
omega=v_a/(IA) # radian/second
v_b=omega*IB # m/s
v_c=omega*IC # m/s
# Results
print('The velocity at point B is %f m/s'%v_b)
print('The velocity at point C is %f m/s'%v_c)

# Initilization of variables
v_a=2 # m/s # velocity at end A
r=0.05 # m # radius of the disc
alpha=30 # degree # angle made by the bar with the horizontal
# Calculations 
# Soving eqn's 1 & 2 and substuting eqn 1 in it we get eq'n for omega as,
omega=(v_a*(math.sin(alpha*math.pi/180))**2)/(r*math.cos(alpha*math.pi/180)) # radian/second
# Results
print('The anguar veocity of the bar is %f radian/second'%omega)

# Initilization of variables
l=0.6 # m 
r=0.12 # m 
theta=30 # degree # angle made by OA with the horizontal
phi=5.7 # degree # from EX 21.5
N=300
# Calculations
# Let the angular velocity of the connecting rod be (omega_ab) which is given from eqn's 1 & 4 as,
omega_oa=(2*math.pi*N)/(60) # radian/ second
# Now,in triangle IBO.
IB=(l*math.cos(phi*math.pi/180)*math.tan(theta*math.pi/180))+(r*math.sin(theta*math.pi/180)) # m
IA=(l*math.cos(phi*math.pi/180))/(math.cos(theta*math.pi/180)) # m
# from eq'n 5
v_b=(r*omega_oa*IB)/(IA) # m/s
# From eq'n 6
omega_ab=(r*omega_oa)/(IA) # radian/second
# Results
print('The velocity at B is %f m/s'%v_b)
print('The angular velocity of the connecting rod is %f radian/second'%omega_ab)

# Initilization of variables
omega_ab=5 # rad/s # angular veocity of the bar
AB=0.20 # m
BC=0.15 # m
CD=0.3 # m
theta=30 # degree # where theta= angle made by AB with the horizontal
alpha=60 # degree # where alpha=angle made by CD with the horizontal
# Calculations
# Consider triangle BIC
IB=math.sin(alpha*math.pi/180)*BC*1 # m
IC=math.sin(theta*math.pi/180)*BC*1 # m
v_b=omega_ab*AB # m/s
# let the angular velocity of the bar BC be omega_bc
omega_bc=v_b/IB # radian/second
v_c=omega_bc*IC # m/s
# let the angular velocity of bar DC be omega_dc
omega_dc=v_c/CD # radian/second
# Results
print('The angular velocity of bar BC is %f rad/s'%omega_bc)
print('The angular velocity of bar CD is %f rad/s'%omega_dc)

