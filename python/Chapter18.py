import numpy
# Initilization of variables
m_a=1 # kg # mass of the ball A
v_a=2 # m/s # velocity of ball A
m_b=2 # kg # mass of ball B
v_b=0 # m/s # ball B at rest
e=1/2 # coefficient of restitution
# Calculations
# Solving eqn's 1 & 2 using matrix for v'_a & v'_b,
A=numpy.matrix('1 2;-1 1')
B=numpy.matrix('2;1')
C=numpy.linalg.inv(A)*B
# Results
print('The velocity of ball A after impact is %f m/s'%C[0])
print('The velocity of ball B after impact is %f m/s'%C[1])

import numpy
# Initilization of variables
m_a=2 # kg # mass of ball A
m_b=6 # kg # mass of ball B
m_c=12 # kg # mass of ball C
v_a=12 # m/s # velocity of ball A
v_b=4 # m/s # velocity of ball B
v_c=2 # m/s # velocity of ball C
e=1 # coefficient of restitution for perfectly elastic body
# Calculations
# (A)
# Solving eq'n 1 & 2 using matrix for v'_a & v'_b,
A=numpy.matrix('2 6;-1 1')
B=numpy.matrix('48;8')
C=numpy.linalg.inv(A)*B
# Calculations
# (B)
# Solving eq'ns 3 & 4 simultaneously using matrix for v'_b & v'_c
P=numpy.matrix('1 2;-1 1')
Q=numpy.matrix('12;6')
R=numpy.linalg.inv(P)*Q
# Results (A&B)
print('The velocity of ball A after impact on ball B is %f m/s'%C[0]) # here the ball of mass 2 kg is bought to rest
print('The velocity of ball B after getting impacted by ball A is %f m/s'%C[1])
print('The final velocity of ball B is %f m/s'%R[0]) # here the ball of mass 6 kg is bought to rest
print('The velocity of ball C after getting impacted by ball B is %f m/s'%R[1])

import math
# Initilization of variables
h_1=9 # m # height of first bounce
h_2=6 # m # height of second bounce
# Calculations
# From eq'n (5) we have, Coefficient of restitution between the glass and the floor is,
e=math.sqrt(h_2/h_1)
# From eq'n 3 we get height of drop as,
h=h_1/e**2 # m
# Results
print('The ball was dropped from a height of %f m'%h)
print('The coefficient of restitution between the glass and the floor is %f '%e)

import math
# Initilization of variables
e=0.90 # coefficient o restitution
v_a=10 # m/s # velocity of ball A
v_b=15 # m/s # velocity of ball B
alpha_1=30 # degree # angle made by v_a with horizontal
alpha_2=60 # degree # angle made by v_b with horizontal
# Calculations
# The components of initial velocity of ball A:
v_a_x=v_a*math.cos(alpha_1*math.pi/180) # m/s
v_a_y=v_a*math.sin(alpha_1*math.pi/180) # m/s
# The components of initial velocity of ball B:
v_b_x=-v_b*math.cos(alpha_2*math.pi/180) # m/s
v_b_y=v_b*math.sin(alpha_2*math.pi/180) # m/s
# From eq'n 1 & 2 we get,
v_ay=v_a_y # m/s # Here, v_ay=(v'_a)_y
v_by=v_b_y # m/s # Here, v_by=(v'_b)_y
# On adding eq'n 3 & 4 we get,
v_bx=((v_a_x+v_b_x)+(-e*(v_b_x-v_a_x)))/2 # m/s # Here. v_bx=(v'_b)_x
# On substuting the value of v'_b_x in eq'n 3 we get,
v_ax=(v_a_x+v_b_x)-(v_bx) # m/s # here, v_ax=(v'_a)_x
# Now the eq'n for resultant velocities of balls A & B after impact are,
v_A=math.sqrt(v_ax**2+v_ay**2) # m/s
v_B=math.sqrt(v_bx**2+v_by**2) # m/s
# The direction of the ball after Impact is,
theta_1=math.degrees(math.atan(-(v_ay/v_ax))) # degree
theta_2=math.degrees(math.atan(v_by/v_bx)) # degree
# Results
print('The velocity of ball A after impact is %f m/s'%v_A)
print('The velocity of ball B after impact is %f m/s'%v_B)
print('The direction of ball A after impact is %f degree'%theta_1)
print('The direction of ball B after impact is %f degree'%theta_2)

# Initiization of variables
theta=30 # degrees # ange made by the ball against the wall
e=0.50
# Calculations
# The notations have been changed
# Resolving the velocity v as,
v_x=math.cos(theta*math.pi/180)
v_y=math.sin(theta*math.pi/180)
V_y=v_y
# from coefficient of restitution reation
V_x=-e*v_x
# Resultant velocity
V=math.sqrt(V_x**2+V_y**2)
theta=math.degrees(math.atan(V_y/(-V_x))) # taking +ve value for V_x
# NOTE: Here all the terms are multiplied with velocity i.e (v).
# Results
print('The velocity of the ball is %f v'%V)
print('The direction of the ball is %f degrees'%theta)

# Initilization of variables
e=0.8 # coefficient of restitution
g=9.81 # m/s**2 # acc due to gravity
# Calcuations
# Squaring eqn's 1 &2 and Solving eqn's 1 & 2 using matrix for the value of h
A=numpy.matrix([[-1,(2*g)],[-1,-(1.28*g)]])
B=numpy.matrix([[0.945**2],[(-0.4*9.81)]])
C=numpy.linalg.inv(A)*B # m
# Results
print('The height from which the ball A should be released is %f m'%C[1])
# The answer given in the book i.e 0.104 is wrong.

# Initilization of variables
theta_a=60 # degree # angle made by sphere A with the verticle
e=1 # coefficient of restitution for elastic impact
# Calculations
# theta_b is given by the eq'n cosd*theta_b=0.875, hence theta_b is,
theta_b=math.degrees(math.acos(0.875)) # degree
# Results
print('The angle through which the sphere B will swing after the impact is %f degree'%theta_b)

# Initilization of variables
m_a=0.01 # kg # mass of bullet A
v_a=100 # m/s # velocity of bullet A
m_b=1 # kg # mass of the bob
v_b=0 # m/s # velocity of the bob
l=1 # m # length of the pendulum
v_r=-20 # m/s # velocity at which the bullet rebounds the surface of the bob # here the notation for v'_a is shown by v_r
v_e=20 # m/s # velocity at which the bullet escapes through the surface of the bob # here the notation for v_a is shown by v_e
g=9.81 # m/s**2 # acc due to gravity
# Calculations
# Momentum of the bullet & the bob before impact is,
M=(m_a*v_a)+(m_b*v_b) # kg.m/s......(eq'n 1)
# The common velocity v_c ( we use v_c insted of v' for notation of common velocity) is given by equating eq'n 1 & eq'n 2 as,
# (a) When the bullet gets embedded into the bob
v_c=M/(m_a+m_b) # m/s
# The height h to which the bob rises is given by eq'n 3 as,
h_1=(1/2)*(v_c**2/g) # m
# The angle (theta_1) by which the bob swings corresponding to the value of height h_1 is,
theta_1=math.degrees(math.acos((l-h_1)/l)) # degree
# (b) When the bullet rebounds from the surface of the bob
# The velocity of the bob after the rebound of the bullet from its surface is given by equating eq'n 1 & eq'n 4 as,
v_bob_rebound=M-(m_a*v_r) # m/s # here v_bob_rebound=v'_b
# The equation for the height which the bob attains after impact is,
h_2=(v_bob_rebound**2)/(2*g) # m
# The corresponding angle of swing 
theta_2=math.degrees(math.acos((l-h_2)/l)) # degree
# (c) When the bullet pierces and escapes through the bob
# From eq'n 1 & 5 the velocity attained by the bob after impact is given as,
v_b_escape=M-(m_a*v_e) # m/s # here we use, v_b_escape insted of v'_b
# The equation for the height which the bob attains after impact is,
h_3=(v_b_escape**2)/(2*g) # m
# The corresponding angle of swing 
theta_3=math.degrees(math.acos((l-h_3)/(l))) # degree
# Results
print('(a) The maximum angle through which the pendulum swings when the bullet gets embeded into the bob is %f degree'%theta_1)
print('(b) The maximum angle through which the pendulum swings when the bullet rebounds from the surface of the bob is %f degree'%theta_2)
print('(c) The maximum angle through which the pendulum swings when the bullet escapes from other end of the bob the bob is %f degree'%theta_3)

# Initilization of variables
W_a=50 # N # falling weight
W_b=50 # N # weight on which W_a falls
g=9.81 # m/s**2 # acc due to gravity
m_a=W_a/g # kg # mass of W_a
m_b=W_b/g # kg # mass of W_b
k=2*10**3 # N/m # stiffness of spring
h=0.075 # m # height through which W_a falls
# The velocity of weight W_a just before the impact and after falling from a height of h is given from the eq'n, ( Principle of conservation of energy)
v_a=math.sqrt(2*g*h) # m/s
# Let the mutual velocity after the impact be v_m (i.e v_m=v'), (by principle of conservation of momentum)
v_m=(m_a*v_a)/(m_a+m_b) # m/s
# Initial compression of the spring due to weight W_b is given by,
delta_st=(W_b/k)*(10**2) # cm
# Let the total compression of the spring be delta_t, Then delta_t is found by finding the roots from the eq'n:
#delta_t**2-0.1*delta_t-0.000003=0. In this eq'n let,
a=1
b=-0.1
c=-0.000003
delta_t=((-b+(math.sqrt(b**2-(4*a*c))))/2*a)*(10**2) # cm # we consider the -ve value
delta=delta_t-delta_st # cm
# Results
print('The compression of the spring over and above caused by the static action of weight W_a is %f cm \n'%delta)

# Initilization of variables
v_a=600 # m/s # velocity of the bullet before impact
v_b=0 # m/s # velocity of the block before impact
w_b=0.25 # N # weight of the bullet
w_wb=50 # N # weight of wodden block
mu=0.5 # coefficient of friction between the floor and the block
g=9.81 # m/s**2 # acc due to gravity
# Calculations
m_a=w_b/g # kg # mass of the bullet
m_b=w_wb/g # kg # mass of the block
# Let the common velocity be v_c which is given by eq'n (Principle of conservation of momentum)
v_c=(w_b*v_a)/(w_wb+w_b) # m/s
# Let the distance through which the block is displaced be s, Then s is given by eq'n
s=v_c**2/(2*g*mu) # m
# Results
print('The distance through which the block is displaced from its initial position is %f m'%s)

# Initilization of variables
M=750 # kg # mass of hammer
m=200 # kg # mass of the pile
h=1.2 # m # height of fall of the hammer
delta=0.1 # m # distance upto which the pile is driven into the ground
g=9.81 # m/s**2 # acc due to gravity
# Caculations
# The resistance to penetration to the pile is given by eq'n,
R=(((M+m)*g)+((M**2*g*h)/((M+m)*delta)))*(10**-3) # kN 
# Results
print('The resistance to penetration to the pile is %f kN'%R)

