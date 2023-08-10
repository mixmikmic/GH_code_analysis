import math
# Initilization of variables
v_t=10 # m/s # velocity of the train
v_s=5 # m/s # velocity of the stone
# Calculations
# Let v_r be the relative velocity, which is given as, (from triangle law)
v_r=math.sqrt(v_t**2+v_s**2) # m/s
# The direction ofthe stone is,
theta=math.degrees(math.atan(v_s/v_t)) # degree
# Results
print('The velocity at which the stone appears to hit the person travelling in the train is %f m/s'%v_r)
print('The direction of the stone is %f degree'%theta)

# Initilization of variables
v_A=5 # m/s # speed of ship A
v_B=2.5 # m/s # speed of ship B
theta=135 # degree # angle between the two ships
# Calculations
# Here,
OA=v_A # m/s
OB=v_B # m/s
# The magnitude of relative velocity is given by cosine law as,
AB=math.sqrt((OA**2)+(OB**2)-(2*OA*OB*math.cos(theta*math.pi/180))) # m/s
# where AB gives the relative velocity of ship B with respect to ship A
# Applying sine law to find the direction, Let alpha be the direction of the reative velocity, then
alpha=math.degrees(math.asin((OB*math.sin(theta*math.pi/180))/(AB))) # degree
# Results
print('The magnitude of relative velocity of ship B with respect to ship A is %f m/s'%AB)
print('The direction of the relative velocity is %f degree'%alpha)

import numpy
# Initilization of variables
v_c=20 # km/hr # speed at which the cyclist is riding to west
theta_1=45 # degree # angle made by rain with the cyclist when he rides at 20 km/hr
V_c=12 # km/hr # changed speed
theta_2=30 # degree # changed angle when the cyclist rides at 12 km/hr
# Calculations
# Solving eq'ns 1 & 2 simultaneously to get the values of components(v_R_x & v_R_y) of absolute velocity v_R. We use matrix to solve eqn's 1 & 2.
A=numpy.matrix('1 1;1 0.577')
B=numpy.matrix('20;12')
C=numpy.linalg.inv(A)*B # km/hr
# The X component of relative velocity (v_R_x) is C(1)
# The Y component of relative velocity (v_R_y) is C(2)
# Calculations
# Relative velocity (v_R) is given as,
v_R=math.sqrt((C[0])**2+(C[1])**2) # km/hr
# And the direction of absolute velocity of rain is theta, is given as
theta=math.degrees(math.atan(C[1]/C[0])) # degree
# Results 
print('The magnitude of absolute velocity is %f km/hr'%v_R)
print('The direction of absolute velocity is %f degree'%theta)

# Initiization of variables
a=1 # m/s**2 # acceleration of car A
u_B=36*(1000/3600) # m/s # velocity of car B
u=0 # m/s # initial velocity of car A
d=32.5 # m # position of car A from north of crossing
t=5 # seconds
# Calculations
# CAR A: Absolute motion using eq'n v=u+at we have,
v=u+(a*t) # m/s
# Now distance travelled by car A after 5 seconds is given by, s_A=u*t+(1/2)*a*t**2
s_A=(u*t)+((1/2)*a*t**2)
# Now, let the position of car A after 5 seconds be y_A
y_A=d-s_A # m # 
# CAR B:
# let a_B be the acceleration of car B
a_B=0 # m/s
# Now position of car B is s_B
s_B=(u_B*t)+((1/2)*a_B*t**2) # m
x_B=s_B # m
# Let the Relative position of car A with respect to car B be BA & its direction be theta, then from fig. 19.9(b)
OA=y_A
OB=x_B
BA=math.sqrt(OA**2+OB**2) # m
theta=math.degrees(math.atan(OA/OB)) # degree
# Let the relative velocity of car A w.r.t. the car B be v_AB & the angle be phi. Then from fig 19.9(c). Consider small alphabets
oa=v
ob=u_B
v_AB=math.sqrt(oa**2+ob**2) # m/s
phi=math.degrees(math.atan(oa/ob)) # degree
# Let the relative acceleration of car A w.r.t. car B be a_A/B.Then,
a_AB=a-a_B # m/s^2
# Results
print('The relative position of car A relative to car B is %f m'%BA)
print('The direction of car A w.r.t car B is %f degree'%theta)
print('The velocity of car A relative to car B is %f m/s'%v_AB)
print('The direction of car A w.r.t (for relative velocity)is %f degree'%phi)
print('The acceleration of car A relative to car B is %d m/s**2'%a_AB)

