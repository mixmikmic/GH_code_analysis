import math
#Initialization of variables
N=1500 # r.p.m
r=0.5 # m , radius of the disc
m=300 # N , weight of the disc
t=120 #seconds , time in which the disc comes to rest
omega=0 
g=9.81 #m/s**2
#Calculations
omega_0=(2*math.pi*N)/60 #rad/s
#angular deceleration is given as,
alpha=-(omega_0/t) #radian/second**2
theta=(omega_0**2)/(2*(-alpha)) #radian
#Let n be the no of revolutions taken by the disc before it comes to rest, then
n=theta/(2*math.pi)
#Now,
I_G=((1/2)*m*r**2)/g
#The frictional torque is given as,
M=I_G*alpha #N-m
#Results
print('(a) The no of revolutions executed by the disc before coming to rest is %d'%n)
print('(b) The frictional torque is %f N-m'%M)

# Initilization of variables
s=1 # m
mu=0.192 # coefficient of static friction
g=9.81 # m/s**2
# Calculations
# The maximum angle of the inclined plane is given as,
theta=math.degrees(math.atan(3*mu)) # degree
a=(2/3)*g*math.sin(theta*180/math.pi) # m/s**2 # by solving eq'n 4
v=math.sqrt(2*a*s) # m/s
# Let the acceleration at the centre be A which is given as,
A=g*math.sin(theta*math.pi/180) # m/s**2 # from eq'n 1
# Results
print('(a) The acceleration at the centre is %f m/s**2'%A)
print('(b) The maximum angle of the inclined plane is %f degree'%theta)

# Initilization of variables
W_a=25 # N 
W_b=25 # N 
W=200 # N # weight of the pulley
i_g=0.2 # m # radius of gyration
g=9.81 # m/s^2
# Calculations
# Solving eqn's 1 & 2 for acceleration of weight A (assume a)
a=(0.15*W_a*g)/(((W*i_g**2)/(0.45))+(0.45*W_a)+((0.6*W_b)/(3))) # m/s^2
# Results
print('The acceleration of weight A is %f m/s**2'%a)

# Initilization of variables
r_1=0.075 # m
r_2=0.15 # m
P=50 # N
W=100 # N
i_g=0.05 # m
theta=30 # degree
g=9.81 # m/s^2
# Calculations
# The eq'n for acceleration of the pool is given by solving eqn's 1,2 &3 as,
a=(50*g*(r_2*math.cos(theta*math.pi/180)-r_1))/(100*((i_g**2/r_2)+r_2)) # m/s**2
# Results
print('The acceleration of the pool is %f m/s**2'%a)

# Initilization of variables
L=1 # m # length of rod AB
m=10 # kg # mass of the rod
g=9.81 
theta=30 # degree
# Calculations
# solving eq'n 4 for omega we get,
omega=math.sqrt(2*16.82*math.sin(theta*math.pi/180)) # rad/s
# Now solving eq'ns 1 &3 for alpha we get,
alpha=(12/7)*g*math.cos(theta*math.pi/180) # rad/s
# Components of reaction are given as,
R_t=((m*g*math.cos(theta*math.pi/180))-((m*alpha*L)/4)) # N
R_n=((m*omega**2*L)/(4))+(m*g*math.sin(theta*math.pi/180)) # N
R=math.sqrt(R_t**2+R_n**2) # N 
# Results
print('(a) The angular velocity of the rod is %f rad/sec'%omega)
print('(b) The reaction at the hinge is %f N'%R)

