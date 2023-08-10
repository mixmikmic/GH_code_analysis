# Initilization of variables
k=1000 # N/m # stiffness of spring
x_1=0.1 # m # distance upto which the spring is stretched
x_2=0.2 # m 
x_0=0 # initial position of spring
# Calculations
# Work required to stretch the spring by 10 cm from undeformed position is given as,
U_10=-(k/2)*(x_1**2-x_0**2) # N-m 
# Work required to stretch from 10 cm to 20 cm is,
U=-(1/2)*k*(x_2**2-x_1**2) # N-m
# Results
print('The work of the spring force is %f N-m'%U_10)
print('The work required to stretch the spring by 20 cm is %f N-m'%U)

import math
# Initilization of variables
M_A=100 # kg # mass of block A
M_B=150 # kg # mass of block B
mu=0.2 # coefficient of friction between the blocks and the surface
x=1 # m # distance by which block A moves
g=9.81 # m/s^2 # acc due to gravity
# Calculations
# Consider the respective F.B.D
# Applying the principle of work and energy to the system of blocks A&B and on simplifying we get the value of v as,
v=math.sqrt(((-mu*M_A*g)+(M_B*g))/(125)) # m/s 
# Results
print('The velocity of block A is %f m/s'%v)

# Initilization of variables
M=500*10**3 # kg # mass of the train
u=0 # m/s # initial speed
v=90*(1000/3600) # m/s # final speed
t=50 # seconds
F_r=15*10**3 # N # Frictioal resistance to motion
# Calculations
# Acceleration is given as,
a=v/t # m/s**2
# The total force required to accelerate the train is,
F=M*a # N
# The maximum power required is at, t=50s & v=25 m/s
P=(F+F_r)*v*(10**-6) # MW
# At any time after 50 seconds, the force required only to overcome the frictional resistance of 15*10**3 N is,
P_req=F_r*v*(10**-3) # kW
# Results
print('(a) The maximum power required is %f MW'%P)
print('(b) The power required to maintain a speed of 90 km/hr is %f kW'%P_req)

# Initilization of variables
W=50 # N # Weight suspended on spring
k=10 # N/cm # stiffness of the spring
x_2=15 # cm # measured extensions
h=10 # cm # height for position 2
# Calculations
# Consider the required F.B.D.
# POSITION 1: The force exerted by the spring is,
F_1=W # N
# Extension of spring from undeformed position is x_1,
x_1=F_1/k # cm
# POSITION 2: When pulled by 10 cm to the floor. PE of weight is,
PE_g=-W*h # N-cm # (PE_g= PE_gravity)
# PE of the spring with respect to position 1
PE_s=(1/2)*k*(x_2**2-x_1**2) # N-cm  # (PE_s= PE_ spring)
# Total PE of the system with respect to position 1
PE_t=PE_g+PE_s # N-cm # (PE_t= PE_total)
# Total energy of the system,
E_2=PE_t # N-cm
# Total energy of the system in position 3 w.r.t position 1 is:
x=-math.sqrt(100) # cm
x=+math.sqrt(100) # cm
# Results
print('The potential energy of the system is %f N-cm'%E_2)
print('The maximum height above the floor that the weight W will attain after release is %f cm'%x)

# Initilization of variables
m=5 # kg # mass of the ball
k=500 # N/m # stiffness of the spring
h=10 # cm # height of drop
g=9.81 # m/s**2 # acc due to gravity
# Calculations
# Consider the respective F.B.D.
# In eq'n 1 substitute the respective values and simplify it further. In this eq'n of 2nd degree a=1 b=-0.1962 & c=-0.01962. Thus the roots of the eq'n is given as,
a=1 
b=-0.1962
c=-0.01962
delta=((-b+(math.sqrt((b**2)-(4*a*c))))/(2*a))*(10**2) # cm # We consider the +ve value of delta
# Results
print('The maximum deflection of the spring is %f cm'%delta)

# Initilization of variables
m=5 # kg # mass of the ball
k=500 # N/m # stiffness of the spring
h=0.1 # m # height of vertical fall
g=9.81 # m/s**2 # acc due to gravity
# Calculations
# Consider the respective F.B.D
# On equating the total energies at position 1 & 2 we get eq'n of delta as,
delta=math.sqrt((2*m*g*h)/(k)) # m 
# Results
print('The maximum compression of the spring is %f m'%delta)

# Initilization of variables
m=5 # kg # mass of the collar
k=500 # N/m # stiffness of the spring
AB=0.15 # m # Refer the F.B.D for AB
AC=0.2 # m # Refer the F.B.D for AC
g=9.81 # m/s**2 # acc due to gravity
# Calculations
# Consider the respective F.B.D
# POSITION 1: 
PE_1=m*g*(AB)+0 
KE_1=0
E_1=PE_1+KE_1 #
# POSITION 2 : Length of the spring in position 2
CB=math.sqrt(AB**2+AC**2) # m 
# x is the extension in the spring
x=CB-AC # m
# On substuting and Equating equations of total energies for position1 & position 2 we get the value of v as,
v=math.sqrt(((E_1-((1/2)*k*x**2))*2)/m) # m/s
# Results
print('The velocity of the collar will be %f m/s'%v)
# The answer given in the text book (v=16.4 m/s) is wrong

# Initilization of variables
m=5 # kg # mass of the block
theta=30 # degree # inclination of the plane
x=0.5 # m # distance travelled by the block
k=1500 # N/m # stiffness of the spring
mu=0.2 # coefficient of friction between the block and the surface
g=9.81 # m/s**2 # acc due to gravity
# Calculations
# Consider the F.B.D of the block
# Applying the principle of work and energy between the positions 1 & 2 and on further simplification we get the generic eq'n for delta as, 750*delta^2-16.03*delta-8.015=0. From this eq'n e have values of a.b & c as,
a=750
b=-16.03
c=-8.015
# Thus the roots of the eq'n are given as,
delta=(-b+(math.sqrt(b**2-(4*a*c))))/(2*a) # m
# Results
print('The maximum compression of the spring is %f m'%delta)

# Initilization of variables
M=10 # kg # Here M=M_1=M_2
g=9.81 # m/s^2 # acc due to gravity
# Calculations
# Consider the respective F.B.D
# Applying the principle of conservation of energy and by equating the total energies at position 1 & position 2 we get v as,
v=math.sqrt((M*g*4)/(25)) # m/s
# Results
print('The velocity of mass M_2 is %f m/s'%v)

