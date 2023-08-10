import math
# Initilization of variables
d1=24 # cm # diameter of larger pulley
d2=12 # cm # diameter of smaller pulley
d=30 #cm # seperation betweem 1st & the 2nd pulley
# Calcuations
r1=d1/2 #cm # radius of 1st pulley
r2=d2/2 #cm # radius of 2nd pulley
theta=math.degrees(math.asin((r1-r2)/d)) #degrees 
# Angle of lap
beta_1=180+(2*theta) #degree # for larger pulley
beta_2=180-(2*theta) #degree #for smaller pulley
L=math.pi*(r1+r2)+(2*d)+((r1-r2)**2/d) #cm # Length of the belt
# Results
print('The angle of lap for the larger pulley is %f degree'%beta_1)
print('The angle of lap for the smaller pulley is %f degree'%beta_2)
print('The length of pulley required is %f cm'%L)

from __future__ import division
import math
# Initilization of variables
d1=0.6 #m # diameter of larger pulley
d2=0.3 #m # diameter of smaller pulley
d=3.5 #m # separation between the pulleys
# Calculations
r1=d1/2 #m # radius of larger pulley
r2=d2/2 #m # radius of smaller pulley
theta=math.degrees(math.asin((r1+r2)/d)) #degree
# Angle of lap for both the pulleys is same, i.e
beta=180+(2*theta) # degree
L=((math.pi*(r1+r2))+(2*d)+((r1-r2)**2/d)) #cm # Length of the belt
# Results
print('The angle of lap for the pulley is %f degree'%beta)
print('The length of pulley required is %f m'%L)

from __future__ import division
import math
# Initilization of variables
W1=1000 #N
mu=0.25 #coefficient of friction
T1=W1 # Tension in the 1st belt carrying W1
e=2.718 #constant
# Calculations
T2=T1/(e**(mu*math.pi)) #N # Tension in the 2nd belt
W2=T2 #N
# Results
print('The minimum weight W2 to keep W1 in equilibrium is %f N'%W2)

from __future__ import division
import math
# Initilization of variables
mu=0.5 # coefficient of friction between the belt and the wheel
W=100 #N
theta=45 #degree
e=2.718
Lac=0.75 #m # ength of the lever
Lab=0.25 #m
Lbc=0.50 #m
r=0.25 #m
# Calculations
beta=((180+theta)*math.pi)/180 # radian # angle of lap
# from eq'n 2
T1=(W*Lbc)/Lab #N 
T2=T1/(e**(mu*beta)) #N # from eq'n 1
# consider the F.B.D of the pulley and take moment about its center, we get Braking Moment (M)
M=r*(T1-T2) #N-m
# Results
print('The braking moment (M) exerted by the vertical weight W is %f N-m'%M)

from __future__ import division
import math
# Initiization of variables
W= 1000 #N # or 1kN
mu=0.3 # coefficient of friction between the rope and the cylinder
e=2.718 # constant
alpha=90 # degree # since 2*alpha=180 egree
# Calculations
beta=2*math.pi*3 # radian # for 3 turn of the rope
# Here T1 is the larger tension in that section of the rope which is about to slip
T1=W #N
F=W/e**(mu*(1/(math.sin(alpha*math.pi/180)))*(beta)) #N  Here T2=F
# Results
print('The  force required to suport the weight of 1000 N i.e 1kN is %f N'%F)

from __future__ import division
import math
# Initilization of variables
Pw=50 #kW
T_max=1500 #N
v=10 # m/s # velocity of rope
w=4 # N/m # weight of rope
mu=0.2 # coefficient of friction 
g=9.81 # m/s**2 # acceleration due to gravity
e=2.718 # constant
alpha=30 # degree # since 2*alpha=60 
# Calcuations
T_e=(w*v**2)/g # N # where T_e is the centrifugal tension
T1=(T_max)-(T_e) #N
T2=T1/(e**(mu*(1/math.sin(alpha*math.pi/180))*(math.pi))) #N # From eq'n T1/T2=e^(mu*cosec(alpha)*beta)
P=(T1-T2)*v*(10**-3) #kW # power transmitted by a single rope
N=Pw/P # Number of ropes required
# Results
print('The number of ropes required to transmit 50 kW is %f or ~ %.0f'%(N,N))

from __future__ import division
import math
# Initilization of variables
d1=0.45 #m # diameter of larger pulley
d2=0.20 #m # diameter of smaller pulley
d=1.95 #m # separation between the pulley's
T_max=1000 #N # or 1kN which is the maximum permissible tension
mu=0.20 # coefficient of friction
N=100 # r.p.m # speed of larger pulley
e=2.718 # constant
T_e=0 #N # as the data for calculating T_e is not given we assume T_e=0
# Calculations
r1=d1/2 #m # radius of larger pulley
r2=d2/2 #m # radius of smaller pulley
theta=math.degrees(math.asin((r1+r2)/d)) # degree
# for cross drive the angle of lap for both the pulleys is same
beta=((180+(2*(theta)))*math.pi)/180 #radian
T1=T_max-T_e #N
T2=T1/(e**(mu*(beta))) #N # from formulae, T1/T2=e**(mu*beta)
v=((2*math.pi)*N*r1)/60 # m/s # where v=velocity of belt which is given as, v=wr=2*pie*N*r/60
P=(T1-T2)*v*(10**-3) #kW # Power
# Results
print('The power transmitted by the cross belt drive is %f kW'%P)
#answer given in the textbook is incorrect

from __future__ import division
import math
# Initilization of variabes
b=0.1 #m #width of the belt
t=0.008 #m #thickness of the belt
v=26.67 # m/s # belt speed
beta=165 # radian # angle of lap for the smaller belt
mu=0.3 # coefficient of friction
sigma_max=2 # MN/m**2 # maximum permissible stress in the belt
m=0.9 # kg/m # mass of the belt
g=9.81 # m/s**2
e=2.718 # constant
# Calculations
A=b*t # m**2 # cross-sectional area of the belt
T_e=m*v**2 # N # where T_e is the Centrifugal tension
T_max=(sigma_max)*(A)*(10**6) # N # maximum tension in the belt
T1=(T_max)-(T_e) # N 
T2=T1/(e**((mu*math.pi*beta)/180)) #N # from formulae T1/T2=e**(mu*beta)
P=(T1-T2)*v*(10**-3) #kW # Power transmitted
T_o=(T1+T2)/2 # N # Initial tension
# Now calculations to transmit maximum power
Te=T_max/3 # N # max tension
u=math.sqrt(T_max/(3*m)) # m/s # belt speed for max power
T_1=T_max-Te # N # T1 for case 2
T_2=T_1/(e**((mu*math.pi*beta)/180)) # N 
P_max=(T_1-T_2)*u*(10**-3) # kW # Max power transmitted
# Results
print('The initial power transmitted is %f kW'%P)
print('The initial tension in the belt is %f N'%T_o)
print('The maximum power that can be transmitted is %f kW'%P_max)
print('The maximum power is transmitted at a belt speed of %f m/s'%u)

from __future__ import division
import math
# Initilization of variables
p=0.0125 # m # pitch of screw
d=0.1 #m # diameter of the screw
r=0.05 #m # radius of the screw
l=0.5 #m # length of the lever
W=50 #kN # load on the lever
mu=0.20 # coefficient of friction 
# Calculations
theta=math.degrees(math.atan(p/(2*math.pi*r))) #degree # theta is the Helix angle
phi=math.degrees(math.atan(mu)) # degree # phi is the angle of friction
# Taking the leverage due to handle into account,force F1 required is,
a=theta+phi
b=theta-phi
F1=(W*(math.tan(a*math.pi/180)))*(r/l) #kN
# To lower the load
F2=(W*(math.tan(b*math.pi/180)))*(r/l) #kN # -ve sign of F2 indicates force required is in opposite sense
E=(math.tan(theta*math.pi/180)/math.tan((theta+phi)*math.pi/180))*100 # % # here E=eata=efficiency in %
# Results
print('The force required (i.e F1) to raise the weight is %f kN'%F1)
print('The force required (i.e F2) to lower the weight is %f kN'%F2)
print('The efficiency of the jack is %f percent'%E)

from __future__ import division
import math
# Initilization of variabes
P=20000 #N #Weight of the shaft
D=0.30 #m #diameter of the shaft
R=0.15 #m #radius of the shaft
mu=0.12 # coefficient of friction
# Calculations
# Friction torque T is given by formulae,
T=(2/3)*P*R*mu #N-m
M=T #N-m
# Results
print('The frictional torque is %f N-m'%M)

