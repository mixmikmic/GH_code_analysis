import math
# Initilization of variables
v_o=500 # m/s # velocity of the projectile
alpha=30 # angle at which the projectile is fired
t=30 # seconds
g=9.81 # m/s**2 # acc due to gravity
# Calculations
v_x=v_o*(math.cos(alpha*math.pi/180)) # m/s # Initial velocity in the horizontal direction
v_y=v_o*(math.sin(alpha*math.pi/180)) # m/s # Initial velocity in the vertical direction
# MOTION IN HORIZONTA DIRECTION:
V_x=v_x # m/s # V_x=Horizontal velocity after 30 seconds
# MOTION IN VERTICAL DIRECTION: # using the eq'n v=u+a*t
V_y=v_y-(g*t) # m/s # -ve sign denotes downward motion
# Let the Resultant velocity be v_R. It is given as,
v_R=math.sqrt((V_x)**2+(-V_y)**2) # m/s
theta=math.degrees(math.atan((-V_y)/V_x)) # degree # direction of the projectile
# Results
print('The velocity of the projectile is %f m/s'%v_R) 
# The answer of velocity is wrong in the text book.
print('The direction of the projectile is %f degree'%theta)

# Initilization of variables
v_A=10 # m/s # velocity of body A
alpha_A=60 # degree # direction of body A
alpha_B=45 # degree # direction of body B
# Calculations
# (a) The velocity (v_B) for the same range is given by eq'n;
v_B=math.sqrt((v_A**2*math.sin(2*alpha_A*math.pi/180))/(math.sin(2*alpha_B*math.pi/180))) # m/s
# (b) Now velocity v_B for the same maximum height is given as,
v_b=math.sqrt((v_A**2)*((math.sin(alpha_A*math.pi/180))**2/(math.sin(alpha_B*math.pi/180))**2)) # m/s
# (c) Now the velocity (v) for the equal time of flight is;
v=(v_A*math.sin(alpha_A*math.pi/180))/(math.sin(alpha_B*math.pi/180)) # m/s
# Results
print('(a) The velocity of body B for horizontal range is %f m/s'%v_B)
print('(b) The velocity of body B for the maximum height is %f m/s'%v_b)
print('(c) The velocity of body B for equal time of flight is %f m/s'%v)

# Initilization of variables
y=3.6 # m # height of the wall
x_1=4.8 # m # position of the boy w.r.t the wall
x_2=3.6 # m # distance from the wall where the ball hits the ground
g=9.81 # m/s**2 # acc due to gravity
# Calculations
# The range of the projectile is r, given as,
r=x_1+x_2 # m
# Let the angle of the projection be alpha, which is derived and given as,
alpha=math.degrees(math.atan((y)/(x_1-(x_1**2/r)))) # degree
# Now substuting the value of alpha in eq'n 3 we get the least velocity (v_o) as;
v_o=math.sqrt((g*r)/(math.sin(2*alpha*math.pi/180))) # m/s
# Results
print('The least velocity with which the ball can be thrown is %f m/s'%v_o)
print('The angle of projection for the same is %f degree'%alpha)

# Initilization of variables
v_o=400 # m/s # initial velocity of each gun
r=5000 # m # range of each of the guns
g=9.81 # m/s**2 # acc due to gravity
pi=180 # degree 
# Calculations
# now from eq'n 1
theta_1=math.degrees(math.asin((r*g)/(v_o**2)))/(2) 
# from eq'n 3
theta_2=(pi-(2*theta_1))/2 # degree 
# For 1st & 2nd gun, s is
s=r # m
# For 1st gun 
v_x=v_o*math.cos(theta_1*math.pi/180) # m/s
# Now the time of flight for 1st gun is t_1, which is given by relation,
t_1=s/(v_x) # seconds
# For 2nd gun
V_x=v_o*math.cos(theta_2*math.pi/180)
# Now the time of flight for 2nd gun is t_2
t_2=s/(V_x) # seconds
# Let the time difference between the two hits be delta.T. Then,
deltaT=t_2-t_1 # seconds
# Results
print('The time difference between the two hits is %f seconds'%deltaT)

# Initilization of variables
h=2000 # m/ height of the plane
v=540*(1000/3600) # m/s # velocity of the plane
g=9.81 # m/s**2 # acc due to gravity
# Calculations
# Time t required to travel down a height 2000 m is given by eq'n,
u=0 # m/s # initial velocity
t=math.sqrt((2*h)/(g)) # seconds
# Now let s be the horizonta distance travelled by the bomb in time t seconds, then
s=v*t # m
# angle is given as theta,
theta=math.degrees(math.atan(h/s)) # degree
# Results
print('The pilot should release the bomb from a distance of %f m'%s)
print('The angle at which the target would appear is %f degree'%theta)

# Initilization of variables
theta=30 # degree # angle at which the bullet is fired
s=-50 # position of target below hill
v=100 # m/s # velocity at which the bullet if fired
g=9.81 # m/s**2 
# Calculations
v_x=v*math.cos(theta*math.pi/180) # m/s # Initial velocity in horizontal direction
v_y=v*math.sin(theta*math.pi/180) # m/s # Initial velocity in vertical direction
# (a) Max height attained by the bullet
h=v_y**2/(2*g) # m
# (b)Let the vertical Velocity with which the bullet will hit the target be V_y. Then,
V_y=math.sqrt((2*-9.81*s)+(v_y)**2) # m/s # the value of V_y is +ve & -ve
# Let V be the velocity with wich it hits the target
V=math.sqrt((v_x)**2+(V_y)**2) # m/s
# (c) The time required to hit the target
a=g # m/s**2
t=(v_y-(-V_y))/a # seconds
# Results
print('(a) The maximum height to which the bullet will rise above the soldier is %f m'%h)
print('(b) The velocity with which the bullet will hit the target is %f m/s'%V)
print('(c) The time required to hit the target is %f seconds'%t)

# Initilization of variables
W=30 # N # Weight of the hammer
theta=30 # degree # ref fig.20.12
mu=0.18 # coefficient of friction
s=10 # m # distance travelled by the hammer # fig 20.12
g=9.81 # m/s**2 # acc due to gravity
# Calculations
# The acceleration of the hammer is given as,
a=g*((math.sin(theta*math.pi/180))-(mu*math.cos(theta*math.pi/180))) # m/s**2
# The velocity of the hammer at point B is,
v=math.sqrt(2*a*s) # m/s
# Let the initial velocity of the hammer in horizontal direction be v_x & v_y in vertical direction, Then,
v_x=v*math.cos(theta*math.pi/180) # m/s
v_y=v*math.sin(theta*math.pi/180) # m/s
# MOTION IN VERTICAL DIRECTION
# Now, let time required to travel vertical distance (i.e BB'=S=5 m) is given by finding the roots of the second degree eq'n as,
# From the eq'n 4.9*t**2+4.1*t-5=0,
a=4.9
b=4.1
c=-5
# The roots of the eq'n are,
t=((-b)+(math.sqrt(b**2-(4*a*c))))/(2*a)
# MOTION IN HORIZONTAL DIRECTION
# Let the horizotal distance travelled by the hammer in time t be s_x.Then,
s_x=v_x*math.cos(theta*math.pi/180)*t # m
x=1+s_x # m
# Results
print('The distance x where the hammer hits the round is %f m'%x)

# Initilization of variables
s=1000 # m # distance OB (ref fig.20.13)
h=19.6 # m # height of shell from ground
g=9.81 # m/s**2 # acc due to gravity
# Calculations
# MOTION OF ENTIRE SHELL FROM O to A.
v_y=math.sqrt(2*(g)*h) # m/s # initial velocity of shell in vertical direction
t=v_y/g # seconds # time taken by the entire shell to reach point A
v_x=s/t # m/s # velocity of shell in vertical direction
# VELOCITIES OF THE TWO PARTS OF THE SHELL AFTER BURSTING AT A:
# Let v_x2 be the horizontal velocity of 1st & the 2nd part after bursting which is given as,
v_x2=v_x*2 # m/s
# Now distance BC travelled by part 2 is
BC=v_x2*t # m
# Distance from firing point OC
OC=s+BC # m
# Results
print('(a) The velocity of shell just before bursting is %f m/s'%v_x)
print('(b) The velocity of first part immediately after the shell burst is %f m/s'%v_x2)
print('(c) The velocity of second part immediately after the shell burst is %f m/s'%v_x2)
print('(b) The distance between the firing point & the point where the second part of the shell hit the ground is %f m'%OC)

# Initilization of variables
v_o=200 # m/s # initial velocity
theta=60 # degree # angle of the incline
y=5 # rise of incline
x=12 # length of incline
g=9.81 # m/s**2 # acc due to gravity
# Calculations
# The angle of the inclined plane with respect to horizontal
beta=math.degrees(math.atan(y/x)) # degree
# The angle of projection with respect to horizontal
alpha=90-theta # degree
# Range is given by eq'n (ref. fig.20.14)
AB=(2*v_o**2*(math.sin((alpha-beta)*math.pi/180))*math.cos(alpha*math.pi/180))/(g*(math.cos(beta*math.pi/180))**2) # m
# Range AC when the short is fired down the plane
AC=(2*v_o**2*(math.sin((alpha+beta)*math.pi/180))*math.cos(alpha*math.pi/180))/(g*(math.cos(beta*math.pi/180))**2) # m
BC=AB+AC # m
# Results
print('The range covered (i.e BC) is %f m'%BC)

