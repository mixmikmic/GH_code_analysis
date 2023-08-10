from __future__ import division
import math
#Initilization of variables
m=0.1 # kg # mass of ball
# Calculations
# Consider the respective F.B.D.
# For component eq'n in x-direction
delta_t=0.015 # seconds # time for which the ball &the bat are in contact
v_x_1=-25 # m/s 
v_x_2=40*math.cos(40*math.pi/180) # m/s
F_x_average=((m*(v_x_2))-(m*(v_x_1)))/(delta_t) # N
# For component eq'n in y-direction
delta_t=0.015 # sceonds
v_y_1=0 # m/s
v_y_2=40*math.sin(40*math.pi/180) # m/s
F_y_average=((m*v_y_2)-(m*(v_y_1)))/(delta_t) # N
F_average=math.sqrt(F_x_average**2+F_y_average**2) # N
# Results
print('The average impules force exerted by the bat on the ball is %f N'%F_average)

from __future__ import division
# Initiliation of variables
m_g=3000 # kg # mass of the gun
m_s=50 # kg # mass of the shell
v_s=300 # m/s # initial velocity of shell
s=0.6 # m # distance at which the gun is brought to rest
v=0 # m/s # initial velocity of gun
# Calculations
# On equating eq'n 1 & eq'n 2 we get v_g as,
v_g=(m_s*v_s)/(-m_g) # m/s
# Using v^2-u^2=2*a*s to find acceleration,
a=(v**2-v_g**2)/(2*s) # m/s**2
# Force required to stop the gun,
F=m_g*(-a) # N # here we make a +ve to find the Force
# Time required to stop the gun, using v=u+a*t:
t=(-v_g)/(-a)  # seconds # we take -a to consider +ve value of acceleration
# Results
print('The recoil velocity of gun is %d m/s'%v_g)
print('The Force required to stop the gun is %f N'%F)
print('The time required to stop the gun is %f seconds'%t)

from __future__ import division
# Initilization of variables
m_m=50 # kg # mass of man
m_b=250 # kg # mass of boat
s=5 # m # length of the boat
v_r=1 # m/s # here v_r=v_(m/b)= relative velocity of man with respect to boat
# Calculations
# Velocity of man is given by, v_m=(-v_r)+v_b
# Final momentum of the man and the boat=m_m*v_m+m_b*v_b. From this eq'n v_b is given as
v_b=(m_m*v_r)/(m_m+m_b) # m/s # this is the absolute velocity of the boat
# Time taken by man to move to the other end of the boat is,
t=s/v_r # seconds
# The distance travelled by the boat in the same time is,
s_b=v_b*t # m to right from O
# Results
print('(a) The velocity of boat as observed from the ground is %f m/s'%v_b)
print('(b) The distance by which the boat gets shifted is %f m'%s_b)

from __future__ import division
# Initilization of variables
M=250 # kg # mass of the boat
M_1=50 # kg # mass of the man
M_2=75 # kg # mass of the man
v=4 # m/s # relative velocity of man w.r.t boat
# Calculations 
# (a)
# Let the increase in the velocity or the final velocity of the boat when TWO MEN DIVE SIMULTANEOUSLY is given by eq'n,
deltaV_1=((M_1+M_2)*v)/(M+(M_1+M_2)) # m/s
# (b) # The increase in the velocity or the final velocity of the boat when man of 75 kg dives 1st followed by man of 50 kg
# Man of 75 kg dives first, So let the final velocity is given as
deltaV_75=(M_2*v)/((M+M_1)+M_2) # m/s
# Now let the man of 50 kg jumps  next, Here
deltaV_50=(M_1*v)/(M+M_1) # m/s
# Let final velocity of boat is,
deltaV_2=0+deltaV_75+deltaV_50 # m/s
# (c) 
# The man of 50 kg jumps first,
delV_50=(M_1*v)/((M+M_2)+(M_1)) # m/s
# the man of 75 kg jumps next,
delV_75=(M_2*v)/(M+M_2) # m/s
# Final velocity of boat is,
deltaV_3=0+delV_50+delV_75 # m/s
# Results
print('(a) The Final velocity of boat when two men dive simultaneously is %f m/s'%deltaV_1)
print('(b) The Final velocity of boat when the man of 75 kg dives first and 50 kg dives second is %f m/s'%deltaV_2)
print('(c) The Final velocity of boat when the man of 50kg dives first followed by the man of 75 kg is %f m/s'%deltaV_3)

from __future__ import division
# Initilization of variables
m_m=70 # kg # mass of man
m_c=35 # kg # mass of canoe
m=25/1000 # kg # mass of bullet
m_wb=2.25 # kg # mass of wodden block
V_b=5 # m/s # velocity of block
# Calculations
# Considering Initial Momentum of bullet=Final momentum of bullet & the block we have,Velocity of  bullet (v) is given by eq'n,
v=(V_b*(m_wb+m))/(m) # m/s 
# Considering, Momentum of the bullet=Momentum of the canoe & the man,the velocity on canoe is given by eq'n
V=(m*v)/(m_m+m_c) # m/s
# Results
print('The velocity of the canoe is %f m/s'%V)

from __future__ import division
# Initilization of variables
m=2 # kg # mass of the particle
v_0=20 # m/s # speed of rotation of the mass attached to the string
r_0=1 # m # radius of the circle along which the particle is rotated
r_1=r_0/2 # m
# Calculations
# here, equating (H_0)_1=(H_0)_2 i.e  (m*v_0)*r_0=(m*v_1)*r_1 (here, r_1=r_0/2). On solving we get v_1 as,
v_1=2*v_0 # m/s
# Tension is given by eq'n,
T=(m*v_1**2)/r_1 # N
# Results
print('The new speed of the particle is %d m/s'%v_1)
print('The tension in the string is %d N'%T)

