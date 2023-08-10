import math
theta=math.degrees(math.acos(1/math.sqrt(2)))
theta=180-theta
print("Polarization angle theta=",theta)

import math
theta_p= math.degrees(math.atan(1.5))
print("Theta_p in degrees=%.5f"%theta_p)
sin_theta_r= (math.sin(theta_p*math.pi/180))/1.5
theta_r=math.degrees(math.asin(sin_theta_r))
print("Angle of refraction fron Snells law in degrees=%.5f"%theta_r)

lamda=5890 #A
n_e=1.553
n_o=1.544
s=(n_e)-(n_o)
x=(lamda)/(4*s)

print("The Value of x in m=",x)
#The answer provided in the textbook is wrong

