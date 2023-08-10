import math
theta_1=30
n_qa=1.4702
theta2=math.degrees(math.asin(math.sin(theta_1*math.pi/180)/n_qa))
print("For 4000 A beam, theta_2 in degree= %.5f"%theta2)

theta_1=30
n_qa=1.4624
theta2=math.degrees(math.asin(math.sin(theta_1*math.pi/180)/n_qa))
print("For 5000 A beam, theta_2 in degree= %.5f"%theta2)

import math
n=1/math.sin(45*math.pi/180)
print("Index reflection= %.5f"%n)

import math
n2=1.33
n1=1.50
theta_c=math.degrees(math.asin(n2/n1))
print("Angle theta_c in degree= %.5f"%theta_c)
print("Actual angle of indices = 45 is less than theta_ c, so there is no internal angle reflection")
print("Angle of refraction:")
x=n1/n2
theta_2=(math.asin(x*math.sin(45*math.pi/180))*180/math.pi)
print("Theta_2 in degree= %.5f"%theta_2)

