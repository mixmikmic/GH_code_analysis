# Initilization of variables
W=1000 # N # weight to be raised
# Calculations
# From the Principle of virtual work,
P=W/2 # N
# Results
print('The value of force (i.e P) that can hold the system in equilibrium is %d N'%P)

import math
# Initilization of variables
P=1000 # N # Force acting at the hinge of the 1st square
Q=1000 # N # Force acting at the hinge of the 2nd square
# Calculations
# Chosing the co-ordinate system with originat A, we can write,
theta=45 # degree
# Forces that do work are P,Q & X_B. Applying the principle of virtual work & Simplyfying and solving for X_B,
X_B=((2*P)/6)*(math.cos(theta*math.pi/180)/math.sin(theta*math.pi/180)) # N 
# Now give a virtual angular displacement to the whole frame about end A such that line AB turns by an angle delta_phi.
# The force doing work are P,Q&Y_B.Applying the principle of virtual work & Simplyfying this eq'n and solving for Y_B,
Y_B=((3*Q)+P)/6 # N
# Simply by removing the support at A & replacing it by the reactions X_A & Y_A we can obtain,
X_A=X_B # N
Y_A=P+Q-Y_B # N
# Results
print('The Horizontal component of reaction at A (X_A) is %f N'%X_A)
print('The Vertical component of reaction at A (Y_A) is %f N'%Y_A)
print('The Horizontal component of reaction at B (X_B) is %f N'%X_B)
print('The Vertical component of reaction at B (Y_B) is %f N'%Y_B)

