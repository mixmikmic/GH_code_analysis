get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Initilization of variables
L_AB=3 # m , length of the beam
L_AC=1 # m
L_BC=2 # m
M_C=12 # kNm , clockwise moment at C

# Calculations
R_B=M_C/L_AB # kN , moment at A
R_A=-M_C/L_AB # kN , moment at B

#S.F
F_A=R_A # kN 
F_B=R_A # kN
# B.M
M_A=0 # kNm
M_C1=R_A*L_AC  #kNm , M_C1 is the BM just before C
M_C2=(R_A*L_AC)+M_C #kNm , M_C2 is the BM just after C
M_B=0  #kNm

# Plotting SFD & BMD
x=([0],[0.99],[1],[3])
y=([-4],[-4],[-4],[-4])
a=([0],[0.99],[1],[3])
b=([0],[-4],[8],[0])
plt.subplot(2,1,1)
plt.xlabel("Span (m)")
plt.ylabel("Shear Force (kN)")
plt.plot(x,y)
plt.subplot(2,1,2)
plt.plot(a,b)
plt.xlabel("Span (m)")
plt.ylabel("Bending Moment (kNm)")

#Results
print('The graphs are the solutions')

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

# Initilization of variables
L_AD=8 # m , length of the beam
L_AB=2  # m 
L_BC=4 # m
L_CD=2 # m
UDL=1 # kN/m
P=2 # kN , point load at A
# Calculations

# Solving eqn's 1&2 using matrix to get R_B & R_C as,
A=np.matrix([[1,1],[1,3]])
B=np.matrix([[8],[30]])
C=np.linalg.inv(A)*B

# SHEAR FORCE
# the term F with suffixes 1 & 2 indicates SF just to left and right 
F_A=-P # kN
F_B1=-P # kN
F_B2=-P+C[0] # kN
F_C1=-P+C[0]-(UDL*L_BC)  #kN
F_C2=-P+C[0]-(UDL*L_BC)+C[1] # kN
F_D=0

# BENDING MOMENT
# the term F with suffixes 1 & 2 indicates BM just to left and right
M_A=0  #kNm
M_B=(-P*L_CD)  #kNm
M_C=(-P*(L_AB+L_BC))+(C[0]*L_BC)-(UDL*L_BC*(L_BC/2))  #kNm
M_D=0  #kNm

# LOCATION OF MAXIMUM BM
#Max BM occurs at E at a distance of 2.5 m from B i.e x=L_AE=4.5 m from free end A. Thus max BM is given by taking moment at B
L_AE=4.5 # m , given
M_E=(-2*L_AE)+(4.5*(L_AE-2))-((1/2)*(L_AE-2)**2) #kNm

# PLOTTING SFD & BMD
x=([0],[1.99],[2],[4.5],[5.99],[6],[8])
y=([-2],[-2],[2.5],[0],[-1.5],[2],[0])
a=([0],[2],[4.5],[6],[8])
b=([0],[-4],[-0.875],[-2],[0])
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(311)
plt.subplot(311)
plt.xlabel("Span (m)")
plt.ylabel("Shear Force (kN)")
ax.plot(x,y)
plt.subplot(313)
plt.plot(a,b)
plt.xlabel("Span (m)")
plt.ylabel("Bending Moment (kNm)")

#Results
print('The graphs are the solutions')

