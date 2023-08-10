import math
# Initilization of variables
W1=2000 #N # load at joint D of the truss
W2=4000 #N # load at joint E of the truss
Lac=6 #m # length of the tie
Lab=3 #m
Lbc=3 #m
theta=60 #degree # interior angles of the truss
# Calculations
# Here A is simply supported & B is  roller support. Now the SUPPORT REACTIONS are given as,
Rc=((W1*(Lab/2))+(W2*(Lab+(Lbc/2))))/Lac #N # Taking moment at A
Ra=W1+W2-Rc #N # Take sum Fy=0
# ANALYSIS OF TRUSS BY METHOD OF JOINT
# ASSUMPTION- we consider the,(1) Forces moving towards each other as +ve i.e TENSILE (T) & (2) Forces moving away from each other as -ve i.e COMPRESSIVE (C)
# (1) JOINT A
Fad=Ra/(math.sin(theta*math.pi/180)) #N #(C) # Umath.sing eq'n 2
Fab=Fad*math.cos(theta*math.pi/180) #N # (T) # Umath.sing eq'n 1
# (2) JOINT C
Fce=Rc/(math.sin(theta*math.pi/180)) #N # (C) # using eq'n 4
Fcb=Fce*math.cos(theta*math.pi/180) #N # (T) # using eq'n 3
# (3) JOINT D
Fdb=((Fad*math.sin(theta*math.pi/180))-(W1))/math.sin(theta*math.pi/180) #N  # (T) # Using eq'n 6
Fde=(Fdb*math.cos(theta*math.pi/180))+(Fad*math.cos(theta*math.pi/180)) #N # (C) # Using eq'n 5
# (4) JOINT E
Feb=((Fce*math.cos(theta*math.pi/180))-(Fde))/math.cos(theta*math.pi/180) #N # (C) # Using eq'n 7
# Results
print('The Axial Force in member AD (Fad) is %f N '%Fad)
print('The Axial Force in member AB (Fab) is %f N '%Fab)
print('The Axial Force in member CE (Fce) is %f N '%Fce)
print('The Axial Force in member CB (Fcb) is %f N '%Fcb)
print('The Axial Force in member DB (Fdb) is %f N '%Fdb)
print('The Axial Force in member DE  (Fde)is %f N '%Fde)
print('The Axial Force in member EB (Feb) is %f N '%Feb)

# Initilization of variables
W1=2000 #N (or 2 kN)# load at joint D of the truss
W2=4000 #N (or 4 kN)# load at joint E of the truss
Lac=6 #m # length of the tie
Lab=3 #m
Lbc=3 #m
theta=60 #degree # interior angles of the truss
# Calculations
# Here A is simply supported & B is  roller support. Now the SUPPORT REACTIONS are given as,
Rc=((W1*(Lab/2))+(W2*(Lab+(Lbc/2))))/Lac #N # Taking moment at A
Ra=W1+W2-Rc #N # Take sum Fy=0
# Calculations
# Calculating the axial forces in the respective members by METHOD OF SECTION
# A section is drawn passing through member DE such that it cuts the respective member. Now consider the equilibrium of the left hand portion of the truss. The three unknown forces are Fde, Fdb, & Fab
# Take moment about B
Fde=((3*Ra)-(W1*Lab*math.sin(30*math.pi/180)))/(3*math.cos(30*math.pi/180)) #N # (T)
# Results
print('The axial force in the member DE (Fde)is %f N \n'%Fde)

# Initilization of variables
W=1 #kN # load on the truss at joint D
theta=45 #degree # angle made by the members AC & BD with the horizontal
Lab=1 #m 
Lcd=1 #m # here Lcd= the distance from B to the line of extension drawn from 1kN force on the horizontal
# Calculations 
# (1) JOINT E
# Here the joint E is in equilibrium under two forces Fec & Fed which are non-collinear. Hence they must be 0. i.e Fec=Fed=0 
Fec=0
Fed=0
# (2) JOINT D
Fdb=W/math.sin(theta*math.pi/180) # kN # (C)# sum Fy=0
Fdc=Fdb*math.cos(theta*math.pi/180) # kN # (T) # sum Fx=0
# (3) JOINT C
Fca=Fdc/math.sin(theta*math.pi/180) # kN # (T) # sum Fx=0
Fcb=-(Fca*math.sin(theta*math.pi/180)) # kN # (C) # sum Fy=0
# Results
print('The axial force in the member DC (Fdc) is %f kN'%Fdc)
print('The axial force in the member DB (Fdb) is %f kN'%Fdb)
print('The axial force in the member CA (Fca) is %f kN'%Fca)
print('The axial force in the member CB (Fcb) is %f kN'%Fcb)
print('The axial force in the member EC (Fec) is %f kN'%Fec)
print('The axial force in the member ED (Fed) is %f kN'%Fed)
# Here -ve sign indicates COMPRESSIVE force & +ve indicates TENSILE force

# Initilization of variables
W1=1000 #N # Load acting at the end pannels and the ridge
W2=2000 #N # Load acting at the intermidiate pannels
Laf=1 #m
Lgf=1 #m
Lag=2 #m
Lbg=1 #m
Lab=3 #m
theta=30 #degree # angle made by the principal rafter with the tie beam
beta=60 #degree # angle made by the slings (i.e members CF & CG) with the tie beam
# Calculations
# consider the equilibrium of the entire truss as a F.B.D
Xa=2*(W1*math.sin(theta*math.pi/180))+(W2*math.sin(theta*math.pi/180)) #N # sum Fx=0
Rb=((W2*Laf*math.cos(theta*math.pi/180))+(W1*Lag*math.cos(theta*math.pi/180)))/Lab # N # Moment at A=0
Ya=2*(W1*math.cos(theta*math.pi/180))+(W2*math.cos(theta*math.pi/180))-(Rb) #N # sum Fy=0
# Now pass a section through the truss such that it cuts the members CE,CG & FG. Now consider the equilibrium of the right hand side of the truss
# Take moment about C
Ffg=(Rb*(Lbg+0.5))/(0.5*math.tan(beta*math.pi/180)) # N # (T) # Here 0.5 is the half distance of Lgf
# Take moment about G
Fce=(-Rb*Lbg)/(Lbg*math.sin(theta*math.pi/180)) # N # (C)
# Take moment about B
Fcg=0/(Lbg*math.sin(beta*math.pi/180)) # N
# Results 
print('The axial force in the member FG (Ffg) is %f N'%Ffg)
print('The axial force in the member CE (Fce) is %f N'%Fce)
print('The axial force in the member CG (Fcg) is %f N'%Fcg)

# Initilization of variables
W1=100 #N # load acting at pt. C vertically
W2=50 #N # load acting at point B horizontaly
L=2 #m # length of each bar in the hexagonal truss
theta=60 #degree # internal angle of the truss
# Calculations
# We calculate the values of different members of the truss
HG=L*math.sin(theta*math.pi/180)
AF=L
# Support A is hinged whereas support F is a roller support. Firstly we find the support reactios as follows,
Rf=(W2*HG)/AF #N # moment at F
Xa=W2 #N # sum Fx=0
Ya=W1-Rf #N # sum Fy=0
# Now pass a section through the truss cutting the members CD,GD,GE & GF and consider equilibrium of right hand portion of the truss
Fcd=(Rf*(L/2))/(L*math.sin(theta*math.pi/180)) # N (C) # Taking moment about G
# Now pass a scetion pq cutting the members CB,GB & GA
Fga=((Rf*(L+(L/2)))-(W1*(L/2)))/(L*math.sin(theta*math.pi/180)) # N (T) # Taking moment about B
# take moment about G
Fcb=((W1*(L/2))+(Rf*(L/2)))/(L*math.sin(theta*math.pi/180)) # N (C)
Fgb=(Fcb*math.cos(theta*math.pi/180))-(Fga*math.cos(theta*math.pi/180)) # N (T) # sum Fx=0
# Results
print('The axial force in the member CD (Fcd) is %f N'%Fcd)
print('The axial force in the member GB (Fgb) is %f N'%Fgb)

# Initilization of variables
W=24 # kN # Load acting at pt C
Laf=12 # m # length of the tie beam
l=4 # m# length of each member in the tie
h=3 # m # height of the slings
Lae=8 # m
# Calculations
s=math.sqrt((l**2)+(h**2)) # m # sloping length 
# From triangle BCD,
theta=math.degrees(math.acos(h/s))
# SUPPORT REACTIONS
Rf=(W*l)/Laf # kN # take moment at A
Ra=W-Rf # kN # sum Fy=0
# now pass a sectio mn through the truss and consider te equilibrium of the left hand portion 
Fce=(Ra*l)/h # kN (T) # Take moment at B
Fbd=((W*l)-(Ra*Lae))/h # kN (C) # take moment at E
Fbe=(Ra-W)/math.cos(theta*math.pi/180) # kN
Fbd=(-Ra*l)/h # kN # take moment at C
Fce=((Ra*Lae)-(W*l))/h # kN (T) # take moment at D
Fcd=(W-Ra)/math.cos(theta*math.pi/180) # kN (T) # sum Fy=0
# Resuts
print('(1) The axial force in the bar CE (Fce) is %f kN'%Fce)
print('(2) The axial force in the bar BD (Fbd) is %f kN'%Fbd)
print('(3) The axial force in the bar BE (Fbe) is %f kN'%Fbe)
print('(4) The axial force in the bar CD (Fcd) is %f kN'%Fcd)

# Initilization of variables
W1=4 # kN # load acting at a distance of 5 m from C
W2=3 # kN # load acting at a distance of 7.5 m from C
L=30 #m # distance AB
L1=15 # dist AC
L2=15 #m #dist BC
l1=10 #m # distance between A and 4 kN load
l2=22.5 #m # distance between A and 3 kN load
# Calculations
# (1) Reactions
Yb=((W1*l1)+(W2*l2))/L # kN # Take moment at A
Ya=W1+W2-Yb # kN # sum Fy=0
# Xa=Xb........(eq'n 1)   # sum Fx=0
# (2) Dismember
# Member AC. Consider equilibrium of member AC
# Xa=Xc ... Consider thus as eq'n 2 # sum Fx=0
Yc=W1-Ya # kN # sum Fy=0
# Take moment about A
Xc=((W1*l1)-(Yc*L1))/L1 # kN 
# now from eq'n 1 & 2
Xa=Xc # kN
Xb=Xa # kN
# The components of reactions at A & B are,
Ra=math.sqrt(Xa**2+Ya**2) # kN
Rb=math.sqrt(Xb**2+Yb**2) # kN
# Results
print('The reaction at A ( Ra) is %f kN'%Ra)
print('The reaction at B ( Rb) is %f kN'%Rb)

# Initilization of variables
W1=2 # kN # load acting at a distance of 1m from point A
W2=1 # kN # load acting at a distance of 1m from point B
theta=30 # degree
L=4 # m # length of the tie beam
l=1 #m # length of each member in the tie
# Calculations
# (a) Reactions
Yb=((W1*l)+(W2*3*l))/L # kN # Taking moment about A
Ya=W1+W2-Yb # kN # sum Fy=0
# (b) Dismember
# MEMBER AB
# Xa=Xb........ (eq'n 1) # sum Fx=0
# MEMBER AC
# Xa=Xc.........(eq'n 2) # sum Fx=0
Yc=W1-Ya # kN # sum Fy=0
# Taking moment about A
Xc=((W1*l)-(Yc*2*l))/(2*math.tan(theta*math.pi/180)) # kN
# From eq'n 1 & 2
Xa=Xc # kN
Xb=Xa # kN
# Results
print('The force in tie bar AB is %f kN'%Xb)

# Initilization of variables
W=1000 # N 
r=0.25 # radius of pulley at E 
Lab=2 #m
Lad=1 # m
Lbd=1 # m 
Ldc=0.75 # m
l1=0.5 #m # c/c distance between bar AB and point E
l2=1.25 # m # dist between rigid support and the weight
# Calculations
# (a) Reactions
Xa=W # N # sum Fx=0
Yb=((W*l1)+(W*l2))/Lab # N # Take moment about A
Ya=W-Yb # N # sum Fy=0
# Dismember
# MEMBER ADB
# consider triangle BCD  to find theta, where s= length of bar BC, 
s=math.sqrt(Lbd**2+Ldc**2) # m
theta=math.degrees(math.acos(Lbd/s)) # degree
# equilibrium eq'n of member ADB
Yd=(Ya*Lab)/Lad #  take moment about B
Fbc=(Yb+Ya-Yd)/math.sin(theta*math.pi/180) # N # sum Fy=0
Xd=(Fbc*math.cos(theta*math.pi/180))+(Xa) # N # sum Fx=0
# PIN D
Rd=math.sqrt(Xd**2+Yd**2) # N # shear force on the pin
# Results
print('The compressive force in bar BC (Fbc) is %f N'%Fbc)
print('The shear force on the pin is %f N'%Rd)

import numpy
# Initiliztion of variables
P=5000 # N
theta=45 # degree # angle made by Rd & Re with the horizontal
Lab=3 # m
Lac=3 # m
Lbd=2 # m
Lce=2 # m
l=1.5 # m # dist of load P from B
# Calculations (BEAM AB )
# Consider the equilibrium of beams 
# We are using matrix to solve the simultaneous eqn's 
A=numpy.matrix([[(Lbd*math.sin(theta*math.pi/180)),Lab],[(Lce*math.sin(theta*math.pi/180)),-Lac]])
B=numpy.matrix([[(P*l)],[0]])
C=numpy.linalg.inv(A)*B
# Calculations (BEAM AC)
Re=C[0] # N (C) # from eq'n 1
Ya=(Re*Lce*math.sin(theta*math.pi/180))/Lac # N # from eq'n 7
Xa=C[0]*math.cos(theta*math.pi/180) # N # from eq'n 2
Ra=math.sqrt(Xa**2+Ya**2) # N (C)
Yb=P-Ya-(C[0]*math.sin(theta*math.pi/180)) # N (C) # eq'n 3
Yc=Ya-(Re*math.sin(theta*math.pi/180)) # N (T)
# Results 
print('(1) The value of axial force (Rd) in bar 2 is %f N'%C[0])
print('(2) The value of axial force (Re) in bar 3 is %f N'%Re)
print('(3) The value of axial force (Yb) in bar 1 is %f N'%Yb)
print('(4) The value of axial force (Yc) in bar 4 is %f N'%Yc)

