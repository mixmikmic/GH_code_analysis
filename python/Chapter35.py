import math 
l=1.0 #length of solenoid in meter
r=3*10**-2 #radius of solenoid in meter
n=200*10**2 #number of turns in solenoid per meter
u0=4*math.pi*10**-7 #in weber/amp-m
i=1.5 #current in amp
N=100 #no.of turns in a close packed coil placed at the center of solenoid
d=2*10**-2 #diameter of coil in meter
delta_T=0.050 #in sec
#(A)
B=u0*i*n
print("Magnetic field at center in wb/m2 is %.7f"%B)
#(B)
A=math.pi*(d/2)**2
Q=B*A
print("Magnetic flux at the center of the solenoid in weber is %.7f"%Q)
delta_Q=Q-(-Q)
E=-(N*delta_Q/delta_T)
print("Induced EMF in volts is %.7f"%E)

B=2 #magnetic field in wb/m2
l=10*10**-2 #in m
v=1.0 #in m/sec
q=1.6*10**-19 #charge in coul
print("Let S be the frame of reference fixed w.r.t  the magnet and Z be the frame of reference w.r.t the loop")
#(A)
E=v*B
print("(A) Induced electric field in volt/m observed by Z",E)
#(B)
F=q*v*B
print("(B) Force acting on charge carrier in nt w.r.t S is %.1e"%F)
F1=q*E
print("Force acting on charge carrier in nt w.r.t Z is %.1e"%F1)
#(C)
emf1=B*l*v
print("(C) Induced emf in volt observed by S is",emf1)
emf2=E*l
print("Induced emf in volt observed by Z is",emf2)

