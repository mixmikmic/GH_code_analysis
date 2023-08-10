import math
c1=10**-6;
f1=60;
L1=1/(4*math.pi*math.pi*(f1**2)*c1);
print ("inductance of the circuit 1 = %.2f H" % L1)
f2=50;
w=2*math.pi*f2;
R1=100;
Z1=complex(R1,((w*L1)-(1/w*c1)));
#Z2=complex(100+j*((2*math.pi*50*L2)-(1/(2*math.pi*50*1.5*10**-6)))));
#for equal currents in two circuits Z1=Z2
print ('inductance of circuit 2 L2=9.82 H')
L2=9.82;
C2=1.5*10**-6;
Rf2=(1/(2*math.pi))*(1/(L2*C2))**0.5;
print ("Resonant frequency of the circuit 1 = %.2f Hz" % Rf2)

