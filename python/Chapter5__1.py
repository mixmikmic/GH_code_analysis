#determine the induced emf in the armature

#varaible declaration
P=4 #poles
A=2 #wave wound
n=50 #number of slots
Sc=24 #slots/conductor
N=600 #speed of armature 
F=10e-3 #webers

#calculations
Z=Sc*n #total conductor
E=F*Z*N*P/(60*A) #emf induced

print " emf induced E = " , E , "volts"

#determine the induced emf in the armature

#variable declaration
P=4 #poles
A=4 #wave wound
n=50 #number of slots
Sc=24 #slots/conductor
N=600 #rpm 
F=10e-3 #webers

#calculations
Z=Sc*n;#total conductor
E=F*Z*N*P/(60*A) #emf induced

#result
print "e.m.f induced E = " , E, "volts"

#determine the speed

#variable declaration
P=6 #poles
A1=2 #wave wound
Z=780 #armature conductors
F=12*10**-3 #webers 
E=400 #volt
A2=6 #wave wound
#calculation
N=(E*60*A1)/(F*Z*P) #rpm
N2=(E*60*A2)/(F*Z*P) #rpm

#result
print " Speed of the armature = " , round(N,2) , "rpm"
print " Speed when lap is wound = " , round(N2,1) , "rpm"

#determine the emf induced

#variable declaration
R=0.5 #ohm
Rs=100.0 
V=250.0 #volts
P=10000.0 #watts

#calculation
I=P/V #ampere
Is=V/Rs 
Ia=I+Is 
Eg=V+(R*Ia) #volts

#result
print " emf induced Eg = " , Eg , "volts"

#calculate the emf induced in the armature

#variable declaration
Il=200 #amperes
Vl=500 #volts
Ra=0.03 #ohm
Rs=0.015
R=150
BCD=2  #one volt per brush

#calculation
I=Vl/R #ampere
Ia=Il+I 
Eg=Vl+(Ia*Ra)+(Ia*Rs)+BCD #volts

#result
print " emf induced Eg = " , round(Eg,2) , "volts"

#round off error in book

#calculate the emf induced in the armature

#variable declaration
I1=200 #ampere
Vl=500 #volts
Ra=0.03 #ohm
Rs=0.015
Is=200  #ampere
R=150 #ohm

#calculation
BCD=2  #one volt per brush
I=(Vl+(Is*Rs))/R #ampere
Ia = I1 + I
Eg=Vl+(Ia*Ra)+(Ia*Rs)+BCD #volts

#result
print " emf induced Eg = " , round(Eg,2) ,"volts"

#Error in book

#calculate the back emf induced on full load

#variable declaration
Ra=0.5  #armature resistance
Rs=250 #shunt resistance
Vl=250  #line volt
Il=40 #ampere

#calculation
Is=Vl/Rs #amperes
Ia=Il-Is
Eb=Vl-(Ia*Ra) #volts

#result
print "emf induced  Eb = ", Eb, "volts"  

#find the power developed in circiut

#variable declaration
Pl=20e3 #watts
Vl=200.0 #volts 
Ra=0.05 #ohms
R=150.0

#calculation
I=Vl/R #ampere
Il=Pl/Vl
Ia=Il+I
Eg=Vl+(Ia*Ra) #volts
P=Eg*Ia #watts

#result
print "power developed = " , round(P,2) , "watt"

#round off error in book

#calculate the speed of the machine when running

#variable declaration
N1=1000  #speed of generator
E1=205.06  #emf generator
E2=195.06  #emf of motor

#calculation
N2=(E2*N1)/E1  #speed of generator

#result
print"speed of motor = " , round(N2,2) ,"rpm"

#dtermine its speed when its take crnt 25 amps

#variable declaration
Vl=250.0 #volts
Ra=0.05 #ohm
R=0.02 #ohm
Ia=30.0 #ampere
I1=30.0
N1=400.0
I2=25.0

#calculation
E1=Vl-(Ia*Ra)-(Ia*R) #volts
N2=(N1*E1*I1)/(E1*I2) #rpm

#result
print "speed of motor = " , round(N2,2) ,"rpm"

#round off error in book

#find the torque whn its take scurnt 60amprs

#variable declaration
Vl=200 #volts
Il=60 #amperes
R=50 #ohm
f=0.03  # flux 
Z=700 #armature conductors
P=4 #pole
A=2

#calculation
I=Vl/R  # amperes
Ia=Il-I  #amperes
T=(0.159*f*Z*Ia*P)/A

#result
print " Torque = " , round(T,2) , "N-m"

#calcute the num of prim turns and prim $sec current

#variable declaration
KVA = 50.0
E1 = 6000.0 #volts
E2 = 250.0 #volts
N2 = 52.0 #number of turns

#calculation
N1=N2*E1/E2
I2=KVA*1000/E2 #ampere
I1=KVA*1000/E1 #ampere

#result
print " primary number of turns = " , N1 , "turns"
print " secondary current = " , I2, "amperes"
print " primary current = " , round(I1,2), "amperes"

#determine the emf induced in the secondry max value of flux density

#calculation
f=50 #Hz
N1=350 #turns
N2=800 #turns
E1=400 #volts
A=75e-4 #m**2

#calculation
E2=(N2*E1)/N1 #volts
Bm=E1/(4.44*f*A*N1) #Wb/m**2

#result
print " flux density = " , round(Bm,3) , "wb/m**2"

import math

#find the magnetic nd iron loss component of current

#variable declaration
E1=440 #volts
E2=200 #volts
I=0.2 #amperea
coso=0.18 #p.f.

#calculation
sino= math.sqrt(1-coso**2) 
Iw=I*coso #ampere
Iu=I*sino #ampere

#result
print " Magnetising compenet of current = " , round(Iw,3), "amperes"
print " iron loss compenet of current = " , round(Iu,4), "amperes"

#calculate the efficiency at loads

#variable declaration
KVA=20
Il=350 #iron loss
Cl=400 #copper loss
x=1 # fraction of load
pf=0.8 # at full load
pf1=0.4  #at half load
x1=0.5 #fraction of load

#calculation
op=KVA*1000*x*pf
op1=KVA*1000*x1*pf1
Tl=Il+(Cl*x*x)
Tl1=Il+(Cl*x1*x1)
ip=op+Tl
ip1=op1+Tl1
n=op/ip*100
n1=op1/ip1*100

#result
print "efficiency at half load = " , round(n,2) 
print "efficiency at full load = " , round(n1,2) 

#calculate the synchronous speed ,slip,frequncy induced emf

#variable declaration
f=50.0 #Hz
p=4 #poles 
N=1460.0 #rpm

#calculation
Ns=120*f/p #rpm
s=(Ns-N)/Ns #slip
f1=(s*f) #Hz

#result
print "synchronous speed Ns = " , Ns , "rpm"
print "slip s = " , round(s,3)
print " Frequency of rotor induced emf f = " , round(f1,2) , "Hz"

#determine the value of slip nd speed of motor

#variable declaration
P=6 #pole
f=50 #Hz
f1=1.5

#calculation
Ns=120*f/P
s=f1/f
N=Ns*(1-s)

#result
print " speed of motor = ", N, " RPM"
print " slip = " , round(s,3)

#calculate the numbers of poles ,slip at full load,frequncy rotor,speed of motor

#variable declaration
Ns=1000.0 #rpm
N=960
f=50 #Hz

#calculation
P=120*f/Ns  #synchronous speed
s=(Ns-N)/Ns #slip 
f1=s*f #Hz
N=Ns*(1-0.08) #speed of motor at 8% slip

#result
print " number of poles  p = " , P
print " slip s = " , round(s,2)
print " Frequency of rotor emf f = " , f , "Hz"
print " Speed of motor at 8% slip N = " , N , "RPM"

#calculate the induced emf per phase

#variable declaration
f=50 #Hz
P=16 #poles
N=160 #rpm
S=6 #slip
F=0.025 #flux

#calculation
n=N*S #conductors
Z=n/3 
e=2.22*F*f*Z #rms value

#result
print "Induced emf per phase e = " , e , "volts"

