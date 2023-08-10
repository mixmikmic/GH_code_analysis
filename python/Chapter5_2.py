import math

#Variable Declaration

Vcc=5                #in V
R1=1*10**3           #in ohm 
Vd=0.7               #Diode voltage in V
I0=1*10**-3          #High output current in A
Vilow=0              #Low input voltage

#Calculation
Voh=Vcc-I0*R1
Vol=Vilow+Vd

print "High output voltage(Voh)=",Voh,"V"
print "Low output voltage(Vol)=",Vol,"V"

import math

#Variable Declaration

Vbe=0.7                 #Base emitter voltage in V
Vce_sat=0.2             #Saturation voltage in V
R1=15*10**3             #in ohm
R2=27*10**3             #in ohm
Vcc=5                   #in V
Vbb=-5                  #in V
Rc1=2.7*10**3           #in ohm
R11=15*10**3            #in ohm
R21=27*10**3            #in ohm

#Calculation
#With Q2 on,
Vc2=Vce_sat             #Q2 is ON  
Vr1r2=Vc2-Vbb           #KVL
Vr1=R1*Vr1r2/(R1+R2)    #Voltage Divider Rule

Vb1=Vc2-Vr1             #KVL

#With Q1 off, 
Vrc1=Rc1*(Vcc-Vbb)/(Rc1+R11+R21)     #Voltage Divider Rule
Vc1=Vcc-Vrc1                         #KVL


#Results

print "With Q2 ON,"
print "Vc2=",round(Vc2,1),"V"
print "Vr1r2=",round(Vr1r2,1),"V"
print "Vr1=",round(Vr1,1),"V"
print "Vb1=",round(Vb1,1),"V"
print
print "With Q1 OFF,"
print "Vrc1=",round(Vrc1,1),"V"
print "Vc1=",round(Vc1,1),"V"

#Note: A round off error of 0.1 V is observed in Vr1 and Vb1 variables

import math

#Variable Declaration

If=20*10**-3                  #Forward current in A

#Calcualtions
#For the LED display
I7=7*If                       #Seven Segment Current in A
I_1by2=2*If                   #Current for 1/2 digit in A
It=3*I7+I_1by2                #Total Current in A

print "For the LED Display,"
print "Current for each 7 segment display=",round(I7*10**3),"mA"
print "Current for 1/2 (2 segment) display=",round(I_1by2*10**3),"mA"
print "Total current for 3 and 1/2 digits=",round(It*10**3),"mA"


#For the LCD Display
If=300*10**-6

I7=7*If                       #Seven Segment Current in A
I_1by2=2*If                   #Current for 1/2 digit in A
It=3*I7+I_1by2                #Total Current in A

print
print "For the LCD Display,"
print "Current for each 7 segment display=",round(I7*10**3,1),"mA"
print "Current for 1/2 (2 segment) display=",round(I_1by2*10**6),"micro ampere"
print "Total current for 3 and 1/2 digits=",round(It*10**3,1),"mA"

import math

#Variable Declaration
T0=1*10**-6        #Oscillator time period in s
N=16              #Modulus of the counters 
n=3               #No. of counters

#Calculations
T=T0*N**n         #Time period in s
f=1/T             #Frequency in Hz

#Results
print "Time period=",round(T*10**3,1),"ms"
print "Frequency=",round(f),"Hz"

import math

#Variable Declaration
Vr=1.25                     #in V
tr=125*10**-3               #in s
f=1.0*10**6                 #in Hz

#For Vi=0.9
Vi=0.9                     #in V
t1=tr*Vi/Vr                #in s 
T=1/f                      #in s
N=t1/T                     #No. of pulses counted  

print "For Vi=0.9V,"
print "t=",round(t1*10**3),"ms"
print "Pulses counted=",round(N)

#For Vi=0.75
Vi=0.75                   #in V
t1=tr*Vi/Vr               #in s 
N=t1/T                    #No. of pulses counted 

print "For Vi=0.75V,"
print "t=",round(t1*10**3),"ms"
print "Pulses counted=",round(N)

#**********************Error********************************
##Note:The count values obtained in text book are 900 and 750 
##Whereas the actual values are 900000 and 75000 respectively

import math

#Variable Declaration

#For 1% quantizing error count, count>=100
N=1
while(N):
    count=2**N-1
    if(count>=100):
        break 
    N=N+1

print "N=",N,"bit ADC is requird for quantizing error less than 1%"
        

import math

#Variable Declaration
a3=1          #bit
a2=0          #bit
a1=1          #bit
a0=0          #bit
Vi=10        #in V

#Calculations

Vo=(2**3*a3+2**2*a2+2**1*a1+a0)*Vi/16.0

print "Vo=",round(Vo,2),"V"
 

