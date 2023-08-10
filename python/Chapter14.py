#importing modules
import math
from __future__ import division

#Variable declaration
Vm=100     #voltage in V
Rf=1*10**3 #resistance in series in ohm
Rl=4*10**3 #load resistance in ohm

#Calculatiions
Im=Vm/(Rf+Rl)*10**3
Idc=Im/math.pi
Irms=Im/2

#Result
print("(a)Maximum current Im is (A)= ",Im,"mA")
print("(b)dc component of current Idc is (A)= %0.2f*10**-3 A" %Idc)
print("(c)rms value of current Irms (A)= ",Irms,"mA")

#importing modules
import math
from __future__ import division

#Variable declaration
Vm=200 #voltage in V
Rf=500 #resistance in series in ohm
Rl=1000 #load resistance in ohm

#Calculatiions
Im=Vm/(Rf+Rl) 
Idc=(2*Im)/math.pi
Irms=Im/math.sqrt(2)
Y=math.sqrt(((Irms/Idc)**2)-1)

#Result
print("(a)Maximum current Im= %0.3f A\n" %Im)
print("(b)dc component of current Idc= %1.4f A\n" %Idc)
print("(c)rms value of current Irms= %1.3f A\n" %Irms)
print("(d)Ripple Factor Y= %1.3f" %Y) #The answers vary due to round off error

#importing modules
import math
from __future__ import division

#Variable declaration
RL=500        #load resistance in ohm
C1=100*10**-6 #capacitance in F
C2=50*10**-6  #capacitance in F
L=5           #in H
f=50          #frequency in Hz

#Calculatiions
Y=0.216/(RL*C1*C2*L*(2*math.pi*f)**3)

#Result
print("Ripple factor Y= %0.5f" %Y) #The answers vary due to round off error

#importing modules
import math
from __future__ import division

#Variable declaration

Iz_min=1492.5*10**-3 #Zener diode current in Ampere
Vz=25                #Zener diode voltage in Volt

#Calculatiions
Pmin=Vz*Iz_min

#Result
print("Minimum Power Rating p= %2.1f W" %Pmin)



