#importing module
import math
from __future__ import division

#Variable declaration
RL=8 #in ohm
VCC=30 #in V

#Calculations
IC_max=VCC/RL 
VCE_max=VCC
IC=VCC/(2*RL)
VCE=VCC-(IC*RL)
PT=VCE*IC

#Result
print("maximum collector current= %1.2f A\n" %IC_max)
print("Maximum collector-emiiter voltage= %i V\n" %VCE_max)
print("Maximum Power rating= %2.2f W" %PT) 

#importing module
import math
from __future__ import division

#Variable declaration
VDD=25 #voltage axis intersection point in V
ID=4 #current in A

#Calculations
RD=VDD/ID
ID=VDD/(2*RD)
VDS=VDD-(ID*RD)
PT=VDS*ID

#Result
print("Drain Resistance= %1.2f ohm\n" %RD)
print("Drain current at maximum power ditribution point= %i A\n" %ID)
print("Drain-to-source voltage at maximum power dissipation point= %2.1f V\n" %VDS)
print("Maximum power dissipation= %i W" %PT)

#importing module
import math
from __future__ import division

#Variable declaration
beta1=20 #bjt gain
beta2=20 #bjt gain

#Calculations
beta0=beta1+beta2+(beta1*beta2)

#Result
print("net common-emitter current gain= %i" %beta0) #The answer in the textbook is mathematically incorrect

#importing module
import math
from __future__ import division

#Variable declaration
TJ_max=150 #in C
Tamb=27 #in C
Rth_dp=1.7 #Thermal resistance in C/W
Rth_pa=40 #in C/W
Rth_ps=1 #in C/W
Rth_sa=4 #in C/W

#Calculations
PD1_max=(TJ_max-Tamb)/(Rth_dp+Rth_pa)
PD2_max=(TJ_max-Tamb)/(Rth_dp+Rth_sa+Rth_ps)

#Result
print("Case(a):No heat sink used :-Maximum power distribution= %1.2f W\n" %PD1_max)
print("Case(b):Heaat sink used :- Maximum power distribution= %2.2f W" %PD2_max)

#importing module
import math
from __future__ import division

#Variable declaration
B=10 #current gain
IB=0.6 #in A
VBE=1 #in V
RC=10 #in ohm
VCC=100 #in Vs

#Calculations
IC=B*IB #in A
VCE=VCC-(IC*RC) #in V
VCB=VCE-VBE #in V
PT=(VCE*IC)+(VBE*IB)

#Result
print("Total power dissipation= %.1f W" %PT)
print("The BJT is working outside the SOA")

#importing module
import math
from __future__ import division

#Variable declaration
Beff=250 #effective gain
B1=25 #current gain of transistor
B2=8.65 #effective gain of Darlington-pair
iB=50*10**-3 #in A

#Calculations
iC2=iB*(Beff-B1)
iE2=(1+(1/B2))*iC2

#Result
print("Emitter current= %2.2f A" %iE2)

#importing module
import math
from __future__ import division

#Variable declaration
VBB=24 #in V
r1=3 #in k-ohm
r2=5 #in k-ohm

#Calculations
n=r1/(r1+r2)
VP=(n*VBB)+0.7

#Result
print("peak-point voltage= %1.1f V" %VP)

#importing module
import math
from __future__ import division

#Variable declaration
Rth_sink=4 #resistance in C/W
Rth_case=1.5 #in C/W
T2=200 #Temperature in C
T1=27 #Room temperature in C
P=20 #power in W

#Calculations
Rth=(T2-T1)/P
Tdev=T2
Tamb=T1
Rth_dp=Rth 
Rth_ps=Rth_case #case-sink resistance
Rth_sa=Rth_sink #sink-ambient resistance
PD=(Tdev-Tamb)/(Rth_dp+Rth_ps+Rth_sa)

#Result
print("Actual power dissipation= %2.2f W" %PD) #The answers vary due to round off error



