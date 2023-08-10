#importing modules
import math
from __future__ import division

#Variable declaration
W=(3.14/3)    #Angular frequency in radian



#Calculations
t=((3.14)/(3*W))

#Result
print"The time taken to move from one end of its path to 0.025m from mean position is %i"%t,"sec"

#importing modules
import math
from __future__ import division

#Variable declaration
T=31.4          #Time Period
A=0.06          #Amplitude


#Calculations
W=((2*3.14)/T)
Vmax=W*A

#Result
print"The Maximum Velocity is",Vmax,"m/sec"

#importing modules
import math
from __future__ import division

#Variable declaration
m=8          #mass
g=9.8        #acceleration due to gravity
x=0.32       #Stretched spring deviation
m2=0.5       #mass of the other body


#Calculations
k=((m*g)/x)
T=((2*3.14)*math.sqrt(m2/k))

#Result
print"The Time Period of Oscillation for the other body is %0.2f"%T,"sec"

#importing modules
import math
from __future__ import division

#Variable declaration
Q=10**4    #Quality Factor
f=250      #Frequency


#Calculations
Tau=((Q)/(2*3.14*f))
t=((math.log(10,10)*20)/(0.4342944819*3.14))

#Result
print"The Time Interval is %2.2f"%t,"sec"

#importing modules
import math
from __future__ import division

#Variable declaration
Q=2000    #Quality Factor
f=240      #Frequency


#Calculations
Tau=((Q)/(2*3.14*f))
t=4*Tau

#Result
print"The Time in which the amplitude decreases is %1.1f"%t,"sec"

#importing modules
import math
from __future__ import division

#Variable declaration
A=50/1.4  #Amplitude which is A=(50f/1.4*W**2)
Amax=50   #Max Amplitude which is Amax=(50f/W**2)


#Calculations
Rat=A/Amax

#Result
print"The Value of A/Amax is %0.2f"%Rat

