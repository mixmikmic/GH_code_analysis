import math

#Variables
   
Vpk = 1.0                     #Peak-to-peak voltage (in volts)
Tby2 = 0.1                    #Half-period (in seconds)
tau = 0.25                    #Time constant (in seconds)  

#Calculation

Vc = 0.5 * math.exp(-Tby2/tau)     #Output voltage (in volts)

#Result

print "Output peak voltage is ",round(Vc,1)," V."

import math

#Variables

RC = 250.0 * 10**-12                   #Time constance (in seconds)
Vomax = 50.0                           #Maximum output voltage (in volts)                         
tau = 0.05  * 10**-6                   #time (in seconds)

#Calculation

alpha = Vomax / RC                     #alpha (in volt per second)
Vp = alpha * tau                       #Peak voltage (in volts)

#Result

print "The peak value of input voltage is ",Vp * 10**-3," kV."

import math
import numpy
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import plot,ylim,xlabel,ylabel,title,annotate

#Variables

Vi = 2.0             #positive clipping (in volts)

#Calculation

Vomax = 5.0          #Maximum output voltage (in volts)    

#Result

print "Following graph shows the output.\nThe part above the line is clipped out."

#Graph

t = numpy.linspace(0.001,2,400)
y = numpy.sin(2*math.pi*t)
plot(t, 5*y)
plot(t,(2*t)/t,'--')
ylim( (-6,6) )
ylabel('Vo')
xlabel('t')
title('Output Waveform')
annotate("Clipping level",xy=(0.575,2))

import math
import numpy
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import xlim,ylim,plot,title,xlabel,ylabel

#Variables 

Vp = 10.0                      #Peak to peak voltage (in volts)

#Calculation

Vi = 4.0                       #Input voltage (in volts)
Vo = 1.0                       #Maximum output voltage (in volts)

#Graph

x = numpy.linspace(0,11,400)
y = numpy.sin(x)
xlim((0,14))
ylim((0,3))
plot(x,5*y - 4)
plot(x,x-x+1,"--")
title("Output waveform")
xlabel("t")
ylabel("Vo")

import math
import numpy
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import plot,ylim,ylabel,xlabel,title,annotate

#Variables

Vp = 10.0               #Peak-to-peak voltage (in volts)

#Result

print "Following is the graph :"

#Graph

t = numpy.arange(0.001, 2.0, 0.005)
y = numpy.sin(2*math.pi*t)
plot(t, 5*y)
plot(t,(-3*t)/t,'--')
ylim( (-6,6) )
ylabel('Vo')
xlabel('t')
title('Output Waveform')
annotate("Clipping level",xy=(0.575,-2.9))

import math
import numpy
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import plot,ylim,ylabel,xlabel,title

#Variables

Vp = 5.0                        #Peak voltage (in volts)

#Calculation

Vpos = 3.0                      #Positive clipping voltage (in volts)
Vneg = -2.0                     #Negative clipping voltage (in volts)

#Result

print "Following is the output :"
#Graph

t = numpy.arange(0.001, 2.0, 0.005)
y = numpy.sin(2*math.pi*t)
plot(t, 5*y)
plot(t,(3*t)/t,'--')
plot(t,(-2*t)/t,'--')
ylim( (-6,6) )
ylabel('Vo')
xlabel('t')
title('Output Waveform')

import math
import numpy
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import plot,ylim,xlabel,ylabel,title

#Variables

Vp = 10.0               #Peak-to-peak voltage (in volts)

#Calculation

Vpos = 3.0              #Positive clipping voltage (in volts)
Vneg = -2.0             #Negative clipping voltage (in volts)

#Result

print "Output waveform is as follows :\nThe parts below and above the clipping levels are clipped off."

#Graph

t = numpy.arange(0.001, 2.0, 0.005)
y = numpy.sin(2*math.pi*t)
plot(t, 10*y)
plot(t,(-2*t)/t,'--')
plot(t,(3*t)/t,'--')
ylim( (-11,11) )
ylabel('Vo')
xlabel('t')
title('Output Waveform')

import math
import numpy
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import plot,ylim,xlabel,ylabel,title

#Variables

Vp = 8.0                   #Peak voltage (in volts) 
Vo = 2.7                   #Maximum acceptance voltage (in volts)
Io = 1.0 * 10**-3          #maximum output current (in Ampere)

#Calculation

VR1 = Vp - Vo              #Maximum voltage drop across R1 (in volts) 
R1min = VR1 / Io           #Resistance (in ohm)

#Result

print "The value of R1 is ",R1min * 10**-3," kilo-ohm."

#Graph

k = numpy.arange(0.0001, 5.0, 0.0005)
k1= numpy.arange(5.0, 10.0, 0.0005)
k2= numpy.arange(10.0, 15.0, 0.0005)
k3= numpy.arange(15.0, 20.0, 0.0005)

m=numpy.arange(-2.7,2.7, 0.0005)
x1=(0.001*m)/m
x5=(5*m)/m
x10=(10*m)/m
x15=(15*m)/m

plot(k,-2.7*k/k,'b')
plot(k1,2.7*k1/k1,'b')
plot(k2,-2.7*k2/k2,'b')
plot(k3,2.7*k3/k3,'b')
plot(x1,m,'b')
plot(x5,m,'b')
plot(x10,m,'b')
plot(x15,m,'b')
ylim( (-5,5) )
ylabel('Vo')
xlabel('t')
title('Output')

import math
import numpy
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import plot,ylim,title,xlabel,ylabel

#Variables

Vp = 10.0            #Peak voltage (in volts)

#Calculation

Vi = 3.0             #Input voltage (in volts)     
Vo = Vi - 2.0        #Output voltage (in volts) 

#Graph

t = numpy.linspace(0,2,100)
t1 = numpy.linspace(2,4,100)
t2 = numpy.linspace(4,9,100)
t3 = numpy.linspace(9,11,100)
t4 = numpy.linspace(11,21,100)
t5 = numpy.linspace(21,23,100)
t6 = numpy.linspace(23,28,100)
t7 = numpy.linspace(28,30,100)
t8 = numpy.linspace(30,32,100)

ylim((0,5))
plot(t1,0.5*t1 -1,'b')
plot(t2,(1*t2)/t2,'b')
plot(t3,1 -0.5*(t3-9))
plot(t5,0.5*(t5-21),'b')
plot(t6,(1*t6)/t6,'b')
plot(t7,1-0.5*(t7-28),'b')
plot(t8,(0*t8/t8),'b')
title("Output waveform")
xlabel("time (t) ->")
ylabel("Output voltage (Vo) ->")

import math
import numpy
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import ylim,xlim,plot,title,xlabel,ylabel

#Variables

VSmax = 20.0                    #peak voltage (in volts)
VD2 = 0.7                       #Voltage drop across diode D2 (in volts) 

#Calculation

Vomin = 5.0 - VD2               #Minimum output voltage (in volts)  
Vomax = 10.7                    #Maximum output voltage (in volts)  

#Graph

t = numpy.linspace(0,4,100)
t1 = numpy.linspace(4,10,100)
t2 = numpy.linspace(10,20,100)
t3 = numpy.linspace(20,24,100)
t4 = numpy.linspace(24,30,100)
t5 = numpy.linspace(30,40,100)

ylim((0,15))
xlim((0,40))
plot(t,4.3+t-t,'b')
plot(t1,4.3 + 6.4/6*(t1 - 4))
plot(t2,(10.7)+t2-t2,'b')
plot(t3,(4.3+t3)-t3,'b')
plot(t4,4.3 + 6.4/6*(t4 - 24),'b')
plot(t5,(10.7)+t5-t5,'b')

title("Output waveform")
xlabel("t (in ms) ->")
ylabel("Vo (in volt) ->")

import math
import numpy
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import plot,xlabel,ylabel,title

#Variables

Vp = 48.0                 #Peak-to-peak voltage (in volts)

#Graph

x = numpy.linspace(0,2 * math.pi,100)
y = numpy.sin(x)
plot(x,24 + 24*y)
plot(x,(24)+x-x,'--')
xlabel("Time(t)")
ylabel("Output voltage (Vo)")
title("Output waveform")

import math
import numpy
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import plot,ylim,xlabel,ylabel,title,subplot


#Variables

Vi = 10.0                    #Input a.c. voltage (in volts)

#Graph

#Positive Clipping
subplot(211)
k1= numpy.arange(0.0001, 0.500, 0.0005)
k2= numpy.arange(0.500, 1.000, 0.0005)
k3= numpy.arange(1.000,1.500, 0.0005)
k4= numpy.arange(1.500,2.000, 0.0005)
m = numpy.arange(-10,10,0.0005)

x5=(0.0500*m)/m
x10=(0.500*m)/m
x15=(1.000*m)/m
x25=(1.500*m)/m

plot(k1,10*k1/k1,'b')
plot(k2,-10*k2/k2,'b')
plot(k3,10*k3/k3,'b')
plot(k4,-10*k4/k4,'b')
plot(x10,m,'b')
plot(x15,m,'b')
plot(x25,m,'b')

ylim( (-12,12) )
ylabel('Vo')
xlabel('Positive Clipping')
title('Output Waveform 1')

#Negative Clipping
subplot(212)
k1= numpy.arange(0.0001, 0.500, 0.0005)
k2= numpy.arange(0.500, 1.000, 0.0005)
k3= numpy.arange(1.000,1.500, 0.0005)
k4= numpy.arange(1.500,2.000, 0.0005)
m = numpy.arange(-20,0,0.0005)

x5=(0.500*m)/m
x10=(0.500*m)/m
x15=(1.000*m)/m
x25=(1.500*m)/m

plot(k1,0*k1/k1,'b')
plot(k2,-20*k2/k2,'b')
plot(k3,0*k3/k3,'b')
plot(k4,-20*k4/k4,'b')
plot(x10,m,'b')
plot(x15,m,'b')
plot(x25,m,'b')

ylim( (-22,0) )
ylabel('Vo')
xlabel('t')
title('Output Waveform 2')

