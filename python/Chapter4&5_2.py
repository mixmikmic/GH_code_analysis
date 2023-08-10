#importing modules
import math
from __future__ import division

#Variable declaration
muclad=1.48   #Refractive index of claddings
mucore=1.5    #Refractive index of core

#Calculations
thetac=math.degrees(math.asin(muclad/mucore))
fri=(mucore-muclad)/mucore
aa=(math.sqrt((mucore**2)-(muclad**2)))
NA=math.sin(aa)
#Result
print"(a) The critical angle is :%2.2f"%thetac,"degrees"
print"(b) The Fractional refractive index is :%1.3f"%fri
print"(c) The Acceptance angle is :%1.3f"%aa,"Radians"
print"(d) The Numerical Apperture is :%1.3f"%NA

#importing modules
import math
from __future__ import division

#Variable declaration
a=25*10**-6                   #core radius
lambdaa=0.85*10**-6    #Wavelength
NA=0.22                #Numerical Aperture

#Calculations
V=((2*3.14*a*0.22)/lambdaa)
N=((V**2)/4)

#Result
print"(a) The V number is %2.2f"%V
print"(b) The number of modes are %3.2f"%N

#Note: The answer in the book is wrongly stated as 40.66 and 413.31

#importing modules
import math
from __future__ import division

#Variable declaration
c=3*10**8
delf=3000    #Bandwidth

#Calculations
lc=(c/delf)

#Result
print"The coherence length of the laser beam is",lc,"m or 10**5 m"

#importing modules
import math
from __future__ import division

#Variable declaration
lambdaa=5*10**-5        #Wavelength
theta=32                #Angle subtended by the sun at the slit

#Calculations
l=((lambdaa*60*180)/(theta*3.14))

#Result
print"The transverse coherence length is %1.3f"%l,"cm"

#importing modules
import math
from __future__ import division

#Variable declaration
lambdaa=5400*10**-10        #Wavelength
tc=10**-10                  #coherence time
c=3*10**-8                 

#Calculations
dom=((lambdaa)/(tc*c))*10**-10

#Result
print"The Degree of Monochromaticity is %2.0f"%dom,"*10**-6"

