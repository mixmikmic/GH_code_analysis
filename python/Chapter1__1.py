#importing modules
import math
from __future__ import division

#Variable declaration
D=0.5         #Distance from Screen
d=0.5        #Distance between parallel slits
lambdaa=5890  #Wavelength

#Calculations
Beta=(D*lambdaa)/(d)/10**4

#Result
print"The Fringe width in Youngs Double Slit Experiment is Beta= %1.4f" %Beta,"*10**-3 m"

#importing modules
import math
from __future__ import division

#Variable declaration
D=2             #Distance from screen
lambdaa=5100    #Wavelength
Beta=0.02       #Fringe Width
x=10            #No. of fringes


#Calculations
d=(x*D*lambdaa)/Beta/10**6

#Result
print"The Double slit separation 2d=",d,"mu m"

#importing modules
import math
from __future__ import division

#Variable declaration
D=1                     #Distance from screen
Beta=0.31*10**-3        #Fringe Width
d=1.9*10**-3            #Slit separation


#Calculations
lambdaa=(Beta*d*10**6)/D

#Result
print"The Wavelength lamda=%0.4f"%lambdaa,"*10**-6 m"

#importing modules
import math
from __future__ import division

#Variable declaration
D=0.04                     #Distance from screen
Lambdaa=5890*10**-10        #Wavelength
d=2*10**-3                 #Slit separation
n=10                       #No. of fringes


#Calculations
x10=(n*D*lambdaa*10**-2)/d

#Result
print"The position of the 10th fringe is",x10,"*10**-4 m"

#importing modules
import math
from __future__ import division

#Variable declaration
D=0.8                       #Distance from screen
Lambdaa=5890*10**-10        #Wavelength
Beta=9.424*10**-4           #Fringe Width


#Calculations
d=(D*lambdaa*10**-2)/Beta

#Result
print"The position of the 10th fringe is %i"%d,"*10**-4 m"

#importing modules
import math
from __future__ import division

#Variable declaration
D=1.1                       #Distance from screen
Lambdaa=5900*10**-10        #Wavelength
d=0.00174                   #Fringe separation


#Calculations
Beta=(D*lambdaa*10**-1)/d

#Result
print"The Fringe width observed at a distance of 1m from BP is %1.1f"%Beta,"*10**-5 m"

#importing modules
import math
from __future__ import division

#Variable declaration
D=2                     #Distance from screen
Lambdaa=5890*10**-10    #Wavelength
mu=1.5                  #refractive index of glass
a=0.25                  #distance from slit
Beta=0.2*10**-3         #Fringe width


#Calculations
alpha=(D*lambdaa*180*10**-6)/(2*a*(mu-1)*Beta*3.14)
A=(180-2*(round(alpha,2)))

#Result
print"The Angle of prism at the vertex is is %i"%A,"deg 17.8 min"

#importing modules
import math
from __future__ import division

#Variable declaration
D=1                        #Distance from screen
mu=1.5                     #refractive index of glass
a=0.5                      #distance from slit
Beta=0.0135*10**-2         #Fringe width
alpha=0.0087               #angleof prism


#Calculations
lambdaa=(Beta*2*a*(mu-1)*alpha*10**10)/D

#Result
print"The Wavelength is",lambdaa,"Angstrom"

#importing modules
import math
from __future__ import division

#Variable declaration
d=0.75                   #slit separation
Beta=0.087*10**-3        #Fringe width


#Calculations
Beta2=Beta*10**3/d

#Result
print"The fringe width would become ",Beta2,"mm"

#importing modules
import math
from __future__ import division

#Variable declaration
d=7.5*10**-4                   #slit separation
Beta=0.094*10**-2              #Fringe width
D=1.2                          #Distance from Screen


#Calculations
lambdaa=(Beta*d*10**10)/D

#Result
print"The wavelength is %i"%lambdaa,"Angstrom"

#importing modules
import math
from __future__ import division

#Variable declaration
d=3.6125*10**-3                #slit separation
D=1                            #Distance from Screen
lambdaa=5870*10**-10           #Wavelength


#Calculations
Beta=(D*lambdaa*10**4)/d

#Result
print"The Fringe width is",round(Beta,3),"*10**-4 m"

#importing modules
import math
from __future__ import division

#Variable declaration
d=0.3*10**-2                #slit separation
D=1                            #Distance from Screen
Beta=0.0195*10**-2           #Wavelength


#Calculations
lambdaa=(Beta*d*10**10)/D

#Result
print"The wavelength is %i"%lambdaa,"*10**-10 m"

#importing modules
import math
from __future__ import division

#Variable declaration
n1=62                   #fringes
lambdaa1=5893*10**-10   #Wavelength 1
lambdaa2=5461*10**-10   #Wavelength 2


#Calculations
n2=(n1*lambdaa1)/lambdaa2

#Result
print"The number of fringes would be %i"%round(n2)

#importing modules
import math
from __future__ import division

#Variable declaration
lambdaa=5.46*10**-7   #Wavelength
t=6.3*10**-6          #thickness

#Calculations
mu=((6*lambdaa)/t)+1

#Result
print"The refractive index is ",mu

#importing modules
import math
from __future__ import division

#Variable declaration
s=2.143*10**-3
mu=1.542  #refractive index
lambdaa=5893*10**-10 #Wavelength
Beta=0.347*10**-3

#Calculations
t=(s*lambdaa*10**6)/(Beta*(mu-1))

#Result
print"The refractive index is ",round(t,2),"mu m"

#importing modules
import math
from __future__ import division

#Variable declaration
mu=1.4                #Refractive index
cosr=0.8631
t=0.01*10**-3         #thickness
lambda1=4000*10**-10  #Wavelength 1
lambda2=5000*10**-10  #Wavelength 2


#Calculations
n1=(2*mu*t*cosr)/lambda1
n2=(2*mu*t*cosr)/lambda2
deln=round(n1)-round(n2)

#Result
print"The number of dark bands seen betwween 4000 A and 5000A  is %i"%deln

#importing modules
import math
from __future__ import division

#Variable declaration
mu=1.33                #Refractive index
cosr=0.7989
lambda1=6.1*10**-5     #Wavelength 1
lambda2=6*10**-5       #Wavelength 2


#Calculations
t=(lambda1*lambda2*10**-5)/(2*mu*cosr*(lambda1-lambda2)*10**-5)

#Result
print"The Thickness is ",round(t,4)," cm"

#importing modules
import math
from __future__ import division

#Variable declaration
n=8                   #number of fringes
lambdaa=5893*10**-10   #Wavelength
mu=1.5                #Refractive index
cosr=(2*math.sqrt(2))/3
#Calculations
t=(n*lambdaa*10**6)/(2*mu*cosr)

#Result
print"The Thickness is ",round(t,3)," mu m"

#importing modules
import math
from __future__ import division

#Variable declaration
mu=4/3            #refractive index
t=1.5             #thickness
cosr=0.7603
lambdaa=5*10**-7  #Wavelength


#Calculations
n=(2*mu*t*cosr*10**-6)/lambdaa

#Result
print"The order of interference of dark band is %i "%n

#importing modules
import math
from __future__ import division

#Variable declaration
mu=1.33    #refractive index
n1=0
n2=1
n3=2
t=5*10**-7 #thickness


#Calculations
lambda1=(4*mu*t*10**10)/(2*n1+1)
lambda2=(4*mu*t*10**10)/(2*n2+1)
lambda3=(4*mu*t*10**10)/(2*n3+1)

#Result
print"For n=0 Lambda is",lambda1  #The answer given in the book is 26000 however it is mathematically incorrect
print"For n=1 Lambda is %i"%lambda2    #The answer given in the book is 8666 however it is mathematically incorrect
print"For n=2 Lambda is",lambda3       # The answer given in the book is 5200 however it is mathematically incorrect
print"Out of these only",lambda3,"lies in the visible range for n=2"  

#importing modules
import math
from __future__ import division

#Variable declaration
R=100   #radius
D25=0.8 #Diameter of the 25th ring
D5=0.3  #Diameter of the 5th ring
p=20    


#Calculations
lambdaa=((D25**2)-(D5**2))*10**8/(4*20*100)

#Result
print"The Wavelength is %i"%lambdaa,"Angstrom"

#importing modules
import math
from __future__ import division

#Variable declaration
n=10                 #no. of ring
D10=0.5              #Diameter of the 10th ring
lambdaa=5893*10**-8  #Wavelength
    
#Calculations
R=(D10**2)/(4*10*5893*10**-8)
t=(D10**2)*10**4/(8*R)

#Result
print"The Thickness is",round(t,3),"cm"
print"The Radius is",round(R,1),"cm"

#importing modules
import math
from __future__ import division

#Variable declaration
n=59                 #no. of ring
lambdaa=6*10**-7     #Wavelength
R=0.9                #Radius
    
#Calculations
D59=math.sqrt(4*R*n*lambdaa)*10**2

#Result
print"The Diameter of the nth dark ring is",round(D59,3),"cm"

#importing modules
import math
from __future__ import division

#Variable declaration
n=20                 #no. of ring
lambdaaR=0.0103      #Wavelength*R
    
#Calculations
D20=math.sqrt(4*n*lambdaaR)

#Result
print"The Diameter of the 20th dark ring is",round(D20,3),"cm"

#importing modules
import math
from __future__ import division

#Variable declaration
D3=10**-2
lambdaa=5890*10**-10

    
#Calculations
R=(D3*math.sqrt(3))*10**-2/(24*lambdaa)

#Result
print"The Radius is",round(R,2),"m"

#importing modules
import math
from __future__ import division

#Variable declaration
n=8            #no. of ring
D8=0.72*10**-2 #Diameter of the 8th ring
R=3            #Radius

    
#Calculations
lambdaa=(D8**2)*10**10/((2*(2*n-1))*R)

#Result
print"The Wavelength is %i"%lambdaa,"Angstrom"

#importing modules
import math
from __future__ import division

#Variable declaration
c=3*10**10  #Speed of Light in Vacuum
mu=1.44     #Refractive Index

#Calculations
u=c*10**-10/mu

#Result
print"The Velocity in the liquid is",round(u,2),"*10**10   m/s"

#importing modules
import math
from __future__ import division

#Variable declaration
lambdaa=5400*10**-10  #Wavelength
n1=5
n2=15
R=100       #Radius of both rings

#Calculations
r5=math.sqrt((R*n1*lambdaa)/2)
r15=math.sqrt((R*n2*lambdaa)/2)
d=round(r15,4)-round(r5,4)

#Result
print"The Distance between 5th and 15th Dark ring is",d,"m"

#importing modules
import math
from __future__ import division

#Variable declaration
l=0.0025 #Distance moved
t=0.005  #thickness of mica sheet

#Calculations
mu=((l/t)+1)

#Result
print"The Refractive Index is",mu

#importing modules
import math
from __future__ import division

#Variable declaration
l=0.02948*10**-3 #Distance moved
n=100            #number of fringes

#Calculations
lambdaa=(2*l)*10**10/n

#Result
print"The Wavelength is %i"%lambdaa,"Angstrom"

#importing modules
import math
from __future__ import division

#Variable declaration
lambdaa1=5896  #Wavelength1
lambdaa2=5890  #Wavelength2


#Calculations
l=(lambdaa1*lambdaa2)/(2*(lambdaa1-lambdaa2))

#Result
print"The Distance by which the mirror moved is %i"%l,"*10**-10 m"

#importing modules
import math
from __future__ import division

#Variable declaration
lambdaa=5893*10**-10  #Wavelength
l=0.2945*10**-3       #Distance by which mirror is displaced


#Calculations
dellambdaa=(lambdaa**2)*10**10/(2*l)

#Result
print"The Difference between two wavelengths is ",round(dellambdaa,1),"Angstrom"

