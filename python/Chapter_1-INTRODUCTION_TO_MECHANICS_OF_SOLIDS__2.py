#downstream direction as x
#direction across river as y

from math import sqrt,atan,pi

#variable declaration

Vx= 8                       #velocity of stream, km/hour
Vy=float(20)                       #velocity of boat,km/hour

V=sqrt(pow(Vx,2)+pow(Vy,2)) #resultant velocity, km/hour
theta=Vy/Vx

alpha= atan(theta)*180/pi   #angle, degrees     

print " The resultant velocity :",round(V,2),"km/hour"
print round(alpha,2),"°"




#components of force in horizontal and vertical components. 
from math import cos,sin,pi
#variable declaration

F= 20                        #force in wire, KN

#calculations
Fx= F*cos(60*pi/180)          
Fy= F*sin(60*pi/180)

print round(Fx,2),"KN" ,"(to the left)"
print round(Fy,2), "KN" ,"(downward)"




 #The plane makes an angle of 20° to the horizontal. Hence the normal to the plane makes an angles of 70° to the horizontal i.e., 20° to the vertical
from math import cos,sin,pi
#variable declaration
W= 10                        # black weighing, KN

#calculations

Nor= W*cos(20*pi/180)             #Component normal to the plane
para= W*sin(20*pi/180)            #Component parallel to the plane

print "Component normal to the plane :",round(Nor,2),"KN"
print "Component parallel to the plane :",round(para,2) , "KN"





#Let the magnitude of the smaller force be F. Hence the magnitude of the larger force is 2F

from math import pi,sqrt, acos
#variable declaration
R1=260            #resultant of two forces,N
R2=float(180)          #resultant of two forces if larger force is reversed,N



#calculations

F=sqrt((pow(R1,2)+pow(R2,2))/10)
F1=F
F2=2*F
theta=acos((pow(R1,2)-pow(F1,2)-pow(F2,2))/(2*F1*F2))*180/pi

print "F1=",F1,"N"
print  "F2=",F2,"N"
print "theta=",round(theta,1),"°"




#Let ?ABC be the triangle of forces drawn to some scale
#Two forces F1 and F2 are acting at point A
#angle in degrees '°'

from math import  sin,pi
  
#variabble declaration
cnv=pi/180

BAC = 20*cnv                           #Resultant R makes angle with F1    
 
ABC = 130*cnv                    

ACB = 30*cnv   

R =  500                            #resultant force,N

#calculations
#sinerule

F1=R*sin(ACB)/sin(ABC)
F2=R*sin(BAC)/sin(ABC)

print "F1=",round(F1,2),"N"
print "F2=",round(F2,2),"N"




#Let ABC  be the triangle of forces,'theta' be the angle between F1 and F2, and 'alpha' be the angle between resultant and F1 

from math import sin,acos,asin,pi

#variable declaration
cnv= 180/pi
F1=float(400)                         #all forces are in newtons,'N'
F2=float(260)
R=float(520)

#calculations

theta=acos((pow(R,2)-pow(F1,2)-pow(F2,2))/(2*F1*F2))*cnv

alpha=asin(F2*sin(theta*pi/180)/R)*cnv

print"theta=",round(theta,2),"°"
print "alpha=",round(alpha,2),"°"



#The force of 3000 N acts along line AB. Let AB make angle alpha with horizontal.

from math import cos,sin,pi,asin,acos

#variable declaration
F=3000                        #force in newtons,'N'
BC=80                         #length of crank BC, 'mm'
AB=200                        #length of connecting rod AB ,'mm'
theta=60*pi/180               #angle b/w BC & AC

#calculations

alpha=asin(BC*sin(theta)/200)*180/pi

HC=F*cos(alpha*pi/180)                    #Horizontal component 
VC= F*sin(alpha*pi/180)                   #Vertical component 

#Components along and normal to crank
#The force makes angle alpha + 60  with crank.
alpha2=alpha+60
CAC=F*cos(alpha2*pi/180)             # Component along crank 
CNC= F*sin(alpha2*pi/180)             #Component normal to crank 


print "horizontal component=",round(HC,1),"N"
print "Vertical component = ",round(VC,1),"N"
print "Component along crank =",round(CAC,1),"N"
print "Component normal to crank=",round(CNC,1),"N"

