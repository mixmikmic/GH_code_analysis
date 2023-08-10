import math
mcsquared=938
pc=500
Etb=math.sqrt(mcsquared**2+pc**2)
print(Etb)

def energyloss(Etb,losspercm):
    distancetraveled=[]
    Et=1000
    while Et>=500:
        Et=Etb-losspercm
        Etb=Et
        distancetraveled.append(Etb)
    print(len(distancetraveled))

energyloss(Etb,1.5)

import numpy
import matplotlib.pylab as plt
get_ipython().magic('matplotlib notebook')

x=numpy.linspace(0,900,1000)
E=1062-1.5*x

p = numpy.sqrt(E**2 - mcsquared**2)
#y=10*x
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.plot(x,p,'b-',markersize=12)
plt.title("Analytic Momentum vs. Distance")
plt.xlabel('Distance(cm)',fontsize=14)
plt.ylabel('Momentum',fontsize=14)

plt.subplot(1,2,2)
plt.plot(x,E,'b-',markersize=12)
plt.title("Analytic Energy vs. Distance")
plt.xlabel('Distance (m)',fontsize=14)
plt.ylabel('Energy',fontsize=14)



p=500
m = 938.0
E = math.sqrt(m**2 + p**2)
b=math.sqrt(1/((m/p)**2+1))   
print(b)
#b = p/E
#print(b)

v=b*(3e8)
print("The proton is originally traveling at %f meters per second."%v)

DISTANCETRAVELED=(1e-9)*v
print("The proton travels %f meters in one nanosecond."%DISTANCETRAVELED)

#1.5MeV for each cm
newenergy=Etb-(100*DISTANCETRAVELED)
print("The proton now has %f MeV of energy"%newenergy)



p=500
b=math.sqrt(1/((m/p)**2+1))   
v=b*(3e8)
print("The proton is originally traveling at %f meters per second."%v)

DISTANCETRAVELED=(10**-9)*v
print("The proton travels %f meters in one nanosecond."%DISTANCETRAVELED)

#1.5MeV for each cm
newenergy=Etb-(100*DISTANCETRAVELED)
print("The proton now has %f MeV of energy"%newenergy)

dt = 1e-9 # time step-size

def simenergyloss(p,mcsquared):
    momenta=[]
    distance=[]
    deltaE = []
    Etb=math.sqrt(mcsquared**2+p**2)
    energy=Etb
    newenergy=Etb
    totdist = 0
    
    while newenergy >=mcsquared:
        #b=math.sqrt(1/((.511/p)**2+1)) 
        #print(newenergy)
        newenergy = math.sqrt(p**2 + mcsquared**2)

        b = p/newenergy
        v=b*(3e8)
        #print(v)
        if v<=3000:
            break
            

        DISTANCETRAVELED=(dt)*v
        totdist += DISTANCETRAVELED
        
        #print(DISTANCETRAVELED,newenergy)
        
        dE = (1.5*DISTANCETRAVELED)
        newenergy -= dE
        
        #print("new energy here: %f" % (newenergy))
        psq = (newenergy**2-mcsquared**2)
        if psq<0:
            break
        p = math.sqrt(psq)
        momenta.append(p)
        distance.append(totdist)
        deltaE.append(dE)
        
    print(len(momenta))
    
    return momenta,distance,deltaE

    
    
#simenergyloss(500,938)

p,d,dE = simenergyloss(500,938)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.plot(d,p)
plt.xlabel('Distance (cm)')
plt.ylabel('Momentum (MeV/c)')




plt.subplot(1,2,2)
plt.plot(d,dE,'.',markersize=1)
plt.xlabel('Distance (cm)')
plt.ylabel('Momentum (MeV/c)')

newdeltaE = []
newdist = []

distwidth = 0.5
startdist = 0

etot = 0
for x,deltaE in zip(d,dE):
    
    if x>=startdist and x<startdist+distwidth:
        etot += deltaE
        
    elif x>startdist+distwidth:
        
        newdeltaE.append(etot)
        newdist.append(startdist + distwidth/2.0)
        
        startdist += distwidth
        etot = deltaE

plt.figure()
#plt.subplot(1,2,2)
plt.plot(newdist,newdeltaE,'o',markersize=2)
plt.xlabel('Distance (cm)')
plt.ylabel('Momentum (MeV/c)')
plt.ylim(0,10)



