import math as m
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np

def thetaI(M=2, thick=1/8):
#    thick   = 1/8
#    M       = 2
    thetaMu = m.asin(1/M)
    deltaY  = thick/(2*m.tan(thetaMu))
    thetaS  = m.atan(9/6.42)
    deltaE  = deltaY*m.sin(m.pi/2-thetaS)
    thetaE  = m.atan(thick/(2*deltaE))
    thetaI  = m.pi-2*thetaE
    return thetaI

print(thetaI()*360/2/m.pi)

Ms= np.linspace(start=1.1, stop=3, num=1e3)
thetas= [thetaI(M)*360/2/m.pi for M in Ms]
plt.plot(Ms, thetas)
plt.grid()

thick= 1/8
thetaI= 120/360*m.pi*2
thetaE= m.pi/2-thetaI/2
# tan(thetaE)=thick/2/deltaE
deltaE= thick/2/m.tan(thetaE)
print(thick)
print(thetaI*360/2/m.pi)
print(thetaE*360/2/m.pi)
print(deltaE)
print(deltaE*2)

