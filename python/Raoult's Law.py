from IPython.display import YouTubeVideo
YouTubeVideo('Adr9_2LnQdw') 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Antoine's equations
A = 'acetone'
B = 'ethanol'

def PsatA(T):
    return 10**(7.02447 - 1161.0/(T + 224))

def PsatB(T):
    return 10**(8.04494 - 1554.3/(T + 222.65))

PsatA(56)
PsatB(78.3)

def PsatMixture(T):
    return 0.30*PsatA(T) + (1-0.30)*PsatB(T)

PsatMixture(70.9)
Tbub = 70.9

yA = 0.30*PsatA(Tbub)/760
print(yA)



def fDew(T):
    return 0.3*760/PsatA(T) + 0.7*760/PsatB(T)

fDew(74.05)

# Plot pure component vapor pressures
T = np.linspace(20,80)

plt.plot(T,PsatA(T),'b')
plt.plot(T,PsatB(T),'r')

plt.ylabel('Pressure [mmHg]')
plt.xlabel('Temperature [C]')
plt.legend([A,B],loc='best')
plt.title("Vapor Pressure of the Pure Components from Antoine's Equations")
plt.grid();

# Plot vapor pressure using Raoult's law at 32 deg C
T = 32
x = np.linspace(0,1)

plt.plot(x,[x*PsatA(T) + (1-x)*PsatB(T) for x in x],'b')
plt.plot(x,[x*PsatA(T) for x in x],'r--')
plt.plot(x,[(1-x)*PsatB(T) for x in x],'g--')

plt.xlim(0,1)
plt.ylim(0,350)
plt.xlabel('Mole fraction ' + A, fontsize = 14)
plt.ylabel('Vapor Pressure [mmHg]', fontsize = 14)
plt.title('Raoult\'s Law: ' + A + ' / ' + B + ' at {:.1f} deg C'.format(T), fontsize = 14)
plt.legend(['Total Pressure: Raoult\'s Law','Vapor Pressure of Acetone','Vapor Pressure of Ethanol'],loc='best')
plt.grid()

plt.plot(x,[x*PsatA(T) + (1-x)*PsatB(T) for x in x],'b')
plt.plot(x,[x*PsatA(T) for x in x],'r--')
plt.plot(x,[(1-x)*PsatB(T) for x in x],'g--')

# Experimental data of (P,x) observations
Px = np.array([    [11.679, 0.00000],    [14.999, 0.04220],    [16.585, 0.06730],    [19.358, 0.11300],    [22.571, 0.17870],    [24.811, 0.23610],    [25.585, 0.26650],    [27.384, 0.31280],    [28.371, 0.34700],    [31.037, 0.44580],    [33.037, 0.53720],    [35.370, 0.65480],    [37.584, 0.77210],    [38.890, 0.84740],    [40.130, 0.92520],    [41.317, 1.00000]])

# Convert kPa to mmHg
P = Px.T[0]*760/101.3

# Convert K to C
T = 305.15 - 273.15

# Extract measured composition
x = Px.T[1]

# Overlay plot of experimental data and label$$ 
plt.plot(x,P,'ro')
plt.xlim(0,1)
plt.ylim(0,350)
plt.xlabel('Mole fraction ' + A, fontsize = 14)
plt.ylabel('Vapor Pressure [mmHg]', fontsize = 14)
plt.title('Raoult\'s Law: ' + A + ' / ' + B + ' at {:.1f} deg C'.format(T), fontsize = 14)
plt.legend(['Total Pressure: Raoult\'s Law','Vapor Pressure of Acetone','Vapor Pressure of Ethanol','Experimental'])
plt.grid()



