import math as m
import numpy as np
import matplotlib
import matplotlib.pyplot as plot
import pylab
from matplotlib import rc
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_formats=['svg']")

## Try ZAERO
## Fin Geometry
## Note: Shorten and Thicken:
##       From t = 0.25 in. to 0.4 in.

Cr = 18*25.4/1e3# Chord, root (m)
Ct = 5*25.4/1e3# Chord, tip (m)
W  = 6.4*25.4/1e3# Semispan (m)
MAC = (Cr+Ct)/2# Chord, mean aero (m)
S = W*MAC # Fin area (m^2)
AR = W/MAC # Aspect ratio
La = Ct/Cr # Taper ratio
# t = 0.25 # Fin thickness (in.)
thick= 0.25

## Atmospheric Parameters

t = np.linspace(0, 200, 100000)
x = np.arange(100000) # Altitude array (m)
def atmo(x):
    if np.all(x < 11000):
        T = 15.04-0.0065*x
        P = 101.3*((T+273.1)/288.1)**5.26
    elif np.all(11000 <= x) & np.all(x < 25000):
        T = -56.46
        P = 22.65*m.exp(1.73-0.00016*x) 
    else:
        T = -131.2+0.003*x
        P = 2.488*((T+273.1)/216.6)**(-11.4)

    rho = P/(0.29*(T+273.1)) # Density, ambient air (kg/m^3)
    Pa  = P*1000             # Pressure, ambient air (Pa)
    Ta  = T+273.1            # Temperature, ambient air (K)
    a   = 20.05*m.sqrt(Ta)     # Speed of sound (m/s)
    return Pa, rho, Ta, a

# def atmo(x):
#    return [atmo_helper(xi) for xi in x] # give atmo_helper() the x values one-at-a-time

a = [atmo(xi)[3] for xi in x] # Speed of sound (m/s)
P = [atmo(xi)[0] for xi in x] # Pressure at altitude x (Pa)
Po = atmo(0)[0] # Sea level pressure (Pa)

## Material Properties

GE = 25.5 # Effective Shear Modulus

## Flutter
## Influences: Stiffness
##             Mach
##             Mass
## Guidelines: 15% FS for velocity V/Vf
##             32% FS for pressure Q/Qf

K1 = (39.3*AR**3)/((AR+2)*(thick/MAC)**3)
K2 = (La+1)/2
K3 = [Pi/Po for Pi in P]
Vf = [a[i]*m.sqrt(GE/(K1*K2*K3[i])) for i in range(0,len(a))] # Flutter velocity (m/s)

#def velo(t):
#    if np.all(t < number):

#Qf = # Flutter dynamic pressure (Pa)

print('\n')

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
pylab.rcParams['figure.figsize'] = (10.0, 10.0)
f, (ax1, ax2) = plot.subplots(2, sharex=True)
ax1.plot(x, Vf)
ax1.yaxis.major.locator.set_params(nbins=6)
ax1.set_title('LV3, Altitude (m) v Flutter Velocity (m/s)')
# ax1.plot(x, V)
ax1.yaxis.major.locator.set_params(nbins=6)
ax1.set_title('LV3, Altitude (m) v Velocity (m/s)')

