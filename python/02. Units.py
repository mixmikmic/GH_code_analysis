import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from astropy import units as u
from astropy import constants as c

R = 8.5 * u.kpc

R = u.Quantity(8.5,unit=u.kpc)

R

R.value

R.unit

R2 = 8500 * u.pc

R == R2

R = R.to(u.AU)
R

c.G

v = 220 * u.km / u.s
v

Mg = v**2 * R / c.G
Mg

Mg.decompose()

Mg.decompose(u.cgs.bases)

Mg.to(u.Msun)

Mg / c.M_sun

(Mg / c.M_sun).decompose()

@u.quantity_input(mg=u.kg,r=u.AU)
def velocity(mg,r):
    return (c.G * mg / r)**0.5
    
velocity(Mg,5*u.AU)

velocity(Mg,5*u.kg)

r = np.arange(1,100)
r = r * u.AU

v = velocity(Mg,r)

plt.plot(r,v)
plt.show()

v

from astropy.units import imperial
imperial.tsp


V = 4/3 * np.pi * c.R_sun**3
V.to(imperial.tsp)

(450. * u.nm).to(u.GHz)

(450. * u.nm).to(u.GHz, equivalencies=u.spectral())

u.nm.find_equivalent_units()

u.nm.find_equivalent_units(equivalencies=u.spectral())

