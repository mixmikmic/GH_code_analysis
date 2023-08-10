import numpy as np
import matplotlib.pyplot as plt

# The constants in SI units:
G = 6.67*10**(-11) # The Gravitational constant in m^3 s^(-2) kg^-1
M = 6*10**24 # The Earth's mass in kg
R = 6370*10**3 # The radius the Earth in m

#  A function to compute the magnitude of the (radial) gravitational acceleration, inside or outside the Earth:
def gravity(r):
    if r>R:
        g = M*G/r**2
    else:
        g =  M*G*r/R**3
    return g

# setting up an array with points from the centre (0) to well beyond the radius of the Earth:
rs= np.arange(0,8*10**6,1000)

# compute the magntiude of the gravitational acceleration for all points in the array rs:
gs = []
for r in rs:
    gs.append(gravity(r))

# plotting the results:
plt.plot(gs,rs/1000)
plt.axhline(R/1000,linestyle='dashed',color='r')
plt.xlabel("Gravitational acceleration (m/s$^2$)")
plt.ylabel("Distance to the centre of the Earth (km)")
plt.show()
                    



