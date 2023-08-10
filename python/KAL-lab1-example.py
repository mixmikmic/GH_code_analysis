''' 
Let's import the Python modules we'll need for this example!
This normally goes at the top of your script or notebook, so that all code 
below it can see these packages.

By the way, the triple-quotes give you blocks of comments
'''
# while the pound (or hash) symbol just comments things

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

# Let's read some data in

# file from: http://ds.iris.edu/ds/products/emc-prem/
# based on Dziewonski & Anderson (1981) http://adsabs.harvard.edu/abs/1981PEPI...25..297D
file = 'PREM_1s.csv'
radius, density = np.loadtxt(file, delimiter=',', usecols=(0,2), comments='#', unpack=True)

radius

density

# Now let's plot these data
plt.plot(radius, density)
plt.xlabel('Radius (km)')
plt.ylabel('Density (g/cm$^3$)')

radius.size


M = np.zeros(radius.size)

# loop over these, and save something in each one

for i in range(1,radius.size-1):
    M[i] = 4.0*np.pi*( (radius[i-1])**3.0-(radius[i])**3.0)*density[i]*1.0E12/3.0

plt.plot(radius,M)
plt.ylabel('Mass in each shell (kg)')
plt.xlabel('Radius (km)')

M_Earth=5.9E24
M_total=0.0
i=1

while M_total<0.5*M_Earth:
    M_total+=M[i]
    i+=1

print radius[i],M_total











