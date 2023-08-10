''' 
Let's import the Python modules we'll need for this example!
This normally goes at the top of your script or notebook, so that all code below it can see these packages.

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

# some beginning thoughts...

# create an array of 100 zeros
x = np.zeros(100)

# loop over these, and save something in each one
for i in range(1, 100):
    x[i] = i*2 + 5 + x[i-1]

plt.plot(x)

z = np.where((x > 8000))
num = np.arange(0,100)

plt.plot(num, x)
plt.scatter(num[z], x[z], c='red', lw=0)



